"""Tests for the metadata upgrade module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_exaspim_data_transformation.upgrade_metadata import (
    _load_metadata_file,
    _needs_upgrade,
    _write_json_to_tempfile,
    upgrade_metadata,
)


class TestNeedsUpgrade(unittest.TestCase):
    """Tests for _needs_upgrade helper."""

    def test_v1_needs_upgrade(self):
        self.assertTrue(_needs_upgrade({"schema_version": "1.0.4"}))

    def test_v0_needs_upgrade(self):
        self.assertTrue(_needs_upgrade({"schema_version": "0.5.0"}))

    def test_missing_version_needs_upgrade(self):
        self.assertTrue(_needs_upgrade({}))

    def test_v2_does_not_need_upgrade(self):
        self.assertFalse(_needs_upgrade({"schema_version": "2.0.0"}))

    def test_v2_5_does_not_need_upgrade(self):
        self.assertFalse(_needs_upgrade({"schema_version": "2.5.1"}))

    def test_v3_does_not_need_upgrade(self):
        self.assertFalse(_needs_upgrade({"schema_version": "3.0.0"}))


class TestLoadMetadataFile(unittest.TestCase):
    """Tests for _load_metadata_file helper."""

    def test_returns_none_for_missing_file(self):
        result = _load_metadata_file(Path("/nonexistent/path.json"))
        self.assertIsNone(result)

    def test_loads_valid_json(self):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump({"schema_version": "1.0.0", "key": "value"}, f)
            f.flush()
            path = Path(f.name)

        try:
            result = _load_metadata_file(path)
            self.assertIsNotNone(result)
            self.assertEqual(result["schema_version"], "1.0.0")
        finally:
            path.unlink()

    def test_returns_none_for_empty_file(self):
        """An empty file should yield an empty dict → None."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            f.write("")
            path = Path(f.name)

        try:
            result = _load_metadata_file(path)
            self.assertIsNone(result)
        finally:
            path.unlink()


class TestWriteJsonToTempfile(unittest.TestCase):
    """Tests for _write_json_to_tempfile helper."""

    def test_creates_valid_json_file(self):
        data = {"schema_version": "2.5.0", "tiles": []}
        tmp = _write_json_to_tempfile(data)
        try:
            self.assertTrue(tmp.exists())
            loaded = json.loads(tmp.read_text())
            self.assertEqual(loaded["schema_version"], "2.5.0")
        finally:
            tmp.unlink()


EXAMPLE_V1_ACQ = Path(__file__).resolve().parent.parent / (
    "docs/examples/acquisition.json"
)


class TestUpgradeMetadata(unittest.TestCase):
    """Integration-style tests for upgrade_metadata (with mocked I/O)."""

    def _make_source_dir(self, tmpdir, acq_data=None, inst_data=None):
        """Create a fake dataset directory structure.

        Returns the source_dir path (equivalent to ``input_source``).
        """
        dataset_dir = Path(tmpdir) / "exaSPIM_test_2026-01-01_00-00-00"
        source_dir = dataset_dir / "exaSPIM"
        source_dir.mkdir(parents=True)

        if acq_data is not None:
            (dataset_dir / "acquisition.json").write_text(
                json.dumps(acq_data, indent=2)
            )
        if inst_data is not None:
            (dataset_dir / "instrument.json").write_text(
                json.dumps(inst_data, indent=2)
            )

        return str(source_dir)

    def test_raises_when_no_acquisition(self):
        """Should raise FileNotFoundError when acquisition.json is
        missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir)

            with self.assertRaises(FileNotFoundError) as ctx:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

            self.assertIn("acquisition.json", str(ctx.exception))

    def test_skips_when_already_v2(self):
        """Should skip silently when acquisition is already v2+."""
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(
                tmpdir,
                acq_data={"schema_version": "2.5.0", "data_streams": []},
            )

            with patch(
                "aind_exaspim_data_transformation.upgrade_metadata"
                "._upload_bytes_to_s3"
            ) as mock_upload:
                upgrade_metadata(source_dir, "s3://bucket/dataset")
                mock_upload.assert_not_called()

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_upgrades_v1_acquisition(self, mock_upload):
        """Should call the upgrader and upload files for v1 data."""
        v1_acq = {"schema_version": "1.0.4", "tiles": [], "axes": []}

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.5.1",
            "data_streams": [],
        }

        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq
        fake_metadata.instrument = None

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ) as mock_upgrade_cls:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Upgrader was called with the v1 record
                mock_upgrade_cls.assert_called_once()
                call_args = mock_upgrade_cls.call_args
                self.assertIn("acquisition", call_args[0][0])
                self.assertTrue(call_args[1]["skip_metadata_validation"])

                # Should have uploaded: backup v1 acq + upgraded acq = 2 calls
                self.assertEqual(mock_upload.call_count, 2)

                # Check backup went to derived/
                backup_call = mock_upload.call_args_list[0]
                self.assertIn(
                    "derived/v1_acquisition.json", backup_call[0][1]
                )

                # Check upgraded went to root
                upload_call = mock_upload.call_args_list[1]
                self.assertIn("acquisition.json", upload_call[0][1])
                self.assertNotIn("derived", upload_call[0][1])

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_upgrades_both_acquisition_and_instrument(self, mock_upload):
        """Should process both files when both are present."""
        v1_acq = {"schema_version": "1.0.4", "tiles": [], "axes": []}
        v1_inst = {
            "schema_version": "1.0.0",
            "instrument_id": "exaSPIM",
            "fluorescence_filters": [],
            "light_sources": [],
        }

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.5.1",
        }
        fake_upgraded_inst = MagicMock()
        fake_upgraded_inst.model_dump.return_value = {
            "schema_version": "2.5.1",
        }

        # First Upgrade() call → acq, second → inst
        acq_upgrade = MagicMock()
        acq_upgrade.metadata.acquisition = fake_upgraded_acq

        inst_upgrade = MagicMock()
        inst_upgrade.metadata.instrument = fake_upgraded_inst

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(
                tmpdir, acq_data=v1_acq, inst_data=v1_inst
            )

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                side_effect=[acq_upgrade, inst_upgrade],
            ) as mock_upgrade_cls:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Upgrade called twice — once per file
                self.assertEqual(mock_upgrade_cls.call_count, 2)

                # First call: acquisition only
                acq_record = mock_upgrade_cls.call_args_list[0][0][0]
                self.assertIn("acquisition", acq_record)
                self.assertNotIn("instrument", acq_record)

                # Second call: instrument only
                inst_record = mock_upgrade_cls.call_args_list[1][0][0]
                self.assertIn("instrument", inst_record)
                self.assertNotIn("acquisition", inst_record)

                # backup v1_acq + backup v1_inst + upgraded acq + upgraded inst
                self.assertEqual(mock_upload.call_count, 4)

                s3_dests = [
                    call[0][1] for call in mock_upload.call_args_list
                ]
                self.assertTrue(
                    any("derived/v1_acquisition.json" in d for d in s3_dests)
                )
                self.assertTrue(
                    any("derived/v1_instrument.json" in d for d in s3_dests)
                )

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_proceeds_without_instrument(self, mock_upload):
        """Should upgrade acquisition even when instrument.json is absent."""
        v1_acq = {"schema_version": "1.0.4", "tiles": [], "axes": []}

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.5.1",
        }

        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq
        fake_metadata.instrument = None

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ) as mock_upgrade_cls:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Record should not contain 'instrument' key
                call_args = mock_upgrade_cls.call_args
                self.assertNotIn("instrument", call_args[0][0])

                # Only backup + upload for acquisition = 2 calls
                self.assertEqual(mock_upload.call_count, 2)

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_upgrades_acquisition_without_instrument(
        self, mock_upload
    ):
        """Acquisition upgrades independently when instrument.json is missing.

        The acquisition and instrument are upgraded in separate Upgrade()
        calls, so a missing instrument should NOT prevent the acquisition
        from being upgraded and uploaded.
        """
        v1_acq = {
            "schema_version": "1.0.4",
            "tiles": [
                {
                    "channel": {
                        "channel_name": "488",
                        "light_source_name": "488 nm",
                        "excitation_wavelength": 488,
                    }
                }
            ],
            "axes": [],
        }

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.4.0",
            "data_streams": [],
        }
        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ) as mock_upgrade_cls:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Upgrade called once — only for acquisition
                self.assertEqual(mock_upgrade_cls.call_count, 1)
                record = mock_upgrade_cls.call_args_list[0][0][0]
                self.assertIn("acquisition", record)
                self.assertNotIn("instrument", record)

                # backup v1 acq + upload upgraded acq = 2 uploads
                self.assertEqual(mock_upload.call_count, 2)

                s3_dests = [
                    call[0][1] for call in mock_upload.call_args_list
                ]
                self.assertTrue(
                    any("derived/v1_acquisition.json" in d for d in s3_dests),
                    "Expected backup of original acquisition.json",
                )
                self.assertTrue(
                    any(
                        d.endswith("/acquisition.json")
                        for d in s3_dests
                    ),
                    "Expected upload of upgraded acquisition.json",
                )

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_s3_trailing_slash_handled(self, mock_upload):
        """S3 location with trailing slash should not produce double slashes."""
        v1_acq = {"schema_version": "1.0.4", "tiles": []}

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.5.1",
        }

        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq
        fake_metadata.instrument = None

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ):
                upgrade_metadata(
                    source_dir, "s3://bucket/dataset/"  # trailing slash
                )

                for call in mock_upload.call_args_list:
                    s3_dest = call[0][1]
                    self.assertNotIn("//", s3_dest.replace("s3://", ""))


    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_upgraded_version_is_v2_or_higher(self, mock_upload):
        """Uploaded acquisition.json must have schema_version >= 2.0.0."""
        v1_acq = {"schema_version": "1.0.4", "tiles": [], "axes": []}

        fake_upgraded_acq = MagicMock()
        fake_upgraded_acq.model_dump.return_value = {
            "schema_version": "2.5.1",
            "data_streams": [],
        }

        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq
        fake_metadata.instrument = None

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ):
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Find the non-backup upload call (the upgraded acquisition)
                for call in mock_upload.call_args_list:
                    s3_dest = call[0][1]
                    if "derived" in s3_dest:
                        continue
                    # This is the upgraded file — parse the JSON body
                    body = call[0][0]
                    uploaded = json.loads(body.decode("utf-8"))
                    from packaging.version import parse as vparse

                    self.assertGreaterEqual(
                        vparse(uploaded["schema_version"]),
                        vparse("2.0.0"),
                        f"Uploaded acquisition.json has "
                        f"schema_version={uploaded['schema_version']} "
                        f"which is below 2.0.0",
                    )


class TestUpgradeMetadataRealUpgrader(unittest.TestCase):
    """Integration test using the real aind-metadata-upgrader (no mocks).

    Uses ``docs/examples/acquisition.json`` from this repository with a
    minimal ``instrument.json`` stub so the upgrader can run end-to-end.
    Runs with ``dry_run=True`` so no S3 credentials are needed.
    """

    EXAMPLE_ACQ = Path(__file__).resolve().parent.parent / (
        "docs/examples/acquisition.json"
    )

    def _make_dataset(self, tmpdir):
        """Copy the example acquisition.json and create a minimal instrument
        stub into a temporary dataset directory.  Returns the source_dir."""
        dataset_dir = Path(tmpdir) / "exaSPIM_test_2026-01-01_00-00-00"
        source_dir = dataset_dir / "exaSPIM"
        source_dir.mkdir(parents=True)

        # Copy the real v1 acquisition.json (use copy, not copy2,
        # to avoid preserving restrictive source file permissions)
        import shutil

        shutil.copy(self.EXAMPLE_ACQ, dataset_dir / "acquisition.json")

        # Create a minimal instrument.json to satisfy the upgrader.
        # Based on the real v1 fixture from aind-metadata-upgrader tests.
        instrument_stub = {
            "schema_version": "0.10.20",
            "describedBy": "https://raw.githubusercontent.com/AllenNeuralDynamics/aind-data-schema/main/src/aind_data_schema/core/instrument.py",
            "instrument_id": "exaSPIM",
            "instrument_type": "exaSPIM",
            "manufacturer": {"name": "Other"},
            "objectives": [
                {
                    "device_type": "Objective",
                    "name": "Custom Objective",
                    "serial_number": None,
                    "manufacturer": {"name": "Other"},
                    "model": "JM_DIAMOND 5.0X/1.3",
                    "numerical_aperture": "0.305",
                    "magnification": "5",
                    "immersion": "air",
                    "notes": None,
                }
            ],
            "detectors": [
                {
                    "device_type": "Detector",
                    "name": "Camera 1",
                    "serial_number": None,
                    "manufacturer": {"name": "Vieworks"},
                    "detector_type": "Camera",
                    "data_interface": "Coax",
                    "cooling": "Air",
                    "notes": None,
                }
            ],
            "light_sources": [
                {
                    "device_type": "Laser",
                    "name": "LAS-001",
                    "serial_number": None,
                    "manufacturer": {"name": "Oxxius"},
                    "wavelength": 488,
                    "wavelength_unit": "nanometer",
                    "coupling": "Single-mode fiber",
                    "notes": None,
                }
            ],
            "fluorescence_filters": [],
            "lenses": [],
            "scanning_stages": [
                {
                    "device_type": "Motorized stage",
                    "name": "stage-x",
                    "serial_number": None,
                    "manufacturer": {
                        "name": "Applied Scientific Instrumentation",
                        "abbreviation": "ASI",
                    },
                    "model": "MS-8000",
                    "travel": "1000",
                    "travel_unit": "millimeter",
                    "stage_axis_direction": "Detection axis",
                    "stage_axis_name": "X",
                    "notes": None,
                }
            ],
            "motorized_stages": [],
            "additional_devices": [],
            "com_ports": [],
            "daqs": [],
            "calibration_date": None,
            "calibration_data": None,
            "notes": None,
        }
        (dataset_dir / "instrument.json").write_text(
            json.dumps(instrument_stub, indent=2)
        )

        return str(source_dir)

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_real_upgrade_produces_v2(self, mock_upload):
        """The real upgrader must produce schema_version >= 2.0.0.

        This test uses the actual ``aind_metadata_upgrader.upgrade.Upgrade``
        class (no mocks) and validates the version of the JSON that would
        be uploaded to S3.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_dataset(tmpdir)

            upgrade_metadata(
                source_dir, "s3://test-bucket/test-dataset", dry_run=False
            )

            # At least one non-backup upload should have occurred
            non_backup_uploads = [
                call
                for call in mock_upload.call_args_list
                if "derived" not in call[0][1]
            ]
            self.assertGreater(
                len(non_backup_uploads),
                0,
                "Expected at least one non-backup upload",
            )

            for call in non_backup_uploads:
                s3_dest = call[0][1]
                body = call[0][0]
                uploaded = json.loads(body.decode("utf-8"))
                from packaging.version import parse as vparse

                self.assertGreaterEqual(
                    vparse(uploaded.get("schema_version", "0.0.0")),
                    vparse("2.0.0"),
                    f"Uploaded {s3_dest} has "
                    f"schema_version={uploaded.get('schema_version')} "
                    f"which is below 2.0.0",
                )


    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_real_upgrade_without_instrument_produces_v2(self, mock_upload):
        """Acquisition upgrades to v2 even without instrument.json.

        Uses the real upgrader with only acquisition.json (no instrument).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_test_no_inst"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            import shutil
            shutil.copy(self.EXAMPLE_ACQ, dataset_dir / "acquisition.json")

            upgrade_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
                dry_run=False,
            )

            # Should have uploads: backup v1 acq + upgraded acq
            non_backup_uploads = [
                call
                for call in mock_upload.call_args_list
                if "derived" not in call[0][1]
            ]
            self.assertGreater(
                len(non_backup_uploads),
                0,
                "Expected at least one non-backup upload",
            )

            for call in non_backup_uploads:
                s3_dest = call[0][1]
                body = call[0][0]
                uploaded = json.loads(body.decode("utf-8"))
                from packaging.version import parse as vparse

                self.assertGreaterEqual(
                    vparse(uploaded.get("schema_version", "0.0.0")),
                    vparse("2.0.0"),
                    f"Uploaded {s3_dest} has "
                    f"schema_version={uploaded.get('schema_version')} "
                    f"which is below 2.0.0",
                )


if __name__ == "__main__":
    unittest.main()
