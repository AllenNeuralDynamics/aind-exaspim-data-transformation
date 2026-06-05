"""Tests for the metadata upgrade module."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from aind_exaspim_data_transformation.upgrade_metadata import (
    _coerce_instrument_id,
    _derive_subject_id,
    _load_metadata_file,
    _needs_upgrade,
    _s3_object_exists,
    _write_json_to_tempfile,
    get_additional_metadata,
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


class TestCoerceInstrumentId(unittest.TestCase):
    """Tests for _coerce_instrument_id helper."""

    def test_keeps_existing_value(self):
        """A non-empty acquisition instrument_id is preserved as-is."""
        acq = {"instrument_id": "exaSPIM-01", "data_streams": []}
        inst = {"instrument_id": "other-id"}
        out = _coerce_instrument_id(acq, inst)
        self.assertEqual(out["instrument_id"], "exaSPIM-01")

    def test_falls_back_to_instrument_when_empty(self):
        """Empty acquisition instrument_id is replaced from instrument."""
        acq = {"instrument_id": "", "data_streams": []}
        inst = {"instrument_id": "exaSPIM-01"}
        out = _coerce_instrument_id(acq, inst)
        self.assertEqual(out["instrument_id"], "exaSPIM-01")

    def test_falls_back_when_literal_none_string(self):
        """The literal string 'None' is treated as missing."""
        acq = {"instrument_id": "None", "data_streams": []}
        inst = {"instrument_id": "exaSPIM-01"}
        out = _coerce_instrument_id(acq, inst)
        self.assertEqual(out["instrument_id"], "exaSPIM-01")

    def test_returns_original_when_neither_set(self):
        """If both sides are empty the value is left unchanged."""
        acq = {"instrument_id": "", "data_streams": []}
        inst: dict = {"instrument_id": ""}
        out = _coerce_instrument_id(acq, inst)
        self.assertEqual(out["instrument_id"], "")

    def test_does_not_mutate_input(self):
        """The original acquisition dict must not be modified in place."""
        acq = {"instrument_id": "", "data_streams": []}
        inst = {"instrument_id": "exaSPIM-01"}
        _coerce_instrument_id(acq, inst)
        self.assertEqual(acq["instrument_id"], "")


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

        fake_upgraded_dict = {
            "schema_version": "2.5.1",
            "data_streams": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_exaspim_data_transformation.upgrade_metadata"
                "._upgrade_acquisition_only",
                return_value=fake_upgraded_dict,
            ) as mock_acq_only:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Direct path was called (no instrument)
                mock_acq_only.assert_called_once()

                # Should have uploaded: backup v1 acq + upgraded acq = 2
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

        fake_metadata = MagicMock()
        fake_metadata.acquisition = fake_upgraded_acq
        fake_metadata.instrument = fake_upgraded_inst

        mock_upgrade_instance = MagicMock()
        mock_upgrade_instance.metadata = fake_metadata

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(
                tmpdir, acq_data=v1_acq, inst_data=v1_inst
            )

            with patch(
                "aind_metadata_upgrader.upgrade.Upgrade",
                return_value=mock_upgrade_instance,
            ) as mock_upgrade_cls:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Single Upgrade() call with both files
                mock_upgrade_cls.assert_called_once()
                record = mock_upgrade_cls.call_args[0][0]
                self.assertIn("acquisition", record)
                self.assertIn("instrument", record)

                # backup v1_acq + backup v1_inst + upgraded acq + upgraded inst
                self.assertEqual(mock_upload.call_count, 4)

                s3_dests = [call[0][1] for call in mock_upload.call_args_list]
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
    def test_proceeds_without_instrument_using_stub(self, mock_upload):
        """Should upgrade acquisition via the direct upgrader path when
        instrument.json is absent."""
        v1_acq = {"schema_version": "1.0.4", "tiles": [], "axes": []}

        fake_upgraded_dict = {
            "schema_version": "2.5.1",
            "data_streams": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_exaspim_data_transformation.upgrade_metadata"
                "._upgrade_acquisition_only",
                return_value=fake_upgraded_dict,
            ) as mock_acq_only:
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Direct path was called (not Upgrade())
                mock_acq_only.assert_called_once()

                # Only backup + upload for acquisition = 2 calls
                self.assertEqual(mock_upload.call_count, 2)

    @patch(
        "aind_exaspim_data_transformation.upgrade_metadata"
        "._upload_bytes_to_s3"
    )
    def test_s3_trailing_slash_handled(self, mock_upload):
        """S3 location with trailing slash should not produce double slashes."""
        v1_acq = {"schema_version": "1.0.4", "tiles": []}

        fake_upgraded_dict = {"schema_version": "2.5.1"}

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_exaspim_data_transformation.upgrade_metadata"
                "._upgrade_acquisition_only",
                return_value=fake_upgraded_dict,
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

        fake_upgraded_dict = {
            "schema_version": "2.5.1",
            "data_streams": [],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            source_dir = self._make_source_dir(tmpdir, acq_data=v1_acq)

            with patch(
                "aind_exaspim_data_transformation.upgrade_metadata"
                "._upgrade_acquisition_only",
                return_value=fake_upgraded_dict,
            ):
                upgrade_metadata(source_dir, "s3://bucket/dataset")

                # Find the non-backup upload call
                for call in mock_upload.call_args_list:
                    s3_dest = call[0][1]
                    if "derived" in s3_dest:
                        continue
                    body = call[0][0]
                    uploaded = json.loads(body.decode("utf-8"))
                    from packaging.version import parse as vparse

                    self.assertGreaterEqual(
                        vparse(uploaded["schema_version"]),
                        vparse("2.0.0"),
                        f"Uploaded acquisition.json has "
                        f"schema_version="
                        f"{uploaded['schema_version']} "
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
    def test_real_upgrade_without_instrument_produces_v2(
        self, mock_upload
    ):
        """Without instrument.json, the stub allows upgrade to v2.

        Uses the real upgrader with only acquisition.json (no instrument).
        A minimal stub is injected automatically so the upgrade proceeds.
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

            # Should have uploaded: backup v1 acq + upgraded acq = 2
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
                body = call[0][0]
                uploaded = json.loads(body.decode("utf-8"))
                from packaging.version import parse as vparse

                self.assertGreaterEqual(
                    vparse(uploaded.get("schema_version", "0.0.0")),
                    vparse("2.0.0"),
                    f"Uploaded acquisition has "
                    f"schema_version="
                    f"{uploaded.get('schema_version')} "
                    f"which is below 2.0.0",
                )


class TestDeriveSubjectId(unittest.TestCase):
    """Tests for _derive_subject_id helper."""

    def test_exaspim_path(self):
        """Subject ID is the second underscore-delimited segment."""
        path = "/allen/aind/stage/exaSPIM/exaSPIM_765830_2025-11-21_12-01-47/exaSPIM"
        self.assertEqual(_derive_subject_id(path), "765830")

    def test_non_exaspim_path(self):
        """Fallback: first underscore-delimited segment."""
        path = "/data/my_dataset_2025-01-01/SPIM"
        self.assertEqual(_derive_subject_id(path), "my")

    def test_no_underscores(self):
        """No underscores returns the full folder name."""
        path = "/data/singlename/exaSPIM"
        self.assertEqual(_derive_subject_id(path), "singlename")

    def test_exaspim_uppercase_variant(self):
        """ExaSPIM in folder name triggers second-segment extraction."""
        path = "/data/ExaSPIM_123456_2026-04-28/exaSPIM"
        self.assertEqual(_derive_subject_id(path), "123456")


@patch("aind_exaspim_data_transformation.upgrade_metadata._s3_object_exists")
@patch("aind_exaspim_data_transformation.upgrade_metadata._upload_bytes_to_s3")
class TestGetAdditionalMetadata(unittest.TestCase):
    """Tests for get_additional_metadata."""

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_fetches_and_uploads_both_files(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """Happy path: not local, not in S3 → fetch and upload."""
        mock_s3_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test_subject"}
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            self.assertEqual(mock_get.call_count, 2)
            # Two files uploaded to S3
            self.assertEqual(mock_upload.call_count, 2)
            # Both files written locally
            self.assertTrue((dataset_dir / "subject.json").exists())
            self.assertTrue((dataset_dir / "procedures.json").exists())

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_skips_existing_local_files(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """Files already present locally are not re-fetched."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            # Pre-create both files
            (dataset_dir / "subject.json").write_text("{}")
            (dataset_dir / "procedures.json").write_text("{}")

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            mock_get.assert_not_called()
            mock_upload.assert_not_called()
            # S3 check not even reached when local file exists
            mock_s3_exists.assert_not_called()

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_skips_when_already_in_s3(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """Files placed in S3 by gather_preliminary_metadata are not re-fetched."""
        mock_s3_exists.return_value = True

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            # S3 was checked for both files
            self.assertEqual(mock_s3_exists.call_count, 2)
            # No HTTP fetch, no upload
            mock_get.assert_not_called()
            mock_upload.assert_not_called()

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_s3_check_failure_falls_through_to_fetch(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """If S3 check fails (e.g., permissions), fall through to fetch."""
        mock_s3_exists.return_value = False  # treated as not found
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test_subject"}
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            # Falls through to fetch
            self.assertEqual(mock_get.call_count, 2)
            self.assertEqual(mock_upload.call_count, 2)

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_http_error_does_not_raise(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """HTTP errors are logged but do not crash."""
        mock_s3_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            # Should not raise
            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            mock_upload.assert_not_called()

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_network_exception_does_not_raise(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """Network errors are caught and logged."""
        import requests

        mock_s3_exists.return_value = False
        mock_get.side_effect = requests.ConnectionError("timeout")

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            # Should not raise
            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            mock_upload.assert_not_called()

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_dry_run_skips_s3_check_and_upload(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """dry_run=True skips S3 check, writes locally, no upload."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"name": "test_subject"}
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
                dry_run=True,
            )

            # Files written locally
            self.assertTrue((dataset_dir / "subject.json").exists())
            self.assertTrue((dataset_dir / "procedures.json").exists())
            # No S3 check or upload in dry_run
            mock_s3_exists.assert_not_called()
            mock_upload.assert_not_called()

    @patch("aind_exaspim_data_transformation.upgrade_metadata.requests.get")
    def test_status_400_still_writes(
        self, mock_get, mock_upload, mock_s3_exists
    ):
        """HTTP 400 is treated as a valid response (writes the JSON body)."""
        mock_s3_exists.return_value = False
        mock_response = MagicMock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"message": "not found"}
        mock_get.return_value = mock_response

        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_dir = Path(tmpdir) / "exaSPIM_765830_2025-11-21"
            source_dir = dataset_dir / "exaSPIM"
            source_dir.mkdir(parents=True)

            get_additional_metadata(
                str(source_dir),
                "s3://test-bucket/test-dataset",
            )

            self.assertEqual(mock_upload.call_count, 2)
            self.assertTrue((dataset_dir / "subject.json").exists())


class TestS3ObjectExists(unittest.TestCase):
    """Tests for _s3_object_exists helper."""

    @patch("aind_exaspim_data_transformation.upgrade_metadata.boto3.client")
    def test_returns_true_when_object_exists(self, mock_boto):
        mock_s3 = MagicMock()
        mock_boto.return_value = mock_s3
        self.assertTrue(
            _s3_object_exists("s3://bucket/key/subject.json")
        )
        mock_s3.head_object.assert_called_once_with(
            Bucket="bucket", Key="key/subject.json"
        )

    @patch("aind_exaspim_data_transformation.upgrade_metadata.boto3.client")
    def test_returns_false_on_client_error(self, mock_boto):
        from botocore.exceptions import ClientError

        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = ClientError(
            {"Error": {"Code": "404", "Message": "Not Found"}},
            "HeadObject",
        )
        mock_boto.return_value = mock_s3
        self.assertFalse(
            _s3_object_exists("s3://bucket/key/subject.json")
        )

    @patch("aind_exaspim_data_transformation.upgrade_metadata.boto3.client")
    def test_returns_false_on_any_exception(self, mock_boto):
        mock_s3 = MagicMock()
        mock_s3.head_object.side_effect = Exception("network error")
        mock_boto.return_value = mock_s3
        self.assertFalse(
            _s3_object_exists("s3://bucket/key/subject.json")
        )


if __name__ == "__main__":
    unittest.main()
