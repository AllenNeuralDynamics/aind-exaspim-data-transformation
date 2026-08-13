"""Tests for neuroglancer_state module."""

import json
import unittest
from unittest.mock import MagicMock, patch

from aind_exaspim_data_transformation.neuroglancer_state import (
    _build_dimensions,
    _build_layers,
    _build_position,
    _build_shader,
    _build_source_transform,
    build_neuroglancer_state,
    compute_contrast_limits,
    compute_contrast_limits_by_channel,
    generate_neuroglancer_url,
    parse_tiles_from_acquisition,
    wavelength_to_hex,
)


class TestWavelengthToHex(unittest.TestCase):
    """Tests for wavelength_to_hex helper."""

    def test_known_wavelengths(self):
        self.assertEqual(wavelength_to_hex(488), "00ff00")
        self.assertEqual(wavelength_to_hex(561), "ffff00")
        self.assertEqual(wavelength_to_hex(638), "ff0000")
        self.assertEqual(wavelength_to_hex(405), "ff00ff")

    def test_unknown_wavelength_returns_white(self):
        self.assertEqual(wavelength_to_hex(999), "ffffff")
        self.assertEqual(wavelength_to_hex(123), "ffffff")


class TestBuildDimensions(unittest.TestCase):
    """Tests for _build_dimensions."""

    def test_basic_conversion(self):
        # voxel sizes in µm: Z=1.0, Y=0.748, X=0.748
        result = _build_dimensions([1.0, 0.748, 0.748])
        self.assertAlmostEqual(result["x"][0], 0.748e-6)
        self.assertAlmostEqual(result["y"][0], 0.748e-6)
        self.assertAlmostEqual(result["z"][0], 1.0e-6)
        self.assertEqual(result["x"][1], "m")
        self.assertEqual(result["y"][1], "m")
        self.assertEqual(result["z"][1], "m")

    def test_exaspim_typical_voxels(self):
        # Typical exaSPIM: Z=20µm, Y=15.04µm, X=15.04µm
        result = _build_dimensions([20.0, 15.04, 15.04])
        self.assertAlmostEqual(result["z"][0], 20.0e-6)
        self.assertAlmostEqual(result["x"][0], 15.04e-6)


class TestBuildSourceTransform(unittest.TestCase):
    """Tests for _build_source_transform."""

    def test_zero_translation(self):
        result = _build_source_transform(
            translation_um=[0.0, 0.0, 0.0],
            voxel_sizes_um=[20.0, 15.04, 15.04],
        )
        matrix = result["matrix"]
        # 4x5 matrix (rows z, y, x, t; input dims [t, z, y, x] since
        # neuroglancer handles the OME channel axis separately) with
        # zero translation.
        self.assertEqual(matrix[0], [0, 1, 0, 0, 0.0])  # z
        self.assertEqual(matrix[1], [0, 0, 1, 0, 0.0])  # y
        self.assertEqual(matrix[2], [0, 0, 0, 1, 0.0])  # x
        self.assertEqual(matrix[3], [1, 0, 0, 0, 0])  # t

    def test_nonzero_translation(self):
        # Translation in µm [Z=1000, Y=2000, X=3000]
        # Voxel sizes [Z=20, Y=10, X=10]
        result = _build_source_transform(
            translation_um=[1000.0, 2000.0, 3000.0],
            voxel_sizes_um=[20.0, 10.0, 10.0],
        )
        matrix = result["matrix"]
        # Rows ordered z, y, x, t; translation is the last column (index 4)
        # X translation in voxels: 3000/10 = 300
        self.assertAlmostEqual(matrix[2][4], 300.0)
        # Y translation in voxels: 2000/10 = 200
        self.assertAlmostEqual(matrix[1][4], 200.0)
        # Z translation in voxels: 1000/20 = 50
        self.assertAlmostEqual(matrix[0][4], 50.0)

    def test_output_dimensions(self):
        result = _build_source_transform(
            translation_um=[0.0, 0.0, 0.0],
            voxel_sizes_um=[20.0, 15.04, 15.04],
        )
        od = result["outputDimensions"]
        self.assertAlmostEqual(od["x"][0], 15.04e-6)
        self.assertAlmostEqual(od["y"][0], 15.04e-6)
        self.assertAlmostEqual(od["z"][0], 20.0e-6)


class TestBuildLayers(unittest.TestCase):
    """Tests for _build_layers."""

    def test_single_channel_single_tile(self):
        tiles_by_channel = {
            488: [
                {
                    "file_name": "tile_000_ch_488.ims",
                    "translation_um": [0.0, 0.0, 0.0],
                }
            ]
        }
        layers = _build_layers(
            tiles_by_channel=tiles_by_channel,
            s3_modality_path="s3://bucket/dataset/SPIM",
            voxel_sizes_um=[20.0, 15.04, 15.04],
        )
        self.assertEqual(len(layers), 1)
        layer = layers[0]
        self.assertEqual(layer["name"], "CH_488")
        self.assertEqual(layer["type"], "image")
        self.assertTrue(layer["visible"])
        self.assertEqual(layer["blend"], "additive")
        self.assertEqual(len(layer["source"]), 1)
        self.assertIn(
            "zarr3://s3://bucket/dataset/SPIM/tile_000_ch_488.ome.zarr",
            layer["source"][0]["url"],
        )

    def test_multi_channel_grouped(self):
        tiles_by_channel = {
            488: [{"file_name": "a_ch_488.ims", "translation_um": [0, 0, 0]}],
            638: [{"file_name": "a_ch_638.ims", "translation_um": [0, 0, 0]}],
        }
        layers = _build_layers(
            tiles_by_channel=tiles_by_channel,
            s3_modality_path="s3://bucket/ds/SPIM",
            voxel_sizes_um=[1.0, 1.0, 1.0],
        )
        self.assertEqual(len(layers), 2)
        names = [layer["name"] for layer in layers]
        self.assertIn("CH_488", names)
        self.assertIn("CH_638", names)

    def test_tile_without_translation(self):
        tiles_by_channel = {
            488: [{"file_name": "tile.ims"}]
        }
        layers = _build_layers(
            tiles_by_channel=tiles_by_channel,
            s3_modality_path="s3://bucket/ds/SPIM",
            voxel_sizes_um=[1.0, 1.0, 1.0],
        )
        # Source should not have a transform key
        self.assertNotIn("transform", layers[0]["source"][0])

    def test_layers_sorted_by_wavelength(self):
        tiles_by_channel = {
            638: [{"file_name": "b.ims"}],
            405: [{"file_name": "a.ims"}],
            561: [{"file_name": "c.ims"}],
        }
        layers = _build_layers(
            tiles_by_channel=tiles_by_channel,
            s3_modality_path="s3://b/d/SPIM",
            voxel_sizes_um=[1.0, 1.0, 1.0],
        )
        self.assertEqual(layers[0]["name"], "CH_405")
        self.assertEqual(layers[1]["name"], "CH_561")
        self.assertEqual(layers[2]["name"], "CH_638")


class TestBuildNeuroglancerState(unittest.TestCase):
    """Tests for build_neuroglancer_state."""

    def test_complete_state_structure(self):
        tiles_by_channel = {
            488: [
                {
                    "file_name": "tile_000_ch_488.ims",
                    "translation_um": [0.0, 100.0, 200.0],
                }
            ]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[20.0, 15.04, 15.04],
            s3_modality_path="s3://aind-open-data/exaSPIM_123/SPIM",
        )
        self.assertIn("dimensions", state)
        self.assertIn("layers", state)
        self.assertIn("showAxisLines", state)
        self.assertIn("showScaleBar", state)
        self.assertFalse(state["showAxisLines"])
        self.assertTrue(state["showScaleBar"])

    def test_state_is_json_serializable(self):
        tiles_by_channel = {
            488: [{"file_name": "t.ims", "translation_um": [0, 0, 0]}]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[1.0, 1.0, 1.0],
            s3_modality_path="s3://b/d/SPIM",
        )
        # Should not raise
        json_str = json.dumps(state)
        self.assertIsInstance(json_str, str)

    def test_dimensions_include_singleton_t_no_channel(self):
        tiles_by_channel = {
            488: [{"file_name": "t.ims", "translation_um": [0, 0, 0]}]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[20.0, 15.04, 15.04],
            s3_modality_path="s3://b/d/SPIM",
        )
        # t must be declared so neuroglancer pins t at 0; the redundant
        # singleton channel (c) dimension is not exposed. Dimensions are
        # ordered z, y, x, t.
        self.assertEqual(
            list(state["dimensions"].keys()), ["z", "y", "x", "t"]
        )
        self.assertEqual(state["dimensions"]["t"], [0.001, "s"])
        self.assertNotIn("c", state["dimensions"])

    def test_position_pins_t_to_zero(self):
        tiles_by_channel = {
            488: [
                {"file_name": "a.ims", "translation_um": [0.0, 0.0, 0.0]},
                {"file_name": "b.ims", "translation_um": [20.0, 30.08, 30.08]},
            ]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[20.0, 15.04, 15.04],
            s3_modality_path="s3://b/d/SPIM",
        )
        # position is ordered [z, y, x, t]; t must be pinned to 0 so the
        # viewer initialises on the single timepoint.
        self.assertEqual(len(state["position"]), 4)
        self.assertEqual(state["position"][3], 0.0)
        # spatial coords are the mosaic centre (midpoint in voxel units)
        self.assertAlmostEqual(state["position"][0], 0.5)  # z: (0+1)/2
        self.assertAlmostEqual(state["position"][1], 1.0)  # y: (0+2)/2
        self.assertAlmostEqual(state["position"][2], 1.0)  # x: (0+2)/2

    def test_build_position_defaults_without_translations(self):
        # Tiles missing translation_um fall back to origin; t stays 0.
        position = _build_position(
            tiles_by_channel={488: [{"file_name": "a.ims"}]},
            voxel_sizes_um=[20.0, 15.04, 15.04],
        )
        self.assertEqual(position, [0.0, 0.0, 0.0, 0.0])

    def test_ng_link_is_first_key(self):
        tiles_by_channel = {
            488: [{"file_name": "t.ims", "translation_um": [0, 0, 0]}]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[1.0, 1.0, 1.0],
            s3_modality_path="s3://b/d/SPIM",
            ng_link="https://viewer/#!x",
        )
        self.assertEqual(list(state.keys())[0], "ng_link")
        self.assertEqual(state["ng_link"], "https://viewer/#!x")

    def test_contrast_limits_applied_to_layers(self):
        tiles_by_channel = {
            488: [{"file_name": "t.ims", "translation_um": [0, 0, 0]}]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[1.0, 1.0, 1.0],
            s3_modality_path="s3://b/d/SPIM",
            contrast_limits_by_channel={488: (12.0, 3400.0)},
        )
        rng = state["layers"][0]["shaderControls"]["normalized"]["range"]
        self.assertEqual(rng, [12.0, 3400.0])

    def test_contrast_limits_default_when_missing(self):
        tiles_by_channel = {
            488: [{"file_name": "t.ims", "translation_um": [0, 0, 0]}]
        }
        state = build_neuroglancer_state(
            tiles_by_channel=tiles_by_channel,
            voxel_sizes_um=[1.0, 1.0, 1.0],
            s3_modality_path="s3://b/d/SPIM",
        )
        rng = state["layers"][0]["shaderControls"]["normalized"]["range"]
        self.assertEqual(rng, [0, 200])


class TestComputeContrastLimits(unittest.TestCase):
    """Tests for compute_contrast_limits helpers."""

    @patch(
        "aind_exaspim_data_transformation.utils.io_utils.ImarisReader"
    )
    def test_compute_contrast_limits_ignores_zeros(self, mock_reader_cls):
        import numpy as np

        reader = MagicMock()
        reader.n_levels = 3
        # Coarsest level (index 2) is the first that fits the voxel budget.
        reader.get_true_shape_for_level.side_effect = (
            lambda lvl: (1000, 1000, 1000) if lvl < 2 else (1, 2, 3)
        )
        reader.as_array.return_value = np.array(
            [0, 0, 0, 100, 200, 300], dtype="uint16"
        )
        mock_reader_cls.return_value.__enter__.return_value = reader

        low, high = compute_contrast_limits(
            "fake.ims", low_percentile=0, high_percentile=100
        )
        # Zeros excluded → min nonzero 100, max 300
        self.assertEqual(low, 100.0)
        self.assertEqual(high, 300.0)
        # Finest level within the voxel budget (index 2) is read
        args, _ = reader.as_array.call_args
        self.assertIn("ResolutionLevel 2", args[0])

    @patch(
        "aind_exaspim_data_transformation.utils.io_utils.ImarisReader"
    )
    def test_compute_contrast_limits_high_gt_low(self, mock_reader_cls):
        import numpy as np

        reader = MagicMock()
        reader.n_levels = 1
        reader.get_true_shape_for_level.return_value = (1, 1, 3)
        reader.as_array.return_value = np.array([5, 5, 5], dtype="uint16")
        mock_reader_cls.return_value.__enter__.return_value = reader

        low, high = compute_contrast_limits("fake.ims")
        self.assertGreater(high, low)

    def test_by_channel_skips_missing_files(self):
        tiles_by_channel = {
            488: [{"file_name": "missing.ims"}],
        }
        limits = compute_contrast_limits_by_channel(
            tiles_by_channel=tiles_by_channel,
            input_source_dir="/nonexistent",
        )
        self.assertEqual(limits, {})

    @patch(
        "aind_exaspim_data_transformation.neuroglancer_state."
        "_sample_intensities"
    )
    def test_by_channel_aggregates_samples(self, mock_sample):
        import tempfile
        from pathlib import Path as _Path

        import numpy as np

        # Two tiles contribute overlapping foreground distributions that
        # are pooled before percentiles are computed.
        mock_sample.side_effect = [
            np.array([10, 20, 30], dtype="uint16"),
            np.array([400, 450, 500], dtype="uint16"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            for name in ("tile_a.ims", "tile_b.ims"):
                (_Path(tmp) / name).write_bytes(b"")
            limits = compute_contrast_limits_by_channel(
                tiles_by_channel={
                    488: [
                        {"file_name": "tile_a.ims"},
                        {"file_name": "tile_b.ims"},
                    ]
                },
                input_source_dir=tmp,
                low_percentile=0,
                high_percentile=100,
            )
        # Pooled min is 10 (tile_a) and pooled max is 500 (tile_b).
        self.assertEqual(limits, {488: (10.0, 500.0)})
        self.assertEqual(mock_sample.call_count, 2)


class TestGenerateNeuroglancerUrl(unittest.TestCase):
    """Tests for generate_neuroglancer_url."""

    def test_default_viewer(self):
        url = generate_neuroglancer_url(
            "s3://aind-open-data/dataset/neuroglancer.json"
        )
        self.assertTrue(
            url.startswith("https://neuroglancer-demo.appspot.com/#!")
        )
        self.assertIn("aind-open-data.s3.amazonaws.com", url)
        self.assertIn("dataset/neuroglancer.json", url)

    def test_custom_viewer(self):
        url = generate_neuroglancer_url(
            "s3://bucket/key.json",
            viewer_url="https://custom-ng.example.com",
        )
        self.assertTrue(url.startswith("https://custom-ng.example.com/#!"))

    def test_url_with_spaces_encoded(self):
        url = generate_neuroglancer_url(
            "s3://bucket/my dataset/file.json"
        )
        self.assertIn("my%20dataset", url)
        self.assertNotIn(" ", url)


class TestParseTilesFromAcquisition(unittest.TestCase):
    """Tests for parse_tiles_from_acquisition (v1 and v2 schemas)."""

    def test_schema_v1_basic(self):
        acquisition = {
            "schema_version": "1.0.4",
            "tiles": [
                {
                    "file_name": "right_000000_ch_488.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["15.04", "15.04", "20.0"]},
                        {
                            "type": "translation",
                            "translation": ["-55.4196", "7.605", "-24.6392"],
                        },
                    ],
                    "channel": {
                        "excitation_wavelength": 488,
                    },
                },
                {
                    "file_name": "left_000001_ch_488.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["15.04", "15.04", "20.0"]},
                        {
                            "type": "translation",
                            "translation": ["-55.4196", "7.605", "-24.6392"],
                        },
                    ],
                    "channel": {
                        "excitation_wavelength": 488,
                    },
                },
            ],
        }
        tiles_by_channel, voxel_sizes_um = parse_tiles_from_acquisition(
            acquisition
        )
        # Both tiles are 488nm → one channel group with 2 tiles
        self.assertIn(488, tiles_by_channel)
        self.assertEqual(len(tiles_by_channel[488]), 2)
        # Voxel sizes: scale = [X=15.04, Y=15.04, Z=20.0] → [Z, Y, X]
        self.assertAlmostEqual(voxel_sizes_um[0], 20.0)
        self.assertAlmostEqual(voxel_sizes_um[1], 15.04)
        self.assertAlmostEqual(voxel_sizes_um[2], 15.04)

    def test_schema_v1_translation_mm_to_um(self):
        acquisition = {
            "schema_version": "1.0.0",
            "tiles": [
                {
                    "file_name": "tile.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["1.0", "1.0", "1.0"]},
                        {
                            "type": "translation",
                            "translation": ["1.0", "2.0", "3.0"],
                        },
                    ],
                    "channel": {"excitation_wavelength": 561},
                }
            ],
        }
        tiles_by_channel, _ = parse_tiles_from_acquisition(acquisition)
        tile = tiles_by_channel[561][0]
        # translation [X=1mm, Y=2mm, Z=3mm] → [Z=3000µm, Y=2000µm, X=1000µm]
        self.assertAlmostEqual(tile["translation_um"][0], 3000.0)
        self.assertAlmostEqual(tile["translation_um"][1], 2000.0)
        self.assertAlmostEqual(tile["translation_um"][2], 1000.0)

    def test_schema_v1_multi_channel(self):
        acquisition = {
            "schema_version": "1.0.0",
            "tiles": [
                {
                    "file_name": "t_488.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["1", "1", "1"]},
                    ],
                    "channel": {"excitation_wavelength": 488},
                },
                {
                    "file_name": "t_638.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["1", "1", "1"]},
                    ],
                    "channel": {"excitation_wavelength": 638},
                },
            ],
        }
        tiles_by_channel, _ = parse_tiles_from_acquisition(acquisition)
        self.assertIn(488, tiles_by_channel)
        self.assertIn(638, tiles_by_channel)

    def test_schema_v1_no_tiles_returns_empty(self):
        acquisition = {"schema_version": "1.0.0", "tiles": []}
        tiles_by_channel, voxel_sizes = parse_tiles_from_acquisition(
            acquisition
        )
        self.assertEqual(tiles_by_channel, {})
        # Fallback voxel sizes
        self.assertEqual(voxel_sizes, [1.0, 1.0, 1.0])

    def test_schema_v2_basic(self):
        acquisition = {
            "schema_version": "2.1.0",
            "data_streams": [
                {
                    "configurations": [
                        {
                            "images": [
                                {
                                    "file_name": "tile_488.ims",
                                    "channel": {
                                        "excitation_wavelength": 488,
                                    },
                                    "image_to_acquisition_transform": [
                                        {
                                            "object_type": "Scale",
                                            "scale": [
                                                "15.04",
                                                "15.04",
                                                "20.0",
                                            ],
                                        },
                                        {
                                            "object_type": "Translation",
                                            "translation": [
                                                "-55.0",
                                                "7.0",
                                                "-24.0",
                                            ],
                                        },
                                    ],
                                }
                            ]
                        }
                    ]
                }
            ],
        }
        tiles_by_channel, voxel_sizes_um = parse_tiles_from_acquisition(
            acquisition
        )
        self.assertIn(488, tiles_by_channel)
        self.assertEqual(len(tiles_by_channel[488]), 1)
        self.assertAlmostEqual(voxel_sizes_um[0], 20.0)
        self.assertAlmostEqual(voxel_sizes_um[1], 15.04)
        self.assertAlmostEqual(voxel_sizes_um[2], 15.04)


class TestGenerateNeuroglancerStateIntegration(unittest.TestCase):
    """Integration test for _generate_neuroglancer_state on ImarisJob."""

    @patch("boto3.client")
    @patch(
        "aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict"
    )
    def test_generate_neuroglancer_state_uploads_json(
        self, mock_read_json, mock_boto3_client
    ):
        """Verify that _generate_neuroglancer_state uploads valid JSON."""
        from aind_exaspim_data_transformation.imaris_job import (
            ImarisCompressionJob,
        )
        from aind_exaspim_data_transformation.models import ImarisJobSettings

        settings = ImarisJobSettings(
            input_source="/data/exaSPIM_123/SPIM",
            output_directory="/tmp/output",
            num_of_partitions=1,
            partition_to_process=0,
            s3_location="s3://aind-open-data/exaSPIM_123/SPIM",
        )
        job = ImarisCompressionJob(job_settings=settings)

        mock_read_json.return_value = {
            "schema_version": "1.0.4",
            "tiles": [
                {
                    "file_name": "tile_ch_488.ims",
                    "coordinate_transformations": [
                        {"type": "scale", "scale": ["15.04", "15.04", "20.0"]},
                        {
                            "type": "translation",
                            "translation": ["-55.0", "7.0", "-24.0"],
                        },
                    ],
                    "channel": {"excitation_wavelength": 488},
                }
            ],
        }

        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client

        with patch("pathlib.Path.is_file", return_value=True):
            job._generate_neuroglancer_state()

        # Verify put_object was called
        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        self.assertEqual(call_kwargs["Bucket"], "aind-open-data")
        self.assertEqual(
            call_kwargs["Key"], "exaSPIM_123/neuroglancer.json"
        )
        self.assertEqual(call_kwargs["ContentType"], "application/json")

        # Verify the uploaded JSON is valid
        body = call_kwargs["Body"].decode("utf-8")
        state = json.loads(body)
        self.assertIn("dimensions", state)
        self.assertIn("layers", state)
        self.assertEqual(len(state["layers"]), 1)
        self.assertEqual(state["layers"][0]["name"], "CH_488")

    @patch("boto3.client")
    def test_generate_neuroglancer_state_skips_when_no_s3(
        self, mock_boto3_client
    ):
        """Verify graceful skip when s3_location is None."""
        from aind_exaspim_data_transformation.imaris_job import (
            ImarisCompressionJob,
        )
        from aind_exaspim_data_transformation.models import ImarisJobSettings

        settings = ImarisJobSettings(
            input_source="/data/exaSPIM_123/SPIM",
            output_directory="/tmp/output",
            num_of_partitions=1,
            partition_to_process=0,
            s3_location=None,
        )
        job = ImarisCompressionJob(job_settings=settings)
        job._generate_neuroglancer_state()

        # Should not attempt S3 upload
        mock_boto3_client.assert_not_called()


class TestBuildShader(unittest.TestCase):
    """Tests for _build_shader."""

    def test_shader_contains_color(self):
        shader = _build_shader("00ff00")
        self.assertIn("00ff00", shader)
        self.assertIn("emitRGB", shader)
        self.assertIn("normalized", shader)


if __name__ == "__main__":
    unittest.main()
