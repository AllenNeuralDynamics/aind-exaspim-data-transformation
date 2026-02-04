"""Tests for omezarr_metadata module"""

import unittest
from unittest.mock import patch

from aind_exaspim_data_transformation.compress.omezarr_metadata import (
    _build_ome,
    _compute_scales,
    _downscale_origin,
    _get_axes_5d,
    _get_pyramid_metadata,
    _validate_axes_for_format,
    _validate_omero_metadata,
    add_multiscales_metadata,
    write_ome_ngff_metadata,
)


class TestOmeZarrMetadata(unittest.TestCase):
    """Test suite for OME-Zarr metadata functions"""

    def test_get_pyramid_metadata(self):
        """Test _get_pyramid_metadata returns correct structure"""
        result = _get_pyramid_metadata()

        self.assertIn("metadata", result)
        self.assertEqual(
            result["metadata"]["method"], "tensorstore.downsample"
        )
        self.assertEqual(
            result["metadata"]["description"],
            "Downscaling tensorstore downsample",
        )

    def test_build_ome_minimal(self):
        """Test _build_ome with minimal parameters"""
        data_shape = (1, 2, 100, 200, 300)  # TCZYX
        image_name = "test_image"

        result = _build_ome(data_shape, image_name)

        self.assertIn("channels", result)
        self.assertEqual(len(result["channels"]), 2)
        self.assertEqual(
            result["channels"][0]["label"], "Channel:test_image:0"
        )

    def test_build_ome_with_channel_names(self):
        """Test _build_ome with custom channel names"""
        data_shape = (1, 2, 100, 200, 300)
        image_name = "test_image"
        channel_names = ["Red", "Green"]

        result = _build_ome(
            data_shape, image_name, channel_names=channel_names
        )

        self.assertEqual(result["channels"][0]["label"], "Red")
        self.assertEqual(result["channels"][1]["label"], "Green")

    def test_build_ome_with_colors(self):
        """Test _build_ome with custom channel colors"""
        data_shape = (1, 2, 100, 200, 300)
        image_name = "test_image"
        channel_colors = [255, 65280]  # Red and Green in hex

        result = _build_ome(
            data_shape, image_name, channel_colors=channel_colors
        )

        self.assertEqual(result["channels"][0]["color"], "0000ff")
        self.assertEqual(result["channels"][1]["color"], "00ff00")

    def test_build_ome_with_minmax(self):
        """Test _build_ome with custom min/max values"""
        data_shape = (1, 1, 100, 200, 300)
        image_name = "test_image"
        channel_minmax = [(0.0, 65535.0)]

        result = _build_ome(
            data_shape, image_name, channel_minmax=channel_minmax
        )

        self.assertEqual(result["channels"][0]["window"]["min"], 0.0)
        self.assertEqual(result["channels"][0]["window"]["max"], 65535.0)

    def test_compute_scales_single_level(self):
        """Test _compute_scales with single pyramid level"""
        scale_num_levels = 1
        scale_factor = (2.0, 2.0, 2.0)
        pixelsizes = (1.0, 0.5, 0.5)
        chunks = (1, 1, 64, 128, 128)
        data_shape = (1, 1, 100, 200, 300)

        transforms, chunk_sizes = _compute_scales(
            scale_num_levels, scale_factor, pixelsizes, chunks, data_shape
        )

        self.assertEqual(len(transforms), 1)
        self.assertEqual(len(chunk_sizes), 1)
        self.assertEqual(transforms[0][0]["scale"], [1.0, 1.0, 1.0, 0.5, 0.5])

    def test_compute_scales_multiple_levels(self):
        """Test _compute_scales with multiple pyramid levels"""
        scale_num_levels = 3
        scale_factor = (2.0, 2.0, 2.0)
        pixelsizes = (1.0, 0.5, 0.5)
        chunks = (1, 1, 64, 128, 128)
        data_shape = (1, 1, 100, 200, 300)

        transforms, chunk_sizes = _compute_scales(
            scale_num_levels, scale_factor, pixelsizes, chunks, data_shape
        )

        self.assertEqual(len(transforms), 3)
        self.assertEqual(len(chunk_sizes), 3)

        # Check scaling progression
        self.assertEqual(transforms[0][0]["scale"], [1.0, 1.0, 1.0, 0.5, 0.5])
        self.assertEqual(transforms[1][0]["scale"], [1.0, 1.0, 2.0, 1.0, 1.0])
        self.assertEqual(transforms[2][0]["scale"], [1.0, 1.0, 4.0, 2.0, 2.0])

    def test_compute_scales_with_translations(self):
        """Test _compute_scales with translation transforms"""
        scale_num_levels = 2
        scale_factor = (2.0, 2.0, 2.0)
        pixelsizes = (1.0, 0.5, 0.5)
        chunks = (1, 1, 64, 128, 128)
        data_shape = (1, 1, 100, 200, 300)
        translations = [[0, 0, 0, 0, 0], [0, 0, 1, 1, 1]]

        transforms, chunk_sizes = _compute_scales(
            scale_num_levels,
            scale_factor,
            pixelsizes,
            chunks,
            data_shape,
            translations=translations,
        )

        self.assertEqual(len(transforms[0]), 2)  # Scale + translation
        self.assertEqual(transforms[0][1]["type"], "translation")
        self.assertEqual(transforms[1][1]["translation"], [0, 0, 1, 1, 1])

    def test_get_axes_5d_default(self):
        """Test _get_axes_5d with default units"""
        axes = _get_axes_5d()

        self.assertEqual(len(axes), 5)
        self.assertEqual(axes[0]["name"], "t")
        self.assertEqual(axes[0]["unit"], "millisecond")
        self.assertEqual(axes[2]["name"], "z")
        self.assertEqual(axes[2]["unit"], "micrometer")

    def test_get_axes_5d_custom_units(self):
        """Test _get_axes_5d with custom units"""
        axes = _get_axes_5d(time_unit="second", space_unit="nanometer")

        self.assertEqual(axes[0]["unit"], "second")
        self.assertEqual(axes[2]["unit"], "nanometer")

    def test_validate_axes_for_format_version_01(self):
        """Test _validate_axes_for_format with version 0.1"""
        from ome_zarr.format import FormatV01

        fmt = FormatV01()
        axes = [{"name": "z"}, {"name": "y"}, {"name": "x"}]

        result_axes, ndim = _validate_axes_for_format(axes, fmt)

        self.assertIsNone(result_axes)  # Axes ignored for 0.1
        self.assertEqual(ndim, -1)

    def test_validate_omero_metadata_valid(self):
        """Test _validate_omero_metadata with valid metadata"""
        omero_metadata = {
            "channels": [
                {
                    "color": "ff0000",
                    "window": {
                        "min": 0.0,
                        "max": 255.0,
                        "start": 0.0,
                        "end": 255.0,
                    },
                }
            ]
        }

        # Should not raise any errors
        _validate_omero_metadata(omero_metadata)

    def test_validate_omero_metadata_invalid_color(self):
        """Test _validate_omero_metadata with invalid color"""
        omero_metadata = {
            "channels": [{"color": "red"}]  # Should be 6-char hex
        }

        with self.assertRaises(TypeError):
            _validate_omero_metadata(omero_metadata)

    def test_validate_omero_metadata_missing_window_key(self):
        """Test _validate_omero_metadata with missing window key"""
        omero_metadata = {
            "channels": [
                {
                    "color": "ff0000",
                    "window": {
                        "min": 0.0,
                        "max": 255.0,
                        "start": 0.0,
                    },  # Missing 'end'
                }
            ]
        }

        with self.assertRaises(KeyError):
            _validate_omero_metadata(omero_metadata)

    def test_validate_omero_metadata_none(self):
        """Test _validate_omero_metadata with None"""
        # Should not raise error
        _validate_omero_metadata(None)

    def test_add_multiscales_metadata_basic(self):
        """Test add_multiscales_metadata with basic parameters"""
        from ome_zarr.format import CurrentFormat

        group = {}
        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            },
            {
                "path": "1",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [2.0, 2.0, 2.0]}
                ],
            },
        ]
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        result = add_multiscales_metadata(
            group, datasets, fmt=CurrentFormat(), axes=axes
        )

        self.assertIn("attributes", result)
        self.assertIn("ome", result["attributes"])
        self.assertIn("multiscales", result["attributes"]["ome"])

    def test_add_multiscales_metadata_with_name(self):
        """Test add_multiscales_metadata with custom name"""
        from ome_zarr.format import CurrentFormat

        group = {}
        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            },
        ]
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]

        result = add_multiscales_metadata(
            group, datasets, fmt=CurrentFormat(), axes=axes, name="test_image"
        )

        self.assertEqual(
            result["attributes"]["ome"]["multiscales"][0]["name"], "test_image"
        )

    def test_add_multiscales_metadata_with_omero(self):
        """Test add_multiscales_metadata with OMERO metadata"""
        from ome_zarr.format import CurrentFormat

        group = {}
        datasets = [
            {
                "path": "0",
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1.0, 1.0, 1.0]}
                ],
            },
        ]
        axes = [
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        omero_metadata = {
            "channels": [
                {
                    "color": "ff0000",
                    "window": {"min": 0, "max": 255, "start": 0, "end": 255},
                }
            ]
        }

        result = add_multiscales_metadata(
            group,
            datasets,
            fmt=CurrentFormat(),
            axes=axes,
            omero_metadata=omero_metadata,
        )

        self.assertIn("omero", result["attributes"]["ome"])
        self.assertEqual(result["attributes"]["ome"]["omero"], omero_metadata)

    def test_downscale_origin_single_level(self):
        """Test _downscale_origin with single level"""
        array_shape = [1, 1, 100, 200, 300]
        origin = [0.0, 0.0, 0.0]
        voxel_size = [1.0, 0.5, 0.5]
        scale_factors = [2, 2, 2]
        n_levels = 1

        origins = _downscale_origin(
            array_shape, origin, voxel_size, scale_factors, n_levels
        )

        self.assertEqual(len(origins), 1)
        self.assertEqual(origins[0], [0.0, 0.0, 0.0, 0.0, 0.0])

    def test_downscale_origin_multiple_levels(self):
        """Test _downscale_origin with multiple levels"""
        array_shape = [1, 1, 100, 200, 300]
        origin = [0.0, 0.0, 0.0]
        voxel_size = [1.0, 0.5, 0.5]
        scale_factors = [2, 2, 2]
        n_levels = 3

        origins = _downscale_origin(
            array_shape, origin, voxel_size, scale_factors, n_levels
        )

        self.assertEqual(len(origins), 3)
        # First level should be padded to 5D
        self.assertEqual(len(origins[0]), 5)
        # Subsequent levels should have shifted origins
        self.assertTrue(origins[1][2] > origins[0][2])  # Z shifts

    @patch(
        "aind_exaspim_data_transformation.compress.omezarr_metadata._build_ome"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.omezarr_metadata._compute_scales"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.omezarr_metadata._downscale_origin"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.omezarr_metadata.add_multiscales_metadata"
    )
    def test_write_ome_ngff_metadata(
        self,
        mock_add_multiscales,
        mock_downscale,
        mock_compute,
        mock_build_ome,
    ):
        """Test write_ome_ngff_metadata integration"""
        # Mock return values
        mock_build_ome.return_value = {"channels": []}
        # Return 3 transformations (one per level) to match n_lvls=3
        # Each transformation must have 'type' and 'scale' keys
        mock_compute.return_value = (
            [
                [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 1.0, 1.0, 1.0],
                    }
                ],  # Level 0
                [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 2.0, 2.0, 2.0],
                    }
                ],  # Level 1
                [
                    {
                        "type": "scale",
                        "scale": [1.0, 1.0, 4.0, 4.0, 4.0],
                    }
                ],  # Level 2
            ],
            [{"chunks": (64, 64, 64)} for _ in range(3)],
        )
        mock_downscale.return_value = [[0, 0, 0, 0, 0]]
        mock_add_multiscales.return_value = {"attributes": {"ome": {}}}

        result = write_ome_ngff_metadata(
            arr_shape=[1, 1, 100, 200, 300],
            chunk_size=[1, 1, 64, 128, 128],
            image_name="test_image",
            n_lvls=3,
            scale_factors=(2, 2, 2),
            voxel_size=(1.0, 0.5, 0.5),
        )

        # Verify functions were called
        mock_build_ome.assert_called_once()
        mock_compute.assert_called_once()
        mock_downscale.assert_not_called()  # No origin provided
        mock_add_multiscales.assert_called_once()

        # Verify result structure
        self.assertIn("attributes", result)


if __name__ == "__main__":
    unittest.main()
