"""
Tests to verify that translation fields are included in zarr.json metadata
for downsampled pyramid levels.
"""

import unittest

from aind_exaspim_data_transformation.compress.omezarr_metadata import (
    write_ome_ngff_metadata,
)


class TestTranslationFields(unittest.TestCase):
    """Test that translation coordinate transformations are properly generated."""

    def setUp(self):
        """Set up test parameters."""
        self.arr_shape = [1, 1, 768, 2688, 3584]
        self.chunk_size = [1, 1, 128, 128, 128]
        self.image_name = "test_image"
        self.n_lvls = 5
        self.scale_factors = (1.0, 2.0, 2.0)  # Z, Y, X
        self.voxel_size = (1.0, 1.0, 1.0)  # Z, Y, X in micrometers
        self.channel_names = ["test_channel"]
        self.origin = [0.0, 0.0, 0.0]  # Z, Y, X origin

    def test_translation_fields_present_with_origin(self):
        """Test that all pyramid levels have translation fields when origin is provided."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=self.channel_names,
            origin=self.origin,
        )

        multiscales = metadata["attributes"]["ome"]["multiscales"][0]
        datasets = multiscales["datasets"]

        # Check that we have the expected number of levels
        self.assertEqual(len(datasets), self.n_lvls)

        # Verify each level has a translation field
        for i, dataset in enumerate(datasets):
            with self.subTest(level=i):
                transforms = dataset.get("coordinateTransformations", [])
                
                # Find translation transform
                translation_transform = None
                for transform in transforms:
                    if transform.get("type") == "translation":
                        translation_transform = transform
                        break

                # Assert translation exists
                self.assertIsNotNone(
                    translation_transform,
                    f"Level {i} is missing translation field"
                )

                # Verify translation is a list of 5 floats
                translation = translation_transform.get("translation")
                self.assertIsInstance(translation, list)
                self.assertEqual(len(translation), 5)
                for val in translation:
                    self.assertIsInstance(val, (int, float))

    def test_translation_values_follow_expected_pattern(self):
        """Test that translation values follow the expected downsampling pattern."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=self.channel_names,
            origin=self.origin,
        )

        multiscales = metadata["attributes"]["ome"]["multiscales"][0]
        datasets = multiscales["datasets"]

        # Expected translations for 2x downsampling with scale_factors=(1, 2, 2)
        expected_translations = [
            [0.0, 0.0, 0.0, 0.0, 0.0],    # Level 0 (1x)
            [0.0, 0.0, 0.0, 0.5, 0.5],    # Level 1 (2x)
            [0.0, 0.0, 0.0, 1.5, 1.5],    # Level 2 (4x)
            [0.0, 0.0, 0.0, 3.5, 3.5],    # Level 3 (8x)
            [0.0, 0.0, 0.0, 7.5, 7.5],    # Level 4 (16x)
        ]

        for i, (dataset, expected) in enumerate(zip(datasets, expected_translations)):
            with self.subTest(level=i):
                transforms = dataset.get("coordinateTransformations", [])
                
                # Find translation transform
                translation = None
                for transform in transforms:
                    if transform.get("type") == "translation":
                        translation = transform.get("translation")
                        break

                self.assertIsNotNone(translation)
                self.assertEqual(
                    translation,
                    expected,
                    f"Level {i} translation mismatch"
                )

    def test_scale_and_translation_both_present(self):
        """Test that both scale and translation transforms are present for each level."""
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=self.channel_names,
            origin=self.origin,
        )

        multiscales = metadata["attributes"]["ome"]["multiscales"][0]
        datasets = multiscales["datasets"]

        for i, dataset in enumerate(datasets):
            with self.subTest(level=i):
                transforms = dataset.get("coordinateTransformations", [])
                
                # Check for scale transform
                has_scale = any(t.get("type") == "scale" for t in transforms)
                self.assertTrue(has_scale, f"Level {i} missing scale transform")
                
                # Check for translation transform
                has_translation = any(t.get("type") == "translation" for t in transforms)
                self.assertTrue(has_translation, f"Level {i} missing translation transform")

    def test_without_origin_parameter(self):
        """Test that metadata can be generated without origin (backwards compatibility)."""
        # This should not raise an error, but translations won't be included
        metadata = write_ome_ngff_metadata(
            arr_shape=self.arr_shape,
            chunk_size=self.chunk_size,
            image_name=self.image_name,
            n_lvls=self.n_lvls,
            scale_factors=self.scale_factors,
            voxel_size=self.voxel_size,
            channel_names=self.channel_names,
            # origin parameter omitted
        )

        # Should still generate valid metadata
        self.assertIn("attributes", metadata)
        self.assertIn("ome", metadata["attributes"])
        multiscales = metadata["attributes"]["ome"]["multiscales"][0]
        datasets = multiscales["datasets"]
        self.assertEqual(len(datasets), self.n_lvls)


if __name__ == "__main__":
    unittest.main()
