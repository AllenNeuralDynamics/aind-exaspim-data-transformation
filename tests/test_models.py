"""Tests for models module"""

import unittest

from aind_exaspim_data_transformation.models import (
    CompressorName,
    ImarisJobSettings,
)


class TestCompressorName(unittest.TestCase):
    """Test suite for CompressorName enum"""

    def test_blosc_value(self):
        """Test BLOSC enum value"""
        self.assertEqual(CompressorName.BLOSC, "blosc")


class TestImarisJobSettings(unittest.TestCase):
    """Test suite for ImarisJobSettings model"""

    def test_minimal_settings(self):
        """Test creating settings with minimal required fields"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=4,
            partition_to_process=0,
        )

        self.assertEqual(settings.input_source, "/path/to/input")
        self.assertEqual(settings.output_directory, "/path/to/output")
        self.assertEqual(settings.num_of_partitions, 4)
        self.assertEqual(settings.partition_to_process, 0)

    def test_default_compressor(self):
        """Test default compressor settings"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertEqual(settings.compressor_name, CompressorName.BLOSC)
        self.assertEqual(settings.compressor_kwargs["cname"], "zstd")
        self.assertEqual(settings.compressor_kwargs["clevel"], 3)

    def test_default_chunk_size(self):
        """Test default chunk size"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertEqual(settings.chunk_size, [128, 128, 128])

    def test_default_scale_factor(self):
        """Test default scale factor"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertEqual(settings.scale_factor, [2, 2, 2])

    def test_default_downsample_levels(self):
        """Test default downsample levels"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertEqual(settings.downsample_levels, 5)

    def test_default_downsample_mode(self):
        """Test default downsample mode"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertEqual(settings.downsample_mode, "mean")

    def test_custom_settings(self):
        """Test creating settings with custom values"""
        settings = ImarisJobSettings(
            input_source="/custom/input",
            output_directory="/custom/output",
            s3_location="s3://my-bucket/prefix",
            num_of_partitions=8,
            partition_to_process=3,
            compressor_name=CompressorName.BLOSC,
            compressor_kwargs={"cname": "lz4", "clevel": 5, "shuffle": 2},
            chunk_size=[256, 256, 256],
            scale_factor=[3, 3, 3],
            downsample_levels=4,
            downsample_mode="median",
            shard_size=[1024, 1024, 1024],
        )

        self.assertEqual(settings.s3_location, "s3://my-bucket/prefix")
        self.assertEqual(settings.num_of_partitions, 8)
        self.assertEqual(settings.partition_to_process, 3)
        self.assertEqual(settings.compressor_kwargs["cname"], "lz4")
        self.assertEqual(settings.chunk_size, [256, 256, 256])
        self.assertEqual(settings.scale_factor, [3, 3, 3])
        self.assertEqual(settings.downsample_levels, 4)
        self.assertEqual(settings.downsample_mode, "median")
        self.assertEqual(settings.shard_size, [1024, 1024, 1024])

    def test_s3_location_optional(self):
        """Test that s3_location is optional"""
        settings = ImarisJobSettings(
            input_source="/path/to/input",
            output_directory="/path/to/output",
            num_of_partitions=1,
            partition_to_process=0,
        )

        self.assertIsNone(settings.s3_location)

    def test_valid_downsample_modes(self):
        """Test all valid downsample modes"""
        valid_modes = ["stride", "median", "mode", "mean", "min", "max"]

        for mode in valid_modes:
            settings = ImarisJobSettings(
                input_source="/path/to/input",
                output_directory="/path/to/output",
                num_of_partitions=1,
                partition_to_process=0,
                downsample_mode=mode,
            )
            self.assertEqual(settings.downsample_mode, mode)


if __name__ == "__main__":
    unittest.main()
