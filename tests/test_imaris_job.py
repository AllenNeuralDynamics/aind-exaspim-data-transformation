"""Tests for ImarisCompressionJob class"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

from aind_exaspim_data_transformation.imaris_job import ImarisCompressionJob
from aind_exaspim_data_transformation.models import (
    CompressorName,
    ImarisJobSettings,
)


class TestImarisCompressionJob(unittest.TestCase):
    """Test suite for ImarisCompressionJob class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=2,
            partition_to_process=0,
            compressor_name=CompressorName.BLOSC,
            compressor_kwargs={"cname": "zstd", "clevel": 3, "shuffle": 1},
            chunk_size=[128, 128, 128],
            scale_factor=[2, 2, 2],
            downsample_levels=3,
        )

    def test_partition_list_basic(self):
        """Test partition_list static method with basic input"""
        test_list = [1, 2, 3, 4, 5, 6]
        result = ImarisCompressionJob.partition_list(test_list, 2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [1, 3, 5])
        self.assertEqual(result[1], [2, 4, 6])

    def test_partition_list_uneven(self):
        """Test partition_list with uneven distribution"""
        test_list = [1, 2, 3, 4, 5]
        result = ImarisCompressionJob.partition_list(test_list, 2)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], [1, 3, 5])
        self.assertEqual(result[1], [2, 4])

    def test_partition_list_single_partition(self):
        """Test partition_list with single partition"""
        test_list = [1, 2, 3]
        result = ImarisCompressionJob.partition_list(test_list, 1)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], [1, 2, 3])

    def test_partition_list_more_partitions_than_items(self):
        """Test partition_list with more partitions than items"""
        test_list = [1, 2]
        result = ImarisCompressionJob.partition_list(test_list, 5)

        self.assertEqual(len(result), 5)
        self.assertEqual(result[0], [1])
        self.assertEqual(result[1], [2])
        self.assertEqual(result[2], [])
        self.assertEqual(result[3], [])
        self.assertEqual(result[4], [])

    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_get_partitioned_list_ims_files(self, mock_path_cls):
        """Test _get_partitioned_list_of_stack_paths finds .ims files"""
        job = ImarisCompressionJob(job_settings=self.test_settings)

        # Mock Path objects
        mock_file1 = MagicMock(spec=Path)
        mock_file1.is_file.return_value = True
        mock_file1.__str__ = lambda x: "/fake/input/file1.ims"

        mock_file2 = MagicMock(spec=Path)
        mock_file2.is_file.return_value = True
        mock_file2.__str__ = lambda x: "/fake/input/file2.ims"

        mock_file3 = MagicMock(spec=Path)
        mock_file3.is_file.return_value = True
        mock_file3.__str__ = lambda x: "/fake/input/file3.ims"

        mock_file4 = MagicMock(spec=Path)
        mock_file4.is_file.return_value = True
        mock_file4.__str__ = lambda x: "/fake/input/file4.ims"

        # Mock glob to return .ims files
        mock_path = MagicMock()
        mock_path.glob.return_value = [
            mock_file1,
            mock_file2,
            mock_file3,
            mock_file4,
        ]
        mock_path_cls.return_value = mock_path

        result = job._get_partitioned_list_of_stack_paths()

        # Should partition into 2 groups
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)
        self.assertEqual(len(result[1]), 2)

    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_get_partitioned_list_fallback_h5(self, mock_path_cls):
        """Test _get_partitioned_list_of_stack_paths falls back to .h5 files"""
        job = ImarisCompressionJob(job_settings=self.test_settings)

        mock_file1 = MagicMock(spec=Path)
        mock_file1.is_file.return_value = True
        mock_file1.__str__ = lambda x: "/fake/input/file1.h5"

        mock_file2 = MagicMock(spec=Path)
        mock_file2.is_file.return_value = True
        mock_file2.__str__ = lambda x: "/fake/input/file2.h5"

        # Mock path that returns empty for .ims, then .h5 files
        mock_path = MagicMock()
        mock_path.glob.side_effect = [[], [mock_file1, mock_file2]]
        mock_path_cls.return_value = mock_path

        result = job._get_partitioned_list_of_stack_paths()

        # Should find h5 files
        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 1)
        self.assertEqual(len(result[1]), 1)

    @patch(
        "aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict"
    )
    def test_get_voxel_resolution_schema_1(self, mock_read_json):
        """Test _get_voxel_resolution with schema version < 2.0"""
        mock_acquisition_path = MagicMock()
        mock_acquisition_path.is_file.return_value = True

        mock_read_json.return_value = {
            "schema_version": "1.0.0",
            "tiles": [
                {
                    "coordinate_transformations": [
                        {"type": "scale", "scale": [0.5, 1.0, 1.0]}
                    ]
                }
            ],
        }

        result = ImarisCompressionJob._get_voxel_resolution(
            mock_acquisition_path
        )

        self.assertEqual(result, [1.0, 1.0, 0.5])  # [Z, Y, X]

    @patch(
        "aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict"
    )
    def test_get_voxel_resolution_schema_2(self, mock_read_json):
        """Test _get_voxel_resolution with schema version >= 2.0"""
        mock_acquisition_path = MagicMock()
        mock_acquisition_path.is_file.return_value = True

        mock_read_json.return_value = {
            "schema_version": "2.0.0",
            "data_streams": [
                {
                    "configurations": [
                        {
                            "images": [
                                {
                                    "image_to_acquisition_transform": [
                                        {
                                            "object_type": "Scale",
                                            "scale": [0.75, 1.5, 1.5],
                                        }
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
        }

        result = ImarisCompressionJob._get_voxel_resolution(
            mock_acquisition_path
        )

        self.assertEqual(result, [1.5, 1.5, 0.75])  # [Z, Y, X]

    def test_get_voxel_resolution_file_not_found(self):
        """Test _get_voxel_resolution raises error when file not found"""
        mock_acquisition_path = MagicMock()
        mock_acquisition_path.is_file.return_value = False

        with self.assertRaises(FileNotFoundError):
            ImarisCompressionJob._get_voxel_resolution(mock_acquisition_path)

    @patch(
        "aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict"
    )
    def test_get_voxel_resolution_schema_2_invalid_structure(
        self, mock_read_json
    ):
        """Test _get_voxel_resolution_schema_2 with invalid structure"""
        mock_read_json.return_value = {
            "schema_version": "2.0.0",
            "data_streams": [],  # Empty, will cause IndexError
        }

        mock_acquisition_path = MagicMock()
        mock_acquisition_path.is_file.return_value = True

        with self.assertRaises(ValueError):
            ImarisCompressionJob._get_voxel_resolution(mock_acquisition_path)

    @patch("aind_exaspim_data_transformation.imaris_job.ImarisReader")
    def test_get_voxel_size_from_imaris(self, mock_imaris_reader_cls):
        """Test _get_voxel_size_from_imaris method"""
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([2.0, 1.0, 1.0], b"um")

        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        mock_path = MagicMock()
        mock_path.name = "test.ims"

        result = ImarisCompressionJob._get_voxel_size_from_imaris(mock_path)

        self.assertEqual(result, [2.0, 1.0, 1.0])
        mock_reader.get_voxel_size.assert_called_once()

    def test_get_compressor_blosc(self):
        """Test _get_compressor returns compressor kwargs for BLOSC"""
        job = ImarisCompressionJob(job_settings=self.test_settings)

        result = job._get_compressor()

        self.assertEqual(result, {"cname": "zstd", "clevel": 3, "shuffle": 1})

    def test_get_compressor_none(self):
        """Test _get_compressor returns None when no compressor set"""
        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=1,
            partition_to_process=0,
            compressor_name=None,
        )
        job = ImarisCompressionJob(job_settings=settings)

        result = job._get_compressor()

        self.assertIsNone(result)

    @patch("aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_writer")
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_with_acquisition_json(
        self, mock_path_cls, mock_writer
    ):
        """Test _write_stacks with acquisition.json present"""
        # Create settings with mocked input_source that has working joinpath
        mock_input_source = MagicMock()
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = True
        mock_input_source.joinpath.return_value = mock_acq_path

        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=2,
            partition_to_process=0,
            compressor_name=CompressorName.BLOSC,
            compressor_kwargs={"cname": "zstd", "clevel": 3, "shuffle": 1},
            chunk_size=[128, 128, 128],
            scale_factor=[2, 2, 2],
            downsample_levels=3,
        )
        job = ImarisCompressionJob(job_settings=settings)

        # Override the input_source on the job settings for mocking
        job.job_settings.input_source = mock_input_source

        # Mock stack files
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        mock_output_path = MagicMock()
        mock_path_cls.return_value = mock_output_path

        with patch.object(
            job, "_get_voxel_resolution", return_value=[1.0, 0.5, 0.5]
        ):
            job._write_stacks([mock_stack1])

        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        self.assertEqual(kwargs["imaris_path"], "/fake/input/stack1.ims")
        self.assertEqual(kwargs["voxel_size"], [1.0, 0.5, 0.5])

    @patch("aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_writer")
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    @patch("aind_exaspim_data_transformation.imaris_job.ImarisReader")
    def test_write_stacks_without_acquisition_json(
        self, mock_imaris_reader_cls, mock_path_cls, mock_writer
    ):
        """Test _write_stacks without acquisition.json.

        Voxel size should be extracted from the Imaris file.
        """
        job = ImarisCompressionJob(job_settings=self.test_settings)

        # Mock stack files
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.5, 0.75, 0.75], b"um")
        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path for acquisition.json not existing
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        job._write_stacks([mock_stack1])

        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        self.assertEqual(kwargs["voxel_size"], [1.5, 0.75, 0.75])

    @patch("aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_writer")
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    @patch("aind_exaspim_data_transformation.imaris_job.urlparse")
    def test_write_stacks_with_s3(
        self, mock_urlparse, mock_path_cls, mock_writer
    ):
        """Test _write_stacks with S3 location"""
        settings_with_s3 = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            s3_location="s3://my-bucket/my-prefix",
            num_of_partitions=1,
            partition_to_process=0,
            compressor_name=CompressorName.BLOSC,
        )
        job = ImarisCompressionJob(job_settings=settings_with_s3)

        # Mock urlparse
        mock_parsed = MagicMock()
        mock_parsed.netloc = "my-bucket"
        mock_parsed.path = "/my-prefix"
        mock_urlparse.return_value = mock_parsed

        # Mock stack
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        # Mock paths
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if isinstance(arg, str) and "input" in arg:
                return mock_input_path
            return mock_output_path

        mock_path_cls.side_effect = path_side_effect

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            job._write_stacks([mock_stack1])

        mock_writer.assert_called_once()
        args, kwargs = mock_writer.call_args
        self.assertEqual(kwargs["bucket_name"], "my-bucket")

    def test_write_stacks_empty_list(self):
        """Test _write_stacks with empty list does nothing"""
        job = ImarisCompressionJob(job_settings=self.test_settings)

        # Should not raise error
        job._write_stacks([])

    @patch("aind_exaspim_data_transformation.imaris_job.utils.sync_dir_to_s3")
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_upload_derivatives_folder_exists(self, mock_path_cls, mock_sync):
        """Test _upload_derivatives_folder when folder exists"""
        settings_with_s3 = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            s3_location="s3://my-bucket/prefix",
            num_of_partitions=1,
            partition_to_process=0,
        )
        job = ImarisCompressionJob(job_settings=settings_with_s3)

        # Mock derivatives path
        mock_deriv_path = MagicMock()
        mock_deriv_path.exists.return_value = True

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_deriv_path

        mock_path_cls.return_value = mock_input_path

        job._upload_derivatives_folder()

        mock_sync.assert_called_once()

    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_upload_derivatives_folder_not_exists(self, mock_path_cls):
        """Test _upload_derivatives_folder when folder doesn't exist"""
        settings_with_s3 = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            s3_location="s3://my-bucket/prefix",
            num_of_partitions=1,
            partition_to_process=0,
        )
        job = ImarisCompressionJob(job_settings=settings_with_s3)

        # Mock derivatives path not existing
        mock_deriv_path = MagicMock()
        mock_deriv_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_deriv_path

        mock_path_cls.return_value = mock_input_path

        # Should not raise error
        job._upload_derivatives_folder()

    @patch("aind_exaspim_data_transformation.imaris_job.time")
    def test_run_job(self, mock_time):
        """Test run_job method"""
        mock_time.return_value = 1000.0

        job = ImarisCompressionJob(job_settings=self.test_settings)

        with patch.object(
            job, "_get_partitioned_list_of_stack_paths"
        ) as mock_get_list:
            with patch.object(
                job, "_upload_derivatives_folder"
            ) as mock_upload:
                with patch.object(job, "_write_stacks") as mock_write:
                    mock_get_list.return_value = [["file1.ims"], ["file2.ims"]]

                    response = job.run_job()

        self.assertEqual(response.status_code, 200)
        self.assertIn("Job finished", response.message)
        mock_get_list.assert_called_once()
        mock_upload.assert_called_once()  # partition 0
        mock_write.assert_called_once()

    @patch(
        "aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_distributed"
    )
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_with_tensorstore(
        self, mock_path_cls, mock_distributed_writer
    ):
        """Test _write_stacks with use_tensorstore=True"""
        settings_with_tensorstore = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=1,
            partition_to_process=0,
            use_tensorstore=True,
            translate_imaris_pyramid=False,
            # Use distributed writer, not translator
            chunk_size=[128, 128, 128],
            shard_size=[256, 256, 256],
            downsample_levels=3,
            tensorstore_batch_size=8,
        )
        job = ImarisCompressionJob(job_settings=settings_with_tensorstore)

        # Mock stack files
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        # Mock Path for acquisition.json
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            job._write_stacks([mock_stack1])

        # Verify distributed writer was called instead of standard writer
        mock_distributed_writer.assert_called_once()
        args, kwargs = mock_distributed_writer.call_args
        self.assertEqual(kwargs["imaris_path"], "/fake/input/stack1.ims")
        self.assertEqual(kwargs["voxel_size"], [1.0, 0.5, 0.5])
        self.assertEqual(kwargs["chunk_shape"], (128, 128, 128))
        self.assertEqual(kwargs["shard_shape"], (256, 256, 256))

    @patch(
        "aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_distributed"
    )
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_tensorstore_with_s3(
        self, mock_path_cls, mock_distributed_writer
    ):
        """Test _write_stacks with TensorStore and S3 output"""
        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            s3_location="s3://my-bucket/my-prefix",
            num_of_partitions=1,
            partition_to_process=0,
            use_tensorstore=True,
            translate_imaris_pyramid=False,
            # Use distributed writer, not translator
        )
        job = ImarisCompressionJob(job_settings=settings)

        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if isinstance(arg, str) and "input" in arg:
                return mock_input_path
            return mock_output_path

        mock_path_cls.side_effect = path_side_effect

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            job._write_stacks([mock_stack1])

        mock_distributed_writer.assert_called_once()
        args, kwargs = mock_distributed_writer.call_args
        self.assertEqual(kwargs["bucket_name"], "my-bucket")

    @patch(
        "aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_translate_pyramid"
    )
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_with_translate_pyramid(
        self, mock_path_cls, mock_translate_writer
    ):
        """Test _write_stacks with translate_imaris_pyramid=True"""
        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=1,
            partition_to_process=0,
            use_tensorstore=True,
            translate_imaris_pyramid=True,  # Use translator
            chunk_size=[128, 128, 128],
            shard_size=[256, 256, 256],
            downsample_levels=3,
            tensorstore_batch_size=16,
        )
        job = ImarisCompressionJob(job_settings=settings)

        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            job._write_stacks([mock_stack1])

        # Verify translate writer was called
        mock_translate_writer.assert_called_once()
        args, kwargs = mock_translate_writer.call_args
        self.assertEqual(kwargs["imaris_path"], "/fake/input/stack1.ims")
        self.assertEqual(kwargs["max_concurrent_writes"], 16)

    @patch("aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_writer")
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_acquisition_json_error(
        self, mock_path_cls, mock_writer
    ):
        """Test _write_stacks handles acquisition.json read error gracefully"""
        job = ImarisCompressionJob(job_settings=self.test_settings)

        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        # Mock acquisition.json exists but raises error when reading
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = True

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        # Make _get_voxel_resolution raise an exception
        with patch.object(
            ImarisCompressionJob,
            "_get_voxel_resolution",
            side_effect=ValueError("Invalid JSON"),
        ):
            with patch.object(
                job,
                "_get_voxel_size_from_imaris",
                return_value=[1.0, 0.5, 0.5],
            ):
                # Should not raise - falls back to extracting from imaris file
                job._write_stacks([mock_stack1])

        mock_writer.assert_called_once()

    @patch(
        "aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_distributed"
    )
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_with_dask_workers(
        self, mock_path_cls, mock_distributed_writer
    ):
        """Test _write_stacks creates Dask cluster when dask_workers > 0"""
        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=1,
            partition_to_process=0,
            use_tensorstore=True,
            translate_imaris_pyramid=False,
            dask_workers=4,  # Request 4 workers
        )
        job = ImarisCompressionJob(job_settings=settings)

        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        # Mock Dask client and cluster
        mock_client = MagicMock()
        mock_client.dashboard_link = "http://localhost:8787"
        mock_cluster = MagicMock()

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            # Mock the dask.distributed imports inside the function
            with patch.dict(
                "sys.modules",
                {
                    "dask.distributed": MagicMock(
                        Client=MagicMock(return_value=mock_client),
                        LocalCluster=MagicMock(return_value=mock_cluster),
                    )
                },
            ):
                job._write_stacks([mock_stack1])

        # Verify distributed writer was called with dask_client
        mock_distributed_writer.assert_called_once()
        args, kwargs = mock_distributed_writer.call_args
        self.assertEqual(kwargs["dask_client"], mock_client)

        # Verify cleanup
        mock_client.close.assert_called_once()
        mock_cluster.close.assert_called_once()

    @patch(
        "aind_exaspim_data_transformation.imaris_job.imaris_to_zarr_distributed"
    )
    @patch("aind_exaspim_data_transformation.imaris_job.Path")
    def test_write_stacks_dask_import_error(
        self, mock_path_cls, mock_distributed_writer
    ):
        """Test _write_stacks handles missing dask.distributed gracefully"""
        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=1,
            partition_to_process=0,
            use_tensorstore=True,
            translate_imaris_pyramid=False,
            dask_workers=4,  # Request workers but dask not available
        )
        job = ImarisCompressionJob(job_settings=settings)

        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"

        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = False

        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path

        mock_output_path = MagicMock()

        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()

        mock_path_cls.side_effect = path_side_effect

        with patch.object(
            job, "_get_voxel_size_from_imaris", return_value=[1.0, 0.5, 0.5]
        ):
            # Simulate ImportError for dask.distributed
            import builtins

            original_import = builtins.__import__

            def mock_import(name, *args, **kwargs):
                if "dask.distributed" in name:
                    raise ImportError("No module named dask.distributed")
                return original_import(name, *args, **kwargs)

            with patch.object(builtins, "__import__", side_effect=mock_import):
                job._write_stacks([mock_stack1])

        # Should still call distributed writer, just with dask_client=None
        mock_distributed_writer.assert_called_once()
        args, kwargs = mock_distributed_writer.call_args
        self.assertIsNone(kwargs["dask_client"])

    @patch("aind_exaspim_data_transformation.imaris_job.time")
    def test_run_job_partition_not_zero(self, mock_time):
        """Test run_job doesn't upload derivatives for partition != 0"""
        mock_time.return_value = 1000.0

        settings = ImarisJobSettings(
            input_source="/fake/input",
            output_directory="/fake/output",
            num_of_partitions=2,
            partition_to_process=1,  # Not 0
        )
        job = ImarisCompressionJob(job_settings=settings)

        with patch.object(
            job, "_get_partitioned_list_of_stack_paths"
        ) as mock_get_list:
            with patch.object(
                job, "_upload_derivatives_folder"
            ) as mock_upload:
                with patch.object(job, "_write_stacks") as mock_write:
                    mock_get_list.return_value = [["file1.ims"], ["file2.ims"]]

                    response = job.run_job()

        self.assertEqual(response.status_code, 200)
        mock_upload.assert_not_called()  # Not called for partition 1


class TestJobEntrypoint(unittest.TestCase):
    """Test suite for job_entrypoint function"""

    @patch("aind_exaspim_data_transformation.imaris_job.ImarisCompressionJob")
    @patch("aind_exaspim_data_transformation.imaris_job.ImarisJobSettings")
    @patch("aind_exaspim_data_transformation.imaris_job.get_parser")
    @patch("aind_exaspim_data_transformation.imaris_job.multiprocessing")
    def test_job_entrypoint_with_job_settings(
        self, mock_mp, mock_get_parser, mock_settings_cls, mock_job_cls
    ):
        """Test job_entrypoint with --job-settings argument"""
        from aind_exaspim_data_transformation.imaris_job import job_entrypoint

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.job_settings = (
            '{"input_source": "/input", "output_directory": "/output", '
            '"num_of_partitions": 1, "partition_to_process": 0}'
        )
        mock_args.config_file = None
        mock_parser.parse_args.return_value = mock_args
        mock_get_parser.return_value = mock_parser

        mock_settings = MagicMock()
        mock_settings_cls.model_validate_json.return_value = mock_settings

        mock_job = MagicMock()
        mock_job.run_job.return_value = MagicMock(
            model_dump_json=lambda: '{"status": 200}'
        )
        mock_job_cls.return_value = mock_job

        job_entrypoint(["--job-settings", "{}"])

        mock_mp.set_start_method.assert_called_once_with("spawn", force=True)
        mock_settings_cls.model_validate_json.assert_called_once()
        mock_job.run_job.assert_called_once()

    @patch("aind_exaspim_data_transformation.imaris_job.ImarisCompressionJob")
    @patch("aind_exaspim_data_transformation.imaris_job.ImarisJobSettings")
    @patch("aind_exaspim_data_transformation.imaris_job.get_parser")
    @patch("aind_exaspim_data_transformation.imaris_job.multiprocessing")
    def test_job_entrypoint_with_config_file(
        self, mock_mp, mock_get_parser, mock_settings_cls, mock_job_cls
    ):
        """Test job_entrypoint with --config-file argument"""
        from aind_exaspim_data_transformation.imaris_job import job_entrypoint

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.job_settings = None
        mock_args.config_file = "/path/to/config.json"
        mock_parser.parse_args.return_value = mock_args
        mock_get_parser.return_value = mock_parser

        mock_settings = MagicMock()
        mock_settings_cls.from_config_file.return_value = mock_settings

        mock_job = MagicMock()
        mock_job.run_job.return_value = MagicMock(
            model_dump_json=lambda: '{"status": 200}'
        )
        mock_job_cls.return_value = mock_job

        job_entrypoint([])

        mock_settings_cls.from_config_file.assert_called_once_with(
            "/path/to/config.json"
        )

    @patch("aind_exaspim_data_transformation.imaris_job.ImarisCompressionJob")
    @patch("aind_exaspim_data_transformation.imaris_job.ImarisJobSettings")
    @patch("aind_exaspim_data_transformation.imaris_job.get_parser")
    @patch("aind_exaspim_data_transformation.imaris_job.multiprocessing")
    def test_job_entrypoint_with_env_vars(
        self, mock_mp, mock_get_parser, mock_settings_cls, mock_job_cls
    ):
        """Test job_entrypoint with no arguments (env vars)"""
        from aind_exaspim_data_transformation.imaris_job import job_entrypoint

        mock_parser = MagicMock()
        mock_args = MagicMock()
        mock_args.job_settings = None
        mock_args.config_file = None
        mock_parser.parse_args.return_value = mock_args
        mock_get_parser.return_value = mock_parser

        mock_settings = MagicMock()
        mock_settings_cls.return_value = mock_settings

        mock_job = MagicMock()
        mock_job.run_job.return_value = MagicMock(
            model_dump_json=lambda: '{"status": 200}'
        )
        mock_job_cls.return_value = mock_job

        job_entrypoint([])

        # Settings created from env vars (no arguments)
        mock_settings_cls.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
