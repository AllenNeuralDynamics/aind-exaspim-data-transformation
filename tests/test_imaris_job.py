"""Tests for ImarisCompressionJob class"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

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
        mock_path.glob.return_value = [mock_file1, mock_file2, mock_file3, mock_file4]
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

    @patch("aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict")
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
        
        result = ImarisCompressionJob._get_voxel_resolution(mock_acquisition_path)
        
        self.assertEqual(result, [1.0, 1.0, 0.5])  # [Z, Y, X]

    @patch("aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict")
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
                                        {"object_type": "Scale", "scale": [0.75, 1.5, 1.5]}
                                    ]
                                }
                            ]
                        }
                    ]
                }
            ],
        }
        
        result = ImarisCompressionJob._get_voxel_resolution(mock_acquisition_path)
        
        self.assertEqual(result, [1.5, 1.5, 0.75])  # [Z, Y, X]

    def test_get_voxel_resolution_file_not_found(self):
        """Test _get_voxel_resolution raises error when file not found"""
        mock_acquisition_path = MagicMock()
        mock_acquisition_path.is_file.return_value = False
        
        with self.assertRaises(FileNotFoundError):
            ImarisCompressionJob._get_voxel_resolution(mock_acquisition_path)

    @patch("aind_exaspim_data_transformation.imaris_job.utils.read_json_as_dict")
    def test_get_voxel_resolution_schema_2_invalid_structure(self, mock_read_json):
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
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
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
        job = ImarisCompressionJob(job_settings=self.test_settings)
        
        # Mock stack files
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"
        
        # Mock Path for acquisition.json
        mock_acq_path = MagicMock()
        mock_acq_path.exists.return_value = True
        
        mock_input_path = MagicMock()
        mock_input_path.joinpath.return_value = mock_acq_path
        
        mock_output_path = MagicMock()
        
        # Control Path() calls
        def path_side_effect(arg):
            if arg == "/fake/input":
                return mock_input_path
            elif arg == "/fake/output":
                return mock_output_path
            return MagicMock()
        
        mock_path_cls.side_effect = path_side_effect
        
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
        """Test _write_stacks without acquisition.json (extracts from Imaris)"""
        job = ImarisCompressionJob(job_settings=self.test_settings)
        
        # Mock stack files
        mock_stack1 = MagicMock()
        mock_stack1.stem = "stack1"
        mock_stack1.__str__ = lambda x: "/fake/input/stack1.ims"
        
        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.5, 0.75, 0.75], b"um")
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
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
    def test_write_stacks_with_s3(self, mock_urlparse, mock_path_cls, mock_writer):
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
            with patch.object(job, "_upload_derivatives_folder") as mock_upload:
                with patch.object(job, "_write_stacks") as mock_write:
                    mock_get_list.return_value = [["file1.ims"], ["file2.ims"]]
                    
                    response = job.run_job()
        
        self.assertEqual(response.status_code, 200)
        self.assertIn("Job finished", response.message)
        mock_get_list.assert_called_once()
        mock_upload.assert_called_once()  # partition 0
        mock_write.assert_called_once()


if __name__ == "__main__":
    unittest.main()
