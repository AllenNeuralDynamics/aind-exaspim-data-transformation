"""Tests for imaris_to_zarr_writer function"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, call

import dask.array as da
import numpy as np
import zarr

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    ImarisReader,
    imaris_to_zarr_writer,
)


class TestImarisToZarrWriter(unittest.TestCase):
    """Test suite for imaris_to_zarr_writer function"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_imaris_path = "/fake/input/test.ims"
        self.test_output_path = "/fake/output"
        self.test_voxel_size = [1.0, 0.5, 0.5]
        self.test_chunk_size = [64, 64, 64]
        self.test_scale_factor = [2, 2, 2]

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.zarr.open_group")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.to_zarr")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_basic_conversion(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_to_zarr,
        mock_zarr_open,
    ):
        """Test basic Imaris to Zarr conversion"""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_shape.return_value = (100, 200, 300)
        mock_reader.n_levels = 3
        
        # Create mock dask arrays
        mock_dask_arrays = [
            MagicMock(shape=(100, 200, 300), chunksize=(64, 64, 64), dtype=np.uint16),
            MagicMock(shape=(50, 100, 150), chunksize=(64, 64, 64), dtype=np.uint16),
            MagicMock(shape=(25, 50, 75), chunksize=(64, 64, 64), dtype=np.uint16),
        ]
        mock_reader.get_dask_pyramid.return_value = mock_dask_arrays
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)
        
        # Mock Path
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: MagicMock()
        mock_path_cls.return_value = mock_path
        
        # Mock zarr group
        mock_root = MagicMock()
        mock_zarr_array = MagicMock()
        mock_root.create_dataset.return_value = mock_zarr_array
        mock_zarr_open.return_value = mock_root
        
        # Mock metadata
        mock_write_metadata.return_value = {"multiscales": []}
        
        # Call function
        imaris_to_zarr_writer(
            imaris_path=self.test_imaris_path,
            output_path=self.test_output_path,
            voxel_size=self.test_voxel_size,
            chunk_size=self.test_chunk_size,
            scale_factor=self.test_scale_factor,
            n_lvls=3,
        )
        
        # Assertions
        mock_imaris_reader_cls.assert_called_once_with(self.test_imaris_path)
        mock_reader.get_dask_pyramid.assert_called_once()
        self.assertEqual(mock_root.create_dataset.call_count, 3)
        self.assertEqual(mock_to_zarr.call_count, 3)
        mock_write_metadata.assert_called_once()
        mock_root.attrs.update.assert_called_once()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.zarr.open_group")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.to_zarr")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_with_voxel_size_extraction(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_to_zarr,
        mock_zarr_open,
    ):
        """Test conversion with voxel size extracted from file"""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_shape.return_value = (100, 200, 300)
        mock_reader.n_levels = 2
        mock_reader.get_voxel_size.return_value = ([1.5, 0.75, 0.75], b"um")
        
        mock_dask_arrays = [
            MagicMock(shape=(100, 200, 300), chunksize=(64, 64, 64), dtype=np.uint16),
            MagicMock(shape=(50, 100, 150), chunksize=(64, 64, 64), dtype=np.uint16),
        ]
        mock_reader.get_dask_pyramid.return_value = mock_dask_arrays
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)
        
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: MagicMock()
        mock_path_cls.return_value = mock_path
        
        mock_root = MagicMock()
        mock_zarr_array = MagicMock()
        mock_root.create_dataset.return_value = mock_zarr_array
        mock_zarr_open.return_value = mock_root
        
        mock_write_metadata.return_value = {"multiscales": []}
        
        # Call function without voxel_size
        imaris_to_zarr_writer(
            imaris_path=self.test_imaris_path,
            output_path=self.test_output_path,
            n_lvls=2,
        )
        
        # Verify voxel size was extracted
        mock_reader.get_voxel_size.assert_called_once()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.zarr.open_group")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.to_zarr")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_with_s3_bucket(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_to_zarr,
        mock_zarr_open,
    ):
        """Test conversion with S3 bucket specified"""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_shape.return_value = (100, 200, 300)
        mock_reader.n_levels = 1
        
        mock_dask_arrays = [
            MagicMock(shape=(100, 200, 300), chunksize=(64, 64, 64), dtype=np.uint16),
        ]
        mock_reader.get_dask_pyramid.return_value = mock_dask_arrays
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)
        
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: MagicMock()
        mock_path_cls.return_value = mock_path
        
        mock_root = MagicMock()
        mock_zarr_array = MagicMock()
        mock_root.create_dataset.return_value = mock_zarr_array
        mock_zarr_open.return_value = mock_root
        
        mock_write_metadata.return_value = {"multiscales": []}
        
        # Call function with bucket
        imaris_to_zarr_writer(
            imaris_path=self.test_imaris_path,
            output_path=self.test_output_path,
            voxel_size=self.test_voxel_size,
            bucket_name="test-bucket",
            n_lvls=1,
        )
        
        # Verify S3 path was constructed
        args, kwargs = mock_zarr_open.call_args
        self.assertIn("s3://", args[0])
        self.assertIn("test-bucket", args[0])

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.zarr.open_group")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.to_zarr")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_fewer_levels_than_requested(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_to_zarr,
        mock_zarr_open,
    ):
        """Test when file has fewer levels than requested"""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_shape.return_value = (100, 200, 300)
        mock_reader.n_levels = 2  # Only 2 levels available
        
        mock_dask_arrays = [
            MagicMock(shape=(100, 200, 300), chunksize=(64, 64, 64), dtype=np.uint16),
            MagicMock(shape=(50, 100, 150), chunksize=(64, 64, 64), dtype=np.uint16),
        ]
        mock_reader.get_dask_pyramid.return_value = mock_dask_arrays
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)
        
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: MagicMock()
        mock_path_cls.return_value = mock_path
        
        mock_root = MagicMock()
        mock_zarr_array = MagicMock()
        mock_root.create_dataset.return_value = mock_zarr_array
        mock_zarr_open.return_value = mock_root
        
        mock_write_metadata.return_value = {"multiscales": []}
        
        # Request 5 levels but only 2 available
        imaris_to_zarr_writer(
            imaris_path=self.test_imaris_path,
            output_path=self.test_output_path,
            voxel_size=self.test_voxel_size,
            n_lvls=5,
        )
        
        # Should only create 2 levels
        self.assertEqual(mock_root.create_dataset.call_count, 2)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.zarr.open_group")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.to_zarr")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Blosc")
    def test_custom_compressor(
        self,
        mock_blosc,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_to_zarr,
        mock_zarr_open,
    ):
        """Test with custom compressor settings"""
        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_shape.return_value = (100, 200, 300)
        mock_reader.n_levels = 1
        
        mock_dask_arrays = [
            MagicMock(shape=(100, 200, 300), chunksize=(64, 64, 64), dtype=np.uint16),
        ]
        mock_reader.get_dask_pyramid.return_value = mock_dask_arrays
        
        mock_imaris_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)
        
        mock_path = MagicMock()
        mock_path.__truediv__ = lambda self, other: MagicMock()
        mock_path_cls.return_value = mock_path
        
        mock_root = MagicMock()
        mock_zarr_array = MagicMock()
        mock_root.create_dataset.return_value = mock_zarr_array
        mock_zarr_open.return_value = mock_root
        
        mock_write_metadata.return_value = {"multiscales": []}
        
        custom_compressor = {"cname": "lz4", "clevel": 5, "shuffle": 1}
        
        # Call with custom compressor
        imaris_to_zarr_writer(
            imaris_path=self.test_imaris_path,
            output_path=self.test_output_path,
            voxel_size=self.test_voxel_size,
            compressor_kwargs=custom_compressor,
            n_lvls=1,
        )
        
        # Verify Blosc was called with custom settings
        mock_blosc.assert_called_once_with(**custom_compressor)


if __name__ == "__main__":
    unittest.main()
