"""Tests for ImarisReader and DataReaderFactory"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import dask.array as da
import h5py
import numpy as np

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    DataReader,
    DataReaderFactory,
    ImarisReader,
    MissingDatasetError,
)


class TestImarisReader(unittest.TestCase):
    """Test suite for ImarisReader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)
        
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_init(self, mock_h5py_file):
        """Test ImarisReader initialization"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        reader = ImarisReader(self.test_filepath)
        
        self.assertEqual(reader.filepath, self.test_filepath)
        mock_h5py_file.assert_called_once_with(self.test_filepath, mode="r")
        self.assertEqual(reader.handle, self.mock_h5_file)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_context_manager(self, mock_h5py_file):
        """Test ImarisReader as context manager"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        with ImarisReader(self.test_filepath) as reader:
            self.assertIsNotNone(reader.handle)
        
        self.mock_h5_file.close.assert_called_once()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_shape(self, mock_h5py_file):
        """Test get_shape method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        # Mock dataset info
        mock_dataset_info = MagicMock()
        mock_dataset_info.attrs = {
            "Z": np.array(b"100"),
            "Y": np.array(b"200"),
            "X": np.array(b"300"),
        }
        self.mock_h5_file.__getitem__.return_value = mock_dataset_info
        
        reader = ImarisReader(self.test_filepath)
        shape = reader.get_shape()
        
        self.assertEqual(shape, (100, 200, 300))

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_voxel_size(self, mock_h5py_file):
        """Test get_voxel_size method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        # Mock dataset info with voxel size data
        mock_dataset_info = MagicMock()
        mock_dataset_info.attrs = {
            "X": np.array(b"1000"),
            "Y": np.array(b"1000"),
            "Z": np.array(b"500"),
            "ExtMin0": np.array(b"0.0"),
            "ExtMin1": np.array(b"0.0"),
            "ExtMin2": np.array(b"0.0"),
            "ExtMax0": np.array(b"1000.0"),
            "ExtMax1": np.array(b"1000.0"),
            "ExtMax2": np.array(b"500.0"),
            "Unit": np.array(b"um"),
        }
        self.mock_h5_file.__getitem__.return_value = mock_dataset_info
        
        reader = ImarisReader(self.test_filepath)
        voxel_size, unit = reader.get_voxel_size()
        
        self.assertEqual(len(voxel_size), 3)
        self.assertEqual(unit, b"um")
        self.assertAlmostEqual(voxel_size[0], 1.0)  # Z
        self.assertAlmostEqual(voxel_size[1], 1.0)  # Y
        self.assertAlmostEqual(voxel_size[2], 1.0)  # X

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_origin(self, mock_h5py_file):
        """Test get_origin method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        mock_dataset_info = MagicMock()
        mock_dataset_info.attrs = {
            "ExtMin0": np.array(b"10.5"),
            "ExtMin1": np.array(b"20.5"),
            "ExtMin2": np.array(b"30.5"),
        }
        self.mock_h5_file.__getitem__.return_value = mock_dataset_info
        
        reader = ImarisReader(self.test_filepath)
        origin = reader.get_origin()
        
        self.assertEqual(origin, [30.5, 20.5, 10.5])  # [Z, Y, X]

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_extent(self, mock_h5py_file):
        """Test get_extent method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        mock_dataset_info = MagicMock()
        mock_dataset_info.attrs = {
            "ExtMax0": np.array(b"100.5"),
            "ExtMax1": np.array(b"200.5"),
            "ExtMax2": np.array(b"300.5"),
        }
        self.mock_h5_file.__getitem__.return_value = mock_dataset_info
        
        reader = ImarisReader(self.test_filepath)
        extent = reader.get_extent()
        
        self.assertEqual(extent, [300.5, 200.5, 100.5])  # [Z, Y, X]

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.da.from_array")
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_as_dask_array(self, mock_h5py_file, mock_from_array):
        """Test as_dask_array method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.shape = (100, 200, 300)
        self.mock_h5_file.__getitem__.return_value = mock_dataset
        
        # Mock dask array
        mock_dask_array = MagicMock()
        mock_dask_array.__getitem__ = MagicMock(return_value=mock_dask_array)
        mock_from_array.return_value = mock_dask_array
        
        # Mock get_shape
        with patch.object(ImarisReader, "get_shape", return_value=(100, 200, 300)):
            reader = ImarisReader(self.test_filepath)
            result = reader.as_dask_array(chunks=True)
        
        mock_from_array.assert_called_once()
        self.assertEqual(result, mock_dask_array)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_chunks(self, mock_h5py_file):
        """Test get_chunks method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        mock_dataset = MagicMock()
        mock_dataset.chunks = (32, 64, 128)
        self.mock_h5_file.__getitem__.return_value = mock_dataset
        
        reader = ImarisReader(self.test_filepath)
        chunks = reader.get_chunks()
        
        self.assertEqual(chunks, (32, 64, 128))

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_itemsize(self, mock_h5py_file):
        """Test get_itemsize method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        mock_dataset = MagicMock()
        mock_dataset.dtype = np.dtype(np.uint16)
        self.mock_h5_file.__getitem__.return_value = mock_dataset
        
        reader = ImarisReader(self.test_filepath)
        itemsize = reader.get_itemsize()
        
        self.assertEqual(itemsize, 2)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_n_levels_property(self, mock_h5py_file):
        """Test n_levels property"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        # Mock that levels 0, 1, 2 exist but 3 doesn't
        def getitem_side_effect(key):
            if "ResolutionLevel 0" in key or "ResolutionLevel 1" in key or "ResolutionLevel 2" in key:
                return MagicMock()
            raise KeyError(key)
        
        self.mock_h5_file.__contains__ = lambda self, key: "ResolutionLevel 0" in key or "ResolutionLevel 1" in key or "ResolutionLevel 2" in key
        
        reader = ImarisReader(self.test_filepath)
        
        # Mock the contains check
        with patch.object(reader, 'get_handle') as mock_get_handle:
            mock_handle = MagicMock()
            mock_get_handle.return_value = mock_handle
            
            def mock_contains(path):
                level = int(path.split("ResolutionLevel ")[1].split("/")[0])
                return level < 3
            
            mock_handle.__contains__ = mock_contains
            n_levels = reader.n_levels
        
        self.assertEqual(n_levels, 3)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_res_lvl(self, mock_h5py_file):
        """Test _get_res_lvl internal method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        reader = ImarisReader(self.test_filepath)
        
        level = reader._get_res_lvl("/DataSet/ResolutionLevel 3/TimePoint 0/Channel 0/Data")
        self.assertEqual(level, 3)
        
        with self.assertRaises(ValueError):
            reader._get_res_lvl("/DataSet/InvalidPath/Data")

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_shape_at_lvl(self, mock_h5py_file):
        """Test _get_shape_at_lvl internal method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        with patch.object(ImarisReader, "get_shape", return_value=(100, 200, 300)):
            reader = ImarisReader(self.test_filepath)
            
            shape_lvl0 = reader._get_shape_at_lvl(0)
            self.assertEqual(shape_lvl0, (100, 200, 300))
            
            shape_lvl1 = reader._get_shape_at_lvl(1)
            self.assertEqual(shape_lvl1, (50, 100, 150))
            
            shape_lvl2 = reader._get_shape_at_lvl(2)
            self.assertEqual(shape_lvl2, (25, 50, 75))

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_close(self, mock_h5py_file):
        """Test close method"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        reader = ImarisReader(self.test_filepath)
        self.assertIsNotNone(reader.handle)
        
        reader.close()
        self.mock_h5_file.close.assert_called_once()
        self.assertIsNone(reader.handle)
        
        # Calling close again should not raise error
        reader.close()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File")
    def test_get_dask_pyramid_missing_dataset(self, mock_h5py_file):
        """Test get_dask_pyramid raises error for missing dataset"""
        mock_h5py_file.return_value = self.mock_h5_file
        
        reader = ImarisReader(self.test_filepath)
        
        with patch.object(reader, 'get_handle') as mock_get_handle:
            mock_handle = MagicMock()
            mock_get_handle.return_value = mock_handle
            mock_handle.__contains__ = lambda x: False
            
            with self.assertRaises(MissingDatasetError):
                reader.get_dask_pyramid(num_levels=2)


class TestDataReaderFactory(unittest.TestCase):
    """Test suite for DataReaderFactory class"""

    def test_init(self):
        """Test factory initialization"""
        factory = DataReaderFactory()
        self.assertEqual(factory.VALID_EXTENSIONS, [".h5", ".ims"])
        self.assertIn(".h5", factory.factory)
        self.assertIn(".ims", factory.factory)

    def test_get_valid_extensions(self):
        """Test get_valid_extensions method"""
        factory = DataReaderFactory()
        extensions = factory.get_valid_extensions()
        self.assertEqual(extensions, [".h5", ".ims"])

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    def test_create_ims_file(self, mock_imaris_reader):
        """Test creating reader for .ims file"""
        factory = DataReaderFactory()
        mock_reader = MagicMock()
        mock_imaris_reader.return_value = mock_reader
        
        reader = factory.create("/path/to/file.ims")
        
        mock_imaris_reader.assert_called_once_with("/path/to/file.ims")
        self.assertEqual(reader, mock_reader)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader")
    def test_create_h5_file(self, mock_imaris_reader):
        """Test creating reader for .h5 file"""
        factory = DataReaderFactory()
        mock_reader = MagicMock()
        mock_imaris_reader.return_value = mock_reader
        
        reader = factory.create("/path/to/file.h5")
        
        mock_imaris_reader.assert_called_once_with("/path/to/file.h5")
        self.assertEqual(reader, mock_reader)

    def test_create_unsupported_extension(self):
        """Test creating reader for unsupported extension raises error"""
        factory = DataReaderFactory()
        
        with self.assertRaises(NotImplementedError) as context:
            factory.create("/path/to/file.tif")
        
        self.assertIn("not supported", str(context.exception))


if __name__ == "__main__":
    unittest.main()
