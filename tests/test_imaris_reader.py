"""Tests for ImarisReader"""

import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch, PropertyMock

import dask.array as da
import h5py
import numpy as np

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    ImarisReader,
    MissingDatasetError,
)


class TestImarisReader(unittest.TestCase):
    """Test suite for ImarisReader class"""

    def setUp(self):
        """Set up test fixtures"""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_init(self, mock_h5py_file):
        """Test ImarisReader initialization"""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)

        self.assertEqual(reader.filepath, self.test_filepath)
        mock_h5py_file.assert_called_once_with(self.test_filepath, mode="r")
        self.assertEqual(reader._handle, self.mock_h5_file)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_context_manager(self, mock_h5py_file):
        """Test ImarisReader as context manager"""
        mock_h5py_file.return_value = self.mock_h5_file

        with ImarisReader(self.test_filepath) as reader:
            self.assertIsNotNone(reader._handle)

        self.mock_h5_file.close.assert_called_once()

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_repr(self, mock_h5py_file):
        """Test ImarisReader __repr__"""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        repr_str = repr(reader)

        self.assertIn("ImarisReader", repr_str)
        self.assertIn(self.test_filepath, repr_str)
        self.assertIn("open", repr_str)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_get_shape(self, mock_h5py_file):
        """Test get_shape method"""
        mock_h5py_file.return_value = self.mock_h5_file

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.shape = (100, 200, 300)
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)
        shape = reader.get_shape()

        self.assertEqual(shape, (100, 200, 300))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_get_metadata_shape(self, mock_h5py_file):
        """Test get_metadata_shape method"""
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
        shape = reader.get_metadata_shape()

        self.assertEqual(shape, (100, 200, 300))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
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

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
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

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
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

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.da.from_array"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_as_dask_array(self, mock_h5py_file, mock_from_array):
        """Test as_dask_array method"""
        mock_h5py_file.return_value = self.mock_h5_file

        # Mock dataset
        mock_dataset = MagicMock()
        mock_dataset.shape = (100, 200, 300)
        mock_dataset.chunks = (32, 64, 128)
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        # Mock dask array
        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)
        result = reader.as_dask_array(chunks="native")

        mock_from_array.assert_called_once()
        self.assertEqual(result, mock_dask_array)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_get_chunks(self, mock_h5py_file):
        """Test get_chunks method"""
        mock_h5py_file.return_value = self.mock_h5_file

        mock_dataset = MagicMock()
        mock_dataset.chunks = (32, 64, 128)
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)
        chunks = reader.get_chunks()

        self.assertEqual(chunks, (32, 64, 128))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_get_dtype(self, mock_h5py_file):
        """Test get_dtype method"""
        mock_h5py_file.return_value = self.mock_h5_file

        mock_dataset = MagicMock()
        mock_dataset.dtype = np.dtype(np.uint16)
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)
        dtype = reader.get_dtype()

        self.assertEqual(dtype, np.dtype(np.uint16))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_n_levels_property(self, mock_h5py_file):
        """Test n_levels property"""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)

        # Mock the contains check to return True for levels 0, 1, 2
        def mock_contains(self_or_path, path=None):
            # Handle both mock.method(self, path) and direct call(path)
            actual_path = path if path is not None else self_or_path
            if "ResolutionLevel" not in actual_path:
                return False
            level = int(actual_path.split("ResolutionLevel ")[1].split("/")[0])
            return level < 3

        self.mock_h5_file.__contains__ = mock_contains
        n_levels = reader.n_levels

        self.assertEqual(n_levels, 3)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_close(self, mock_h5py_file):
        """Test close method"""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        self.assertIsNotNone(reader._handle)

        reader.close()
        self.mock_h5_file.close.assert_called_once()
        self.assertIsNone(reader._handle)

        # Calling close again should not raise error
        reader.close()

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_require_handle_raises_when_closed(self, mock_h5py_file):
        """Test _require_handle raises RuntimeError when file is closed"""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        reader.close()

        with self.assertRaises(RuntimeError) as context:
            reader._require_handle()

        self.assertIn("closed", str(context.exception))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.h5py.File"
    )
    def test_get_dask_pyramid_missing_dataset(self, mock_h5py_file):
        """Test get_dask_pyramid raises error for missing dataset"""
        mock_h5py_file.return_value = self.mock_h5_file

        # Mock that no resolution levels exist
        def mock_contains(self_or_path, path=None):
            return False

        self.mock_h5_file.__contains__ = mock_contains

        reader = ImarisReader(self.test_filepath)

        with self.assertRaises(MissingDatasetError):
            reader.get_dask_pyramid(num_levels=2)


if __name__ == "__main__":
    unittest.main()
