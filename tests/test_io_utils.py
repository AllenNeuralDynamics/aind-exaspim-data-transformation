"""Tests for io_utils module - ImarisReader and related utilities."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock

import h5py
import numpy as np

from aind_exaspim_data_transformation.utils.io_utils import (
    ImarisReader,
    MissingDatasetError,
)


class TestImarisReaderBasic(unittest.TestCase):
    """Test suite for ImarisReader basic functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_init(self, mock_h5py_file):
        """Test ImarisReader initialization."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)

        self.assertEqual(reader.filepath, self.test_filepath)
        mock_h5py_file.assert_called_once_with(self.test_filepath, mode="r")

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_context_manager(self, mock_h5py_file):
        """Test ImarisReader as context manager."""
        mock_h5py_file.return_value = self.mock_h5_file

        with ImarisReader(self.test_filepath) as reader:
            self.assertIsNotNone(reader._handle)

        self.mock_h5_file.close.assert_called_once()

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_repr_open(self, mock_h5py_file):
        """Test ImarisReader __repr__ when open."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        repr_str = repr(reader)

        self.assertIn("ImarisReader", repr_str)
        self.assertIn(self.test_filepath, repr_str)
        self.assertIn("open", repr_str)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_repr_closed(self, mock_h5py_file):
        """Test ImarisReader __repr__ when closed."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        reader.close()
        repr_str = repr(reader)

        self.assertIn("closed", repr_str)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_close(self, mock_h5py_file):
        """Test close method."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        reader.close()

        self.mock_h5_file.close.assert_called_once()
        self.assertIsNone(reader._handle)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_close_twice(self, mock_h5py_file):
        """Test closing twice doesn't raise."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        reader.close()
        reader.close()  # Should not raise

        # Only closed once
        self.mock_h5_file.close.assert_called_once()

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_require_handle_raises_when_closed(self, mock_h5py_file):
        """Test _require_handle raises RuntimeError when closed."""
        mock_h5py_file.return_value = self.mock_h5_file

        reader = ImarisReader(self.test_filepath)
        reader.close()

        with self.assertRaises(RuntimeError) as ctx:
            reader._require_handle()

        self.assertIn("closed", str(ctx.exception))


class TestImarisReaderDataAccess(unittest.TestCase):
    """Test suite for ImarisReader data access methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)
        self.mock_dataset = MagicMock()
        self.mock_dataset.shape = (64, 128, 256)
        self.mock_dataset.chunks = (32, 64, 128)
        self.mock_dataset.dtype = np.dtype("uint16")

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_shape(self, mock_h5py_file):
        """Test get_shape method."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        shape = reader.get_shape()

        self.assertEqual(shape, (64, 128, 256))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_shape_custom_path(self, mock_h5py_file):
        """Test get_shape with custom data path."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        custom_path = "/DataSet/ResolutionLevel 1/TimePoint 0/Channel 0/Data"
        shape = reader.get_shape(custom_path)

        self.mock_h5_file.__getitem__.assert_called_with(custom_path)
        self.assertEqual(shape, (64, 128, 256))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_chunks(self, mock_h5py_file):
        """Test get_chunks method."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        chunks = reader.get_chunks()

        self.assertEqual(chunks, (32, 64, 128))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_dtype(self, mock_h5py_file):
        """Test get_dtype method."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        dtype = reader.get_dtype()

        self.assertEqual(dtype, np.dtype("uint16"))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_as_array(self, mock_h5py_file):
        """Test as_array method."""
        mock_h5py_file.return_value = self.mock_h5_file
        test_data = np.random.randint(0, 65535, (64, 128, 256), dtype=np.uint16)
        self.mock_dataset.__getitem__.return_value = test_data
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        result = reader.as_array()

        np.testing.assert_array_equal(result, test_data)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_read_block(self, mock_h5py_file):
        """Test read_block method."""
        mock_h5py_file.return_value = self.mock_h5_file
        test_data = np.random.randint(0, 65535, (32, 64, 64), dtype=np.uint16)
        self.mock_dataset.__getitem__.return_value = test_data
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset

        reader = ImarisReader(self.test_filepath)
        slices = (slice(0, 32), slice(0, 64), slice(0, 64))
        result = reader.read_block(slices)

        np.testing.assert_array_equal(result, test_data)

    @patch("aind_exaspim_data_transformation.utils.io_utils.da.from_array")
    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_as_dask_array_native_chunks(self, mock_h5py_file, mock_from_array):
        """Test as_dask_array with native chunks."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset
        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)
        result = reader.as_dask_array(chunks="native")

        mock_from_array.assert_called_once_with(
            self.mock_dataset, chunks=(32, 64, 128), lock=True
        )
        self.assertEqual(result, mock_dask_array)

    @patch("aind_exaspim_data_transformation.utils.io_utils.da.from_array")
    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_as_dask_array_explicit_chunks(self, mock_h5py_file, mock_from_array):
        """Test as_dask_array with explicit chunks."""
        mock_h5py_file.return_value = self.mock_h5_file
        self.mock_h5_file.__getitem__.return_value = self.mock_dataset
        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)
        explicit_chunks = (16, 32, 64)
        result = reader.as_dask_array(chunks=explicit_chunks)

        mock_from_array.assert_called_once_with(
            self.mock_dataset, chunks=explicit_chunks, lock=True
        )


class TestImarisReaderMetadata(unittest.TestCase):
    """Test suite for ImarisReader metadata methods."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_metadata_shape(self, mock_h5py_file):
        """Test get_metadata_shape method."""
        mock_h5py_file.return_value = self.mock_h5_file

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

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_voxel_size(self, mock_h5py_file):
        """Test get_voxel_size method."""
        mock_h5py_file.return_value = self.mock_h5_file

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

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_origin(self, mock_h5py_file):
        """Test get_origin method."""
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

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_extent(self, mock_h5py_file):
        """Test get_extent method."""
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

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_n_levels(self, mock_h5py_file):
        """Test n_levels property."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Simulate 3 resolution levels
        def contains_check(path):
            if "ResolutionLevel 0" in path:
                return True
            if "ResolutionLevel 1" in path:
                return True
            if "ResolutionLevel 2" in path:
                return True
            return False

        self.mock_h5_file.__contains__.side_effect = contains_check

        reader = ImarisReader(self.test_filepath)
        n_levels = reader.n_levels

        self.assertEqual(n_levels, 3)


class TestImarisReaderIterBlocks(unittest.TestCase):
    """Test suite for ImarisReader iter_blocks method."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_iter_blocks_basic(self, mock_h5py_file):
        """Test iter_blocks yields correct blocks."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Create a test array
        test_data = np.arange(64 * 64 * 64, dtype=np.uint16).reshape(64, 64, 64)
        mock_dataset = MagicMock()
        mock_dataset.shape = (64, 64, 64)

        def getitem_side_effect(slices):
            return test_data[slices]

        mock_dataset.__getitem__.side_effect = getitem_side_effect
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)
        block_shape = (32, 32, 32)
        blocks = list(reader.iter_blocks(block_shape))

        # Should have 2x2x2 = 8 blocks
        self.assertEqual(len(blocks), 8)

        # Check first block
        slices, data = blocks[0]
        self.assertEqual(slices, (slice(0, 32), slice(0, 32), slice(0, 32)))
        self.assertEqual(data.shape, (32, 32, 32))

        # Check last block
        slices, data = blocks[-1]
        self.assertEqual(slices, (slice(32, 64), slice(32, 64), slice(32, 64)))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_iter_blocks_edge_case(self, mock_h5py_file):
        """Test iter_blocks handles non-divisible shapes."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Non-divisible shape
        test_data = np.arange(50 * 50 * 50, dtype=np.uint16).reshape(50, 50, 50)
        mock_dataset = MagicMock()
        mock_dataset.shape = (50, 50, 50)

        def getitem_side_effect(slices):
            return test_data[slices]

        mock_dataset.__getitem__.side_effect = getitem_side_effect
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)
        block_shape = (32, 32, 32)
        blocks = list(reader.iter_blocks(block_shape))

        # ceil(50/32) = 2 in each dim, so 2x2x2 = 8 blocks
        self.assertEqual(len(blocks), 8)

        # Check edge block has truncated data
        slices, data = blocks[-1]
        self.assertEqual(slices, (slice(32, 50), slice(32, 50), slice(32, 50)))
        self.assertEqual(data.shape, (18, 18, 18))  # 50 - 32 = 18


class TestImarisReaderIterSuperchunks(unittest.TestCase):
    """Test suite for ImarisReader iter_superchunks method."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_iter_superchunks_basic(self, mock_h5py_file):
        """Test iter_superchunks yields correct blocks."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Create test array: 128x128x128
        test_data = np.arange(128 * 128 * 128, dtype=np.uint16).reshape(
            128, 128, 128
        )
        mock_dataset = MagicMock()
        mock_dataset.shape = (128, 128, 128)

        def getitem_side_effect(slices):
            return test_data[slices]

        mock_dataset.__getitem__.side_effect = getitem_side_effect
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)

        # Superchunk = 128x128x128 (whole array)
        # Yield = 64x64x64
        superchunk_shape = (128, 128, 128)
        yield_shape = (64, 64, 64)

        blocks = list(
            reader.iter_superchunks(superchunk_shape, yield_shape)
        )

        # 128/64 = 2 in each dim, so 2x2x2 = 8 blocks
        self.assertEqual(len(blocks), 8)

        # All blocks should be 64x64x64
        for slices, data in blocks:
            self.assertEqual(data.shape, (64, 64, 64))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_iter_superchunks_validates_divisibility(self, mock_h5py_file):
        """Test iter_superchunks raises error for non-divisible shapes."""
        mock_h5py_file.return_value = self.mock_h5_file

        mock_dataset = MagicMock()
        mock_dataset.shape = (128, 128, 128)
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)

        # yield_shape doesn't divide superchunk_shape
        superchunk_shape = (128, 128, 128)
        yield_shape = (30, 30, 30)  # 128 % 30 != 0

        with self.assertRaises(ValueError) as ctx:
            list(reader.iter_superchunks(superchunk_shape, yield_shape))

        self.assertIn("evenly divide", str(ctx.exception))

    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_iter_superchunks_multiple_superchunks(self, mock_h5py_file):
        """Test iter_superchunks with multiple superchunks."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Create test array: 256x256x256
        test_data = np.zeros((256, 256, 256), dtype=np.uint16)
        mock_dataset = MagicMock()
        mock_dataset.shape = (256, 256, 256)

        def getitem_side_effect(slices):
            return test_data[slices]

        mock_dataset.__getitem__.side_effect = getitem_side_effect
        self.mock_h5_file.__getitem__.return_value = mock_dataset

        reader = ImarisReader(self.test_filepath)

        # Superchunk = 128x128x128, Yield = 64x64x64
        superchunk_shape = (128, 128, 128)
        yield_shape = (64, 64, 64)

        blocks = list(
            reader.iter_superchunks(superchunk_shape, yield_shape)
        )

        # 2x2x2 superchunks, each with 2x2x2 yield blocks = 64 total
        self.assertEqual(len(blocks), 64)


class TestImarisReaderGetDaskPyramid(unittest.TestCase):
    """Test suite for ImarisReader get_dask_pyramid method."""

    def setUp(self):
        """Set up test fixtures."""
        self.test_filepath = "/fake/path/test.ims"
        self.mock_h5_file = MagicMock(spec=h5py.File)

    @patch("aind_exaspim_data_transformation.utils.io_utils.da.from_array")
    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_dask_pyramid(self, mock_h5py_file, mock_from_array):
        """Test get_dask_pyramid returns multiple levels."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Mock datasets for 3 levels
        mock_datasets = []
        shapes = [(128, 256, 512), (64, 128, 256), (32, 64, 128)]
        for shape in shapes:
            mock_ds = MagicMock()
            mock_ds.shape = shape
            mock_ds.chunks = (32, 64, 128)
            mock_datasets.append(mock_ds)

        def getitem_side_effect(path):
            for i, shape in enumerate(shapes):
                if f"ResolutionLevel {i}" in path:
                    return mock_datasets[i]
            raise KeyError(path)

        self.mock_h5_file.__getitem__.side_effect = getitem_side_effect

        # Make all levels "exist"
        def contains_check(path):
            return any(f"ResolutionLevel {i}" in path for i in range(3))

        self.mock_h5_file.__contains__.side_effect = contains_check

        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)
        pyramid = reader.get_dask_pyramid(num_levels=3)

        self.assertEqual(len(pyramid), 3)
        self.assertEqual(mock_from_array.call_count, 3)

    @patch("aind_exaspim_data_transformation.utils.io_utils.da.from_array")
    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_dask_pyramid_missing_level(self, mock_h5py_file, mock_from_array):
        """Test get_dask_pyramid raises error for missing level."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Mock dataset for level 0 only
        mock_ds0 = MagicMock()
        mock_ds0.shape = (128, 256, 512)
        mock_ds0.chunks = (32, 64, 128)

        def getitem_side_effect(path):
            if "ResolutionLevel 0" in path:
                return mock_ds0
            raise KeyError(path)

        self.mock_h5_file.__getitem__.side_effect = getitem_side_effect

        # Only level 0 exists - check __contains__ for path lookup
        def contains_check(path):
            return "ResolutionLevel 0" in path

        self.mock_h5_file.__contains__.side_effect = contains_check

        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)

        with self.assertRaises(MissingDatasetError):
            reader.get_dask_pyramid(num_levels=3)

    @patch("aind_exaspim_data_transformation.utils.io_utils.da.from_array")
    @patch("aind_exaspim_data_transformation.utils.io_utils.h5py.File")
    def test_get_dask_pyramid_explicit_chunks(self, mock_h5py_file, mock_from_array):
        """Test get_dask_pyramid with explicit chunks clamped to level shape."""
        mock_h5py_file.return_value = self.mock_h5_file

        # Level 0: large, Level 1: smaller than requested chunks
        mock_ds0 = MagicMock()
        mock_ds0.shape = (128, 256, 512)
        mock_ds0.chunks = (32, 64, 128)

        mock_ds1 = MagicMock()
        mock_ds1.shape = (32, 64, 128)  # Smaller than requested chunks
        mock_ds1.chunks = (16, 32, 64)

        def getitem_side_effect(path):
            if "ResolutionLevel 0" in path:
                return mock_ds0
            elif "ResolutionLevel 1" in path:
                return mock_ds1
            raise KeyError(path)

        self.mock_h5_file.__getitem__.side_effect = getitem_side_effect
        self.mock_h5_file.__contains__.side_effect = lambda p: (
            "Level 0" in p or "Level 1" in p
        )

        mock_dask_array = MagicMock()
        mock_from_array.return_value = mock_dask_array

        reader = ImarisReader(self.test_filepath)
        pyramid = reader.get_dask_pyramid(num_levels=2, chunks=(64, 128, 256))

        self.assertEqual(len(pyramid), 2)

        # Check that chunks are clamped for level 1
        calls = mock_from_array.call_args_list
        # Level 0: (64, 128, 256) - fits
        # Level 1: min(64,32), min(128,64), min(256,128) = (32, 64, 128)
        self.assertEqual(calls[1][1]["chunks"], (32, 64, 128))


class TestMissingDatasetError(unittest.TestCase):
    """Test MissingDatasetError exception."""

    def test_missing_dataset_error_is_file_not_found(self):
        """Test MissingDatasetError inherits from FileNotFoundError."""
        self.assertTrue(issubclass(MissingDatasetError, FileNotFoundError))

    def test_missing_dataset_error_message(self):
        """Test MissingDatasetError stores message."""
        error = MissingDatasetError("Dataset /path/to/data not found")
        self.assertIn("not found", str(error))


if __name__ == "__main__":
    unittest.main()
