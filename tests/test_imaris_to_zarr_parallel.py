"""Tests for the TensorStore-based parallel writer functions."""

import unittest
from unittest.mock import MagicMock, Mock, patch, PropertyMock
import math

import numpy as np


class TestCreateTensorstoreSpec(unittest.TestCase):
    """Test suite for create_tensorstore_spec function."""

    def test_local_spec_without_sharding(self):
        """Test creating a local file spec without sharding."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_tensorstore_spec,
        )

        spec = create_tensorstore_spec(
            path="/tmp/test.zarr",
            shape=(100, 200, 300),
            dtype=np.dtype("uint16"),
            chunk_shape=(64, 64, 64),
            shard_shape=None,
            codec="zstd",
            codec_level=3,
            is_s3=False,
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertEqual(spec["kvstore"]["driver"], "file")
        self.assertEqual(spec["kvstore"]["path"], "/tmp/test.zarr")
        self.assertEqual(spec["metadata"]["shape"], [100, 200, 300])
        self.assertEqual(
            spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"],
            [64, 64, 64],
        )
        self.assertTrue(spec["create"])
        self.assertTrue(spec["delete_existing"])

    def test_local_spec_with_sharding(self):
        """Test creating a local file spec with sharding."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_tensorstore_spec,
        )

        spec = create_tensorstore_spec(
            path="/tmp/test.zarr",
            shape=(100, 200, 300),
            dtype=np.dtype("uint16"),
            chunk_shape=(64, 64, 64),
            shard_shape=(128, 128, 128),
            codec="zstd",
            codec_level=3,
            is_s3=False,
        )

        # With sharding, the outer chunk_shape should be the shard shape
        self.assertEqual(
            spec["metadata"]["chunk_grid"]["configuration"]["chunk_shape"],
            [128, 128, 128],
        )

        # Check for sharding codec
        codecs = spec["metadata"]["codecs"]
        self.assertEqual(len(codecs), 1)
        self.assertEqual(codecs[0]["name"], "sharding_indexed")
        self.assertEqual(
            codecs[0]["configuration"]["chunk_shape"], [64, 64, 64]
        )

    def test_s3_spec(self):
        """Test creating an S3 spec."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_tensorstore_spec,
        )

        spec = create_tensorstore_spec(
            path="s3://my-bucket/path/to/test.zarr",
            shape=(100, 200, 300),
            dtype=np.dtype("uint16"),
            chunk_shape=(64, 64, 64),
            shard_shape=None,
            codec="zstd",
            codec_level=3,
            is_s3=True,
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertEqual(spec["kvstore"]["driver"], "s3")
        self.assertEqual(spec["kvstore"]["bucket"], "my-bucket")
        self.assertEqual(spec["kvstore"]["path"], "path/to/test.zarr")


class TestS3Kvstore(unittest.TestCase):
    """Test suite for _s3_kvstore function."""

    def test_valid_s3_url(self):
        """Test parsing a valid S3 URL."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _s3_kvstore,
        )

        result = _s3_kvstore("s3://my-bucket/some/path")
        self.assertEqual(result["driver"], "s3")
        self.assertEqual(result["bucket"], "my-bucket")
        self.assertEqual(result["path"], "some/path")

    def test_s3_url_no_path(self):
        """Test parsing an S3 URL with no path."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _s3_kvstore,
        )

        result = _s3_kvstore("s3://my-bucket")
        self.assertEqual(result["bucket"], "my-bucket")
        self.assertEqual(result["path"], "")

    def test_invalid_url(self):
        """Test that invalid URLs raise ValueError."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _s3_kvstore,
        )

        with self.assertRaises(ValueError):
            _s3_kvstore("/local/path")


class TestIterBlockAlignedSlices(unittest.TestCase):
    """Test suite for iter_block_aligned_slices function."""

    def test_exact_fit(self):
        """Test when shape is exact multiple of block shape."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            iter_block_aligned_slices,
        )

        shape = (100, 100, 100)
        block_shape = (50, 50, 50)

        slices = list(iter_block_aligned_slices(shape, block_shape))

        # Should have 2x2x2 = 8 blocks
        self.assertEqual(len(slices), 8)

        # Check first and last slices
        self.assertEqual(slices[0], (slice(0, 50), slice(0, 50), slice(0, 50)))
        self.assertEqual(
            slices[-1], (slice(50, 100), slice(50, 100), slice(50, 100))
        )

    def test_partial_blocks(self):
        """Test when shape doesn't evenly divide by block shape."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            iter_block_aligned_slices,
        )

        shape = (75, 75, 75)
        block_shape = (50, 50, 50)

        slices = list(iter_block_aligned_slices(shape, block_shape))

        # Should have 2x2x2 = 8 blocks (with partial blocks at edges)
        self.assertEqual(len(slices), 8)

        # Check that last block is clamped to shape
        last_slice = slices[-1]
        self.assertEqual(last_slice[0], slice(50, 75))  # Not slice(50, 100)
        self.assertEqual(last_slice[1], slice(50, 75))
        self.assertEqual(last_slice[2], slice(50, 75))

    def test_single_block(self):
        """Test when entire array fits in one block."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            iter_block_aligned_slices,
        )

        shape = (50, 50, 50)
        block_shape = (100, 100, 100)

        slices = list(iter_block_aligned_slices(shape, block_shape))

        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0], (slice(0, 50), slice(0, 50), slice(0, 50)))

    def test_covers_entire_array(self):
        """Test that all slices together cover the entire array."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            iter_block_aligned_slices,
        )

        shape = (123, 234, 345)
        block_shape = (64, 64, 64)

        # Create a coverage array
        coverage = np.zeros(shape, dtype=bool)

        for slices in iter_block_aligned_slices(shape, block_shape):
            coverage[slices] = True

        # All elements should be covered
        self.assertTrue(np.all(coverage))


class TestWriteBlockToTensorstore(unittest.TestCase):
    """Test suite for write_block_to_tensorstore function."""

    def test_write_block(self):
        """Test writing a block to TensorStore."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            write_block_to_tensorstore,
        )

        # Create mock store
        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_store.__getitem__.return_value.write.return_value = mock_future

        data = np.random.randint(0, 1000, size=(64, 64, 64), dtype=np.uint16)
        slices = (slice(0, 64), slice(0, 64), slice(0, 64))

        result = write_block_to_tensorstore(mock_store, data, slices)

        mock_store.__getitem__.assert_called_once_with(slices)
        mock_store.__getitem__.return_value.write.assert_called_once()
        self.assertEqual(result, mock_future)


class TestImarisToZarrParallel(unittest.TestCase):
    """Test suite for imaris_to_zarr_parallel function."""

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_basic_parallel_conversion(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_ts,
    ):
        """Test basic parallel Imaris to Zarr conversion."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (64, 64, 64)  # Small for testing
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 32, 32)
        mock_reader.n_levels = 1

        # Create mock dask array
        mock_dask_array = MagicMock()
        mock_dask_array.__getitem__ = MagicMock(return_value=MagicMock())
        mock_dask_array.__getitem__.return_value.compute.return_value = (
            np.zeros((64, 64, 64), dtype=np.uint16)
        )
        mock_reader.as_dask_array.return_value = mock_dask_array

        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_path = MagicMock()
        mock_path.stem = "test"
        mock_path.__truediv__ = lambda self, other: MagicMock(
            mkdir=MagicMock(),
            __str__=lambda s: "/fake/output/test.ome.zarr",
        )
        mock_path_cls.return_value = mock_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_store.__getitem__.return_value.write.return_value = mock_future
        mock_ts.open.return_value.result.return_value = mock_store

        # Mock metadata
        mock_write_metadata.return_value = {"multiscales": []}

        # Call function
        result = imaris_to_zarr_parallel(
            imaris_path="/fake/input/test.ims",
            output_path="/fake/output",
            chunk_shape=(32, 32, 32),
            shard_shape=(64, 64, 64),
            n_lvls=1,
        )

        # Assertions
        mock_imaris_reader_cls.assert_called_once_with("/fake/input/test.ims")
        mock_ts.open.assert_called()
        mock_write_metadata.assert_called_once()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_s3_output(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_ts,
    ):
        """Test parallel conversion with S3 output."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )

        # Setup mocks
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (64, 64, 64)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 32, 32)
        mock_reader.n_levels = 1

        mock_dask_array = MagicMock()
        mock_dask_array.__getitem__ = MagicMock(return_value=MagicMock())
        mock_dask_array.__getitem__.return_value.compute.return_value = (
            np.zeros((64, 64, 64), dtype=np.uint16)
        )
        mock_reader.as_dask_array.return_value = mock_dask_array

        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        mock_path = MagicMock()
        mock_path.stem = "test"
        mock_path.__truediv__ = lambda self, other: MagicMock(
            __str__=lambda s: "fake/output/test.ome.zarr"
        )
        mock_path_cls.return_value = mock_path

        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_store.__getitem__.return_value.write.return_value = mock_future
        mock_ts.open.return_value.result.return_value = mock_store

        mock_write_metadata.return_value = {"multiscales": []}

        # Call function with S3 bucket
        result = imaris_to_zarr_parallel(
            imaris_path="/fake/input/test.ims",
            output_path="output",
            bucket_name="test-bucket",
            n_lvls=1,
        )

        # Verify S3 path was constructed
        self.assertIn("s3://", result)
        self.assertIn("test-bucket", result)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_voxel_size_extraction(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_metadata,
        mock_ts,
    ):
        """Test that voxel size is extracted from file when not provided."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )

        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([2.5, 0.75, 0.75], b"um")
        mock_reader.get_shape.return_value = (64, 64, 64)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 32, 32)
        mock_reader.n_levels = 1

        mock_dask_array = MagicMock()
        mock_dask_array.__getitem__ = MagicMock(return_value=MagicMock())
        mock_dask_array.__getitem__.return_value.compute.return_value = (
            np.zeros((64, 64, 64), dtype=np.uint16)
        )
        mock_reader.as_dask_array.return_value = mock_dask_array

        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        mock_path = MagicMock()
        mock_path.stem = "test"
        mock_path.__truediv__ = lambda self, other: MagicMock(
            mkdir=MagicMock(),
            __str__=lambda s: "/fake/output/test.ome.zarr",
        )
        mock_path_cls.return_value = mock_path

        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_store.__getitem__.return_value.write.return_value = mock_future
        mock_ts.open.return_value.result.return_value = mock_store

        mock_write_metadata.return_value = {"multiscales": []}

        # Call function without voxel_size
        imaris_to_zarr_parallel(
            imaris_path="/fake/input/test.ims",
            output_path="/fake/output",
            n_lvls=1,
        )

        # Verify voxel size was extracted
        mock_reader.get_voxel_size.assert_called_once()


if __name__ == "__main__":
    unittest.main()
