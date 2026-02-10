"""Tests for the TensorStore-based parallel writer functions."""

import unittest
from unittest.mock import MagicMock, Mock, patch

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
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "_write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_basic_parallel_conversion(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_zarr_metadata,
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

        self.assertIsNotNone(result)
        mock_write_zarr_metadata.assert_called_once()

        # Assertions
        mock_imaris_reader_cls.assert_called_once_with("/fake/input/test.ims")
        mock_ts.open.assert_called()
        mock_write_metadata.assert_called_once()

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "_write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_s3_output(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_zarr_metadata,
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

        mock_write_zarr_metadata.assert_called_once()

        # Verify S3 path was constructed
        self.assertIn("s3://", result)
        self.assertIn("test-bucket", result)

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "_write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_voxel_size_extraction(
        self,
        mock_path_cls,
        mock_imaris_reader_cls,
        mock_write_zarr_metadata,
        mock_write_metadata,
        mock_ts,
    ):
        """Test voxel size extraction when not provided."""
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
        mock_write_zarr_metadata.assert_called_once()

        # Verify voxel size was extracted
        mock_reader.get_voxel_size.assert_called_once()


class TestShardUtilities(unittest.TestCase):
    """Test suite for shard computation utilities."""

    def test_compute_shard_grid_exact_division(self):
        """Test compute_shard_grid with exact division."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_shard_grid,
        )

        data_shape = (256, 512, 1024)
        shard_shape = (128, 256, 512)

        grid = compute_shard_grid(data_shape, shard_shape)

        self.assertEqual(grid, (2, 2, 2))

    def test_compute_shard_grid_with_remainder(self):
        """Test compute_shard_grid with non-exact division."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_shard_grid,
        )

        data_shape = (300, 500, 700)
        shard_shape = (128, 256, 512)

        grid = compute_shard_grid(data_shape, shard_shape)

        # ceil(300/128)=3, ceil(500/256)=2, ceil(700/512)=2
        self.assertEqual(grid, (3, 2, 2))

    def test_shard_index_to_slices_basic(self):
        """Test shard_index_to_slices computes correct slices."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            shard_index_to_slices,
        )

        shard_index = (1, 2, 0)
        shard_shape = (128, 256, 512)
        data_shape = (512, 1024, 2048)

        slices = shard_index_to_slices(shard_index, shard_shape, data_shape)

        self.assertEqual(slices[0], slice(128, 256))  # Z: 1*128 to 2*128
        self.assertEqual(slices[1], slice(512, 768))  # Y: 2*256 to 3*256
        self.assertEqual(slices[2], slice(0, 512))  # X: 0*512 to 1*512

    def test_shard_index_to_slices_edge_case(self):
        """Test shard_index_to_slices handles edge shards correctly."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            shard_index_to_slices,
        )

        # Last shard in a dimension that doesn't divide evenly
        shard_index = (2, 0, 0)
        shard_shape = (128, 256, 512)
        data_shape = (300, 256, 512)  # Z only goes to 300, not 384

        slices = shard_index_to_slices(shard_index, shard_shape, data_shape)

        # Edge shard should be clamped to data_shape
        self.assertEqual(slices[0], slice(256, 300))  # Clamped to 300

    def test_enumerate_shard_indices(self):
        """Test enumerate_shard_indices yields correct indices."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            enumerate_shard_indices,
        )

        data_shape = (256, 768, 512)
        shard_shape = (128, 256, 256)

        indices = enumerate_shard_indices(data_shape, shard_shape)

        # 2 x 3 x 2 = 12 shards
        self.assertEqual(len(indices), 12)

        # Check first and last
        self.assertEqual(indices[0], (0, 0, 0))
        self.assertEqual(indices[-1], (1, 2, 1))

    def test_create_shard_tasks(self):
        """Test create_shard_tasks generates correct task list."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_shard_tasks,
        )

        tasks = create_shard_tasks(
            imaris_path="/path/to/input.ims",
            output_spec={"driver": "zarr3"},
            data_shape=(256, 512, 512),
            shard_shape=(128, 256, 256),
            data_path="/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        )

        # 2 x 2 x 2 = 8 shards
        self.assertEqual(len(tasks), 8)

        # Check first task structure
        task = tasks[0]
        self.assertEqual(task["imaris_path"], "/path/to/input.ims")
        self.assertEqual(task["shard_index"], (0, 0, 0))
        self.assertIn("output_spec", task)


class TestProcessSingleShard(unittest.TestCase):
    """Test suite for process_single_shard function."""

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    def test_process_single_shard(self, mock_imaris_reader_cls, mock_ts):
        """Test process_single_shard reads and writes correctly."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            process_single_shard,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        test_data = np.zeros((128, 256, 256), dtype=np.uint16)
        mock_reader.read_block.return_value = test_data

        mock_imaris_reader_cls.return_value.__enter__ = Mock(
            return_value=mock_reader
        )
        mock_imaris_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock TensorStore
        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_store.__getitem__.return_value.write.return_value = mock_future
        mock_ts.open.return_value.result.return_value = mock_store

        result = process_single_shard(
            imaris_path="/path/to/input.ims",
            output_spec={"driver": "zarr3"},
            shard_index=(0, 0, 0),
            shard_shape=(128, 256, 256),
            data_shape=(256, 512, 512),
            data_path="/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        )

        # Check result structure
        self.assertEqual(result["shard_index"], (0, 0, 0))
        self.assertIn("bytes_written", result)
        self.assertIn("elapsed_seconds", result)
        self.assertGreater(result["bytes_written"], 0)

        # Verify read was called with correct slices
        mock_reader.read_block.assert_called_once()


class TestWriteZarrMetadata(unittest.TestCase):
    """Test suite for _write_zarr_metadata function."""

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    def test_write_zarr_metadata_s3(self, mock_ts):
        """Test writing metadata to S3."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _write_zarr_metadata,
        )

        mock_store = MagicMock()
        mock_future = MagicMock()
        mock_future.result.return_value = None
        mock_store.write.return_value = mock_future
        mock_ts.open.return_value.result.return_value = mock_store

        metadata = {"multiscales": [{"name": "test"}]}

        _write_zarr_metadata(
            store_path="s3://bucket/path/zarr",
            metadata_dict=metadata,
            is_s3=True,
        )

        mock_ts.open.assert_called_once()
        mock_store.write.assert_called_once()

    @patch("builtins.open", new_callable=MagicMock)
    @patch("json.dump")
    def test_write_zarr_metadata_local(self, mock_json_dump, mock_open):
        """Test writing metadata to local file."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _write_zarr_metadata,
        )

        metadata = {"multiscales": [{"name": "test"}]}

        _write_zarr_metadata(
            store_path="/local/path/zarr",
            metadata_dict=metadata,
            is_s3=False,
        )

        mock_open.assert_called_once()
        mock_json_dump.assert_called_once()


class TestComputeDownsampledShape(unittest.TestCase):
    """Test suite for compute_downsampled_shape function."""

    def test_compute_downsampled_shape_basic(self):
        """Test compute_downsampled_shape with basic input."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        shape = (100, 200, 300)
        factor = (2, 2, 2)

        result = compute_downsampled_shape(shape, factor)

        self.assertEqual(result, (50, 100, 150))

    def test_compute_downsampled_shape_uneven(self):
        """Test compute_downsampled_shape with uneven division."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        shape = (101, 201, 301)
        factor = (2, 2, 2)

        result = compute_downsampled_shape(shape, factor)

        # ceil(101/2)=51, ceil(201/2)=101, ceil(301/2)=151
        self.assertEqual(result, (51, 101, 151))

    def test_compute_downsampled_shape_different_factors(self):
        """Test compute_downsampled_shape with different factors."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        shape = (100, 200, 300)
        factor = (1, 2, 4)

        result = compute_downsampled_shape(shape, factor)

        self.assertEqual(result, (100, 100, 75))


class TestCreateScaleSpec(unittest.TestCase):
    """Test suite for create_scale_spec function."""

    def test_create_scale_spec_local(self):
        """Test create_scale_spec for local storage."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="/path/to/output.zarr",
            data_shape=(1, 1, 64, 128, 256),
            data_dtype="uint16",
            shard_shape=(1, 1, 32, 64, 128),
            chunk_shape=(1, 1, 16, 32, 64),
            scale=0,
            codec="zstd",
            codec_level=3,
            bucket_name=None,
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertIn("kvstore", spec)
        self.assertTrue(spec["create"])

    def test_create_scale_spec_s3(self):
        """Test create_scale_spec for S3 storage."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="path/to/output.zarr",
            data_shape=(1, 1, 64, 128, 256),
            data_dtype="uint16",
            shard_shape=(1, 1, 32, 64, 128),
            chunk_shape=(1, 1, 16, 32, 64),
            scale=0,
            codec="zstd",
            codec_level=3,
            bucket_name="my-bucket",
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertIn("kvstore", spec)
        self.assertEqual(spec["kvstore"]["driver"], "s3")


class TestImarisToZarrDistributed(unittest.TestCase):
    """Test suite for imaris_to_zarr_distributed function."""

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_distributed_sequential_local(
        self,
        mock_path_cls,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed with sequential processing (no Dask)."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (64, 128, 256)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_output_path = MagicMock()
        mock_output_path.__truediv__ = Mock(return_value=mock_output_path)
        mock_output_path.__str__ = Mock(return_value="/output/test.ome.zarr")
        mock_path_cls.return_value = mock_output_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock process_single_shard to return success
        mock_process_shard.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        result = imaris_to_zarr_distributed(
            imaris_path="/input/test.ims",
            output_path="/output",
            voxel_size=[1.0, 0.5, 0.5],
            chunk_shape=(32, 64, 128),
            shard_shape=(64, 128, 256),
            n_lvls=1,  # No downsampling
            channel_name="ch0",
            stack_name="test.ome.zarr",
            bucket_name=None,
            dask_client=None,  # Sequential
        )

        # Verify ImarisReader was used
        mock_reader_cls.assert_called_once_with("/input/test.ims")

        # Verify TensorStore was opened
        mock_ts_open.assert_called()

        # Verify process_single_shard was called for each shard
        # With shape (64, 128, 256) and shard (64, 128, 256), there's 1 shard
        self.assertEqual(mock_process_shard.call_count, 1)

        # Verify metadata was written
        mock_write_ome.assert_called_once()
        mock_write_metadata.assert_called_once()

        # Should return the store path
        self.assertIsNotNone(result)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_distributed_with_downsampling(
        self,
        mock_path_cls,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed with downsampling levels."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (128, 256, 512)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_output_path = MagicMock()
        mock_output_path.__truediv__ = Mock(return_value=mock_output_path)
        mock_output_path.__str__ = Mock(return_value="/output/test.ome.zarr")
        mock_path_cls.return_value = mock_output_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock process_single_shard
        mock_process_shard.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        result = imaris_to_zarr_distributed(
            imaris_path="/input/test.ims",
            output_path="/output",
            chunk_shape=(32, 64, 128),
            shard_shape=(64, 128, 256),
            n_lvls=3,  # Request 3 levels
            scale_factor=(2, 2, 2),
            downsample_mode="mean",
            channel_name="ch0",
            stack_name="test.ome.zarr",
            bucket_name=None,
            dask_client=None,
        )

        # Verify downsampling was called (since n_lvls > 1)
        mock_downsample.assert_called_once()
        call_kwargs = mock_downsample.call_args[1]
        self.assertEqual(call_kwargs["n_levels"], 3)
        self.assertEqual(call_kwargs["downsample_factor"], (2, 2, 2))
        self.assertEqual(call_kwargs["downsample_mode"], "mean")

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    def test_distributed_with_s3(
        self,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed with S3 output."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (64, 128, 256)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock process_single_shard
        mock_process_shard.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        result = imaris_to_zarr_distributed(
            imaris_path="/input/test.ims",
            output_path="/prefix",
            chunk_shape=(32, 64, 128),
            shard_shape=(64, 128, 256),
            n_lvls=1,
            channel_name="ch0",
            stack_name="test.ome.zarr",
            bucket_name="my-bucket",  # S3 output
            dask_client=None,
        )

        # Verify S3 path is returned
        self.assertIn("s3://", result)
        self.assertIn("my-bucket", result)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_distributed_with_dask_client(
        self,
        mock_path_cls,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed with Dask client for parallel processing."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (128, 256, 512)  # Multiple shards
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_output_path = MagicMock()
        mock_output_path.__truediv__ = Mock(return_value=mock_output_path)
        mock_output_path.__str__ = Mock(return_value="/output/test.ome.zarr")
        mock_path_cls.return_value = mock_output_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        # Mock Dask client
        mock_client = MagicMock()

        # Mock future results
        mock_future = MagicMock()
        mock_future.result.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }
        mock_client.submit.return_value = mock_future

        # Mock as_completed from dask.distributed
        mock_as_completed = MagicMock()
        mock_as_completed.return_value = iter([mock_future] * 8)

        with patch.dict(
            "sys.modules",
            {
                "dask.distributed": MagicMock(as_completed=mock_as_completed),
            },
        ):
            result = imaris_to_zarr_distributed(
                imaris_path="/input/test.ims",
                output_path="/output",
                chunk_shape=(32, 64, 128),
                shard_shape=(64, 128, 256),
                n_lvls=1,
                channel_name="ch0",
                stack_name="test.ome.zarr",
                bucket_name=None,
                dask_client=mock_client,  # Use Dask
            )

        # Verify client.submit was called for each shard
        # Shape (128, 256, 512) with shards (64, 128, 256) = 2x2x2 = 8 shards
        self.assertEqual(mock_client.submit.call_count, 8)

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_distributed_extracts_voxel_size(
        self,
        mock_path_cls,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed extracts voxel size when not provided."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([2.0, 1.0, 1.0], b"um")
        mock_reader.get_shape.return_value = (64, 128, 256)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_output_path = MagicMock()
        mock_output_path.__truediv__ = Mock(return_value=mock_output_path)
        mock_output_path.__str__ = Mock(return_value="/output/test.ome.zarr")
        mock_path_cls.return_value = mock_output_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock process_single_shard
        mock_process_shard.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        result = imaris_to_zarr_distributed(
            imaris_path="/input/test.ims",
            output_path="/output",
            voxel_size=None,  # Don't provide voxel size
            chunk_shape=(32, 64, 128),
            shard_shape=(64, 128, 256),
            n_lvls=1,
            dask_client=None,
        )

        # Verify voxel size was extracted from file
        mock_reader.get_voxel_size.assert_called_once()

        # Verify the extracted voxel size was passed to metadata writer
        mock_write_ome.assert_called_once()
        call_kwargs = mock_write_ome.call_args[1]
        self.assertEqual(call_kwargs["voxel_size"], (2.0, 1.0, 1.0))

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.write_ome_ngff_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr._write_zarr_metadata"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.create_downsample_levels"
    )
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.process_single_shard"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.Path")
    def test_distributed_uses_defaults(
        self,
        mock_path_cls,
        mock_reader_cls,
        mock_ts_open,
        mock_process_shard,
        mock_downsample,
        mock_write_metadata,
        mock_write_ome,
    ):
        """Test imaris_to_zarr_distributed uses default chunk/shard shapes."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_distributed,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        mock_reader.get_voxel_size.return_value = ([1.0, 0.5, 0.5], b"um")
        mock_reader.get_shape.return_value = (64, 128, 256)
        mock_reader.get_dtype.return_value = np.dtype("uint16")
        mock_reader.get_chunks.return_value = (32, 64, 128)
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock Path
        mock_output_path = MagicMock()
        mock_output_path.__truediv__ = Mock(return_value=mock_output_path)
        mock_output_path.__str__ = Mock(return_value="/output/input.ome.zarr")
        mock_path_cls.return_value = mock_output_path

        # Mock TensorStore
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        # Mock process_single_shard
        mock_process_shard.return_value = {
            "shard_index": (0, 0, 0),
            "bytes_written": 1024 * 1024,
            "elapsed_seconds": 0.5,
        }

        # Mock metadata writer
        mock_write_ome.return_value = {"multiscales": []}

        result = imaris_to_zarr_distributed(
            imaris_path="/input/test.ims",
            output_path="/output",
            # Don't provide chunk_shape, shard_shape, or stack_name
            n_lvls=1,
            dask_client=None,
        )

        # Verify defaults were used - check the TensorStore spec
        mock_ts_open.assert_called()


class TestProcessSingleShardFunction(unittest.TestCase):
    """Test suite for process_single_shard function - additional tests."""

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts.open")
    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr.ImarisReader"
    )
    def test_process_single_shard_basic(self, mock_reader_cls, mock_ts_open):
        """Test process_single_shard reads and writes a shard."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            process_single_shard,
        )

        # Mock ImarisReader
        mock_reader = MagicMock()
        test_data = np.zeros((32, 64, 128), dtype=np.uint16)
        mock_reader.read_block.return_value = test_data
        mock_reader_cls.return_value.__enter__ = Mock(return_value=mock_reader)
        mock_reader_cls.return_value.__exit__ = Mock(return_value=False)

        # Mock TensorStore - use MagicMock which auto-chains
        # TensorStore uses: store[slices].write(data).result()
        mock_store = MagicMock()
        mock_ts_open.return_value.result.return_value = mock_store

        output_spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "/output/test.zarr/0"},
        }

        result = process_single_shard(
            imaris_path="/input/test.ims",
            output_spec=output_spec,
            shard_index=(0, 0, 0),
            shard_shape=(32, 64, 128),
            data_shape=(64, 128, 256),
            data_path="/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        )

        # Verify result structure
        self.assertIn("shard_index", result)
        self.assertIn("bytes_written", result)
        self.assertIn("elapsed_seconds", result)
        self.assertEqual(result["shard_index"], (0, 0, 0))

        # Verify read_block was called with correct slices
        mock_reader.read_block.assert_called_once()

        # Verify TensorStore was subscripted and write was called
        # TensorStore uses: store[slices].write(data).result()
        mock_store.__getitem__.assert_called()
        mock_store.__getitem__().write.assert_called()


class TestCreateShardTasksFunction(unittest.TestCase):
    """Test suite for create_shard_tasks function - additional tests."""

    def test_create_shard_tasks_single_shard(self):
        """Test create_shard_tasks with a single shard."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_shard_tasks,
        )

        output_spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "/output/test.zarr/0"},
        }

        tasks = create_shard_tasks(
            imaris_path="/input/test.ims",
            output_spec=output_spec,
            data_shape=(64, 128, 256),
            shard_shape=(64, 128, 256),  # Single shard
            data_path="/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        )

        self.assertEqual(len(tasks), 1)
        self.assertEqual(tasks[0]["shard_index"], (0, 0, 0))
        self.assertEqual(tasks[0]["shard_shape"], (64, 128, 256))
        self.assertEqual(tasks[0]["data_shape"], (64, 128, 256))

    def test_create_shard_tasks_multiple_shards(self):
        """Test create_shard_tasks with multiple shards."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_shard_tasks,
        )

        output_spec = {
            "driver": "zarr3",
            "kvstore": {"driver": "file", "path": "/output/test.zarr/0"},
        }

        tasks = create_shard_tasks(
            imaris_path="/input/test.ims",
            output_spec=output_spec,
            data_shape=(128, 256, 512),
            shard_shape=(64, 128, 256),  # 2x2x2 = 8 shards
            data_path="/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data",
        )

        self.assertEqual(len(tasks), 8)

        # Check first shard
        self.assertEqual(tasks[0]["shard_index"], (0, 0, 0))

        # Check last shard
        self.assertEqual(tasks[-1]["shard_index"], (1, 1, 1))


if __name__ == "__main__":
    unittest.main()
