"""Tests for the multiscale pyramid downsampling functions."""

import asyncio
import math
import os
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import psutil


# S3 test configuration
S3_BUCKET = "aind-scratch-data"
S3_PREFIX = "exaSPIM_683791-screen_2026-01-26_14-53-41"
DATA_DIR = Path(
    "/allen/aind/stage/exaSPIM/exaSPIM_683791-screen_2026-01-26_14-53-41/exaSPIM"
)


class ResourceMonitor:
    """Monitor CPU and memory usage during test execution."""

    def __init__(self, interval: float = 1.0):
        """
        Initialize the resource monitor.

        Parameters
        ----------
        interval : float
            Sampling interval in seconds (default 1.0)
        """
        self.interval = interval
        self.process = psutil.Process(os.getpid())
        self._stop_event = threading.Event()
        self._thread = None

        # Metrics storage
        self.memory_samples = []  # RSS memory in MB
        self.cpu_samples = []  # CPU percent
        self.timestamps = []  # Time since start

    def start(self):
        """Start monitoring in background thread."""
        self._stop_event.clear()
        self.memory_samples = []
        self.cpu_samples = []
        self.timestamps = []
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self):
        """Stop monitoring and wait for thread to finish."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def _monitor_loop(self):
        """Background loop to sample resource usage."""
        while not self._stop_event.is_set():
            try:
                mem_info = self.process.memory_info()
                cpu_percent = self.process.cpu_percent(interval=None)

                self.memory_samples.append(mem_info.rss / (1024 * 1024))  # MB
                self.cpu_samples.append(cpu_percent)
                self.timestamps.append(time.time() - self._start_time)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break

            self._stop_event.wait(self.interval)

    def get_stats(self) -> dict:
        """Get summary statistics of resource usage."""
        if not self.memory_samples:
            return {"error": "No samples collected"}

        return {
            "memory_mb": {
                "min": min(self.memory_samples),
                "max": max(self.memory_samples),
                "avg": sum(self.memory_samples) / len(self.memory_samples),
                "final": self.memory_samples[-1],
            },
            "cpu_percent": {
                "min": min(self.cpu_samples) if self.cpu_samples else 0,
                "max": max(self.cpu_samples) if self.cpu_samples else 0,
                "avg": (
                    sum(self.cpu_samples) / len(self.cpu_samples)
                    if self.cpu_samples
                    else 0
                ),
            },
            "duration_seconds": self.timestamps[-1] if self.timestamps else 0,
            "num_samples": len(self.memory_samples),
        }

    def print_stats(self, operation_name: str = "Operation"):
        """Print formatted statistics."""
        stats = self.get_stats()
        if "error" in stats:
            print(f"  ⚠ {stats['error']}")
            return

        print(f"\n  📊 Resource Usage for '{operation_name}':")
        print(f"     Duration: {stats['duration_seconds']:.2f}s")
        print(
            f"     Memory (MB): min={stats['memory_mb']['min']:.1f}, "
            f"max={stats['memory_mb']['max']:.1f}, "
            f"avg={stats['memory_mb']['avg']:.1f}"
        )
        print(
            f"     CPU (%): min={stats['cpu_percent']['min']:.1f}, "
            f"max={stats['cpu_percent']['max']:.1f}, "
            f"avg={stats['cpu_percent']['avg']:.1f}"
        )


@contextmanager
def timed_operation(name: str, monitor_resources: bool = True, interval: float = 0.5):
    """
    Context manager for timing operations with optional resource monitoring.

    Parameters
    ----------
    name : str
        Name of the operation for logging
    monitor_resources : bool
        Whether to monitor CPU/memory (default True)
    interval : float
        Resource sampling interval in seconds

    Yields
    ------
    dict
        Stats dictionary that will be populated after the operation
    """
    stats = {}
    monitor = ResourceMonitor(interval=interval) if monitor_resources else None

    print(f"\n  ⏱ Starting: {name}")
    start_time = time.time()

    if monitor:
        monitor.start()

    try:
        yield stats
    finally:
        elapsed = time.time() - start_time

        if monitor:
            monitor.stop()
            resource_stats = monitor.get_stats()
            stats.update(resource_stats)
            monitor.print_stats(name)
        else:
            print(f"  ⏱ {name} completed in {elapsed:.2f}s")

        stats["elapsed_seconds"] = elapsed


class TestComputeDownsampledShape(unittest.TestCase):
    """Test suite for compute_downsampled_shape function."""

    def test_exact_division(self):
        """Test when shape divides evenly by factor."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        result = compute_downsampled_shape(
            shape=(100, 200, 300),
            downsample_factor=(2, 2, 2),
        )
        self.assertEqual(result, (50, 100, 150))

    def test_ceiling_division(self):
        """Test when shape doesn't divide evenly (uses ceiling)."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        result = compute_downsampled_shape(
            shape=(101, 201, 301),
            downsample_factor=(2, 2, 2),
        )
        self.assertEqual(result, (51, 101, 151))

    def test_5d_shape(self):
        """Test with 5D shape (T, C, Z, Y, X)."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        result = compute_downsampled_shape(
            shape=(1, 1, 100, 200, 300),
            downsample_factor=(1, 1, 2, 2, 2),
        )
        self.assertEqual(result, (1, 1, 50, 100, 150))

    def test_mixed_factors(self):
        """Test with different factors per dimension."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        result = compute_downsampled_shape(
            shape=(1, 1, 100, 200, 400),
            downsample_factor=(1, 1, 2, 4, 4),
        )
        self.assertEqual(result, (1, 1, 50, 50, 100))

    def test_mismatched_dimensions_raises(self):
        """Test that mismatched dimensions raise ValueError."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        with self.assertRaises(ValueError):
            compute_downsampled_shape(
                shape=(100, 200, 300),
                downsample_factor=(2, 2),  # Only 2D factor for 3D shape
            )

    def test_single_element_dimensions(self):
        """Test with dimensions that become 1 after downsampling."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
        )

        result = compute_downsampled_shape(
            shape=(2, 3, 5),
            downsample_factor=(2, 2, 2),
        )
        self.assertEqual(result, (1, 2, 3))


class TestBuildKvstoreSpec(unittest.TestCase):
    """Test suite for _build_kvstore_spec function."""

    def test_local_kvstore(self):
        """Test building local file kvstore spec."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _build_kvstore_spec,
        )

        result = _build_kvstore_spec(
            path="/data/test.zarr",
            bucket_name=None,
        )

        self.assertEqual(result["driver"], "file")
        self.assertEqual(result["path"], "/data/test.zarr")

    def test_s3_kvstore(self):
        """Test building S3 kvstore spec."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _build_kvstore_spec,
        )

        result = _build_kvstore_spec(
            path="path/to/data.zarr",
            bucket_name="my-bucket",
            aws_region="us-east-1",
            cpu_cnt=8,
            read_cache_bytes=2 << 30,
        )

        self.assertEqual(result["driver"], "s3")
        self.assertEqual(result["bucket"], "my-bucket")
        self.assertEqual(result["path"], "path/to/data.zarr")
        self.assertEqual(result["aws_region"], "us-east-1")
        self.assertIn("context", result)
        self.assertEqual(
            result["context"]["cache_pool"]["total_bytes_limit"], 2 << 30
        )

    def test_s3_kvstore_default_cpu_count(self):
        """Test S3 kvstore uses multiprocessing cpu_count when not specified."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            _build_kvstore_spec,
        )

        with patch("multiprocessing.cpu_count", return_value=16):
            result = _build_kvstore_spec(
                path="data.zarr",
                bucket_name="bucket",
                cpu_cnt=None,
            )

            self.assertEqual(
                result["context"]["data_copy_concurrency"]["limit"], 16
            )


class TestCreateScaleSpec(unittest.TestCase):
    """Test suite for create_scale_spec function."""

    def test_basic_5d_spec(self):
        """Test creating a basic 5D scale spec."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="/data/dataset.zarr",
            data_shape=(1, 1, 100, 200, 300),
            data_dtype="uint16",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
            scale=0,
            codec="zstd",
            codec_level=3,
            bucket_name=None,
        )

        self.assertEqual(spec["driver"], "zarr3")
        self.assertEqual(spec["metadata"]["shape"], [1, 1, 100, 200, 300])
        self.assertEqual(spec["metadata"]["data_type"], "uint16")
        self.assertTrue(spec["create"])

    def test_scale_path_appended(self):
        """Test that scale index is appended to path."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="/data/dataset.zarr",
            data_shape=(1, 1, 50, 100, 150),
            data_dtype="uint16",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
            scale=3,
            bucket_name=None,
        )

        self.assertEqual(
            spec["kvstore"]["path"], "/data/dataset.zarr/3"
        )

    def test_shard_and_chunk_clamped_to_shape(self):
        """Test that shard and chunk shapes are clamped to data shape."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        # Shape smaller than shard/chunk
        spec = create_scale_spec(
            output_path="/data/dataset.zarr",
            data_shape=(1, 1, 10, 20, 30),
            data_dtype="uint16",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
            scale=5,
            bucket_name=None,
        )

        # Shard shape should be clamped to data shape
        chunk_grid_shape = spec["metadata"]["chunk_grid"]["configuration"][
            "chunk_shape"
        ]
        self.assertEqual(chunk_grid_shape, [1, 1, 10, 20, 30])

        # Inner chunk shape should also be clamped
        sharding_config = spec["metadata"]["codecs"][0]["configuration"]
        self.assertEqual(sharding_config["chunk_shape"], [1, 1, 10, 20, 30])

    def test_s3_bucket_spec(self):
        """Test spec creation with S3 bucket."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="prefix/dataset.zarr",
            data_shape=(1, 1, 100, 200, 300),
            data_dtype="float32",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
            scale=0,
            bucket_name="my-bucket",
            aws_region="us-west-2",
        )

        self.assertEqual(spec["kvstore"]["driver"], "s3")
        self.assertEqual(spec["kvstore"]["bucket"], "my-bucket")

    def test_sharding_codec_structure(self):
        """Test that sharding codec is properly structured."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
        )

        spec = create_scale_spec(
            output_path="/data/dataset.zarr",
            data_shape=(1, 1, 100, 200, 300),
            data_dtype="uint16",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
            scale=0,
            codec="blosc",
            codec_level=5,
            bucket_name=None,
        )

        codecs = spec["metadata"]["codecs"]
        self.assertEqual(len(codecs), 1)
        self.assertEqual(codecs[0]["name"], "sharding_indexed")

        sharding_config = codecs[0]["configuration"]
        self.assertEqual(sharding_config["index_location"], "end")

        # Check inner codecs
        inner_codecs = sharding_config["codecs"]
        self.assertEqual(inner_codecs[0]["name"], "transpose")
        self.assertEqual(inner_codecs[1]["name"], "blosc")
        self.assertEqual(inner_codecs[1]["configuration"]["level"], 5)


class TestCreateDownsampleDataset(unittest.TestCase):
    """Test suite for create_downsample_dataset function."""

    @patch("aind_exaspim_data_transformation.compress.imaris_to_zarr.ts")
    def test_downsample_spec_structure(self, mock_ts):
        """Test that the downsample spec is correctly structured."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_downsample_dataset,
        )

        # Setup mocks
        mock_downsampled_view = AsyncMock()
        mock_downsampled_view.shape = (1, 1, 50, 100, 150)
        mock_downsampled_view.read = AsyncMock(
            return_value=np.zeros((1, 1, 50, 100, 150))
        )

        mock_source = MagicMock()
        mock_source.dtype.name = "uint16"
        mock_downsampled_view.base = mock_source

        mock_dest = AsyncMock()

        async def mock_open(spec=None, **kwargs):
            if spec and spec.get("driver") == "downsample":
                return mock_downsampled_view
            return mock_dest

        mock_ts.open = mock_open

        # Run the function
        result = asyncio.run(
            create_downsample_dataset(
                dataset_path="/data/test.zarr",
                start_scale=0,
                downsample_factor=[2, 2, 2],
                downsample_mode="mean",
                shard_shape=(1, 1, 64, 64, 64),
                chunk_shape=(1, 1, 32, 32, 32),
            )
        )

        self.assertEqual(result, (1, 1, 50, 100, 150))

    def test_downsample_factor_padded_to_5d(self):
        """Test that 3D downsample factor is padded to 5D."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_downsample_dataset,
        )

        # This test verifies the padding logic by checking a ValueError
        # would occur if padding wasn't correct
        # We mock ts.open to capture the spec
        captured_spec = {}

        async def capture_spec(spec=None, **kwargs):
            if spec and spec.get("driver") == "downsample":
                captured_spec.update(spec)
                mock_view = AsyncMock()
                mock_view.shape = (1, 1, 50, 100, 150)
                mock_view.read = AsyncMock(
                    return_value=np.zeros((1, 1, 50, 100, 150))
                )
                mock_source = MagicMock()
                mock_source.dtype.name = "uint16"
                mock_view.base = mock_source
                return mock_view
            return AsyncMock()

        with patch(
            "aind_exaspim_data_transformation.compress.imaris_to_zarr.ts"
        ) as mock_ts:
            mock_ts.open = capture_spec
            asyncio.run(
                create_downsample_dataset(
                    dataset_path="/data/test.zarr",
                    start_scale=0,
                    downsample_factor=[2, 2, 2],  # 3D factor
                    downsample_mode="mean",
                    shard_shape=(1, 1, 64, 64, 64),
                    chunk_shape=(1, 1, 32, 32, 32),
                )
            )

        # Verify it was padded to 5D: [1, 1, 2, 2, 2]
        self.assertEqual(
            captured_spec["downsample_factors"], [1, 1, 2, 2, 2]
        )


class TestCreateDownsampleLevels(unittest.TestCase):
    """Test suite for create_downsample_levels function."""

    def test_no_downsampling_for_single_level(self):
        """Test that single level returns just base shape."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_downsample_levels,
        )

        result = create_downsample_levels(
            dataset_path="/data/test.zarr",
            base_shape=(1, 1, 100, 200, 300),
            n_levels=1,
            downsample_factor=(2, 2, 2),
            downsample_mode="mean",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
        )

        self.assertEqual(result, [(1, 1, 100, 200, 300)])

    @patch(
        "aind_exaspim_data_transformation.compress.imaris_to_zarr."
        "create_downsample_dataset"
    )
    def test_multiple_levels_calls_downsample(self, mock_create_ds):
        """Test that multiple levels triggers downsampling."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_downsample_levels,
        )

        # Mock create_downsample_dataset to return progressively smaller shapes
        shapes = [
            (1, 1, 50, 100, 150),
            (1, 1, 25, 50, 75),
            (1, 1, 13, 25, 38),
        ]
        
        # Create async mock that returns shapes sequentially
        async def mock_downsample(*args, **kwargs):
            return shapes.pop(0)
        
        mock_create_ds.side_effect = mock_downsample

        result = create_downsample_levels(
            dataset_path="/data/test.zarr",
            base_shape=(1, 1, 100, 200, 300),
            n_levels=4,
            downsample_factor=(2, 2, 2),
            downsample_mode="mean",
            shard_shape=(1, 1, 64, 64, 64),
            chunk_shape=(1, 1, 32, 32, 32),
        )

        # Should have called create_downsample_dataset 3 times (levels 1, 2, 3)
        self.assertEqual(mock_create_ds.call_count, 3)

        # Verify the calls were made with correct start_scale
        calls = mock_create_ds.call_args_list
        self.assertEqual(calls[0].kwargs["start_scale"], 0)
        self.assertEqual(calls[1].kwargs["start_scale"], 1)
        self.assertEqual(calls[2].kwargs["start_scale"], 2)


class TestImarisToZarrParallelNewParams(unittest.TestCase):
    """Test suite for updated imaris_to_zarr_parallel function."""

    def test_function_accepts_new_parameters(self):
        """Test that function accepts scale_factor and downsample_mode."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )
        import inspect

        sig = inspect.signature(imaris_to_zarr_parallel)
        params = list(sig.parameters.keys())

        self.assertIn("scale_factor", params)
        self.assertIn("downsample_mode", params)

    def test_default_parameter_values(self):
        """Test default values for new parameters."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )
        import inspect

        sig = inspect.signature(imaris_to_zarr_parallel)
        params = sig.parameters

        self.assertEqual(params["scale_factor"].default, (2, 2, 2))
        self.assertEqual(params["downsample_mode"].default, "mean")


class TestLiveDownsamplePyramid(unittest.TestCase):
    """Live integration tests using real IMS files and S3 storage.
    
    These tests require:
    - Access to the staging data directory
    - AWS credentials configured for S3 access
    - The aind-scratch-data bucket to be accessible
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with real data paths."""
        cls.data_dir = DATA_DIR
        cls.s3_bucket = S3_BUCKET
        cls.s3_prefix = S3_PREFIX
        
        # Find IMS files
        if cls.data_dir.exists():
            cls.ims_files = list(cls.data_dir.glob("*.ims"))
        else:
            cls.ims_files = []
        
        # Create temp directory for local tests
        cls.temp_dir = tempfile.mkdtemp(prefix="test_downsample_pyramid_")
        cls.output_dir = Path(cls.temp_dir)
        
        # System info
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        print(f"\n{'='*60}")
        print("=== Live Downsample Pyramid Test Setup ===")
        print(f"{'='*60}")
        print(
            f"System: {cpu_count} CPUs, {mem.total / (1024**3):.1f} GB RAM "
            f"({mem.available / (1024**3):.1f} GB available)"
        )
        print(f"Data directory: {cls.data_dir}")
        print(f"Data directory exists: {cls.data_dir.exists()}")
        if cls.ims_files:
            print(f"Found {len(cls.ims_files)} IMS files:")
            for f in cls.ims_files:
                print(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
        else:
            print("Found 0 IMS files")
        print(f"S3 bucket: {cls.s3_bucket}")
        print(f"S3 prefix: {cls.s3_prefix}")
        print(f"Temp output dir: {cls.output_dir}")
        print("=" * 60)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary directory."""
        import shutil
        if cls.output_dir.exists():
            print(f"\nCleaning up: {cls.output_dir}")
            shutil.rmtree(cls.output_dir)

    def test_compute_downsampled_shape_real_dimensions(self):
        """Test shape calculation with real-world image dimensions."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            compute_downsampled_shape,
            ImarisReader,
        )
        
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        print(f"\nTesting with: {ims_file.name}")
        
        with ImarisReader(str(ims_file)) as reader:
            shape_3d = reader.get_shape()
            shape_5d = (1, 1) + tuple(shape_3d)
            print(f"  Original shape (5D): {shape_5d}")
            
            # Test downsampling calculation
            for level in range(5):
                factor_5d = (1, 1, 2**level, 2**level, 2**level)
                if level == 0:
                    factor_5d = (1, 1, 1, 1, 1)
                else:
                    factor_5d = (1, 1, 2, 2, 2)
                    
                if level == 0:
                    downsampled = shape_5d
                else:
                    # Apply factor iteratively
                    current = shape_5d
                    for _ in range(level):
                        current = compute_downsampled_shape(current, (1, 1, 2, 2, 2))
                    downsampled = current
                    
                print(f"  Level {level}: {downsampled}")
                
                # Verify dimensions are positive
                self.assertTrue(all(d > 0 for d in downsampled))

    def test_create_scale_spec_for_real_data(self):
        """Test creating TensorStore specs with real data dimensions."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            create_scale_spec,
            ImarisReader,
        )
        
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        stack_name = f"{ims_file.stem}.ome.zarr"
        s3_output_path = f"{self.s3_prefix}/{stack_name}"
        
        print(f"\nTesting create_scale_spec for: {ims_file.name}")
        print(f"  S3 output path: s3://{self.s3_bucket}/{s3_output_path}")
        
        with ImarisReader(str(ims_file)) as reader:
            shape_3d = reader.get_shape()
            shape_5d = (1, 1) + tuple(shape_3d)
            dtype = reader.get_dtype()
            
            # Test spec creation for S3
            spec = create_scale_spec(
                output_path=s3_output_path,
                data_shape=shape_5d,
                data_dtype=str(dtype),
                shard_shape=(1, 1, 512, 512, 512),
                chunk_shape=(1, 1, 128, 128, 128),
                scale=0,
                codec="zstd",
                codec_level=3,
                bucket_name=self.s3_bucket,
                aws_region="us-west-2",
            )
            
            print(f"  Generated spec for shape: {shape_5d}")
            print(f"  Spec driver: {spec['driver']}")
            print(f"  Kvstore driver: {spec['kvstore']['driver']}")
            print(f"  Kvstore bucket: {spec['kvstore']['bucket']}")
            
            self.assertEqual(spec["driver"], "zarr3")
            self.assertEqual(spec["kvstore"]["driver"], "s3")
            self.assertEqual(spec["kvstore"]["bucket"], self.s3_bucket)
            self.assertEqual(spec["metadata"]["shape"], list(shape_5d))

    def test_imaris_to_zarr_parallel_local_single_level(self):
        """Test parallel writer with local output and single level."""
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )
        
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        stack_name = f"{ims_file.stem}_test_local.ome.zarr"
        
        print(f"\n{'='*60}")
        print(f"Testing local parallel write: {ims_file.name}")
        print(f"Output: {self.output_dir / stack_name}")
        print(f"{'='*60}")
        
        with timed_operation(
            f"Local write (single level): {ims_file.name}",
            monitor_resources=True,
            interval=0.5,
        ) as stats:
            result_path = imaris_to_zarr_parallel(
                imaris_path=str(ims_file),
                output_path=str(self.output_dir),
                chunk_shape=(128, 128, 128),
                shard_shape=(256, 256, 256),
                n_lvls=1,  # Single level only for speed
                scale_factor=(2, 2, 2),
                downsample_mode="mean",
                stack_name=stack_name,
                codec="zstd",
                codec_level=3,
                bucket_name=None,  # Local
                max_concurrent_writes=8,
            )
        
        print(f"  Output path: {result_path}")
        
        # Verify output exists
        output_path = Path(result_path)
        self.assertTrue(output_path.exists())
        
        # Check for level 0
        level_0 = output_path / "0"
        self.assertTrue(level_0.exists())
        
        print(f"  ✓ Output verified at {output_path}")

    @unittest.skip("S3 write test - run manually when needed")
    def test_imaris_to_zarr_parallel_s3_with_pyramid(self):
        """Test parallel writer with S3 output and multiple pyramid levels.
        
        This test writes to S3 and should be run manually to avoid
        unintended S3 writes during automated testing.
        """
        from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
            imaris_to_zarr_parallel,
        )
        
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        stack_name = f"{ims_file.stem}_pyramid_test.ome.zarr"
        s3_output_path = f"s3://{self.s3_bucket}/{self.s3_prefix}/{stack_name}"
        
        print(f"\n{'='*60}")
        print(f"Testing S3 parallel write with pyramid: {ims_file.name}")
        print(f"Output: {s3_output_path}")
        print(f"{'='*60}")
        
        with timed_operation(
            f"S3 write with pyramid (8 levels): {ims_file.name}",
            monitor_resources=True,
            interval=0.5,
        ) as stats:
            result_path = imaris_to_zarr_parallel(
                imaris_path=str(ims_file),
                output_path=self.s3_prefix,
                chunk_shape=(128, 128, 128),
                shard_shape=(512, 512, 512),
                n_lvls=8,  # Generate 8 pyramid levels
                scale_factor=(2, 2, 2),
                downsample_mode="mean",
                stack_name=stack_name,
                codec="zstd",
                codec_level=3,
                bucket_name=self.s3_bucket,
                max_concurrent_writes=16,
            )
        
        print(f"  Output path: {result_path}")
        
        # Print summary
        if "memory_mb" in stats:
            print(f"\n  📈 Performance Summary:")
            print(f"     Peak memory: {stats['memory_mb']['max']:.1f} MB")
            print(f"     Avg memory: {stats['memory_mb']['avg']:.1f} MB")
            print(f"     Total time: {stats['elapsed_seconds']:.2f}s")
        
        self.assertTrue(result_path.startswith("s3://"))


if __name__ == "__main__":
    unittest.main()