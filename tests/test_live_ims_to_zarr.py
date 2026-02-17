"""Live integration tests for IMS to Zarr conversion using real data files

This test module is designed to work with actual IMS files to validate
the conversion pipeline end-to-end.
"""

import os
import shutil
import tempfile
import threading
import time
import unittest
from contextlib import contextmanager
from pathlib import Path

import psutil

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    imaris_to_zarr_distributed,
    imaris_to_zarr_parallel,
    imaris_to_zarr_writer,
)
from aind_exaspim_data_transformation.utils.io_utils import ImarisReader


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
def timed_operation(
    name: str, monitor_resources: bool = True, interval: float = 0.5
):
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


@unittest.skip("S3 write test - run manually when needed")
class TestLiveImsToZarr(unittest.TestCase):
    """Live tests using real IMS files from staging area"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with real data paths"""
        cls.data_dir = Path(
            "/allen/aind/stage/exaSPIM/"
            "exaSPIM_683791-screen_2026-01-26_14-53-41/exaSPIM"
        )
        cls.ims_files = list(cls.data_dir.glob("*.ims"))

        # Create a temporary output directory for test results
        cls.temp_dir = tempfile.mkdtemp(prefix="test_ims_to_zarr_")
        cls.output_dir = Path(cls.temp_dir)
        cls.delete_output_dir = False

        # System info
        mem = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()

        print(f"\n{'='*60}")
        print("=== Live Test Setup ===")
        print(f"{'='*60}")
        print(
            f"System: {cpu_count} CPUs, {mem.total / (1024**3):.1f} GB RAM "
            f"({mem.available / (1024**3):.1f} GB available)"
        )
        print(f"Data directory: {cls.data_dir}")
        print(f"Found {len(cls.ims_files)} IMS files:")
        for f in cls.ims_files:
            print(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
        print(f"Output directory: {cls.output_dir}")
        print("=" * 60)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory"""
        if cls.output_dir.exists() and cls.delete_output_dir:
            print(f"\nCleaning up test output directory: {cls.output_dir}")
            shutil.rmtree(cls.output_dir)

    def test_imaris_reader_opens_ims_file(self):
        """Test that ImarisReader correctly opens .ims files"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        print(f"\nTesting ImarisReader with: {ims_file.name}")

        reader = ImarisReader(str(ims_file))

        self.assertIsInstance(reader, ImarisReader)
        reader.close()

    print("✓ Successfully created ImarisReader")

    def test_imaris_reader_open_and_properties(self):
        """Test opening IMS file and reading basic properties"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        print(f"\nTesting ImarisReader with: {ims_file.name}")

        reader = ImarisReader(str(ims_file))

        with reader:
            # Test basic properties - get_shape returns actual HDF5 shape
            shape = reader.get_shape()
            print(f"  HDF5 Shape: {shape}")
            self.assertEqual(len(shape), 3, "Shape should be 3D (ZYX)")
            self.assertTrue(
                all(s > 0 for s in shape), "All dimensions should be > 0"
            )

            # Test metadata shape (excludes padding)
            metadata_shape = reader.get_metadata_shape()
            print(f"  Metadata Shape: {metadata_shape}")
            self.assertEqual(
                len(metadata_shape), 3, "Metadata shape should be 3D (ZYX)"
            )

            # Test voxel size - returns (voxel_size, unit)
            voxel_size, unit = reader.get_voxel_size()
            print(f"  Voxel size: {voxel_size} (unit: {unit})")
            self.assertEqual(
                len(voxel_size), 3, "Voxel size should be 3D (ZYX)"
            )
            self.assertTrue(
                all(v > 0 for v in voxel_size), "All voxel sizes should be > 0"
            )

            # Test chunks - from the underlying HDF5 dataset (3D)
            chunks = reader.get_chunks()
            print(f"  Chunks: {chunks}")
            self.assertEqual(len(chunks), 3, "Chunks should be 3D")

            # Test origin
            origin = reader.get_origin()
            print(f"  Origin: {origin}")
            self.assertEqual(len(origin), 3, "Origin should be 3D")

            # Test extent
            extent = reader.get_extent()
            print(f"  Extent: {extent}")
            self.assertEqual(len(extent), 3, "Extent should be 3D")

            # Test n_levels property
            n_levels = reader.n_levels
            print(f"  Number of resolution levels: {n_levels}")
            self.assertGreater(
                n_levels, 0, "Should have at least one resolution level"
            )

            print("✓ All properties read successfully")

    @unittest.skip("S3 write test - run manually when needed")
    def test_imaris_reader_chunking_strategy(self):
        """Report Imaris chunking strategy and per-dimension chunk counts"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        print(f"\nChunking strategy for: {ims_file.name}")

        with ImarisReader(str(ims_file)) as reader:
            shape = reader.get_shape()
            chunks = reader.get_chunks()

        self.assertEqual(len(shape), 3, "Shape should be 3D (ZYX)")
        self.assertEqual(len(chunks), 3, "Chunks should be 3D (ZYX)")

        chunk_counts = tuple(
            (dim + chunk - 1) // chunk for dim, chunk in zip(shape, chunks)
        )
        total_chunks = chunk_counts[0] * chunk_counts[1] * chunk_counts[2]

        print(f"  Shape (ZYX): {shape}")
        print(f"  Chunk size (ZYX): {chunks}")
        print(f"  Chunk counts (ZYX): {chunk_counts}")
        print(f"  Total chunks: {total_chunks}")

    def test_imaris_reader_as_dask_array(self):
        """Test loading IMS data as dask array"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        print(f"\n{'='*60}")
        print(f"Testing dask array loading with: {ims_file.name}")
        print(f"{'='*60}")

        reader = ImarisReader(str(ims_file))

        with reader:
            # Get the native chunks from the HDF5 file for efficient reading
            native_chunks = reader.get_chunks()
            print(f"  Using native chunks: {native_chunks}")

            with timed_operation("Create dask array", monitor_resources=False):
                dask_array = reader.as_dask_array(chunks=native_chunks)

            print(f"  Dask array shape: {dask_array.shape}")
            print(f"  Dask array dtype: {dask_array.dtype}")
            print(f"  Dask array chunks: {dask_array.chunksize}")

            self.assertEqual(dask_array.shape, reader.get_shape())
            self.assertEqual(dask_array.ndim, 3)  # ZYX - 3D array

            # Compute a small slice to verify data access
            with timed_operation(
                "Compute 10x10x10 slice", interval=0.5
            ) as stats:
                slice_data = dask_array[
                    0:10, 0:10, 0:10
                ].compute()  # ZYX slicing

            print(f"  Slice shape: {slice_data.shape}")
            print(f"  Slice dtype: {slice_data.dtype}")
            print(f"  Slice min/max: {slice_data.min()}/{slice_data.max()}")

            if stats.get("elapsed_seconds") is not None:
                print(f"  Slice compute time: {stats['elapsed_seconds']:.2f}s")
            print("\n✓ Dask array created and data accessible")

    def test_imaris_reader_get_dask_pyramid(self):
        """Test generating multi-resolution pyramid"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        print(f"\n{'='*60}")
        print(f"Testing pyramid generation with: {ims_file.name}")
        print(f"{'='*60}")

        reader = ImarisReader(str(ims_file))

        with reader:
            num_levels = 3

            print(f"  Generating {num_levels} pyramid levels")
            with timed_operation(
                "Generate dask pyramid", monitor_resources=False
            ):
                pyramid = reader.get_dask_pyramid(num_levels=num_levels)

            print(f"  Number of pyramid levels: {len(pyramid)}")
            self.assertEqual(len(pyramid), num_levels)

            for i, level in enumerate(pyramid):
                print(
                    f"Level {i}: shape={level.shape}, chunks={level.chunksize}"
                )
                self.assertEqual(level.ndim, 3)  # ZYX - 3D

                # Each level should be roughly half the size of the previous
                if i > 0:
                    prev_level = pyramid[i - 1]
                    for dim in [0, 1, 2]:  # ZYX dimensions
                        ratio = prev_level.shape[dim] / level.shape[dim]
                        self.assertGreater(
                            ratio, 1.5, f"Level {i} should be downsampled"
                        )

            print("\n✓ Pyramid generated successfully")

    @unittest.skip("S3 write test - run manually when needed")
    def test_imaris_to_zarr_writer_single_file(self):
        """Test full conversion pipeline: IMS -> Zarr with single file"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        output_path = self.output_dir
        expected_zarr_path = output_path / f"{ims_file.stem}.ome.zarr"

        print(f"\n{'='*60}")
        print("Testing full conversion pipeline:")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  Output directory: {output_path}")
        print(f"  Expected zarr: {expected_zarr_path}")

        # Conversion parameters
        n_lvls = 3
        scale_factor = [2, 2, 2]

        print(f"  Pyramid levels: {n_lvls}")
        print(f"  Scale factor: {scale_factor}")

        # Run conversion with resource monitoring
        with timed_operation(
            f"IMS to Zarr conversion ({ims_file.name})", interval=0.5
        ) as stats:
            imaris_to_zarr_writer(
                imaris_path=str(ims_file),
                output_path=str(output_path),
                n_lvls=n_lvls,
                scale_factor=scale_factor,
                voxel_size=None,  # Extract from file
                compressor_kwargs={"cname": "zstd", "clevel": 5, "shuffle": 1},
            )

        # Print throughput stats
        input_size_gb = ims_file.stat().st_size / (1024**3)
        if stats.get("elapsed_seconds", 0) > 0:
            throughput = input_size_gb / stats["elapsed_seconds"]
            print(f"\n  📈 Throughput: {throughput:.2f} GB/s")
            max_memory_mb = stats.get("memory_mb", {}).get("max", 0)
            print(f"  Peak memory: {max_memory_mb:.1f} MB")

        # Verify output
        self.assertTrue(
            expected_zarr_path.exists(),
            f"Output zarr should exist at {expected_zarr_path}",
        )

        # Check for pyramid levels
        for i in range(n_lvls):
            level_path = expected_zarr_path / str(i)
            self.assertTrue(
                level_path.exists(),
                f"Pyramid level {i} should exist at {level_path}",
            )
            print(f"  ✓ Level {i} created: {level_path}")

        # Check metadata
        zattrs_path = expected_zarr_path / ".zattrs"
        self.assertTrue(zattrs_path.exists(), "Zarr attributes should exist")
        print(f"  ✓ Metadata file created: {zattrs_path}")

        # Print output size
        import subprocess

        result = subprocess.run(
            ["du", "-sh", str(expected_zarr_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            output_size = result.stdout.strip().split()[0]
            print(f"  Output size: {output_size}")

    print("\n✓ Conversion completed successfully")

    @unittest.skip("S3 write test - run manually when needed")
    def test_imaris_to_zarr_writer_custom_voxel_size(self):
        """Test conversion with custom voxel size override"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]
        output_path = self.output_dir
        custom_stack_name = f"{ims_file.stem}_custom_voxel.ome.zarr"
        expected_zarr_path = output_path / custom_stack_name

        print(f"\n{'='*60}")
        print("Testing conversion with custom voxel size:")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")

        custom_voxel_size = [2.0, 1.0, 1.0]  # ZYX in microns
        print(f"  Custom voxel size: {custom_voxel_size}")

        with timed_operation(
            f"Custom voxel conversion ({ims_file.name})", interval=0.5
        ) as stats:
            imaris_to_zarr_writer(
                imaris_path=str(ims_file),
                output_path=str(output_path),
                n_lvls=2,
                scale_factor=[2, 2, 2],
                voxel_size=custom_voxel_size,
                stack_name=custom_stack_name,
            )

        self.assertTrue(expected_zarr_path.exists())
        if stats.get("elapsed_seconds") is not None:
            print(f"  Custom voxel elapsed: {stats['elapsed_seconds']:.2f}s")
        print("\n✓ Conversion with custom voxel size completed")

    def test_process_multiple_files(self):
        """Test processing multiple IMS files (if available)"""
        if len(self.ims_files) < 2:
            self.skipTest("Need at least 2 IMS files for this test")

        print(f"\n{'='*60}")
        print(f"Testing batch conversion of {len(self.ims_files)} files:")
        print(f"{'='*60}")

        total_size_gb = sum(f.stat().st_size for f in self.ims_files) / (
            1024**3
        )
        print(f"  Total input size: {total_size_gb:.2f} GB")

        with timed_operation(
            f"Batch conversion ({len(self.ims_files)} files)", interval=0.5
        ) as total_stats:
            for i, ims_file in enumerate(self.ims_files):
                expected_zarr_path = (
                    self.output_dir / f"{ims_file.stem}.ome.zarr"
                )

                with timed_operation(
                    f"File {i+1}/{len(self.ims_files)}: {ims_file.name}",
                    interval=0.5,
                ):
                    imaris_to_zarr_writer(
                        imaris_path=str(ims_file),
                        output_path=str(self.output_dir),
                        n_lvls=2,
                        scale_factor=[2, 2, 2],
                        voxel_size=None,
                    )

                self.assertTrue(expected_zarr_path.exists())
                print(f"    ✓ Completed: {expected_zarr_path}")

        # Print overall throughput
        if total_stats.get("elapsed_seconds", 0) > 0:
            throughput = total_size_gb / total_stats["elapsed_seconds"]
            print(f"\n  📈 Overall throughput: {throughput:.2f} GB/s")

        print(f"\n✓ All {len(self.ims_files)} files converted successfully")

    def test_imaris_to_zarr_parallel_to_s3(self):
        """Test TensorStore parallel writer with S3 output"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]

        # S3 test configuration
        s3_bucket = "aind-scratch-data"
        s3_prefix = "exaSPIM_683791-screen_2026-01-26_14-53-41"
        stack_name = f"{ims_file.stem}.ome.zarr"
        s3_output_path = f"s3://{s3_bucket}/{s3_prefix}/{stack_name}"

        print(f"\n{'='*60}")
        print("Testing TensorStore parallel writer to S3:")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  S3 Output: {s3_output_path}")

        # Conversion parameters optimized for S3
        chunk_shape = (128, 128, 128)
        shard_shape = (256, 256, 256)
        n_lvls = 1  # Start with just base level for testing

        print(f"  Chunk shape: {chunk_shape}")
        print(f"  Shard shape: {shard_shape}")
        print(f"  Pyramid levels: {n_lvls}")

        # Run conversion with resource monitoring
        with timed_operation(
            f"IMS to S3 Zarr ({ims_file.name})", interval=0.5
        ) as stats:
            result_path = imaris_to_zarr_parallel(
                imaris_path=str(ims_file),
                output_path=s3_prefix,
                voxel_size=None,  # Extract from file
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                n_lvls=n_lvls,
                channel_name=ims_file.stem,
                stack_name=stack_name,
                codec="zstd",
                codec_level=3,
                bucket_name=s3_bucket,
                max_concurrent_writes=16,
            )

        # Print throughput stats
        input_size_gb = ims_file.stat().st_size / (1024**3)
        if stats.get("elapsed_seconds", 0) > 0:
            throughput = input_size_gb / stats["elapsed_seconds"]
            print(f"\n  📈 Throughput: {throughput:.2f} GB/s")
            print(
                f"Peak memory: {stats.get('memory_mb', {}).get('max', 0):.1f} MB"
            )

        print(f"  Result path: {result_path}")
        print("\n✓ S3 upload completed successfully")

        # Verify the upload using boto3
        try:
            import boto3

            s3_client = boto3.client("s3")

            # List objects at the output path
            response = s3_client.list_objects_v2(
                Bucket=s3_bucket,
                Prefix=f"{s3_prefix}/{stack_name}/",
                MaxKeys=10,
            )

            if "Contents" in response:
                print("\n  📁 S3 Contents (first 10):")
                for obj in response["Contents"][:10]:
                    size_kb = obj["Size"] / 1024
                    print(f"     - {obj['Key']} ({size_kb:.1f} KB)")
                print(
                    f"     ... and {response.get('KeyCount', 0)} total objects"
                )
            else:
                print(f"\n  ⚠ No objects found at {s3_prefix}/{stack_name}/")

        except ImportError:
            print("\n  ⚠ boto3 not available for S3 verification")
        except Exception as e:
            print(f"\n  ⚠ S3 verification error: {e}")

    def test_imaris_to_zarr_parallel_to_s3_multi_level(self):
        """Test TensorStore parallel writer with S3 output and multiple pyramid levels"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]

        # S3 test configuration
        s3_bucket = "aind-scratch-data"
        s3_prefix = "exaSPIM_683791-screen_2026-01-26_14-53-41"
        stack_name = f"{ims_file.stem}_multilevel.ome.zarr"
        s3_output_path = f"s3://{s3_bucket}/{s3_prefix}/{stack_name}"

        print(f"\n{'='*60}")
        print("Testing TensorStore multi-level pyramid to S3:")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  S3 Output: {s3_output_path}")

        # Conversion parameters
        chunk_shape = (128, 128, 128)
        shard_shape = (256, 256, 256)
        n_lvls = 3  # Multiple pyramid levels

        print(f"  Chunk shape: {chunk_shape}")
        print(f"  Shard shape: {shard_shape}")
        print(f"  Pyramid levels: {n_lvls}")

        # Run conversion with resource monitoring
        with timed_operation(
            f"IMS to S3 Zarr multi-level ({ims_file.name})", interval=0.5
        ) as stats:
            result_path = imaris_to_zarr_parallel(
                imaris_path=str(ims_file),
                output_path=s3_prefix,
                voxel_size=None,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                n_lvls=n_lvls,
                channel_name=ims_file.stem,
                stack_name=stack_name,
                codec="zstd",
                codec_level=3,
                bucket_name=s3_bucket,
                max_concurrent_writes=16,
            )

        # Print throughput stats
        input_size_gb = ims_file.stat().st_size / (1024**3)
        if stats.get("elapsed_seconds", 0) > 0:
            throughput = input_size_gb / stats["elapsed_seconds"]
            print(f"\n  📈 Throughput: {throughput:.2f} GB/s")
            max_memory_mb = stats.get("memory_mb", {}).get("max", 0)
            print(f"  Peak memory: {max_memory_mb:.1f} MB")

        print(f"  Result path: {result_path}")

        # Verify pyramid levels in S3
        try:
            import boto3

            s3_client = boto3.client("s3")

            for level in range(n_lvls):
                response = s3_client.list_objects_v2(
                    Bucket=s3_bucket,
                    Prefix=f"{s3_prefix}/{stack_name}/{level}/",
                    MaxKeys=5,
                )

                if "Contents" in response:
                    print(f"\n  📁 Level {level} contents:")
                    for obj in response["Contents"][:5]:
                        size_kb = obj["Size"] / 1024
                        print(
                            f"     - {obj['Key'].split('/')[-1]} ({size_kb:.1f} KB)"
                        )
                else:
                    print(f"\n  ⚠ Level {level} not found")

        except ImportError:
            print("\n  ⚠ boto3 not available for S3 verification")
        except Exception as e:
            print(f"\n  ⚠ S3 verification error: {e}")

        print("\n✓ Multi-level S3 upload completed successfully")

    @unittest.skip("Distributed test - run manually when needed")
    def test_imaris_to_zarr_distributed_local(self):
        """Test distributed worker-centric conversion to local filesystem"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]

        print(f"\n{'='*60}")
        print("Testing distributed worker-centric conversion (local):")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  Output: {self.output_dir}")

        # Test with different shard sizes
        chunk_shape = (128, 128, 128)
        shard_shape = (256, 256, 256)  # ~32MB per shard for uint16
        n_lvls = 3

        print(f"  Chunk shape: {chunk_shape}")
        print(f"  Shard shape: {shard_shape}")
        print(f"  Pyramid levels: {n_lvls}")
        print(f"  translate_pyramid_levels: True")

        # Get shard info before conversion
        with ImarisReader(str(ims_file)) as reader:
            shape = reader.get_shape()
            print(f"  Data shape: {shape}")

        # Calculate expected shards
        import math

        shard_grid = tuple(
            math.ceil(s / sh) for s, sh in zip(shape, shard_shape)
        )
        total_shards = shard_grid[0] * shard_grid[1] * shard_grid[2]
        print(f"  Shard grid: {shard_grid} = {total_shards} total shards")

        # Run conversion WITHOUT Dask client (sequential for testing)
        with timed_operation(
            f"Distributed IMS to Zarr - sequential ({ims_file.name})",
            interval=0.5,
        ) as stats:
            result_path = imaris_to_zarr_distributed(
                imaris_path=str(ims_file),
                output_path=str(self.output_dir),
                voxel_size=None,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                n_lvls=n_lvls,
                channel_name=ims_file.stem,
                stack_name=f"{ims_file.stem}_distributed.ome.zarr",
                codec="zstd",
                codec_level=3,
                bucket_name=None,  # Local filesystem
                dask_client=None,  # Sequential processing
                translate_pyramid_levels=True,
            )

        # Calculate throughput
        if stats.get("elapsed_seconds", 0) > 0:
            throughput = input_size_gb / stats["elapsed_seconds"]
            print(f"\n  📈 Throughput: {throughput:.2f} GB/s")
            print(
                f"  Peak memory: {stats.get('memory_mb', {}).get('max', 0):.1f} MB"
            )

        print(f"  Result path: {result_path}")

        # Verify the output exists
        output_path = Path(result_path)
        self.assertTrue(
            output_path.exists() or result_path.startswith("s3://"),
            f"Output path should exist: {result_path}",
        )

        print("\n✓ Distributed conversion (sequential) completed successfully")

    @unittest.skip("Distributed S3 test - run manually when needed")
    def test_imaris_to_zarr_distributed_to_s3(self):
        """Test distributed worker-centric conversion to S3 with Dask.

        Benchmarks multiple worker configurations to find optimal throughput.

        ============================================================
        BENCHMARK RESULTS
        ============================================================
        File: tile_000000_ch_488.ims (9.63 GB)
        Shard: (512, 512, 512), Levels: 3
        Workers   Time (s)     GB/s    Peak MB
        ----------------------------------------
                4      143.0    0.067        163
                8      115.2    0.084        168
                16      114.8    0.084        178
        ============================================================
        """
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]

        # S3 test configuration
        s3_bucket = "aind-scratch-data"
        s3_prefix = "exaSPIM_683791-screen_2026-01-26_14-53-41"

        # Larger shards for S3 efficiency
        chunk_shape = (128, 128, 128)
        shard_shape = (512, 512, 512)  # ~256MB per shard
        n_lvls = 3

        input_size_gb = ims_file.stat().st_size / (1024**3)

        # Get shard info
        with ImarisReader(str(ims_file)) as reader:
            shape = reader.get_shape()

        import math

        shard_grid = tuple(
            math.ceil(s / sh) for s, sh in zip(shape, shard_shape)
        )
        total_shards = shard_grid[0] * shard_grid[1] * shard_grid[2]

        # Benchmark with different worker counts
        worker_configs = [4, 8, 16]

        print(f"\n{'='*60}")
        print("Benchmarking distributed S3 upload with Dask")
        print(f"{'='*60}")
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  Data shape: {shape}")
        print(f"  Chunk: {chunk_shape}  Shard: {shard_shape}")
        print(f"  Shard grid: {shard_grid} = {total_shards} total shards")
        print(f"  Pyramid levels: {n_lvls} (translate from Imaris)")
        print(f"  Worker configs to test: {worker_configs}")
        print(f"{'='*60}")

        from dask.distributed import Client, LocalCluster

        results = []

        for n_workers in worker_configs:
            stack_name = f"{ims_file.stem}_dask_{n_workers}w.ome.zarr"
            s3_output = f"s3://{s3_bucket}/{s3_prefix}/{stack_name}"

            print(f"\n--- {n_workers} workers ---")
            print(f"  Output: {s3_output}")

            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            print(f"  Dashboard: {client.dashboard_link}")

            try:
                with timed_operation(
                    f"{n_workers} workers", interval=0.5
                ) as stats:
                    result_path = imaris_to_zarr_distributed(
                        imaris_path=str(ims_file),
                        output_path=s3_prefix,
                        voxel_size=None,
                        chunk_shape=chunk_shape,
                        shard_shape=shard_shape,
                        n_lvls=n_lvls,
                        channel_name=ims_file.stem,
                        stack_name=stack_name,
                        codec="zstd",
                        codec_level=3,
                        bucket_name=s3_bucket,
                        dask_client=client,
                        translate_pyramid_levels=True,
                    )

                elapsed = stats.get("elapsed_seconds", 0)
                throughput = input_size_gb / elapsed if elapsed > 0 else 0
                peak_mem = stats.get("memory_mb", {}).get("max", 0)
                results.append((n_workers, elapsed, throughput, peak_mem))
                print(
                    f"  ✓ {elapsed:.1f}s  {throughput:.3f} GB/s  "
                    f"peak_mem={peak_mem:.0f} MB"
                )
            finally:
                client.close()
                cluster.close()

        # Print summary table
        print(f"\n{'='*60}")
        print("BENCHMARK RESULTS")
        print(f"{'='*60}")
        print(f"  File: {ims_file.name} ({input_size_gb:.2f} GB)")
        print(f"  Shard: {shard_shape}, Levels: {n_lvls}")
        print(
            f"  {'Workers':>8} {'Time (s)':>10} {'GB/s':>8} "
            f"{'Peak MB':>10}"
        )
        print(f"  {'-'*40}")
        for n_w, t, tp, mem in results:
            print(f"  {n_w:>8} {t:>10.1f} {tp:>8.3f} {mem:>10.0f}")
        print(f"{'='*60}")

        # Verify at least one upload worked
        self.assertTrue(len(results) > 0)
        # All runs should complete
        self.assertEqual(len(results), len(worker_configs))

    @unittest.skip("Distributed Dask test - run manually with cluster")
    def test_imaris_to_zarr_distributed_with_dask(self):
        """Test distributed conversion with actual Dask workers"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")

        ims_file = self.ims_files[0]

        print(f"\n{'='*60}")
        print("Testing distributed conversion with Dask cluster:")
        print(f"{'='*60}")
        input_size_gb = ims_file.stat().st_size / (1024**3)
        print(f"  Input: {ims_file.name} ({input_size_gb:.2f} GB)")

        # Configuration
        chunk_shape = (128, 128, 128)
        shard_shape = (256, 256, 256)
        n_lvls = 1
        n_workers = 4  # Use 4 local workers for testing

        print(f"  Chunk shape: {chunk_shape}")
        print(f"  Shard shape: {shard_shape}")
        print(f"  Dask workers: {n_workers}")

        # Create a local Dask cluster
        try:
            from dask.distributed import Client, LocalCluster

            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            print(f"  Dask dashboard: {client.dashboard_link}")
        except ImportError:
            self.skipTest("dask.distributed not available")
            return

        try:
            with timed_operation(
                f"Distributed IMS to Zarr with Dask ({ims_file.name})",
                interval=0.5,
            ) as stats:
                result_path = imaris_to_zarr_distributed(
                    imaris_path=str(ims_file),
                    output_path=str(self.output_dir),
                    voxel_size=None,
                    chunk_shape=chunk_shape,
                    shard_shape=shard_shape,
                    n_lvls=n_lvls,
                    channel_name=ims_file.stem,
                    stack_name=f"{ims_file.stem}_dask_distributed.ome.zarr",
                    codec="zstd",
                    codec_level=3,
                    bucket_name=None,
                    dask_client=client,
                )

            # Calculate throughput
            if stats.get("elapsed_seconds", 0) > 0:
                throughput = input_size_gb / stats["elapsed_seconds"]
                print(f"\n  📈 Throughput: {throughput:.2f} GB/s")
                print(
                    f"  Peak memory: {stats.get('memory_mb', {}).get('max', 0):.1f} MB"
                )

            print(f"  Result path: {result_path}")
            print("\n✓ Distributed Dask conversion completed successfully")

        finally:
            client.close()
            cluster.close()


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
