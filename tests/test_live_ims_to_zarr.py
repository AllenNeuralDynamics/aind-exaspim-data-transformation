"""Live integration tests for IMS to Zarr conversion using real data files

This test module is designed to work with actual IMS files to validate
the conversion pipeline end-to-end.
"""

import unittest
from pathlib import Path
import shutil
import tempfile

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    DataReaderFactory,
    ImarisReader,
    imaris_to_zarr_writer,
)


class TestLiveImsToZarr(unittest.TestCase):
    """Live tests using real IMS files from staging area"""

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures with real data paths"""
        cls.data_dir = Path("/allen/aind/stage/exaSPIM/exaSPIM_683791-screen_2026-01-26_14-53-41/exaSPIM")
        cls.ims_files = list(cls.data_dir.glob("*.ims"))
        
        # Create a temporary output directory for test results
        cls.temp_dir = tempfile.mkdtemp(prefix="test_ims_to_zarr_")
        cls.output_dir = Path(cls.temp_dir)
        cls.delete_output_dir = False
        
        print(f"\n=== Live Test Setup ===")
        print(f"Data directory: {cls.data_dir}")
        print(f"Found {len(cls.ims_files)} IMS files:")
        for f in cls.ims_files:
            print(f"  - {f.name} ({f.stat().st_size / (1024**3):.2f} GB)")
        print(f"Output directory: {cls.output_dir}")
        print("=" * 50)

    @classmethod
    def tearDownClass(cls):
        """Clean up temporary test directory"""
        if cls.output_dir.exists() and cls.delete_output_dir:
            print(f"\nCleaning up test output directory: {cls.output_dir}")
            shutil.rmtree(cls.output_dir)

    def test_data_reader_factory_creates_imaris_reader(self):
        """Test that DataReaderFactory correctly creates ImarisReader for .ims files"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        print(f"\nTesting DataReaderFactory with: {ims_file.name}")
        
        reader = DataReaderFactory.create(str(ims_file))
        
        self.assertIsInstance(reader, ImarisReader)
        print(f"✓ Successfully created ImarisReader")

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
            self.assertTrue(all(s > 0 for s in shape), "All dimensions should be > 0")
            
            # Test metadata shape (excludes padding)
            metadata_shape = reader.get_metadata_shape()
            print(f"  Metadata Shape: {metadata_shape}")
            self.assertEqual(len(metadata_shape), 3, "Metadata shape should be 3D (ZYX)")
            
            # Test voxel size - returns (voxel_size, unit)
            voxel_size, unit = reader.get_voxel_size()
            print(f"  Voxel size: {voxel_size} (unit: {unit})")
            self.assertEqual(len(voxel_size), 3, "Voxel size should be 3D (ZYX)")
            self.assertTrue(all(v > 0 for v in voxel_size), "All voxel sizes should be > 0")
            
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
            self.assertGreater(n_levels, 0, "Should have at least one resolution level")
            
            print(f"✓ All properties read successfully")

    def test_imaris_reader_as_dask_array(self):
        """Test loading IMS data as dask array"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        print(f"\nTesting dask array loading with: {ims_file.name}")
        
        reader = ImarisReader(str(ims_file))
        
        with reader:
            # Get the native chunks from the HDF5 file for efficient reading
            native_chunks = reader.get_chunks()
            print(f"  Using native chunks: {native_chunks}")
            
            dask_array = reader.as_dask_array(chunks=native_chunks)
            
            print(f"  Dask array shape: {dask_array.shape}")
            print(f"  Dask array dtype: {dask_array.dtype}")
            print(f"  Dask array chunks: {dask_array.chunksize}")
            
            self.assertEqual(dask_array.shape, reader.get_shape())
            self.assertEqual(dask_array.ndim, 3)  # ZYX - 3D array
            
            # Compute a small slice to verify data access
            print(f"  Computing small slice to verify data access...")
            slice_data = dask_array[0:10, 0:10, 0:10].compute()  # ZYX slicing
            print(f"  Slice shape: {slice_data.shape}")
            print(f"  Slice dtype: {slice_data.dtype}")
            print(f"  Slice min/max: {slice_data.min()}/{slice_data.max()}")
            
            print(f"✓ Dask array created and data accessible")

    def test_imaris_reader_get_dask_pyramid(self):
        """Test generating multi-resolution pyramid"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        print(f"\nTesting pyramid generation with: {ims_file.name}")
        
        reader = ImarisReader(str(ims_file))
        
        with reader:
            num_levels = 3
            
            print(f"  Generating {num_levels} pyramid levels")
            pyramid = reader.get_dask_pyramid(num_levels=num_levels)
            
            print(f"  Number of pyramid levels: {len(pyramid)}")
            self.assertEqual(len(pyramid), num_levels)
            
            for i, level in enumerate(pyramid):
                print(f"  Level {i}: shape={level.shape}, chunks={level.chunksize}")
                self.assertEqual(level.ndim, 3)  # ZYX - 3D
                
                # Each level should be roughly half the size of the previous
                if i > 0:
                    prev_level = pyramid[i-1]
                    for dim in [0, 1, 2]:  # ZYX dimensions
                        ratio = prev_level.shape[dim] / level.shape[dim]
                        self.assertGreater(ratio, 1.5, f"Level {i} should be downsampled")
            
            print(f"✓ Pyramid generated successfully")

    def test_imaris_to_zarr_writer_single_file(self):
        """Test full conversion pipeline: IMS -> Zarr with single file"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        output_path = self.output_dir
        # The writer appends stack_name to output_path, defaulting to {stem}.ome.zarr
        expected_zarr_path = output_path / f"{ims_file.stem}.ome.zarr"
        
        print(f"\nTesting full conversion pipeline:")
        print(f"  Input: {ims_file.name}")
        print(f"  Output directory: {output_path}")
        print(f"  Expected zarr: {expected_zarr_path}")
        
        # Conversion parameters
        n_lvls = 3
        scale_factor = [2, 2, 2]
        
        print(f"  Pyramid levels: {n_lvls}")
        print(f"  Scale factor: {scale_factor}")
        
        # Run conversion
        print(f"  Starting conversion...")
        imaris_to_zarr_writer(
            imaris_path=str(ims_file),
            output_path=str(output_path),
            n_lvls=n_lvls,
            scale_factor=scale_factor,
            voxel_size=None,  # Extract from file
            compressor_kwargs={"cname": "zstd", "clevel": 5, "shuffle": 1},
        )
        
        # Verify output
        self.assertTrue(expected_zarr_path.exists(), f"Output zarr should exist at {expected_zarr_path}")
        
        # Check for pyramid levels
        for i in range(n_lvls):
            level_path = expected_zarr_path / str(i)
            self.assertTrue(level_path.exists(), f"Pyramid level {i} should exist at {level_path}")
            print(f"  ✓ Level {i} created: {level_path}")
        
        # Check metadata
        zattrs_path = expected_zarr_path / ".zattrs"
        self.assertTrue(zattrs_path.exists(), "Zarr attributes should exist")
        print(f"  ✓ Metadata file created: {zattrs_path}")
        
        print(f"✓ Conversion completed successfully")
        
        # Print output size
        import subprocess
        result = subprocess.run(
            ["du", "-sh", str(expected_zarr_path)],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print(f"  Output size: {result.stdout.strip().split()[0]}")

    def test_imaris_to_zarr_writer_custom_voxel_size(self):
        """Test conversion with custom voxel size override"""
        if not self.ims_files:
            self.skipTest("No IMS files found in data directory")
        
        ims_file = self.ims_files[0]
        output_path = self.output_dir
        custom_stack_name = f"{ims_file.stem}_custom_voxel.ome.zarr"
        expected_zarr_path = output_path / custom_stack_name
        
        print(f"\nTesting conversion with custom voxel size:")
        print(f"  Input: {ims_file.name}")
        
        custom_voxel_size = [2.0, 1.0, 1.0]  # ZYX in microns
        print(f"  Custom voxel size: {custom_voxel_size}")
        
        imaris_to_zarr_writer(
            imaris_path=str(ims_file),
            output_path=str(output_path),
            n_lvls=2,
            scale_factor=[2, 2, 2],
            voxel_size=custom_voxel_size,
            stack_name=custom_stack_name,
        )
        
        self.assertTrue(expected_zarr_path.exists())
        print(f"✓ Conversion with custom voxel size completed")

    def test_process_multiple_files(self):
        """Test processing multiple IMS files (if available)"""
        if len(self.ims_files) < 2:
            self.skipTest("Need at least 2 IMS files for this test")
        
        print(f"\nTesting batch conversion of {len(self.ims_files)} files:")
        
        for ims_file in self.ims_files:
            expected_zarr_path = self.output_dir / f"{ims_file.stem}.ome.zarr"
            print(f"  Converting {ims_file.name}...")
            
            imaris_to_zarr_writer(
                imaris_path=str(ims_file),
                output_path=str(self.output_dir),
                n_lvls=2,
                scale_factor=[2, 2, 2],
                voxel_size=None,
            )
            
            self.assertTrue(expected_zarr_path.exists())
            print(f"    ✓ Completed: {expected_zarr_path}")
        
        print(f"✓ All {len(self.ims_files)} files converted successfully")


if __name__ == "__main__":
    # Run with verbose output
    unittest.main(verbosity=2)
