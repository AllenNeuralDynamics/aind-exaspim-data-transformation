"""
Imaris to OME-Zarr conversion.

This module provides functionality to read Imaris (.ims) files and convert them
to OME-Zarr format with multi-resolution pyramids.

Classes
-------
ImarisReader
    Reader for Imaris HDF5-based image files.

Functions
---------
imaris_to_zarr_writer
    Convert an Imaris file to OME-Zarr format.
"""

from __future__ import annotations

import asyncio
import logging
import math
import multiprocessing
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import dask.array as da
import h5py

# noqa: F401 - registers HDF5 compression filters (LZ4, etc.)
import hdf5plugin  # noqa: F401
import numpy as np
import tensorstore as ts
import zarr
from numcodecs import Blosc
from aind_exaspim_data_transformation.utils.io_utils import ImarisReader, MissingDatasetError
from aind_exaspim_data_transformation.compress.omezarr_metadata import (
    write_ome_ngff_metadata,
)

logger = logging.getLogger(__name__) 



# =============================================================================
# Utility Functions for Multiscale Pyramid Generation
# =============================================================================


def compute_downsampled_shape(
    shape: Tuple[int, ...],
    downsample_factor: Tuple[int, ...],
) -> Tuple[int, ...]:
    """
    Compute the shape after downsampling using ceiling division.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Original array shape.
    downsample_factor : Tuple[int, ...]
        Downsample factor for each dimension. Must have same length as shape.

    Returns
    -------
    Tuple[int, ...]
        Downsampled shape where each dimension is ceil(shape[i] / factor[i]).

    Examples
    --------
    >>> compute_downsampled_shape((100, 200, 300), (2, 2, 2))
    (50, 100, 150)
    >>> compute_downsampled_shape((101, 201, 301), (2, 2, 2))
    (51, 101, 151)
    """
    if len(shape) != len(downsample_factor):
        raise ValueError(
            f"Shape ({len(shape)}D) and downsample_factor "
            f"({len(downsample_factor)}D) must have the same dimensions"
        )
    return tuple(math.ceil(s / f) for s, f in zip(shape, downsample_factor))


def _build_kvstore_spec(
    path: str,
    bucket_name: Optional[str] = None,
    aws_region: str = "us-west-2",
    cpu_cnt: Optional[int] = None,
    read_cache_bytes: int = 1 << 30,
) -> Dict[str, Any]:
    """
    Build a TensorStore kvstore specification for local or S3 storage.

    Parameters
    ----------
    path : str
        Path to the data (local path or S3 prefix without bucket).
    bucket_name : Optional[str]
        S3 bucket name. If None, creates a local file kvstore.
    aws_region : str
        AWS region for S3 bucket. Default is "us-west-2".
    cpu_cnt : Optional[int]
        Number of threads for concurrent operations. If None, uses all CPUs.
    read_cache_bytes : int
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    Dict[str, Any]
        TensorStore kvstore specification dictionary.
    """
    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    if bucket_name is not None:
        return {
            "driver": "s3",
            "bucket": bucket_name,
            "path": path,
            "aws_region": aws_region,
            "endpoint": f"https://s3.{aws_region}.amazonaws.com",
            "context": {
                "cache_pool": {"total_bytes_limit": read_cache_bytes},
                "data_copy_concurrency": {"limit": cpu_cnt},
                "s3_request_concurrency": {"limit": cpu_cnt},
            },
        }
    else:
        return {
            "driver": "file",
            "path": path,
        }


def create_scale_spec(
    output_path: str,
    data_shape: Tuple[int, ...],
    data_dtype: str,
    shard_shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    scale: int,
    codec: str = "zstd",
    codec_level: int = 3,
    cpu_cnt: Optional[int] = None,
    aws_region: str = "us-west-2",
    bucket_name: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a TensorStore specification for a pyramid scale level.

    This creates a Zarr v3 spec with sharding for efficient parallel writes
    and reads. The specification is suitable for both local and S3 storage.

    Parameters
    ----------
    output_path : str
        Base path to the zarr store (without scale level suffix).
    data_shape : Tuple[int, ...]
        Shape of this scale level (5D: T, C, Z, Y, X).
    data_dtype : str
        Data type name (e.g., "uint16", "float32").
    shard_shape : Tuple[int, ...]
        Shard shape for the sharding codec. This determines the file
        granularity for parallel writes.
    chunk_shape : Tuple[int, ...]
        Inner chunk shape within shards. This determines the compression
        unit and read granularity.
    scale : int
        Scale level index (0 = full resolution).
    codec : str
        Compression codec name. Default is "zstd".
    codec_level : int
        Compression level. Default is 3.
    cpu_cnt : Optional[int]
        Number of threads for concurrent operations. If None, uses all CPUs.
    aws_region : str
        AWS region for S3 bucket. Default is "us-west-2".
    bucket_name : Optional[str]
        S3 bucket name. If None, writes locally.

    Returns
    -------
    Dict[str, Any]
        TensorStore specification dictionary ready for ts.open().

    Notes
    -----
    The spec uses sharding_indexed codec for efficient parallel writes.
    Each shard is an independent file containing multiple chunks, allowing
    different workers to write different shards without coordination.
    """
    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    # Build the scale-specific path
    scale_path = f"{output_path}/{scale}"

    # Clamp chunk shape to data shape first
    clamped_chunk = tuple(min(c, d) for c, d in zip(chunk_shape, data_shape))

    # Clamp shard shape to data shape, then ensure it's a multiple of chunk
    # This is required by Zarr v3 sharding: shard must be divisible by chunk
    clamped_shard = []
    for s, c, d in zip(shard_shape, clamped_chunk, data_shape):
        # Clamp shard to data dimension
        clamped = min(s, d)
        # Round down to nearest multiple of chunk size
        if c > 0:
            clamped = (clamped // c) * c
            # Ensure at least one chunk
            if clamped < c:
                clamped = c
        clamped_shard.append(clamped)
    clamped_shard = tuple(clamped_shard)

    # Build inner codecs for compression
    inner_codecs = [
        {"name": "transpose", "configuration": {"order": "C"}},
        {"name": codec, "configuration": {"level": codec_level}},
    ]

    # Build sharding codec
    codecs = [
        {
            "name": "sharding_indexed",
            "configuration": {
                "chunk_shape": list(clamped_chunk),
                "codecs": inner_codecs,
                "index_codecs": [
                    {"name": "bytes", "configuration": {"endian": "little"}},
                    {"name": "crc32c"},
                ],
                "index_location": "end",
            },
        }
    ]

    # Build kvstore spec
    kvstore = _build_kvstore_spec(
        path=scale_path,
        bucket_name=bucket_name,
        aws_region=aws_region,
        cpu_cnt=cpu_cnt,
    )

    spec = {
        "driver": "zarr3",
        "kvstore": kvstore,
        "metadata": {
            "shape": list(data_shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": list(clamped_shard)},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": data_dtype,
            "codecs": codecs,
        },
        "create": True,
        "delete_existing": True,
    }

    return spec


async def create_downsample_dataset(
    dataset_path: str,
    start_scale: int,
    downsample_factor: List[int],
    downsample_mode: str,
    shard_shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    codec: str = "zstd",
    codec_level: int = 3,
    cpu_cnt: Optional[int] = None,
    aws_region: str = "us-west-2",
    bucket_name: Optional[str] = None,
    read_cache_bytes: int = 1 << 30,
) -> Tuple[int, ...]:
    """
    Create a new downsampled scale level using TensorStore's downsample driver.

    This function reads from an existing scale level, applies downsampling
    using TensorStore's efficient downsample driver, and writes the result
    to a new scale level in the same dataset.

    Parameters
    ----------
    dataset_path : str
        Base path of the dataset (local path or S3 prefix without bucket).
        Should contain scale level subdirectories (0, 1, 2, ...).
    start_scale : int
        The scale level to downsample from (e.g., 0 for original resolution).
    downsample_factor : List[int]
        Downsampling factor for each dimension [t, c, z, y, x].
        For spatial-only downsampling, use [1, 1, 2, 2, 2].
    downsample_mode : str
        Downsampling method. Options are:
        - "stride": Take every Nth sample (fastest, may alias)
        - "median": Median of the downsampling window
        - "mode": Most common value (good for label data)
        - "mean": Average of the window (good for intensity data)
        - "min": Minimum value in window
        - "max": Maximum value in window
    shard_shape : Tuple[int, ...]
        Shard shape for the output Zarr v3 sharding (5D).
    chunk_shape : Tuple[int, ...]
        Inner chunk shape within shards (5D).
    codec : str
        Compression codec for the output. Default is "zstd".
    codec_level : int
        Compression level. Default is 3.
    cpu_cnt : int, optional
        Number of threads for concurrent operations. If None, uses all CPUs.
    aws_region : str
        AWS region for S3 bucket. Default is "us-west-2".
    bucket_name : str, optional
        S3 bucket name. If None, operates on local filesystem.
    read_cache_bytes : int
        Size of the read cache pool in bytes. Default is 1GB.

    Returns
    -------
    Tuple[int, ...]
        Shape of the newly created downsampled dataset.

    Notes
    -----
    TensorStore's downsample driver performs efficient streaming downsampling
    without loading the entire source array into memory. This is critical for
    handling 120TB datasets.

    The new scale level is written to {dataset_path}/{start_scale + 1}.
    """
    if cpu_cnt is None:
        cpu_cnt = multiprocessing.cpu_count()

    # Ensure downsample factor is 5D
    if len(downsample_factor) < 5:
        downsample_factor = [1] * (5 - len(downsample_factor)) + list(
            downsample_factor
        )

    # Build kvstore for the source dataset
    source_kvstore = _build_kvstore_spec(
        path=dataset_path,
        bucket_name=bucket_name,
        aws_region=aws_region,
        cpu_cnt=cpu_cnt,
        read_cache_bytes=read_cache_bytes,
    )

    # Create the spec with downsample driver wrapping the source
    source_with_downsample_spec = {
        "driver": "downsample",
        "downsample_factors": downsample_factor,
        "downsample_method": downsample_mode,
        "base": {
            "driver": "zarr3",
            "kvstore": source_kvstore,
            "path": str(start_scale),
            "recheck_cached_metadata": False,
            "recheck_cached_data": False,
        },
    }

    # Open the downsampled view of the source
    downsampled_view = await ts.open(spec=source_with_downsample_spec)
    source_dataset = downsampled_view.base

    # Get properties for the new scale
    new_scale = start_scale + 1
    downsampled_shape = tuple(downsampled_view.shape)
    source_dtype = source_dataset.dtype.name

    logger.info(
        f"Creating scale {new_scale}: "
        f"shape {downsampled_shape} from scale {start_scale}"
    )

    # Create spec for the destination dataset
    dest_spec = create_scale_spec(
        output_path=dataset_path,
        data_shape=downsampled_shape,
        data_dtype=source_dtype,
        shard_shape=shard_shape,
        chunk_shape=chunk_shape,
        scale=new_scale,
        codec=codec,
        codec_level=codec_level,
        cpu_cnt=cpu_cnt,
        aws_region=aws_region,
        bucket_name=bucket_name,
    )

    # Open destination for writing
    dest_dataset = await ts.open(dest_spec)

    # Read the downsampled data and write to destination
    # TensorStore handles this efficiently with streaming
    downsampled_data = await downsampled_view.read()
    await dest_dataset.write(downsampled_data)

    logger.info(f"Completed writing scale {new_scale}")

    return downsampled_shape


def create_downsample_levels(
    dataset_path: str,
    base_shape: Tuple[int, ...],
    n_levels: int,
    downsample_factor: Tuple[int, int, int],
    downsample_mode: str,
    shard_shape: Tuple[int, ...],
    chunk_shape: Tuple[int, ...],
    codec: str = "zstd",
    codec_level: int = 3,
    cpu_cnt: Optional[int] = None,
    aws_region: str = "us-west-2",
    bucket_name: Optional[str] = None,
) -> List[Tuple[int, ...]]:
    """
    Generate all downsampled pyramid levels for a dataset.

    This function orchestrates the creation of a multi-resolution pyramid
    by sequentially generating each level from the previous one. Level 0
    (base resolution) must already exist in the dataset.

    Parameters
    ----------
    dataset_path : str
        Base path to the zarr store containing level 0.
    base_shape : Tuple[int, ...]
        Shape of the base resolution level (5D: T, C, Z, Y, X).
    n_levels : int
        Total number of pyramid levels to create (including base).
        For example, n_levels=5 creates levels 0, 1, 2, 3, 4.
    downsample_factor : Tuple[int, int, int]
        Downsample factors for Z, Y, X dimensions (spatial only).
        Typically (2, 2, 2) for isotropic downsampling.
    downsample_mode : str
        Downsampling method: "stride", "median", "mode", "mean", "min", "max".
    shard_shape : Tuple[int, ...]
        Shard shape for output Zarr v3 sharding (5D: T, C, Z, Y, X).
    chunk_shape : Tuple[int, ...]
        Inner chunk shape within shards (5D).
    codec : str
        Compression codec for output arrays. Default is "zstd".
    codec_level : int
        Compression level. Default is 3.
    cpu_cnt : Optional[int]
        Number of threads for concurrent operations. If None, uses all CPUs.
    aws_region : str
        AWS region for S3 bucket. Default is "us-west-2".
    bucket_name : Optional[str]
        S3 bucket name. If None, operates on local filesystem.

    Returns
    -------
    List[Tuple[int, ...]]
        List of shapes for all levels created (including base).

    Examples
    --------
    >>> shapes = create_downsample_levels(
    ...     dataset_path="s3://bucket/dataset.zarr",
    ...     base_shape=(1, 1, 1000, 2000, 3000),
    ...     n_levels=4,
    ...     downsample_factor=(2, 2, 2),
    ...     downsample_mode="mean",
    ...     shard_shape=(1, 1, 256, 256, 256),
    ...     chunk_shape=(1, 1, 64, 64, 64),
    ...     bucket_name="bucket",
    ... )
    >>> # shapes will be:
    >>> # [(1, 1, 1000, 2000, 3000),  # level 0 (base)
    >>> #  (1, 1, 500, 1000, 1500),   # level 1
    >>> #  (1, 1, 250, 500, 750),     # level 2
    >>> #  (1, 1, 125, 250, 375)]     # level 3

    Notes
    -----
    - Level 0 must already exist before calling this function.
    - Levels are created sequentially because each level depends on
      the previous one.
    - The function uses asyncio to run the async downsample operations.
    """
    if n_levels < 2:
        logger.info("n_levels < 2, no downsampling needed")
        return [base_shape]

    # Build 5D downsample factor (T, C, Z, Y, X)
    # Only downsample spatial dimensions
    downsample_factor_5d = [1, 1] + list(downsample_factor)

    # Track shapes for all levels
    shapes: List[Tuple[int, ...]] = [base_shape]

    logger.info(
        f"Creating {n_levels - 1} downsampled levels with factor "
        f"{downsample_factor} using '{downsample_mode}' method"
    )

    async def _generate_all_levels():
        """Async helper to generate all levels sequentially."""
        for level_idx in range(n_levels - 1):
            start_scale = level_idx
            new_shape = await create_downsample_dataset(
                dataset_path=dataset_path,
                start_scale=start_scale,
                downsample_factor=downsample_factor_5d,
                downsample_mode=downsample_mode,
                shard_shape=shard_shape,
                chunk_shape=chunk_shape,
                codec=codec,
                codec_level=codec_level,
                cpu_cnt=cpu_cnt,
                aws_region=aws_region,
                bucket_name=bucket_name,
            )
            shapes.append(new_shape)

    # Run the async generation
    asyncio.run(_generate_all_levels())

    logger.info(f"Completed creating {n_levels} pyramid levels")
    for i, shape in enumerate(shapes):
        logger.info(f"  Level {i}: {shape}")

    return shapes


# =============================================================================
# Writer function
# =============================================================================


def imaris_to_zarr_writer(
    imaris_path: str,
    output_path: str,
    voxel_size: Optional[List[float]] = None,
    chunk_size: Optional[List[int]] = None,
    scale_factor: Optional[List[int]] = None,
    n_lvls: int = 5,
    channel_name: Optional[str] = None,
    stack_name: Optional[str] = None,
    compressor_kwargs: Optional[Dict] = None,
    bucket_name: Optional[str] = None,
) -> str:
    """
    Convert an Imaris file to OME-Zarr format.

    Reads an Imaris (.ims) file and writes it as an OME-NGFF compliant
    Zarr store with multi-resolution pyramids.

    Parameters
    ----------
    imaris_path : str
        Path to the input Imaris file (.ims or .h5)
    output_path : str
        Directory where the output OME-Zarr will be written
    voxel_size : Optional[List[float]]
        Voxel size in [Z, Y, X] order in micrometers.
        If None, extracted from Imaris metadata.
    chunk_size : Optional[List[int]]
        Chunk size for zarr arrays in [Z, Y, X] order.
        Default is [128, 128, 128].
    scale_factor : Optional[List[int]]
        Downsampling scale factors in [Z, Y, X] order.
        Default is [2, 2, 2].
    n_lvls : int
        Number of pyramid levels to generate. Default is 5.
    channel_name : Optional[str]
        Name for the channel in metadata. If None, uses stack name.
    stack_name : Optional[str]
        Name for the output zarr store (e.g., "image.ome.zarr").
        If None, derives from input filename.
    compressor_kwargs : Optional[Dict]
        Compression settings for Blosc compressor.
        Default is {"cname": "zstd", "clevel": 3, "shuffle": Blosc.SHUFFLE}
    bucket_name : Optional[str]
        S3 bucket name for cloud storage. If None, writes locally.

    Returns
    -------
    str
        Path to the created OME-Zarr store

    Examples
    --------
    >>> imaris_to_zarr_writer(
    ...     imaris_path="input.ims",
    ...     output_path="/data/output",
    ...     n_lvls=3,
    ...     voxel_size=[2.0, 0.5, 0.5],
    ... )
    '/data/output/input.ome.zarr'
    """
    logger.info(f"Starting Imaris to Zarr conversion: {imaris_path}")

    # Set defaults
    if chunk_size is None:
        chunk_size = [128, 128, 128]
    if scale_factor is None:
        scale_factor = [2, 2, 2]
    if compressor_kwargs is None:
        compressor_kwargs = {
            "cname": "zstd",
            "clevel": 3,
            "shuffle": Blosc.SHUFFLE,
        }
    if stack_name is None:
        stack_name = Path(imaris_path).stem + ".ome.zarr"

    compressor = Blosc(**compressor_kwargs)

    with ImarisReader(imaris_path) as reader:
        # Get voxel size from file if not provided
        if voxel_size is None:
            voxel_size, unit = reader.get_voxel_size()
            logger.info(
                f"Extracted voxel size from Imaris: {voxel_size} {unit}"
            )
        else:
            logger.info(f"Using provided voxel size: {voxel_size}")

        # Determine actual number of levels available
        available_levels = reader.n_levels
        actual_n_lvls = min(n_lvls, available_levels)
        if actual_n_lvls < n_lvls:
            logger.warning(
                f"Requested {n_lvls} levels but only {available_levels} "
                f"available in Imaris file. Using {actual_n_lvls}."
            )

        # Get pyramid as dask arrays using native HDF5 chunks
        native_chunks = reader.get_chunks()
        logger.info(f"Using native HDF5 chunks: {native_chunks}")

        pyramid = reader.get_dask_pyramid(
            num_levels=actual_n_lvls,
            timepoint=0,
            channel=0,
            chunks=native_chunks,
        )

        shape = pyramid[0].shape
        logger.info(f"Image shape (Z, Y, X): {shape}")

        # Prepare output path
        output_zarr_path = Path(output_path) / stack_name

        if bucket_name:
            store_path = f"s3://{bucket_name}/{output_zarr_path}"
            logger.info(f"Writing to S3: {store_path}")
        else:
            store_path = str(output_zarr_path)
            output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing to local: {store_path}")

        # Create root zarr group
        root = zarr.open_group(store_path, mode="w")

        # Write each pyramid level
        for level_idx, dask_array in enumerate(pyramid):
            logger.info(
                f"Writing level {level_idx}: shape={dask_array.shape}, "
                f"chunks={dask_array.chunksize}"
            )

            # Create zarr array for this level
            z_array = root.create_dataset(
                str(level_idx),
                shape=dask_array.shape,
                chunks=dask_array.chunksize,
                dtype=dask_array.dtype,
                compressor=compressor,
                overwrite=True,
            )

            # Write dask array to zarr
            da.to_zarr(dask_array, z_array, overwrite=True, compute=True)

        # Generate and write OME-NGFF metadata
        # Shape in 5D format: (T, C, Z, Y, X)
        shape_5d = (1, 1) + tuple(shape)
        chunk_size_5d = (1, 1) + tuple(native_chunks)

        metadata_dict = write_ome_ngff_metadata(
            arr_shape=list(shape_5d),
            chunk_size=list(chunk_size_5d),
            image_name=channel_name or stack_name,
            n_lvls=actual_n_lvls,
            scale_factors=tuple(scale_factor),
            voxel_size=tuple(voxel_size),
            channel_names=[channel_name or stack_name],
        )

        root.attrs.update(metadata_dict)
        logger.info(
            f"Successfully wrote {stack_name} with {actual_n_lvls} levels"
        )

        return store_path


# =============================================================================
# TensorStore-based Parallel Writer (Zarr v3 with Sharding)
# =============================================================================


def create_tensorstore_spec(
    path: str,
    shape: Tuple[int, ...],
    dtype: np.dtype,
    chunk_shape: Tuple[int, ...],
    shard_shape: Optional[Tuple[int, ...]] = None,
    codec: str = "zstd",
    codec_level: int = 3,
    is_s3: bool = False,
) -> Dict[str, Any]:
    """
    Create a TensorStore specification for Zarr v3 with optional sharding.

    Parameters
    ----------
    path : str
        Path or S3 URL for the zarr store
    shape : Tuple[int, ...]
        Array shape (Z, Y, X)
    dtype : np.dtype
        Data type of the array
    chunk_shape : Tuple[int, ...]
        Chunk size for the array (Z, Y, X)
    shard_shape : Optional[Tuple[int, ...]]
        Shard shape for sharding codec. If None, no sharding is used.
    codec : str
        Compression codec ("zstd", "blosc", etc.)
    codec_level : int
        Compression level
    is_s3 : bool
        Whether the path is an S3 URL

    Returns
    -------
    Dict[str, Any]
        TensorStore specification dictionary
    """
    # Build the codec chain
    if shard_shape is not None:
        # With sharding: codecs go inside the sharding codec
        inner_codecs = [
            {"name": "transpose", "configuration": {"order": "C"}},
            {"name": codec, "configuration": {"level": codec_level}},
        ]
        codecs = [
            {
                "name": "sharding_indexed",
                "configuration": {
                    "chunk_shape": list(chunk_shape),
                    "codecs": inner_codecs,
                    "index_codecs": [
                        {
                            "name": "bytes",
                            "configuration": {"endian": "little"},
                        },
                        {"name": "crc32c"},
                    ],
                    "index_location": "end",
                },
            }
        ]
        effective_chunk_shape = list(shard_shape)
    else:
        # No sharding: direct codecs
        codecs = [
            {"name": "transpose", "configuration": {"order": "C"}},
            {"name": codec, "configuration": {"level": codec_level}},
        ]
        effective_chunk_shape = list(chunk_shape)

    # Build kvstore spec based on storage type
    if is_s3:
        kvstore = _s3_kvstore(path)
    else:
        kvstore = {"driver": "file", "path": path}

    spec = {
        "driver": "zarr3",
        "kvstore": kvstore,
        "metadata": {
            "shape": list(shape),
            "chunk_grid": {
                "name": "regular",
                "configuration": {"chunk_shape": effective_chunk_shape},
            },
            "chunk_key_encoding": {"name": "default"},
            "data_type": str(dtype),
            "codecs": codecs,
        },
        "create": True,
        "delete_existing": True,
    }

    return spec


def _s3_kvstore(s3_url: str, region: str = "us-west-2") -> Dict[str, Any]:
    """
    Create a TensorStore S3 kvstore configuration from an S3 URL.

    Parameters
    ----------
    s3_url : str
        S3 URL in format "s3://bucket/path"
    region : str
        AWS region for the S3 bucket. Default is "us-west-2".

    Returns
    -------
    Dict[str, Any]
        TensorStore kvstore specification for S3
    """
    if not s3_url.startswith("s3://"):
        raise ValueError(f"Invalid S3 URL: {s3_url}")

    # Parse s3://bucket/path
    parts = s3_url[5:].split("/", 1)
    bucket = parts[0]
    path = parts[1] if len(parts) > 1 else ""

    return {
        "driver": "s3",
        "bucket": bucket,
        "path": path,
        "aws_region": region,
        "endpoint": f"https://s3.{region}.amazonaws.com",
        "aws_credentials": {"type": "default"},
    }


def iter_block_aligned_slices(
    shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
) -> Iterator[Tuple[slice, ...]]:
    """
    Iterate over block-aligned slices covering the entire array.

    This yields slices that are aligned to block boundaries, which is
    essential for efficient parallel writes with sharding.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the full array
    block_shape : Tuple[int, ...]
        Shape of each block (should match shard shape for optimal performance)

    Yields
    ------
    Tuple[slice, ...]
        Tuple of slices for each dimension defining the block region
    """
    # Calculate number of blocks in each dimension
    n_blocks = tuple(math.ceil(s / b) for s, b in zip(shape, block_shape))

    # Iterate over all blocks in Z, Y, X order
    for z_idx in range(n_blocks[0]):
        for y_idx in range(n_blocks[1]):
            for x_idx in range(n_blocks[2]):
                z_start = z_idx * block_shape[0]
                y_start = y_idx * block_shape[1]
                x_start = x_idx * block_shape[2]

                z_end = min(z_start + block_shape[0], shape[0])
                y_end = min(y_start + block_shape[1], shape[1])
                x_end = min(x_start + block_shape[2], shape[2])

                yield (
                    slice(z_start, z_end),
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                )


def write_block_to_tensorstore(
    store: ts.TensorStore,
    data: np.ndarray,
    slices: Tuple[slice, ...],
) -> ts.Future:
    """
    Write a single block to a TensorStore array.

    Parameters
    ----------
    store : ts.TensorStore
        Open TensorStore array
    data : np.ndarray
        Block data to write
    slices : Tuple[slice, ...]
        Target slices in the store

    Returns
    -------
    ts.Future
        Future that completes when the write is done
    """
    return store[slices].write(data)


def imaris_to_zarr_parallel(
    imaris_path: str,
    output_path: str,
    voxel_size: Optional[List[float]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    shard_shape: Optional[Tuple[int, int, int]] = None,
    n_lvls: int = 1,
    scale_factor: Tuple[int, int, int] = (2, 2, 2),
    downsample_mode: str = "mean",
    channel_name: Optional[str] = None,
    stack_name: Optional[str] = None,
    codec: str = "zstd",
    codec_level: int = 3,
    bucket_name: Optional[str] = None,
    max_concurrent_writes: int = 16,
) -> str:
    """
    Convert an Imaris file to OME-Zarr v3 format using TensorStore.

    This function uses TensorStore for parallel, shard-aligned writes which
    enables horizontal scaling across multiple workers in a distributed
    environment (e.g., SLURM cluster with Dask).

    The output is a 5D OME-NGFF compliant Zarr store with shape (T,C,Z,Y,X).
    Multiscale pyramid levels are generated using TensorStore's downsample
    driver for efficient streaming downsampling of large datasets.

    Parameters
    ----------
    imaris_path : str
        Path to the input Imaris file (.ims or .h5)
    output_path : str
        Directory where the output OME-Zarr will be written
    voxel_size : Optional[List[float]]
        Voxel size in [Z, Y, X] order in micrometers.
        If None, extracted from Imaris metadata.
    chunk_shape : Optional[Tuple[int, int, int]]
        Chunk size for zarr arrays in (Z, Y, X) order.
        Default is (128, 128, 128). Will be converted to 5D internally.
    shard_shape : Optional[Tuple[int, int, int]]
        Shard size for Zarr v3 sharding in (Z, Y, X) order.
        Default is (256, 256, 256). Will be converted to 5D internally.
    n_lvls : int
        Number of pyramid levels to generate. Default is 1 (base level only).
        For example, n_lvls=5 creates levels 0, 1, 2, 3, 4 with each level
        downsampled by scale_factor from the previous.
    scale_factor : Tuple[int, int, int]
        Downsample factors for (Z, Y, X) dimensions. Default is (2, 2, 2).
        Applied to generate each successive pyramid level.
    downsample_mode : str
        Downsampling method for pyramid generation. Default is "mean".
        Options: "stride", "median", "mode", "mean", "min", "max".
    channel_name : Optional[str]
        Name for the channel in metadata. If None, uses stack name.
    stack_name : Optional[str]
        Name for the output zarr store (e.g., "image.ome.zarr").
        If None, derives from input filename.
    codec : str
        Compression codec. Default is "zstd".
    codec_level : int
        Compression level. Default is 3.
    bucket_name : Optional[str]
        S3 bucket name for cloud storage. If None, writes locally.
    max_concurrent_writes : int
        Maximum number of concurrent block writes. Default is 16.

    Returns
    -------
    str
        Path to the created OME-Zarr store

    Notes
    -----
    This writer is designed for horizontal scaling:
    - Block processing is shard-aligned for efficient parallel writes
    - Each worker can write different shards without coordination
    - TensorStore handles efficient S3 writes with proper chunking
    - Pyramid levels are generated using TensorStore's downsample driver
      which streams data efficiently without loading entire arrays

    For distributed execution, call this function from Dask workers
    where each worker processes a subset of blocks.

    Examples
    --------
    >>> imaris_to_zarr_parallel(
    ...     imaris_path="input.ims",
    ...     output_path="/data/output",
    ...     chunk_shape=(128, 128, 128),
    ...     shard_shape=(256, 256, 256),
    ...     n_lvls=5,
    ...     scale_factor=(2, 2, 2),
    ...     downsample_mode="mean",
    ... )
    '/data/output/input.ome.zarr'
    """
    logger.info(f"Starting parallel Imaris to Zarr conversion: {imaris_path}")

    # Set defaults
    if chunk_shape is None:
        chunk_shape = (128, 128, 128)
    if shard_shape is None:
        shard_shape = (256, 256, 256)
    if stack_name is None:
        stack_name = Path(imaris_path).stem + ".ome.zarr"

    # Convert 3D shapes to 5D (T, C, Z, Y, X)
    chunk_shape_5d = (1, 1) + tuple(chunk_shape)
    shard_shape_5d = (1, 1) + tuple(shard_shape)

    def _data_path(level: int) -> str:
        """Build the HDF5 data path for a resolution level."""
        return f"/DataSet/ResolutionLevel {level}/TimePoint 0/Channel 0/Data"

    with ImarisReader(imaris_path) as reader:
        # Get voxel size from file if not provided
        if voxel_size is None:
            voxel_size, unit = reader.get_voxel_size()
            logger.info(
                f"Extracted voxel size from Imaris: {voxel_size} {unit}"
            )
        else:
            logger.info(f"Using provided voxel size: {voxel_size}")

        # Get shape and dtype from base resolution
        base_path = _data_path(0)
        shape_3d = reader.get_shape(base_path)
        dtype = reader.get_dtype(base_path)
        native_chunks = reader.get_chunks(base_path)

        # Create 5D shape (T, C, Z, Y, X)
        shape_5d = (1, 1) + tuple(shape_3d)

        logger.info(f"Image shape (Z, Y, X): {shape_3d}")
        logger.info(f"Image shape 5D (T, C, Z, Y, X): {shape_5d}")
        logger.info(f"Native HDF5 chunks: {native_chunks}")
        logger.info(f"Output chunk shape (5D): {chunk_shape_5d}")
        logger.info(f"Shard shape (5D): {shard_shape_5d}")
        logger.info(
            f"Pyramid: {n_lvls} levels, factor={scale_factor}, "
            f"mode={downsample_mode}"
        )

        # Prepare output path
        output_zarr_path = Path(output_path) / stack_name
        is_s3 = bucket_name is not None

        if is_s3:
            # For S3, store_path is used for logging, but dataset_path is
            # the path within the bucket (without s3://bucket/ prefix)
            store_path = f"s3://{bucket_name}/{output_zarr_path}"
            dataset_path = str(output_zarr_path)
            logger.info(f"Writing to S3: {store_path}")
        else:
            store_path = str(output_zarr_path)
            dataset_path = store_path
            output_zarr_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing to local: {store_path}")

        # =====================================================================
        # Write base level (level 0) from Imaris source
        # =====================================================================
        logger.info(f"Writing base level 0: shape={shape_5d}")

        # Create TensorStore spec for base level (5D)
        base_spec = create_scale_spec(
            output_path=dataset_path,
            data_shape=shape_5d,
            data_dtype=str(dtype),
            shard_shape=shard_shape_5d,
            chunk_shape=chunk_shape_5d,
            scale=0,
            codec=codec,
            codec_level=codec_level,
            bucket_name=bucket_name,
        )

        # Open TensorStore for writing
        store = ts.open(base_spec).result()

        # Get dask array for base level (using native chunks for reading)
        dask_array = reader.as_dask_array(
            data_path=base_path,
            chunks=native_chunks,
        )

        # Write blocks aligned to shard boundaries (iterate over Z, Y, X)
        write_shape_3d = shard_shape if shard_shape else chunk_shape
        pending_writes: List[ts.Future] = []

        for block_slices_3d in iter_block_aligned_slices(
            shape_3d, write_shape_3d
        ):
            # Read block from source (this triggers dask computation)
            block_data_3d = dask_array[block_slices_3d].compute()

            # Expand to 5D: (Z, Y, X) -> (1, 1, Z, Y, X)
            block_data_5d = block_data_3d[np.newaxis, np.newaxis, ...]

            # Create 5D slices for writing
            block_slices_5d = (
                slice(0, 1),  # T
                slice(0, 1),  # C
            ) + block_slices_3d

            # Write block to destination
            future = write_block_to_tensorstore(
                store, block_data_5d, block_slices_5d
            )
            pending_writes.append(future)

            # Limit concurrent writes
            if len(pending_writes) >= max_concurrent_writes:
                # Wait for all current writes to complete
                for f in pending_writes:
                    f.result()
                pending_writes.clear()

        # Wait for remaining writes
        for f in pending_writes:
            f.result()

        logger.info("Completed writing base level 0")

    # =========================================================================
    # Generate downsampled pyramid levels (1, 2, 3, ...)
    # =========================================================================
    if n_lvls > 1:
        logger.info(f"Generating {n_lvls - 1} downsampled pyramid levels...")

        create_downsample_levels(
            dataset_path=dataset_path,
            base_shape=shape_5d,
            n_levels=n_lvls,
            downsample_factor=scale_factor,
            downsample_mode=downsample_mode,
            shard_shape=shard_shape_5d,
            chunk_shape=chunk_shape_5d,
            codec=codec,
            codec_level=codec_level,
            bucket_name=bucket_name,
        )

    # =========================================================================
    # Write OME-NGFF metadata
    # =========================================================================
    metadata_dict = write_ome_ngff_metadata(
        arr_shape=list(shape_5d),
        chunk_size=list(chunk_shape_5d),
        image_name=channel_name or stack_name,
        n_lvls=n_lvls,
        scale_factors=scale_factor,
        voxel_size=tuple(voxel_size),
        channel_names=[channel_name or stack_name],
    )

    # Write metadata to zarr.json
    _write_zarr_metadata(store_path, metadata_dict, is_s3)

    logger.info(f"Successfully wrote {stack_name} with {n_lvls} levels")
    return store_path


def _write_zarr_metadata(
    store_path: str,
    metadata_dict: Dict[str, Any],
    is_s3: bool = False,
) -> None:
    """
    Write OME-NGFF metadata to zarr.json.

    Parameters
    ----------
    store_path : str
        Path to the zarr store root
    metadata_dict : Dict[str, Any]
        OME-NGFF metadata dictionary
    is_s3 : bool
        Whether the store is on S3
    """
    import json

    if is_s3:
        # Use TensorStore to write to S3
        spec = {
            "driver": "json",
            "kvstore": _s3_kvstore(f"{store_path}/zarr.json"),
        }
        store = ts.open(spec, create=True, delete_existing=True).result()
        store.write(metadata_dict).result()
    else:
        # Write locally
        metadata_path = Path(store_path) / "zarr.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata_dict, f, indent=2)


def imaris_to_zarr_translate_pyramid(
    imaris_path: str,
    output_path: str,
    voxel_size: Optional[List[float]] = None,
    chunk_shape: Optional[Tuple[int, int, int]] = None,
    shard_shape: Optional[Tuple[int, int, int]] = None,
    n_lvls: Optional[int] = None,
    channel_name: Optional[str] = None,
    stack_name: Optional[str] = None,
    codec: str = "zstd",
    codec_level: int = 3,
    bucket_name: Optional[str] = None,
    max_concurrent_writes: int = 16,
) -> str:
    """
    Convert Imaris file to OME-Zarr v3 by translating existing pyramids.

    This function reads the pre-computed pyramid levels from an Imaris file
    and writes them directly to Zarr format. This is significantly faster
    than re-computing downsampled levels, as Imaris files typically contain
    multiple resolution levels pre-computed by the acquisition software.

    The output is a 5D OME-NGFF compliant Zarr store with shape (T,C,Z,Y,X).

    Parameters
    ----------
    imaris_path : str
        Path to the input Imaris file (.ims or .h5)
    output_path : str
        Directory where the output OME-Zarr will be written
    voxel_size : Optional[List[float]]
        Voxel size in [Z, Y, X] order in micrometers.
        If None, extracted from Imaris metadata.
    chunk_shape : Optional[Tuple[int, int, int]]
        Chunk size for zarr arrays in (Z, Y, X) order.
        Default is (128, 128, 128). Will be converted to 5D internally.
    shard_shape : Optional[Tuple[int, int, int]]
        Shard size for Zarr v3 sharding in (Z, Y, X) order.
        Default is (256, 256, 256). Will be converted to 5D internally.
    n_lvls : Optional[int]
        Number of pyramid levels to write. If None, writes all available
        levels from the Imaris file.
    channel_name : Optional[str]
        Name for the channel in metadata. If None, uses stack name.
    stack_name : Optional[str]
        Name for the output zarr store (e.g., "image.ome.zarr").
        If None, derives from input filename.
    codec : str
        Compression codec. Default is "zstd".
    codec_level : int
        Compression level. Default is 3.
    bucket_name : Optional[str]
        S3 bucket name for cloud storage. If None, writes locally.
    max_concurrent_writes : int
        Maximum number of concurrent block writes. Default is 16.

    Returns
    -------
    str
        Path to the created OME-Zarr store

    Notes
    -----
    This function is preferred over `imaris_to_zarr_parallel` when:
    - The Imaris file already contains pre-computed pyramid levels
    - Speed is important (avoids re-computing downsampled data)
    - You want to preserve the original pyramid structure

    The function automatically detects available resolution levels in the
    Imaris file and translates each one to the corresponding Zarr level.

    Examples
    --------
    >>> imaris_to_zarr_translate_pyramid(
    ...     imaris_path="input.ims",
    ...     output_path="/data/output",
    ...     chunk_shape=(128, 128, 128),
    ...     shard_shape=(512, 512, 512),
    ... )
    '/data/output/input.ome.zarr'
    """
    logger.info(f"Starting Imaris pyramid translation to Zarr: {imaris_path}")

    # Set defaults
    if chunk_shape is None:
        chunk_shape = (128, 128, 128)
    if shard_shape is None:
        shard_shape = (256, 256, 256)
    if stack_name is None:
        stack_name = Path(imaris_path).stem + ".ome.zarr"

    # Convert 3D shapes to 5D (T, C, Z, Y, X)
    chunk_shape_5d = (1, 1) + tuple(chunk_shape)
    shard_shape_5d = (1, 1) + tuple(shard_shape)

    def _data_path(level: int) -> str:
        """Build the HDF5 data path for a resolution level."""
        return f"/DataSet/ResolutionLevel {level}/TimePoint 0/Channel 0/Data"

    def _count_imaris_levels(reader: ImarisReader) -> int:
        """Count the number of resolution levels in the Imaris file."""
        count = 0
        with h5py.File(reader.filepath, "r") as f:
            dataset = f["DataSet"]
            for key in dataset.keys():
                if "ResolutionLevel" in key:
                    count += 1
        return count

    with ImarisReader(imaris_path) as reader:
        # Get voxel size from file if not provided
        if voxel_size is None:
            voxel_size, unit = reader.get_voxel_size()
            logger.info(
                f"Extracted voxel size from Imaris: {voxel_size} {unit}"
            )
        else:
            logger.info(f"Using provided voxel size: {voxel_size}")

        # Count available pyramid levels in Imaris file
        available_levels = _count_imaris_levels(reader)
        logger.info(
            f"Imaris file contains {available_levels} resolution levels"
        )

        # Determine how many levels to write
        if n_lvls is None:
            n_lvls = available_levels
        else:
            n_lvls = min(n_lvls, available_levels)

        logger.info(f"Will translate {n_lvls} pyramid levels")

        # Get shape and dtype from base resolution
        base_path = _data_path(0)
        shape_3d = reader.get_shape(base_path)
        dtype = reader.get_dtype(base_path)
        native_chunks = reader.get_chunks(base_path)

        # Create 5D shape (T, C, Z, Y, X)
        shape_5d = (1, 1) + tuple(shape_3d)

        logger.info(f"Base image shape (Z, Y, X): {shape_3d}")
        logger.info(f"Base image shape 5D (T, C, Z, Y, X): {shape_5d}")
        logger.info(f"Native HDF5 chunks: {native_chunks}")
        logger.info(f"Output chunk shape (5D): {chunk_shape_5d}")
        logger.info(f"Shard shape (5D): {shard_shape_5d}")

        # Prepare output path
        output_zarr_path = Path(output_path) / stack_name
        is_s3 = bucket_name is not None

        if is_s3:
            store_path = f"s3://{bucket_name}/{output_zarr_path}"
            dataset_path = str(output_zarr_path)
            logger.info(f"Writing to S3: {store_path}")
        else:
            store_path = str(output_zarr_path)
            dataset_path = store_path
            output_zarr_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing to local: {store_path}")

        # Track shapes for each level (needed for metadata)
        level_shapes = []

        # =====================================================================
        # Translate each Imaris resolution level to Zarr
        # =====================================================================
        for level in range(n_lvls):
            imaris_data_path = _data_path(level)

            # Get shape for this level
            level_shape_3d = reader.get_shape(imaris_data_path)
            level_shape_5d = (1, 1) + tuple(level_shape_3d)
            level_shapes.append(level_shape_5d)
            level_chunks = reader.get_chunks(imaris_data_path)

            logger.info(
                f"Translating level {level}: "
                f"shape={level_shape_5d}, native_chunks={level_chunks}"
            )

            # Create TensorStore spec for this level
            level_spec = create_scale_spec(
                output_path=dataset_path,
                data_shape=level_shape_5d,
                data_dtype=str(dtype),
                shard_shape=shard_shape_5d,
                chunk_shape=chunk_shape_5d,
                scale=level,
                codec=codec,
                codec_level=codec_level,
                bucket_name=bucket_name,
            )

            # Open TensorStore for writing
            store = ts.open(level_spec).result()

            # Get dask array for this level (using native chunks for reading)
            dask_array = reader.as_dask_array(
                data_path=imaris_data_path,
                chunks=level_chunks,
            )

            # Write blocks aligned to shard boundaries
            write_shape_3d = shard_shape if shard_shape else chunk_shape
            pending_writes: List[ts.Future] = []

            for block_slices_3d in iter_block_aligned_slices(
                level_shape_3d, write_shape_3d
            ):
                # Read block from source (this triggers dask computation)
                block_data_3d = dask_array[block_slices_3d].compute()

                # Expand to 5D: (Z, Y, X) -> (1, 1, Z, Y, X)
                block_data_5d = block_data_3d[np.newaxis, np.newaxis, ...]

                # Create 5D slices for writing
                block_slices_5d = (
                    slice(0, 1),  # T
                    slice(0, 1),  # C
                ) + block_slices_3d

                # Write block to destination
                future = write_block_to_tensorstore(
                    store, block_data_5d, block_slices_5d
                )
                pending_writes.append(future)

                # Limit concurrent writes
                if len(pending_writes) >= max_concurrent_writes:
                    for f in pending_writes:
                        f.result()
                    pending_writes.clear()

            # Wait for remaining writes
            for f in pending_writes:
                f.result()

            logger.info(f"Completed level {level}")

    # =========================================================================
    # Compute scale factors from actual level shapes
    # =========================================================================
    # Compute the scale factors between consecutive levels
    scale_factors_per_level = []
    for i in range(1, len(level_shapes)):
        prev_shape = level_shapes[i - 1]
        curr_shape = level_shapes[i]
        # Calculate factors for Z, Y, X (indices 2, 3, 4)
        factors = tuple(
            round(prev_shape[j] / curr_shape[j]) if curr_shape[j] > 0 else 1
            for j in range(2, 5)
        )
        scale_factors_per_level.append(factors)

    # Use the first scale factor as representative (they should all be similar)
    if scale_factors_per_level:
        representative_factor = scale_factors_per_level[0]
    else:
        representative_factor = (2, 2, 2)

    logger.info(f"Detected scale factors: {scale_factors_per_level}")
    logger.info(f"Using representative factor: {representative_factor}")

    # =========================================================================
    # Write OME-NGFF metadata
    # =========================================================================
    metadata_dict = write_ome_ngff_metadata(
        arr_shape=list(shape_5d),
        chunk_size=list(chunk_shape_5d),
        image_name=channel_name or stack_name,
        n_lvls=n_lvls,
        scale_factors=representative_factor,
        voxel_size=tuple(voxel_size),
        channel_names=[channel_name or stack_name],
    )

    # Write metadata to zarr.json
    _write_zarr_metadata(store_path, metadata_dict, is_s3)

    logger.info(
        f"Successfully translated {stack_name} with {n_lvls} levels "
        f"from Imaris pyramid"
    )
    return store_path
