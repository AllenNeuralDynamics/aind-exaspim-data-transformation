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

import logging
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import dask.array as da
import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters (LZ4, etc.)
import numpy as np
import tensorstore as ts
import zarr
from numcodecs import Blosc

from aind_exaspim_data_transformation.compress.omezarr_metadata import (
    write_ome_ngff_metadata,
)

logger = logging.getLogger(__name__)


class MissingDatasetError(FileNotFoundError):
    """Raised when an expected dataset path is absent in the HDF5 file."""


class ImarisReader:
    """
    Reader for Imaris HDF5-based image files (.ims, .h5).

    Imaris files store 3D image stacks in HDF5 format with pre-computed
    multi-resolution pyramids. This reader provides efficient access to
    the image data as dask arrays for lazy, chunked processing.

    Parameters
    ----------
    filepath : str
        Path to the Imaris file (.ims or .h5)

    Examples
    --------
    >>> with ImarisReader("image.ims") as reader:
    ...     shape = reader.get_shape()
    ...     voxel_size, unit = reader.get_voxel_size()
    ...     dask_array = reader.as_dask_array()

    Notes
    -----
    Imaris files use LZ4 compression which requires the hdf5plugin library
    to be imported for the HDF5 filter to be registered.
    """

    DEFAULT_DATA_PATH = "/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._handle: Optional[h5py.File] = h5py.File(filepath, mode="r")

    def __enter__(self) -> "ImarisReader":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def __repr__(self) -> str:
        status = "open" if self._handle is not None else "closed"
        return f"ImarisReader({self.filepath!r}, {status})"

    # -------------------------------------------------------------------------
    # Core data access methods
    # -------------------------------------------------------------------------

    def as_dask_array(
        self,
        data_path: str = DEFAULT_DATA_PATH,
        chunks: Any = "native",
    ) -> da.Array:
        """
        Return the image stack as a Dask Array.

        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file.
            Default is the full resolution, first timepoint, first channel.
        chunks : Any
            Chunk specification for dask array:
            - "native" (default): Use HDF5 native chunks for optimal I/O
            - True: Let dask choose chunks automatically
            - None: No chunking (single chunk)
            - tuple: Explicit chunk sizes (Z, Y, X)

        Returns
        -------
        da.Array
            Dask array with shape (Z, Y, X)

        Notes
        -----
        Uses lock=True to ensure thread-safe access to the HDF5 file,
        enabling parallel reads with the threaded scheduler.
        """
        dataset = self._get_dataset(data_path)

        # Resolve chunk specification
        if chunks == "native":
            chunks = dataset.chunks

        return da.from_array(dataset, chunks=chunks, lock=True)

    def as_array(self, data_path: str = DEFAULT_DATA_PATH) -> np.ndarray:
        """
        Return the image stack as a NumPy array.

        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file

        Returns
        -------
        np.ndarray
            NumPy array with shape (Z, Y, X)

        Warning
        -------
        This loads the entire dataset into memory. For large files,
        use as_dask_array() instead.
        """
        return self._get_dataset(data_path)[:]

    def get_dask_pyramid(
        self,
        num_levels: int,
        timepoint: int = 0,
        channel: int = 0,
        chunks: Any = "native",
    ) -> List[da.Array]:
        """
        Get multiple resolution levels as dask arrays.

        Parameters
        ----------
        num_levels : int
            Number of pyramid levels to retrieve
        timepoint : int
            Timepoint index (default 0)
        channel : int
            Channel index (default 0)
        chunks : Any
            Chunk specification:
            - "native" (default): Use HDF5 native chunks for optimal I/O
            - True: Let dask choose chunks automatically
            - None: No chunking
            - tuple: Explicit chunk sizes

        Returns
        -------
        List[da.Array]
            List of dask arrays, one per pyramid level (highest resolution first)

        Raises
        ------
        MissingDatasetError
            If a requested resolution level doesn't exist
        """
        pyramid: List[da.Array] = []

        for lvl in range(num_levels):
            data_path = (
                f"/DataSet/ResolutionLevel {lvl}/"
                f"TimePoint {timepoint}/Channel {channel}/Data"
            )

            if data_path not in self._require_handle():
                raise MissingDatasetError(
                    f"Resolution level {lvl} not found: {data_path}"
                )

            # Determine chunks for this level
            if chunks == "native":
                lvl_chunks = self.get_chunks(data_path)
            elif isinstance(chunks, tuple):
                # Clamp explicit chunks to level shape
                lvl_shape = self.get_shape(data_path)
                lvl_chunks = tuple(min(c, s) for c, s in zip(chunks, lvl_shape))
            else:
                lvl_chunks = chunks

            pyramid.append(self.as_dask_array(data_path, chunks=lvl_chunks))

        return pyramid

    # -------------------------------------------------------------------------
    # Metadata access methods
    # -------------------------------------------------------------------------

    def get_shape(self, data_path: str = DEFAULT_DATA_PATH) -> Tuple[int, ...]:
        """
        Return the HDF5 dataset shape.

        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file

        Returns
        -------
        tuple
            Shape as (Z, Y, X)

        Note
        ----
        This returns the actual HDF5 dataset shape, which may include
        padding added by Imaris. Use get_metadata_shape() for the
        "true" image dimensions from metadata.
        """
        return self._get_dataset(data_path).shape

    def get_metadata_shape(self) -> Tuple[int, int, int]:
        """
        Return the image shape from Imaris metadata.

        Returns
        -------
        tuple
            Shape as (Z, Y, X) from metadata (excludes any padding)
        """
        info = self._get_dataset_info()
        return (
            int(info.attrs["Z"].tobytes()),
            int(info.attrs["Y"].tobytes()),
            int(info.attrs["X"].tobytes()),
        )

    def get_chunks(self, data_path: str = DEFAULT_DATA_PATH) -> Tuple[int, ...]:
        """
        Return the HDF5 native chunk shape.

        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file

        Returns
        -------
        tuple
            Chunk shape as (Z, Y, X)
        """
        return self._get_dataset(data_path).chunks

    def get_dtype(self, data_path: str = DEFAULT_DATA_PATH) -> np.dtype:
        """
        Return the dataset data type.

        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file

        Returns
        -------
        np.dtype
            NumPy dtype of the dataset
        """
        return self._get_dataset(data_path).dtype

    def get_voxel_size(self) -> Tuple[List[float], bytes]:
        """
        Get voxel size from Imaris metadata.

        Returns
        -------
        voxel_size : List[float]
            Voxel dimensions as [Z, Y, X] in the file's units
        unit : bytes
            Unit string (e.g., b'um' for micrometers)
        """
        info = self._get_dataset_info()
        origin = self.get_origin()
        extent = self.get_extent()

        x = int(info.attrs["X"].tobytes())
        y = int(info.attrs["Y"].tobytes())
        z = int(info.attrs["Z"].tobytes())
        unit = info.attrs["Unit"].tobytes()

        voxel_size = [
            (extent[0] - origin[0]) / z,  # Z voxel size
            (extent[1] - origin[1]) / y,  # Y voxel size
            (extent[2] - origin[2]) / x,  # X voxel size
        ]
        return voxel_size, unit

    def get_origin(self) -> List[float]:
        """
        Get the image origin (minimum extent) from metadata.

        Returns
        -------
        List[float]
            Origin coordinates as [Z, Y, X]
        """
        info = self._get_dataset_info()
        return [
            float(info.attrs["ExtMin2"].tobytes()),  # Z
            float(info.attrs["ExtMin1"].tobytes()),  # Y
            float(info.attrs["ExtMin0"].tobytes()),  # X
        ]

    def get_extent(self) -> List[float]:
        """
        Get the image extent (maximum bounds) from metadata.

        Returns
        -------
        List[float]
            Extent coordinates as [Z, Y, X]
        """
        info = self._get_dataset_info()
        return [
            float(info.attrs["ExtMax2"].tobytes()),  # Z
            float(info.attrs["ExtMax1"].tobytes()),  # Y
            float(info.attrs["ExtMax0"].tobytes()),  # X
        ]

    @property
    def n_levels(self) -> int:
        """
        Number of pre-computed resolution levels in the file.

        Returns
        -------
        int
            Number of pyramid levels available
        """
        lvl = 0
        while True:
            data_path = (
                f"/DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0/Data"
            )
            if data_path not in self._require_handle():
                return lvl
            lvl += 1

    # -------------------------------------------------------------------------
    # Resource management
    # -------------------------------------------------------------------------

    def close(self) -> None:
        """Close the HDF5 file handle."""
        if self._handle is not None:
            self._handle.close()
            self._handle = None

    # -------------------------------------------------------------------------
    # Private helper methods
    # -------------------------------------------------------------------------

    def _require_handle(self) -> h5py.File:
        """Get the file handle, raising if closed."""
        if self._handle is None:
            raise RuntimeError("Imaris file handle is closed")
        return self._handle

    def _get_dataset(self, data_path: str) -> h5py.Dataset:
        """Get an HDF5 dataset by path."""
        return self._require_handle()[data_path]

    def _get_dataset_info(self) -> h5py.Dataset:
        """Get the DataSetInfo/Image group containing metadata."""
        return self._require_handle()["DataSetInfo/Image"]


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
            logger.info(f"Extracted voxel size from Imaris: {voxel_size} {unit}")
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
        logger.info(f"Successfully wrote {stack_name} with {actual_n_lvls} levels")

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
                        {"name": "bytes", "configuration": {"endian": "little"}},
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

    spec = {
        "driver": "zarr3",
        "kvstore": {"driver": "file", "path": path} if not is_s3 else _s3_kvstore(path),
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


def _s3_kvstore(s3_url: str) -> Dict[str, Any]:
    """
    Create a TensorStore S3 kvstore configuration from an S3 URL.

    Parameters
    ----------
    s3_url : str
        S3 URL in format "s3://bucket/path"

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
    n_blocks = tuple(
        math.ceil(s / b) for s, b in zip(shape, block_shape)
    )

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
        Default is (128, 128, 128).
    shard_shape : Optional[Tuple[int, int, int]]
        Shard size for Zarr v3 sharding in (Z, Y, X) order.
        Default is (256, 256, 256). Set to None to disable sharding.
    n_lvls : int
        Number of pyramid levels to write. Default is 1 (base level only).
        Additional levels can be computed separately.
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

    For distributed execution, call this function from Dask workers
    where each worker processes a subset of blocks.

    Examples
    --------
    >>> imaris_to_zarr_parallel(
    ...     imaris_path="input.ims",
    ...     output_path="/data/output",
    ...     chunk_shape=(128, 128, 128),
    ...     shard_shape=(256, 256, 256),
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

    def _data_path(level: int) -> str:
        """Build the HDF5 data path for a resolution level."""
        return f"/DataSet/ResolutionLevel {level}/TimePoint 0/Channel 0/Data"

    with ImarisReader(imaris_path) as reader:
        # Get voxel size from file if not provided
        if voxel_size is None:
            voxel_size, unit = reader.get_voxel_size()
            logger.info(f"Extracted voxel size from Imaris: {voxel_size} {unit}")
        else:
            logger.info(f"Using provided voxel size: {voxel_size}")

        # Get shape and dtype from base resolution
        base_path = _data_path(0)
        shape = reader.get_shape(base_path)
        dtype = reader.get_dtype(base_path)
        native_chunks = reader.get_chunks(base_path)

        logger.info(f"Image shape (Z, Y, X): {shape}")
        logger.info(f"Native HDF5 chunks: {native_chunks}")
        logger.info(f"Output chunk shape: {chunk_shape}")
        logger.info(f"Shard shape: {shard_shape}")

        # Determine actual number of levels to process
        available_levels = reader.n_levels
        actual_n_lvls = min(n_lvls, available_levels)
        if actual_n_lvls < n_lvls:
            logger.warning(
                f"Requested {n_lvls} levels but only {available_levels} "
                f"available in Imaris file. Using {actual_n_lvls}."
            )

        # Prepare output path
        output_zarr_path = Path(output_path) / stack_name
        is_s3 = bucket_name is not None

        if is_s3:
            store_path = f"s3://{bucket_name}/{output_zarr_path}"
            logger.info(f"Writing to S3: {store_path}")
        else:
            store_path = str(output_zarr_path)
            output_zarr_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Writing to local: {store_path}")

        # Write each pyramid level
        for level_idx in range(actual_n_lvls):
            level_data_path = _data_path(level_idx)
            level_shape = reader.get_shape(level_data_path)
            level_path = f"{store_path}/{level_idx}"

            logger.info(f"Writing level {level_idx}: shape={level_shape}")

            # Create TensorStore spec for this level
            spec = create_tensorstore_spec(
                path=level_path,
                shape=level_shape,
                dtype=dtype,
                chunk_shape=chunk_shape,
                shard_shape=shard_shape,
                codec=codec,
                codec_level=codec_level,
                is_s3=is_s3,
            )

            # Open TensorStore for writing
            store = ts.open(spec).result()

            # Get dask array for this level (using native chunks for reading)
            dask_array = reader.as_dask_array(
                data_path=level_data_path,
                chunks=native_chunks,
            )

            # Write blocks aligned to shard boundaries
            write_shape = shard_shape if shard_shape else chunk_shape
            pending_writes: List[ts.Future] = []

            for block_slices in iter_block_aligned_slices(level_shape, write_shape):
                # Read block from source (this triggers dask computation)
                block_data = dask_array[block_slices].compute()

                # Write block to destination
                future = write_block_to_tensorstore(store, block_data, block_slices)
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

            logger.info(f"Completed level {level_idx}")

        # Write OME-NGFF metadata
        # Shape in 5D format: (T, C, Z, Y, X)
        shape_5d = (1, 1) + tuple(shape)
        chunk_size_5d = (1, 1) + tuple(chunk_shape)

        metadata_dict = write_ome_ngff_metadata(
            arr_shape=list(shape_5d),
            chunk_size=list(chunk_size_5d),
            image_name=channel_name or stack_name,
            n_lvls=actual_n_lvls,
            scale_factors=(2, 2, 2),
            voxel_size=tuple(voxel_size),
            channel_names=[channel_name or stack_name],
        )

        # Write metadata to zarr.json
        _write_zarr_metadata(store_path, metadata_dict, is_s3)

        logger.info(f"Successfully wrote {stack_name} with {actual_n_lvls} levels")
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

