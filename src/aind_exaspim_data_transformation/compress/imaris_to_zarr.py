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
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import dask.array as da
import h5py
import hdf5plugin  # noqa: F401 - registers HDF5 compression filters (LZ4, etc.)
import numpy as np
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

