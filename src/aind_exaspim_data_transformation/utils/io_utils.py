from __future__ import annotations

import math

import dask.array as da
import h5py
import hdf5plugin  # noqa: F401
import numpy as np

from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

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

    def read_block(
        self,
        slices: Tuple[slice, ...],
        data_path: str = DEFAULT_DATA_PATH,
    ) -> np.ndarray:
        """
        Read a specific block region directly from HDF5.

        This method bypasses dask and reads directly via HDF5 hyperslab
        selection, which can be more efficient for single-threaded scenarios
        or when reading specific regions.

        Parameters
        ----------
        slices : Tuple[slice, ...]
            Tuple of slices defining the region to read (Z, Y, X).
        data_path : str
            Path to the dataset within the HDF5 file.

        Returns
        -------
        np.ndarray
            NumPy array containing the requested region.

        Examples
        --------
        >>> with ImarisReader("image.ims") as reader:
        ...     block = reader.read_block((slice(0, 256), slice(0, 256), slice(0, 256)))
        """
        return self._get_dataset(data_path)[slices]

    def iter_superchunks(
        self,
        superchunk_shape: Tuple[int, int, int],
        yield_shape: Tuple[int, int, int],
        data_path: str = DEFAULT_DATA_PATH,
    ) -> Iterator[Tuple[Tuple[slice, ...], np.ndarray]]:
        """
        Load superchunks and yield smaller blocks for efficient worker distribution.

        This method optimizes I/O by reading larger contiguous regions (superchunks)
        aligned to Imaris native chunks, then yielding smaller worker-sized blocks
        from the loaded data. This amortizes HDF5 overhead and reduces redundant
        reads when multiple workers need adjacent data.

        Parameters
        ----------
        superchunk_shape : Tuple[int, int, int]
            Shape of the superchunk to load at once (Z, Y, X).
            Should be a multiple of the native Imaris chunk size (32, 128, 128)
            for optimal I/O. Typical values: (256, 512, 512) or (512, 512, 512).
        yield_shape : Tuple[int, int, int]
            Shape of each block to yield to workers (Z, Y, X).
            Typically matches the output shard size, e.g., (256, 256, 256).
            Must evenly divide superchunk_shape in each dimension.
        data_path : str
            Path to the dataset within the HDF5 file.

        Yields
        ------
        Tuple[Tuple[slice, ...], np.ndarray]
            A tuple of (global_slices, block_data) where:
            - global_slices: Slices in the original array coordinates
            - block_data: NumPy array containing the block data

        Examples
        --------
        >>> with ImarisReader("image.ims") as reader:
        ...     for slices, block in reader.iter_superchunks(
        ...         superchunk_shape=(256, 512, 512),
        ...         yield_shape=(256, 256, 256),
        ...     ):
        ...         # Process block, write to output at slices
        ...         output[slices] = block

        Notes
        -----
        Memory usage is bounded by superchunk_shape. For a (256, 512, 512)
        superchunk of uint16 data, memory usage is ~128MB per superchunk.

        The superchunk is read once, then multiple yield_shape blocks are
        extracted without additional I/O. This is efficient when:
        - superchunk_shape is aligned to native Imaris chunks (32, 128, 128)
        - yield_shape divides evenly into superchunk_shape
        - Multiple workers can process blocks from the same superchunk
        """
        dataset = self._get_dataset(data_path)
        shape = dataset.shape

        # Validate yield_shape divides superchunk_shape
        for dim, (sc, ys) in enumerate(zip(superchunk_shape, yield_shape)):
            if sc % ys != 0:
                raise ValueError(
                    f"yield_shape[{dim}]={ys} must evenly divide "
                    f"superchunk_shape[{dim}]={sc}"
                )

        # Calculate number of superchunks in each dimension
        n_superchunks = tuple(
            math.ceil(s / sc) for s, sc in zip(shape, superchunk_shape)
        )

        # Iterate over superchunks
        for sz_idx in range(n_superchunks[0]):
            for sy_idx in range(n_superchunks[1]):
                for sx_idx in range(n_superchunks[2]):
                    # Calculate superchunk bounds (global coordinates)
                    sz_start = sz_idx * superchunk_shape[0]
                    sy_start = sy_idx * superchunk_shape[1]
                    sx_start = sx_idx * superchunk_shape[2]

                    sz_end = min(sz_start + superchunk_shape[0], shape[0])
                    sy_end = min(sy_start + superchunk_shape[1], shape[1])
                    sx_end = min(sx_start + superchunk_shape[2], shape[2])

                    # Read the superchunk from HDF5
                    superchunk_slices = (
                        slice(sz_start, sz_end),
                        slice(sy_start, sy_end),
                        slice(sx_start, sx_end),
                    )
                    superchunk_data = dataset[superchunk_slices]

                    # Actual superchunk dimensions (may be smaller at edges)
                    actual_sc_shape = superchunk_data.shape

                    # Calculate number of yield blocks within this superchunk
                    n_yields = tuple(
                        math.ceil(s / ys)
                        for s, ys in zip(actual_sc_shape, yield_shape)
                    )

                    # Yield blocks from the loaded superchunk
                    for yz_idx in range(n_yields[0]):
                        for yy_idx in range(n_yields[1]):
                            for yx_idx in range(n_yields[2]):
                                # Local coordinates within superchunk
                                local_z_start = yz_idx * yield_shape[0]
                                local_y_start = yy_idx * yield_shape[1]
                                local_x_start = yx_idx * yield_shape[2]

                                local_z_end = min(
                                    local_z_start + yield_shape[0],
                                    actual_sc_shape[0],
                                )
                                local_y_end = min(
                                    local_y_start + yield_shape[1],
                                    actual_sc_shape[1],
                                )
                                local_x_end = min(
                                    local_x_start + yield_shape[2],
                                    actual_sc_shape[2],
                                )

                                # Extract block from superchunk (no I/O)
                                block_data = superchunk_data[
                                    local_z_start:local_z_end,
                                    local_y_start:local_y_end,
                                    local_x_start:local_x_end,
                                ]

                                # Global coordinates for output
                                global_z_start = sz_start + local_z_start
                                global_y_start = sy_start + local_y_start
                                global_x_start = sx_start + local_x_start

                                global_slices = (
                                    slice(global_z_start, global_z_start + block_data.shape[0]),
                                    slice(global_y_start, global_y_start + block_data.shape[1]),
                                    slice(global_x_start, global_x_start + block_data.shape[2]),
                                )

                                yield global_slices, block_data

    def iter_blocks(
        self,
        block_shape: Tuple[int, int, int],
        data_path: str = DEFAULT_DATA_PATH,
    ) -> Iterator[Tuple[Tuple[slice, ...], np.ndarray]]:
        """
        Iterate over the dataset in block-aligned chunks.

        This is a simpler alternative to iter_superchunks that reads
        one block at a time directly from HDF5. Use this when memory
        is very constrained or when blocks are processed independently.

        Parameters
        ----------
        block_shape : Tuple[int, int, int]
            Shape of each block to yield (Z, Y, X).
        data_path : str
            Path to the dataset within the HDF5 file.

        Yields
        ------
        Tuple[Tuple[slice, ...], np.ndarray]
            A tuple of (slices, block_data) where:
            - slices: Slices defining the block position in the full array
            - block_data: NumPy array containing the block data

        Examples
        --------
        >>> with ImarisReader("image.ims") as reader:
        ...     for slices, block in reader.iter_blocks((256, 256, 256)):
        ...         output[slices] = compress(block)
        """
        dataset = self._get_dataset(data_path)
        shape = dataset.shape

        # Calculate number of blocks in each dimension
        n_blocks = tuple(
            math.ceil(s / b) for s, b in zip(shape, block_shape)
        )

        for z_idx in range(n_blocks[0]):
            for y_idx in range(n_blocks[1]):
                for x_idx in range(n_blocks[2]):
                    z_start = z_idx * block_shape[0]
                    y_start = y_idx * block_shape[1]
                    x_start = x_idx * block_shape[2]

                    z_end = min(z_start + block_shape[0], shape[0])
                    y_end = min(y_start + block_shape[1], shape[1])
                    x_end = min(x_start + block_shape[2], shape[2])

                    slices = (
                        slice(z_start, z_end),
                        slice(y_start, y_end),
                        slice(x_start, x_end),
                    )

                    yield slices, dataset[slices]

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
            List of dask arrays, one per pyramid level (highest res first)

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
                lvl_chunks = tuple(
                    min(c, s) for c, s in zip(chunks, lvl_shape)
                )
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

    def get_chunks(
        self, data_path: str = DEFAULT_DATA_PATH
    ) -> Tuple[int, ...]:
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
