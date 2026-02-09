from __future__ import annotations

import dask.array as da
import h5py
import hdf5plugin  # noqa: F401

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
