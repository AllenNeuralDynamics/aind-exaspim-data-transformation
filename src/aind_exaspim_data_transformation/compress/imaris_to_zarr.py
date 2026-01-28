"""Data readers for Imaris and TIFF sources."""

from __future__ import annotations

import logging
import math
import os
import re
from abc import ABC, abstractmethod
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


class MissingDatasetError(FileNotFoundError):
    """Raised when an expected dataset path is absent in the file."""


class DataReader(ABC):
    """Abstract base class for image data readers."""

    def __init__(self, filepath: str):
        self.filepath = filepath

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pragma: no cover - passthrough
        self.close()

    @abstractmethod
    def as_dask_array(self, *args, **kwargs) -> da.Array:
        """Return the image as a dask array."""

    @abstractmethod
    def as_array(self, *args, **kwargs) -> np.ndarray:
        """Return the image as a numpy array."""

    @abstractmethod
    def get_shape(self) -> tuple:
        """Return the array shape."""

    @abstractmethod
    def get_chunks(self, *args, **kwargs) -> tuple | None:
        """Return the chunk shape if available."""

    @abstractmethod
    def get_itemsize(self, *args, **kwargs) -> int:
        """Return the bytes per element."""

    @abstractmethod
    def get_handle(self):
        """Return the underlying file handle."""

    @abstractmethod
    def close(self) -> None:
        """Close any open resources."""


class ImarisReader(DataReader):
    DEFAULT_DATA_PATH = "/DataSet/ResolutionLevel 0/TimePoint 0/Channel 0/Data"

    def __init__(self, filepath):
        super().__init__(filepath)
        self.handle: Optional[h5py.File] = h5py.File(self.filepath, mode="r")

    def _require_handle(self) -> h5py.File:
        if self.handle is None:
            raise RuntimeError("Imaris file handle is closed")
        return self.handle

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def as_dask_array(
        self, data_path: str = DEFAULT_DATA_PATH, chunks: Any = True,
        trim_padding: bool = False
    ) -> da.Array:
        """Return the image stack as a Dask Array.
        
        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file
        chunks : Any
            Chunk specification for dask array. Use True for auto, 
            None for no chunking, or tuple for explicit chunks.
        trim_padding : bool
            If True, trim the array to the "real" shape from metadata.
            If False (default), return the full padded HDF5 dataset.
            Note: Trimming may cause chunk misalignment issues.
        
        Uses lock=True to ensure thread-safe access to the HDF5 file,
        enabling parallel reads with the threaded scheduler.
        """
        dataset = self.get_dataset(data_path)
        
        # Use lock=True to enable thread-safe HDF5 access
        # This allows dask to use multiple threads for parallel reads
        a = da.from_array(dataset, chunks=chunks, lock=True)
        
        if trim_padding:
            lvl = self._get_res_lvl(data_path)
            real_shape_at_lvl = self._get_shape_at_lvl(lvl)
            a = a[
                : real_shape_at_lvl[0],
                : real_shape_at_lvl[1],
                : real_shape_at_lvl[2],
            ]
        
        return a

    def as_array(self, data_path: str = DEFAULT_DATA_PATH, trim_padding: bool = False) -> np.ndarray:
        """Return the image stack as a NumPy array.
        
        Parameters
        ----------
        data_path : str
            Path to the dataset within the HDF5 file
        trim_padding : bool
            If True, trim the array to the "real" shape from metadata.
            If False (default), return the full padded HDF5 dataset.
        """
        dataset = self.get_dataset(data_path)
        
        if trim_padding:
            lvl = self._get_res_lvl(data_path)
            real_shape_at_lvl = self._get_shape_at_lvl(lvl)
            return dataset[
                : real_shape_at_lvl[0],
                : real_shape_at_lvl[1],
                : real_shape_at_lvl[2],
            ]
        else:
            return dataset[:]

    def get_dataset(self, data_path: str = DEFAULT_DATA_PATH) -> h5py.Dataset:
        return self._require_handle()[data_path]

    def get_shape(self, data_path: str = DEFAULT_DATA_PATH) -> tuple:
        """Return the actual HDF5 dataset shape (may include padding)."""
        return self.get_dataset(data_path).shape

    def get_metadata_shape(self) -> tuple:
        """Return the 'real' image shape from Imaris metadata (excludes padding)."""
        info = self.get_dataset_info()
        return (
            int(info.attrs["Z"].tobytes()),
            int(info.attrs["Y"].tobytes()),
            int(info.attrs["X"].tobytes()),
        )

    def get_chunks(self, data_path: str = DEFAULT_DATA_PATH) -> tuple:
        return self.get_dataset(data_path).chunks

    def get_itemsize(self, data_path: str = DEFAULT_DATA_PATH) -> int:
        return self.get_dataset(data_path).dtype.itemsize

    def get_handle(self) -> h5py.File:
        return self._require_handle()

    def get_dask_pyramid(
        self,
        num_levels: int,
        timepoint: int = 0,
        channel: int = 0,
        chunks: Any = "native",
    ) -> List[da.Array]:
        """Get multiple resolution levels as dask arrays.
        
        Parameters
        ----------
        num_levels : int
            Number of pyramid levels to retrieve
        timepoint : int
            Timepoint index (default 0)
        channel : int
            Channel index (default 0)
        chunks : Any
            Chunk specification. Use "native" (default) to use HDF5 native chunks,
            True for dask auto-chunking, None for no chunking, or tuple for explicit.
        """
        darrays: List[da.Array] = []
        for lvl in range(0, num_levels):
            ds_path = f"/DataSet/ResolutionLevel {lvl}/TimePoint {timepoint}/Channel {channel}/Data"
            if ds_path not in self.get_handle():
                raise MissingDatasetError(f"{ds_path} does not exist")

            # Determine chunks for this level
            if chunks == "native":
                # Use the native HDF5 chunks - most efficient for reading
                lvl_chunks = self.get_chunks(ds_path)
            elif isinstance(chunks, bool) or chunks is None:
                lvl_chunks = chunks
            else:
                # Explicit chunks provided - clamp to level shape
                lvl_shape = self.get_shape(ds_path)
                assert len(chunks) == len(lvl_shape)
                lvl_chunks = tuple(min(c, s) for c, s in zip(chunks, lvl_shape))

            darrays.append(self.as_dask_array(ds_path, chunks=lvl_chunks))

        return darrays

    def get_dataset_info(self) -> h5py.Dataset:
        return self.get_handle()["DataSetInfo/Image"]

    def get_origin(self) -> List[float]:
        info = self.get_dataset_info()
        x_min = float(info.attrs["ExtMin0"].tobytes())
        y_min = float(info.attrs["ExtMin1"].tobytes())
        z_min = float(info.attrs["ExtMin2"].tobytes())
        return [z_min, y_min, x_min]

    def get_extent(self) -> List[float]:
        info = self.get_dataset_info()
        x_max = float(info.attrs["ExtMax0"].tobytes())
        y_max = float(info.attrs["ExtMax1"].tobytes())
        z_max = float(info.attrs["ExtMax2"].tobytes())
        return [z_max, y_max, x_max]

    def get_voxel_size(self) -> Tuple[List[float], bytes]:
        info = self.get_dataset_info()
        extmin = self.get_origin()
        extmax = self.get_extent()
        x = int(info.attrs["X"].tobytes())
        y = int(info.attrs["Y"].tobytes())
        z = int(info.attrs["Z"].tobytes())
        unit = info.attrs["Unit"].tobytes()
        voxsize = [
            (extmax[0] - extmin[0]) / z,
            (extmax[1] - extmin[1]) / y,
            (extmax[2] - extmin[2]) / x,
        ]
        return voxsize, unit

    def close(self) -> None:
        if self.handle is not None:
            self.handle.close()
            self.handle = None

    def _get_res_lvl(self, data_path: str):
        lvl_rgx = r"/ResolutionLevel (\d+)/"
        m = re.search(lvl_rgx, data_path)
        if m is None:
            raise ValueError(
                f"Could not parse resolution level from path {data_path}"
            )
        return int(m.group(1))

    def _get_shape_at_lvl(self, lvl: int):
        shape = self.get_shape()
        return tuple(int(math.ceil(s / 2**lvl)) for s in shape)

    @property
    def n_levels(self):
        lvl = 0
        while True:
            ds_path = (
                f"/DataSet/ResolutionLevel {lvl}/TimePoint 0/Channel 0/Data"
            )
            if ds_path not in self.get_handle():
                return lvl
            lvl += 1


class DataReaderFactory:
    VALID_EXTENSIONS = [".h5", ".ims"]

    _factory = {
        ".h5": ImarisReader,
        ".ims": ImarisReader,
    }

    @classmethod
    def get_valid_extensions(cls):
        return cls.VALID_EXTENSIONS

    @classmethod
    def create(cls, filepath: str) -> DataReader:
        _, ext = os.path.splitext(filepath)
        if ext not in cls.VALID_EXTENSIONS:
            raise NotImplementedError(f"File type {ext} not supported")
        return cls._factory[ext](filepath)


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
) -> None:
    """
    Convert an Imaris file to OME-Zarr format.

    Parameters
    ----------
    imaris_path : str
        Path to the input Imaris file (.ims or .h5)
    output_path : str
        Path where the output OME-Zarr will be written
    voxel_size : Optional[List[float]]
        Voxel size in [Z, Y, X] order in micrometers. 
        If None, will be extracted from Imaris metadata.
    chunk_size : Optional[List[int]]
        Chunk size for zarr arrays in [Z, Y, X] order.
        Default is [128, 128, 128]
    scale_factor : Optional[List[int]]
        Downsampling scale factors in [Z, Y, X] order.
        Default is [2, 2, 2]
    n_lvls : int
        Number of pyramid levels to generate. Default is 5.
    channel_name : Optional[str]
        Name for the channel. If None, uses stack name.
    stack_name : Optional[str]
        Name for the output zarr store. If None, derives from input filename.
    compressor_kwargs : Optional[Dict]
        Compression settings for Blosc compressor.
        Default is {"cname": "zstd", "clevel": 3, "shuffle": Blosc.SHUFFLE}
    bucket_name : Optional[str]
        S3 bucket name for cloud storage. If None, writes locally.

    Returns
    -------
    None
    """
    logging.info(f"Starting Imaris to Zarr conversion: {imaris_path}")

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

    # Initialize compressor
    compressor = Blosc(**compressor_kwargs)

    # Open Imaris file
    with ImarisReader(imaris_path) as reader:
        # Get voxel size from file if not provided
        if voxel_size is None:
            voxel_size_from_file, unit = reader.get_voxel_size()
            voxel_size = voxel_size_from_file
            logging.info(f"Extracted voxel size from Imaris: {voxel_size} {unit}")
        else:
            logging.info(f"Using provided voxel size: {voxel_size}")

        # Determine actual number of levels available
        actual_n_lvls = min(n_lvls, reader.n_levels)
        if actual_n_lvls < n_lvls:
            logging.warning(
                f"Requested {n_lvls} levels but only {actual_n_lvls} available in Imaris file"
            )

        # Get pyramid as dask arrays (using native HDF5 chunks for efficient reading)
        native_chunks = reader.get_chunks()
        logging.info(f"Using native HDF5 chunks: {native_chunks}")
        
        pyramid = reader.get_dask_pyramid(
            num_levels=actual_n_lvls,
            timepoint=0,
            channel=0,
            chunks=native_chunks,  # Use native chunks to avoid rechunking overhead
        )
        
        # Get the shape from the first pyramid level
        shape = pyramid[0].shape
        logging.info(f"Image shape (Z, Y, X): {shape}")

        # Prepare output path
        output_zarr_path = Path(output_path) / stack_name
        if bucket_name:
            # S3 path
            store_path = f"s3://{bucket_name}/{output_zarr_path}"
            logging.info(f"Writing to S3: {store_path}")
        else:
            # Local filesystem
            store_path = str(output_zarr_path)
            output_zarr_path.parent.mkdir(parents=True, exist_ok=True)
            logging.info(f"Writing to local: {store_path}")

        # Create root zarr group
        root = zarr.open_group(store_path, mode="w")

        # Write each pyramid level
        for level_idx, dask_array in enumerate(pyramid):
            level_path = f"{level_idx}"
            logging.info(
                f"Writing level {level_idx}: shape={dask_array.shape}, "
                f"chunks={dask_array.chunksize}"
            )

            # Create zarr array for this level
            z_array = root.create_dataset(
                level_path,
                shape=dask_array.shape,
                chunks=dask_array.chunksize,
                dtype=dask_array.dtype,
                compressor=compressor,
                overwrite=True,
            )

            # Write dask array to zarr using compute=True to process lazily
            # This streams chunks through memory instead of loading everything
            da.to_zarr(
                dask_array, 
                z_array, 
                overwrite=True,
                compute=True,  # Execute the write
            )

        # Generate and write OME-NGFF metadata
        # Wrap shape in 5D format: (T, C, Z, Y, X)
        # Use the native chunks for metadata (what's actually stored)
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

        # Write metadata to zarr attributes
        root.attrs.update(metadata_dict)

        logging.info(f"Successfully wrote {stack_name} with {actual_n_lvls} levels")

