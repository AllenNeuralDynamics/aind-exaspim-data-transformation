"""Data readers for Imaris and TIFF sources."""

from __future__ import annotations

import math
import os
import re
from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple

import dask.array as da
import h5py
import numpy as np


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
        self, data_path: str = DEFAULT_DATA_PATH, chunks: Any = True
    ) -> da.Array:
        """Return the image stack as a Dask Array."""

        lvl = self._get_res_lvl(data_path)
        real_shape_at_lvl = self._get_shape_at_lvl(lvl)

        a = da.from_array(self.get_dataset(data_path), chunks=chunks)
        return a[
            : real_shape_at_lvl[0],
            : real_shape_at_lvl[1],
            : real_shape_at_lvl[2],
        ]

    def as_array(self, data_path: str = DEFAULT_DATA_PATH) -> np.ndarray:
        lvl = self._get_res_lvl(data_path)
        real_shape_at_lvl = self._get_shape_at_lvl(lvl)
        return self.get_dataset(data_path)[
            : real_shape_at_lvl[0],
            : real_shape_at_lvl[1],
            : real_shape_at_lvl[2],
        ]

    def get_dataset(self, data_path: str = DEFAULT_DATA_PATH) -> h5py.Dataset:
        return self._require_handle()[data_path]

    def get_shape(self) -> tuple:
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
        chunks: Any = True,
    ) -> List[da.Array]:
        darrays: List[da.Array] = []
        for lvl in range(0, num_levels):
            ds_path = f"/DataSet/ResolutionLevel {lvl}/TimePoint {timepoint}/Channel {channel}/Data"
            if ds_path not in self.get_handle():
                raise MissingDatasetError(f"{ds_path} does not exist")

            if isinstance(chunks, bool) or chunks is None:
                lvl_chunks = chunks
            else:
                lvl_shape = self._get_shape_at_lvl(lvl)
                assert len(chunks) == len(lvl_shape)
                lvl_chunks = list(chunks)
                for i in range(len(chunks)):
                    lvl_chunks[i] = min(chunks[i], lvl_shape[i])

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

    def __init__(self):
        self.factory = {
            ".h5": ImarisReader,
            ".ims": ImarisReader,
        }

    def get_valid_extensions(self):
        return self.VALID_EXTENSIONS

    def create(self, filepath: str) -> DataReader:
        _, ext = os.path.splitext(filepath)
        if ext not in self.VALID_EXTENSIONS:
            raise NotImplementedError(f"File type {ext} not supported")
        return self.factory[ext](filepath)
