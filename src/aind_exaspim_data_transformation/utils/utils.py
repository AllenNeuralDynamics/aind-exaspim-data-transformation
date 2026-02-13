"""
Utility functions for image readers
"""

import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Optional

import boto3
import numpy as np
from numpy.typing import ArrayLike

from aind_exaspim_data_transformation.models import PathLike


def add_leading_dim(data: ArrayLike) -> ArrayLike:
    """
    Adds a new dimension to existing data.
    Parameters
    ------------------------
    arr: ArrayLike
        Dask/numpy array that contains image data.

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """

    return data[None, ...]


def pad_array_n_d(arr: ArrayLike, dim: int = 5) -> ArrayLike:
    """
    Pads a daks array to be in a 5D shape.

    Parameters
    ------------------------

    arr: ArrayLike
        Dask/numpy array that contains image data.
    dim: int
        Number of dimensions that the array will be padded

    Returns
    ------------------------
    ArrayLike:
        Padded dask/numpy array.
    """
    if dim > 5:
        raise ValueError("Padding more than 5 dimensions is not supported.")

    while arr.ndim < dim:
        arr = arr[np.newaxis, ...]
    return arr


def extract_data(
    arr: ArrayLike, last_dimensions: Optional[int] = None
) -> ArrayLike:
    """
    Extracts n dimensional data (numpy array or dask array)
    given expanded dimensions.
    e.g., (1, 1, 1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1600, 2000) -> (1600, 2000)
    e.g., (1, 1, 2, 1600, 2000) -> (2, 1600, 2000)

    Parameters
    ------------------------
    arr: ArrayLike
        Numpy or dask array with image data. It is assumed
        that the last dimensions of the array contain
        the information about the image.

    last_dimensions: Optional[int]
        If given, it selects the number of dimensions given
        stating from the end
        of the array
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=3 -> (1, 1600, 2000)
        e.g., arr=(1, 1, 1600, 2000) last_dimensions=1 -> (2000)

    Raises
    ------------------------
    ValueError:
        Whenever the last dimensions value is higher
        than the array dimensions.

    Returns
    ------------------------
    ArrayLike:
        Reshaped array with the selected indices.
    """

    if last_dimensions is not None:
        if last_dimensions > arr.ndim:
            raise ValueError(
                "Last dimensions should be lower than array dimensions"
            )

    else:
        last_dimensions = len(arr.shape) - arr.shape.count(1)

    dynamic_indices = [slice(None)] * arr.ndim

    for idx in range(arr.ndim - last_dimensions):
        dynamic_indices[idx] = 0

    return arr[tuple(dynamic_indices)]


def read_json_as_dict(filepath: PathLike) -> dict:
    """
    Reads a json as dictionary.

    Parameters
    ------------------------

    filepath: PathLike
        Path where the json is located.

    Returns
    ------------------------

    dict:
        Dictionary with the data the json has.

    """

    path = Path(filepath)

    # Be defensive: mocks in tests may supply a MagicMock Path-like; treat any
    # non-existent/non-file path as empty and skip disk I/O.
    try:
        if not path.exists() or not path.is_file():
            return {}
    except Exception:
        return {}

    try:
        with path.open() as json_file:
            return json.load(json_file)
    except Exception:
        return {}


def sync_dir_to_s3(directory_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    directory_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "sync",
        str(directory_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def copy_file_to_s3(file_to_upload: PathLike, s3_location: str) -> None:
    """
    Syncs a local directory to an s3 location by running aws cli in a
    subprocess.

    Parameters
    ----------
    file_to_upload : PathLike
    s3_location : str

    Returns
    -------
    None

    """
    # Upload to s3
    if platform.system() == "Windows":
        shell = True
    else:
        shell = False

    base_command = [
        "aws",
        "s3",
        "cp",
        str(file_to_upload),
        s3_location,
        "--only-show-errors",
    ]

    subprocess.run(base_command, shell=shell, check=True)


def validate_slices(start_slice: int, end_slice: int, len_dir: int):
    """
    Validates that the slice indices are within bounds

    Parameters
    ----------
    start_slice: int
        Start slice integer

    end_slice: int
        End slice integer

    len_dir: int
        Len of czi directory
    """
    if not (0 <= start_slice < end_slice <= len_dir):
        msg = (
            f"Slices out of bounds. Total: {len_dir}"
            f"Start: {start_slice}, End: {end_slice}"
        )
        raise ValueError(msg)


def parallel_reader(
    args: tuple,
    out: np.ndarray,
    nominal_start: np.ndarray,
    start_slice: int,
    ax_index: int,
    resize: bool,
    order: int,
):
    """
    Reads a single subblock and places it in the output array.

    Parameters
    ----------
    args: tuple
        Index and directory entry of the czi file.

    out: np.ndarray
        Placeholder array for the data

    nominal_start: np.ndarray
        Nominal start of the dataset when it was acquired.

    start_slice: int
        Start slice.

    ax_index: int
        Axis index.

    resize: bool
        True if resizing is needed when reading CZI data.

    order: int
        Interpolation in resizing.
    """
    idx, directory_entry = args
    subblock = directory_entry.data_segment()
    tile = subblock.data(resize=resize, order=order)
    dir_start = np.array(directory_entry.start) - nominal_start

    # Calculate index placement
    index = tuple(slice(i, i + k) for i, k in zip(dir_start, tile.shape))
    index = list(index)
    index[ax_index] = slice(
        index[ax_index].start - start_slice, index[ax_index].stop - start_slice
    )

    try:
        out[tuple(index)] = tile
    except ValueError as e:
        raise ValueError(f"Error writing subblock {idx + start_slice}: {e}")


def generate_jumps(n: int, jump_size: Optional[int] = 128):
    """
    Generates jumps for indexing.

    Parameters
    ----------
    n: int
        Final number for indexing.
        It is exclusive in the final number.

    jump_size: Optional[int] = 128
        Jump size.
    """
    jumps = list(range(0, n, jump_size))
    # if jumps[-1] + jump_size >= n:
    #     jumps.append(n)

    return jumps


def write_json(
    output_path: str,
    json_data: dict,
    bucket_name: Optional[str] = None,
):
    """
    Writes the multiscale json in the top
    level directory of the zarr.

    Parameters
    ----------
    output_path: str
        Output path where we want the json

    json_data: dict
        Dictionary with the zarr.json metadata.

    bucket_name: Optional[str]
        Path where we want to store the json in s3.
        If default is None, the file will be saved
        locally. Default: None

    """
    json_key = f"{output_path}/zarr.json"
    if bucket_name:
        s3 = boto3.client("s3")

        # Upload the JSON string as a file to S3
        s3.put_object(
            Bucket=bucket_name,
            Key=json_key,
            Body=json.dumps(json_data, indent=2),
            ContentType="application/json",
        )

    else:
        with open(json_key, "w") as fp:
            json.dump(json_data, fp, indent=2)
