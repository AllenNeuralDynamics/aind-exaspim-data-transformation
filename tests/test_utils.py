"""Tests for utils module."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import MagicMock, patch

import numpy as np
from numpy.typing import ArrayLike

from aind_exaspim_data_transformation.utils import utils


class TestArrayHelpers(unittest.TestCase):
    """Tests for array helper utilities."""

    def test_add_leading_dim(self):
        """Adds a leading axis to data."""
        data = cast(ArrayLike, np.zeros((2, 3)))
        result = np.asarray(utils.add_leading_dim(data))
        self.assertEqual(result.shape, (1, 2, 3))

    def test_pad_array_n_d(self):
        """Pads arrays to expected dimension."""
        data = cast(ArrayLike, np.zeros((2, 3)))
        result = np.asarray(utils.pad_array_n_d(data, dim=5))
        self.assertEqual(result.shape, (1, 1, 1, 2, 3))

    def test_pad_array_n_d_raises(self):
        """Reject padding beyond 5D."""
        data = cast(ArrayLike, np.zeros((2, 3)))
        with self.assertRaises(ValueError):
            utils.pad_array_n_d(data, dim=6)

    def test_extract_data_default(self):
        """Extracts data when leading singleton dims exist."""
        data = cast(ArrayLike, np.zeros((1, 1, 2, 3)))
        result = np.asarray(utils.extract_data(data))
        self.assertEqual(result.shape, (2, 3))

    def test_extract_data_last_dimensions(self):
        """Extracts last N dimensions."""
        data = cast(ArrayLike, np.zeros((1, 1, 2, 3)))
        result = np.asarray(utils.extract_data(data, last_dimensions=3))
        self.assertEqual(result.shape, (1, 2, 3))

    def test_extract_data_invalid_last_dimensions(self):
        """Raises when last_dimensions exceeds ndim."""
        data = cast(ArrayLike, np.zeros((1, 2)))
        with self.assertRaises(ValueError):
            utils.extract_data(data, last_dimensions=3)


class TestJsonHelpers(unittest.TestCase):
    """Tests for JSON utilities."""

    def test_read_json_as_dict_missing(self):
        """Returns empty dict when file is missing."""
        missing_path = Path(tempfile.gettempdir()) / "missing.json"
        if missing_path.exists():
            missing_path.unlink()
        self.assertEqual(utils.read_json_as_dict(missing_path), {})

    def test_read_json_as_dict(self):
        """Loads JSON from file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.json"
            payload = {"a": 1, "b": "c"}
            path.write_text(json.dumps(payload))
            result = utils.read_json_as_dict(path)
            self.assertEqual(result, payload)


class TestS3Helpers(unittest.TestCase):
    """Tests for S3 helper utilities."""

    @patch("aind_exaspim_data_transformation.utils.utils.subprocess.run")
    @patch("aind_exaspim_data_transformation.utils.utils.platform.system")
    def test_sync_dir_to_s3(self, mock_system, mock_run):
        """Syncs directory using AWS CLI."""
        mock_system.return_value = "Linux"
        utils.sync_dir_to_s3("/tmp/data", "s3://bucket/prefix")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertIn("sync", args[0])
        self.assertFalse(kwargs["shell"])

    @patch("aind_exaspim_data_transformation.utils.utils.subprocess.run")
    @patch("aind_exaspim_data_transformation.utils.utils.platform.system")
    def test_copy_file_to_s3(self, mock_system, mock_run):
        """Copies file using AWS CLI."""
        mock_system.return_value = "Windows"
        utils.copy_file_to_s3("/tmp/data.txt", "s3://bucket/prefix")
        mock_run.assert_called_once()
        args, kwargs = mock_run.call_args
        self.assertIn("cp", args[0])
        self.assertTrue(kwargs["shell"])

    @patch("aind_exaspim_data_transformation.utils.utils.boto3.client")
    def test_write_json_s3(self, mock_client):
        """Writes JSON to S3 via boto3."""
        mock_s3 = MagicMock()
        mock_client.return_value = mock_s3
        utils.write_json("prefix", {"hello": "world"}, bucket_name="bucket")
        mock_s3.put_object.assert_called_once()

    def test_write_json_local(self):
        """Writes JSON to local filesystem."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "output"
            output_path.mkdir()
            utils.write_json(str(output_path), {"a": 1})
            json_path = output_path / "zarr.json"
            self.assertTrue(json_path.exists())


class TestMiscHelpers(unittest.TestCase):
    """Tests for remaining utilities."""

    def test_validate_slices_valid(self):
        """Accepts valid slice boundaries."""
        utils.validate_slices(0, 2, 5)

    def test_validate_slices_invalid(self):
        """Rejects invalid slice boundaries."""
        with self.assertRaises(ValueError):
            utils.validate_slices(2, 1, 5)

    def test_generate_jumps(self):
        """Generates jump indices."""
        result = utils.generate_jumps(10, jump_size=4)
        self.assertEqual(result, [0, 4, 8])

    def test_parallel_reader(self):
        """Writes tile data into output array."""
        out = np.zeros((2, 2, 2), dtype=np.int32)
        nominal_start = np.array([0, 0, 0])

        subblock = MagicMock()
        subblock.data.return_value = np.ones((2, 2, 2), dtype=np.int32)
        directory_entry = MagicMock()
        directory_entry.data_segment.return_value = subblock
        directory_entry.start = [0, 0, 0]

        utils.parallel_reader(
            args=(0, directory_entry),
            out=out,
            nominal_start=nominal_start,
            start_slice=0,
            ax_index=0,
            resize=False,
            order=1,
        )

        self.assertTrue(np.all(out == 1))

    def test_parallel_reader_error(self):
        """Raises a friendly error on write failures."""
        out = np.zeros((1, 1), dtype=np.int32)
        nominal_start = np.array([0, 0])

        subblock = MagicMock()
        subblock.data.return_value = np.ones((2, 2), dtype=np.int32)
        directory_entry = MagicMock()
        directory_entry.data_segment.return_value = subblock
        directory_entry.start = [0, 0]

        with self.assertRaises(ValueError):
            utils.parallel_reader(
                args=(0, directory_entry),
                out=out,
                nominal_start=nominal_start,
                start_slice=0,
                ax_index=0,
                resize=False,
                order=1,
            )


if __name__ == "__main__":
    unittest.main()
