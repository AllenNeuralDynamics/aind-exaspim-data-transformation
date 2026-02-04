"""Module to handle zeiss data compression"""

import logging
import multiprocessing
import os
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from aind_data_transformation.core import GenericEtl, JobResponse, get_parser
from packaging import version

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    ImarisReader,
    imaris_to_zarr_parallel,
    imaris_to_zarr_translate_pyramid,
    imaris_to_zarr_writer,
)
from aind_exaspim_data_transformation.models import (
    CompressorName,
    ImarisJobSettings,
)
from aind_exaspim_data_transformation.utils import utils

logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))


class ImarisCompressionJob(GenericEtl[ImarisJobSettings]):
    """Job to handle compressing and uploading Imaris data."""

    @staticmethod
    def partition_list(
        lst: List[Any], num_of_partitions: int
    ) -> List[List[Any]]:
        """Partitions a list"""
        accumulated_list = []
        for _ in range(num_of_partitions):
            accumulated_list.append([])
        for list_item_index, list_item in enumerate(lst):
            a_index = list_item_index % num_of_partitions
            accumulated_list[a_index].append(list_item)
        return accumulated_list

    def _get_partitioned_list_of_stack_paths(self) -> List[List[Path]]:
        """
        Scans through the input source and partitions a list of stack
        paths that it finds there. Looks for Imaris (.ims) files.
        """
        all_stack_paths = []
        total_counter = 0

        # Look for .ims files (Imaris format)
        for p in Path(self.job_settings.input_source).glob("**/*.ims"):
            if p.is_file():
                total_counter += 1
                all_stack_paths.append(p)

        # Also support .h5 files as fallback
        if total_counter == 0:
            logging.info("No .ims files found, searching for .h5 files...")
            for p in Path(self.job_settings.input_source).glob("**/*.h5"):
                if p.is_file():
                    total_counter += 1
                    all_stack_paths.append(p)

        logging.info(f"Found {total_counter} Imaris files to process")

        # Important to sort paths so every node computes the same list
        all_stack_paths.sort(key=lambda x: str(x))
        return self.partition_list(
            all_stack_paths, self.job_settings.num_of_partitions
        )

    @staticmethod
    def _get_voxel_resolution(acquisition_path: Path) -> List[float]:
        """
        Get the voxel resolution from an acquisition.json file.

        This method is kept for backward compatibility with datasets that
        have acquisition.json. For Imaris files, voxel size can be extracted
        directly from the file metadata.
        """

        if not acquisition_path.is_file():
            raise FileNotFoundError(
                f"acquisition.json file not found at: {acquisition_path}"
            )

        acquisition_config = utils.read_json_as_dict(acquisition_path)

        schema_version = acquisition_config.get("schema_version")
        logging.info(f"Schema version: {schema_version}")

        if version.parse(schema_version) >= version.parse("2.0.0"):
            return ImarisCompressionJob._get_voxel_resolution_schema_2(
                acquisition_config
            )

        # Grabbing a tile with metadata from acquisition - we assume all
        # dataset was acquired with the same resolution
        tile_coord_transforms = acquisition_config["tiles"][0][
            "coordinate_transformations"
        ]

        scale_transform = [
            x["scale"] for x in tile_coord_transforms if x["type"] == "scale"
        ][0]

        x = float(scale_transform[0])
        y = float(scale_transform[1])
        z = float(scale_transform[2])

        return [z, y, x]

    @staticmethod
    def _get_voxel_size_from_imaris(imaris_path: Path) -> List[float]:
        """
        Extract voxel size directly from an Imaris file.

        Parameters
        ----------
        imaris_path : Path
            Path to the Imaris file

        Returns
        -------
        List[float]
            Voxel size in [Z, Y, X] order in micrometers
        """
        with ImarisReader(str(imaris_path)) as reader:
            voxel_size, unit = reader.get_voxel_size()
            unit_display = unit.decode() if isinstance(unit, bytes) else unit
            logging.info(
                "Extracted voxel size from %s: %s %s",
                imaris_path.name,
                voxel_size,
                unit_display,
            )
            return voxel_size

    @staticmethod
    def _get_voxel_resolution_schema_2(
        acquisition_config: Dict,
    ) -> List[float]:
        """Get the voxel resolution from an acquisition.json file
        for aind-data-schema==2.0.0"""

        # Grabbing a tile with metadata from acquisition - we assume all
        # dataset was acquired with the same resolution
        try:
            data_stream = acquisition_config.get("data_streams", [])[0]
            configuration = data_stream.get("configurations", [])[0]
            image = configuration.get("images", [])[0]
            image_to_acquisition_transform = image[
                "image_to_acquisition_transform"
            ]
        except (IndexError, AttributeError, KeyError) as e:
            raise ValueError(
                "acquisition_config structure is invalid or missing "
                "required fields"
            ) from e

        scale_transform = [
            x["scale"]
            for x in image_to_acquisition_transform
            if x["object_type"] == "Scale"
        ][0]

        x = float(scale_transform[0])
        y = float(scale_transform[1])
        z = float(scale_transform[2])

        return [z, y, x]

    def _get_compressor(self) -> Optional[Dict]:
        """
        Utility method to construct a compressor class.
        Returns
        -------
        Blosc | None
          An instantiated Blosc compressor. Return None if not set in configs.

        """
        if self.job_settings.compressor_name == CompressorName.BLOSC:
            return self.job_settings.compressor_kwargs
        else:
            return None

    def _write_stacks(self, stacks_to_process: List) -> None:
        """
        Write a list of Imaris stacks to OME-Zarr format.

        Parameters
        ----------
        stacks_to_process : List
            List of Path objects pointing to Imaris files

        Returns
        -------
        None

        """

        if not len(stacks_to_process):
            logging.info("No stacks to process!")
            return

        compressor = self._get_compressor()

        # Try to get voxel resolution from acquisition.json if it exists
        # Otherwise, it will be extracted from each Imaris file individually
        acquisition_path = self.job_settings.input_source.joinpath(
            "acquisition.json"
        )

        voxel_size_zyx = None
        if acquisition_path.exists():
            try:
                voxel_size_zyx = self._get_voxel_resolution(
                    acquisition_path=acquisition_path
                )
                logging.info(
                    f"Using voxel size from acquisition.json: {voxel_size_zyx}"
                )
            except Exception as e:
                logging.warning(
                    f"Could not read acquisition.json: {e}. "
                    "Will extract voxel size from individual Imaris files."
                )

        # Output path for VAST
        output_path = Path(self.job_settings.output_directory)
        bucket_name = None

        # If s3_location is provided, we define the bucket name
        # and parse the output path to the prefix
        if self.job_settings.s3_location:
            parsed = urlparse(self.job_settings.s3_location)
            bucket_name = parsed.netloc
            output_path = Path(parsed.path.lstrip("/"))

        # Converting Imaris files to Multiscale OMEZarr
        for stack in stacks_to_process:
            logging.info(f"Converting {stack}")
            stack_name = stack.stem

            # If voxel size wasn't available from acquisition.json,
            # extract it from the Imaris file
            if voxel_size_zyx is None:
                voxel_size_zyx = self._get_voxel_size_from_imaris(stack)

            msg = (
                f"Voxel resolution ZYX {voxel_size_zyx} for {stack} "
                f"with name {stack_name} - output: {output_path} "
                f"with bucket name: {bucket_name}"
            )
            logging.info(msg)

            if self.job_settings.use_tensorstore:
                # Use TensorStore-based parallel writer (Zarr v3 with sharding)
                if self.job_settings.translate_imaris_pyramid:
                    # Translate existing Imaris pyramid levels (faster)
                    logging.info(
                        "Using TensorStore pyramid translator "
                        "(translating existing Imaris pyramids)"
                    )
                    imaris_to_zarr_translate_pyramid(
                        imaris_path=str(stack),
                        output_path=str(output_path),
                        voxel_size=voxel_size_zyx,
                        chunk_shape=tuple(self.job_settings.chunk_size),
                        shard_shape=tuple(self.job_settings.shard_size),
                        n_lvls=self.job_settings.downsample_levels,
                        channel_name=stack_name,
                        stack_name=f"{stack_name}.ome.zarr",
                        bucket_name=bucket_name,
                        max_concurrent_writes=(
                            self.job_settings.tensorstore_batch_size
                        ),
                    )
                else:
                    # Re-compute pyramid levels using TensorStore downsample
                    logging.info(
                        "Using TensorStore parallel writer "
                        "(re-computing pyramid levels)"
                    )
                    imaris_to_zarr_parallel(
                        imaris_path=str(stack),
                        output_path=str(output_path),
                        voxel_size=voxel_size_zyx,
                        chunk_shape=tuple(self.job_settings.chunk_size),
                        shard_shape=tuple(self.job_settings.shard_size),
                        n_lvls=self.job_settings.downsample_levels,
                        scale_factor=tuple(self.job_settings.scale_factor),
                        downsample_mode=self.job_settings.downsample_mode,
                        channel_name=stack_name,
                        stack_name=f"{stack_name}.ome.zarr",
                        bucket_name=bucket_name,
                        max_concurrent_writes=(
                            self.job_settings.tensorstore_batch_size
                        ),
                    )
            else:
                # Use standard dask-based writer
                imaris_to_zarr_writer(
                    imaris_path=str(stack),
                    output_path=str(output_path),
                    voxel_size=voxel_size_zyx,
                    chunk_size=self.job_settings.chunk_size,
                    scale_factor=self.job_settings.scale_factor,
                    n_lvls=self.job_settings.downsample_levels,
                    channel_name=stack_name,
                    stack_name=f"{stack_name}.ome.zarr",
                    compressor_kwargs=compressor,
                    bucket_name=bucket_name,
                )

    def _upload_derivatives_folder(self):
        """
        Uploads the 'derivatives' folder if it exists in the input source.
        This is optional and will be skipped if the folder doesn't exist.
        """
        derivatives_path = Path(self.job_settings.input_source).joinpath(
            "derivatives"
        )

        if not derivatives_path.exists():
            logging.info(
                f"No derivatives folder found at {derivatives_path}, "
                "skipping upload."
            )
            return

        if self.job_settings.s3_location is not None:
            s3_derivatives_dir = f"{self.job_settings.s3_location}/derivatives"
            logging.info(
                f"Uploading {derivatives_path} to {s3_derivatives_dir}"
            )
            utils.sync_dir_to_s3(derivatives_path, s3_derivatives_dir)
            logging.info(f"{derivatives_path} uploaded to s3.")

    def run_job(self):
        """Main entrypoint to run the job."""
        job_start_time = time()

        # Reading data within the SPIM folder
        partitioned_list = self._get_partitioned_list_of_stack_paths()

        # Upload derivatives folder
        if self.job_settings.partition_to_process == 0:
            self._upload_derivatives_folder()

        stacks_to_process = partitioned_list[
            self.job_settings.partition_to_process
        ]

        self._write_stacks(stacks_to_process=stacks_to_process)
        total_job_duration = time() - job_start_time
        return JobResponse(
            status_code=200, message=f"Job finished in {total_job_duration}"
        )


def job_entrypoint(sys_args: list):
    """Main function"""
    multiprocessing.set_start_method("spawn", force=True)

    parser = get_parser()
    cli_args = parser.parse_args(sys_args)
    if cli_args.job_settings is not None:
        job_settings = ImarisJobSettings.model_validate_json(
            cli_args.job_settings
        )
    elif cli_args.config_file is not None:
        job_settings = ImarisJobSettings.from_config_file(cli_args.config_file)
    else:
        # Construct settings from env vars
        job_settings = ImarisJobSettings()
    job = ImarisCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())


if __name__ == "__main__":
    job_entrypoint(sys.argv[1:])
