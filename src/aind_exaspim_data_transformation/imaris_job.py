"""Module to handle zeiss data compression"""

import logging
import multiprocessing
import os
import sys
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Tuple, cast
from urllib.parse import urlparse

from aind_data_transformation.core import GenericEtl, JobResponse, get_parser
from packaging import version

from aind_exaspim_data_transformation.compress.imaris_to_zarr import (
    enumerate_shard_indices,
    imaris_to_zarr_distributed,
    imaris_to_zarr_parallel,
    imaris_to_zarr_translate_pyramid,
    imaris_to_zarr_writer,
)
from aind_exaspim_data_transformation.models import (
    CompressorName,
    ImarisJobSettings,
)
from aind_exaspim_data_transformation.utils import utils
from aind_exaspim_data_transformation.utils.io_utils import ImarisReader

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

        # Filter to single tile if requested
        if self.job_settings.single_tile_upload and all_stack_paths:
            logging.info(
                "Single tile upload mode enabled (file partition mode): "
                "selecting first tile '%s'",
                all_stack_paths[0].name,
            )
            all_stack_paths = all_stack_paths[:1]

        return self.partition_list(
            all_stack_paths, self.job_settings.num_of_partitions
        )

    def _get_sorted_stack_paths(self) -> List[Path]:
        """Return a deterministically sorted list of stack paths."""
        all_stack_paths: List[Path] = []

        for p in Path(self.job_settings.input_source).glob("**/*.ims"):
            if p.is_file():
                all_stack_paths.append(p)

        if not all_stack_paths:
            logging.info("No .ims files found, searching for .h5 files...")
            for p in Path(self.job_settings.input_source).glob("**/*.h5"):
                if p.is_file():
                    all_stack_paths.append(p)

        all_stack_paths.sort(key=lambda x: str(x))

        # Filter to single tile if requested
        if self.job_settings.single_tile_upload and all_stack_paths:
            logging.info(
                "Single tile upload mode enabled: selecting first tile '%s'",
                all_stack_paths[0].name,
            )
            all_stack_paths = all_stack_paths[:1]

        return all_stack_paths

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

        schema_version_str = schema_version or "0.0.0"

        if version.parse(schema_version_str) >= version.parse("2.0.0"):
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

    @staticmethod
    def _get_tile_translation_from_acquisition(
        acquisition_path: Path,
        tile_filename: str,
    ) -> Optional[List[float]]:
        """
        Get the physical world-space translation for a specific tile from
        acquisition.json.

        Tile positions are matched by ``file_name`` within the ``tiles``
        array. The function looks for a ``translation`` entry inside each
        tile's ``coordinate_transformations`` list.

        The acquisition.json schema stores tile positions in **millimetres**
        (X, Y, Z order); this method converts them to **micrometres** in
        **Z, Y, X** order so the result can be passed directly to
        ``write_ome_ngff_metadata`` as the ``origin`` parameter.

        .. note::
            The mm → µm conversion factor assumes the snapshot of
            aind-data-schema seen in production (schema_version 1.x).
            Verify against a known tile by comparing the returned value to
            expected neuroglancer world coordinates before relying on this
            in production.

        Parameters
        ----------
        acquisition_path : Path
            Path to the ``acquisition.json`` file.
        tile_filename : str
            Basename of the Imaris file to look up (e.g.
            ``"tile_000000_ch_561.ims"``).

        Returns
        -------
        List[float] or None
            Translation as ``[Z, Y, X]`` in micrometres, or ``None`` if the
            file does not exist, the tile is not found in the manifest, or no
            translation transform is present for that tile.
        """
        # acquisition.json stores tile positions in mm; convert to µm.
        # NOTE: verify this factor against a real acquisition before
        # deploying to a new instrument or schema version.
        _MM_TO_UM = 1000.0

        if not acquisition_path.is_file():
            return None

        try:
            acquisition_config = utils.read_json_as_dict(acquisition_path)
        except Exception as exc:  # pylint: disable=broad-except
            logging.warning(
                "Could not parse acquisition.json for tile translation: %s",
                exc,
            )
            return None

        for tile in acquisition_config.get("tiles", []):
            if tile.get("file_name") != tile_filename:
                continue

            for transform in tile.get("coordinate_transformations", []):
                if transform.get("type") != "translation":
                    continue

                raw = transform.get("translation", [])
                if len(raw) != 3:
                    logging.warning(
                        "Unexpected translation length %d for tile %s; "
                        "expected 3 (X, Y, Z).",
                        len(raw),
                        tile_filename,
                    )
                    return None

                # acquisition.json: translation = [X, Y, Z] in mm
                x_mm = float(raw[0])
                y_mm = float(raw[1])
                z_mm = float(raw[2])

                translation_zyx_um = [
                    z_mm * _MM_TO_UM,
                    y_mm * _MM_TO_UM,
                    x_mm * _MM_TO_UM,
                ]
                logging.info(
                    "Tile %s: acquisition.json translation ZYX (µm) = %s",
                    tile_filename,
                    translation_zyx_um,
                )
                return translation_zyx_um

        logging.warning(
            "Tile '%s' not found in acquisition.json or has no translation "
            "transform; falling back to Imaris ExtMin values.",
            tile_filename,
        )
        return None

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

        input_source_path = Path(self.job_settings.input_source)
        chunk_shape: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.chunk_size)
        )
        shard_shape: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.shard_size)
        )
        scale_factor: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.scale_factor)
        )

        # Try to get voxel resolution from acquisition.json if it exists
        # Otherwise, it will be extracted from each Imaris file individually
        acquisition_path = input_source_path.joinpath("acquisition.json")

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

            # Look up this tile's world-space position from acquisition.json.
            # Returns ZYX in µm, or None (fall back to Imaris ExtMin values).
            tile_origin = self._get_tile_translation_from_acquisition(
                acquisition_path, stack.name
            )
            msg = (
                f"Voxel resolution ZYX {voxel_size_zyx} for {stack} "
                f"with name {stack_name} - output: {output_path} "
                f"with bucket name: {bucket_name}"
            )
            logging.info(msg)

            if self.job_settings.use_tensorstore:
                # Use TensorStore-based distributed writer (Zarr v3 with sharding)
                # Each worker independently reads and writes one shard at a time
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
                        chunk_shape=chunk_shape,
                        shard_shape=shard_shape,
                        n_lvls=self.job_settings.downsample_levels,
                        channel_name=stack_name,
                        stack_name=f"{stack_name}.ome.zarr",
                        bucket_name=bucket_name,
                        max_concurrent_writes=(
                            self.job_settings.tensorstore_batch_size
                        ),
                        origin=tile_origin,
                    )
                else:
                    # Use distributed worker-centric processing
                    # Each worker reads one shard from Imaris and writes it
                    dask_client = None
                    dask_cluster = None

                    if self.job_settings.dask_workers > 0:
                        try:
                            from dask.distributed import Client, LocalCluster

                            logging.info(
                                f"Creating Dask cluster with "
                                f"{self.job_settings.dask_workers} workers"
                            )
                            dask_cluster = LocalCluster(
                                n_workers=self.job_settings.dask_workers,
                                threads_per_worker=1,
                            )
                            dask_client = Client(dask_cluster)
                            logging.info(
                                f"Dask dashboard: {dask_client.dashboard_link}"
                            )
                        except ImportError:
                            logging.warning(
                                "dask.distributed not available, "
                                "falling back to sequential processing"
                            )

                    logging.info(
                        "Using distributed worker-centric writer "
                        f"(shard-per-worker, workers={self.job_settings.dask_workers})"
                    )

                    try:
                        imaris_to_zarr_distributed(
                            imaris_path=str(stack),
                            output_path=str(output_path),
                            voxel_size=voxel_size_zyx,
                            chunk_shape=chunk_shape,
                            shard_shape=shard_shape,
                            n_lvls=self.job_settings.downsample_levels,
                            scale_factor=scale_factor,
                            downsample_mode=self.job_settings.downsample_mode,
                            channel_name=stack_name,
                            stack_name=f"{stack_name}.ome.zarr",
                            bucket_name=bucket_name,
                            dask_client=dask_client,
                            origin=tile_origin,
                        )
                    finally:
                        # Clean up Dask resources
                        if dask_client is not None:
                            dask_client.close()
                        if dask_cluster is not None:
                            dask_cluster.close()
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

    def _build_global_shard_task_list(
        self, stack_paths: List[Path]
    ) -> List[tuple[Path, tuple[int, int, int]]]:
        """Enumerate all (file, shard_index) pairs across all stacks."""

        shard_shape: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.shard_size)
        )
        tasks: List[tuple[Path, tuple[int, int, int]]] = []

        for stack_path in stack_paths:
            with ImarisReader(str(stack_path)) as reader:
                data_shape = cast(
                    Tuple[int, int, int], tuple(reader.get_shape())
                )
            for shard_idx in enumerate_shard_indices(data_shape, shard_shape):
                tasks.append((stack_path, shard_idx))

        return tasks

    def _run_shard_partitioned(self, all_stacks: List[Path]):
        """Process shard-level work across all files for this partition."""

        global_tasks = self._build_global_shard_task_list(all_stacks)
        logging.info(
            "Global shard task list: %s shards across %s files",
            len(global_tasks),
            len(all_stacks),
        )

        partitioned = self.partition_list(
            global_tasks, self.job_settings.num_of_partitions
        )
        my_tasks = partitioned[self.job_settings.partition_to_process]
        if not my_tasks:
            logging.info(
                "Worker %s: no shards assigned",
                self.job_settings.partition_to_process,
            )
            return

        # Try to read voxel size from acquisition.json once
        acquisition_path = Path(self.job_settings.input_source).joinpath(
            "acquisition.json"
        )
        voxel_size_zyx = None
        if acquisition_path.exists():
            try:
                voxel_size_zyx = self._get_voxel_resolution(
                    acquisition_path=acquisition_path
                )
                logging.info(
                    "Using voxel size from acquisition.json: %s",
                    voxel_size_zyx,
                )
            except Exception as exc:  # pylint: disable=broad-except
                logging.warning(
                    "Could not read acquisition.json: %s. "
                    "Will extract voxel size from individual Imaris files.",
                    exc,
                )

        # Group tasks by file
        from collections import defaultdict

        tasks_by_file: dict[Path, List[tuple[int, int, int]]] = defaultdict(
            list
        )
        for stack_path, shard_idx in my_tasks:
            tasks_by_file[stack_path].append(shard_idx)

        # Create Dask client once and reuse across all files
        dask_client = None
        dask_cluster = None

        if self.job_settings.dask_workers > 0:
            try:
                from dask.distributed import Client, LocalCluster

                logging.info(
                    "Creating Dask cluster with %s workers",
                    self.job_settings.dask_workers,
                )
                dask_cluster = LocalCluster(
                    n_workers=self.job_settings.dask_workers,
                    threads_per_worker=1,
                )
                dask_client = Client(dask_cluster)
                logging.info("Dask dashboard: %s", dask_client.dashboard_link)
            except ImportError:
                logging.warning(
                    "dask.distributed not available, "
                    "falling back to sequential processing"
                )

        try:
            for stack_path, shard_indices in tasks_by_file.items():
                self._process_file_shards(
                    stack_path=stack_path,
                    shard_indices=shard_indices,
                    voxel_size_zyx=voxel_size_zyx,
                    dask_client=dask_client,
                    acquisition_path=acquisition_path,
                )
        finally:
            if dask_client is not None:
                dask_client.close()
            if dask_cluster is not None:
                dask_cluster.close()

    def _process_file_shards(
        self,
        stack_path: Path,
        shard_indices: List[tuple[int, int, int]],
        voxel_size_zyx: Optional[List[float]] = None,
        dask_client: Optional[Any] = None,
        acquisition_path: Optional[Path] = None,
    ) -> None:
        """Write assigned shards (and pyramid levels) for a single file."""

        if not shard_indices:
            logging.info("No shards for %s", stack_path)
            return

        output_path = Path(self.job_settings.output_directory)
        bucket_name = None

        if self.job_settings.s3_location:
            parsed = urlparse(self.job_settings.s3_location)
            bucket_name = parsed.netloc
            output_path = Path(parsed.path.lstrip("/"))

        if voxel_size_zyx is None:
            voxel_size_zyx = self._get_voxel_size_from_imaris(stack_path)

        chunk_shape: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.chunk_size)
        )
        shard_shape: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.shard_size)
        )
        scale_factor: Tuple[int, int, int] = cast(
            Tuple[int, int, int], tuple(self.job_settings.scale_factor)
        )

        logging.info(
            "Worker %s processing %s shards for %s",
            self.job_settings.partition_to_process,
            len(shard_indices),
            stack_path,
        )

        # Look up this tile's world-space position from acquisition.json.
        # Returns ZYX in µm, or None (fall back to Imaris ExtMin values).
        tile_origin = None
        if acquisition_path is not None:
            tile_origin = self._get_tile_translation_from_acquisition(
                acquisition_path, stack_path.name
            )

        imaris_to_zarr_distributed(
            imaris_path=str(stack_path),
            output_path=str(output_path),
            voxel_size=voxel_size_zyx,
            chunk_shape=chunk_shape,
            shard_shape=shard_shape,
            n_lvls=self.job_settings.downsample_levels,
            scale_factor=scale_factor,
            downsample_mode=self.job_settings.downsample_mode,
            channel_name=stack_path.stem,
            stack_name=f"{stack_path.stem}.ome.zarr",
            bucket_name=bucket_name,
            dask_client=dask_client,
            shard_indices=shard_indices,
            translate_pyramid_levels=True,
            partition_to_process=self.job_settings.partition_to_process,
            num_of_partitions=self.job_settings.num_of_partitions,
            origin=tile_origin,
        )

    def run_job(self):
        """Main entrypoint to run the job."""
        job_start_time = time()

        # Upload derivatives folder (only from partition 0)
        if self.job_settings.partition_to_process == 0:
            self._upload_derivatives_folder()

        if self.job_settings.partition_mode == "shard":
            # Shard-level partitioning: each worker processes a subset of
            # (file, shard_index) tuples across all files.
            all_stacks = self._get_sorted_stack_paths()
            logging.info(
                "Using shard-level partitioning (partition %s of %s)",
                self.job_settings.partition_to_process,
                self.job_settings.num_of_partitions,
            )
            self._run_shard_partitioned(all_stacks)
        else:
            # File-level partitioning: each worker processes whole files.
            partitioned_list = self._get_partitioned_list_of_stack_paths()
            stacks_to_process = partitioned_list[
                self.job_settings.partition_to_process
            ]
            self._write_stacks(stacks_to_process=stacks_to_process)

        total_job_duration = time() - job_start_time
        return JobResponse(
            status_code=200,
            message=f"Job finished in {total_job_duration}",
            data=None,
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
        # Construct settings from env vars or defaults (backwards compatible)
        job_settings = ImarisJobSettings()  # type: ignore[call-arg]
    job = ImarisCompressionJob(job_settings=job_settings)
    job_response = job.run_job()
    logging.info(job_response.model_dump_json())


if __name__ == "__main__":
    job_entrypoint(sys.argv[1:])
