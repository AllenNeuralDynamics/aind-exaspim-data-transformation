"""
Submit an ExaSPIM IMS-to-Zarr transformation job to the
aind-data-transfer-service-dev Airflow cluster.

Usage
-----
    python test_airflow.py /path/to/folder/with/ims_files

The input folder should contain .ims (Imaris) files directly.
No metadata files are required.
"""

import argparse
import os
import re
from datetime import datetime
from glob import glob

import requests
from aind_data_schema_models.modalities import Modality
from aind_data_transfer_service.models.core import (
    SubmitJobRequestV2,
    Task,
    UploadJobConfigsV2,
)

# from aind_data_schema_models.platforms import Platform


# ── Configurable defaults ──────────────────────────────────────────
IMAGE = "ghcr.io/allenneuraldynamics/aind-exaspim-data-transformation"
IMAGE_VERSION = "dev-9bf19b7"
ENDPOINT = "http://aind-data-transfer-service-dev"
S3_BUCKET = "open"  # maps to aind-open-data-dev
JOB_TYPE = "default"  # registered job type on the dev cluster

# Resource limits
MAX_PARTITIONS = 32
CPUS_PER_NODE = 4
MIN_RAM_MB = 24_000
MAX_RAM_MB = 40_000
SCHEDULING_OVERHEAD_MB = 1_300  # per tile, from profiling
PROCESSING_OVERHEAD_MB = 4_400  # per shard, from profiling
BUFFER_MB = 1_000
PROCESSING_SPEED_MB_PER_HOUR = 12_200
# ────────────────────────────────────────────────────────────────────


def _discover_ims_files(source: str) -> list[str]:
    """Return sorted list of .ims file paths found anywhere under *source*."""
    files = sorted(glob(os.path.join(source, "**", "*.ims"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .ims files found under {source}")
    return files


def _first_file_size_mb(ims_files: list[str]) -> float:
    """Size of the first .ims file in MB (used for resource estimation)."""
    return os.path.getsize(ims_files[0]) / (1024 * 1024)


def _estimate_resources(
    n_tiles: int, tile_size_mb: float
) -> tuple[int, int, int]:
    """Return (num_partitions, memory_per_cpu_mb, timeout_minutes)."""
    # num_partitions = min(n_tiles, MAX_PARTITIONS)
    # tiles_per_partition = max(n_tiles // num_partitions, 1)
    # we want to run multiple partitions per file
    num_partitions = 8

    # estimated_mem = (
    #     tiles_per_partition * SCHEDULING_OVERHEAD_MB
    #     + PROCESSING_OVERHEAD_MB
    #     + BUFFER_MB
    # ) // CPUS_PER_NODE
    estimated_mem = (
        MIN_RAM_MB // CPUS_PER_NODE
    )  # from profiling, seems to need at least this much per CPU regardless of tile size

    memory_per_cpu = min(
        max(estimated_mem, MIN_RAM_MB // CPUS_PER_NODE),
        MAX_RAM_MB // CPUS_PER_NODE,
    )

    timeout_min = int(
        (n_tiles * tile_size_mb / 1024) / PROCESSING_SPEED_MB_PER_HOUR + 60
    )

    return num_partitions, memory_per_cpu, timeout_min


def _parse_acq_datetime(path: str) -> datetime:
    """Extract an acquisition datetime from the folder path.

    Expects a ``YYYY-MM-DD_HH-MM-SS`` pattern somewhere in *path*.
    """
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", path)
    if not match:
        raise ValueError(
            f"Could not find a YYYY-MM-DD_HH-MM-SS datetime in: {path}"
        )
    date_part = match.group(1)
    time_part = match.group(2).replace("-", ":")
    return datetime.fromisoformat(f"{date_part}T{time_part}")


def _derive_subject_id(path: str) -> str:
    """Best-effort subject ID from the folder name."""
    basename = os.path.basename(os.path.normpath(path))
    if "_" in basename:
        return (
            basename.split("_")[1]
            if "ExaSPIM" in basename
            else basename.split("_")[0]
        )
    return basename


def submit_exaspim_job(
    source: str,
    project_name: str = "MSMA Platform",
    subject_id: str | None = None,
) -> None:
    """Build and POST an ExaSPIM transformation job.

    Parameters
    ----------
    source : str
        Folder containing .ims files.
    project_name : str
        Project name for the upload job.
    subject_id : str | None
        Subject ID. Derived from *source* folder name when ``None``.
    """
    if subject_id is None:
        subject_id = _derive_subject_id(source)

    acq_datetime = _parse_acq_datetime(source)
    ims_files = _discover_ims_files(source)
    n_tiles = len(ims_files)
    tile_size_mb = _first_file_size_mb(ims_files)
    num_partitions, mem_mb, timeout_min = _estimate_resources(
        n_tiles, tile_size_mb
    )

    print(f"Tiles found       : {n_tiles}")
    print(f"First tile size   : {tile_size_mb:,.0f} MB")
    print(f"Partitions        : {num_partitions}")
    print(f"Memory / CPU      : {mem_mb:,} MB")
    print(f"Timeout           : {timeout_min} min")

    exaspim_job_settings = {
        "input_source": source,
        "output_directory": "%OUTPUT_LOCATION",
        "s3_location": "%S3_LOCATION",
        "num_of_partitions": num_partitions,
        "use_tensorstore": True,
        "translate_imaris_pyramid": True,
    }

    custom_exaspim_task = Task(
        skip_task=False,
        image=IMAGE,
        image_version=IMAGE_VERSION,
        image_resources={
            "partition": "aind",
            "qos": "dev",
            "array": f"0-{num_partitions - 1}",
            "time_limit": {"set": True, "number": timeout_min},
            "memory_per_cpu": {"set": True, "number": mem_mb},
            "minimum_cpus_per_node": CPUS_PER_NODE,
            "standard_error": "/allen/aind/scratch/svc_aind_airflow/dev/logs/%x_%j_error.out",
            "tasks": 1,
            "standard_output": "/allen/aind/scratch/svc_aind_airflow/dev/logs/%x_%j.out",
            "environment": [
                "PATH=/bin:/usr/bin/:/usr/local/bin/",
                "LD_LIBRARY_PATH=/lib/:/lib64/:/usr/local/lib",
            ],
            "maximum_nodes": 1,
            "minimum_nodes": 1,
            "current_working_directory": ".",
            "comment": "retry 2",
        },
        job_settings=exaspim_job_settings,
        command_script=(
            "#!/bin/bash \nexport "
            "SINGULARITYENV_TRANSFORMATION_JOB_PARTITION_TO_PROCESS=$SLURM_ARRAY_TASK_ID"
            " \nsingularity exec --cleanenv --env-file %ENV_FILE docker://%IMAGE:%IMAGE_VERSION "
            "python -m aind_exaspim_data_transformation.imaris_job --job-settings ' %JOB_SETTINGS '"
        ),
    )

    upload_job = UploadJobConfigsV2(
        job_type=JOB_TYPE,
        s3_bucket=S3_BUCKET,
        project_name=project_name,
        # platform=Platform.EXASPIM,
        modalities=[Modality.SPIM],
        subject_id=subject_id,
        acq_datetime=acq_datetime,
        user_email="carson.berry@alleninstitute.org",
        email_notification_types=["all"],
        tasks={
            "modality_transformation_settings": {
                Modality.SPIM.abbreviation: custom_exaspim_task,
            },
            "check_s3_folder_exists": {"skip_task": True},
            "final_check_s3_folder_exist": {"skip_task": True},
            "check_metadata_files": {"skip_task": True},
            "gather_preliminary_metadata": {"skip_task": True},
            "register_data_asset": {"skip_task": True},
            "get_codeocean_asset_id": {"skip_task": True},
            "run_codeocean_pipeline": {"skip_task": True},
        },
    )

    submit_request = SubmitJobRequestV2(upload_jobs=[upload_job])
    payload = submit_request.model_dump(mode="json", exclude_none=True)

    resp = requests.post(
        url=f"{ENDPOINT}/api/v2/submit_jobs",
        json=payload,
    )
    print(f"\nStatus : {resp.status_code}")
    print(f"Response: {resp.json()}")
    print(f"Review your job at {ENDPOINT}/jobs/")


def test_submit_exaspim_job():
    data_dir = (
        "/allen/aind/stage/exaSPIM/"
        "exaSPIM_683791-screen_2026-01-26_14-53-41/exaSPIM"
    )

    submit_exaspim_job(
        source=data_dir,
        project_name="MSMA Platform",
        subject_id="683791-screen",
    )


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Submit an ExaSPIM IMS-to-Zarr job to "
            "aind-data-transfer-service-dev."
        ),
    )
    parser.add_argument(
        "input_folder",
        type=str,
        help="Path to a folder containing .ims (Imaris) files.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="MSMA Platform",
        help="Project name (default: MSMA Platform).",
    )
    parser.add_argument(
        "--subject-id",
        type=str,
        default=None,
        help="Subject ID. Derived from folder name if omitted.",
    )
    args = parser.parse_args()

    submit_exaspim_job(
        source=args.input_folder,
        project_name=args.project_name,
        subject_id=args.subject_id,
    )


if __name__ == "__main__":
    # main()
    test_submit_exaspim_job()
