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
import json

import requests
from aind_codeocean_pipeline_monitor.models import (
    CaptureSettings,
    PipelineMonitorSettings,
)
from aind_data_schema_models.modalities import Modality
from aind_data_transfer_service.models.core import (
    SubmitJobRequestV2,
    Task,
    UploadJobConfigsV2,
)
from codeocean.computation import DataAssetsRunParam, RunParams

# from aind_data_schema_models.platforms import Platform


# ── Configurable defaults ─────────────────────────────────
IMAGE = "ghcr.io/allenneuraldynamics/aind-exaspim-data-transformation"
IMAGE_VERSION = "dev-df92e49"
ENDPOINT = "http://aind-data-transfer-service"
S3_BUCKET = "open"  # maps to aind-open-data-dev
JOB_TYPE = "exaSPIM"  # registered job type on the dev cluster
MAX_PARTITIONS = 256
PROCESSING_SPEED_GB_PER_HOUR = 12_200

# Code Ocean pipeline is only triggered for exaSPIM *screening* datasets, which
# are small (< 300 GB total). Larger datasets are transformed/uploaded only.
CODEOCEAN_PIPELINE_ID = "5ad577f1-a6e6-49a7-af95-876480c7ed21"
CODEOCEAN_MOUNT = "exaSPIM_screening_dataset"
SCREENING_MAX_BYTES = 300 * 1024**3  # 300 GB
SCREENING_PARTITIONS = 16
# ────────────────────────────────────────────────────────────────────


def _discover_ims_files(source: str) -> list[str]:
    """Return sorted list of .ims file paths found anywhere under *source*."""
    files = sorted(glob(os.path.join(source, "**", "*.ims"), recursive=True))
    if not files:
        raise FileNotFoundError(f"No .ims files found under {source}")
    return files


def _first_file_size_mb(ims_files: list[str]) -> float:
    """Size of the first .ims file in MB (used for timeout estimation)."""
    return os.path.getsize(ims_files[0]) / (1024 * 1024)


def _total_ims_size_bytes(ims_files: list[str]) -> int:
    """Total size in bytes of all .ims files in the dataset."""
    return sum(os.path.getsize(f) for f in ims_files)


def _estimate_timeout(n_tiles: int, tile_size_mb: float) -> int:
    """Return estimated timeout in minutes based on data volume."""
    buffer_min = 8*60  # Add 8 hours as a buffer for overhead, variability, etc.
    variability_factor = 2 # Account for variability in processing speed
    return int(
        (n_tiles * tile_size_mb / 1024) / (PROCESSING_SPEED_GB_PER_HOUR) * 60 * variability_factor + buffer_min
    )


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
    single_tile_upload: bool = False,
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
    single_tile_upload : bool
        If True, only process the first tile for integration testing.
        Default is False (process all tiles).

    Notes
    -----
    Datasets are expected to arrive with complete aind-data-schema v2
    metadata off the rig. Subject and procedures metadata are gathered
    upstream by the Airflow ``gather_preliminary_metadata`` step; this
    package no longer upgrades or fetches metadata.
    """
    if subject_id is None:
        subject_id = _derive_subject_id(source)

    acq_datetime = _parse_acq_datetime(source)
    ims_files = _discover_ims_files(source)
    n_tiles = len(ims_files)
    tile_size_mb = _first_file_size_mb(ims_files)

    total_bytes = _total_ims_size_bytes(ims_files)
    total_gb = total_bytes / (1024**3)
    is_screening = total_bytes < SCREENING_MAX_BYTES
    if is_screening:
        num_partitions = SCREENING_PARTITIONS
    else:
        num_partitions = MAX_PARTITIONS
    timeout_min = _estimate_timeout(n_tiles, tile_size_mb)

    
    

    print(f"Tiles found       : {n_tiles}")
    print(f"First tile size   : {tile_size_mb:,.0f} MB")
    print(f"Total dataset size: {total_gb:,.1f} GB")
    print(f"Partitions        : {num_partitions}")
    print(f"Timeout           : {timeout_min} min")
    print(
        f"Screening dataset : {is_screening} "
        f"(Code Ocean pipeline {'ENABLED' if is_screening else 'SKIPPED'})"
    )
    if single_tile_upload:
        print(f"Single tile mode  : ENABLED (testing first tile only)")

    # Only per-job overrides; image, resources, command_script, and most
    # job_settings are provided by the "exaSPIM" job type on the server.
    exaspim_job_settings = {
        "input_source": source,
        "num_of_partitions": num_partitions,
        "dask_workers": 0,
        "use_tensorstore": True,
        "single_tile_upload": single_tile_upload,
    }

    custom_exaspim_task = Task(
        image=IMAGE,
        image_version=IMAGE_VERSION,
        image_resources={
            "array": f"0-{num_partitions - 1}",
            "time_limit": {"set": True, "number": timeout_min},
        },
        job_settings=exaspim_job_settings,
    )

    # Only screening datasets (< 300 GB) trigger the Code Ocean pipeline.
    tasks = {
        "modality_transformation_settings": {
            Modality.SPIM.abbreviation: custom_exaspim_task,
        },
        "check_s3_folder_exists": {"skip_task": True},
        "final_check_s3_folder_exist": {"skip_task": True},
        "check_metadata_files": {"skip_task": True},
        "gather_preliminary_metadata": {"skip_task": False},
        "register_data_asset": {"skip_task": not is_screening},
        "get_codeocean_asset_id": {"skip_task": not is_screening},
        "run_codeocean_pipeline": {"skip_task": not is_screening},
        "remove_source_folders": {"skip_task": True},
    }

    if is_screening:
        codeocean_pipeline_settings = PipelineMonitorSettings(
            run_params=RunParams(
                pipeline_id=CODEOCEAN_PIPELINE_ID,
                data_assets=[
                    DataAssetsRunParam(id="", mount=CODEOCEAN_MOUNT)
                ],
            ),
            capture_settings=CaptureSettings(
                tags=["exaSPIM", "RAW"],
            ),
        )
        tasks["codeocean_pipeline_settings"] = {
            Modality.SPIM.abbreviation: {
                "skip_task": False,
                "job_settings": {
                    "pipeline_monitor_settings": (
                        codeocean_pipeline_settings.model_dump(
                            mode="json", exclude_none=True
                        )
                    )
                },
            }
        }

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
        tasks=tasks,
    )

    submit_request = SubmitJobRequestV2(upload_jobs=[upload_job], 
                                        user_email= "carson.berry@alleninstitute.org",)
    payload = submit_request.model_dump(mode="json", exclude_none=True)

    resp = requests.post(
        url=f"{ENDPOINT}/api/v2/submit_jobs",
        json=payload,
    )
    print(f"\nStatus : {resp.status_code}")
    print(f"Response: {resp.json()}")
    print(f"Review your job at {ENDPOINT}/jobs/")


def test_submit_exaspim_job():
    # dataset_name = "exaSPIM_718162_2026-01-29_19-28-50"
    dataset_name = "exaSPIM_823507_2026-06-30_16-49-27"
    data_dir = f"/allen/aind/stage/exaSPIM/{dataset_name}/exaSPIM"

    submit_exaspim_job(
        source=data_dir,
        project_name="Single Neuron Reconstructions",
        subject_id="823507",
        single_tile_upload=False,  # Set to True for testing with a single tile
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
    parser.add_argument(
        "--single-tile",
        action="store_true",
        help="Process only the first tile (for integration testing).",
    )
    args = parser.parse_args()

    submit_exaspim_job(
        source=args.input_folder,
        project_name=args.project_name,
        subject_id=args.subject_id,
        single_tile_upload=args.single_tile,
    )


if __name__ == "__main__":
    # main()
    test_submit_exaspim_job()