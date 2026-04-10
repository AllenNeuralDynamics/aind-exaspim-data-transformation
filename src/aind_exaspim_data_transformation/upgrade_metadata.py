"""
Upgrade aind-data-schema metadata files from v1 to v2.5+.

Reads ``acquisition.json`` (and optionally ``instrument.json``) from a
local ExaSPIM dataset directory, upgrades them using
``aind-metadata-upgrader``, backs up the originals to S3 under
``derived/``, and writes the upgraded versions to the S3 dataset root.

Usage
-----
    python -m aind_exaspim_data_transformation.upgrade_metadata \
        --source-dir /allen/aind/stage/exaSPIM/<dataset>/exaSPIM \
        --s3-location s3://aind-open-data/<dataset_name>
"""

import argparse
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import boto3
from packaging import version

from aind_exaspim_data_transformation.utils import utils

logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

# Metadata files we know how to upgrade
_METADATA_FILES = ("acquisition.json", "instrument.json")

# Schema version threshold — anything below this gets upgraded
_V2_THRESHOLD = "2.0.0"


def _load_metadata_file(path: Path) -> dict | None:
    """Load a JSON metadata file, returning *None* if missing/empty/unreadable."""
    data = utils.read_json_as_dict(path)
    if not data:
        # read_json_as_dict returns {} for missing/unreadable files;
        # treat both None and empty dict as "not available".
        return None
    return data


def _needs_upgrade(data: dict) -> bool:
    """Return True when the schema_version in *data* is below v2."""
    sv = data.get("schema_version", "0.0.0")
    return version.parse(sv) < version.parse(_V2_THRESHOLD)


def _write_json_to_tempfile(data: dict) -> Path:
    """Serialize *data* to a temporary JSON file and return its path.

    The caller is responsible for cleaning up the temp file.
    """
    fd, tmp_path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(data, fh, indent=3, default=str)
    except Exception:
        # fd is already closed by os.fdopen's context manager __exit__;
        # do not call os.close(fd) again.
        raise
    return Path(tmp_path)


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split ``s3://bucket/key`` into ``(bucket, key)``."""
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


def _upload_bytes_to_s3(body: bytes, s3_uri: str) -> None:
    """Upload raw bytes to an S3 URI using boto3."""
    bucket, key = _parse_s3_uri(s3_uri)
    s3 = boto3.client("s3")
    s3.put_object(
        Bucket=bucket, Key=key, Body=body, ContentType="application/json"
    )
    logger.info("Uploaded %d bytes → %s", len(body), s3_uri)


def _backup_original_to_s3(
    local_path: Path, s3_location: str, filename: str
) -> None:
    """Copy a local file to ``{s3_location}/derived/v1_{filename}``."""
    s3_dest = f"{s3_location.rstrip('/')}/derived/v1_{filename}"
    logger.info("Backing up %s → %s", local_path, s3_dest)
    _upload_bytes_to_s3(local_path.read_bytes(), s3_dest)


def _upload_upgraded_to_s3(
    data: dict, s3_location: str, filename: str
) -> None:
    """Serialize *data* to JSON and upload to S3 dataset root."""
    s3_dest = f"{s3_location.rstrip('/')}/{filename}"
    logger.info("Uploading upgraded %s → %s", filename, s3_dest)
    body = json.dumps(data, indent=3, default=str).encode("utf-8")
    _upload_bytes_to_s3(body, s3_dest)


def _to_json_dict(data: Any) -> dict | None:
    """Convert upgrader output object to plain dict when available."""
    if data is None:
        return None
    if hasattr(data, "model_dump"):
        return data.model_dump(mode="json", exclude_none=True)
    return data


def _upgrade_with_instrument(
    acq_data: dict, inst_data: dict
) -> tuple[dict, dict | None]:
    """Upgrade acquisition and instrument together via ``Upgrade()``.

    Returns ``(upgraded_acq_dict, upgraded_inst_dict)``.
    """
    from aind_metadata_upgrader.upgrade import Upgrade

    record: dict = {"acquisition": acq_data, "instrument": inst_data}
    upgraded = Upgrade(record, skip_metadata_validation=True)

    upgraded_acq = _to_json_dict(upgraded.metadata.acquisition)
    upgraded_inst = _to_json_dict(upgraded.metadata.instrument)
    return upgraded_acq, upgraded_inst


def _upgrade_acquisition_only(acq_data: dict) -> dict:
    """Upgrade acquisition without a real instrument.json.

    Calls the acquisition upgrader directly with a minimal metadata
    stub containing empty ``fluorescence_filters`` and
    ``light_sources`` lists, bypassing the full ``Upgrade()`` pipeline
    (which would also try to upgrade the instrument and fail
    validation).

    Returns the upgraded acquisition as a plain dict.
    """
    from aind_data_schema.core.acquisition import Acquisition
    from aind_metadata_upgrader.acquisition.v1v2 import AcquisitionV1V2

    target_version = Acquisition.model_fields[
        "schema_version"
    ].default

    metadata_stub = {
        "instrument": {
            "fluorescence_filters": [],
            "light_sources": [],
        },
    }

    upgraded_data = AcquisitionV1V2().upgrade(
        acq_data.copy(), target_version, metadata=metadata_stub
    )

    # Validate / normalise through the Acquisition model
    acq_model = Acquisition.model_construct(**upgraded_data)
    return acq_model.model_dump(mode="json", exclude_none=True)


def _backup_originals(
    acq_path: Path,
    inst_path: Path,
    inst_data: dict | None,
    s3_location: str,
    dry_run: bool,
) -> None:
    """Backup original v1 metadata files to S3 derived/ paths."""
    if dry_run:
        logger.info(
            "[DRY RUN] Would back up %s → %s/derived/v1_acquisition.json",
            acq_path,
            s3_location.rstrip("/"),
        )
        if inst_data is not None:
            logger.info(
                "[DRY RUN] Would back up %s → %s/derived/v1_instrument.json",
                inst_path,
                s3_location.rstrip("/"),
            )
        return

    _backup_original_to_s3(acq_path, s3_location, "acquisition.json")
    if inst_data is not None:
        _backup_original_to_s3(inst_path, s3_location, "instrument.json")


def _upload_upgraded_outputs(
    upgraded_acq: dict,
    upgraded_inst: dict | None,
    s3_location: str,
    dry_run: bool,
) -> None:
    """Upload upgraded metadata files to dataset root."""
    if dry_run:
        logger.info(
            "[DRY RUN] Would upload upgraded acquisition.json "
            "(schema_version %s) → %s/acquisition.json",
            upgraded_acq.get("schema_version"),
            s3_location.rstrip("/"),
        )
    else:
        _upload_upgraded_to_s3(upgraded_acq, s3_location, "acquisition.json")
    logger.info(
        "Upgraded acquisition.json to schema_version %s",
        upgraded_acq.get("schema_version"),
    )

    if upgraded_inst is None:
        return

    if dry_run:
        logger.info(
            "[DRY RUN] Would upload upgraded instrument.json "
            "(schema_version %s) → %s/instrument.json",
            upgraded_inst.get("schema_version"),
            s3_location.rstrip("/"),
        )
    else:
        _upload_upgraded_to_s3(upgraded_inst, s3_location, "instrument.json")
    logger.info(
        "Upgraded instrument.json to schema_version %s",
        upgraded_inst.get("schema_version"),
    )


def upgrade_metadata(
    source_dir: str, s3_location: str, dry_run: bool = False
) -> None:
    """Upgrade v1 metadata files and upload results to S3.

    Parameters
    ----------
    source_dir : str
        Path to the folder containing ``.ims`` files (the ``input_source``).
        Metadata files are expected one level up, e.g.::

            /allen/aind/stage/exaSPIM/<dataset>/acquisition.json
            /allen/aind/stage/exaSPIM/<dataset>/exaSPIM/  ← source_dir

    s3_location : str
        Full S3 URI of the dataset root, e.g.
        ``s3://aind-open-data/exaSPIM_12345_2026-01-01_00-00-00``.
    dry_run : bool
        If True, perform the upgrade and validate results but skip all
        S3 uploads.  Useful for local testing without write permissions.
    """
    metadata_dir = Path(source_dir).parent
    logger.info("Looking for metadata in %s", metadata_dir)

    # ── Load acquisition.json (required) ────────────────────────────
    acq_path = metadata_dir / "acquisition.json"
    acq_data = _load_metadata_file(acq_path)

    if acq_data is None:
        raise FileNotFoundError(
            f"acquisition.json not found at {acq_path}. "
            "This file is required for metadata upgrade."
        )

    if not _needs_upgrade(acq_data):
        logger.info(
            "acquisition.json is already schema_version %s (>= %s), "
            "skipping upgrade.",
            acq_data.get("schema_version"),
            _V2_THRESHOLD,
        )
        return

    logger.info(
        "acquisition.json is schema_version %s — upgrading …",
        acq_data.get("schema_version"),
    )

    # ── Load instrument.json (optional) ─────────────────────────────
    inst_path = metadata_dir / "instrument.json"
    inst_data = _load_metadata_file(inst_path)

    if inst_data is None:
        logger.warning(
            "No instrument.json found at %s — "
            "acquisition will be upgraded with empty "
            "fluorescence_filters / light_sources.",
            inst_path,
        )

    # ── Build the upgrade record ────────────────────────────────────
    # When both files are present we hand them to Upgrade() together so
    # the acquisition upgrader can read fluorescence_filters and
    # light_sources from the instrument metadata.
    #
    # When instrument.json is absent we call the acquisition upgrader
    # directly with a minimal metadata stub containing empty filter /
    # source lists.  This avoids running the full instrument upgrade
    # pipeline (which requires valid Laser, Objective, ScanningStage
    # components) while still producing a v2 acquisition — the
    # resulting data_streams simply won't have filter / laser details.
    if inst_data is not None:
        upgraded_acq, upgraded_inst = _upgrade_with_instrument(
            acq_data, inst_data
        )
    else:
        upgraded_acq = _upgrade_acquisition_only(acq_data)
        upgraded_inst = None

    if upgraded_acq is None:
        raise RuntimeError(
            "Upgrader did not produce an upgraded acquisition.json. "
            "Check the input data and upgrader logs above."
        )

    logger.info(
        "acquisition.json upgraded: %s → %s",
        acq_data.get("schema_version"),
        upgraded_acq.get("schema_version"),
    )
    if upgraded_inst is not None:
        logger.info(
            "instrument.json upgraded: %s → %s",
            inst_data.get("schema_version"),
            upgraded_inst.get("schema_version"),
        )

    # ── Backup originals and upload upgraded files ──────────────────
    _backup_originals(
        acq_path=acq_path,
        inst_path=inst_path,
        inst_data=inst_data,
        s3_location=s3_location,
        dry_run=dry_run,
    )
    _upload_upgraded_outputs(
        upgraded_acq=upgraded_acq,
        upgraded_inst=upgraded_inst,
        s3_location=s3_location,
        dry_run=dry_run,
    )

    logger.info("Metadata upgrade complete.")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=(
            "Upgrade ExaSPIM metadata files (acquisition.json, "
            "instrument.json) from aind-data-schema v1 to v2.5+."
        ),
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help=(
            "Path to the folder containing .ims files (the input_source). "
            "Metadata files are expected one level up."
        ),
    )
    parser.add_argument(
        "--s3-location",
        type=str,
        required=True,
        help=(
            "Full S3 URI of the dataset root, e.g. "
            "s3://aind-open-data/exaSPIM_12345_2026-01-01_00-00-00"
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Perform the upgrade but skip S3 uploads. "
            "Useful for local testing without write permissions."
        ),
    )
    args = parser.parse_args()

    upgrade_metadata(
        source_dir=args.source_dir,
        s3_location=args.s3_location,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
