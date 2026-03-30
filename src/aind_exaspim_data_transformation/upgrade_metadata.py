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

from packaging import version

from aind_exaspim_data_transformation.utils import utils

logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))
logger = logging.getLogger(__name__)

# Metadata files we know how to upgrade
_METADATA_FILES = ("acquisition.json", "instrument.json")

# Schema version threshold — anything below this gets upgraded
_V2_THRESHOLD = "2.0.0"


def _load_metadata_file(path: Path) -> dict | None:
    """Load a JSON metadata file, returning *None* if missing/empty."""
    data = utils.read_json_as_dict(path)
    if not data:
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
        os.close(fd)
        raise
    return Path(tmp_path)


def _backup_original_to_s3(
    local_path: Path, s3_location: str, filename: str
) -> None:
    """Copy a local file to ``{s3_location}/derived/v1_{filename}``."""
    s3_dest = f"{s3_location.rstrip('/')}/derived/v1_{filename}"
    logger.info("Backing up %s → %s", local_path, s3_dest)
    utils.copy_file_to_s3(local_path, s3_dest)


def _upload_upgraded_to_s3(
    data: dict, s3_location: str, filename: str
) -> None:
    """Write *data* to a temp file and upload to S3 dataset root."""
    tmp = _write_json_to_tempfile(data)
    try:
        s3_dest = f"{s3_location.rstrip('/')}/{filename}"
        logger.info("Uploading upgraded %s → %s", filename, s3_dest)
        utils.copy_file_to_s3(tmp, s3_dest)
    finally:
        tmp.unlink(missing_ok=True)


def upgrade_metadata(source_dir: str, s3_location: str) -> None:
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
    """
    # Lazy import — only needed when we actually upgrade
    from aind_metadata_upgrader.upgrade import Upgrade

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
        logger.info(
            "No instrument.json found at %s — proceeding without it.",
            inst_path,
        )

    # ── Build upgrader input record ─────────────────────────────────
    record: dict = {"acquisition": acq_data}
    if inst_data is not None:
        record["instrument"] = inst_data

    # skip_metadata_validation=True because we only have partial metadata
    # (no subject, procedures, data_description, etc.)
    upgraded = Upgrade(record, skip_metadata_validation=True)

    # ── Extract upgraded dicts ──────────────────────────────────────
    upgraded_acq = None
    if upgraded.metadata.acquisition is not None:
        upgraded_acq = upgraded.metadata.acquisition.model_dump(
            mode="json", exclude_none=True
        )

    upgraded_inst = None
    if (
        inst_data is not None
        and upgraded.metadata.instrument is not None
    ):
        upgraded_inst = upgraded.metadata.instrument.model_dump(
            mode="json", exclude_none=True
        )

    # ── Backup originals to S3 derived/ ─────────────────────────────
    _backup_original_to_s3(acq_path, s3_location, "acquisition.json")

    if inst_data is not None:
        _backup_original_to_s3(
            inst_path, s3_location, "instrument.json"
        )

    # ── Upload upgraded files to S3 root ────────────────────────────
    if upgraded_acq is not None:
        _upload_upgraded_to_s3(
            upgraded_acq, s3_location, "acquisition.json"
        )
        logger.info(
            "Upgraded acquisition.json to schema_version %s",
            upgraded_acq.get("schema_version"),
        )
    else:
        raise RuntimeError(
            "Upgrader did not produce an upgraded acquisition.json. "
            "Check the input data and upgrader logs above."
        )

    if upgraded_inst is not None:
        _upload_upgraded_to_s3(
            upgraded_inst, s3_location, "instrument.json"
        )
        logger.info(
            "Upgraded instrument.json to schema_version %s",
            upgraded_inst.get("schema_version"),
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
    args = parser.parse_args()

    upgrade_metadata(
        source_dir=args.source_dir,
        s3_location=args.s3_location,
    )


if __name__ == "__main__":
    main()
