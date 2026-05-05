"""
Generate a Neuroglancer JSON state file for visualizing exaSPIM
OME-Zarr v3 data stored on S3.

This module produces a JSON state file compatible with Neuroglancer's
``zarr3://`` source protocol. Tiles are grouped into layers by
excitation wavelength (channel).
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

logger = logging.getLogger(__name__)

# Neuroglancer viewer base URL (public demo instance)
DEFAULT_VIEWER_URL = "https://neuroglancer-demo.appspot.com"

# Mapping of common laser wavelengths (nm) to hex colours for rendering.
_WAVELENGTH_TO_HEX: Dict[int, str] = {
    405: "ff00ff",  # violet
    445: "0000ff",  # blue
    488: "00ff00",  # green
    514: "00ff80",  # cyan-green
    532: "80ff00",  # yellow-green
    561: "ffff00",  # yellow
    594: "ff8000",  # orange
    638: "ff0000",  # red
    647: "ff0000",  # red
    660: "ff0000",  # red
    680: "cc0000",  # dark red
    730: "880000",  # far-red
    750: "880000",  # far-red
    785: "660000",  # near-IR
    808: "440000",  # near-IR
}


def wavelength_to_hex(wavelength_nm: int) -> str:
    """Map a laser wavelength to a hex colour string.

    Falls back to white (``"ffffff"``) for unknown wavelengths.
    """
    return _WAVELENGTH_TO_HEX.get(wavelength_nm, "ffffff")


def _build_dimensions(
    voxel_sizes_um: List[float],
) -> Dict[str, List]:
    """Build the neuroglancer ``dimensions`` block.

    Parameters
    ----------
    voxel_sizes_um : list of float
        Voxel sizes in micrometres, **[Z, Y, X]** order.

    Returns
    -------
    dict
        Neuroglancer dimensions with values expressed in metres.
    """
    z_um, y_um, x_um = voxel_sizes_um
    return {
        "x": [x_um * 1e-6, "m"],
        "y": [y_um * 1e-6, "m"],
        "z": [z_um * 1e-6, "m"],
    }


def _build_source_transform(
    translation_um: List[float],
    voxel_sizes_um: List[float],
) -> Dict[str, Any]:
    """Build a per-source affine transform for one tile.

    The transform encodes the tile's world-space translation so that
    neuroglancer places tiles correctly relative to each other.

    Parameters
    ----------
    translation_um : list of float
        Tile translation in µm, **[Z, Y, X]** order.
    voxel_sizes_um : list of float
        Voxel sizes in µm, **[Z, Y, X]** order.

    Returns
    -------
    dict
        A neuroglancer source transform with a 3×4 affine matrix
        (identity scale, translation in voxel units) and
        ``outputDimensions``.
    """
    z_um, y_um, x_um = voxel_sizes_um
    tz, ty, tx = translation_um

    # Neuroglancer source transforms express translation in the output
    # coordinate system (physical). We use voxel-unit translations so that
    # when multiplied by the dimension scale the position is correct.
    tx_vox = tx / x_um if x_um else 0.0
    ty_vox = ty / y_um if y_um else 0.0
    tz_vox = tz / z_um if z_um else 0.0

    return {
        "matrix": [
            [1, 0, 0, tx_vox],
            [0, 1, 0, ty_vox],
            [0, 0, 1, tz_vox],
        ],
        "outputDimensions": {
            "x": [x_um * 1e-6, "m"],
            "y": [y_um * 1e-6, "m"],
            "z": [z_um * 1e-6, "m"],
        },
    }


def _build_shader(hex_color: str) -> str:
    """Return a GLSL shader string with controllable colour and range."""
    return (
        f'#uicontrol vec3 color color(default="#{hex_color}")\n'
        "#uicontrol invlerp normalized\n"
        "void main() {\n"
        "  emitRGB(color * normalized());\n"
        "}"
    )


def _build_layers(
    tiles_by_channel: Dict[int, List[Dict]],
    s3_modality_path: str,
    voxel_sizes_um: List[float],
) -> List[Dict[str, Any]]:
    """Build neuroglancer image layers grouped by channel.

    Parameters
    ----------
    tiles_by_channel : dict
        Mapping of ``excitation_wavelength`` (int, nm) to a list of tile
        dicts. Each tile dict has keys: ``file_name``, ``translation_um``
        (list[float] in ZYX µm).
    s3_modality_path : str
        S3 URI prefix for the modality folder, e.g.
        ``s3://bucket/dataset/SPIM``.
    voxel_sizes_um : list of float
        Voxel sizes in µm, [Z, Y, X] order.

    Returns
    -------
    list of dict
        Neuroglancer layer objects.
    """
    layers: List[Dict[str, Any]] = []

    for wavelength_nm in sorted(tiles_by_channel.keys()):
        tiles = tiles_by_channel[wavelength_nm]
        hex_color = wavelength_to_hex(wavelength_nm)
        layer_name = f"CH_{wavelength_nm}"

        sources = []
        for tile in tiles:
            tile_stem = Path(tile["file_name"]).stem
            zarr_uri = f"{s3_modality_path.rstrip('/')}/{tile_stem}.ome.zarr"
            source_url = f"zarr3://{zarr_uri}"

            source_entry: Dict[str, Any] = {"url": source_url}

            translation = tile.get("translation_um")
            if translation is not None:
                source_entry["transform"] = _build_source_transform(
                    translation_um=translation,
                    voxel_sizes_um=voxel_sizes_um,
                )

            sources.append(source_entry)

        layer: Dict[str, Any] = {
            "name": layer_name,
            "type": "image",
            "source": sources,
            "shader": _build_shader(hex_color),
            "shaderControls": {"normalized": {"range": [0, 200]}},
            "visible": True,
            "opacity": 1.0,
        }
        layers.append(layer)

    return layers


def build_neuroglancer_state(
    tiles_by_channel: Dict[int, List[Dict]],
    voxel_sizes_um: List[float],
    s3_modality_path: str,
) -> Dict[str, Any]:
    """Build the full neuroglancer JSON state dictionary.

    Parameters
    ----------
    tiles_by_channel : dict
        Mapping of wavelength (int nm) → list of tile info dicts.
        Each tile dict: ``{"file_name": str, "translation_um": [Z,Y,X]}``.
    voxel_sizes_um : list of float
        Voxel sizes in µm, [Z, Y, X] order.
    s3_modality_path : str
        Full S3 URI to the modality folder (e.g.
        ``s3://aind-open-data/exaSPIM_123/SPIM``).

    Returns
    -------
    dict
        A dictionary representing the neuroglancer JSON state, ready to
        be serialized with ``json.dumps()``.
    """
    state: Dict[str, Any] = {
        "dimensions": _build_dimensions(voxel_sizes_um),
        "layers": _build_layers(
            tiles_by_channel=tiles_by_channel,
            s3_modality_path=s3_modality_path,
            voxel_sizes_um=voxel_sizes_um,
        ),
        "showAxisLines": False,
        "showScaleBar": True,
    }
    return state


def generate_neuroglancer_url(
    s3_json_uri: str,
    viewer_url: str = DEFAULT_VIEWER_URL,
) -> str:
    """Construct a clickable neuroglancer viewer URL.

    The URL encodes the S3 path to the JSON state file so that
    neuroglancer fetches and renders it on load.

    Parameters
    ----------
    s3_json_uri : str
        S3 URI where the state JSON is hosted, e.g.
        ``s3://bucket/dataset/neuroglancer.json``.
    viewer_url : str
        Base URL of the neuroglancer web viewer.

    Returns
    -------
    str
        Full viewer URL with the JSON state reference.
    """
    # Convert s3:// URI → https:// URL that neuroglancer can fetch
    parsed = urlparse(s3_json_uri)
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    https_url = f"https://{bucket}.s3.amazonaws.com/{quote(key, safe='/')}"
    return f"{viewer_url}/#!{https_url}"


def parse_tiles_from_acquisition(
    acquisition_config: Dict,
) -> Tuple[Dict[int, List[Dict]], List[float]]:
    """Extract tile info and voxel sizes from an acquisition.json dict.

    Handles both schema v1 (``tiles`` array) and v2
    (``data_streams`` array).

    Parameters
    ----------
    acquisition_config : dict
        Parsed contents of ``acquisition.json``.

    Returns
    -------
    tuple of (tiles_by_channel, voxel_sizes_um)
        - tiles_by_channel: dict mapping wavelength (int) → list of
          tile dicts with keys ``file_name`` and ``translation_um``
          (ZYX, µm).
        - voxel_sizes_um: voxel size as [Z, Y, X] in µm.
    """
    from packaging import version as pkg_version

    schema_version_str = acquisition_config.get("schema_version", "0.0.0")

    if pkg_version.parse(schema_version_str) >= pkg_version.parse("2.0.0"):
        return _parse_tiles_schema_v2(acquisition_config)
    return _parse_tiles_schema_v1(acquisition_config)


def _parse_tiles_schema_v1(
    acquisition_config: Dict,
) -> Tuple[Dict[int, List[Dict]], List[float]]:
    """Parse tiles from acquisition.json schema v1.x."""
    _MM_TO_UM = 1000.0
    tiles_by_channel: Dict[int, List[Dict]] = {}
    voxel_sizes_um: Optional[List[float]] = None

    for tile in acquisition_config.get("tiles", []):
        file_name = tile.get("file_name", "")
        channel = tile.get("channel", {})
        wavelength = channel.get("excitation_wavelength", 0)
        if isinstance(wavelength, str):
            wavelength = int(float(wavelength))

        # Parse coordinate transformations
        coord_transforms = tile.get("coordinate_transformations", [])

        # Extract voxel size from scale transform (first tile only)
        if voxel_sizes_um is None:
            for ct in coord_transforms:
                if ct.get("type") == "scale":
                    scale = ct["scale"]
                    # acquisition.json: scale = [X, Y, Z] in µm
                    x_um = float(scale[0])
                    y_um = float(scale[1])
                    z_um = float(scale[2])
                    voxel_sizes_um = [z_um, y_um, x_um]
                    break

        # Extract translation
        translation_um: Optional[List[float]] = None
        for ct in coord_transforms:
            if ct.get("type") == "translation":
                raw = ct.get("translation", [])
                if len(raw) == 3:
                    # acquisition.json: translation = [X, Y, Z] in mm
                    x_mm = float(raw[0])
                    y_mm = float(raw[1])
                    z_mm = float(raw[2])
                    translation_um = [
                        z_mm * _MM_TO_UM,
                        y_mm * _MM_TO_UM,
                        x_mm * _MM_TO_UM,
                    ]
                break

        tile_info: Dict[str, Any] = {"file_name": file_name}
        if translation_um is not None:
            tile_info["translation_um"] = translation_um

        tiles_by_channel.setdefault(wavelength, []).append(tile_info)

    if voxel_sizes_um is None:
        # Fallback — should not happen if acquisition.json is well-formed
        voxel_sizes_um = [1.0, 1.0, 1.0]
        logger.warning(
            "Could not extract voxel sizes from acquisition.json; "
            "defaulting to [1, 1, 1] µm."
        )

    return tiles_by_channel, voxel_sizes_um


def _parse_tiles_schema_v2(
    acquisition_config: Dict,
) -> Tuple[Dict[int, List[Dict]], List[float]]:
    """Parse tiles from acquisition.json schema v2.x (data_streams)."""
    _MM_TO_UM = 1000.0
    tiles_by_channel: Dict[int, List[Dict]] = {}
    voxel_sizes_um: Optional[List[float]] = None

    for stream in acquisition_config.get("data_streams", []):
        for config in stream.get("configurations", []):
            # Build a channel_name → wavelength lookup from the
            # configuration-level channels array.
            channel_wavelength_map: Dict[str, int] = {}
            for ch in config.get("channels", []):
                ch_name = ch.get("channel_name", "")
                # Wavelength can be in light_sources or directly
                wl = 0
                for ls in ch.get("light_sources", []):
                    wl = ls.get("wavelength", 0)
                    if wl:
                        break
                if not wl:
                    wl = ch.get("excitation_wavelength", 0)
                if isinstance(wl, str):
                    wl = int(float(wl))
                if ch_name and wl:
                    channel_wavelength_map[ch_name] = int(wl)

            for image in config.get("images", []):
                file_name = image.get("file_name", "")

                # Resolve wavelength: first try the image's own channel
                # field, then fall back to matching channel_name in the
                # configuration-level channels list.
                img_channel = image.get("channel", {})
                wavelength = img_channel.get("excitation_wavelength", 0)
                if not wavelength:
                    ch_name = image.get("channel_name", "")
                    wavelength = channel_wavelength_map.get(ch_name, 0)
                if isinstance(wavelength, str):
                    wavelength = int(float(wavelength))

                transforms = image.get(
                    "image_to_acquisition_transform", []
                )

                # Voxel sizes from scale
                if voxel_sizes_um is None:
                    for t in transforms:
                        if t.get("object_type") == "Scale":
                            scale = t["scale"]
                            x_um = float(scale[0])
                            y_um = float(scale[1])
                            z_um = float(scale[2])
                            voxel_sizes_um = [z_um, y_um, x_um]
                            break

                # Translation
                translation_um: Optional[List[float]] = None
                for t in transforms:
                    if t.get("object_type") == "Translation":
                        raw = t.get("translation", [])
                        if len(raw) == 3:
                            x_mm = float(raw[0])
                            y_mm = float(raw[1])
                            z_mm = float(raw[2])
                            translation_um = [
                                z_mm * _MM_TO_UM,
                                y_mm * _MM_TO_UM,
                                x_mm * _MM_TO_UM,
                            ]
                        break

                tile_info: Dict[str, Any] = {"file_name": file_name}
                if translation_um is not None:
                    tile_info["translation_um"] = translation_um

                tiles_by_channel.setdefault(wavelength, []).append(tile_info)

    if voxel_sizes_um is None:
        voxel_sizes_um = [1.0, 1.0, 1.0]
        logger.warning(
            "Could not extract voxel sizes from acquisition.json v2; "
            "defaulting to [1, 1, 1] µm."
        )

    return tiles_by_channel, voxel_sizes_um
