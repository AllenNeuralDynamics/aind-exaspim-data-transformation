"""
Generate a Neuroglancer JSON state file for visualizing exaSPIM
OME-Zarr v3 data stored on S3.

This module produces a JSON state file compatible with Neuroglancer's
``zarr3://`` source protocol. Tiles are grouped into layers by
excitation wavelength (channel).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, urlparse

import numpy as np

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

    The exaSPIM OME-Zarr arrays are 5D (``[t, c, z, y, x]``), but each
    tile file holds a single channel and every wavelength is rendered as
    its own layer, so the ``c`` (channel) dimension is not exposed. The
    singleton ``t`` (time) dimension **is** declared to keep neuroglancer
    from auto-assigning a non-zero ``t`` position. Dimensions are ordered
    ``z, y, x, t``.

    Parameters
    ----------
    voxel_sizes_um : list of float
        Voxel sizes in micrometres, **[Z, Y, X]** order.

    Returns
    -------
    dict
        Neuroglancer dimensions with spatial values expressed in metres.
    """
    z_um, y_um, x_um = voxel_sizes_um
    return {
        "z": [z_um * 1e-6, "m"],
        "y": [y_um * 1e-6, "m"],
        "x": [x_um * 1e-6, "m"],
        "t": [0.001, "s"],
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
        A neuroglancer source transform with a 4×5 affine matrix
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

    # The source arrays are 5D ([t, c, z, y, x]), but neuroglancer pulls
    # the OME ``channel``-type axis out as a separate (non-spatial)
    # channel dimension, so the coordinate transform operates on the
    # remaining 4 dims. The matrix has one row per output dimension
    # (ordered z, y, x, t) and one column per input dimension in the
    # source order ([t, z, y, x]) plus a translation column (4 + 1 = 5).
    # ``t`` is carried through so it stays pinned at 0.
    return {
        "matrix": [
            [0, 1, 0, 0, tz_vox],
            [0, 0, 1, 0, ty_vox],
            [0, 0, 0, 1, tx_vox],
            [1, 0, 0, 0, 0],
        ],
        "outputDimensions": {
            "z": [z_um * 1e-6, "m"],
            "y": [y_um * 1e-6, "m"],
            "x": [x_um * 1e-6, "m"],
            "t": [0.001, "s"],
        },
    }


def _build_position(
    tiles_by_channel: Dict[int, List[Dict]],
    voxel_sizes_um: List[float],
) -> List[float]:
    """Build the initial viewer ``position`` (ordered ``z, y, x, t``).

    The singleton ``t`` (time) coordinate is pinned to ``0`` so the
    viewer initialises on the only timepoint (declaring the ``t``
    dimension alone is not enough; neuroglancer otherwise auto-assigns a
    non-zero ``t`` position and nothing renders). The spatial ``z, y, x``
    coordinates are centred on the tile mosaic (midpoint of the tile
    translation bounding box, in voxel units) so data is in view.

    Parameters
    ----------
    tiles_by_channel : dict
        Mapping of wavelength (int, nm) to a list of tile dicts, each with
        an optional ``translation_um`` (list[float], ZYX µm).
    voxel_sizes_um : list of float
        Voxel sizes in µm, **[Z, Y, X]** order.

    Returns
    -------
    list of float
        Position ``[z, y, x, t]`` in voxel units, with ``t == 0``.
    """
    z_um, y_um, x_um = voxel_sizes_um
    translations_vox: List[List[float]] = []
    for tiles in tiles_by_channel.values():
        for tile in tiles:
            tz, ty, tx = tile.get("translation_um") or (0.0, 0.0, 0.0)
            translations_vox.append(
                [
                    tz / z_um if z_um else 0.0,
                    ty / y_um if y_um else 0.0,
                    tx / x_um if x_um else 0.0,
                ]
            )

    if translations_vox:
        arr = np.asarray(translations_vox, dtype=float)
        z_c, y_c, x_c = ((arr.min(axis=0) + arr.max(axis=0)) / 2.0).tolist()
    else:
        z_c = y_c = x_c = 0.0

    return [z_c, y_c, x_c, 0.0]


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
    contrast_limits_by_channel: Optional[
        Dict[int, Tuple[float, float]]
    ] = None,
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
    contrast_limits_by_channel : dict, optional
        Mapping of wavelength (int, nm) to a ``(low, high)`` intensity
        window used to initialise each layer's ``invlerp`` shader range.
        Channels without an entry fall back to a default range.

    Returns
    -------
    list of dict
        Neuroglancer layer objects.
    """
    contrast_limits_by_channel = contrast_limits_by_channel or {}
    _DEFAULT_RANGE = [0, 200]
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

        limits = contrast_limits_by_channel.get(wavelength_nm)
        shader_range = list(limits) if limits is not None else _DEFAULT_RANGE

        layer: Dict[str, Any] = {
            "name": layer_name,
            "type": "image",
            "source": sources,
            "shader": _build_shader(hex_color),
            "shaderControls": {"normalized": {"range": shader_range}},
            "blend": "additive",
            "visible": True,
            "opacity": 1.0,
        }
        layers.append(layer)

    return layers


def build_neuroglancer_state(
    tiles_by_channel: Dict[int, List[Dict]],
    voxel_sizes_um: List[float],
    s3_modality_path: str,
    contrast_limits_by_channel: Optional[
        Dict[int, Tuple[float, float]]
    ] = None,
    ng_link: Optional[str] = None,
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
    contrast_limits_by_channel : dict, optional
        Mapping of wavelength (int, nm) to a ``(low, high)`` intensity
        window used to initialise each layer's shader range.
    ng_link : str, optional
        Self-referential neuroglancer viewer URL. When provided it is
        inserted as the first key of the state for easy access.

    Returns
    -------
    dict
        A dictionary representing the neuroglancer JSON state, ready to
        be serialized with ``json.dumps()``.
    """
    state: Dict[str, Any] = {}
    if ng_link is not None:
        state["ng_link"] = ng_link
    state.update(
        {
            "dimensions": _build_dimensions(voxel_sizes_um),
            "position": _build_position(
                tiles_by_channel=tiles_by_channel,
                voxel_sizes_um=voxel_sizes_um,
            ),
            "layers": _build_layers(
                tiles_by_channel=tiles_by_channel,
                s3_modality_path=s3_modality_path,
                voxel_sizes_um=voxel_sizes_um,
                contrast_limits_by_channel=contrast_limits_by_channel,
            ),
            "showAxisLines": False,
            "showScaleBar": True,
        }
    )
    return state


def _select_sample_level(reader, max_sample_voxels: int) -> int:
    """Choose the finest pyramid level that fits a voxel budget.

    Sampling the coarsest level is cheap but repeated mean-downsampling
    averages away bright peaks, collapsing the high percentile toward the
    mean. Instead we pick the *finest* level whose voxel count is within
    ``max_sample_voxels`` so the sampled intensities preserve real signal
    peaks while keeping the read affordable.

    Parameters
    ----------
    reader : ImarisReader
        Open Imaris reader.
    max_sample_voxels : int
        Maximum number of voxels to load for a single level.

    Returns
    -------
    int
        Selected resolution-level index. Falls back to the coarsest
        level if every level exceeds the budget.
    """
    n_levels = reader.n_levels
    coarsest = max(n_levels - 1, 0)
    for level in range(n_levels):
        shape = reader.get_true_shape_for_level(level)
        if int(np.prod(shape)) <= max_sample_voxels:
            return level
    return coarsest


def _sample_intensities(
    imaris_path: str,
    max_sample_voxels: int = 64_000_000,
) -> np.ndarray:
    """Load nonzero intensity samples from a single Imaris tile.

    Reads the finest resolution level that fits ``max_sample_voxels`` and
    returns its nonzero (foreground) voxels flattened. If the level is
    entirely zero, the raw voxels are returned so percentiles are still
    defined.

    Parameters
    ----------
    imaris_path : str
        Path to the ``.ims`` file.
    max_sample_voxels : int
        Voxel budget used to select the resolution level.

    Returns
    -------
    numpy.ndarray
        Flattened intensity samples.
    """
    from aind_exaspim_data_transformation.utils.io_utils import ImarisReader

    with ImarisReader(str(imaris_path)) as reader:
        level = _select_sample_level(reader, max_sample_voxels)
        data_path = (
            f"/DataSet/ResolutionLevel {level}/"
            "TimePoint 0/Channel 0/Data"
        )
        arr = reader.as_array(data_path)

    nonzero = arr[arr > 0]
    return nonzero if nonzero.size else arr.ravel()


def compute_contrast_limits(
    imaris_path: str,
    low_percentile: float = 1.0,
    high_percentile: float = 99.9,
    max_sample_voxels: int = 64_000_000,
) -> Tuple[float, float]:
    """Estimate intensity contrast limits from an Imaris file.

    Samples the finest pyramid level that fits within
    ``max_sample_voxels`` (see :func:`_sample_intensities`) and returns
    percentile-based low/high limits. Background (zero) voxels are
    excluded so the window reflects the signal distribution.

    Parameters
    ----------
    imaris_path : str
        Path to the ``.ims`` file.
    low_percentile : float
        Lower percentile for the contrast window (default 1.0).
    high_percentile : float
        Upper percentile for the contrast window (default 99.9).
    max_sample_voxels : int
        Voxel budget used to select the sampled resolution level.

    Returns
    -------
    tuple of float
        ``(low, high)`` intensity limits. ``high`` is guaranteed to be
        strictly greater than ``low``.
    """
    sample = _sample_intensities(imaris_path, max_sample_voxels)
    low = float(np.percentile(sample, low_percentile))
    high = float(np.percentile(sample, high_percentile))
    if high <= low:
        high = low + 1.0
    return low, high


def _select_representative_tiles(
    tiles: List[Dict],
    max_tiles: int,
) -> List[Dict]:
    """Pick tiles evenly spread across the mosaic.

    Sampling only the first tile biases contrast toward one corner of
    the acquisition (often background or an edge). Selecting tiles at
    evenly spaced indices covers the spatial extent so the pooled
    intensity distribution is representative.

    Parameters
    ----------
    tiles : list of dict
        Tile dicts for one channel, in acquisition order.
    max_tiles : int
        Maximum number of tiles to select. Values ``<= 0`` select all.

    Returns
    -------
    list of dict
        Selected subset of ``tiles``.
    """
    if max_tiles <= 0 or len(tiles) <= max_tiles:
        return list(tiles)
    idxs = np.linspace(0, len(tiles) - 1, max_tiles)
    unique = sorted({int(round(i)) for i in idxs})
    return [tiles[i] for i in unique]


def compute_contrast_limits_by_channel(
    tiles_by_channel: Dict[int, List[Dict]],
    input_source_dir: str,
    low_percentile: float = 1.0,
    high_percentile: float = 99.9,
    max_tiles_per_channel: int = 8,
    max_sample_voxels: int = 64_000_000,
) -> Dict[int, Tuple[float, float]]:
    """Estimate per-channel contrast limits from sampled Imaris tiles.

    For each channel, several tiles spread evenly across the mosaic are
    sampled (see :func:`_select_representative_tiles`), and their
    foreground voxels are pooled into a single distribution before
    computing percentiles. Pooling across tiles and using the finest
    affordable resolution level gives a far more representative contrast
    window than sampling one coarse tile. Tiles that are missing or
    unreadable are skipped; channels with no readable samples are omitted
    from the result so the caller can fall back to a default range.

    Parameters
    ----------
    tiles_by_channel : dict
        Mapping of wavelength (int, nm) → list of tile dicts (each with a
        ``file_name`` key).
    input_source_dir : str
        Directory containing the ``.ims`` tile files.
    low_percentile, high_percentile : float
        Percentiles used for the contrast window.
    max_tiles_per_channel : int
        Maximum number of tiles to sample per channel.
    max_sample_voxels : int
        Per-tile voxel budget used to select the sampled resolution
        level.

    Returns
    -------
    dict
        Mapping of wavelength (int, nm) → ``(low, high)`` limits.
    """
    source_dir = Path(input_source_dir)
    limits: Dict[int, Tuple[float, float]] = {}

    for wavelength_nm, tiles in tiles_by_channel.items():
        selected = _select_representative_tiles(
            tiles, max_tiles_per_channel
        )
        samples: List[np.ndarray] = []
        for tile in selected:
            tile_path = source_dir / tile.get("file_name", "")
            if not tile_path.is_file():
                logger.warning(
                    "Tile %s not found — skipping contrast sampling.",
                    tile_path,
                )
                continue
            try:
                samples.append(
                    _sample_intensities(
                        str(tile_path), max_sample_voxels
                    )
                )
            except (OSError, ValueError, KeyError, RuntimeError) as exc:
                logger.warning(
                    "Failed to sample intensities for %s: %s",
                    tile_path,
                    exc,
                )
        if samples:
            pooled = np.concatenate(samples)
            low = float(np.percentile(pooled, low_percentile))
            high = float(np.percentile(pooled, high_percentile))
            if high <= low:
                high = low + 1.0
            logger.info(
                "Channel %snm: pooled %s voxels from %s tile(s) → "
                "contrast (%.1f, %.1f)",
                wavelength_nm,
                pooled.size,
                len(samples),
                low,
                high,
            )
            limits[wavelength_nm] = (low, high)

    return limits


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
