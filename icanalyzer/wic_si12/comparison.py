"""Paired access to legacy and rerun WIC/SI12 precipitation products.

Product-2 is the timing and SZA authority. Legacy frames are admitted only
when their product timestamp exactly matches a selected Product-2 timestamp.
"""

#%% Imports

from pathlib import Path

import numpy as np

from .data import (
    EXPECTED_GRID_ID,
    REQUIRED_CAMERA_FIELDS,
    _NoFramesRemainError,
    _apply_sza_frame_selection,
    _frame_selection,
    _materialize_precipitation_cs,
    _orbit_from_path,
    _require_product_contract,
    _validated_minimum_sza,
)


#%% File and time pairing

def pair_orbit_files(current_files, legacy_directory):
    """Pair current orbit paths with same-named legacy files."""

    current_files = [Path(path) for path in current_files]
    legacy_directory = Path(legacy_directory)
    if not legacy_directory.is_dir():
        raise NotADirectoryError(f"legacy comparison path is not a directory: {legacy_directory}")

    legacy_files = sorted(legacy_directory.rglob("or_*.nc"))
    by_name = {}
    for path in legacy_files:
        if path.name in by_name:
            raise ValueError(f"legacy comparison contains duplicate filename {path.name!r}")
        by_name[path.name] = path.resolve()

    pairs = [
        (path.resolve(), by_name[path.name])
        for path in current_files
        if path.name in by_name
    ]
    if not pairs:
        raise FileNotFoundError("current and legacy inputs contain no common orbit filenames")
    return pairs


def _time_key(value):
    if hasattr(value, "year") and hasattr(value, "month"):
        return (
            int(value.year), int(value.month), int(value.day),
            int(value.hour), int(value.minute), int(value.second),
            int(getattr(value, "microsecond", 0)),
        )
    if isinstance(value, np.datetime64):
        return int(value.astype("datetime64[us]").astype(np.int64))
    return str(value)


def exact_time_indices(current_times, legacy_times):
    """Return aligned indices for unique, exactly equal product timestamps."""

    current_keys = [_time_key(value) for value in current_times]
    legacy_keys = [_time_key(value) for value in legacy_times]
    if len(set(current_keys)) != len(current_keys):
        raise ValueError("current product contains duplicate timestamps")
    if len(set(legacy_keys)) != len(legacy_keys):
        raise ValueError("legacy product contains duplicate timestamps")

    legacy_lookup = {key: index for index, key in enumerate(legacy_keys)}
    current_indices = []
    legacy_indices = []
    for current_index, key in enumerate(current_keys):
        if key in legacy_lookup:
            current_indices.append(current_index)
            legacy_indices.append(legacy_lookup[key])
    return (
        np.asarray(current_indices, dtype=int),
        np.asarray(legacy_indices, dtype=int),
    )


#%% Legacy contract and materialization

def _require_legacy_contract(product):
    if product.product_type != "precipitation":
        raise ValueError(f"expected legacy precipitation, got {product.product_type!r}")
    if int(product.schema_version) != 2:
        raise ValueError(f"expected legacy schema 2, got {product.schema_version!r}")
    if product.precipitation_method != "image_ratio":
        raise ValueError("legacy comparison requires precipitation_method='image_ratio'")
    if product.proton_flux_source != "SI12":
        raise ValueError("legacy comparison requires proton_flux_source='SI12'")
    if product.proton_energy_model != "hardy":
        raise ValueError("legacy comparison requires proton_energy_model='hardy'")

    missing = [
        name for name in (*REQUIRED_CAMERA_FIELDS, "w")
        if not hasattr(product, name)
    ]
    if missing:
        raise ValueError("legacy precipitation is missing: " + ", ".join(missing))
    if tuple(product.shape[1:]) != (46, 46):
        raise ValueError(f"legacy product has grid shape {product.shape[1:]}; expected (46, 46)")


def _float_array(values):
    return np.asarray(np.ma.filled(values, np.nan), dtype=np.float32)


def _materialize_legacy(product, filename, selection, sza, current_grid, metadata):
    legacy_grid = {
        "mlat": np.asarray(product.mlat, dtype=np.float64),
        "mlt": np.asarray(product.mlt, dtype=np.float64),
    }
    for coordinate in ("mlat", "mlt"):
        if not np.allclose(legacy_grid[coordinate], current_grid[coordinate]):
            raise ValueError(f"legacy and current products disagree on grid {coordinate}")

    arrays = {
        name: _float_array(getattr(product, name)[selection])
        for name in REQUIRED_CAMERA_FIELDS
    }
    robust_weight = _float_array(product.w[selection])
    wic = arrays["wic_corrected"]
    si12 = arrays["si12"]
    dwic = arrays["dwic_corrected"]
    dsi12 = arrays["dsi12"]
    negative_wic = np.isfinite(wic) & (wic < 0)
    negative_si12 = np.isfinite(si12) & (si12 < 0)
    wic = np.maximum(wic, 0.0)
    si12 = np.maximum(si12, 0.0)

    common_finite = (
        np.isfinite(wic)
        & np.isfinite(arrays["si13_corrected"])
        & np.isfinite(si12)
        & np.isfinite(dwic)
        & np.isfinite(arrays["dsi13_corrected"])
        & np.isfinite(dsi12)
    )
    valid_pixels = common_finite & (dwic >= 0) & (dsi12 >= 0)
    if sza is not None:
        valid_pixels &= np.isfinite(sza) & (sza >= metadata["minimum_sza_degrees"])
    metadata["wic_valid_input_cells"] = int(np.count_nonzero(valid_pixels))
    metadata["si12_valid_input_cells"] = int(np.count_nonzero(valid_pixels))
    metadata["negative_wic_clipped_valid_cells"] = int(
        np.count_nonzero(negative_wic & valid_pixels)
    )
    metadata["negative_si12_clipped_valid_cells"] = int(
        np.count_nonzero(negative_si12 & valid_pixels)
    )

    valid = np.stack((valid_pixels, valid_pixels), axis=1)
    auroral_weight = np.clip(1.0 - robust_weight, 0.0, 1.0)
    quality = np.broadcast_to(auroral_weight[:, None], valid.shape).copy()
    quality = np.where(valid & np.isfinite(quality), quality, 0.0)
    times = np.asarray(product.time[selection], dtype=object).copy()
    source_indices = {
        sensor: np.asarray(product.source_indices[sensor][selection]).copy()
        for sensor in ("wic", "si12", "si13")
    }

    metadata.update({
        "filename": str(filename),
        "orbit": _orbit_from_path(filename),
        "grid_id": EXPECTED_GRID_ID,
        "source_shape": tuple(int(value) for value in product.shape),
        "shape": tuple(int(value) for value in valid_pixels.shape),
        "schema_version": int(product.schema_version),
        "proton_flux_source": str(product.proton_flux_source),
        "proton_energy_model": str(product.proton_energy_model),
        "legacy_precipitation_method": str(product.precipitation_method),
        "quality_weight_source": "one_minus_legacy_combined_w",
        "upstream_quality_weight_interpretation": "background_fit_weight",
        "negative_input_policy": "clip_to_zero",
        "finite_wic_input_cells": int(np.count_nonzero(np.isfinite(wic))),
        "finite_si12_input_cells": int(np.count_nonzero(np.isfinite(si12))),
        "negative_wic_clipped_cells": int(np.count_nonzero(negative_wic)),
        "negative_si12_clipped_cells": int(np.count_nonzero(negative_si12)),
        "source_time_proxy": "matched_product_time",
    })
    return {
        "values": np.stack((wic, si12), axis=1),
        "standard_deviation": np.stack((dwic, dsi12), axis=1),
        "valid": valid,
        "quality": quality,
        "si13_corrected": arrays["si13_corrected"],
        "dsi13_corrected": arrays["dsi13_corrected"],
        "time": times,
        "grid": legacy_grid,
        "source_indices": source_indices,
        "source_times": {sensor: times.copy() for sensor in ("wic", "si12", "si13")},
        "frame_quality": {
            sensor: np.full(times.shape, -1, dtype=np.int8)
            for sensor in ("wic", "si12", "si13")
        },
        "metadata": metadata,
    }


#%% Paired loading

def load_paired_orbit(
    current_filename,
    legacy_filename,
    data_version="both",
    wic_source_interval=None,
    minimum_sza=None,
    current_loader=None,
    legacy_loader=None,
):
    """Load exact-time-paired fields from one current/legacy orbit pair."""

    if data_version not in ("current", "legacy", "both"):
        raise ValueError("data_version must be 'current', 'legacy', or 'both'")
    minimum_sza = _validated_minimum_sza(minimum_sza)
    if current_loader is None or legacy_loader is None:
        from icreader import load
        current_loader = load if current_loader is None else current_loader
        legacy_loader = load if legacy_loader is None else legacy_loader

    current_filename = Path(current_filename)
    legacy_filename = Path(legacy_filename)
    legacy = legacy_loader(legacy_filename)
    _require_legacy_contract(legacy)

    with current_loader(current_filename) as current:
        _require_product_contract(current)
        current_selection, annotation = _frame_selection(current, wic_source_interval)
        current_selection, sza, sza_diagnostics = _apply_sza_frame_selection(
            current, current_selection, minimum_sza
        )
        current_times = np.asarray(current.time[current_selection], dtype=object)
        current_indices, legacy_selection = exact_time_indices(current_times, legacy.time)
        if current_indices.size == 0:
            raise _NoFramesRemainError("no exact product timestamps are shared by the orbit pair")

        current_selection = np.asarray(current_selection)[current_indices]
        if sza is not None:
            sza = sza[current_indices]
        current_grid = {
            "mlat": np.asarray(current.mlat, dtype=np.float64),
            "mlt": np.asarray(current.mlt, dtype=np.float64),
        }
        pairing = {
            "comparison_mode": "exact_product_time",
            "comparison_data_version": data_version,
            "paired_current_filename": str(current_filename),
            "paired_legacy_filename": str(legacy_filename),
            "current_selected_frames_before_pairing": int(current_times.size),
            "legacy_total_frames": int(len(legacy.time)),
            "paired_frames": int(current_indices.size),
            "minimum_sza_degrees": minimum_sza,
            "manual_frame_selection": annotation,
        }

        result = {}
        if data_version in ("current", "both"):
            current_product = _materialize_precipitation_cs(
                current,
                current_filename,
                current_selection,
                annotation,
                minimum_sza,
                sza,
                sza_diagnostics,
            )
            current_product["metadata"].update(pairing)
            current_product["metadata"]["comparison_data_version"] = "current"
            result["current"] = current_product

        if data_version in ("legacy", "both"):
            legacy_metadata = dict(pairing)
            legacy_metadata["comparison_data_version"] = "legacy"
            result["legacy"] = _materialize_legacy(
                legacy,
                legacy_filename,
                legacy_selection,
                sza,
                current_grid,
                legacy_metadata,
            )

    if data_version == "both":
        return result
    return result[data_version]


def _concatenate_products(products, excluded_files):
    if not products:
        raise ValueError("no paired frames remain after input selection")

    grid = products[0]["grid"]
    for product in products[1:]:
        for coordinate in ("mlat", "mlt"):
            if not np.allclose(product["grid"][coordinate], grid[coordinate]):
                raise ValueError(f"paired input files disagree on grid {coordinate}")

    result = {
        name: np.concatenate([product[name] for product in products], axis=0)
        for name in (
            "values", "standard_deviation", "valid", "quality",
            "si13_corrected", "dsi13_corrected", "time",
        )
    }
    result["metadata"] = [product["metadata"] for product in products]
    result["excluded_files"] = excluded_files
    result["grid"] = grid
    for group in ("source_indices", "source_times", "frame_quality"):
        result[group] = {
            sensor: np.concatenate([product[group][sensor] for product in products])
            for sensor in ("wic", "si12", "si13")
        }
    result["orbit"] = np.concatenate([
        np.full(product["values"].shape[0], product["metadata"]["orbit"], dtype=int)
        for product in products
    ])
    return result


def load_paired_files(
    pairs,
    data_version,
    wic_source_intervals=None,
    minimum_sza=None,
    current_loader=None,
    legacy_loader=None,
):
    """Load one side of several pre-matched orbit pairs."""

    products = []
    excluded_files = []
    for current_path, legacy_path in pairs:
        orbit = _orbit_from_path(current_path)
        if wic_source_intervals is None:
            interval = None
        else:
            if orbit not in wic_source_intervals:
                raise ValueError(f"orbit {orbit} has no manual frame interval")
            interval = wic_source_intervals[orbit]
        try:
            product = load_paired_orbit(
                current_path,
                legacy_path,
                data_version=data_version,
                wic_source_interval=interval,
                minimum_sza=minimum_sza,
                current_loader=current_loader,
                legacy_loader=legacy_loader,
            )
        except _NoFramesRemainError:
            excluded_files.append(str(Path(current_path)))
            continue
        products.append(product)
    return _concatenate_products(products, excluded_files)
