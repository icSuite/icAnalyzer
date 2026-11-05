"""Native conductance_cs ingestion for the unchanged two-field spline VAE."""

#%% Imports and the native product contract

from pathlib import Path
import re
import warnings

import numpy as np

CONDUCTANCE_ARTIFACT_TYPE = "conductance_vae_run_v1"
EXPECTED_GRID_ID = "image_apex_130km_46x46_v1"
EXPECTED_IMAGE_DIMENSIONS = ("time", "dim1", "dim2")
METHOD_QUALITY_WEIGHT_ORIENTATION_MARKERS = ("auroral-quality", "1 minus")


class _NoFramesRemainError(ValueError):
    """Internal signal for a conductance orbit removed by input selection."""


def _orbit_from_path(path):
    match = re.search(r"or_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def _frame_selection(product, wic_source_interval):
    if wic_source_interval is None:
        return slice(None), None
    start, stop = (int(value) for value in wic_source_interval)
    source_indices = np.asarray(product.source_indices["wic"])
    positions = np.flatnonzero((source_indices >= start) & (source_indices <= stop))
    if positions.size == 0:
        raise ValueError(f"no Product-2 frames map to annotated WIC interval {start}:{stop}")
    selection = slice(int(positions[0]), int(positions[-1]) + 1) if np.all(np.diff(positions) == 1) else positions
    return selection, {
        "wic_source_start": start,
        "wic_source_stop": stop,
        "selected_product_frames": int(positions.size)
    }


def _validated_minimum_sza(minimum_sza):
    if minimum_sza is None:
        return None
    minimum_sza = float(minimum_sza)
    if not np.isfinite(minimum_sza) or not 0 <= minimum_sza <= 180:
        raise ValueError("minimum_sza must be finite and between 0 and 180 degrees")
    return minimum_sza


def _read_sza(product, selection):
    if not hasattr(product, "sza"):
        raise ValueError("conductance_cs is missing the SZA field")
    if product.sza.dimensions != EXPECTED_IMAGE_DIMENSIONS or product.sza.units != "degrees":
        raise ValueError("sza must use IMAGE dimensions and degree units")
    return np.asarray(product.read("sza", selection), dtype=np.float32)


def _apply_sza_frame_selection(product, selection, minimum_sza):
    if minimum_sza is None:
        return selection, None, None
    sza = _read_sza(product, selection)
    eligible_cells = np.isfinite(sza) & (sza >= minimum_sza)
    eligible_frames = np.any(eligible_cells, axis=(1, 2))
    selected_positions = np.arange(int(product.shape[0]))[selection]
    retained_positions = selected_positions[eligible_frames]
    diagnostics = {
        "input_frames": int(sza.shape[0]),
        "retained_frames": int(eligible_frames.sum()),
        "total_cells": int(sza.size),
        "retained_cells": int(eligible_cells.sum())
    }
    if retained_positions.size == 0:
        raise _NoFramesRemainError(f"no frames contain SZA >= {minimum_sza:g} degrees")
    return retained_positions, sza[eligible_frames], diagnostics


def load_conductance_cs(filename, product_loader=None, wic_source_interval=None, minimum_sza=None, include_kp=False):
    """Read H/P and dH/dP in S; use declared auroral quality without inversion."""
    if product_loader is None:
        from icreader import load as product_loader
    minimum_sza = _validated_minimum_sza(minimum_sza)
    filename = Path(filename)
    with product_loader(filename) as product:
        if product.product_type != "conductance_cs" or product.grid_id != EXPECTED_GRID_ID:
            raise ValueError("expected conductance_cs on " + EXPECTED_GRID_ID)
        if product.attrs.get("conductance_model") != "robinson":
            raise ValueError("the direct baseline requires conductance_model='robinson'")
        quality_method = str(product.attrs.get("method_quality_weight_method", "")).lower()
        if not all(marker in quality_method for marker in METHOD_QUALITY_WEIGHT_ORIENTATION_MARKERS):
            raise ValueError("conductance_cs must explicitly declare auroral quality as 1 minus fuvpy weight; refusing ambiguous orientation")
        for name in ("H", "P", "dH", "dP", "coverage", "uncertainty_coverage", "conductance_valid", "conductance_uncertainty_valid", "method_quality_weight"):
            field = getattr(product, name, None)
            if name in ("H", "P", "dH", "dP"):
                units = ("S",)
            elif name in ("conductance_valid", "conductance_uncertainty_valid"):
                units = (None, "1")
            else:
                units = ("1",)
            if field is None or field.dimensions != EXPECTED_IMAGE_DIMENSIONS or field.units not in units:
                raise ValueError(f"{name} must have dimensions {EXPECTED_IMAGE_DIMENSIONS} and units in {units}")
        if minimum_sza is not None and not hasattr(product, "sza"):
            raise ValueError("this conductance_cs product has no SZA field; --minimum-sza cannot be applied without a coordinated companion product")

        selection, annotation = _frame_selection(product, wic_source_interval)
        selection, sza, sza_diagnostics = _apply_sza_frame_selection(product, selection, minimum_sza)
        if np.arange(product.shape[0])[selection].size == 0:
            raise _NoFramesRemainError("no conductance_cs frames remain after selection")
        arrays = {name: np.asarray(product.read(name, selection)) for name in ("H", "P", "dH", "dP", "coverage", "uncertainty_coverage", "conductance_valid", "conductance_uncertainty_valid", "method_quality_weight")}
        values = np.stack((arrays["H"], arrays["P"]), axis=1).astype(np.float32)
        standard_deviation = np.stack((arrays["dH"], arrays["dP"]), axis=1).astype(np.float32)
        quality_weight = arrays["method_quality_weight"]
        finite_weights = quality_weight[np.isfinite(quality_weight)]
        if np.any((finite_weights < 0) | (finite_weights > 1)):
            raise ValueError("method_quality_weight must lie in [0, 1]")
        common = arrays["conductance_valid"].astype(bool) & arrays["conductance_uncertainty_valid"].astype(bool)
        common &= np.all(np.isfinite(values) & np.isfinite(standard_deviation) & (standard_deviation >= 0), axis=1)
        common &= np.isfinite(quality_weight) & np.isfinite(arrays["coverage"]) & (arrays["coverage"] > 0)
        common &= np.isfinite(arrays["uncertainty_coverage"]) & (arrays["uncertainty_coverage"] > 0)
        frame_missing_fraction = np.mean(~common, axis=(1, 2))
        if minimum_sza is not None:
            common &= np.isfinite(sza) & (sza >= minimum_sza)
        valid = np.broadcast_to(common[:, None], values.shape).copy()
        negative_counts = np.count_nonzero(np.isfinite(values) & (values < 0), axis=(0, 2, 3))
        if negative_counts.any():
            warnings.warn(f"upstream conductance_cs contains negative H/P; clipping to zero: {negative_counts.tolist()} cells", stacklevel=2)
        values = np.maximum(values, 0)
        common_quality = quality_weight * np.clip(arrays["coverage"], 0, 1)
        quality = np.broadcast_to(np.where(common, common_quality, 0)[:, None], values.shape).copy().astype(np.float32)
        grid = {name: np.asarray(getattr(product, name), dtype=float).copy() for name in ("mlat", "mlt")}
        metadata = {
            "filename": str(filename.resolve()), "orbit": _orbit_from_path(filename),
            "product_type": "conductance_cs", "grid_id": product.grid_id,
            "schema_version": int(product.schema_version), "software_version": str(product.software_version),
            "conductance_model": "robinson", "manual_frame_selection": annotation,
            "minimum_sza_degrees": minimum_sza, "sza_filter": sza_diagnostics,
            "quality_weight_source": "method_quality_weight_times_conductance_coverage",
            "upstream_quality_weight_interpretation": "auroral_quality_weight",
            "method_quality_weight_method": str(product.attrs["method_quality_weight_method"]),
            "negative_input_policy": "clip_to_zero", "negative_H_P_clipped_cells": negative_counts.tolist()
        }
        for name in ("nonlinear_ordering", "conductance_uncertainty_method", "uncertainty_method", "proton_energy_model", "proton_flux_source", "source_conductance_detector_sha256", "source_precipitation_detector_sha256", "source_fuv_detector_sha256", "count_source", "smoothing_method", "method_quality_weight_spatial_propagation"):
            metadata[name] = str(product.attrs.get(name, "unrecorded"))
        if filename.is_file():
            source_stat = filename.stat()
            metadata.update(file_size_bytes=source_stat.st_size, file_mtime_ns=source_stat.st_mtime_ns)
        result = {
            "values": values, "standard_deviation": standard_deviation,
            "valid": valid, "quality": quality, "frame_missing_fraction": frame_missing_fraction,
            "time": np.asarray(product.time[selection]).copy(), "grid": grid,
            "source_indices": {name: np.asarray(product.source_indices[name][selection]).copy() for name in ("wic", "si12", "si13")},
            "metadata": metadata
        }
        if include_kp:
            field = getattr(product, "Kp", None)
            # icReader eagerly reads Kp into an array; image fields are lazy objects.
            attributes = getattr(product, "variable_attrs", {}).get("Kp", {})
            units = attributes.get("units", getattr(field, "units", None))
            dimensions = getattr(field, "dimensions", ("time",) if isinstance(field, np.ndarray) else None)
            if field is None or dimensions != ("time",) or units != "1":
                raise ValueError("Kp conditioning requires dimensionless per-frame source Kp")
            result["kp"] = np.asarray(product.read("Kp", selection), dtype=np.float32)
            result["kp_interval_start"] = np.asarray(product.Kp_interval_start[selection], dtype="datetime64[ns]").copy()
            times = np.asarray(result["time"], dtype="datetime64[ns]")
            intervals = result["kp_interval_start"]
            if result["kp"].shape != times.shape or intervals.shape != times.shape:
                raise ValueError("Kp and interval times must align with selected images")
            if np.isnat(times).any() or np.isnat(intervals).any() or np.any(times < intervals) or np.any(times >= intervals + np.timedelta64(3, "h")):
                raise ValueError("source Kp interval must contain its image time")
            metadata["kp_provenance"] = dict(product.kp_provenance)
        return result


#%% Concatenate native orbits, rejecting mixed grids and upstream conventions

def load_conductance_cs_files(filenames, wic_source_intervals=None, minimum_sza=None, product_loader=None, include_kp=False):
    products, excluded = [], []
    for filename in filenames:
        orbit = _orbit_from_path(filename)
        if wic_source_intervals is not None and orbit not in wic_source_intervals:
            raise ValueError(f"orbit {orbit} has no manual frame interval")
        interval = None if wic_source_intervals is None else wic_source_intervals[orbit]
        try:
            options = {"include_kp": True} if include_kp else {}
            products.append(load_conductance_cs(filename, product_loader, interval, minimum_sza, **options))
        except _NoFramesRemainError:
            excluded.append(str(filename))
    if not products:
        raise ValueError("no conductance_cs frames remain after selection")
    grid = products[0]["grid"]
    for product in products[1:]:
        for name in grid:
            if not np.allclose(product["grid"][name], grid[name]):
                raise ValueError(f"native input files disagree on grid {name}")
        for name in ("proton_energy_model", "proton_flux_source", "nonlinear_ordering", "method_quality_weight_method"):
            if product["metadata"][name] != products[0]["metadata"][name]:
                raise ValueError(f"native input files disagree on {name}")
    result = {name: np.concatenate([product[name] for product in products]) for name in ("values", "standard_deviation", "valid", "quality", "frame_missing_fraction", "time")}
    result["grid"] = grid
    result["metadata"] = [product["metadata"] for product in products]
    result["excluded_files"] = excluded
    result["source_indices"] = {name: np.concatenate([product["source_indices"][name] for product in products]) for name in ("wic", "si12", "si13")}
    result["orbit"] = np.concatenate([np.full(len(product["values"]), product["metadata"]["orbit"]) for product in products])
    if include_kp:
        for name in ("kp", "kp_interval_start"):
            result[name] = np.concatenate([product[name] for product in products])
    return result


def fit_kp_conditioning(kp):
    """Fit the scalar condition normalization on retained training images only."""
    kp = np.asarray(kp, dtype=float)
    if kp.ndim != 1 or len(kp) < 2 or not np.isfinite(kp).all() or np.any((kp < 0) | (kp > 9)):
        raise ValueError("training Kp must be a finite 0–9 vector with at least two images")
    scale = float(kp.std())
    if scale <= 0:
        raise ValueError("Kp-conditioned training requires more than one Kp value")
    return {"center": float(kp.mean()), "scale": scale, "training_min": float(kp.min()),
            "training_max": float(kp.max()), "embedding_dim": 8}
