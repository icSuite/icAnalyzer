"""Load and transform corrected-WIC/SI12 fields from precipitation_cs.

The loader deliberately validates a stronger contract than icReader schema 1:
the new SI12 fields are optional in icReader for backward compatibility, but
they are mandatory for this VAE.  Corrected WIC is never proton corrected in
this package; that operation already happened in icBuilder Product 2.
"""

#%% Imports

import csv
from pathlib import Path
import re

import numpy as np
import torch
from torch.utils.data import Dataset


#%% Product contract

EXPECTED_GRID_ID = "image_apex_130km_46x46_v1"
EXPECTED_IMAGE_DIMENSIONS = ("time", "dim1", "dim2")
REQUIRED_CAMERA_FIELDS = (
    "wic_corrected",
    "dwic_corrected",
    "si13_corrected",
    "dsi13_corrected",
    "si12",
    "dsi12",
)


class _NoFramesRemainError(ValueError):
    """Internal signal for an orbit removed entirely by input selection."""


def _orbit_from_path(path):
    match = re.search(r"or_(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def load_good_orbit_intervals(filename):
    """Read inclusive manually accepted WIC-frame intervals from the CSV.

    The historical annotation convention treats a blank good start as frame
    zero and a blank or zero good stop as no accepted interval.
    """

    intervals = {}
    with Path(filename).open(newline="", encoding="utf-8") as stream:
        for line_number, row in enumerate(csv.reader(stream), start=1):
            if not row or all(not value.strip() for value in row):
                continue
            if len(row) != 5:
                raise ValueError(
                    f"annotation row {line_number} has {len(row)} columns; expected 5"
                )
            try:
                orbit = int(row[0])
                start = int(float(row[3])) if row[3].strip() else 0
                stop = int(float(row[4])) if row[4].strip() else None
            except ValueError as error:
                raise ValueError(
                    f"annotation row {line_number} contains a nonnumeric value"
                ) from error

            if stop in (None, 0):
                continue
            if start < 0 or stop < start:
                raise ValueError(
                    f"annotation row {line_number} has invalid interval {start}:{stop}"
                )
            if orbit in intervals:
                raise ValueError(f"orbit {orbit} occurs more than once in annotations")
            intervals[orbit] = (start, stop)

    if not intervals:
        raise ValueError("annotation file contains no accepted orbit intervals")
    return intervals


def select_annotated_orbit_files(filenames, intervals):
    """Return files whose orbit has one manually accepted interval."""

    selected = []
    for path in filenames:
        orbit = _orbit_from_path(path)
        if orbit in intervals:
            selected.append(Path(path))
    if not selected:
        raise FileNotFoundError(
            "none of the input files has a manually accepted orbit interval"
        )
    return selected


def _frame_selection(product, wic_source_interval):
    if wic_source_interval is None:
        return slice(None), None

    start, stop = (int(value) for value in wic_source_interval)
    source_indices = np.asarray(product.source_indices["wic"])
    positions = np.flatnonzero((source_indices >= start) & (source_indices <= stop))
    if positions.size == 0:
        raise ValueError(
            f"no Product-2 frames map to annotated WIC interval {start}:{stop}"
        )
    if np.all(np.diff(positions) == 1):
        selection = slice(int(positions[0]), int(positions[-1]) + 1)
    else:
        selection = positions
    return selection, {
        "wic_source_start": start,
        "wic_source_stop": stop,
        "selected_product_frames": int(positions.size),
    }


def _require_product_contract(product):
    if product.product_type != "precipitation_cs":
        raise ValueError(
            f"expected precipitation_cs, got {product.product_type!r}"
        )
    if product.grid_id != EXPECTED_GRID_ID:
        raise ValueError(
            f"expected grid {EXPECTED_GRID_ID!r}, got {product.grid_id!r}"
        )

    missing = [name for name in REQUIRED_CAMERA_FIELDS if not hasattr(product, name)]
    if missing:
        raise ValueError(
            "precipitation_cs is missing VAE fields: " + ", ".join(missing)
        )

    for name in REQUIRED_CAMERA_FIELDS:
        field = getattr(product, name)
        if field.dimensions != EXPECTED_IMAGE_DIMENSIONS:
            raise ValueError(
                f"{name} dimensions are {field.dimensions}; "
                f"expected {EXPECTED_IMAGE_DIMENSIONS}"
            )
        if field.units != "counts":
            raise ValueError(f"{name} units are {field.units!r}; expected 'counts'")

    if product.proton_flux_source != "SI12":
        raise ValueError(
            f"proton_flux_source is {product.proton_flux_source!r}; expected 'SI12'"
        )
    if product.proton_energy_model != "hardy":
        raise ValueError(
            "the accepted Product-2 baseline requires "
            "proton_energy_model='hardy'"
        )


def _validated_minimum_sza(minimum_sza):
    if minimum_sza is None:
        return None
    minimum_sza = float(minimum_sza)
    if not np.isfinite(minimum_sza) or not 0.0 <= minimum_sza <= 180.0:
        raise ValueError("minimum_sza must be finite and between 0 and 180 degrees")
    return minimum_sza


def _read_sza(product, selection):
    if not hasattr(product, "sza"):
        raise ValueError("precipitation_cs is missing the SZA field")
    if product.sza.dimensions != EXPECTED_IMAGE_DIMENSIONS:
        raise ValueError(
            f"sza dimensions are {product.sza.dimensions}; "
            f"expected {EXPECTED_IMAGE_DIMENSIONS}"
        )
    if product.sza.units != "degrees":
        raise ValueError(f"sza units are {product.sza.units!r}; expected 'degrees'")
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
        "retained_frames": int(np.count_nonzero(eligible_frames)),
        "total_cells": int(sza.size),
        "retained_cells": int(np.count_nonzero(eligible_cells)),
    }
    if retained_positions.size == 0:
        raise _NoFramesRemainError(
            f"no frames contain SZA >= {minimum_sza:g} degrees"
        )
    return retained_positions, sza[eligible_frames], diagnostics


def _materialize_precipitation_cs(
    product,
    filename,
    selection,
    annotation,
    minimum_sza,
    sza,
    sza_diagnostics,
):
    """Read one already-validated current product at selected frame positions."""

    arrays = {
        name: np.asarray(product.read(name, selection), dtype=np.float32)
        for name in REQUIRED_CAMERA_FIELDS
    }
    method_valid = np.asarray(
        product.read("method_valid", selection), dtype=bool
    )
    method_quality = np.asarray(
        product.read("method_quality_weight", selection), dtype=np.float32
    )
    wic_coverage = np.asarray(
        product.read("wic_coverage", selection), dtype=np.float32
    )
    si12_coverage = np.asarray(
        product.read("si12_coverage", selection), dtype=np.float32
    )
    metadata = {
        "filename": str(filename),
        "orbit": _orbit_from_path(filename),
        "grid_id": str(product.grid_id),
        "source_shape": tuple(int(value) for value in product.shape),
        "shape": tuple(int(value) for value in method_valid.shape),
        "schema_version": int(product.schema_version),
        "software_version": str(product.software_version),
        "proton_flux_source": str(product.proton_flux_source),
        "proton_energy_model": str(product.proton_energy_model),
        "proton_energy_uncertainty_method": str(
            product.attrs.get("proton_energy_uncertainty_method", "unknown")
        ),
        "proton_energy_coordinate_note": str(
            product.attrs.get("proton_energy_coordinate_note", "unknown")
        ),
        "proton_operation_order": str(
            product.attrs.get("proton_operation_order", "unknown")
        ),
        "count_source": str(product.attrs.get("count_source", "unknown")),
        "spatial_smoothing_kernel": str(
            product.attrs.get("spatial_smoothing_kernel", "unknown")
        ),
        "source_precipitation_detector_sha256": str(
            product.attrs.get("source_precipitation_detector_sha256", "")
        ),
        "source_fuv_detector_sha256": str(
            product.attrs.get("source_fuv_detector_sha256", "")
        ),
        "manual_frame_selection": annotation,
        "minimum_sza_degrees": minimum_sza,
        "quality_weight_source": (
            "one_minus_method_quality_weight_times_channel_coverage"
        ),
        "upstream_quality_weight_interpretation": "background_fit_weight",
    }
    if sza_diagnostics is not None:
        metadata.update({
            "sza_filter_input_frames": sza_diagnostics["input_frames"],
            "sza_filter_retained_frames": sza_diagnostics["retained_frames"],
            "sza_filter_retained_cells": sza_diagnostics["retained_cells"],
            "sza_filter_total_cells": sza_diagnostics["total_cells"],
        })
    times = np.asarray(product.time[selection], dtype=object).copy()
    grid = {
        "mlat": np.asarray(product.mlat, dtype=np.float64).copy(),
        "mlt": np.asarray(product.mlt, dtype=np.float64).copy(),
    }
    source_indices = {
        sensor: np.asarray(product.source_indices[sensor][selection]).copy()
        for sensor in ("wic", "si12", "si13")
    }
    source_times = {
        sensor: np.asarray(
            product.source_times[sensor][selection], dtype=object
        ).copy()
        for sensor in ("wic", "si12", "si13")
    }
    frame_quality = {
        sensor: np.asarray(product.frame_quality[sensor][selection]).copy()
        for sensor in ("wic", "si12", "si13")
    }

    wic = arrays["wic_corrected"]
    si12 = arrays["si12"]
    dwic = arrays["dwic_corrected"]
    dsi12 = arrays["dsi12"]

    negative_wic = np.isfinite(wic) & (wic < 0)
    negative_si12 = np.isfinite(si12) & (si12 < 0)
    wic = np.maximum(wic, 0.0)
    si12 = np.maximum(si12, 0.0)
    metadata["negative_input_policy"] = "clip_to_zero"
    metadata["finite_wic_input_cells"] = int(np.count_nonzero(np.isfinite(wic)))
    metadata["finite_si12_input_cells"] = int(np.count_nonzero(np.isfinite(si12)))
    metadata["negative_wic_clipped_cells"] = int(np.count_nonzero(negative_wic))
    metadata["negative_si12_clipped_cells"] = int(np.count_nonzero(negative_si12))

    common_finite = (
        np.isfinite(wic)
        & np.isfinite(si12)
        & np.isfinite(dwic)
        & np.isfinite(dsi12)
    )
    nonnegative_uncertainty = (dwic >= 0) & (dsi12 >= 0)
    base_valid = (
        method_valid
        & common_finite
        & nonnegative_uncertainty
        & np.isfinite(method_quality)
    )
    frame_common_valid = (
        base_valid & (wic_coverage > 0) & (si12_coverage > 0)
    )
    frame_missing_fraction = np.mean(~frame_common_valid, axis=(1, 2))
    if minimum_sza is not None:
        base_valid &= np.isfinite(sza) & (sza >= minimum_sza)
    wic_valid = base_valid & (wic_coverage > 0)
    si12_valid = base_valid & (si12_coverage > 0)
    metadata["wic_valid_input_cells"] = int(np.count_nonzero(wic_valid))
    metadata["si12_valid_input_cells"] = int(np.count_nonzero(si12_valid))
    metadata["negative_wic_clipped_valid_cells"] = int(
        np.count_nonzero(negative_wic & wic_valid)
    )
    metadata["negative_si12_clipped_valid_cells"] = int(
        np.count_nonzero(negative_si12 & si12_valid)
    )

    values = np.stack((wic, si12), axis=1)
    standard_deviation = np.stack((dwic, dsi12), axis=1)
    coverage = np.stack((wic_coverage, si12_coverage), axis=1)
    valid = np.stack(
        (wic_valid, si12_valid),
        axis=1,
    )
    auroral_weight = np.clip(1.0 - method_quality[:, None, :, :], 0.0, 1.0)
    quality = auroral_weight * np.clip(coverage, 0.0, 1.0)
    quality = np.where(valid & np.isfinite(quality), quality, 0.0)

    return {
        "values": values,
        "standard_deviation": standard_deviation,
        "valid": valid,
        "quality": quality,
        "frame_missing_fraction": frame_missing_fraction,
        "si13_corrected": arrays["si13_corrected"],
        "dsi13_corrected": arrays["dsi13_corrected"],
        "time": times,
        "grid": grid,
        "source_indices": source_indices,
        "source_times": source_times,
        "frame_quality": frame_quality,
        "metadata": metadata,
    }


def load_precipitation_cs(
    filename,
    product_loader=None,
    wic_source_interval=None,
    minimum_sza=None,
):
    """Materialize the VAE fields from one Product-2 Cubed-Sphere file.

    Parameters
    ----------
    filename : path-like
        A schema-1 ``precipitation_cs`` file containing the coordinated camera
        fields.
    product_loader : callable, optional
        Dependency-injection hook used by focused tests.  The default is
        ``icreader.load``.
    wic_source_interval : pair of int, optional
        Inclusive manually accepted WIC source-frame bounds. Product frames
        are selected through ``wic_source_index`` before camera arrays are
        materialized.
    minimum_sza : float, optional
        Retain only grid cells whose solar zenith angle is at least this many
        degrees. Frames with no remaining weighted cells are later removed by
        ``WICSI12ArrayDataset``.
    """

    minimum_sza = _validated_minimum_sza(minimum_sza)
    if product_loader is None:
        from icreader import load as product_loader

    filename = Path(filename)
    with product_loader(filename) as product:
        _require_product_contract(product)
        selection, annotation = _frame_selection(product, wic_source_interval)
        selection, sza, sza_diagnostics = _apply_sza_frame_selection(
            product, selection, minimum_sza
        )

        return _materialize_precipitation_cs(
            product,
            filename,
            selection,
            annotation,
            minimum_sza,
            sza,
            sza_diagnostics,
        )


def load_precipitation_cs_files(
    filenames,
    wic_source_intervals=None,
    minimum_sza=None,
    product_loader=None,
):
    """Load and concatenate several orbit files with one verified grid."""

    products = []
    excluded_files = []
    for path in filenames:
        orbit = _orbit_from_path(path)
        if wic_source_intervals is None:
            interval = None
        else:
            if orbit not in wic_source_intervals:
                raise ValueError(f"orbit {orbit} has no manual frame interval")
            interval = wic_source_intervals[orbit]
        try:
            product = load_precipitation_cs(
                path,
                product_loader=product_loader,
                wic_source_interval=interval,
                minimum_sza=minimum_sza,
            )
        except _NoFramesRemainError:
            excluded_files.append(str(Path(path)))
            continue
        products.append(product)
    if not products:
        raise ValueError("no precipitation_cs frames remain after input selection")

    grid_ids = {product["metadata"]["grid_id"] for product in products}
    if len(grid_ids) != 1:
        raise ValueError(f"input files use multiple grids: {sorted(grid_ids)}")

    grid = products[0]["grid"]
    for product in products[1:]:
        for coordinate in ("mlat", "mlt"):
            if not np.allclose(product["grid"][coordinate], grid[coordinate]):
                raise ValueError(f"input files disagree on grid {coordinate}")

    result = {}
    for name in (
        "values", "standard_deviation", "valid", "quality",
        "frame_missing_fraction", "si13_corrected", "dsi13_corrected", "time",
    ):
        result[name] = np.concatenate([product[name] for product in products], axis=0)

    result["metadata"] = [product["metadata"] for product in products]
    result["excluded_files"] = excluded_files
    result["grid"] = grid
    for group in ("source_indices", "source_times", "frame_quality"):
        result[group] = {
            sensor: np.concatenate(
                [product[group][sensor] for product in products], axis=0
            )
            for sensor in ("wic", "si12", "si13")
        }
    orbit_values = []
    for product in products:
        orbit_values.extend(
            [product["metadata"]["orbit"]] * product["values"].shape[0]
        )
    result["orbit"] = np.asarray(orbit_values)
    return result


def split_orbit_files(filenames, validation_fraction=0.15, test_fraction=0.15, seed=0):
    """Split complete orbit files reproducibly without frame leakage."""

    paths = [Path(path) for path in filenames]
    if not paths:
        raise ValueError("at least one orbit file is required")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")

    orbit_path_pairs = []
    for path in paths:
        orbit = _orbit_from_path(path)
        if orbit is None:
            raise ValueError(f"cannot determine orbit from {path.name!r}")
        orbit_path_pairs.append((orbit, path))
    if len({orbit for orbit, _ in orbit_path_pairs}) != len(orbit_path_pairs):
        raise ValueError("each orbit must appear in exactly one input file")

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(paths))
    shuffled = [orbit_path_pairs[index][1] for index in order]

    n_files = len(shuffled)
    n_test = int(np.floor(n_files * test_fraction))
    n_validation = int(np.floor(n_files * validation_fraction))
    if test_fraction > 0 and n_files >= 3:
        n_test = max(1, n_test)
    if validation_fraction > 0 and n_files - n_test >= 2:
        n_validation = max(1, n_validation)

    return {
        "test": shuffled[:n_test],
        "validation": shuffled[n_test:n_test + n_validation],
        "train": shuffled[n_test + n_validation:],
    }


#%% Channel transformation

class ChannelTransform:
    """Versioned channel transform followed by channel-wise standardization."""

    ASINH_KIND = "channel_asinh_standardization_v1"
    LOG10_KIND = "channel_log10p_standardization_v1"

    def __init__(self, scale, center, spread, kind=ASINH_KIND):
        self.scale = np.asarray(scale, dtype=np.float32)
        self.center = np.asarray(center, dtype=np.float32)
        self.spread = np.asarray(spread, dtype=np.float32)
        self.kind = str(kind)
        if self.scale.shape != (2,) or self.center.shape != (2,) or self.spread.shape != (2,):
            raise ValueError("scale, center, and spread must each have two values")
        if np.any(self.scale <= 0) or np.any(self.spread <= 0):
            raise ValueError("scale and spread must be positive")
        if self.kind not in (self.ASINH_KIND, self.LOG10_KIND):
            raise ValueError(f"unsupported channel transform {self.kind!r}")

    def transform(self, values):
        if torch.is_tensor(values):
            scale = torch.as_tensor(self.scale, device=values.device, dtype=values.dtype)
            center = torch.as_tensor(self.center, device=values.device, dtype=values.dtype)
            spread = torch.as_tensor(self.spread, device=values.device, dtype=values.dtype)
            shape = (1, 2) + (1,) * (values.ndim - 2)
            if self.kind == self.LOG10_KIND:
                if torch.any(values < 0):
                    raise ValueError("log10 channel transform requires nonnegative values")
                transformed = torch.log10(values + 1.0)
            else:
                transformed = torch.asinh(values / scale.reshape(shape))
            return (
                transformed - center.reshape(shape)
            ) / spread.reshape(shape)

        values = np.asarray(values)
        shape = (1, 2) + (1,) * (values.ndim - 2)
        if self.kind == self.LOG10_KIND:
            if np.any(values < 0):
                raise ValueError("log10 channel transform requires nonnegative values")
            transformed = np.log10(values + 1.0)
        else:
            transformed = np.arcsinh(values / self.scale.reshape(shape))
        return (
            transformed - self.center.reshape(shape)
        ) / self.spread.reshape(shape)

    def inverse(self, transformed):
        if torch.is_tensor(transformed):
            scale = torch.as_tensor(self.scale, device=transformed.device, dtype=transformed.dtype)
            center = torch.as_tensor(self.center, device=transformed.device, dtype=transformed.dtype)
            spread = torch.as_tensor(self.spread, device=transformed.device, dtype=transformed.dtype)
            shape = (1, 2) + (1,) * (transformed.ndim - 2)
            restored = transformed * spread.reshape(shape) + center.reshape(shape)
            if self.kind == self.LOG10_KIND:
                return torch.clamp(torch.pow(10.0, restored) - 1.0, min=0.0)
            return scale.reshape(shape) * torch.sinh(restored)

        transformed = np.asarray(transformed)
        shape = (1, 2) + (1,) * (transformed.ndim - 2)
        restored = transformed * self.spread.reshape(shape) + self.center.reshape(shape)
        if self.kind == self.LOG10_KIND:
            return np.maximum(np.power(10.0, restored) - 1.0, 0.0)
        return self.scale.reshape(shape) * np.sinh(restored)

    def variance(self, values, standard_deviation):
        values = np.asarray(values, dtype=np.float32)
        standard_deviation = np.asarray(standard_deviation, dtype=np.float32)
        shape = (1, 2) + (1,) * (values.ndim - 2)
        scale = self.scale.reshape(shape)
        spread = self.spread.reshape(shape)
        if self.kind == self.LOG10_KIND:
            if np.any(values < 0):
                raise ValueError("log10 variance transform requires nonnegative values")
            derivative = 1.0 / ((values + 1.0) * np.log(10.0) * spread)
        else:
            derivative = 1.0 / (
                scale * np.sqrt(1.0 + (values / scale)**2) * spread
            )
        return (standard_deviation * derivative)**2

    def to_dict(self):
        return {
            "kind": self.kind,
            "scale": self.scale.tolist(),
            "center": self.center.tolist(),
            "spread": self.spread.tolist(),
        }

    @classmethod
    def from_dict(cls, values):
        return cls(
            values["scale"], values["center"], values["spread"],
            kind=values.get("kind", cls.ASINH_KIND)
        )


def fit_channel_transform(values, valid):
    """Fit the legacy log10(value + 1) transform using training data only."""

    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.shape != valid.shape or values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values and valid must have shape (sample, 2, y, x)")
    if np.any(values[valid] < 0):
        raise ValueError(
            "log10 channel transform received negative values; "
            "clip them at the ingestion boundary"
        )

    scale = np.empty(2, dtype=np.float32)
    center = np.empty(2, dtype=np.float32)
    spread = np.empty(2, dtype=np.float32)
    for channel in range(2):
        selected = values[:, channel][valid[:, channel]]
        if selected.size == 0:
            raise ValueError(f"channel {channel} has no valid training values")
        scale[channel] = 1.0
        transformed = np.log10(selected + 1.0)
        center[channel] = np.mean(transformed)
        spread[channel] = np.std(transformed)
        if not np.isfinite(spread[channel]) or spread[channel] <= 0:
            raise ValueError(f"channel {channel} has zero transformed spread")
    return ChannelTransform(
        scale, center, spread, kind=ChannelTransform.LOG10_KIND
    )


def fit_precision_weighting(
    product_data,
    transform,
    variance_floor=1e-3,
    maximum_relative_precision=100.0,
    maximum_scale_samples=2_000_000,
):
    """Fit one transformed-space precision scale per channel on training data.

    The scale is the median valid precision from a deterministic frame sample.
    It is fitted once, rather than independently for every image, so absolute
    uncertainty differences between training frames remain in the loss.
    """

    values = np.asarray(product_data["values"], dtype=np.float32)
    standard_deviation = np.asarray(
        product_data["standard_deviation"], dtype=np.float32
    )
    valid = np.asarray(product_data["valid"], dtype=bool)
    quality = np.asarray(product_data["quality"], dtype=np.float32)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("training values must have shape (sample, 2, y, x)")
    for name, array in (
        ("standard_deviation", standard_deviation),
        ("valid", valid),
        ("quality", quality),
    ):
        if array.shape != values.shape:
            raise ValueError(f"{name} shape {array.shape} does not match {values.shape}")

    variance_floor = float(variance_floor)
    maximum_relative_precision = float(maximum_relative_precision)
    maximum_scale_samples = int(maximum_scale_samples)
    if not np.isfinite(variance_floor) or variance_floor <= 0:
        raise ValueError("variance_floor must be finite and positive")
    if not np.isfinite(maximum_relative_precision) or maximum_relative_precision <= 0:
        raise ValueError("maximum_relative_precision must be finite and positive")
    if maximum_scale_samples <= 0:
        raise ValueError("maximum_scale_samples must be positive")

    possible_values_per_channel = int(values.shape[0] * np.prod(values.shape[2:]))
    frame_stride = max(
        1,
        int(np.ceil(possible_values_per_channel / maximum_scale_samples)),
    )
    selection = slice(None, None, frame_stride)
    sampled_variance = transform.variance(
        values[selection], standard_deviation[selection]
    )
    sampled_valid = valid[selection]
    sampled_quality = quality[selection]

    precision_scale = np.empty(2, dtype=np.float32)
    scale_sample_count = []
    relative_precision_diagnostics = []
    for channel in range(2):
        support = (
            sampled_valid[:, channel]
            & np.isfinite(sampled_quality[:, channel])
            & (sampled_quality[:, channel] > 0)
            & np.isfinite(sampled_variance[:, channel])
            & (sampled_variance[:, channel] >= 0)
        )
        variance = sampled_variance[:, channel][support]
        if variance.size == 0:
            raise ValueError(f"channel {channel} has no valid uncertainty samples")
        precision = 1.0 / np.maximum(variance, variance_floor)
        precision_scale[channel] = np.median(precision)
        if not np.isfinite(precision_scale[channel]) or precision_scale[channel] <= 0:
            raise ValueError(f"channel {channel} has invalid fitted precision scale")
        scale_sample_count.append(int(precision.size))
        relative_precision = precision / precision_scale[channel]
        clipped_precision = np.minimum(
            relative_precision, maximum_relative_precision
        )
        relative_precision_diagnostics.append({
            "median": float(np.median(relative_precision)),
            "mean_before_cap": float(np.mean(relative_precision)),
            "mean_after_cap": float(np.mean(clipped_precision)),
            "fraction_capped": float(np.mean(
                relative_precision > maximum_relative_precision
            )),
        })

    return {
        "kind": "training_median_precision_v1",
        "precision_scale": precision_scale.tolist(),
        "variance_floor": variance_floor,
        "maximum_relative_precision": maximum_relative_precision,
        "maximum_scale_samples": maximum_scale_samples,
        "frame_stride": frame_stride,
        "scale_sample_count": scale_sample_count,
        "relative_precision_diagnostics": relative_precision_diagnostics,
        "loss_reduction": "mean_valid_support_per_sample_channel",
    }


#%% Torch dataset

class WICSI12ArrayDataset(Dataset):
    """Torch dataset for an already materialized orbit split."""

    def __init__(self, product_data, transform, maximum_missing_fraction=0.33):
        self.values = np.asarray(product_data["values"], dtype=np.float32)
        self.standard_deviation = np.asarray(
            product_data["standard_deviation"], dtype=np.float32
        )
        self.valid = np.asarray(product_data["valid"], dtype=bool)
        self.quality = np.asarray(product_data["quality"], dtype=np.float32)
        self.transform = transform
        self.maximum_missing_fraction = float(maximum_missing_fraction)
        if not 0 <= self.maximum_missing_fraction <= 1:
            raise ValueError("maximum_missing_fraction must be between zero and one")

        expected = self.values.shape
        for name, array in (
            ("standard_deviation", self.standard_deviation),
            ("valid", self.valid),
            ("quality", self.quality),
        ):
            if array.shape != expected:
                raise ValueError(f"{name} shape {array.shape} does not match {expected}")

        common_valid = np.all(self.valid, axis=1, keepdims=True)
        missing_fraction = np.asarray(
            product_data.get(
                "frame_missing_fraction",
                np.mean(~common_valid, axis=(1, 2, 3)),
            ),
            dtype=np.float64,
        )
        if missing_fraction.shape != (self.values.shape[0],):
            raise ValueError("frame_missing_fraction must contain one value per frame")
        usable = missing_fraction <= self.maximum_missing_fraction
        self.selected_indices = np.flatnonzero(usable)
        self.dropped_indices = np.flatnonzero(~usable)
        if self.selected_indices.size == 0:
            raise ValueError(
                "dataset contains no frames within the maximum missing fraction"
            )

        self.values = self.values[self.selected_indices]
        self.standard_deviation = self.standard_deviation[self.selected_indices]
        self.valid = self.valid[self.selected_indices]
        self.quality = self.quality[self.selected_indices]
        self.common_valid = common_valid[self.selected_indices]
        self.missing_fraction = missing_fraction[self.selected_indices]
        self.weight = np.min(self.quality, axis=1, keepdims=True)
        self.weight = np.where(
            self.common_valid & np.isfinite(self.weight), self.weight, 0.0
        ).astype(np.float32)

        self.target = self.transform.transform(self.values).astype(np.float32)
        self.transformed_variance = self.transform.variance(
            self.values, self.standard_deviation
        ).astype(np.float32)

    def __len__(self):
        return self.values.shape[0]

    def __getitem__(self, index):
        target = self.target[index].copy()
        mask = self.common_valid[index].copy()
        target[:, ~mask[0]] = 0.0
        weight = self.weight[index]
        encoder_input = np.concatenate((target, weight), axis=0)
        return {
            "encoder_input": torch.from_numpy(encoder_input),
            "target": torch.from_numpy(target),
            "mask": torch.from_numpy(mask),
            "weight": torch.from_numpy(weight),
            "variance": torch.from_numpy(self.transformed_variance[index]),
            "index": int(self.selected_indices[index]),
        }
