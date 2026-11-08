"""Empirical condition regimes for named WIC-CVAE sampling."""

#%% Imports and physical definitions

import hashlib
import json
from pathlib import Path

import numpy as np

from .cache import AutomaticFieldDataset
from .conditioning import CONDITIONING_FEATURE_NAMES, align_conditioning_features


REGIME_NAMES = ("bz_south", "bz_north", "by_positive", "by_negative")


def standard_imf_clock_angle(by_gsm, bz_gsm):
    """Return GSM IMF clock angle with 0 north and 180 degrees south."""

    by_gsm, bz_gsm = np.broadcast_arrays(
        np.asarray(by_gsm, dtype=float), np.asarray(bz_gsm, dtype=float)
    )
    return np.mod(np.rad2deg(np.arctan2(by_gsm, bz_gsm)), 360.0)


def _circular_distance(angle, center):
    return np.abs((angle - center + 180.0) % 360.0 - 180.0)


def empirical_regime_masks(
    matrix, field_minimum=2.0, field_maximum=10.0,
    clock_half_width=30.0, maximum_absolute_tilt=5.0
):
    """Select physical IMF regimes without restricting nuisance variables."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(CONDITIONING_FEATURE_NAMES):
        raise ValueError("conditioning matrix does not match the named feature contract")
    limits = np.asarray([
        field_minimum, field_maximum, clock_half_width,
        maximum_absolute_tilt
    ], dtype=float)
    if (
        not np.isfinite(limits).all() or field_minimum <= 0
        or field_maximum <= field_minimum
        or clock_half_width <= 0 or clock_half_width >= 90
        or maximum_absolute_tilt < 0
    ):
        raise ValueError("regime field, clock and tilt limits are invalid")

    by_gsm = matrix[:, CONDITIONING_FEATURE_NAMES.index("by_gsm")]
    bz_gsm = matrix[:, CONDITIONING_FEATURE_NAMES.index("bz_gsm")]
    tilt = matrix[:, CONDITIONING_FEATURE_NAMES.index("dipole_tilt")]
    clock = standard_imf_clock_angle(by_gsm, bz_gsm)
    common = np.isfinite(matrix).all(axis=1) & (np.abs(tilt) <= maximum_absolute_tilt)

    masks = {
        "bz_south": (
            common & (bz_gsm >= -field_maximum) & (bz_gsm <= -field_minimum)
            & (_circular_distance(clock, 180.0) <= clock_half_width)
        ),
        "bz_north": (
            common & (bz_gsm >= field_minimum) & (bz_gsm <= field_maximum)
            & (_circular_distance(clock, 0.0) <= clock_half_width)
        ),
        "by_positive": (
            common & (by_gsm >= field_minimum) & (by_gsm <= field_maximum)
            & (_circular_distance(clock, 90.0) <= clock_half_width)
        ),
        "by_negative": (
            common & (by_gsm >= -field_maximum) & (by_gsm <= -field_minimum)
            & (_circular_distance(clock, 270.0) <= clock_half_width)
        ),
    }
    definitions = {
        "bz_south": {
            "bz_gsm_nT": [-field_maximum, -field_minimum],
            "standard_clock_degrees": [
                180 - clock_half_width, 180 + clock_half_width
            ],
        },
        "bz_north": {
            "bz_gsm_nT": [field_minimum, field_maximum],
            "standard_clock_center_degrees": 0.0,
            "standard_clock_half_width_degrees": clock_half_width,
        },
        "by_positive": {
            "by_gsm_nT": [field_minimum, field_maximum],
            "standard_clock_degrees": [
                90 - clock_half_width, 90 + clock_half_width
            ],
        },
        "by_negative": {
            "by_gsm_nT": [-field_maximum, -field_minimum],
            "standard_clock_degrees": [
                270 - clock_half_width, 270 + clock_half_width
            ],
        },
    }
    for definition in definitions.values():
        definition["dipole_tilt_degrees"] = [
            -maximum_absolute_tilt, maximum_absolute_tilt
        ]
        definition["other_features"] = "unrestricted empirical training distribution"
    return masks, definitions, clock


#%% Exact training-row recovery

def load_complete_training_conditions(cache_directory, conditioning_directory):
    """Recover the finite condition rows used by the named CVAE train split."""

    cache_directory = Path(cache_directory).expanduser().resolve()
    conditioning_directory = Path(conditioning_directory).expanduser().resolve()
    cache_manifest_path = cache_directory / "manifest.json"
    cache_manifest = json.loads(cache_manifest_path.read_text())
    if cache_manifest.get("field_names") != ["wic_corrected"]:
        raise ValueError("expected a one-field WIC CVAE cache")
    dataset = AutomaticFieldDataset(
        cache_directory, "train", include_uncertainty=False,
        expected_cache_format=cache_manifest["cache_format"]
    )
    try:
        matrix, condition_manifest, condition_sha256 = align_conditioning_features(
            conditioning_directory, dataset.frame_orbit, dataset.source_frames,
            dataset.frame_time, dataset.kp
        )
        finite = np.all(np.isfinite(matrix), axis=1)
        result = {
            "matrix": matrix[finite],
            "orbit": dataset.frame_orbit[finite],
            "source_frame": dataset.source_frames[finite],
            "time": dataset.frame_time[finite],
        }
    finally:
        dataset.close()
    result["provenance"] = {
        "cache_directory": str(cache_directory),
        "cache_manifest_sha256": hashlib.sha256(
            cache_manifest_path.read_bytes()
        ).hexdigest(),
        "conditioning_directory": str(conditioning_directory),
        "conditioning_manifest_sha256": condition_sha256,
        "conditioning_format": condition_manifest["format"],
        "split": "train",
        "complete_frames": int(len(result["matrix"])),
    }
    return result
