"""Select representative observed condition vectors for WIC-CVAE sampling."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from icanalyzer.cvae.cache import AutomaticFieldDataset
from icanalyzer.cvae.conditioning import (
    CONDITIONING_FEATURE_NAMES,
    align_conditioning_features,
)


#%% Representative observed regimes

def representative_indices(matrix):
    """Choose actual rows near the center of explicit physical regimes."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[1] != len(CONDITIONING_FEATURE_NAMES):
        raise ValueError("conditioning matrix does not match the named feature contract")
    if len(matrix) < 100 or not np.isfinite(matrix).all():
        raise ValueError("case selection needs at least 100 complete condition rows")

    column = {
        name: matrix[:, CONDITIONING_FEATURE_NAMES.index(name)]
        for name in ("sml", "smr", "by_gsm", "bz_gsm", "kp")
    }
    quantile = {
        name: np.quantile(values, [.1, .2, .25, .35, .65, .75, .8, .9])
        for name, values in column.items()
    }
    q = lambda name, position: quantile[name][position]

    central_sml = (column["sml"] >= q("sml", 2)) & (column["sml"] <= q("sml", 5))
    central_kp = (column["kp"] >= q("kp", 2)) & (column["kp"] <= q("kp", 5))
    central_by = (column["by_gsm"] >= q("by_gsm", 3)) & (column["by_gsm"] <= q("by_gsm", 4))
    central_bz = (column["bz_gsm"] >= q("bz_gsm", 3)) & (column["bz_gsm"] <= q("bz_gsm", 4))

    regimes = [
        ("typical", np.ones(len(matrix), dtype=bool)),
        ("quiet", (column["sml"] >= q("sml", 6)) & (column["kp"] <= q("kp", 1))),
        ("active_substorm", (column["sml"] <= q("sml", 0)) & (column["kp"] >= q("kp", 4))),
        ("storm", (column["smr"] <= q("smr", 0)) & (column["kp"] >= q("kp", 4))),
        ("by_negative", (column["by_gsm"] <= q("by_gsm", 0)) & central_sml & central_kp & central_bz),
        ("by_positive", (column["by_gsm"] >= q("by_gsm", 7)) & central_sml & central_kp & central_bz),
        ("bz_south", (column["bz_gsm"] <= q("bz_gsm", 0)) & central_sml & central_kp & central_by),
        ("bz_north", (column["bz_gsm"] >= q("bz_gsm", 7)) & central_sml & central_kp & central_by),
    ]

    center = np.median(matrix, axis=0)
    scale = np.quantile(matrix, .75, axis=0) - np.quantile(matrix, .25, axis=0)
    scale = np.where(scale > 0, scale, np.std(matrix, axis=0))
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError("conditioning columns need finite nonzero spread")
    standardized = (matrix - center) / scale

    selected = []
    used = set()
    for label, mask in regimes:
        candidates = np.flatnonzero(mask)
        if not len(candidates):
            raise ValueError(f"no complete observed rows satisfy the {label} regime")
        target = np.median(standardized[candidates], axis=0)
        distance = np.sum((standardized[candidates] - target) ** 2, axis=1)
        ordered = candidates[np.argsort(distance)]
        index = next((int(value) for value in ordered if int(value) not in used), None)
        if index is None:
            raise ValueError(f"the {label} regime has no distinct representative row")
        selected.append((label, index, int(len(candidates))))
        used.add(index)
    return selected


def condition_record(label, matrix_row, orbit, source_frame, time, candidates):
    """Convert one observed internal row to the sampler's physical JSON form."""

    values = dict(zip(CONDITIONING_FEATURE_NAMES, np.asarray(matrix_row, dtype=float)))
    day_phase = np.arctan2(values["day_of_year_sin"], values["day_of_year_cos"]) % (2 * np.pi)
    clock_phase = np.arctan2(values["clock_angle_sin"], values["clock_angle_cos"]) % (2 * np.pi)
    return {
        "label": label,
        "sml": values["sml"],
        "smr": values["smr"],
        "imf_by": values["by_gsm"],
        "imf_bz": values["bz_gsm"],
        "vx": values["vx_gse"],
        "clock_angle": float(np.rad2deg(clock_phase)),
        "dynamic_pressure": values["dynamic_pressure"],
        "newell": values["newell"],
        "kp": values["kp"],
        "day_of_year": float(1 + day_phase * 365.25 / (2 * np.pi)),
        "dipole_tilt": values["dipole_tilt"],
        "source_orbit": int(orbit),
        "source_frame": int(source_frame),
        "source_time": np.datetime_as_string(time, unit="s"),
        "regime_candidate_frames": int(candidates),
    }


#%% Cache join and output

def run(arguments):
    cache = arguments.cache.expanduser().resolve()
    conditioning = arguments.conditioning.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("condition-case output already exists")

    manifest_path = cache / "manifest.json"
    metadata = json.loads(manifest_path.read_text())
    if metadata.get("field_names") != ["wic_corrected"]:
        raise ValueError("expected a one-field WIC CVAE cache")
    dataset = AutomaticFieldDataset(
        cache, arguments.split, include_uncertainty=False,
        expected_cache_format=metadata["cache_format"]
    )
    try:
        matrix, condition_manifest, condition_sha256 = align_conditioning_features(
            conditioning, dataset.frame_orbit, dataset.source_frames,
            dataset.frame_time, dataset.kp
        )
        finite = np.all(np.isfinite(matrix), axis=1)
        matrix = matrix[finite]
        orbits = dataset.frame_orbit[finite]
        source_frames = dataset.source_frames[finite]
        times = dataset.frame_time[finite]
    finally:
        dataset.close()

    selected = representative_indices(matrix)
    records = [
        condition_record(
            label, matrix[index], orbits[index], source_frames[index],
            times[index], candidate_count
        )
        for label, index, candidate_count in selected
    ]
    result = {
        "selection": {
            "policy": "actual complete training rows nearest each regime's multivariate median",
            "split": arguments.split,
            "complete_frames": int(len(matrix)),
            "cache_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "conditioning_manifest_sha256": condition_sha256,
            "conditioning_format": condition_manifest["format"],
        },
        "conditions": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Saved {len(records)} observed condition cases to {output}", flush=True)
    for record in records:
        print(
            f"  {record['label']}: orbit {record['source_orbit']:04d}, "
            f"frame {record['source_frame']}, {record['source_time']}",
            flush=True
        )
    return result


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("conditioning", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "validation", "test"), default="train")
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
