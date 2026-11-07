"""Rank SuperMAG conditioning groups against masked WIC morphology targets."""

#%% Imports and benchmark contract

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score

from icanalyzer.cvae.cache import file_sha256
from scripts.CVAE.prepare_conditioning_data import CONDITIONING_FORMAT


FEATURE_GROUPS = {
    "sml": ("sml",),
    "smr": ("smr",),
    "imf_by_bz": ("by_gsm", "bz_gsm"),
    "newell": ("newell",),
    "velocity": ("vx_gse", "vy_gse", "vz_gsm"),
    "plasma": ("density", "dynamic_pressure"),
}
FEATURE_NAMES = tuple(name for group in FEATURE_GROUPS.values() for name in group)
TARGET_NAMES = (
    "log_observed_total_wic", "log_mean_wic", "peak_mlat",
    "brightness_centroid_mlat", "brightness_width_mlat",
    "dawn_dusk_mean_asymmetry",
)


#%% Exact conditioning join and interpretable WIC targets

def morphology_targets(wic, valid, mlat, mlt):
    valid = valid & np.isfinite(wic) & (wic >= 0)
    values = np.where(valid, wic, 0.0).astype(np.float64)
    counts = valid.sum(axis=(1, 2))
    total = values.sum(axis=(1, 2))
    positive = total > 0
    denominator = np.where(positive, total, 1.0)

    centroid = np.sum(values * mlat, axis=(1, 2)) / denominator
    width = np.sqrt(
        np.sum(values * (mlat[None] - centroid[:, None, None])**2, axis=(1, 2))
        / denominator
    )
    flat = values.reshape(len(values), -1)
    peak = mlat.reshape(-1)[np.argmax(flat, axis=1)]

    dawn = mlt < 12
    dusk = ~dawn
    dawn_count = np.sum(valid & dawn, axis=(1, 2))
    dusk_count = np.sum(valid & dusk, axis=(1, 2))
    dawn_mean = np.sum(values * dawn, axis=(1, 2)) / np.maximum(dawn_count, 1)
    dusk_mean = np.sum(values * dusk, axis=(1, 2)) / np.maximum(dusk_count, 1)
    asymmetry = (dusk_mean - dawn_mean) / (dusk_mean + dawn_mean + 1e-8)

    targets = np.column_stack((
        np.log10(1 + total), np.log10(1 + total / np.maximum(counts, 1)),
        peak, centroid, width, asymmetry
    )).astype(np.float32)
    targets[~positive, 2:] = np.nan
    return targets, (counts / valid.shape[1] / valid.shape[2]).astype(np.float32)


def load_benchmark_data(cache_directory, conditioning_directory):
    cache = Path(cache_directory).expanduser().resolve()
    conditioning = Path(conditioning_directory).expanduser().resolve()
    cache_manifest = json.loads((cache / "manifest.json").read_text())
    condition_manifest = json.loads((conditioning / "manifest.json").read_text())
    if condition_manifest.get("format") != CONDITIONING_FORMAT:
        raise ValueError("expected completed IMAGE/SuperMAG conditioning data")
    if cache_manifest.get("field_names") != ["wic_corrected"]:
        raise ValueError("conditioning benchmark requires a one-field WIC cache")

    with np.load(conditioning / condition_manifest["array_file"], allow_pickle=False) as saved:
        condition = {name: saved[name].copy() for name in saved.files}
    condition_key = condition["orbit"].astype(np.int64) * 1_000_000 + condition["source_frame"]
    order = np.argsort(condition_key)
    sorted_key = condition_key[order]
    if np.any(np.diff(sorted_key) == 0):
        raise ValueError("conditioning table contains duplicate orbit/frame keys")

    grid = cache_manifest["grid"]
    mlat = np.asarray(grid["mlat"], dtype=np.float64)
    mlt = np.asarray(grid["mlt"], dtype=np.float64)
    collected = {name: [] for name in ("features", "targets", "valid_fraction", "orbit", "time", "split")}
    for row in cache_manifest["orbits"]:
        if not row.get("shard"):
            continue
        shard = cache / row["shard"]
        with np.load(shard / "frames.npz", allow_pickle=False) as frames:
            selected = frames["eligible"].astype(bool)
            frame = frames["source_frame"][selected]
            time = frames["time"][selected].astype("datetime64[us]")
        key = int(row["orbit"]) * 1_000_000 + frame.astype(np.int64)
        positions = np.searchsorted(sorted_key, key)
        if np.any(positions == len(sorted_key)) or np.any(sorted_key[np.minimum(positions, len(sorted_key) - 1)] != key):
            raise ValueError(f"orbit {row['orbit']}: cache frames are absent from conditioning data")
        positions = order[positions]
        if not np.array_equal(condition["time"][positions].astype("datetime64[us]"), time):
            raise ValueError(f"orbit {row['orbit']}: conditioning and cache timestamps differ")

        physical = np.load(shard / "physical.npy", mmap_mode="r")
        valid = np.load(shard / "valid.npy", mmap_mode="r")
        targets, valid_fraction = morphology_targets(physical[selected, 0], valid[selected], mlat, mlt)
        collected["features"].append(np.column_stack([condition[name][positions] for name in FEATURE_NAMES]))
        collected["targets"].append(targets)
        collected["valid_fraction"].append(valid_fraction)
        collected["orbit"].append(np.full(len(frame), row["orbit"], dtype=np.int32))
        collected["time"].append(time)
        collected["split"].append(np.full(len(frame), row["split"]))
    return {name: np.concatenate(parts) for name, parts in collected.items()}, cache_manifest, condition_manifest


#%% Time-blocked full/drop-group and grouped-permutation scores

def scores(target, predicted, weight):
    overall = float(r2_score(
        target, predicted, sample_weight=weight, multioutput="variance_weighted"
    ))
    per_target = r2_score(
        target, predicted, sample_weight=weight, multioutput="raw_values"
    )
    return overall, {
        name: float(value) for name, value in zip(TARGET_NAMES, per_target)
    }


def folds(data, mode):
    if mode == "year":
        labels = data["time"].astype("datetime64[Y]").astype(int) + 1970
        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError("year-blocked evaluation requires at least two years")
        return [(f"year_{year}", labels != year, labels == year) for year in unique]
    train = data["split"] == "train"
    return [(name, train, data["split"] == name) for name in ("validation", "test")]


def model(arguments, seed):
    return ExtraTreesRegressor(
        n_estimators=arguments.trees, min_samples_leaf=arguments.minimum_leaf,
        max_features=1.0, n_jobs=arguments.threads, random_state=seed
    )


def run(arguments):
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new importance-analysis directory")
    data, cache_manifest, condition_manifest = load_benchmark_data(arguments.cache, arguments.conditioning)
    finite = np.all(np.isfinite(data["features"]), axis=1) & np.all(np.isfinite(data["targets"]), axis=1)
    data = {name: values[finite] for name, values in data.items()}
    if len(data["features"]) < 100:
        raise ValueError("too few jointly finite cache/conditioning frames")

    rng = np.random.default_rng(arguments.seed)
    fold_results = []
    for number, (name, train, test) in enumerate(folds(data, arguments.split_mode)):
        train_indices = np.flatnonzero(train)
        if len(train_indices) > arguments.maximum_training_frames:
            train_indices = rng.choice(train_indices, arguments.maximum_training_frames, replace=False)
        test_indices = np.flatnonzero(test)
        if not len(train_indices) or not len(test_indices):
            raise ValueError(f"empty training or held-out population for {name}")
        x_train = data["features"][train_indices]
        y_train = data["targets"][train_indices]
        x_test = data["features"][test_indices]
        y_test = data["targets"][test_indices]
        train_weight = data["valid_fraction"][train_indices]
        weight = data["valid_fraction"][test_indices]
        target_center = np.average(y_train, axis=0, weights=train_weight)
        target_scale = np.sqrt(np.average(
            (y_train - target_center)**2, axis=0, weights=train_weight
        ))
        target_scale = np.maximum(target_scale, 1e-6)
        scaled_train = (y_train - target_center) / target_scale

        full = model(arguments, arguments.seed + number).fit(
            x_train, scaled_train, sample_weight=train_weight
        )
        full_prediction = full.predict(x_test) * target_scale + target_center
        full_score, full_target_scores = scores(y_test, full_prediction, weight)
        importance = {}
        for group, names in FEATURE_GROUPS.items():
            columns = [FEATURE_NAMES.index(field) for field in names]
            retained = [column for column in range(len(FEATURE_NAMES)) if column not in columns]
            reduced = model(arguments, arguments.seed + 100 + number).fit(
                x_train[:, retained], scaled_train, sample_weight=train_weight
            )
            reduced_score, reduced_target_scores = scores(
                y_test,
                reduced.predict(x_test[:, retained]) * target_scale + target_center,
                weight
            )
            permuted = x_test.copy()
            permutation = rng.permutation(len(permuted))
            permuted[:, columns] = permuted[permutation][:, columns]
            permutation_score, permutation_target_scores = scores(
                y_test, full.predict(permuted) * target_scale + target_center,
                weight
            )
            importance[group] = {
                "drop_group_delta_r2": full_score - reduced_score,
                "permutation_delta_r2": full_score - permutation_score,
                "drop_group_delta_r2_by_target": {
                    target: full_target_scores[target] - reduced_target_scores[target]
                    for target in TARGET_NAMES
                },
                "permutation_delta_r2_by_target": {
                    target: full_target_scores[target] - permutation_target_scores[target]
                    for target in TARGET_NAMES
                },
            }
        fold_results.append({
            "fold": name, "train_frames": len(train_indices),
            "test_frames": len(test_indices), "full_r2": full_score,
            "full_r2_by_target": full_target_scores, "importance": importance
        })

    aggregate = {}
    for group in FEATURE_GROUPS:
        aggregate[group] = {}
        for metric in ("drop_group_delta_r2", "permutation_delta_r2"):
            values = [fold["importance"][group][metric] for fold in fold_results]
            aggregate[group][metric] = float(np.mean(values))
            aggregate[group][metric + "_fold_std"] = float(np.std(values))
        for metric in ("drop_group_delta_r2_by_target", "permutation_delta_r2_by_target"):
            aggregate[group][metric] = {
                target: float(np.mean([
                    fold["importance"][group][metric][target]
                    for fold in fold_results
                ])) for target in TARGET_NAMES
            }
    output.mkdir(parents=True)
    np.savez_compressed(output / "benchmark_data.npz", **data, feature_names=FEATURE_NAMES, target_names=TARGET_NAMES)
    result = {
        "feature_groups": FEATURE_GROUPS, "target_names": TARGET_NAMES,
        "split_mode": arguments.split_mode, "jointly_finite_frames": len(data["features"]),
        "folds": fold_results, "aggregate_importance": aggregate,
        "cache_manifest_sha256": file_sha256(Path(arguments.cache).expanduser() / "manifest.json"),
        "conditioning_manifest_sha256": file_sha256(Path(arguments.conditioning).expanduser() / "manifest.json"),
        "target_training": "weighted standardization within each training fold",
        "interpretation": "predictive importance for masked WIC summaries; not causal attribution",
    }
    (output / "importance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(aggregate, indent=2), flush=True)
    return result


#%% CLI

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("conditioning", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split-mode", choices=("year", "cache"), default="year")
    parser.add_argument("--trees", type=int, default=200)
    parser.add_argument("--minimum-leaf", type=int, default=20)
    parser.add_argument("--maximum-training-frames", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    arguments = parser.parse_args(argv)
    if min(arguments.trees, arguments.minimum_leaf, arguments.maximum_training_frames, arguments.threads) < 1:
        parser.error("tree, leaf, frame and thread counts must be positive")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
