"""Compare legacy and rerun WIC/SI12 inputs on exact paired timestamps."""

#%% Imports

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from icanalyzer.wic_si12.comparison import load_paired_orbit, pair_orbit_files
from icanalyzer.wic_si12.data import (
    _NoFramesRemainError,
    _orbit_from_path,
    load_good_orbit_intervals,
    select_annotated_orbit_files,
)


#%% Inputs

def discover_files(inputs):
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("or_*.nc")))
        elif path.is_file():
            files.append(path)
        else:
            files.extend(sorted(Path().glob(value)))
    files = sorted({path.resolve() for path in files})
    if not files:
        raise FileNotFoundError("no current precipitation_cs orbit files were found")
    return files


def field_arrays(product):
    return {
        "wic_corrected": product["values"][:, 0],
        "si12": product["values"][:, 1],
        "si13_corrected": product["si13_corrected"],
        "dwic_corrected": product["standard_deviation"][:, 0],
        "dsi12": product["standard_deviation"][:, 1],
        "dsi13_corrected": product["dsi13_corrected"],
        "quality_wic": product["quality"][:, 0],
        "quality_si12": product["quality"][:, 1],
    }


def field_validity(product):
    common = np.all(product["valid"], axis=1)
    return {
        "wic_corrected": product["valid"][:, 0],
        "si12": product["valid"][:, 1],
        "si13_corrected": common,
        "dwic_corrected": product["valid"][:, 0],
        "dsi12": product["valid"][:, 1],
        "dsi13_corrected": common,
        "quality_wic": product["valid"][:, 0],
        "quality_si12": product["valid"][:, 1],
    }


#%% Streaming statistics

def empty_accumulator(shape):
    fields = (
        "wic_corrected", "si12", "si13_corrected",
        "dwic_corrected", "dsi12", "dsi13_corrected",
        "quality_wic", "quality_si12",
    )
    statistics = {
        name: {
            "count": 0,
            "legacy_sum": 0.0,
            "current_sum": 0.0,
            "legacy_square_sum": 0.0,
            "current_square_sum": 0.0,
            "cross_sum": 0.0,
            "difference_sum": 0.0,
            "difference_square_sum": 0.0,
        }
        for name in fields
    }
    maps = {
        name: {
            "count": np.zeros(shape, dtype=np.int64),
            "legacy_sum": np.zeros(shape, dtype=np.float64),
            "current_sum": np.zeros(shape, dtype=np.float64),
        }
        for name in ("wic_corrected", "si12", "si13_corrected")
    }
    return statistics, maps


def update_accumulator(statistics, maps, paired):
    current = paired["current"]
    legacy = paired["legacy"]
    if not np.array_equal(current["time"], legacy["time"]):
        raise ValueError("paired loader returned misaligned product times")

    current_fields = field_arrays(current)
    legacy_fields = field_arrays(legacy)
    current_valid = field_validity(current)
    legacy_valid = field_validity(legacy)

    for name in statistics:
        mask = (
            current_valid[name]
            & legacy_valid[name]
            & np.isfinite(current_fields[name])
            & np.isfinite(legacy_fields[name])
        )
        old = legacy_fields[name][mask].astype(np.float64)
        new = current_fields[name][mask].astype(np.float64)
        difference = new - old
        values = statistics[name]
        values["count"] += int(old.size)
        values["legacy_sum"] += float(np.sum(old))
        values["current_sum"] += float(np.sum(new))
        values["legacy_square_sum"] += float(np.dot(old, old))
        values["current_square_sum"] += float(np.dot(new, new))
        values["cross_sum"] += float(np.dot(old, new))
        values["difference_sum"] += float(np.sum(difference))
        values["difference_square_sum"] += float(np.dot(difference, difference))

        if name in maps:
            maps[name]["count"] += np.sum(mask, axis=0)
            maps[name]["legacy_sum"] += np.sum(np.where(mask, legacy_fields[name], 0), axis=0)
            maps[name]["current_sum"] += np.sum(np.where(mask, current_fields[name], 0), axis=0)

    return {
        "frames": int(current["values"].shape[0]),
        "current_valid_pixels": int(np.count_nonzero(current["valid"])),
        "legacy_valid_pixels": int(np.count_nonzero(legacy["valid"])),
        "common_valid_pixels": int(np.count_nonzero(current["valid"] & legacy["valid"])),
    }


def finish_statistics(statistics):
    result = {}
    for name, values in statistics.items():
        count = values["count"]
        if count == 0:
            result[name] = {"paired_pixel_count": 0}
            continue

        old_mean = values["legacy_sum"] / count
        new_mean = values["current_sum"] / count
        old_variance = max(values["legacy_square_sum"] / count - old_mean**2, 0.0)
        new_variance = max(values["current_square_sum"] / count - new_mean**2, 0.0)
        covariance = values["cross_sum"] / count - old_mean * new_mean
        scale = np.sqrt(old_variance * new_variance)
        result[name] = {
            "paired_pixel_count": count,
            "legacy_mean": old_mean,
            "current_mean": new_mean,
            "mean_change_current_minus_legacy": values["difference_sum"] / count,
            "rmse": np.sqrt(values["difference_square_sum"] / count),
            "correlation": covariance / scale if scale > 0 else None,
        }
    return result


#%% Figure

def mean_map(values, key):
    result = np.full(values["count"].shape, np.nan)
    np.divide(values[key], values["count"], out=result, where=values["count"] > 0)
    return result


def plot_maps(maps, output):
    labels = {
        "wic_corrected": "Corrected WIC [counts]",
        "si12": "SI12 [counts]",
        "si13_corrected": "Corrected SI13 [counts]",
    }
    figure, axes = plt.subplots(3, 3, figsize=(11, 10), constrained_layout=True)
    for row, (name, label) in enumerate(labels.items()):
        legacy = mean_map(maps[name], "legacy_sum")
        current = mean_map(maps[name], "current_sum")
        difference = current - legacy
        finite_values = np.concatenate((legacy[np.isfinite(legacy)], current[np.isfinite(current)]))
        upper = float(np.percentile(finite_values, 99)) if finite_values.size else 1.0
        difference_limit = float(np.nanpercentile(np.abs(difference), 99))
        if not np.isfinite(difference_limit) or difference_limit == 0:
            difference_limit = 1.0

        for column, (image, title) in enumerate((
            (legacy, "Legacy mean"),
            (current, "Rerun mean"),
            (difference, "Rerun - legacy"),
        )):
            if column < 2:
                artist = axes[row, column].imshow(image, origin="lower", vmin=0, vmax=upper)
            else:
                artist = axes[row, column].imshow(
                    image, origin="lower", cmap="coolwarm",
                    vmin=-difference_limit, vmax=difference_limit,
                )
            axes[row, column].set_title(f"{title}\n{label}")
            axes[row, column].set_xlabel("grid column")
            axes[row, column].set_ylabel("grid row")
            figure.colorbar(artist, ax=axes[row, column], shrink=0.78)
    figure.savefig(output, dpi=180)
    plt.close(figure)


#%% Workflow

def run(arguments):
    files = discover_files(arguments.current_inputs)
    intervals = None
    if not arguments.all_frames:
        intervals = load_good_orbit_intervals(arguments.segments)
        files = select_annotated_orbit_files(files, intervals)
    pairs = pair_orbit_files(files, arguments.legacy_directory)
    if arguments.max_files is not None:
        pairs = pairs[:arguments.max_files]

    statistics = None
    maps = None
    totals = {
        "common_orbit_files": len(pairs),
        "paired_frames": 0,
        "current_valid_pixels": 0,
        "legacy_valid_pixels": 0,
        "common_valid_pixels": 0,
        "excluded_orbit_files": [],
    }
    for current_path, legacy_path in pairs:
        orbit = _orbit_from_path(current_path)
        interval = None if intervals is None else intervals[orbit]
        try:
            paired = load_paired_orbit(
                current_path,
                legacy_path,
                data_version="both",
                wic_source_interval=interval,
                minimum_sza=arguments.minimum_sza,
            )
        except _NoFramesRemainError:
            totals["excluded_orbit_files"].append(str(current_path))
            continue
        if statistics is None:
            statistics, maps = empty_accumulator(paired["current"]["values"].shape[-2:])
        counts = update_accumulator(statistics, maps, paired)
        totals["paired_frames"] += counts["frames"]
        for name in ("current_valid_pixels", "legacy_valid_pixels", "common_valid_pixels"):
            totals[name] += counts[name]

    if statistics is None:
        raise ValueError("no paired frames remain after comparison selection")

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    figure_name = "paired_input_mean_maps.png"
    plot_maps(maps, output / figure_name)
    report = {
        "selection": {
            "current_inputs": [str(Path(value).resolve()) for value in arguments.current_inputs],
            "legacy_directory": str(Path(arguments.legacy_directory).resolve()),
            "manual_frame_intervals": None if intervals is None else str(Path(arguments.segments).resolve()),
            "minimum_sza_degrees": arguments.minimum_sza,
            "pairing_rule": "same orbit filename and exact product timestamp",
            "quality_weights": {
                "legacy": "combined robust weight w",
                "current": "method_quality_weight times channel coverage",
            },
        },
        "totals": totals,
        "fields": finish_statistics(statistics),
        "created_figures": [figure_name],
        "interpretation_limit": (
            "This is the net difference between the legacy and rerun product pipelines, "
            "not an isolated fuvpy-only effect; matching, regridding, validity, and "
            "uncertainty propagation also changed."
        ),
    }
    (output / "paired_input_comparison.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current_inputs", nargs="+", help="current Product-2 files or directory")
    parser.add_argument("--legacy-directory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--segments",
        default=str(Path(__file__).resolve().parents[3] / "data" / "good_orbit_segments.csv"),
    )
    parser.add_argument(
        "--all-frames", "--no-csv-selection", dest="all_frames", action="store_true",
        help="bypass good_orbit_segments.csv; SZA and technical validity still apply",
    )
    parser.add_argument("--minimum-sza", type=float)
    parser.add_argument("--max-files", type=int, help="bounded smoke-test subset")
    return parser.parse_args(argv)


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), indent=2))
