"""Compare matched 46x46 and 184x184 precipitation-CS spatial structure."""

#%% Imports

import argparse
import csv
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from icreader import load


FIELDS = (
    ("wic_corrected", "Corrected WIC", "counts"),
    ("si12", "SI12", "counts"),
)


#%% Exact nested-grid reduction and metrics

def block_mean(values, factor=4):
    """Average one fine field onto its exactly nested coarse grid."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or any(size % factor for size in values.shape):
        raise ValueError("fine field must be divisible by the nesting factor")
    rows, columns = values.shape
    blocks = values.reshape(
        rows // factor, factor, columns // factor, factor
    )
    count = np.isfinite(blocks).sum(axis=(1, 3))
    total = np.nansum(blocks, axis=(1, 3))
    return np.divide(
        total, count, out=np.full(total.shape, np.nan), where=count > 0
    )


def expanded(values, factor=4):
    return np.repeat(np.repeat(values, factor, axis=0), factor, axis=1)


def comparison_metrics(coarse, fine):
    fine_block = block_mean(fine)
    fine_coarse_scale = expanded(fine_block)
    fine_valid = np.isfinite(fine) & np.isfinite(fine_coarse_scale)
    residual = fine - fine_coarse_scale
    centered = fine[fine_valid] - np.mean(fine[fine_valid])
    detail_energy = np.sum(residual[fine_valid] ** 2)
    total_energy = np.sum(centered ** 2)

    common = np.isfinite(coarse) & np.isfinite(fine_block)
    if common.sum() < 2:
        correlation = np.nan
        normalized_rmse = np.nan
    else:
        correlation = np.corrcoef(coarse[common], fine_block[common])[0, 1]
        rmse = np.sqrt(np.mean((coarse[common] - fine_block[common]) ** 2))
        scale = np.std(fine_block[common])
        normalized_rmse = rmse / scale if scale > 0 else np.nan
    return {
        "fine_block": fine_block,
        "fine_coarse_scale": fine_coarse_scale,
        "fine_residual": residual,
        "detail_variance_fraction": (
            detail_energy / total_energy if total_energy > 0 else np.nan
        ),
        "coarse_correlation": correlation,
        "coarse_normalized_rmse": normalized_rmse,
        "coarse_common_cells": int(common.sum()),
        "fine_valid_cells": int(np.isfinite(fine).sum()),
    }


#%% One matched frame

def robust_limit(arrays, percentile=99):
    values = np.concatenate([
        np.asarray(array)[np.isfinite(array)] for array in arrays
    ])
    return max(float(np.percentile(np.maximum(values, 0), percentile)), 1e-6)


def plot_frame(record, output):
    figure, axes = plt.subplots(2, 5, figsize=(17, 7.2), layout="constrained")
    column_titles = (
        "Native 46x46\nexpanded 4x4",
        "Native 184x184\naveraged to 46x46",
        "Native 184x184",
        "184 minus its\n46-scale average",
        "Native 46 minus\nfine-grid average",
    )
    for axis, title in zip(axes[0], column_titles):
        axis.set_title(title, fontsize=10)

    for row, (_name, label, units) in enumerate(FIELDS):
        coarse = record["fields"][row]["coarse"]
        fine = record["fields"][row]["fine"]
        metrics = record["fields"][row]["metrics"]
        coarse_expanded = expanded(coarse)
        fine_coarse = metrics["fine_coarse_scale"]
        detail = metrics["fine_residual"]
        coarse_difference = expanded(coarse - metrics["fine_block"])
        maximum = robust_limit((coarse_expanded, fine_coarse, fine))
        difference_maximum = max(
            robust_limit((np.abs(detail), np.abs(coarse_difference)), 99),
            maximum * 0.02,
        )
        images = []
        for column, values in enumerate(
            (coarse_expanded, fine_coarse, fine)
        ):
            images.append(axes[row, column].imshow(
                values, origin="lower", cmap="viridis", vmin=0, vmax=maximum
            ))
        for column, values in enumerate((detail, coarse_difference), start=3):
            images.append(axes[row, column].imshow(
                values, origin="lower", cmap="coolwarm",
                vmin=-difference_maximum, vmax=difference_maximum
            ))
        axes[row, 0].set_ylabel(label)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.colorbar(images[2], ax=axes[row, :3], shrink=.76, label=units)
        figure.colorbar(images[3], ax=axes[row, 3:], shrink=.76, label=units)

    figure.suptitle(
        f"Orbit {record['orbit']:04d}, frame {record['frame']}, "
        f"{record['time']} UTC, Kp={record['kp']:g}\n"
        "Brightest jointly valid corrected-WIC frame in this orbit"
    )
    filename = f"orbit_{record['orbit']:04d}_frame_{record['frame']:03d}.png"
    figure.savefig(output / filename, dpi=170)
    plt.close(figure)
    return filename


#%% Cross-orbit summary

def plot_summary(records, output):
    labels = [f"{record['orbit']:04d}" for record in records]
    positions = np.arange(len(records))
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for sensor, (_name, label, _units) in enumerate(FIELDS):
        detail = [
            100 * record["fields"][sensor]["metrics"][
                "detail_variance_fraction"
            ] for record in records
        ]
        axes[0, sensor].bar(positions, detail)
        axes[0, sensor].set(
            title=f"{label}: variance below the 46x46 cell scale",
            ylabel="fraction of spatial variance [%]",
            xticks=positions, xticklabels=labels,
        )
        axes[0, sensor].tick_params(axis="x", rotation=45)

    width = .36
    for sensor, (_name, label, _units) in enumerate(FIELDS):
        correlation = [
            record["fields"][sensor]["metrics"]["coarse_correlation"]
            for record in records
        ]
        nrmse = [
            record["fields"][sensor]["metrics"]["coarse_normalized_rmse"]
            for record in records
        ]
        offset = (sensor - .5) * width
        axes[1, 0].bar(positions + offset, correlation, width, label=label)
        axes[1, 1].bar(positions + offset, nrmse, width, label=label)
    axes[1, 0].set(
        title="Native 46 versus averaged native 184",
        ylabel="Pearson correlation", ylim=(0, 1.05),
        xticks=positions, xticklabels=labels,
    )
    axes[1, 1].set(
        title="Native 46 versus averaged native 184",
        ylabel="RMSE / fine-grid spatial standard deviation",
        xticks=positions, xticklabels=labels,
    )
    for axis in axes[1]:
        axis.tick_params(axis="x", rotation=45)
        axis.legend()
    figure.suptitle(
        "Precipitation-CS resolution test; one bright frame per matched orbit"
    )
    figure.savefig(output / "test_A_summary.png", dpi=180)
    figure.savefig(output / "test_A_summary.pdf")
    plt.close(figure)


#%% Bounded remote reads and report

def run(arguments):
    coarse_root = arguments.coarse.expanduser().resolve()
    fine_root = arguments.fine.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    records = []

    for number, orbit in enumerate(arguments.orbits, start=1):
        coarse_path = coarse_root / f"or_{orbit:04d}.nc"
        fine_path = fine_root / f"or_{orbit:04d}.nc"
        with load(coarse_path) as coarse, load(fine_path) as fine:
            if (
                coarse.shape[0] != fine.shape[0]
                or not np.array_equal(coarse.time, fine.time)
                or coarse.shape[1:] != (46, 46)
                or fine.shape[1:] != (184, 184)
                or any(
                    not np.array_equal(old, new[::4])
                    for old, new in zip(coarse.grid.edges, fine.grid.edges)
                )
            ):
                raise ValueError(f"orbit {orbit}: products are not exactly nested")

            coarse_fields = [
                np.asarray(coarse.read(name), dtype=np.float64)
                for name, _label, _units in FIELDS
            ]
            jointly_valid = np.isfinite(coarse_fields[0]) & np.isfinite(
                coarse_fields[1]
            )
            score = np.nansum(
                np.where(jointly_valid, np.maximum(coarse_fields[0], 0), np.nan),
                axis=(1, 2),
            )
            enough = jointly_valid.sum(axis=(1, 2)) >= arguments.minimum_cells
            frame = int(np.nanargmax(np.where(enough, score, np.nan)))
            record = {
                "orbit": orbit,
                "frame": frame,
                "time": str(np.asarray(coarse.time)[frame]),
                "kp": float(np.asarray(coarse.Kp)[frame]),
                "fields": [],
            }
            for sensor, (name, _label, _units) in enumerate(FIELDS):
                coarse_values = coarse_fields[sensor][frame]
                fine_values = np.asarray(fine.read(name, frame), dtype=np.float64)
                record["fields"].append({
                    "coarse": coarse_values,
                    "fine": fine_values,
                    "metrics": comparison_metrics(coarse_values, fine_values),
                })
        record["filename"] = plot_frame(record, output)
        records.append(record)
        print(f"{number}/{len(arguments.orbits)} orbit {orbit:04d} complete", flush=True)

    plot_summary(records, output)
    with (output / "test_A_metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow((
            "orbit", "frame", "time", "kp", "field",
            "detail_variance_fraction", "coarse_correlation",
            "coarse_normalized_rmse", "coarse_common_cells", "fine_valid_cells",
        ))
        for record in records:
            for sensor, (name, _label, _units) in enumerate(FIELDS):
                metrics = record["fields"][sensor]["metrics"]
                writer.writerow((
                    record["orbit"], record["frame"], record["time"],
                    record["kp"], name, metrics["detail_variance_fraction"],
                    metrics["coarse_correlation"],
                    metrics["coarse_normalized_rmse"],
                    metrics["coarse_common_cells"], metrics["fine_valid_cells"],
                ))

    panels = "\n".join(
        f'<a href="{record["filename"]}"><img src="{record["filename"]}">'
        f'<span>Orbit {record["orbit"]:04d}: {escape(record["time"])}</span></a>'
        for record in records
    )
    page = (
        '<!doctype html><meta charset="utf-8"><title>CS resolution test A</title>'
        '<style>body{font:15px sans-serif;margin:20px}main{display:grid;'
        'grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:18px}'
        'img{width:100%}a{color:inherit;text-decoration:none}span{display:block;'
        'text-align:center}</style><h1>Test A: 46x46 versus 184x184</h1><main>'
        + panels + '</main>'
    )
    (output / "index.html").write_text(page, encoding="utf-8")
    return records


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("coarse", type=Path)
    parser.add_argument("fine", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--orbits", type=int, nargs="+",
        default=(85, 261, 358, 650, 904, 1189, 1513, 1930),
    )
    parser.add_argument("--minimum-cells", type=int, default=100)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
