"""Step 1: validate posterior reconstruction on the held-out WIC/SI12 cache.

The encoder receives the complete saved input tensor, but the poster figure
shows the physically meaningful comparison: observed image, decoder at the
posterior mean, and residual. Metrics use only originally valid pixels.
"""

#%% Imports

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.cache import AutomaticFieldDataset, WIC_SI12_CACHE_FORMATS


#%% Held-out frame metrics

def frame_metrics(observed, reconstructed, valid):
    """Return one record per frame and channel on the observed support."""

    observed = np.asarray(observed)
    reconstructed = np.asarray(reconstructed)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape == (len(observed), 1, *observed.shape[2:]):
        valid = np.broadcast_to(valid, observed.shape)
    if observed.shape != reconstructed.shape or observed.shape != valid.shape:
        raise ValueError("observed, reconstruction and validity arrays must align")
    records = []
    for frame in range(len(observed)):
        channels = []
        for channel in range(observed.shape[1]):
            mask = valid[frame, channel]
            truth = observed[frame, channel][mask].astype(np.float64)
            estimate = reconstructed[frame, channel][mask].astype(np.float64)
            if truth.size < 2:
                channels.append({"pixels": int(truth.size), "rmse": None, "nrmse": None, "correlation": None, "bias": None})
                continue
            residual = estimate - truth
            scale = float(np.std(truth))
            correlation = float(np.corrcoef(truth, estimate)[0, 1]) if np.std(estimate) > 0 and scale > 0 else None
            rmse = float(np.sqrt(np.mean(residual**2)))
            channels.append({
                "pixels": int(truth.size),
                "rmse": rmse,
                "nrmse": None if scale <= 0 else rmse / scale,
                "correlation": correlation,
                "bias": float(np.mean(residual)),
            })
        records.append(channels)
    return records


def select_representatives(records, eligible=None, count=10):
    """Select distinct frames at evenly spaced joint-NRMSE quantiles."""

    scores = np.array([
        np.mean([entry["nrmse"] for entry in frame if entry["nrmse"] is not None])
        for frame in records
    ])
    finite = np.flatnonzero(np.isfinite(scores))
    if eligible is not None:
        eligible = np.asarray(eligible, dtype=bool)
        if eligible.shape != scores.shape:
            raise ValueError("representative eligibility must contain one value per frame")
        finite = finite[eligible[finite]]
    if finite.size < count:
        raise ValueError(
            f"need at least {count} eligible held-out frames with finite reconstruction scores"
        )
    ordered = finite[np.argsort(scores[finite])]
    quantiles = (np.arange(count) + 0.5) / count
    positions = np.rint(quantiles * (len(ordered) - 1)).astype(int)
    selected = ordered[positions]
    if len(np.unique(selected)) != count:
        raise RuntimeError("representative quantile selection did not produce distinct frames")
    return selected, scores, quantiles


#%% Poster figures

def plot_representative(observed, reconstructed, valid, title, output, stem, dpi):
    residual = reconstructed - observed
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.5), constrained_layout=True)
    for channel, name in enumerate(("WIC", "SI12")):
        mask = valid[channel]
        vmax = max(float(np.nanpercentile(observed[channel][mask], 99.5)), 1.0)
        error_limit = max(float(np.nanpercentile(abs(residual[channel][mask]), 99.5)), 1.0)
        panels = (observed[channel], reconstructed[channel], residual[channel])
        for column, (panel, label) in enumerate(zip(panels, ("Observation", "Posterior reconstruction", "Residual"))):
            image = axes[channel, column].imshow(
                np.where(mask, panel, np.nan), origin="lower",
                cmap="RdBu_r" if column == 2 else "magma",
                vmin=-error_limit if column == 2 else 0,
                vmax=error_limit if column == 2 else vmax,
                interpolation="nearest",
            )
            axes[channel, column].set_title(f"{name}: {label}")
            axes[channel, column].set_axis_off()
            figure.colorbar(image, ax=axes[channel, column], label="counts", shrink=0.78)
    figure.suptitle(title)
    figure.savefig(output / f"{stem}.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_population(records, output, dpi):
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), constrained_layout=True)
    display = {}
    for channel, name in enumerate(("WIC", "SI12")):
        nrmse = np.asarray([frame[channel]["nrmse"] for frame in records if frame[channel]["nrmse"] is not None])
        correlation = np.asarray([frame[channel]["correlation"] for frame in records if frame[channel]["correlation"] is not None])
        display[name] = {}
        for column, (values, label, title) in enumerate((
            (nrmse, "RMSE / observed spatial SD", "Held-out reconstruction error"),
            (correlation, "Spatial correlation", "Held-out spatial agreement"),
        )):
            lower, upper = np.percentile(values, (1, 99))
            if not upper > lower:
                padding = max(abs(float(lower)) * 0.01, 1e-6)
                lower, upper = lower - padding, upper + padding
            clipped = np.clip(values, lower, upper)
            axes[channel, column].hist(clipped, bins=np.linspace(lower, upper, 41), histtype="step", linewidth=1.8)
            below = int(np.count_nonzero(values < lower))
            above = int(np.count_nonzero(values > upper))
            axes[channel, column].set(
                xlabel=label, ylabel="Frames",
                title=f"{name}: {title}\n1st–99th percentile display range",
                xlim=(lower, upper),
            )
            axes[channel, column].text(
                0.02, 0.96, f"tails clipped to edges: {below} low, {above} high",
                transform=axes[channel, column].transAxes, va="top", fontsize=8,
            )
            axes[channel, column].grid(alpha=0.2)
            display[name]["nrmse" if column == 0 else "correlation"] = {
                "display_minimum": float(lower), "display_maximum": float(upper),
                "clipped_low": below, "clipped_high": above,
            }
    figure.savefig(output / "01_validation_population.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / "01_validation_population.pdf", bbox_inches="tight")
    plt.close(figure)
    return display


#%% Chronological validation workflow

def run(arguments):
    run_directory = arguments.run_directory.expanduser().resolve()
    cache_directory = arguments.cache_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty model-validation output directory")
    output.mkdir(parents=True, exist_ok=True)
    device = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    if device == "auto":
        device = "cpu"

    artifact_path = run_directory / "wic_cvae.pt"
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if model.conditioning is not None or model.kp_conditioning is None:
        raise ValueError("this poster validation currently requires the Kp-conditioned WIC/SI12 model")
    manifest_path = cache_directory / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    cache_format = manifest.get("cache_format")
    if cache_format not in WIC_SI12_CACHE_FORMATS:
        raise ValueError("expected a completed WIC/SI12 cache")
    expected_cache = artifact.get("data_provenance", {}).get("cache_manifest_sha256")
    if expected_cache is not None and expected_cache != manifest_sha256:
        raise ValueError("held-out cache manifest differs from the cache used to train the model")
    dataset = AutomaticFieldDataset(
        cache_directory, arguments.split, cache_mib=arguments.cache_mib,
        max_open_orbits=arguments.max_open_orbits, include_uncertainty=False,
        expected_cache_format=cache_format,
    )
    indices = np.arange(len(dataset))
    if arguments.maximum_frames is not None and arguments.maximum_frames < len(indices):
        generator = np.random.default_rng(arguments.seed)
        indices = np.sort(generator.choice(indices, arguments.maximum_frames, replace=False))
    subset = Subset(dataset, indices.tolist())
    loader = DataLoader(subset, batch_size=arguments.batch_size, shuffle=False, num_workers=0)

    observed_batches, reconstructed_batches, valid_batches = [], [], []
    with torch.no_grad():
        for batch in loader:
            encoder_input = batch["encoder_input"].to(device)
            valid = batch["mask"].to(device)
            kp = batch["kp"].to(device)
            mu, _logvar = model.encode(encoder_input, valid, kp=kp)
            reconstructed = transform.inverse(model.decode(mu, kp=kp))
            observed = transform.inverse(batch["target"].to(device))
            observed_batches.append(observed.cpu().numpy())
            reconstructed_batches.append(reconstructed.cpu().numpy())
            valid_batches.append(valid.cpu().numpy())
    observed = np.concatenate(observed_batches)
    reconstructed = np.concatenate(reconstructed_batches)
    valid = np.concatenate(valid_batches)
    valid_full = np.broadcast_to(valid, observed.shape)
    records = frame_metrics(observed, reconstructed, valid_full)
    valid_fraction = np.mean(
        valid_full & np.isfinite(observed), axis=(2, 3)
    )
    representative_eligible = np.all(
        valid_fraction >= arguments.minimum_valid_fraction, axis=1
    )
    representatives, joint_scores, representative_quantiles = select_representatives(
        records, eligible=representative_eligible
    )

    rows = []
    for local, dataset_index in enumerate(indices):
        row = {
            "dataset_index": int(dataset_index),
            "orbit": int(dataset.frame_orbit[dataset_index]),
            "source_frame": int(dataset.source_frames[dataset_index]),
            "time": str(dataset.frame_time[dataset_index]),
            "kp": float(dataset.kp[dataset_index]),
            "joint_nrmse": float(joint_scores[local]),
            "wic_valid_fraction": float(valid_fraction[local, 0]),
            "si12_valid_fraction": float(valid_fraction[local, 1]),
            "minimum_valid_fraction": float(np.min(valid_fraction[local])),
            "representative_eligible": bool(representative_eligible[local]),
        }
        for channel, prefix in enumerate(("wic", "si12")):
            row.update({f"{prefix}_{key}": value for key, value in records[local][channel].items()})
        rows.append(row)
    with (output / "01_validation_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected_rows = []
    for number, (representative, quantile) in enumerate(zip(representatives, representative_quantiles), start=1):
        selected = dict(rows[representative])
        selected["performance_percentile"] = float(100 * quantile)
        selected_rows.append(selected)
        title = (
            f"Held-out performance percentile {100 * quantile:.0f}; "
            f"orbit {selected['orbit']}, frame {selected['source_frame']}, "
            f"{selected['time']}, Kp={selected['kp']:g}; "
            f"joint NRMSE={selected['joint_nrmse']:.3g}; "
            f"minimum coverage={100 * selected['minimum_valid_fraction']:.1f}%"
        )
        stem = f"01_representative_reconstruction_{number:02d}_p{round(100 * quantile):02d}"
        plot_representative(
            observed[representative], reconstructed[representative],
            valid_full[representative], title, output, stem, arguments.dpi,
        )
    population_display = plot_population(records, output, arguments.dpi)
    summary = {
        "purpose": "held-out posterior reconstruction validation",
        "model": str(artifact_path),
        "model_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "cache": str(cache_directory),
        "cache_manifest_sha256": manifest_sha256,
        "split": arguments.split,
        "frames": len(rows),
        "representative_selection": (
            "ten distinct frames at the 5th, 15th, ..., 95th percentiles of "
            "joint WIC/SI12 NRMSE after requiring the minimum valid fraction "
            "in each field"
        ),
        "representative_minimum_valid_fraction_per_field": arguments.minimum_valid_fraction,
        "representative_eligible_frames": int(np.count_nonzero(representative_eligible)),
        "representatives": selected_rows,
        "population_figure_display": population_display,
        "metrics": {
            name: {
                metric: float(np.nanmedian([frame[channel][metric] for frame in records if frame[channel][metric] is not None]))
                for metric in ("rmse", "nrmse", "correlation", "bias")
            }
            for channel, name in enumerate(("WIC", "SI12"))
        },
        "interpretation": "decoder evaluated at posterior mean; scores use only originally valid held-out pixels",
    }
    (output / "01_model_validation.json").write_text(json.dumps(summary, indent=2) + "\n")
    dataset.close()
    print(json.dumps(summary["metrics"], indent=2))
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("cache_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--maximum-frames", type=int, help="seeded diagnostic subset; omit for the full split")
    parser.add_argument(
        "--minimum-valid-fraction", type=float, default=0.6,
        help="minimum finite/valid grid fraction required in each field for the ten representative figures",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cache-mib", type=int, default=4096)
    parser.add_argument("--max-open-orbits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args(argv)
    if min(arguments.batch_size, arguments.cache_mib, arguments.max_open_orbits, arguments.dpi) < 1:
        parser.error("batch, cache, open-orbit and DPI settings must be positive")
    if arguments.maximum_frames is not None and arguments.maximum_frames < 1:
        parser.error("--maximum-frames must be positive")
    if not np.isfinite(arguments.minimum_valid_fraction) or not 0 <= arguments.minimum_valid_fraction <= 1:
        parser.error("--minimum-valid-fraction must lie between zero and one")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
