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


def select_representative(records):
    """Select the frame nearest median joint NRMSE, without cherry-picking."""

    scores = np.array([
        np.mean([entry["nrmse"] for entry in frame if entry["nrmse"] is not None])
        for frame in records
    ])
    finite = np.flatnonzero(np.isfinite(scores))
    if not finite.size:
        raise ValueError("no held-out frame has a finite reconstruction score")
    median = np.median(scores[finite])
    return int(finite[np.argmin(abs(scores[finite] - median))]), scores


#%% Poster figures

def plot_representative(observed, reconstructed, valid, title, output, dpi):
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
    figure.savefig(output / "01_representative_reconstruction.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / "01_representative_reconstruction.pdf", bbox_inches="tight")
    plt.close(figure)


def plot_population(records, output, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for channel, name in enumerate(("WIC", "SI12")):
        nrmse = [frame[channel]["nrmse"] for frame in records if frame[channel]["nrmse"] is not None]
        correlation = [frame[channel]["correlation"] for frame in records if frame[channel]["correlation"] is not None]
        axes[0].hist(nrmse, bins=40, histtype="step", linewidth=1.8, label=name)
        axes[1].hist(correlation, bins=np.linspace(-1, 1, 41), histtype="step", linewidth=1.8, label=name)
    axes[0].set(xlabel="Frame RMSE / observed spatial SD", ylabel="Frames", title="Held-out reconstruction error")
    axes[1].set(xlabel="Spatial correlation", ylabel="Frames", title="Held-out spatial agreement", xlim=(-1, 1))
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend()
    figure.savefig(output / "01_validation_population.png", dpi=dpi, bbox_inches="tight")
    figure.savefig(output / "01_validation_population.pdf", bbox_inches="tight")
    plt.close(figure)


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
    representative, joint_scores = select_representative(records)

    rows = []
    for local, dataset_index in enumerate(indices):
        row = {
            "dataset_index": int(dataset_index),
            "orbit": int(dataset.frame_orbit[dataset_index]),
            "source_frame": int(dataset.source_frames[dataset_index]),
            "time": str(dataset.frame_time[dataset_index]),
            "kp": float(dataset.kp[dataset_index]),
            "joint_nrmse": float(joint_scores[local]),
        }
        for channel, prefix in enumerate(("wic", "si12")):
            row.update({f"{prefix}_{key}": value for key, value in records[local][channel].items()})
        rows.append(row)
    with (output / "01_validation_frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected = rows[representative]
    title = f"Held-out orbit {selected['orbit']}, frame {selected['source_frame']}, {selected['time']}, Kp={selected['kp']:g}"
    plot_representative(observed[representative], reconstructed[representative], valid_full[representative], title, output, arguments.dpi)
    plot_population(records, output, arguments.dpi)
    summary = {
        "purpose": "held-out posterior reconstruction validation",
        "model": str(artifact_path),
        "model_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "cache": str(cache_directory),
        "cache_manifest_sha256": manifest_sha256,
        "split": arguments.split,
        "frames": len(rows),
        "representative_selection": "frame nearest the median joint WIC/SI12 NRMSE",
        "representative": selected,
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
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
