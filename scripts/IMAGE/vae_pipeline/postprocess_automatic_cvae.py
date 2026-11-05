"""Plot cached held-out frames and sample a saved conductance CVAE at fixed Kp.

No training. Existing run/cache files and old plotting workflows are unchanged.
Samples stay on disk; covariance memory depends on grid size, not sample count.
"""

#%% Imports

import argparse
import json
import mmap
from pathlib import Path
import shutil

import numpy as np
import torch
from scipy.sparse.linalg import eigsh
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.CVAE.prepare_cvae_hp_cache import AutomaticConductanceDataset, read_cache, file_sha256
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import (
    plt, image_panel, robust_limits, save_figure, physical_decode,
    infer_posterior_prior, infer_heldout_inpainting, inpainting_skill,
    posterior_reconstruction_skill, plot_posterior_reconstruction_skill,
    plot_loss_history, plot_sample_mosaic, plot_ensemble_moments, plot_covariance_modes
)


#%% Materialize only the requested held-out subset, preserving cached masks

class HeldoutFrames(Dataset):
    def __init__(self, dataset, indices, support):
        frames = [dataset[int(index)] for index in indices]
        self.indices = np.asarray(indices, dtype=int)
        self.values = np.stack([frame["physical_target"].numpy() for frame in frames])
        self.standard_deviation = np.stack([frame["physical_standard_deviation"].numpy() for frame in frames])
        self.transformed_variance = np.stack([frame["variance"].numpy() for frame in frames])
        self.common_valid = np.stack([frame["mask"].numpy() for frame in frames]) & support[None, None]
        self.target = np.where(self.common_valid, np.stack([frame["target"].numpy() for frame in frames]), 0.0)
        self.weight = np.where(self.common_valid, np.stack([frame["weight"].numpy() for frame in frames]), 0.0)
        self.encoder_input = np.concatenate((self.target, self.weight), axis=1)
        self.kp = dataset.kp[indices]
        self.orbits = dataset.frame_orbit[indices]
        self.times = dataset.frame_time[indices]
        self.source_frames = dataset.source_frames[indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        return {"encoder_input": torch.from_numpy(self.encoder_input[index]), "mask": torch.from_numpy(self.common_valid[index]), "kp": torch.tensor(self.kp[index], dtype=torch.float32)}


def plot_reconstructions(heldout, ensembles, inpainting, split, output, created, examples):
    indices = np.linspace(0, len(heldout) - 1, min(examples, len(heldout)), dtype=int)
    figure, axes = plt.subplots(2 * len(indices), 5, figsize=(17, 5.4 * len(indices)), squeeze=False, layout="constrained")
    titles = ("held-out retrieval", "full-input reconstruction", "input after removal", "damaged-input reconstruction", "damaged-input error")
    for example, index in enumerate(indices):
        for channel, name in enumerate(("Hall", "Pedersen")):
            valid = ensembles["valid"][index, channel]
            observed = np.where(valid, heldout.values[index, channel], np.nan)
            full = np.where(valid, ensembles["posterior_mean"][index, channel], np.nan)
            damaged = np.where(inpainting["input_valid"][index, channel], observed, np.nan)
            reconstructed = np.where(valid, inpainting["reconstruction"][index, channel], np.nan)
            fields = (observed, full, damaged, reconstructed, reconstructed - observed)
            limits = robust_limits(np.stack(fields[:4]))
            identity = f"{name}; {split} frame {heldout.indices[index]}, Kp {heldout.kp[index]:.2f}"
            for column, field in enumerate(fields):
                image_panel(axes[2 * example + channel, column], field, titles[column] + "\n" + identity, "S", cmap="coolwarm" if column == 4 else "viridis", limits=robust_limits(field, signed=True) if column == 4 else limits)
    figure.suptitle("Decoder of posterior latent mean; IMAGE retrieval is not independent truth")
    save_figure(figure, output, "heldout_inpainting_examples.png", created)


#%% Batched decoding to disk; no whole-ensemble allocation

def release_pages(array):
    # Linux can reclaim mapped pages immediately; other systems use normal paging.
    if hasattr(array._mmap, "madvise") and hasattr(mmap, "MADV_DONTNEED"):
        array._mmap.madvise(mmap.MADV_DONTNEED)


def save_samples(model, transform, support, output, count, batch_size, kp, seed, device):
    samples = np.lib.format.open_memmap(output / "conductance_samples.npy", mode="w+", dtype=np.float32, shape=(count, 2, *model.image_shape))
    latent_file = np.lib.format.open_memmap(output / "latent_samples.npy", mode="w+", dtype=np.float32, shape=(count, model.latent_dim))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    try:
        with torch.no_grad():
            for start in tqdm(range(0, count, batch_size), desc=f"Prior samples, Kp={kp:g}", unit="batch"):
                stop = min(start + batch_size, count)
                latent = torch.randn(stop - start, model.latent_dim, generator=generator)
                decoded = model.decode(latent.to(device), kp=kp)
                physical = physical_decode(decoded, transform, support)
                if not np.isfinite(physical).all():
                    raise ValueError("decoder produced nonfinite supported conductance; ensemble is incomplete")
                samples[start:stop] = physical
                latent_file[start:stop] = latent.numpy()
                samples.flush()
                latent_file.flush()
                release_pages(samples)
                release_pages(latent_file)
    finally:
        samples._mmap.close()
        latent_file._mmap.close()


#%% Stable joint covariance, including cross-sensor blocks

def streaming_statistics(filename, batch_size, log_space=False):
    samples = np.load(filename, mmap_mode="r", allow_pickle=False)
    state_shape = samples.shape[1:]
    dimension = int(np.prod(state_shape))
    mean = np.zeros(dimension, dtype=np.float64)
    scatter = np.zeros((dimension, dimension), dtype=np.float64)
    count = 0
    try:
        for start in tqdm(range(0, len(samples), batch_size), desc="Log covariance" if log_space else "Physical covariance", unit="batch"):
            batch = np.array(samples[start:start + batch_size], dtype=np.float64).reshape(-1, dimension)
            release_pages(samples)
            if log_space:
                batch = np.log10(batch + 1.0)
            if not np.isfinite(batch).all():
                raise ValueError("covariance requires finite ensemble members")
            current_count = len(batch)
            batch_mean = batch.mean(axis=0)
            delta = batch_mean - mean
            new_count = count + current_count
            centered = batch - batch_mean
            correction = count * current_count / new_count
            # Row blocks bound matrix-product temporaries, independent of samples.
            for row in range(0, dimension, 256):
                stop = min(row + 256, dimension)
                scatter[row:stop] += centered[:, row:stop].T @ centered
                scatter[row:stop] += correction * delta[row:stop, None] * delta[None, :]
            mean += delta * current_count / new_count
            count = new_count
    finally:
        samples._mmap.close()
    if count < 2:
        raise ValueError("at least two samples are needed for covariance")
    scatter /= count - 1
    return mean.reshape(state_shape), scatter


def plot_teleconnections(covariance, support, grid_id, kp, output, created):
    variance = np.diag(covariance)
    field_size = support.size
    figure, axes = plt.subplots(2, 2, figsize=(10, 8), layout="constrained")
    for source, source_name in enumerate(("Hall", "Pedersen")):
        anchor = source * field_size + grid_id
        scale = np.sqrt(np.maximum(variance[anchor] * variance, 0.0))
        correlation = np.full_like(variance, np.nan)
        np.divide(covariance[:, anchor], scale, out=correlation, where=scale > 0)
        maps = np.clip(correlation.reshape(2, *support.shape), -1, 1)
        for target, target_name in enumerate(("Hall", "Pedersen")):
            field = np.where(support, maps[target], np.nan)
            image_panel(axes[source, target], field, f"{source_name} anchor → {target_name}", "correlation", cmap="coolwarm", limits=(-1, 1))
            row, column = np.unravel_index(grid_id, support.shape)
            axes[source, target].scatter(column, row, marker="x", color="black")
    figure.suptitle(f"Prior correlations at Kp={kp:g}; flattened grid ID {grid_id}")
    save_figure(figure, output, "conductance_correlation_maps.png", created)


#%% Saved checkpoint + exact cache provenance, then held-out diagnostics

def run(arguments):
    run_directory = arguments.run_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError("choose a new output directory; existing diagnostics are preserved")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if arguments.device == "auto" else arguments.device
    model, transform, artifact = load_run_artifact(run_directory / "automatic_cvae.pt", device=device)
    if artifact["artifact_type"] != "conductance_vae_run_v1" or model.kp_conditioning is None:
        raise ValueError("expected the native Kp-conditioned conductance model")
    provenance = artifact["data_provenance"]
    cache = (arguments.cache or Path(provenance["cache_directory"])).expanduser().resolve()
    metadata = read_cache(cache)
    if artifact["training_configuration"].get("dataset_backend") != metadata["cache_format"] or metadata["manifest_sha256"] != provenance["cache_manifest_sha256"]:
        raise ValueError("cache manifest differs from the saved training inputs")
    if transform.to_dict() != metadata["transform"]:
        raise ValueError("cache and trained model have different normalization")
    support = np.asarray(provenance["training_support"], dtype=bool)
    if support.shape != tuple(model.image_shape) or not support.any() or not np.array_equal(support, np.load(cache / "training_support.npy", allow_pickle=False)):
        raise ValueError("cache and model training support differ")
    if arguments.grid_id is not None and (not 0 <= arguments.grid_id < support.size or not support.flat[arguments.grid_id]):
        raise ValueError("grid ID must lie on the supported flattened grid")
    if not arguments.diagnostics_only:
        bytes_needed = arguments.samples * (2 * support.size + model.latent_dim) * 4 + 2 * (2 * support.size)**2 * 8
        output.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(output.parent).free < 1.1 * bytes_needed:
            raise OSError("insufficient free disk for samples and covariance outputs")

    dataset = AutomaticConductanceDataset(cache, arguments.split)
    try:
        indices = np.asarray(arguments.frames, dtype=int) if arguments.frames is not None else np.linspace(0, len(dataset) - 1, min(arguments.heldout_count, len(dataset)), dtype=int)
        if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(dataset)):
            raise ValueError("held-out frame IDs must be unique and inside the cached split")
        heldout = HeldoutFrames(dataset, indices, support)
    finally:
        dataset.close()
    output.mkdir(parents=True)
    created = []
    plot_loss_history(artifact["history"], output, created)
    print(f"Reconstructing {len(heldout)} cached {arguments.split} frames on {device}.", flush=True)
    ensembles = infer_posterior_prior(model, transform, heldout, arguments.batch_size, device, arguments.seed, support)
    inpainting = infer_heldout_inpainting(model, transform, heldout, arguments.batch_size, device, arguments.hole_size, policy=arguments.hole_policy, seed=arguments.seed, support_mask=support)
    skill = posterior_reconstruction_skill(ensembles, heldout, transform, artifact["training_configuration"]["loss_configuration"])
    plot_posterior_reconstruction_skill(skill, output, created, family="conductance")
    plot_reconstructions(heldout, ensembles, inpainting, arguments.split, output, created, arguments.examples)
    np.savez_compressed(
        output / "heldout_reconstructions.npz",
        frame_indices=indices, orbit=heldout.orbits, time=heldout.times, source_frame=heldout.source_frames,
        observed=heldout.values, valid=ensembles["valid"], quality=heldout.weight,
        standard_deviation=heldout.standard_deviation, kp=heldout.kp,
        full_reconstruction=ensembles["posterior_mean"], inpainted=inpainting["reconstruction"],
        input_valid=inpainting["input_valid"], removed=inpainting["removed"], hole_bounds=inpainting["bounds"],
        mu=ensembles["mu"], logvar=ensembles["logvar"]
    )
    report = {
        "checkpoint": str(run_directory / "automatic_cvae.pt"), "checkpoint_sha256": file_sha256(run_directory / "automatic_cvae.pt"),
        "cache_directory": str(cache), "cache_manifest_sha256": metadata["manifest_sha256"], "split": arguments.split,
        "heldout_frame_indices": indices.tolist(), "heldout_frame_kp": heldout.kp.tolist(), "seed": arguments.seed,
        "batch_size": arguments.batch_size, "selected_epoch": artifact["training_configuration"]["selected_epoch"],
        "grid_id": provenance["grid_id"], "field_names": ["Hall", "Pedersen"], "field_units": "S",
        "reconstruction_skill": skill, "inpainting_skill": inpainting_skill(ensembles, inpainting, transform, family="conductance"),
        "hole_policy": arguments.hole_policy, "hole_size": arguments.hole_size,
        "state_order": "Hall C-flattened grid, then Pedersen C-flattened grid",
        "reference": "native IMAGE conductance_cs retrieval, not independent truth",
        "reconstruction": "decoder of posterior latent mean, not mean decoded posterior or calibrated uncertainty",
        "unsupported_cells": "zero storage placeholders; excluded from scoring and plots", "uncertainty_calibrated": False
    }
    report["ensemble_generated"] = not arguments.diagnostics_only
    np.save(output / "training_support.npy", support)
    np.savez_compressed(output / "grid.npz", **{key: np.asarray(value) for key, value in provenance["grid"].items()})
    del ensembles, inpainting, heldout

    # Fixed-Kp prior: independent of the held-out observations above.
    if not arguments.diagnostics_only:
        conditioning = model.kp_conditioning
        extrapolation = not conditioning["training_min"] <= arguments.kp <= conditioning["training_max"]
        if extrapolation:
            print("Warning: requested Kp is outside the training range.", flush=True)
        save_samples(model, transform, support, output, arguments.samples, arguments.batch_size, arguments.kp, arguments.seed, device)
        samples = np.load(output / "conductance_samples.npy", mmap_mode="r", allow_pickle=False)
        sample_ids = np.linspace(0, arguments.samples - 1, min(5, arguments.samples), dtype=int)
        examples = samples[sample_ids].copy()
        samples._mmap.close()
        plot_sample_mosaic(np.where(support[None, None], examples, np.nan), sample_ids, "conductance", output, created)
        mean, covariance = streaming_statistics(output / "conductance_samples.npy", arguments.batch_size)
        np.save(output / "conductance_mean.npy", mean)
        np.save(output / "conductance_covariance.npy", covariance)
        variance = np.maximum(np.diag(covariance), 0.0).reshape(mean.shape)
        moments = {"conductance": {"mean": np.where(support[None], mean, np.nan), "standard_deviation": np.where(support[None], np.sqrt(variance), np.nan)}}
        plot_ensemble_moments(moments, output, created)
        total_variance = float(np.trace(covariance))
        if total_variance <= 0:
            raise ValueError("sampled prior has no variance; PCA/teleconnections are undefined")
        modes_count = min(arguments.modes, arguments.samples - 1, covariance.shape[0] - 1)
        eigenvalues, vectors = eigsh(covariance, k=modes_count, which="LA", v0=np.random.default_rng(arguments.seed).standard_normal(covariance.shape[0]), tol=1e-6)
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        modes = vectors[:, order].T.reshape(modes_count, *mean.shape)
        np.savez_compressed(output / "conductance_pca.npz", eigenvalues=eigenvalues, modes=modes, explained_variance=eigenvalues / total_variance)
        plot_covariance_modes(eigenvalues, np.ma.array(modes, mask=np.broadcast_to(~support, modes.shape)), total_variance, "conductance", output, created, min(4, modes_count))
        grid_id = int(np.argmax(variance[0])) if arguments.grid_id is None else arguments.grid_id
        plot_teleconnections(covariance, support, grid_id, arguments.kp, output, created)
        del covariance
        log_mean, log_covariance = streaming_statistics(output / "conductance_samples.npy", arguments.batch_size, log_space=True)
        np.save(output / "conductance_log10p_mean.npy", log_mean)
        np.save(output / "conductance_log10p_covariance.npy", log_covariance)
        report["prior"] = {
            "kp": arguments.kp, "kp_extrapolation": extrapolation, "samples": arguments.samples,
            "sample_example_ids": sample_ids.tolist(), "grid_id": grid_id, "covariance_units": "S^2",
            "log_covariance_units": "dex^2 of log10(conductance/S + 1), not standardized",
            "explained_variance": (eigenvalues / total_variance).tolist()
        }
    report["figures"] = created
    (output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {len(created)} figures and numerical diagnostics to {output}.", flush=True)
    return report


#%% Command line

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, help="relocated copy of the original unchanged cache")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--frames", type=int, nargs="+", help="explicit IDs in the retained cached held-out split")
    for name, default in (("heldout-count", 128), ("examples", 3), ("batch-size", 256), ("samples", 5000), ("modes", 4), ("hole-size", 12), ("seed", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--hole-policy", choices=("auroral", "random"), default="auroral")
    parser.add_argument("--kp", type=float, help="fixed Kp for the prior ensemble; does not replace held-out frame Kp")
    parser.add_argument("--grid-id", type=int, help="teleconnection anchor in flattened grid; default maximum Hall variance")
    parser.add_argument("--diagnostics-only", action="store_true", help="skip prior sampling and covariance")
    arguments = parser.parse_args(argv)
    if any(getattr(arguments, name) < 1 for name in ("heldout_count", "examples", "batch_size", "modes")) or arguments.samples < 2:
        parser.error("counts must be positive; samples must be at least two")
    if not 1 <= arguments.hole_size < 46:
        parser.error("hole size must lie between1 and45")
    if not arguments.diagnostics_only and (arguments.kp is None or not np.isfinite(arguments.kp) or not 0 <= arguments.kp <= 9):
        parser.error("sampling requires --kp between0 and9")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
