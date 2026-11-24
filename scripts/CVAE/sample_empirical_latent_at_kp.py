"""Sample a Kp-local empirical aggregate posterior from a trained WIC/SI12 CVAE."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.cache import AutomaticFieldDataset, WIC_SI12_CACHE_FORMATS
from scripts.CVAE.sample_wic_at_kp import plot_wic, plot_wic_sample_mosaic


FIELD_NAMES = ("wic_corrected", "si12")


#%% Deterministic Kp-local bank selection

def evenly_spaced_subset(indices, maximum):
    """Retain all indices or a mission-spanning deterministic subset."""

    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError("indices must be one-dimensional")
    if maximum is None:
        return indices.copy()
    maximum = int(maximum)
    if maximum < 1:
        raise ValueError("maximum must be positive or None")
    if len(indices) <= maximum:
        return indices.copy()
    positions = np.linspace(0, len(indices) - 1, maximum).astype(np.int64)
    return indices[positions]


def select_bank_indices(kp, targets, half_width, maximum_per_target):
    """Select all or a bounded number of frames around each Kp category."""

    kp = np.asarray(kp, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if (
        kp.ndim != 1 or targets.ndim != 1 or not len(targets)
        or not np.isfinite(kp).all() or not np.isfinite(targets).all()
        or not np.isfinite(half_width) or half_width < 0
    ):
        raise ValueError("Kp arrays and half-width must be finite")
    selections = []
    available = {}
    for target in targets:
        candidates = np.flatnonzero(np.abs(kp - target) <= half_width + 1e-6)
        available[str(float(target))] = int(len(candidates))
        if not len(candidates):
            raise ValueError(f"no training frames lie near Kp={target:g}")
        selections.append(evenly_spaced_subset(candidates, maximum_per_target))
    return np.unique(np.concatenate(selections)), available


#%% Bounded posterior encoding

def encode_bank(model, loader, device):
    """Encode one ordered training subset without retaining image tensors."""

    mus, logvars, kp = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Encoding empirical latent bank", unit="batch"):
            encoder_input = batch["encoder_input"].to(
                device, non_blocking=device.type == "cuda"
            )
            valid = batch["mask"].to(
                device, non_blocking=device.type == "cuda"
            )
            batch_kp = batch["kp"].to(
                device, non_blocking=device.type == "cuda"
            )
            mu, logvar = model.encode(encoder_input, valid, kp=batch_kp)
            mus.append(mu.cpu().numpy().astype(np.float32))
            logvars.append(logvar.cpu().numpy().astype(np.float32))
            kp.append(batch["kp"].numpy().astype(np.float32))
    if not mus:
        raise ValueError("selected empirical latent bank is empty")
    return np.concatenate(mus), np.concatenate(logvars), np.concatenate(kp)


def choose_latents(mu, logvar, pool, draws, rng, posterior_sample):
    """Choose complete joint rows and optionally sample their diagonal posterior."""

    pool = np.asarray(pool, dtype=np.int64)
    if pool.ndim != 1 or not len(pool) or draws < 1:
        raise ValueError("latent sampling requires a nonempty pool and positive draws")
    source_rows = rng.choice(pool, size=draws, replace=len(pool) < draws)
    latent = mu[source_rows].copy()
    if posterior_sample:
        noise = rng.standard_normal(latent.shape).astype(np.float32)
        latent += noise * np.exp(0.5 * logvar[source_rows])
    return latent.astype(np.float32), source_rows


def decode_latents(model, transform, latent, kp, batch_size, device, support):
    """Decode one empirical latent ensemble in bounded GPU batches."""

    outputs = np.empty(
        (len(latent), len(FIELD_NAMES), *model.image_shape), dtype=np.float32
    )
    model.eval()
    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            stop = min(start + batch_size, len(latent))
            values = torch.from_numpy(latent[start:stop]).to(device)
            condition = torch.full(
                (stop - start,), float(kp), dtype=torch.float32, device=device
            )
            decoded = model.decode(values, kp=condition)
            outputs[start:stop] = transform.inverse(decoded.cpu().numpy())
    outputs = np.where(support[None], outputs, 0)
    return outputs


#%% Compact products and galleries

def summarize_and_plot(
    samples, support, mlat, mlt, target_kp, source_kp, mode, output,
    wic_maximum, si12_maximum,
):
    """Save individual coherent members and compact marginal summaries."""

    limits = (wic_maximum, si12_maximum)
    products = {}
    source_label = (
        f"source Kp {np.min(source_kp):g}--{np.max(source_kp):g}"
    )
    mode_label = "posterior means" if mode == "mean" else "posterior samples"
    mode_prefix = "posterior_mean" if mode == "mean" else "posterior_draw"
    title = f"{mode_label}, decoded at Kp {target_kp:g}; {source_label}"
    for channel, (field, label) in enumerate(zip(FIELD_NAMES, ("Corrected WIC", "SI12"))):
        values = samples[:, channel]
        mean = np.mean(values, axis=0)
        quantiles = np.quantile(values, (0.1, 0.5, 0.9), axis=0)
        mosaic = values[:min(36, len(values))].copy()
        indices = np.arange(len(mosaic), dtype=np.int32)
        prefix = f"{field}_{mode_prefix}"
        plot_wic(
            mean, quantiles, support[channel], mlat, mlt, title, output,
            limits[channel], field_label=label, file_prefix=prefix,
            distribution_label="empirical posterior",
        )
        plot_wic_sample_mosaic(
            mosaic, indices, support[channel], mlat, mlt, title, output,
            limits[channel], field_label=label, file_prefix=prefix,
        )
        products.update({
            f"{field}_{mode}_mean": mean,
            f"{field}_{mode}_q10": quantiles[0],
            f"{field}_{mode}_q50": quantiles[1],
            f"{field}_{mode}_q90": quantiles[2],
            f"{field}_{mode}_mosaic": mosaic,
        })
    return products


#%% End-to-end empirical conditional sampling

def run(arguments):
    run_directory = arguments.run_directory.expanduser().resolve()
    cache = arguments.cache.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose an empty empirical-sampling output directory")
    output.mkdir(parents=True, exist_ok=True)

    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if arguments.device == "auto" else arguments.device
    device = torch.device(device_name)
    torch.set_num_threads(arguments.threads)
    artifact_path = run_directory / "wic_cvae.pt"
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if (
        tuple(model.field_names) != FIELD_NAMES
        or model.kp_conditioning is None or model.conditioning is not None
    ):
        raise ValueError("requires the paired scalar-Kp WIC/SI12 model")

    cache_format = json.loads((cache / "manifest.json").read_text())["cache_format"]
    if cache_format not in WIC_SI12_CACHE_FORMATS:
        raise ValueError("requires a paired WIC/SI12 cache")
    workers = arguments.num_workers
    mapping_budget = arguments.cache_mib / max(1, workers)
    dataset = AutomaticFieldDataset(
        cache, "train", cache_mib=mapping_budget,
        max_open_orbits=arguments.max_open_orbits,
        include_uncertainty=False, expected_cache_format=cache_format,
    )
    expected_hash = artifact["data_provenance"]["cache_manifest_sha256"]
    if dataset.metadata["manifest_sha256"] != expected_hash:
        raise ValueError("run artifact and empirical-sampling cache do not match")

    targets = np.asarray(arguments.kp, dtype=float)
    maximum_per_target = (
        None if arguments.all_matching_frames
        else arguments.bank_frames_per_kp
    )
    bank_indices, available = select_bank_indices(
        dataset.kp, targets, arguments.kp_half_width,
        maximum_per_target,
    )
    subset = Subset(dataset, bank_indices.tolist())
    loader_options = {
        "batch_size": arguments.encode_batch_size,
        "shuffle": False,
        "num_workers": workers,
        "pin_memory": device.type == "cuda",
    }
    if workers:
        loader_options.update(
            multiprocessing_context="spawn",
            prefetch_factor=arguments.prefetch_factor,
        )
    loader = DataLoader(subset, **loader_options)
    mu, logvar, bank_kp = encode_bank(model, loader, device)
    if not np.allclose(bank_kp, dataset.kp[bank_indices]):
        raise RuntimeError("encoded bank order does not match selected cache rows")

    support = np.asarray(
        artifact["data_provenance"]["training_support"], dtype=bool
    )
    grid = artifact["data_provenance"]["grid"]
    mlat = np.asarray(grid["mlat"], dtype=float)
    mlt = np.asarray(grid["mlt"], dtype=float)
    expected_shape = (len(FIELD_NAMES), *model.image_shape)
    if support.shape != expected_shape:
        raise ValueError("artifact support does not align with model outputs")

    bank_products = {
        "mu": mu, "logvar": logvar, "kp": bank_kp,
        "dataset_index": bank_indices,
        "orbit": dataset.frame_orbit[bank_indices],
        "source_frame": dataset.source_frames[bank_indices],
        "time": dataset.frame_time[bank_indices],
    }
    np.savez_compressed(output / "empirical_latent_bank.npz", **bank_products)

    summary = {
        "run_directory": str(run_directory),
        "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
        "cache_directory": str(cache),
        "cache_manifest_sha256": dataset.metadata["manifest_sha256"],
        "split": "train",
        "bank_frames": int(len(bank_indices)),
        "bank_frames_per_kp_limit": maximum_per_target,
        "all_matching_frames": arguments.all_matching_frames,
        "available_training_frames_by_target": available,
        "kp_half_width": arguments.kp_half_width,
        "draws": 0 if arguments.bank_only else arguments.draws,
        "posterior_mode": None if arguments.bank_only else arguments.posterior_mode,
        "seed": arguments.seed,
        "bank_selection_policy": (
            "every training frame matching each requested discrete Kp category"
            if arguments.all_matching_frames else
            "mission-spanning capped subset around each requested Kp category"
        ),
        "sampling_policy": (
            None if arguments.bank_only else
            "complete joint global+spatial posterior rows sampled uniformly "
            "from Kp-local training encodings; dimensions and groups are never "
            "resampled independently; decoder receives the requested Kp category"
        ),
        "selections": [],
    }
    if arguments.bank_only:
        (output / "sampling_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n"
        )
        dataset.close()
        print(
            f"Encoded all {len(bank_indices)} selected latent rows; "
            "gallery decoding skipped", flush=True,
        )
        return summary

    modes = (
        ("mean", False), ("sample", True)
    ) if arguments.posterior_mode == "both" else (
        (arguments.posterior_mode, arguments.posterior_mode == "sample"),
    )
    selections = []
    for number, target in enumerate(targets):
        pool = np.flatnonzero(
            np.abs(bank_kp - target) <= arguments.kp_half_width + 1e-6
        )
        if len(pool) < arguments.minimum_pool:
            raise ValueError(
                f"Kp={target:g} has only {len(pool)} encoded source frames; "
                "increase --bank-frames-per-kp or --kp-half-width"
            )
        destination = output / f"kp_{number:02d}_{target:g}"
        destination.mkdir()
        rng = np.random.default_rng(arguments.seed)
        selected_rows = rng.choice(
            pool, size=arguments.draws, replace=len(pool) < arguments.draws
        )
        source_kp = bank_kp[selected_rows]
        saved = {
            "source_bank_row": selected_rows,
            "source_kp": source_kp,
            "source_dataset_index": bank_indices[selected_rows],
            "source_orbit": dataset.frame_orbit[bank_indices[selected_rows]],
            "source_frame": dataset.source_frames[bank_indices[selected_rows]],
            "source_time": dataset.frame_time[bank_indices[selected_rows]],
            "mlat": mlat, "mlt": mlt, "support": support,
        }
        for mode, posterior_sample in modes:
            latent = mu[selected_rows].copy()
            if posterior_sample:
                noise = rng.standard_normal(latent.shape).astype(np.float32)
                latent += noise * np.exp(0.5 * logvar[selected_rows])
            samples = decode_latents(
                model, transform, latent, target, arguments.decode_batch_size,
                device, support,
            )
            saved[f"latent_{mode}"] = latent
            saved.update(summarize_and_plot(
                samples, support, mlat, mlt, target, source_kp, mode,
                destination, arguments.wic_vmax, arguments.si12_vmax,
            ))
            del samples
        np.savez_compressed(destination / "empirical_posterior_products.npz", **saved)
        selections.append({
            "target_kp": float(target),
            "pool_frames": int(len(pool)),
            "source_kp_minimum": float(np.min(source_kp)),
            "source_kp_median": float(np.median(source_kp)),
            "source_kp_maximum": float(np.max(source_kp)),
            "directory": destination.name,
        })
        print(f"Kp {target:g}: empirical latent gallery complete", flush=True)

    summary["selections"] = selections
    (output / "sampling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    dataset.close()
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--kp", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kp-half-width", type=float, default=0.34)
    parser.add_argument("--bank-frames-per-kp", type=int, default=4096)
    parser.add_argument(
        "--all-matching-frames", action="store_true",
        help="encode every training frame in each requested Kp window",
    )
    parser.add_argument(
        "--bank-only", action="store_true",
        help="write the latent bank without decoding empirical galleries",
    )
    parser.add_argument("--minimum-pool", type=int, default=100)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--posterior-mode", choices=("mean", "sample", "both"), default="both")
    parser.add_argument("--encode-batch-size", type=int, default=264)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--cache-mib", type=float, default=4096)
    parser.add_argument("--max-open-orbits", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--wic-vmax", type=float, default=1000)
    parser.add_argument("--si12-vmax", type=float, default=50)
    arguments = parser.parse_args(argv)
    positive = (
        "bank_frames_per_kp", "minimum_pool", "draws", "encode_batch_size",
        "decode_batch_size", "max_open_orbits", "threads",
    )
    if any(getattr(arguments, name) < 1 for name in positive):
        parser.error("frame, pool, draw, batch, orbit and thread counts must be positive")
    if (
        arguments.num_workers < 0 or arguments.prefetch_factor < 1
        or not np.isfinite(arguments.cache_mib) or arguments.cache_mib <= 0
        or not np.isfinite(arguments.kp_half_width) or arguments.kp_half_width < 0
        or not np.isfinite(arguments.kp).all()
        or any(not 0 <= value <= 9 for value in arguments.kp)
        or not np.allclose(
            np.asarray(arguments.kp) * 3,
            np.round(np.asarray(arguments.kp) * 3),
            atol=0.0011, rtol=0,
        )
        or len(set(arguments.kp)) != len(arguments.kp)
        or arguments.wic_vmax <= 0 or arguments.si12_vmax <= 0
    ):
        parser.error(
            "invalid workers, cache, discrete one-third Kp category, "
            "half-width or display limits"
        )
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
