"""Sample and inspect per-image conductance covariance from a saved native VAE."""

#%% Imports

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.validate_conductance_prior import input_evidence
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset, physical_decode, inpainting_occlusion_masks


#%% Fixed image posterior, paired samples and physical covariance

def decode_event(model, transform, mu, logvar, noise, support, batch_size=128):
    latent = np.asarray(mu)[None] + np.exp(0.5 * np.asarray(logvar))[None] * noise
    samples = np.empty((len(latent), 2, *support.shape), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(latent), batch_size):
            stop = min(start + batch_size, len(latent))
            values = model.decode(torch.as_tensor(latent[start:stop], dtype=torch.float32))
            samples[start:stop] = physical_decode(values, transform, support)
    if not np.isfinite(samples[:, :, support]).all():
        raise ValueError("decoder produced nonfinite values on learned support")
    return samples, latent


def physical_covariance(samples, support):
    """Joint state is Hall supported cells followed by Pedersen supported cells."""
    support = np.asarray(support, dtype=bool)
    if samples.ndim != 4 or samples.shape[1] != 2 or samples.shape[-2:] != support.shape or len(samples) < 2 or not support.any():
        raise ValueError("need at least two paired samples on a nonempty aligned support")
    state_indices = np.flatnonzero(np.broadcast_to(support, samples.shape[1:]).ravel())
    fields = np.asarray(samples.reshape(len(samples), -1)[:, state_indices], dtype=float)
    if not np.isfinite(fields).all():
        raise ValueError("nonfinite supported samples")
    mean = fields.mean(axis=0)
    factor = (fields - mean).T / np.sqrt(len(fields) - 1)
    covariance = factor @ factor.T
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    quantiles = np.quantile(fields, [0.05, 0.95], axis=0)
    maps = []
    for values in (mean, std, *quantiles):
        field = np.full(samples.shape[1:], np.nan)
        field.ravel()[state_indices] = values
        maps.append(field)
    return covariance, factor, maps, state_indices


def covariance_checks(covariance, factor, seed=0):
    probes = np.random.default_rng(seed).normal(size=(16, len(covariance)))
    quadratic = np.einsum("ij,ij->i", probes @ covariance, probes)
    direct = np.sum((probes @ factor)**2, axis=1)
    return {
        "symmetry_max_absolute_S2": float(np.max(np.abs(covariance - covariance.T))),
        "minimum_probe_quadratic_form": float(quadratic.min()),
        "quadratic_form_factor_relative_error": float(np.max(np.abs(quadratic - direct) / np.maximum(direct, 1e-30))),
        "diagonal_factor_max_absolute_error_S2": float(np.max(np.abs(np.diag(covariance) - np.sum(factor**2, axis=1))))
    }


def sampling_convergence(samples, support, full_covariance, factor):
    fields = samples[:, :, support].reshape(len(samples), -1).astype(float)
    target_std = np.sqrt(np.diag(full_covariance))
    target_trace = np.trace(full_covariance)
    result = []
    for count in sorted(set(min(n, len(samples)) for n in (256, 512, 1024, 2048, len(samples)))):
        variance = np.var(fields[:count], axis=0, ddof=1)
        result.append({
            "draws": count,
            "trace_relative_difference": float(abs(variance.sum() - target_trace) / max(target_trace, 1e-30)),
            "std_map_relative_L2_difference": float(np.linalg.norm(np.sqrt(variance) - target_std) / max(np.linalg.norm(target_std), 1e-30))
        })
    return result


def anchor_maps(covariance, maps, support, anchor):
    points = np.flatnonzero(support.ravel())
    position = int(np.flatnonzero(points == np.ravel_multi_index(anchor, support.shape))[0])
    std = np.sqrt(np.maximum(np.diag(covariance), 0))
    result = np.full((2, 2, *support.shape), np.nan)
    for channel in (0, 1):
        index = position + channel * len(points)
        denominator = std[index] * std
        correlation = np.divide(covariance[index], denominator, out=np.full_like(std, np.nan), where=denominator > 0)
        result[channel][:, support] = np.clip(correlation.reshape(2, -1), -1, 1)
    return result


def event_summary(samples, latent, mu, logvar, covariance, maps, observed, valid, prior_covariance):
    mean, std, lower, upper = maps
    interval = (observed >= lower) & (observed <= upper)
    diagonal = np.diag(covariance)
    latent_target_covariance = np.diag(np.exp(logvar))
    return {
        "draws": len(samples),
        "latent_mean_error_norm": float(np.linalg.norm(latent.mean(axis=0) - mu)),
        "latent_variance_relative_L2_error": float(np.linalg.norm(np.var(latent, axis=0, ddof=1) - np.exp(logvar)) / np.linalg.norm(np.exp(logvar))),
        "latent_covariance_relative_frobenius_error": float(np.linalg.norm(np.cov(latent, rowvar=False) - latent_target_covariance) / np.linalg.norm(latent_target_covariance)),
        "covariance_trace_S2": float(diagonal.sum()),
        "covariance_relative_frobenius_difference_from_prior": float(np.linalg.norm(covariance - prior_covariance) / np.linalg.norm(prior_covariance)),
        "median_std_S_Hall_Pedersen": [float(np.nanmedian(std[channel])) for channel in (0, 1)],
        "max_std_S_Hall_Pedersen": [float(np.nanmax(std[channel])) for channel in (0, 1)],
        "input_reconstruction_RMSE_S_Hall_Pedersen": [float(np.sqrt(np.mean((mean[channel][valid] - observed[channel][valid])**2))) for channel in (0, 1)],
        "input_retrieval_containment_in90_interval_Hall_Pedersen": [float(interval[channel][valid].mean()) for channel in (0, 1)]
    }


#%% Figures: physical means, conditional spread and joint correlation

def save_figure(figure, output, name):
    figure.savefig(output / (name + ".png"), dpi=160)
    figure.savefig(output / (name + ".pdf"))
    plt.close(figure)


def plot_events(output, events, prior_maps, prior_correlation, anchor):
    figure, axes = plt.subplots(4, len(events), figsize=(3.1 * len(events), 11), layout="constrained")
    for column, event in enumerate(events):
        for row in range(4):
            channel = row % 2
            quantity = event["maps"][0 if row < 2 else 1][channel]
            maximum = (50, 10, 8, 2)[row]
            im = axes[row, column].imshow(quantity, origin="lower", vmin=0, vmax=maximum, cmap="viridis")
            axes[row, column].set(title=f"orbit {event['orbit']}\n{'mean' if row < 2 else 'STD'} {'Hall' if channel == 0 else 'Pedersen'}", xlabel="CS column", ylabel="CS row")
            figure.colorbar(im, ax=axes[row, column], shrink=0.7, label="S", extend="max" if np.nanmax(quantity) > maximum else "neither")
    figure.suptitle("Per-image q(z|x) ensemble: physical mean and conditional standard deviation")
    save_figure(figure, output, "event_mean_and_std")
    figure, axes = plt.subplots(len(events) + 1, 4, figsize=(12, 2.4 * (len(events) + 1)), layout="constrained")
    for row, correlation in enumerate([prior_correlation, *[event["correlation"] for event in events]]):
        for column in range(4):
            source, target = divmod(column, 2)
            im = axes[row, column].imshow(correlation[source, target], origin="lower", cmap="RdBu_r", vmin=-1, vmax=1)
            axes[row, column].plot(anchor[1], anchor[0], "kx", ms=4)
            label = "unconditional prior" if row == 0 else f"orbit {events[row-1]['orbit']}"
            axes[row, column].set(title=f"{label}: {'H' if source == 0 else 'P'} anchor → {'H' if target == 0 else 'P'}")
            figure.colorbar(im, ax=axes[row, column], shrink=0.65)
    figure.suptitle("Event-specific joint correlation; one fixed spatial anchor")
    save_figure(figure, output, "event_correlation_maps")
    first = events[0]
    figure, axes = plt.subplots(2, 5, figsize=(14, 6), layout="constrained")
    for channel, name in enumerate(("Hall", "Pedersen")):
        fields = [np.where(first["valid"], first["observed"][channel], np.nan), first["maps"][0][channel], first["maps"][1][channel], first["maps"][3][channel]-first["maps"][2][channel], prior_maps[1][channel]]
        titles = ["input retrieval", "conditional mean", "conditional STD", "90% interval width", "unconditional STD"]
        for column, (field, title) in enumerate(zip(fields, titles)):
            maximum = (50, 50, 8, 25, 8)[column] if channel == 0 else (10, 10, 2, 6, 2)[column]
            im = axes[channel, column].imshow(field, origin="lower", vmin=0, vmax=maximum)
            axes[channel, column].set_title(f"{name}: {title}", fontsize=9)
            figure.colorbar(im, ax=axes[channel, column], shrink=0.7, label="S", extend="max" if np.nanmax(field) > maximum else "neither")
    figure.suptitle(f"Example orbit {first['orbit']}, {first['time']}; fixed model and learned support")
    save_figure(figure, output, "event_example")


#%% Test fixed event posteriors and one input-removal control

def run(arguments):
    run_directory, output = arguments.run_directory.expanduser(), arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(arguments.threads)
    model, transform, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
    if artifact["artifact_type"] != "conductance_vae_run_v1":
        raise ValueError("requires the native paired Hall/Pedersen model")
    if getattr(model, "kp_conditioning", None) is not None:
        raise ValueError("this image-conditioned covariance audit requires an unconditional VAE; use Kp-matched diagnostics for conditional models")
    evidence = input_evidence(run_directory, artifact)
    dataset, _ = _load_heldout_dataset(artifact, transform, requested_split="test")
    support = np.asarray(artifact["data_provenance"]["training_support"], dtype=bool)
    selections = []
    for orbit in np.unique(dataset.frame_orbit):
        positions = np.flatnonzero(dataset.frame_orbit == orbit)
        order = np.argsort(dataset.frame_time[positions])
        selections.append(int(positions[order[len(order) // 2]]))
    noise = np.random.default_rng(arguments.seed).normal(size=(arguments.draws, model.latent_dim)).astype(np.float32)
    np.save(output / "base_gaussian_noise.npy", noise)
    print(f"Selected {len(selections)} midpoint images from distinct held-out orbits", flush=True)
    samples, _ = decode_event(model, transform, np.zeros(model.latent_dim), np.zeros(model.latent_dim), noise, support)
    prior_covariance, prior_factor, prior_maps, state_indices = physical_covariance(samples, support)
    anchor = tuple(int(value) for value in np.unravel_index(np.nanargmax(prior_maps[1][0]), support.shape))
    prior_correlation = anchor_maps(prior_covariance, prior_maps, support, anchor)
    results = {
        "configuration": {"draws": arguments.draws, "seed": arguments.seed, "device": "cpu", "support_cells": int(support.sum()), "state_size": len(state_indices), "input_policy": "exclude never-trained cells from encoder input and output", "anchor_row_column": anchor},
        "prior": {"trace_S2": float(np.trace(prior_covariance)), "checks": covariance_checks(prior_covariance, prior_factor)},
        "events": [], "evidence": evidence
    }
    events = []
    for index in selections:
        orbit, time = int(dataset.frame_orbit[index]), str(dataset.frame_time[index])
        batch = dataset[index]
        original_valid = batch["mask"].numpy().copy()
        valid = original_valid & support[None]
        encoder_input = np.where(valid, batch["encoder_input"].numpy(), 0).astype(np.float32)
        with torch.no_grad():
            mu, logvar = model.encode(torch.from_numpy(encoder_input[None]), torch.from_numpy(valid[None]))
        mu, logvar = mu[0].numpy(), logvar[0].numpy()
        print(f"Orbit {orbit}, {time}: decoding {arguments.draws} conditional draws", flush=True)
        samples, latent = decode_event(model, transform, mu, logvar, noise, support)
        covariance, factor, maps, indices = physical_covariance(samples, support)
        summary = event_summary(samples, latent, mu, logvar, covariance, maps, dataset.values[index], valid[0], prior_covariance)
        summary.update(orbit=orbit, time=time, frame_index=index, extra_untrained_input_cells_excluded=int((original_valid & ~support[None]).sum()), checks=covariance_checks(covariance, factor), convergence=sampling_convergence(samples, support, covariance, factor))
        correlation = anchor_maps(covariance, maps, support, anchor)
        np.savez_compressed(output / f"event_{orbit:04d}.npz", covariance_S2=covariance, anomaly_factor_S=factor.astype(np.float32), mean_S=maps[0], standard_deviation_S=maps[1], lower05_S=maps[2], upper95_S=maps[3], state_flat_indices=indices, support=support, mu=mu, logvar=logvar, observed_S=dataset.values[index], input_valid=valid[0])
        events.append({"orbit": orbit, "time": time, "maps": maps, "correlation": correlation, "observed": dataset.values[index], "valid": valid[0]})
        results["events"].append(summary)
        if index == selections[0]:
            independent = np.random.default_rng(arguments.seed + 1).normal(size=noise.shape).astype(np.float32)
            repeat, _ = decode_event(model, transform, mu, logvar, independent, support)
            repeat_covariance, _, _, _ = physical_covariance(repeat, support)
            results["independent_seed_check"] = {"orbit": orbit, "trace_relative_difference": float(abs(np.trace(repeat_covariance) - np.trace(covariance)) / np.trace(covariance)), "covariance_relative_frobenius_difference": float(np.linalg.norm(repeat_covariance - covariance) / np.linalg.norm(covariance))}
            del repeat, repeat_covariance
            damaged_valid, bounds, _ = inpainting_occlusion_masks(dataset.values[index:index+1], valid[None], 12, policy="auroral")
            damaged_input = np.where(damaged_valid[0], encoder_input, 0)
            with torch.no_grad():
                damaged_mu, damaged_logvar = model.encode(torch.from_numpy(damaged_input[None]), torch.from_numpy(damaged_valid))
            damaged, _ = decode_event(model, transform, damaged_mu[0].numpy(), damaged_logvar[0].numpy(), noise, support)
            damaged_covariance, _, damaged_maps, _ = physical_covariance(damaged, support)
            removed = valid[0] & ~damaged_valid[0, 0]
            results["masked_patch_control"] = {"orbit": orbit, "bounds": bounds.tolist(), "removed_supported_cells": int(removed.sum()), "covariance_trace_ratio_masked_to_full": float(np.trace(damaged_covariance) / np.trace(covariance)), "removed_patch_variance_ratio_Hall_Pedersen": [float(np.mean(damaged_maps[1][channel][removed]**2) / np.mean(maps[1][channel][removed]**2)) for channel in (0, 1)]}
            np.savez_compressed(output / "masked_patch_control.npz", mean_S=damaged_maps[0], standard_deviation_S=damaged_maps[1], removed=removed, original_mean_S=maps[0], original_std_S=maps[1])
            del damaged, damaged_covariance
        del samples, covariance, factor
    results["interpretation"] = [
        "Each image holds mu/logvar fixed. Covariance is within-event decoded q spread, not covariance pooled across events.",
        "Shared Gaussian noise across images/prior reduces Monte Carlo noise in comparisons; one independent-seed repeat checks sampling stability.",
        "All joint matrices use state [Hall supported cells, Pedersen supported cells] and units S^2; excluded cells are unavailable, not known zeros.",
        "Input retrieval containment is a reconstruction diagnostic, not independent uncertainty calibration.",
        "Encoder receives normalized H/P and quality, not dH/dP. No retrieval-noise propagation or model-weight uncertainty is added.",
        "Finite-sample covariance is PSD by factor construction, but not necessarily positive definite. Nonlinear output covariance rank is not limited to latent dimension.",
        "Masked-input variance growth is measured, not assumed. Source uses image-ratio electrons/Hardy protons; no new Hardy-electron training."
    ]
    if input_evidence(run_directory, artifact) != evidence:
        raise RuntimeError("input/model evidence changed during test")
    plot_events(output, events, prior_maps, prior_correlation, anchor)
    pd.DataFrame(results["events"]).drop(columns=["checks", "convergence"]).to_csv(output / "event_metrics.csv", index=False)
    (output / "event_covariance_test.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({"event_count": len(results["events"]), "independent_seed_check": results["independent_seed_check"], "masked_patch_control": results["masked_patch_control"]}, indent=2), flush=True)
    print(f"Saved event covariance test to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    arguments = parser.parse_args()
    if arguments.draws < 256 or arguments.threads < 1:
        parser.error("need at least 256 draws and positive threads")
    run(arguments)
