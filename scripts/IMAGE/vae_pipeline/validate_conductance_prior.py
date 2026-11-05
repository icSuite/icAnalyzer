"""Validate a native conductance VAE prior against saved train/test populations."""

#%% Imports and input evidence

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2, wasserstein_distance
import torch

from icreader import load
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset, infer_posterior_prior


def input_evidence(run, artifact):
    evidence = {}
    for name in ("conductance_vae.pt", "summary.json"):
        evidence[name] = hashlib.sha256((run / name).read_bytes()).hexdigest()
    evidence["source_files"] = {
        path: [Path(path).stat().st_size, Path(path).stat().st_mtime_ns]
        for split in ("train", "test") for path in artifact["split_files"][split]
    }
    annotations = artifact["data_provenance"].get("manual_annotations")
    if annotations:
        path = Path(annotations["filename"])
        evidence["annotation_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with zipfile.ZipFile(run / "ensemble_products.npz") as archive:
        evidence["saved_array_crc"] = {entry.filename: entry.CRC for entry in archive.infolist()}
    return evidence


def orbit_weights(orbits):
    """Give each orbit equal mass and each of its frames equal conditional mass."""
    _, inverse, count = np.unique(orbits, return_inverse=True, return_counts=True)
    return 1.0 / (len(count) * count[inverse])


#%% Recover complete physical members without loading the full anomaly factor

def saved_members(run, indices):
    indices = np.asarray(indices, dtype=int)
    with np.load(run / "ensemble_products.npz", allow_pickle=False) as archive:
        mean = archive["conductance_mean"]
    with zipfile.ZipFile(run / "ensemble_products.npz") as archive:
        with archive.open("conductance_anomaly_factor.npy") as stream:
            version = np.lib.format.read_magic(stream)
            shape, fortran, dtype = np.lib.format._read_array_header(stream, version)
            if len(shape) != 2 or shape[0] != mean.size or shape[1] < 2 or dtype.kind != "f":
                raise ValueError("saved factor does not match the conductance mean")
            if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= shape[1]):
                raise ValueError("saved member indices are out of range")
            selected = np.empty((shape[0], len(indices)), dtype=dtype)
            width, count = (shape[0], shape[1]) if fortran else (shape[1], shape[0])
            chunk = max(1, 8 * 1024**2 // (width * dtype.itemsize))
            for start in range(0, count, chunk):
                stop = min(start + chunk, count)
                raw = stream.read((stop - start) * width * dtype.itemsize)
                block = np.frombuffer(raw, dtype=dtype).reshape(stop - start, width)
                if fortran:
                    positions = np.flatnonzero((indices >= start) & (indices < stop))
                    selected[:, positions] = block[indices[positions] - start].T
                else:
                    selected[start:stop] = block[:, indices]
    physical = mean.ravel()[None] + np.sqrt(shape[1] - 1) * selected.T
    return physical.reshape(len(indices), *mean.shape)


#%% Exact aggregate-posterior moments and sampled distribution shapes

def aggregate_posterior(mu, logvar, weights):
    """Moments of the finite mixture of encoder Gaussians; no draw approximation."""
    mu, logvar = np.asarray(mu, dtype=float), np.asarray(logvar, dtype=float)
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()
    mean = weights @ mu
    residual = mu - mean
    covariance = (residual * weights[:, None]).T @ residual
    covariance += np.diag(weights @ np.exp(logvar))
    return mean, covariance


def latent_summary(mu, logvar, weights, prior, seed):
    mean, covariance = aggregate_posterior(mu, logvar, weights)
    eigenvalues = np.linalg.eigvalsh(covariance)
    dimension = mu.shape[1]
    gaussian_kl = 0.5 * (np.trace(covariance) + mean @ mean - dimension - np.linalg.slogdet(covariance)[1])
    rng = np.random.default_rng(seed)
    draws = mu[:, None] + np.exp(0.5 * logvar[:, None]) * rng.normal(size=(len(mu), 8, dimension))
    draws = draws.reshape(-1, dimension)
    draw_weights = np.repeat(weights / 8, 8)
    direction = rng.normal(size=(64, dimension))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)
    prior = prior[rng.choice(len(prior), min(8000, len(prior)), replace=False)]
    q_projection, p_projection = draws @ direction.T, prior @ direction.T
    distances = [wasserstein_distance(q_projection[:, i], p_projection[:, i], u_weights=draw_weights) for i in range(64)]
    null = rng.normal(size=draws.shape) @ direction.T
    null_distances = [wasserstein_distance(null[:, i], p_projection[:, i], u_weights=draw_weights) for i in range(64)]
    radial_interval = chi2.ppf([0.05, 0.95], dimension)
    r2 = np.sum(draws**2, axis=1)
    kl_per_frame = 0.5 * np.sum(mu**2 + np.exp(logvar) - 1 - logvar, axis=1)
    result = {
        "aggregate_mean_norm": float(np.linalg.norm(mean)),
        "aggregate_expected_squared_radius": float(np.trace(covariance) + mean @ mean),
        "aggregate_covariance_eigenvalue_range": eigenvalues[[0, -1]].tolist(),
        "aggregate_coordinate_variance_range": [float(np.diag(covariance).min()), float(np.diag(covariance).max())],
        "aggregate_off_diagonal_covariance_rms": float(np.sqrt(np.mean((covariance - np.diag(np.diag(covariance)))**2))),
        "moment_matched_gaussian_kl_lower_bound_nats": float(gaussian_kl),
        "mean_conditional_encoder_kl_nats": float(weights @ kl_per_frame),
        "fraction_in_prior_central90_radial_interval": float(draw_weights @ ((r2 >= radial_interval[0]) & (r2 <= radial_interval[1]))),
        "mean_sliced_wasserstein": float(np.mean(distances)),
        "prior_vs_prior_mean_sliced_wasserstein": float(np.mean(null_distances)),
        "distribution_note": "Eight q draws per frame are Monte Carlo integration of a weighted mixture, not independent observations; no iid p-value."
    }
    return result, mean, covariance, draws, draw_weights


#%% Matched-mask physical distributions and pairwise spatial dependence

def weighted_quantiles(values, weights, probabilities):
    selected = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    values, weights = np.asarray(values)[selected], np.asarray(weights)[selected]
    if values.size == 0:
        return np.full(len(probabilities), np.nan)
    order = np.argsort(values)
    values, weights = values[order], weights[order]
    cumulative = (np.cumsum(weights) - 0.5 * weights) / weights.sum()
    return np.interp(probabilities, cumulative, values)


def cell_weights(valid, area, quality=None):
    """Area-normalized cells within each frame; positive-quality sensitivity."""
    weights = np.asarray(valid, dtype=float) * area[None]
    if quality is not None:
        weights *= np.where(np.isfinite(quality), np.maximum(quality, 0), 0)
    total = weights.sum(axis=(1, 2), keepdims=True)
    return np.divide(weights, total, out=np.zeros_like(weights), where=total > 0)


def weighted_correlation(x, y, weights, axis=0):
    weights = np.where(np.isfinite(x) & np.isfinite(y), weights, 0)
    total = weights.sum(axis=axis)
    safe = np.where(total > 0, total, 1)
    x, y = np.where(weights > 0, x, 0), np.where(weights > 0, y, 0)
    mx, my = (weights * x).sum(axis=axis) / safe, (weights * y).sum(axis=axis) / safe
    covariance = (weights * x * y).sum(axis=axis) / safe - mx * my
    vx = (weights * x * x).sum(axis=axis) / safe - mx**2
    vy = (weights * y * y).sum(axis=axis) / safe - my**2
    denominator = np.sqrt(np.maximum(vx, 0) * np.maximum(vy, 0))
    return np.divide(covariance, denominator, out=np.full_like(covariance, np.nan), where=(denominator > 1e-12) & (total > 0))


def spatial_summary(fields, valid, frame_weights):
    log_fields = np.log10(np.maximum(fields, 0) + 1)
    weights = frame_weights[:, None, None] * valid
    minimum_count = max(20, int(np.ceil(0.1 * len(fields))))
    hp = weighted_correlation(log_fields[:, 0], log_fields[:, 1], weights)
    hp = np.where(valid.sum(axis=0) >= minimum_count, hp, np.nan)
    result = {"same_cell_log_HP_correlation_median": float(np.nanmedian(hp))}
    for lag in (1, 2, 4, 8):
        for channel, name in enumerate(("Hall", "Pedersen")):
            x, y = log_fields[:, channel, :, :-lag], log_fields[:, channel, :, lag:]
            pair_valid = valid[:, :, :-lag] & valid[:, :, lag:]
            weights = frame_weights[:, None, None] * pair_valid
            correlation = weighted_correlation(x, y, weights)
            correlation = np.where(pair_valid.sum(axis=0) >= minimum_count, correlation, np.nan)
            difference = (x - y)**2
            per_frame = np.divide(np.sum(np.where(pair_valid, difference, 0), axis=(1, 2)), pair_valid.sum(axis=(1, 2)), out=np.full(len(fields), np.nan), where=pair_valid.sum(axis=(1, 2)) > 0)
            available = np.isfinite(per_frame)
            result[f"{name}_log_correlation_lag{lag}_median"] = float(np.nanmedian(correlation))
            result[f"{name}_log_structure_function_lag{lag}"] = float(np.average(per_frame[available], weights=frame_weights[available]))
    result["lag_note"] = "Column offsets in CS cells; exact pairwise common validity and orbit-balanced frame weights. Correlations are across frames at fixed cell pairs, then median over pairs with at least max(20,10% of frames) valid observations."
    return result, hp


def physical_summary(fields, valid, quality, area, frame_weights, orbits, weighting):
    within = cell_weights(valid, area, quality if weighting == "quality" else None)
    eligible = within.sum(axis=(1, 2)) > 0
    frame_weights = np.zeros(len(fields))
    frame_weights[eligible] = orbit_weights(orbits[eligible])
    weights = within * frame_weights[:, None, None]
    weights /= weights.sum()
    result, orbit_rows = {}, []
    for channel, name in enumerate(("Hall", "Pedersen")):
        values = np.asarray(fields[:, channel], dtype=float)
        mean = float(np.sum(weights * np.where(weights > 0, values, 0)))
        result[name] = {
            "mean_S": mean,
            "standard_deviation_S": float(np.sqrt(np.sum(weights * np.where(weights > 0, (values - mean)**2, 0)))),
            "quantiles_S_p05_p50_p95_p99": weighted_quantiles(values.ravel(), weights.ravel(), [0.05, 0.5, 0.95, 0.99]).tolist(),
            "zero_fraction": float(np.sum(weights * (values <= 1e-6))),
            "fraction_above_10S": float(np.sum(weights * (values > 10))),
            "fraction_above_20S": float(np.sum(weights * (values > 20)))
        }
        frame_mean = np.sum(within * np.where(within > 0, values, 0), axis=(1, 2))
        frame_tail = np.sum(within * (values > 20), axis=(1, 2))
        frame_usable = within.sum(axis=(1, 2)) > 0
        for orbit in np.unique(orbits):
            selected = (orbits == orbit) & frame_usable
            if selected.any():
                orbit_rows.append({"orbit": int(orbit), "channel": name, "mean_S": float(frame_mean[selected].mean()), "fraction_above_20S": float(frame_tail[selected].mean())})
    ratio_weights = np.where(fields[:, 1] > 0.1, weights, 0)
    ratio = fields[:, 0] / np.maximum(fields[:, 1], 0.1)
    result["HP_ratio_p05_p50_p95_P_above_0p1S"] = weighted_quantiles(ratio.ravel(), ratio_weights.ravel(), [0.05, 0.5, 0.95]).tolist()
    result["eligible_frames"] = int(np.count_nonzero(within.sum(axis=(1, 2)) > 0))
    return result, orbit_rows


def mean_maps(fields, valid):
    count = valid.sum(axis=0)
    mean = np.divide(np.sum(np.where(valid[:, None], fields, 0), axis=0), count[None], out=np.full(fields.shape[1:], np.nan), where=count[None] > 0)
    variance = np.divide(np.sum(np.where(valid[:, None], (fields - mean)**2, 0), axis=0), count[None], out=np.full_like(mean, np.nan), where=count[None] > 0)
    return mean, np.sqrt(np.maximum(variance, 0))


#%% Standalone figures and orbit-level sampling uncertainty

def save_figure(figure, output, name):
    figure.savefig(output / (name + ".png"), dpi=170)
    figure.savefig(output / (name + ".pdf"))
    plt.close(figure)


def latent_figure(output, latent, prior):
    figure, axes = plt.subplots(2, 3, figsize=(13, 8), layout="constrained")
    radius = np.linalg.norm(prior, axis=1)
    upper = max(15, np.ceil(max(np.linalg.norm(latent[split][3], axis=1).max() for split in latent)))
    bins = np.linspace(0, upper, 81)
    for row, split in enumerate(("train", "test")):
        summary, mean, covariance, draws, weights = latent[split]
        axes[row, 0].hist(radius, bins=bins, density=True, histtype="step", label="N(0,I) prior")
        axes[row, 0].hist(np.linalg.norm(draws, axis=1), weights=weights, bins=bins, density=True, histtype="step", label="orbit-balanced encoder q")
        axes[row, 0].set(title=f"{split}: latent radius", xlabel="norm(z)", ylabel="density")
        axes[row, 0].legend()
        axes[row, 1].plot(np.linalg.eigvalsh(covariance), "o-", label="aggregate q covariance")
        axes[row, 1].axhline(1, color="black", linestyle="--", label="prior variance")
        axes[row, 1].set(title="Exact mixture covariance eigenvalues", xlabel="sorted eigenvalue index", ylabel="variance")
        axes[row, 1].legend(fontsize=8)
        im = axes[row, 2].imshow(covariance - np.eye(len(mean)), cmap="RdBu_r", vmin=-2, vmax=2)
        axes[row, 2].set(title="Covariance − identity (colour clipped ±2)")
        figure.colorbar(im, ax=axes[row, 2], shrink=0.8)
    save_figure(figure, output, "latent_prior_validation")


def physical_figures(output, ensembles, valid, frame_weights, area):
    colors = {"observed": "black", "posterior_mean": "tab:blue", "posterior_sample": "tab:orange", "prior": "tab:green"}
    labels = {"observed": "retrieval reference", "posterior_mean": "decoder(mu)", "posterior_sample": "decoded q sample", "prior": "decoded prior draw"}
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    within = cell_weights(valid, area)
    weights = (within * frame_weights[:, None, None]).ravel()
    for channel, name in enumerate(("Hall", "Pedersen")):
        for family, fields in ensembles.items():
            axes[channel].hist(np.log10(np.maximum(fields[:, channel], 0).ravel() + 1), weights=weights, bins=np.linspace(0, 2.5, 81), density=True, histtype="step", color=colors[family], label=labels[family])
        axes[channel].set(title=f"Held-out {name}: equal orbit, area-normalized masks", xlabel="log10(conductance/S + 1)", ylabel="density")
        axes[channel].legend(fontsize=8)
    save_figure(figure, output, "conductance_distributions")
    figure, axes = plt.subplots(4, 4, figsize=(13, 12), layout="constrained")
    for row, (family, fields) in enumerate(ensembles.items()):
        mean, std = mean_maps(fields, valid)
        for column in range(4):
            channel = column % 2
            values = mean[channel] if column < 2 else std[channel]
            limit = (12, 4, 15, 4)[column]
            im = axes[row, column].imshow(values, origin="lower", vmin=0, vmax=limit)
            axes[row, column].set(title=f"{labels[family]}: {'mean' if column < 2 else 'std'} {'Hall' if channel == 0 else 'Pedersen'}", xlabel="CS column", ylabel="CS row")
            figure.colorbar(im, ax=axes[row, column], shrink=0.75, label="S")
    figure.suptitle("Identical held-out masks; cell maps use available frames, not orbit weighting")
    save_figure(figure, output, "conductance_moment_maps")


def spatial_figure(output, results):
    figure, axes = plt.subplots(2, 2, figsize=(11, 8), layout="constrained")
    lags = (1, 2, 4, 8)
    for channel, name in enumerate(("Hall", "Pedersen")):
        for family, summary in results["test"]["spatial"].items():
            axes[channel, 0].plot(lags, [summary[f"{name}_log_correlation_lag{lag}_median"] for lag in lags], "o-", label=family)
            axes[channel, 1].plot(lags, [summary[f"{name}_log_structure_function_lag{lag}"] for lag in lags], "o-", label=family)
        axes[channel, 0].set(title=f"{name}: pairwise temporal correlation", xlabel="CS column offset (cells)", ylabel="median correlation")
        axes[channel, 1].set(title=f"{name}: log-field spatial differences", xlabel="CS column offset (cells)", ylabel="mean squared log10 difference")
        for axis in axes[channel]:
            axis.legend(fontsize=8)
            axis.grid(alpha=0.2)
    figure.suptitle("Held-out, identical pairwise masks, equal-orbit frame weights")
    save_figure(figure, output, "conductance_spatial_dependence")


def orbit_bootstrap(rows, seed):
    frame = pd.DataFrame(rows)
    rng = np.random.default_rng(seed)
    result = {}
    for weighting in ("area", "quality"):
        for channel in ("Hall", "Pedersen"):
            selected = frame.loc[(frame.weighting == weighting) & (frame.channel == channel) & (frame.split == "test")]
            table = selected.pivot(index="orbit", columns="family", values="mean_S").dropna()
            difference = table["prior"].to_numpy() - table["observed"].to_numpy()
            bootstrap = difference[rng.integers(len(difference), size=(2000, len(difference)))].mean(axis=1)
            result[f"{weighting}_{channel}"] = {"n_orbits": len(table), "prior_minus_observed_mean_S": float(difference.mean()), "orbit_bootstrap95_interval_S": np.quantile(bootstrap, [0.025, 0.975]).tolist()}
    return result


#%% Run with the original model/selection and isolated outputs

def run(arguments):
    run_directory, output = arguments.run_directory.expanduser(), arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(arguments.threads)
    model, transform, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
    if artifact["artifact_type"] != "conductance_vae_run_v1":
        raise ValueError("this diagnostic requires a native Hall/Pedersen model")
    if getattr(model, "kp_conditioning", None) is not None:
        raise ValueError("this unconditional-prior audit does not support fixed-Kp ensembles; use Kp-matched held-out diagnostics")
    evidence = input_evidence(run_directory, artifact)
    support = np.asarray(artifact["data_provenance"]["training_support"], dtype=bool)
    with load(artifact["split_files"]["test"][0]) as product:
        area = np.asarray(product.grid.A, dtype=float)
    with np.load(run_directory / "ensemble_products.npz", allow_pickle=False) as archive:
        prior_latent = archive["latent_samples"]
    results, latent, rows = {}, {}, []
    for split in ("train", "test"):
        print(f"Reloading original {split} population", flush=True)
        dataset, _ = _load_heldout_dataset(artifact, transform, requested_split=split)
        print(f"{split}: {len(dataset)} frames, {len(np.unique(dataset.frame_orbit))} orbits", flush=True)
        arrays = infer_posterior_prior(model, transform, dataset, 32, "cpu", arguments.seed, support_mask=support)
        frame_weights = orbit_weights(dataset.frame_orbit)
        latent[split] = latent_summary(arrays["mu"], arrays["logvar"], frame_weights, prior_latent, arguments.seed)
        frame_mean, frame_covariance = aggregate_posterior(arrays["mu"], arrays["logvar"], np.ones(len(dataset)))
        member_indices = np.random.default_rng(arguments.seed + 1).choice(len(prior_latent), len(dataset), replace=False)
        print(f"{split}: streaming {len(member_indices)} original physical prior members", flush=True)
        index_sets = [member_indices]
        if split == "test":
            for repeat in range(1, arguments.prior_repeats):
                index_sets.append(np.random.default_rng(arguments.seed + 1000 * repeat).choice(len(prior_latent), len(dataset), replace=False))
        members = saved_members(run_directory, np.concatenate(index_sets))
        arrays["prior"] = members[:len(dataset)]
        # Missing stored cells and float32 roundoff are masked, not new physical clipping.
        valid = arrays["valid"][:, 0] & support
        families = {name: arrays[name] for name in ("observed", "posterior_mean", "posterior_sample", "prior")}
        results[split] = {
            "frames": len(dataset), "orbits": int(len(np.unique(dataset.frame_orbit))),
            "latent": latent[split][0],
            "frame_weighted_latent_moments": {
                "mean_norm": float(np.linalg.norm(frame_mean)),
                "covariance_eigenvalue_range": np.linalg.eigvalsh(frame_covariance)[[0, -1]].tolist(),
                "expected_squared_radius": float(np.trace(frame_covariance) + frame_mean @ frame_mean)
            },
            "physical": {}, "spatial": {}
        }
        if split == "test":
            results[split]["prior_monte_carlo_repeats_area"] = []
            for repeat in range(arguments.prior_repeats):
                summary, _ = physical_summary(members[repeat * len(dataset):(repeat + 1) * len(dataset)], valid, dataset.quality[:, 0], area, frame_weights, dataset.frame_orbit, "area")
                results[split]["prior_monte_carlo_repeats_area"].append({name: summary[name] for name in ("Hall", "Pedersen")})
        np.savez_compressed(output / f"{split}_latent.npz", mu=arrays["mu"], logvar=arrays["logvar"], orbit=dataset.frame_orbit)
        for family, fields in families.items():
            if not np.isfinite(fields[np.broadcast_to(valid[:, None], fields.shape)]).all():
                raise ValueError(f"{family} has nonfinite fields on common support")
            results[split]["physical"][family] = {}
            for weighting in ("area", "quality"):
                summary, orbit_rows = physical_summary(fields, valid, dataset.quality[:, 0], area, frame_weights, dataset.frame_orbit, weighting)
                results[split]["physical"][family][weighting] = summary
                rows.extend(dict(row, split=split, family=family, weighting=weighting) for row in orbit_rows)
            results[split]["spatial"][family], _ = spatial_summary(fields, valid, frame_weights)
        if split == "test":
            physical_figures(output, families, valid, frame_weights, area)
            np.savez_compressed(output / "heldout_latent.npz", mu=arrays["mu"], logvar=arrays["logvar"], orbit=dataset.frame_orbit, prior_member_indices=member_indices)
            np.savez_compressed(output / "inference.npz", **arrays, quality=dataset.quality,
                                standard_deviation=dataset.standard_deviation, frame_orbit=dataset.frame_orbit,
                                frame_time=dataset.frame_time, support=support)
        del arrays, dataset, members
    results["orbit_bootstrap"] = orbit_bootstrap(rows, arguments.seed)
    results["evidence"] = evidence
    results["configuration"] = {"seed": arguments.seed, "device": "cpu", "threads": arguments.threads, "unsupported_cells": int((~support).sum()), "latent_prior_draws": len(prior_latent), "prior_repeats": arguments.prior_repeats}
    results["interpretation"] = [
        "Native source: image-ratio electron retrievals and Hardy protons; not the new Hardy-electron model.",
        "Equal orbit weights; area-normalized identical per-frame observation masks. Quality sensitivity uses the same quality weights for every field family.",
        "Primary aggregate q is orbit-balanced; supplementary frame-weighted moments check the frame/batch-weighted training population.",
        "Quality depends on auroral signal. Random prior members do not retain the observed signal-quality association; quality deficits alone do not establish unconditional prior bias.",
        "Observed fields are noisy retrieval references, not independent true conductance; no noise deconvolution or extra decoder residual noise.",
        "decoder(mu) is a deterministic reconstruction, not the posterior mean physical conductance.",
        "One randomly selected complete prior member per mask; posterior sample is one q draw per frame. No latent vectors are filtered by density.",
        "Bootstrap units are complete held-out orbits; few orbits limit population generalization. No p-values from correlated pixels/frames.",
        "Matched masks do not eliminate activity-dependent missingness. A diagnostic mismatch has no automatic model-calibration correction."
    ]
    if input_evidence(run_directory, artifact) != evidence:
        raise RuntimeError("model, ensemble, annotation or input file evidence changed during validation")
    latent_figure(output, latent, prior_latent)
    spatial_figure(output, results)
    pd.DataFrame(rows).to_csv(output / "orbit_metrics.csv", index=False)
    (output / "validation.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps({split: results[split]["latent"] for split in ("train", "test")}, indent=2), flush=True)
    print(f"Saved validation to {output}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1012)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--prior-repeats", type=int, default=3)
    arguments = parser.parse_args()
    if arguments.threads < 1 or arguments.prior_repeats < 1:
        parser.error("threads and prior repeats must be positive")
    run(arguments)
