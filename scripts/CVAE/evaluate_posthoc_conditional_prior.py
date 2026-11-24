"""Evaluate fitted Kp-conditional latent priors without refitting them."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ks_2samp
from torch.utils.data import DataLoader

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.prior import ConditionalGaussianMixturePrior
from scripts.CVAE.analyze_empirical_latent_modes import FIELD_NAMES, load_bank
from scripts.CVAE.fit_posthoc_conditional_prior import (
    latent_dataset,
    sample_standardized_posterior,
)


FIELD_LABELS = ("Corrected WIC", "SI12")
DISPLAY_METRICS = ("maximum", "supported_mean", "active_fraction")


#%% Saved prior and held-out rows

def sha256_file(filename, chunk_size=16 * 1024 * 1024):
    digest = hashlib.sha256()
    with filename.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()

def load_prior(filename, device):
    payload = torch.load(filename, map_location="cpu", weights_only=False)
    required = (
        "prior_configuration", "prior_state_dict", "latent_center",
        "latent_scale", "artifact_sha256", "bank_sha256", "validation_rows",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ValueError(f"{filename} is missing " + ", ".join(missing))
    prior = ConditionalGaussianMixturePrior.from_configuration(
        payload["prior_configuration"]
    ).to(device)
    prior.load_state_dict(payload["prior_state_dict"])
    prior.eval()
    return prior, payload


def rows_near_kp(bank, validation_rows, target, tolerance):
    validation_rows = np.asarray(validation_rows, dtype=np.int64)
    selected = np.abs(bank["kp"][validation_rows] - target) <= tolerance + 1e-6
    rows = validation_rows[selected]
    if not len(rows):
        raise ValueError(f"no held-out frames lie within Kp {target:g}+/-{tolerance:g}")
    return rows


def evaluate_density(priors, bank, rows, center, scale, arguments, device):
    loader = DataLoader(
        latent_dataset(bank, rows), batch_size=arguments.density_batch_size,
        shuffle=False,
    )
    center = torch.from_numpy(center).to(device)
    scale = torch.from_numpy(scale).to(device)
    generator = torch.Generator(device=device).manual_seed(arguments.seed + 40_000)
    values = {label: [] for label in priors}
    with torch.no_grad():
        for mu, logvar, kp in loader:
            mu, logvar, kp = mu.to(device), logvar.to(device), kp.to(device)
            batch_values = {label: [] for label in priors}
            for _ in range(arguments.posterior_samples):
                latent = sample_standardized_posterior(
                    mu, logvar, center, scale, generator
                )
                for label, prior in priors.items():
                    batch_values[label].append(prior.log_prob(latent, kp))
            for label in priors:
                mean = torch.stack(batch_values[label]).mean(0)
                values[label].append(mean.cpu().numpy())
    return {
        label: np.concatenate(parts).astype(np.float64)
        for label, parts in values.items()
    }


#%% Decoded intensity and morphology distributions

def field_metrics(fields, support, mlat, mlt, thresholds):
    metrics = {}
    for channel, field in enumerate(FIELD_NAMES):
        mask = support[channel]
        values = fields[:, channel][:, mask].astype(np.float64)
        latitude = mlat[mask]
        angle = mlt[mask] * (2 * np.pi / 24)
        weights = np.maximum(values, 0)
        weight_sum = weights.sum(axis=1)
        safe_sum = np.where(weight_sum > 0, weight_sum, np.nan)
        weighted_latitude = (weights * latitude).sum(axis=1) / safe_sum
        latitude_width = np.sqrt(
            (weights * np.square(latitude - weighted_latitude[:, None])).sum(axis=1)
            / safe_sum
        )
        sine = (weights * np.sin(angle)).sum(axis=1)
        cosine = (weights * np.cos(angle)).sum(axis=1)
        weighted_mlt = np.mod(np.arctan2(sine, cosine), 2 * np.pi) * (24 / (2 * np.pi))
        metrics[field] = {
            "maximum": values.max(axis=1),
            "supported_mean": values.mean(axis=1),
            "active_fraction": np.mean(values > thresholds[channel], axis=1),
            "weighted_mlat": weighted_latitude,
            "mlat_width": latitude_width,
            "weighted_mlt": weighted_mlt,
        }
    return metrics


def decode_metrics(model, transform, latent, target_kp, arguments, device,
                   support, mlat, mlt):
    collected = {
        field: {metric: [] for metric in (
            "maximum", "supported_mean", "active_fraction",
            "weighted_mlat", "mlat_width", "weighted_mlt",
        )} for field in FIELD_NAMES
    }
    thresholds = (arguments.wic_active_threshold, arguments.si12_active_threshold)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(latent), arguments.decode_batch_size):
            stop = min(start + arguments.decode_batch_size, len(latent))
            values = torch.from_numpy(latent[start:stop]).to(device)
            kp = torch.full(
                (len(values),), float(target_kp), dtype=torch.float32, device=device
            )
            fields = transform.inverse(model.decode(values, kp=kp).cpu().numpy())
            batch = field_metrics(fields, support, mlat, mlt, thresholds)
            for field in FIELD_NAMES:
                for metric, metric_values in batch[field].items():
                    collected[field][metric].append(metric_values)
    return {
        field: {
            metric: np.concatenate(parts)
            for metric, parts in field_metrics_.items()
        } for field, field_metrics_ in collected.items()
    }


def summarize_metrics(metrics):
    summary = {}
    for field, field_metrics_ in metrics.items():
        summary[field] = {}
        for metric, values in field_metrics_.items():
            finite = values[np.isfinite(values)]
            summary[field][metric] = (
                {
                    "q05": float(np.quantile(finite, 0.05)),
                    "median": float(np.median(finite)),
                    "q95": float(np.quantile(finite, 0.95)),
                    "q99": float(np.quantile(finite, 0.99)),
                } if len(finite) else {
                    "q05": None, "median": None, "q95": None, "q99": None,
                }
            )
    return summary


def channel_mean_correlation(metrics):
    wic = metrics["wic_corrected"]["supported_mean"]
    si12 = metrics["si12"]["supported_mean"]
    finite = np.isfinite(wic) & np.isfinite(si12)
    if finite.sum() < 2 or np.std(wic[finite]) == 0 or np.std(si12[finite]) == 0:
        return None
    return float(np.corrcoef(wic[finite], si12[finite])[0, 1])


def calibration_comparison(reference, candidate):
    comparison = {}
    for field in FIELD_NAMES:
        comparison[field] = {}
        for metric in reference[field]:
            first = reference[field][metric]
            second = candidate[field][metric]
            finite = np.isfinite(first)
            candidate_finite = np.isfinite(second)
            if not finite.any() or not candidate_finite.any():
                comparison[field][metric] = {"ks_distance": None}
                continue
            record = {"ks_distance": float(ks_2samp(
                first[finite], second[candidate_finite]
            ).statistic)}
            if metric in DISPLAY_METRICS and np.nanmedian(first) != 0:
                record["median_ratio"] = float(
                    np.nanmedian(second) / np.nanmedian(first)
                )
            elif metric != "weighted_mlt":
                record["median_difference"] = float(
                    np.nanmedian(second) - np.nanmedian(first)
                )
            comparison[field][metric] = record
    return comparison


#%% Compact comparison figures

def plot_density(records, labels, output):
    kp = np.asarray([record["target_kp"] for record in records])
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for label in labels:
        mean = [record["priors"][label]["mean_log_density"] for record in records]
        error = [record["priors"][label]["standard_error"] for record in records]
        axes[0].errorbar(kp, mean, yerr=error, marker="o", label=label)
    gain = [record["selected_minus_baseline"]["mean"] for record in records]
    gain_error = [record["selected_minus_baseline"]["standard_error"] for record in records]
    axes[1].errorbar(kp, gain, yerr=gain_error, marker="o", color="tab:green")
    axes[1].axhline(0, color="black", linewidth=1)
    axes[0].set(title="Held-out latent density by Kp", xlabel="Kp", ylabel="mean log density")
    axes[1].set(title="Selected prior gain", xlabel="Kp", ylabel="paired log-density gain")
    axes[0].legend()
    figure.savefig(output / "density_by_kp.png", dpi=180)
    figure.savefig(output / "density_by_kp.pdf")
    plt.close(figure)


def plot_calibration(raw, labels, output):
    targets = sorted(raw)
    colors = {"heldout_posterior": "black", labels[0]: "tab:blue", labels[-1]: "tab:orange"}
    figure, axes = plt.subplots(2, 3, figsize=(14, 8), layout="constrained")
    for row, (field, field_label) in enumerate(zip(FIELD_NAMES, FIELD_LABELS)):
        for column, metric in enumerate(DISPLAY_METRICS):
            axis = axes[row, column]
            for source in ("heldout_posterior", *labels):
                quantiles = np.asarray([
                    np.quantile(raw[target][source][field][metric], (0.05, 0.5, 0.95, 0.99))
                    for target in targets
                ])
                axis.plot(targets, quantiles[:, 1], marker="o", color=colors[source], label=source)
                axis.fill_between(targets, quantiles[:, 0], quantiles[:, 2], color=colors[source], alpha=0.12)
                axis.plot(targets, quantiles[:, 3], linestyle=":", color=colors[source], alpha=0.8)
            axis.set(title=f"{field_label}: {metric.replace('_', ' ')}", xlabel="Kp")
    axes[0, 0].legend(fontsize="small")
    figure.suptitle("Median, 5--95% interval and dotted 99th percentile")
    figure.savefig(output / "decoded_calibration_by_kp.png", dpi=180)
    figure.savefig(output / "decoded_calibration_by_kp.pdf")
    plt.close(figure)


#%% End-to-end evaluation

def run(arguments):
    bank_path = arguments.bank.expanduser().resolve()
    run_directory = arguments.run_directory.expanduser().resolve()
    prior_directory = arguments.prior_directory.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose an empty prior-evaluation output directory")
    output.mkdir(parents=True, exist_ok=True)
    device_name = "cuda" if arguments.device == "auto" and torch.cuda.is_available() else arguments.device
    device = torch.device("cpu" if device_name == "auto" else device_name)
    torch.set_num_threads(arguments.threads)

    bank = load_bank(bank_path)
    fit_summary = json.loads((prior_directory / "posthoc_prior_summary.json").read_text())
    priors, payloads = {}, {}
    for candidate in fit_summary["candidates"]:
        label = f"K={candidate['components']}"
        prior, payload = load_prior(
            prior_directory / candidate["directory"] / "conditional_latent_prior.pt",
            device,
        )
        priors[label], payloads[label] = prior, payload
    labels = list(priors)
    baseline_label = f"K={min(candidate['components'] for candidate in fit_summary['candidates'])}"
    selected_label = f"K={fit_summary['selected_components']}"
    reference = payloads[baseline_label]
    validation_rows = np.asarray(reference["validation_rows"], dtype=np.int64)
    center = np.asarray(reference["latent_center"], dtype=np.float32)
    scale = np.asarray(reference["latent_scale"], dtype=np.float32)
    for label, payload in payloads.items():
        if (
            not np.array_equal(validation_rows, payload["validation_rows"])
            or not np.allclose(center, payload["latent_center"])
            or not np.allclose(scale, payload["latent_scale"])
        ):
            raise ValueError(f"{label} was not fit on the same split and standardization")

    artifact_path = run_directory / "wic_cvae.pt"
    artifact_hash = sha256_file(artifact_path)
    bank_hash = sha256_file(bank_path)
    if any(
        payload["artifact_sha256"] != artifact_hash or payload["bank_sha256"] != bank_hash
        for payload in payloads.values()
    ):
        raise ValueError("saved priors do not match the supplied CVAE artifact and latent bank")
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if (
        tuple(model.field_names) != FIELD_NAMES
        or model.latent_dim != bank["mu"].shape[1]
        or model.kp_conditioning is None or model.conditioning is not None
    ):
        raise ValueError("latent bank and paired scalar-Kp WIC/SI12 artifact do not align")
    support = np.asarray(artifact["data_provenance"]["training_support"], dtype=bool)
    grid = artifact["data_provenance"]["grid"]
    mlat, mlt = np.asarray(grid["mlat"]), np.asarray(grid["mlt"])
    if support.shape != (len(FIELD_NAMES), *model.image_shape):
        raise ValueError("artifact support does not align with model outputs")
    if mlat.shape != model.image_shape or mlt.shape != model.image_shape:
        raise ValueError("artifact grid does not align with model outputs")

    density_records, raw_calibration, component_fractions, saved_arrays = [], {}, {}, {}
    for target in arguments.kp:
        rows = rows_near_kp(bank, validation_rows, target, arguments.kp_tolerance)
        values = evaluate_density(priors, bank, rows, center, scale, arguments, device)
        prior_records = {}
        for label, log_density in values.items():
            prior_records[label] = {
                "mean_log_density": float(np.mean(log_density)),
                "standard_error": float(np.std(log_density, ddof=1) / np.sqrt(len(log_density))),
            }
        gain = values[selected_label] - values[baseline_label]
        density_records.append({
            "target_kp": float(target), "heldout_frames": int(len(rows)),
            "priors": prior_records,
            "selected_minus_baseline": {
                "mean": float(np.mean(gain)),
                "standard_error": float(np.std(gain, ddof=1) / np.sqrt(len(gain))),
                "frame_win_fraction": float(np.mean(gain > 0)),
            },
        })

        rng = np.random.default_rng(arguments.seed + int(round(target * 1000)))
        source_rows = rng.choice(rows, size=arguments.draws, replace=len(rows) < arguments.draws)
        latent = bank["mu"][source_rows].astype(np.float32).copy()
        latent += rng.standard_normal(latent.shape, dtype=np.float32) * np.exp(
            0.5 * bank["logvar"][source_rows]
        )
        target_raw = {
            "heldout_posterior": decode_metrics(
                model, transform, latent, target, arguments, device, support, mlat, mlt
            )
        }
        for number, label in enumerate(labels):
            condition = torch.full((arguments.draws,), float(target), device=device)
            generator = torch.Generator(device=device).manual_seed(
                arguments.seed + 50_000 + number + int(round(target * 1000))
            )
            with torch.no_grad():
                standardized, components = priors[label].sample(condition, generator)
                latent = center + standardized.cpu().numpy() * scale
            target_raw[label] = decode_metrics(
                model, transform, latent.astype(np.float32), target,
                arguments, device, support, mlat, mlt,
            )
            saved_arrays[f"kp_{target:g}_{label}_component"] = components.cpu().numpy()
            component_fractions.setdefault(str(float(target)), {})[label] = {
                str(component + 1): float(np.mean(components.cpu().numpy() == component))
                for component in range(priors[label].components)
            }
        raw_calibration[float(target)] = target_raw

    calibration = {}
    for target, sources in raw_calibration.items():
        source_summaries = {
            source: summarize_metrics(metrics) for source, metrics in sources.items()
        }
        for source, metrics in sources.items():
            source_summaries[source]["wic_si12_supported_mean_correlation"] = (
                channel_mean_correlation(metrics)
            )
        calibration[str(target)] = {
            "summaries": source_summaries,
            "candidate_vs_heldout": {
                label: calibration_comparison(sources["heldout_posterior"], sources[label])
                for label in labels
            },
        }
        for source, fields in sources.items():
            for field, metrics in fields.items():
                for metric, values in metrics.items():
                    saved_arrays[f"kp_{target:g}_{source}_{field}_{metric}"] = values

    plot_density(density_records, labels, output)
    plot_calibration(raw_calibration, labels, output)
    np.savez_compressed(output / "decoded_metric_samples.npz", **saved_arrays)
    summary = {
        "latent_bank": str(bank_path), "run_directory": str(run_directory),
        "prior_directory": str(prior_directory), "device": str(device),
        "kp_tolerance": arguments.kp_tolerance,
        "posterior_samples_per_density_frame": arguments.posterior_samples,
        "decoded_draws_per_source_and_kp": arguments.draws,
        "active_thresholds": {
            "wic_corrected": arguments.wic_active_threshold,
            "si12": arguments.si12_active_threshold,
        },
        "baseline": baseline_label, "selected": selected_label,
        "density_by_kp": density_records, "decoded_calibration": calibration,
        "sampled_component_fractions": component_fractions,
        "reference_definition": (
            "orbit-held-out encoder posterior samples decoded at the target Kp; "
            "this isolates prior calibration from CVAE reconstruction error"
        ),
    }
    (output / "posthoc_prior_evaluation.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"Post-hoc prior evaluation written to {output}", flush=True)
    return summary


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bank", type=Path)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("prior_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kp", type=float, nargs="+", default=(0, 2, 4, 6))
    parser.add_argument("--kp-tolerance", type=float, default=0.34)
    parser.add_argument("--posterior-samples", type=int, default=4)
    parser.add_argument("--density-batch-size", type=int, default=1024)
    parser.add_argument("--draws", type=int, default=2000)
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--wic-active-threshold", type=float, default=100)
    parser.add_argument("--si12-active-threshold", type=float, default=5)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)
    if (
        not np.isfinite(arguments.kp).all()
        or any(not 0 <= value <= 9 for value in arguments.kp)
        or not np.allclose(
            np.asarray(arguments.kp) * 3,
            np.round(np.asarray(arguments.kp) * 3),
            atol=0.0011, rtol=0,
        )
        or len(set(arguments.kp)) != len(arguments.kp)
        or not np.isfinite(arguments.kp_tolerance) or arguments.kp_tolerance < 0
        or arguments.posterior_samples < 1 or arguments.density_batch_size < 1
        or arguments.draws < 2 or arguments.decode_batch_size < 1
        or not np.isfinite((
            arguments.wic_active_threshold, arguments.si12_active_threshold,
        )).all()
        or arguments.wic_active_threshold < 0 or arguments.si12_active_threshold < 0
        or arguments.threads < 1
    ):
        parser.error("invalid Kp, tolerance, sampling, batching, threshold or thread setting")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
