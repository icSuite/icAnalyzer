"""Audit saved VAE loss gradients without training or loading prior ensembles."""

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
from torch.utils.data import DataLoader, Subset

from icanalyzer.wic_si12.count_likelihood import COUNT_UNCERTAINTY_LOSS_KIND, fit_count_uncertainty_configuration
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


#%% Fixed observation strata and exact per-cell loss decomposition
GROUPS = ["zero", "faint", "middle", "bright"]


def observation_groups(values, weight, bright_threshold):
    low = np.array([50.0, 1.0]).reshape(1, 2, 1, 1)
    bright = np.asarray(bright_threshold).reshape(1, 2, 1, 1)
    supported = np.broadcast_to(weight > 0, values.shape)
    return np.stack([supported & (values == 0), supported & (values > 0) & (values <= low),
                     supported & (values > low) & (values < bright),
                     supported & (values > low) & (values >= bright)], axis=2)


def fit_bright_threshold(dataset):
    thresholds = []
    for channel, low in enumerate((50.0, 1.0)):
        y = dataset.values[:, channel]
        q = dataset.weight[:, 0]
        use = (q > 0) & np.isfinite(y) & (y > 0)
        if not np.any(use):
            raise ValueError("intensity audit requires positive observations in each channel")
        order = np.argsort(y[use])
        cumulative = np.cumsum(q[use][order].astype(float))
        threshold = y[use][order][np.searchsorted(cumulative, .9 * cumulative[-1])]
        thresholds.append(float(max(low, threshold)))
    return thresholds


def loss_cells(reconstructed, batch, configuration, measurements=True, fixed_reference=False):
    """Already quality-weighted and reduced cells; sum equals reconstruction.

    Diagnostic formulas are checked against the production loss in tests.
    Fixed-reference count cases are counterfactuals, NOT implemented training
    options: they isolate changing variance from its coupled f**2 temperature.
    """
    q = torch.where(batch["mask"] & torch.isfinite(batch["weight"]), batch["weight"].clamp(min=0), 0.0)
    active = (q > 0).expand_as(reconstructed)
    decoded = torch.where(active, reconstructed, 0.0)
    target = torch.where(active, batch["target"], 0.0)
    kind = configuration["kind"]
    if kind == "legacy_weighted_mse_v1":
        cells = (decoded - target).square()
    elif kind == "quality_weighted_gaussian_v1":
        v = torch.where(active, batch["variance"], 0.0) if measurements else torch.zeros_like(decoded)
        tau2 = decoded.new_tensor(configuration["model_error_std"]).reshape(1, 2, 1, 1).square()
        cells = tau2 * ((decoded - target).square() / (v + tau2) + torch.log1p(v / tau2))
    elif kind == COUNT_UNCERTAINTY_LOSS_KIND:
        center = decoded.new_tensor(configuration["transform"]["center"]).reshape(1, 2, 1, 1)
        spread = decoded.new_tensor(configuration["transform"]["spread"]).reshape(1, 2, 1, 1)
        scale = decoded.new_tensor(configuration["channel_scale_counts"]).reshape(1, 2, 1, 1)
        f2 = decoded.new_tensor(configuration["model_error_fraction"]).reshape(1, 2, 1, 1).square()
        mean = torch.expm1(np.log(10) * (decoded * spread + center)) / scale
        y = torch.where(active, batch["physical_target"], 0.0) / scale
        sigma = torch.where(active, batch["physical_standard_deviation"], 0.0) / scale if measurements else torch.zeros_like(decoded)
        v = sigma.square() + f2
        reference = torch.full_like(f2, .5**2) if fixed_reference else f2
        density = reference * ((mean - y).square() / v + torch.log(v / f2))
        censored = -2 * reference * torch.special.log_ndtr(-mean / torch.sqrt(v))
        cells = torch.where(y == 0, censored, density)
    else:
        raise ValueError(f"unsupported diagnostic loss: {kind}")
    cells = torch.where(active, cells, 0.0) * q * np.prod(reconstructed.shape[1:]) / (q.sum() + 1e-8)
    if not torch.all(torch.isfinite(cells)):
        raise ValueError("nonfinite loss cells; inspect inputs and decoder")
    return cells


def gradient_statistics(wic, si12, kl):
    """L2 norms and alignment on the SAME encoder parameter space."""
    w2 = sum(g.double().square().sum() for g in wic)
    s2 = sum(g.double().square().sum() for g in si12)
    k2 = sum(g.double().square().sum() for g in kl)
    r2 = sum((a.double() + b.double()).square().sum() for a, b in zip(wic, si12))
    rk = sum(((a.double() + b.double()) * k.double()).sum() for a, b, k in zip(wic, si12, kl))
    ws = sum((a.double() * b.double()).sum() for a, b in zip(wic, si12))
    return np.array([float(torch.sqrt(w2)), float(torch.sqrt(s2)), float(torch.sqrt(r2)),
                     float(torch.sqrt(k2)), float(rk / torch.sqrt(r2 * k2).clamp(min=1e-30)),
                     float(ws / torch.sqrt(w2 * s2).clamp(min=1e-30))])


def gradient_difference(reference, actual):
    """Relative L2 change and cosine, including changes in gradient direction."""
    r2 = sum(g.double().square().sum() for g in reference)
    a2 = sum(g.double().square().sum() for g in actual)
    delta2 = sum((a.double() - r.double()).square().sum() for r, a in zip(reference, actual))
    dot = sum((a.double() * r.double()).sum() for r, a in zip(reference, actual))
    return [float(torch.sqrt(delta2 / r2.clamp(min=1e-30))),
            float(dot / torch.sqrt(a2 * r2).clamp(min=1e-30))]


#%% One fixed graph/noise draw per batch, several loss counterfactuals
def audit_model(model, dataset, configuration, count_configuration, indices, batch_size, seed, beta, bright_threshold):
    scenarios = {"native": (configuration, True, False), "native_no_measurements": (configuration, False, False)}
    for fraction in (.5, .1, .03):
        config = dict(count_configuration, model_error_fraction=[fraction, fraction])
        for measured in (True, False):
            scenarios[f"count_f{fraction:g}_{'measured' if measured else 'no_measurements'}"] = (config, measured, False)
        if fraction != .5:
            scenarios[f"count_f{fraction:g}_fixed_reference"] = (config, True, True)
    parameters = [p for name, p in model.named_parameters() if name.startswith(("enc_", "fc_mu.", "fc_logvar.")) and p.requires_grad]
    if not parameters:
        raise ValueError("no canonical shared encoder parameters found")
    model.eval()
    generator = torch.Generator().manual_seed(seed)
    totals = {name: {"pixel_gradient_l1": np.zeros((2, 4)), "loss_sum": np.zeros(2), "encoder_batches": []} for name in scenarios}
    frame_hash = hashlib.sha256()
    for batch in DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False):
        measured_gradients = {}
        mu, logvar = model.encode(batch["encoder_input"], batch["mask"])
        noise = torch.randn(mu.shape, generator=generator)
        reconstructed = model.decode(mu + torch.exp(.5 * logvar) * noise)
        kl_loss = -.5 * beta * torch.sum(1 + logvar - mu.square() - logvar.exp())
        kl_grad = torch.autograd.grad(kl_loss, parameters, retain_graph=True)
        groups = observation_groups(batch["physical_target"].numpy(), batch["weight"].numpy(), bright_threshold)
        frame_hash.update(batch["index"].numpy().tobytes())
        for name, (config, measurements, fixed_reference) in scenarios.items():
            cells = loss_cells(reconstructed, batch, config, measurements, fixed_reference)
            channel_loss = cells.sum(dim=(0, 2, 3))
            pixel_gradient = torch.autograd.grad(channel_loss.sum(), reconstructed, retain_graph=True)[0].detach().abs().numpy()
            wic = torch.autograd.grad(channel_loss[0], parameters, retain_graph=True)
            si12 = torch.autograd.grad(channel_loss[1], parameters, retain_graph=True)
            statistics = gradient_statistics(wic, si12, kl_grad)
            if not np.all(np.isfinite(statistics)) or not np.all(np.isfinite(pixel_gradient)):
                raise ValueError("nonfinite gradients; inspect decoder and likelihood tails")
            totals[name]["encoder_batches"].append(statistics.tolist())
            reconstruction_gradient = [(a + b).detach() for a, b in zip(wic, si12)]
            if name == "native" or name.endswith("_measured"):
                measured_gradients[name] = reconstruction_gradient
            elif name.endswith("_no_measurements"):
                reference_name = name.replace("_no_measurements", "") if name.startswith("native") else name.replace("_no_measurements", "_measured")
                difference = gradient_difference(measured_gradients[reference_name], reconstruction_gradient)
                totals[name].setdefault("encoder_measurement_removal_batches", []).append(difference)
            totals[name]["loss_sum"] += channel_loss.detach().numpy()
            totals[name]["pixel_gradient_l1"] += (pixel_gradient[:, :, None] * groups).sum(axis=(0, 3, 4))
        del reconstructed, cells, wic, si12, kl_grad, measured_gradients, reconstruction_gradient
    for name, record in totals.items():
        norms = np.asarray(record["encoder_batches"])
        record["encoder_mean_norms_wic_si12_reconstruction_betaKL"] = norms[:, :4].mean(axis=0).tolist()
        record["mean_encoder_cosines_reconstruction_KL_and_channels"] = norms[:, 4:].mean(axis=0).tolist()
        record["reconstruction_to_betaKL_norm_ratio"] = float(norms[:, 2].mean() / norms[:, 3].mean()) if np.any(norms[:, 3] > 0) else None
        mass = record["pixel_gradient_l1"]
        record["pixel_gradient_share"] = (mass / max(mass.sum(), 1e-30)).tolist()
        record["pixel_gradient_l1"] = mass.tolist()
        record["loss_sum"] = record["loss_sum"].tolist()
        record["settings"] = {"configuration": scenarios[name][0], "measurements": scenarios[name][1], "fixed_reference": scenarios[name][2]}
        if "encoder_measurement_removal_batches" in record:
            difference = np.asarray(record["encoder_measurement_removal_batches"]).mean(axis=0)
            record["mean_relative_encoder_gradient_change_on_measurement_removal"] = float(difference[0])
            record["mean_encoder_cosine_on_measurement_removal"] = float(difference[1])
    return totals, frame_hash.hexdigest()


#%% Plots and saved report
def plot_audit(results, output):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    labels = []
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]
    for name, result in results.items():
        native = result["scenarios"]["native"]
        share = np.asarray(native["pixel_gradient_share"])
        labels.append(name.replace("wic_si12_smoothed_b16_holes", "holes"))
        index = len(labels) - 1
        for c, field in enumerate(("WIC", "SI12")):
            bottom = 0
            for group, value, color in zip(GROUPS, share[c], colors):
                axes[c].bar(index, value, bottom=bottom, color=color, label=group if index == 0 else None)
                bottom += value
            axes[c].set(title=f"{field}: share of BOTH fields' output-gradient L1", ylim=(0, 1))
    for axis in axes:
        axis.set_xticks(range(len(labels)), labels)
        axis.legend(fontsize=8)
    figure.savefig(output / "native_gradient_budget.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    fractions = [.03, .1, .5]
    for name, result in results.items():
        label = name.replace("wic_si12_smoothed_b16_holes", "holes")
        scenarios = result["scenarios"]
        ratios = [scenarios[f"count_f{f:g}_measured"]["reconstruction_to_betaKL_norm_ratio"] for f in fractions]
        no_errors = [scenarios[f"count_f{f:g}_no_measurements"]["reconstruction_to_betaKL_norm_ratio"] for f in fractions]
        fixed = [scenarios[f"count_f{f:g}_fixed_reference" if f != .5 else "count_f0.5_measured"]["reconstruction_to_betaKL_norm_ratio"] for f in fractions]
        axes[0].plot(fractions, ratios, "o-", label=label)
        axes[0].plot(fractions, no_errors, ":", label=f"{label}, no measurement errors")
        axes[1].plot(fractions, ratios, "o-", label=f"{label}, current scaling")
        axes[1].plot(fractions, fixed, "--", label=f"{label}, fixed reference (diagnostic)")
    for axis, title in zip(axes, ("Measurement-error sensitivity", "Variance vs coupled loss-temperature sensitivity")):
        axis.set(xscale="log", yscale="log", xlabel="Model-error fraction per channel", ylabel="Mean encoder recon norm / beta-KL norm", title=title)
        axis.legend(fontsize=7)
    figure.savefig(output / "allowance_and_KL_sensitivity.png", dpi=140)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    for index, (name, result) in enumerate(results.items()):
        label = name.replace("wic_si12_smoothed_b16_holes", "holes")
        scenarios = result["scenarios"]
        native = 100 * scenarios["native_no_measurements"]["mean_relative_encoder_gradient_change_on_measurement_removal"]
        axes[0].bar(index, native, color=f"C{index}")
        axes[0].text(index, native + 2, f"{native:.2f}%", ha="center")
        change = [100 * scenarios[f"count_f{f:g}_no_measurements"]["mean_relative_encoder_gradient_change_on_measurement_removal"] for f in fractions]
        axes[1].plot(fractions, change, "o-", label=label)
    axes[0].set_xticks(range(len(labels)), labels)
    axes[0].set(title="Native loss: remove measured variance", ylabel="Mean relative encoder-gradient change (%)", ylim=(0, max(5, axes[0].get_ylim()[1] * 1.1)))
    axes[1].set(title="Count counterfactuals: remove measured variance", xscale="log", xlabel="Model-error fraction per channel", ylabel="Mean relative encoder-gradient change (%)")
    axes[1].legend(fontsize=8)
    figure.savefig(output / "measurement_gradient_change.png", dpi=140)
    plt.close(figure)


def run(arguments):
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("choose a fresh output; existing diagnostics are preserved")
    artifacts = [load_run_artifact(path / "wic_si12_vae.pt") for path in arguments.runs]
    first = artifacts[0][2]
    for _, _, artifact in artifacts[1:]:
        if (artifact["split_files"] != first["split_files"] or artifact["transform"] != first["transform"]
                or artifact["model_configuration"] != first["model_configuration"]):
            raise ValueError("loss audit requires identical saved splits, transforms and architectures")
        for key in ("minimum_sza_degrees", "maximum_missing_fraction", "comparison_data_version", "beta"):
            if artifact["training_configuration"].get(key) != first["training_configuration"].get(key):
                raise ValueError(f"loss audit settings differ: {key}")
        for key in ("manual_frame_intervals", "manual_annotations", "loss_quality_weight_source", "upstream_quality_weight_interpretation", "legacy_comparison_directory", "products"):
            saved = json.dumps(artifact["data_provenance"].get(key), sort_keys=True, default=lambda v: np.asarray(v).tolist())
            reference = json.dumps(first["data_provenance"].get(key), sort_keys=True, default=lambda v: np.asarray(v).tolist())
            if saved != reference:
                raise ValueError(f"loss audit data provenance differs: {key}")
    dataset, split = _load_heldout_dataset(first, artifacts[0][1], arguments.split)
    count_configuration = next((a[2]["training_configuration"]["loss_configuration"] for a in artifacts
                                if a[2]["training_configuration"]["loss_configuration"]["kind"] == COUNT_UNCERTAINTY_LOSS_KIND), None)
    if count_configuration is None:
        training, _ = _load_heldout_dataset(first, artifacts[0][1], "train")
        count_configuration = fit_count_uncertainty_configuration(training)
        del training
    bright = fit_bright_threshold(dataset)
    indices = np.arange(len(dataset))
    if arguments.max_frames is not None:
        indices = np.sort(np.random.default_rng(arguments.seed).choice(indices, min(len(indices), arguments.max_frames), replace=False))
    results = {}
    for path, (model, transform, artifact) in zip(arguments.runs, artifacts):
        print(f"Auditing {path.name}: {len(indices)} {split} frames", flush=True)
        before = hashlib.sha256((path / "wic_si12_vae.pt").read_bytes()).hexdigest()
        config = artifact["training_configuration"]["loss_configuration"]
        scenarios, frame_hash = audit_model(model, dataset, config, count_configuration, indices,
                                            arguments.batch_size, arguments.seed, artifact["training_configuration"]["beta"], bright)
        after = hashlib.sha256((path / "wic_si12_vae.pt").read_bytes()).hexdigest()
        if before != after:
            raise RuntimeError("saved model changed during the audit")
        results[path.name] = {"artifact_sha256": before, "frame_indices_sha256": frame_hash, "scenarios": scenarios}
    groups = observation_groups(dataset.values[indices], dataset.weight[indices], bright)
    quality_mass = (dataset.weight[indices][:, :, None] * groups).sum(axis=(0, 3, 4))
    report = {"split": split, "frame_count": len(indices), "batch_size": arguments.batch_size, "seed": arguments.seed,
              "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "run_paths": [str(path.resolve()) for path in arguments.runs], "device": "cpu",
              "frame_orbit": dataset.frame_orbit[indices].tolist(), "frame_time": dataset.frame_time[indices].tolist(),
              "groups": GROUPS, "bright_threshold_counts": bright, "quality_mass_per_channel_group": quality_mass.tolist(),
              "results": results, "limitations": ["Eval-mode, full-input seeded posterior draw; not exact train-mode/augmented optimization",
              "Output-gradient L1 is in standardized-log coordinates; encoder L2 uses identical shared parameter space",
              "Norms are means of per-batch norms, NOT the norm of a corpus-averaged gradient",
              "Different saved models have different encoder Jacobians; compare counterfactuals within a model",
              "Variance offsets contribute to loss values but not gradients; losses cannot rank different objectives",
              "Fixed-reference scenarios are diagnostic only; smaller fractions are not demonstrated skill improvements",
              "No optimizer, parameter update, prior ensemble load or retraining"]}
    plot_audit(results, output)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--max-frames", type=int)
    arguments = parser.parse_args(argv)
    if arguments.batch_size <= 0 or (arguments.max_frames is not None and arguments.max_frames <= 0):
        parser.error("batch size and optional frame limit must be positive")
    if len(set(path.name for path in arguments.runs)) != len(arguments.runs):
        parser.error("run names must be unique")
    return arguments


if __name__ == "__main__":
    report = run(parse_arguments())
    print(f"Saved diagnostic for {len(report['results'])} models and {report['frame_count']} frames")
