"""Create reproducible diagnostics from a saved WIC/SI12 or native conductance VAE."""

#%% Imports

import argparse
import gc
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.wic_si12.comparison import load_paired_files, pair_orbit_files
from scripts.manual_masking.annotated_data import load_annotations, annotation_intervals, apply_annotations
from icanalyzer.wic_si12.data import (
    WICSI12ArrayDataset,
    load_good_orbit_intervals,
    load_precipitation_cs_files,
)
from icanalyzer.wic_si12.diagnostics import (
    anomaly_variance,
    best_validation_epoch,
    covariance_mode_decomposition,
    joint_correlation_maps,
    maximum_variance_anchor,
    reconstruct_ensemble,
)
from icanalyzer.wic_si12.workflow import load_run_artifact


#%% Plot helpers

FIELD_NAMES = {
    "fuv": ("corrected WIC", "SI12"),
    "conductance": ("Hall", "Pedersen"),
}
FIELD_UNITS = {
    "fuv": ("counts", "counts"),
    "conductance": ("S", "S"),
}


def image_panel(
    axis,
    values,
    title,
    colorbar_label,
    cmap="viridis",
    limits=None,
    show_xlabel=True,
    show_ylabel=True,
):
    settings = {}
    if limits is not None:
        settings.update(vmin=limits[0], vmax=limits[1])
    image = axis.imshow(values, origin="lower", cmap=cmap, **settings)
    axis.set_title(title)
    if show_xlabel:
        axis.set_xlabel("CS column")
    if show_ylabel:
        axis.set_ylabel("CS row")
    plt.colorbar(image, ax=axis, shrink=0.78, label=colorbar_label)
    return image


def robust_limits(values, signed=False):
    values = np.asarray(values)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("cannot determine plot limits without finite values")
    if signed:
        limit = float(np.percentile(np.abs(values), 99))
        limit = max(limit, np.finfo(float).eps)
        return -limit, limit
    maximum = max(float(np.percentile(values, 99)), np.finfo(float).eps)
    return 0.0, maximum


def save_figure(figure, output, name, created):
    filename = output / name
    figure.savefig(filename, dpi=180, bbox_inches="tight")
    plt.close(figure)
    created.append(name)


#%% Training history

def plot_loss_history(history, output, created):
    epochs = np.array([record["epoch"] for record in history])
    has_learning_rate = all("learning_rate" in record for record in history)
    columns = 3 if has_learning_rate else 2
    figure, axes = plt.subplots(
        1, columns, figsize=(6 * columns, 4.5), layout="constrained"
    )

    for split, style in (("train", "-"), ("validation", "--")):
        if split not in history[0]:
            continue
        total = [record[split]["total"] for record in history]
        reconstruction = [record[split]["reconstruction"] for record in history]
        kl = [record[split]["kl"] for record in history]
        axes[0].plot(epochs, total, style, label=f"{split} total")
        axes[0].plot(
            epochs,
            reconstruction,
            style,
            alpha=0.65,
            label=f"{split} reconstruction",
        )
        axes[1].plot(epochs, kl, style, label=split)

    best_epoch, best_metrics = best_validation_epoch(history)
    if best_epoch is not None:
        axes[0].axvline(
            best_epoch,
            color="tab:red",
            alpha=0.7,
            label=f"best validation: {best_epoch}",
        )
        axes[0].scatter([best_epoch], [best_metrics["total"]], color="tab:red", zorder=4)

    axes[0].set(title="VAE loss", xlabel="epoch", ylabel="per-sample loss")
    axes[1].set(title="KL term before beta weighting", xlabel="epoch", ylabel="KL")
    if has_learning_rate:
        learning_rates = [record["learning_rate"] for record in history]
        axes[2].semilogy(epochs, learning_rates, color="tab:green")
        axes[2].set(
            title="ReduceLROnPlateau schedule",
            xlabel="epoch",
            ylabel="learning rate",
        )
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, _labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    save_figure(figure, output, "loss_history.png", created)
    return best_epoch, best_metrics


#%% Ensemble samples and moments

def plot_sample_mosaic(samples, sample_indices, family, output, created):
    if len(sample_indices) < 2:
        raise ValueError("at least two sample examples are required")
    n_examples = len(sample_indices)
    names = FIELD_NAMES[family]
    units = FIELD_UNITS[family]
    figure, axes = plt.subplots(
        2, n_examples, figsize=(2.7 * n_examples, 5.5), layout="constrained"
    )

    for channel in range(2):
        signed = bool(np.any(samples[:, channel] < 0))
        limits = robust_limits(samples[:, channel], signed=signed)
        cmap = "coolwarm" if signed else "viridis"
        for column, sample_index in enumerate(sample_indices):
            image_panel(
                axes[channel, column],
                samples[column, channel],
                f"{names[channel]} sample {sample_index}",
                units[channel],
                cmap=cmap,
                limits=limits,
                show_xlabel=channel == 1,
                show_ylabel=column == 0,
            )
    save_figure(figure, output, f"{family}_samples.png", created)
    return np.asarray(sample_indices, dtype=int).tolist()


def plot_ensemble_moments(moments, output, created):
    fields = []
    for family, statistics in moments.items():
        for channel in range(2):
            fields.append((statistics["mean"][channel], statistics["standard_deviation"][channel], FIELD_NAMES[family][channel], FIELD_UNITS[family][channel]))
    figure, axes = plt.subplots(2, len(fields), figsize=(4.25 * len(fields), 7.5), layout="constrained")
    for column, (mean, std, name, unit) in enumerate(fields):
        signed = bool(np.any(mean < 0))
        image_panel(
            axes[0, column], mean, f"mean {name}", unit,
            cmap="coolwarm" if signed else "viridis",
            limits=robust_limits(mean, signed=signed),
            show_xlabel=False,
            show_ylabel=column == 0,
        )
        image_panel(
            axes[1, column], std, f"standard deviation {name}", unit,
            limits=robust_limits(std),
            show_ylabel=column == 0,
        )
    save_figure(figure, output, "ensemble_moments.png", created)


def plot_latent_samples(products, output, created):
    if "latent_samples" not in products:
        return False

    latent = np.asarray(products["latent_samples"])
    if latent.ndim != 2 or latent.shape[0] < 2:
        raise ValueError("latent_samples must be a sample-by-dimension matrix")

    n_samples, latent_dim = latent.shape
    dimensions = np.arange(1, latent_dim + 1)
    means = np.mean(latent, axis=0)
    standard_deviations = np.std(latent, axis=0, ddof=1)
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.5), layout="constrained")

    if latent_dim == 1:
        axes[0].scatter(np.arange(n_samples), latent[:, 0], s=10, alpha=0.6)
        axes[0].set(xlabel="sample", ylabel="z1")
    else:
        axes[0].scatter(latent[:, 0], latent[:, 1], s=10, alpha=0.6)
        axes[0].set(xlabel="z1", ylabel="z2")
    axes[0].set_title("Prior latent draws")

    axes[1].errorbar(dimensions, means, yerr=standard_deviations, fmt="o", ms=3)
    axes[1].axhline(0, color="black", linewidth=1, alpha=0.5)
    axes[1].set(
        title="Marginal mean ± standard deviation",
        xlabel="latent dimension",
        ylabel="z",
    )

    subset = latent[:min(50, n_samples)]
    limit = max(float(np.percentile(np.abs(subset), 99)), np.finfo(float).eps)
    image = axes[2].imshow(
        subset,
        aspect="auto",
        origin="lower",
        cmap="coolwarm",
        vmin=-limit,
        vmax=limit,
    )
    axes[2].set(
        title=f"First {subset.shape[0]} latent samples",
        xlabel="latent dimension",
        ylabel="sample",
    )
    plt.colorbar(image, ax=axes[2], shrink=0.8, label="z")
    save_figure(figure, output, "latent_samples.png", created)
    return True


#%% Correlations and modes

def plot_correlation_maps(factor, state_shape, family, anchor, output, created):
    maps = joint_correlation_maps(factor, state_shape, *anchor)
    names = FIELD_NAMES[family]
    figure, axes = plt.subplots(2, 2, figsize=(10, 8.5), layout="constrained")
    for source in range(2):
        for target in range(2):
            image_panel(
                axes[source, target],
                maps[source, target],
                f"{names[source]} anchor → {names[target]}",
                "correlation",
                cmap="coolwarm",
                limits=(-1.0, 1.0),
                show_xlabel=source == 1,
                show_ylabel=target == 0,
            )
            axes[source, target].scatter(
                anchor[1], anchor[0], marker="x", color="black", s=45
            )
    figure.suptitle(f"Anchor at CS row {anchor[0]}, column {anchor[1]}")
    save_figure(figure, output, f"{family}_correlation_maps.png", created)


def plot_covariance_modes(
    eigenvalues, modes, total_variance, family, output, created, n_modes
):
    eigenvalues = eigenvalues[:n_modes]
    modes = modes[:n_modes]
    names = FIELD_NAMES[family]
    fraction = eigenvalues / total_variance
    figure, axes = plt.subplots(
        2,
        n_modes,
        figsize=(4.0 * n_modes, 7.0),
        squeeze=False,
        layout="constrained",
    )
    for mode_index in range(n_modes):
        for channel in range(2):
            limit = float(np.max(np.abs(modes[mode_index, channel])))
            image_panel(
                axes[channel, mode_index],
                modes[mode_index, channel],
                f"mode {mode_index + 1}: {names[channel]}\n{fraction[mode_index]:.1%} variance",
                "normalized loading",
                cmap="coolwarm",
                limits=(-limit, limit),
                show_xlabel=channel == 1,
                show_ylabel=mode_index == 0,
            )
    save_figure(figure, output, f"{family}_covariance_modes.png", created)
    return eigenvalues, fraction


def plot_covariance_spectra(spectra, output, created):
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for family, spectrum in spectra.items():
        values = spectrum["eigenvalues"]
        fraction = values / spectrum["total_variance"]
        axes[0].semilogy(np.arange(1, len(values) + 1), fraction, label=family)
        axes[1].plot(np.arange(1, len(values) + 1), np.cumsum(fraction), label=family)
    axes[0].set(title="Leading covariance spectrum", xlabel="mode", ylabel="variance fraction")
    axes[1].set(title="Cumulative captured variance", xlabel="mode", ylabel="fraction")
    axes[1].set_ylim(0, 1.01)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    save_figure(figure, output, "covariance_spectrum.png", created)


#%% Sparse reconstruction

def plot_reconstruction(products, output, created, conductance_std=None):
    prior_mean = products.get(
        "reconstruction_prior_mean", products["conductance_mean"]
    )
    posterior_mean = products["reconstruction_mean"]
    prior_std = products.get("reconstruction_prior_standard_deviation")
    if prior_std is None:
        if conductance_std is None:
            raise ValueError("saved run has no prior conductance uncertainty")
        prior_std = conductance_std
    posterior_std = products["reconstruction_standard_deviation"]

    figure, axes = plt.subplots(4, 2, figsize=(10, 16), layout="constrained")
    for channel, name in enumerate(FIELD_NAMES["conductance"]):
        combined_mean = np.r_[
            prior_mean[channel].ravel(), posterior_mean[channel].ravel()
        ]
        maximum = float(np.percentile(combined_mean, 99))
        minimum = min(0.0, float(np.min(posterior_mean[channel])))
        image_panel(
            axes[0, channel], prior_mean[channel], f"prior mean {name}", "S",
            limits=(minimum, maximum), show_xlabel=False,
            show_ylabel=channel == 0,
        )
        image_panel(
            axes[1, channel], posterior_mean[channel],
            f"posterior mean {name}", "S", limits=(minimum, maximum),
            show_xlabel=False, show_ylabel=channel == 0,
        )
        image_panel(
            axes[2, channel], prior_std[channel],
            f"prior uncertainty {name}", "S",
            limits=robust_limits(prior_std[channel]), show_xlabel=False,
            show_ylabel=channel == 0,
        )
        image_panel(
            axes[3, channel], posterior_std[channel],
            f"posterior uncertainty {name}", "S",
            limits=robust_limits(prior_std[channel]), show_ylabel=channel == 0,
        )
    save_figure(figure, output, "reconstruction_summary.png", created)

    if "reconstruction_truth" not in products:
        return False

    truth = products["reconstruction_truth"]
    observation_indices = products["observation_indices"].astype(int)
    rows, columns = truth.shape[-2:]
    field_size = rows * columns
    figure, axes = plt.subplots(3, 2, figsize=(10, 12), layout="constrained")
    for channel, name in enumerate(FIELD_NAMES["conductance"]):
        finite = np.isfinite(truth[channel])
        maximum = float(np.percentile(truth[channel][finite], 99))
        image_panel(
            axes[0, channel], truth[channel], f"truth {name}", "S",
            limits=(0, maximum), show_xlabel=False, show_ylabel=channel == 0,
        )
        prior_error = prior_mean[channel] - truth[channel]
        posterior_error = posterior_mean[channel] - truth[channel]
        image_panel(
            axes[1, channel], prior_error, f"prior error {name}", "S",
            cmap="coolwarm", limits=robust_limits(prior_error, signed=True),
            show_xlabel=False, show_ylabel=channel == 0,
        )
        image_panel(
            axes[2, channel], posterior_error,
            f"posterior error {name}", "S", cmap="coolwarm",
            limits=robust_limits(posterior_error, signed=True),
            show_ylabel=channel == 0,
        )

        selected = observation_indices[(observation_indices // field_size) == channel]
        local = selected % field_size
        for axis in axes[:, channel]:
            axis.scatter(
                local % columns,
                local // columns,
                s=12,
                facecolors="none",
                edgecolors="black",
            )
    save_figure(figure, output, "reconstruction_truth.png", created)
    return True


#%% Held-out posterior versus prior diagnostics

POSTERIOR_PRIOR_LABELS = {
    "observed": "held-out observations",
    "posterior_mean": "decoder at posterior mean",
    "posterior_sample": "posterior samples",
    "prior": "prior samples",
}


def masked_moments(samples, valid):
    """Calculate physical moments without treating missing pixels as zero."""

    samples = np.asarray(samples, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape == (samples.shape[0], 1, *samples.shape[2:]):
        valid = np.broadcast_to(valid, samples.shape)
    if samples.ndim != 4 or samples.shape[1] != 2 or valid.shape != samples.shape:
        raise ValueError("samples and valid must align as (sample, 2, y, x)")
    valid = valid & np.isfinite(samples)

    count = np.sum(valid, axis=0)
    total = np.sum(np.where(valid, samples, 0.0), axis=0)
    mean = np.full(samples.shape[1:], np.nan, dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)

    residual = np.where(valid, samples - mean, 0.0)
    variance = np.full_like(mean, np.nan)
    np.divide(
        np.sum(residual**2, axis=0), count, out=variance, where=count > 0
    )

    global_mean = []
    global_standard_deviation = []
    for channel in range(2):
        selected = samples[:, channel][valid[:, channel]]
        if selected.size == 0:
            raise ValueError(f"channel {channel} has no valid held-out values")
        global_mean.append(float(np.mean(selected)))
        global_standard_deviation.append(float(np.std(selected)))

    return {
        "mean": mean,
        "standard_deviation": np.sqrt(np.maximum(variance, 0.0)),
        "count": count,
        "global_mean": global_mean,
        "global_standard_deviation": global_standard_deviation,
        "mean_spatial_variance": [
            float(np.nanmean(variance[channel])) for channel in range(2)
        ],
        "median_spatial_variance": [
            float(np.nanmedian(variance[channel])) for channel in range(2)
        ],
    }


def coverage_adjusted_anomaly_factor(samples, valid, minimum_coverage=0.1):
    """Construct a PSD masked-covariance diagnostic on common grid support.

    Missing residuals are zero after subtracting each cell's available-sample
    mean. Each state element is divided by sqrt(valid_count - 1), preserving
    its ordinary available-sample variance on the covariance diagonal. The
    resulting cross-covariances are a coverage-adjusted diagnostic, not an
    exact pairwise-complete covariance estimate.
    """

    samples = np.asarray(samples, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if valid.shape == (samples.shape[0], 1, *samples.shape[2:]):
        valid = np.broadcast_to(valid, samples.shape)
    if samples.ndim != 4 or samples.shape[1] != 2 or valid.shape != samples.shape:
        raise ValueError("samples and valid must align as (sample, 2, y, x)")
    if not 0 <= minimum_coverage <= 1:
        raise ValueError("minimum_coverage must be between zero and one")

    flat = samples.reshape(samples.shape[0], -1)
    flat_valid = (valid & np.isfinite(samples)).reshape(samples.shape[0], -1)
    count = np.sum(flat_valid, axis=0)
    minimum_count = max(2, int(np.ceil(minimum_coverage * samples.shape[0])))
    support = count >= minimum_count
    if np.count_nonzero(support) < 2:
        raise ValueError("too few state elements meet the covariance coverage threshold")

    total = np.sum(np.where(flat_valid, flat, 0.0), axis=0)
    mean = np.zeros(flat.shape[1], dtype=np.float64)
    np.divide(total, count, out=mean, where=count > 0)
    residual = np.where(flat_valid, flat - mean, 0.0)
    residual[:, support] /= np.sqrt(count[support] - 1.0)
    return residual[:, support].T, support, count


def _load_heldout_dataset(artifact, transform, requested_split=None):
    configuration = artifact["training_configuration"]
    provenance = artifact["data_provenance"]
    split_files = artifact["split_files"]
    if requested_split is None:
        requested_split = next(
            (name for name in ("test", "validation", "train") if split_files.get(name)),
            None,
        )
    if requested_split not in split_files or not split_files[requested_split]:
        raise ValueError(f"saved run has no files for split {requested_split!r}")

    paths = [Path(path) for path in split_files[requested_split]]
    interval_path = provenance.get("manual_frame_intervals")
    intervals = (
        None if interval_path is None
        else load_good_orbit_intervals(interval_path)
    )
    saved_annotations = provenance.get("manual_annotations")
    if saved_annotations is not None:
        csv_fallback = saved_annotations.get("csv_fallback")
        annotations = load_annotations(saved_annotations["filename"], None if csv_fallback is None else csv_fallback["filename"])
        if annotations["provenance"] != saved_annotations:
            raise ValueError("annotation JSON, CSV or original frame times changed since training")
        intervals = annotation_intervals(annotations)
    minimum_sza = configuration.get("minimum_sza_degrees")
    data_version = configuration.get("comparison_data_version", "current")
    legacy_directory = provenance.get("legacy_comparison_directory")
    if artifact.get("artifact_type") == "conductance_vae_run_v1":
        from icanalyzer.conductance import load_conductance_cs_files
        kp_options = {"include_kp": True} if artifact.get("model_configuration", {}).get("kp_conditioning") is not None else {}
        product = load_conductance_cs_files(paths, wic_source_intervals=intervals, minimum_sza=minimum_sza, **kp_options)
    elif legacy_directory is None:
        if data_version != "current":
            raise ValueError("legacy held-out diagnostics require the saved legacy directory")
        product = load_precipitation_cs_files(
            paths,
            wic_source_intervals=intervals,
            minimum_sza=minimum_sza,
        )
    else:
        pairs = pair_orbit_files(paths, legacy_directory)
        product = load_paired_files(
            pairs,
            data_version=data_version,
            wic_source_intervals=intervals,
            minimum_sza=minimum_sza,
        )

    if saved_annotations is not None:
        product = apply_annotations(product, annotations)

    dataset = WICSI12ArrayDataset(
        product,
        transform,
        maximum_missing_fraction=configuration.get(
            "maximum_missing_fraction", 0.33
        ),
    )
    expected = (
        configuration.get("frame_selection", {})
        .get(requested_split, {})
        .get("used")
    )
    if expected is not None and len(dataset) != int(expected):
        raise ValueError(
            f"reloaded {requested_split} split has {len(dataset)} frames; "
            f"saved run used {expected}"
        )
    saved_kp = provenance.get("retained_frame_kp", {}).get(requested_split)
    if saved_kp is not None and not np.array_equal(dataset.kp, saved_kp):
        raise ValueError("reloaded retained-frame Kp differs from the saved run")
    dataset.frame_orbit = np.asarray(product["orbit"])[dataset.selected_indices]
    dataset.frame_time = np.asarray(product["time"])[dataset.selected_indices].astype(str)
    return dataset, requested_split


def artifact_path(run_directory):
    """Resolve the paired-field artifact without guessing when both are present."""

    candidates = [Path(run_directory) / name for name in ("wic_si12_vae.pt", "conductance_vae.pt")]
    available = [path for path in candidates if path.is_file()]
    if len(available) != 1:
        raise ValueError("run directory must contain exactly one WIC/SI12 or conductance VAE artifact")
    return available[0]


def physical_decode(values, transform, support_mask=None):
    """Exclude never-observed cells before exponentiating unconstrained outputs."""

    if support_mask is not None:
        supported = torch.as_tensor(support_mask, device=values.device)[None, None]
        values = torch.where(supported, values, 0.0)
    physical = transform.inverse(values).cpu().numpy()
    if support_mask is not None:
        physical[:, :, ~support_mask] = 0.0
    return physical


def infer_posterior_prior(model, transform, dataset, batch_size, device, seed, support_mask=None):
    """Decode deterministic posterior, posterior, and prior fields in batches."""

    if batch_size <= 0:
        raise ValueError("posterior batch size must be positive")
    device = torch.device(device)
    model = model.to(device)
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    shape = (len(dataset), 2, *model.image_shape)
    decoded = {
        name: np.empty(shape, dtype=np.float32)
        for name in ("posterior_mean", "posterior_sample", "prior")
    }
    mu_values = np.empty((len(dataset), model.latent_dim), dtype=np.float32)
    logvar_values = np.empty_like(mu_values)
    posterior_latent = np.empty_like(mu_values)
    # Draw on CPU so the saved diagnostic is identical for CPU and CUDA runs.
    posterior_generator = torch.Generator(device="cpu")
    posterior_generator.manual_seed(int(seed))
    prior_generator = torch.Generator(device="cpu")
    prior_generator.manual_seed(int(seed) + 1)

    start = 0
    with torch.no_grad():
        for batch in loader:
            encoder_input = batch["encoder_input"].to(device)
            mask = batch["mask"].to(device)
            condition_arguments = {"kp": batch["kp"].to(device)} if getattr(model, "kp_conditioning", None) is not None else {}
            mu, logvar = model.encode(encoder_input, mask, **condition_arguments)
            standard_deviation = torch.exp(0.5 * logvar)
            noise = torch.randn(
                mu.shape, generator=posterior_generator,
                dtype=mu.dtype, device="cpu"
            ).to(device)
            sampled_latent = mu + standard_deviation * noise
            prior_latent = torch.randn(
                mu.shape, generator=prior_generator,
                dtype=mu.dtype, device="cpu"
            ).to(device)
            transformed = {
                "posterior_mean": model.decode(mu, **condition_arguments),
                "posterior_sample": model.decode(sampled_latent, **condition_arguments),
                "prior": model.decode(prior_latent, **condition_arguments),
            }
            stop = start + mu.shape[0]
            for name, values in transformed.items():
                decoded[name][start:stop] = physical_decode(values, transform, support_mask)
            mu_values[start:stop] = mu.cpu().numpy()
            logvar_values[start:stop] = logvar.cpu().numpy()
            posterior_latent[start:stop] = sampled_latent.cpu().numpy()
            start = stop

    decoded["observed"] = np.asarray(dataset.values, dtype=np.float32)
    decoded["valid"] = np.broadcast_to(
        dataset.common_valid, decoded["observed"].shape
    ).copy()
    if support_mask is not None:
        decoded["valid"] &= support_mask[None, None]
    decoded["mu"] = mu_values
    decoded["logvar"] = logvar_values
    decoded["posterior_latent"] = posterior_latent
    if getattr(model, "kp_conditioning", None) is not None:
        decoded["kp"] = dataset.kp.copy()
    return decoded


def inpainting_occlusion_masks(values, valid, patch_size, policy="auroral", seed=0):
    """Remove a square at peak signal or a seeded uniformly sampled valid cell."""

    values = np.asarray(values, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if values.ndim != 4 or values.shape[1] != 2:
        raise ValueError("values must have shape (sample, 2, y, x)")
    if policy not in ("auroral", "random"):
        raise ValueError("inpainting policy must be auroral or random")
    if valid.shape == (values.shape[0], 1, *values.shape[2:]):
        common_valid = valid
    elif valid.shape == values.shape:
        common_valid = np.all(valid, axis=1, keepdims=True)
    else:
        raise ValueError("valid must align with two-channel image values")
    patch_size = int(patch_size)
    rows, columns = values.shape[-2:]
    if patch_size <= 0 or patch_size > min(rows, columns):
        raise ValueError("inpainting patch size must fit inside the image")

    # Log intensity prevents the brighter channel from dominating patch selection.
    joint_signal = (
        np.log10(np.maximum(values[:, 0], 0.0) + 1.0)
        + np.log10(np.maximum(values[:, 1], 0.0) + 1.0)
    )
    joint_signal = np.where(common_valid[:, 0], joint_signal, -np.inf)
    input_valid = common_valid.copy()
    generator = np.random.default_rng(seed) if policy == "random" else None
    bounds = []
    removed_signal = []
    for frame in range(values.shape[0]):
        if not np.any(np.isfinite(joint_signal[frame])):
            raise ValueError(f"held-out frame {frame} has no valid auroral pixels")
        if policy == "random":
            candidates = np.argwhere(common_valid[frame, 0])
            center = candidates[generator.integers(len(candidates))]
        else:
            center = np.unravel_index(np.argmax(joint_signal[frame]), joint_signal[frame].shape)
        row_start = int(np.clip(center[0] - patch_size // 2, 0, rows - patch_size))
        column_start = int(np.clip(
            center[1] - patch_size // 2, 0, columns - patch_size
        ))
        row_stop = row_start + patch_size
        column_stop = column_start + patch_size
        patch_valid = common_valid[
            frame, 0, row_start:row_stop, column_start:column_stop
        ]
        patch_signal = joint_signal[
            frame, row_start:row_stop, column_start:column_stop
        ][patch_valid]
        removed_signal.append(float(np.mean(patch_signal)))
        input_valid[frame, 0, row_start:row_stop, column_start:column_stop] = False
        if policy == "random" and not np.any(input_valid[frame]):
            raise ValueError(f"random patch removes all observed input in frame {frame}; use a smaller patch")
        bounds.append((row_start, row_stop, column_start, column_stop))
    return input_valid, np.asarray(bounds, dtype=int), np.asarray(removed_signal)


def auroral_occlusion_masks(values, valid, patch_size):
    """Retain the original peak-signal diagnostic interface."""

    return inpainting_occlusion_masks(values, valid, patch_size)


def infer_heldout_inpainting(
    model, transform, dataset, batch_size, device, patch_size, policy="auroral", seed=0, support_mask=None
):
    """Reconstruct held-out frames with identical seeded or peak-signal patches."""

    input_valid, bounds, removed_signal = inpainting_occlusion_masks(
        dataset.values, dataset.common_valid, patch_size, policy=policy, seed=seed
    )
    encoder_input = np.concatenate((dataset.target, dataset.weight), axis=1).copy()
    encoder_input = np.where(input_valid, encoder_input, 0.0).astype(np.float32)
    reconstructed = np.empty_like(dataset.values, dtype=np.float32)

    device = torch.device(device)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            current_input = torch.from_numpy(encoder_input[start:stop]).to(device)
            current_mask = torch.from_numpy(input_valid[start:stop]).to(device)
            condition_arguments = {"kp": torch.as_tensor(dataset.kp[start:stop], device=device)} if getattr(model, "kp_conditioning", None) is not None else {}
            mu, _logvar = model.encode(current_input, current_mask, **condition_arguments)
            decoded = model.decode(mu, **condition_arguments)
            reconstructed[start:stop] = physical_decode(decoded, transform, support_mask)

    original_valid = np.broadcast_to(dataset.common_valid, dataset.values.shape)
    input_valid_two_channel = np.broadcast_to(input_valid, dataset.values.shape)
    if support_mask is not None:
        original_valid = original_valid & support_mask[None, None]
        input_valid_two_channel = input_valid_two_channel & support_mask[None, None]
    return {
        "reconstruction": reconstructed,
        "input_valid": input_valid_two_channel.copy(),
        "removed": (original_valid & ~input_valid_two_channel).copy(),
        "bounds": bounds,
        "removed_signal": removed_signal,
    }


def inpainting_skill(ensembles, inpainting, transform, family="fuv"):
    observed = np.asarray(ensembles["observed"])
    full_reconstruction = np.asarray(ensembles["posterior_mean"])
    inpainted = np.asarray(inpainting["reconstruction"])
    removed = np.asarray(inpainting["removed"], dtype=bool)
    retained = np.asarray(inpainting["input_valid"], dtype=bool)

    observed_transformed = transform.transform(observed)
    full_error = transform.transform(full_reconstruction) - observed_transformed
    inpainted_error = transform.transform(inpainted) - observed_transformed
    unit = "counts" if family == "fuv" else "siemens"
    return {
        "full_input_removed_region_normalized_log10_rmse": _weighted_channel_rmse(
            full_error, removed.astype(np.float64)
        ),
        "inpainted_removed_region_normalized_log10_rmse": _weighted_channel_rmse(
            inpainted_error, removed.astype(np.float64)
        ),
        "inpainted_retained_region_normalized_log10_rmse": _weighted_channel_rmse(
            inpainted_error, retained.astype(np.float64)
        ),
        f"inpainted_removed_region_physical_rmse_{unit}": _weighted_channel_rmse(
            inpainted - observed, removed.astype(np.float64)
        ),
        "removed_cell_count": [
            int(np.count_nonzero(removed[:, channel])) for channel in range(2)
        ],
        "per_frame_removed_cell_count": removed.sum(axis=(2, 3)).tolist(),
        f"per_frame_removed_squared_error_{unit}": np.where(
            removed, (inpainted - observed)**2, 0.0
        ).sum(axis=(2, 3), dtype=np.float64).tolist(),
        "per_frame_removed_squared_error_normalized_log10": np.where(
            removed, inpainted_error**2, 0.0
        ).sum(axis=(2, 3), dtype=np.float64).tolist(),
    }


def plot_heldout_inpainting(
    ensembles, inpainting, output, created, n_examples, family="fuv"
):
    n_examples = min(int(n_examples), ensembles["observed"].shape[0])
    if n_examples < 1:
        raise ValueError("inpainting examples must be positive")
    order = np.argsort(inpainting["removed_signal"])
    positions = np.linspace(0.5 * (len(order) - 1), len(order) - 1, n_examples)
    indices = order[np.asarray(np.round(positions), dtype=int)]
    columns = (
        ("observed", "held-out truth"),
        ("damaged", "input after removal"),
        ("posterior_mean", "reconstruction with full input"),
        ("inpainted", "reconstruction from damaged input"),
        ("error", "damaged-input reconstruction error"),
    )
    figure, axes = plt.subplots(
        2 * n_examples, len(columns),
        figsize=(3.45 * len(columns), 3.05 * 2 * n_examples),
        squeeze=False, layout="constrained"
    )
    for example_number, frame in enumerate(indices):
        row_start, row_stop, column_start, column_stop = inpainting["bounds"][frame]
        for channel in range(2):
            row = 2 * example_number + channel
            truth = np.where(
                ensembles["valid"][frame, channel],
                ensembles["observed"][frame, channel], np.nan
            )
            damaged = np.where(
                inpainting["input_valid"][frame, channel],
                ensembles["observed"][frame, channel], np.nan
            )
            full_reconstruction = np.where(
                ensembles["valid"][frame, channel],
                ensembles["posterior_mean"][frame, channel], np.nan
            )
            reconstructed = np.where(
                ensembles["valid"][frame, channel],
                inpainting["reconstruction"][frame, channel], np.nan
            )
            error = reconstructed - truth
            fields = (truth, damaged, full_reconstruction, reconstructed, error)
            limits = robust_limits(np.stack(fields[:4]))
            error_limits = robust_limits(error, signed=True)
            for column, ((_key, title), field) in enumerate(zip(columns, fields)):
                image_panel(
                    axes[row, column], field,
                    f"{title}\nframe {frame}, {FIELD_NAMES[family][channel]}",
                    FIELD_UNITS[family][channel],
                    cmap="coolwarm" if column == 4 else "viridis",
                    limits=error_limits if column == 4 else limits,
                    show_xlabel=row == 2 * n_examples - 1,
                    show_ylabel=column == 0,
                )
                axes[row, column].add_patch(plt.Rectangle(
                    (column_start - 0.5, row_start - 0.5),
                    column_stop - column_start, row_stop - row_start,
                    fill=False, edgecolor="red", linewidth=1.4
                ))
    save_figure(figure, output, "heldout_inpainting_examples.png", created)
    return indices.tolist()


def plot_posterior_prior_examples(ensembles, output, created, n_examples, family="fuv"):
    n_examples = min(int(n_examples), ensembles["observed"].shape[0])
    if n_examples < 1:
        raise ValueError("posterior examples must be positive")
    indices = np.linspace(
        0, ensembles["observed"].shape[0] - 1, n_examples, dtype=int
    )
    names = ("observed", "posterior_mean", "posterior_sample")
    figure, axes = plt.subplots(
        2 * n_examples, len(names),
        figsize=(3.7 * len(names), 3.1 * 2 * n_examples),
        squeeze=False, layout="constrained"
    )
    for example_number, sample_index in enumerate(indices):
        for channel in range(2):
            row = 2 * example_number + channel
            selected = [
                np.where(
                    ensembles["valid"][sample_index, channel],
                    ensembles[name][sample_index, channel], np.nan
                )
                for name in names
            ]
            limits = robust_limits(np.stack(selected))
            for column, name in enumerate(names):
                image_panel(
                    axes[row, column], selected[column],
                    f"{POSTERIOR_PRIOR_LABELS[name]}\n"
                    f"frame {sample_index}, {FIELD_NAMES[family][channel]}",
                    FIELD_UNITS[family][channel], limits=limits,
                    show_xlabel=row == 2 * n_examples - 1,
                    show_ylabel=column == 0,
                )
    save_figure(figure, output, "posterior_prior_examples.png", created)
    return indices.tolist()


def plot_prior_sample_gallery(ensembles, output, created, n_examples, family="fuv", support_mask=None):
    n_examples = min(int(n_examples), ensembles["prior"].shape[0])
    indices = np.linspace(
        0, ensembles["prior"].shape[0] - 1, n_examples, dtype=int
    )
    figure, axes = plt.subplots(
        2, n_examples, figsize=(3.2 * n_examples, 6.2),
        squeeze=False, layout="constrained"
    )
    for channel in range(2):
        fields = ensembles["prior"][:, channel]
        if support_mask is not None:
            fields = np.where(support_mask, fields, np.nan)
        limits = robust_limits(fields)
        for column, sample_index in enumerate(indices):
            image_panel(
                axes[channel, column],
                fields[sample_index],
                    (f"Kp {ensembles['kp'][sample_index]:g} prior draw {sample_index}\n" if "kp" in ensembles else f"unconditional prior draw {sample_index}\n")
                    + f"{FIELD_NAMES[family][channel]}",
                FIELD_UNITS[family][channel], limits=limits,
                show_xlabel=channel == 1, show_ylabel=column == 0,
            )
    save_figure(figure, output, "prior_samples_gallery.png", created)
    return indices.tolist()


def plot_posterior_prior_moments(moment_sets, output, created, family="fuv"):
    names = tuple(POSTERIOR_PRIOR_LABELS)
    figure, axes = plt.subplots(
        len(names), 4, figsize=(16, 3.5 * len(names)), layout="constrained"
    )
    columns = [(channel, statistic, f"{FIELD_NAMES[family][channel]} {statistic.replace('_', ' ')}", FIELD_UNITS[family][channel]) for channel in range(2) for statistic in ("mean", "standard_deviation")]
    limits = []
    for channel, statistic, _title, _unit in columns:
        combined = np.stack([moment_sets[name][statistic][channel] for name in names])
        limits.append(robust_limits(combined))
    for row, name in enumerate(names):
        for column, (channel, statistic, title, unit) in enumerate(columns):
            image_panel(
                axes[row, column], moment_sets[name][statistic][channel],
                f"{POSTERIOR_PRIOR_LABELS[name]}\n{title}", unit,
                limits=limits[column], show_xlabel=row == len(names) - 1,
                show_ylabel=column == 0,
            )
    save_figure(figure, output, "posterior_prior_moments.png", created)


def plot_posterior_prior_distributions(ensembles, moment_sets, output, created, family="fuv"):
    names = tuple(POSTERIOR_PRIOR_LABELS)
    colors = ("black", "tab:blue", "tab:orange", "tab:green")
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for channel in range(2):
        transformed = {}
        for name in names:
            values = ensembles[name][:, channel][ensembles["valid"][:, channel]]
            transformed[name] = np.log10(np.maximum(values, 0.0) + 1.0)
        combined = np.concatenate(list(transformed.values()))
        lower, upper = np.percentile(combined, [0.25, 99.75])
        bins = np.linspace(lower, upper, 80)
        for name, color in zip(names, colors):
            axes[channel, 0].hist(
                transformed[name], bins=bins, density=True, histtype="step",
                linewidth=1.5, color=color, label=POSTERIOR_PRIOR_LABELS[name]
            )
        axes[channel, 0].set(
            title=f"{FIELD_NAMES[family][channel]} held-out-mask distribution",
            xlabel=f"log10({FIELD_UNITS[family][channel]} + 1)", ylabel="density"
        )
        axes[channel, 0].legend(fontsize=7)

        positions = np.arange(2)
        width = 0.18
        for offset, (name, color) in enumerate(zip(names, colors)):
            axes[channel, 1].bar(
                positions + (offset - 1.5) * width,
                [
                    moment_sets[name]["global_mean"][channel],
                    moment_sets[name]["global_standard_deviation"][channel],
                ],
                width=width, color=color, label=POSTERIOR_PRIOR_LABELS[name]
            )
        axes[channel, 1].set(
            title=f"{FIELD_NAMES[family][channel]} physical moments",
            ylabel=FIELD_UNITS[family][channel], xticks=positions, xticklabels=("mean", "std")
        )
        axes[channel, 1].legend(fontsize=7)
    for axis in axes.ravel():
        axis.grid(alpha=0.2)
    save_figure(figure, output, "posterior_prior_distributions.png", created)


def _weighted_channel_rmse(error, weights):
    error = np.asarray(error, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if error.shape != weights.shape or error.ndim != 4 or error.shape[1] != 2:
        raise ValueError("error and weights must align as (sample, 2, y, x)")
    values = []
    for channel in range(2):
        finite = np.isfinite(error[:, channel]) & np.isfinite(weights[:, channel])
        weight = np.where(finite, np.maximum(weights[:, channel], 0.0), 0.0)
        denominator = np.sum(weight)
        if denominator <= 0:
            raise ValueError(f"channel {channel} has no positive reconstruction weight")
        squared_error = np.where(finite, error[:, channel]**2, 0.0)
        values.append(float(np.sqrt(np.sum(weight * squared_error) / denominator)))
    return values


def posterior_reconstruction_skill(
    ensembles, dataset, transform, loss_configuration
):
    """Compare posterior error under neutral, quality, and training weights."""

    valid = np.asarray(ensembles["valid"], dtype=bool)
    quality = np.broadcast_to(dataset.weight, valid.shape).astype(np.float64)
    schemes = {
        "all_valid": valid.astype(np.float64),
        "auroral_quality": np.where(valid, quality, 0.0),
    }
    training_weight = schemes["auroral_quality"].copy()
    coverage = loss_configuration.get("coverage_weighting")
    if coverage is not None:
        from icanalyzer.wic_si12.coverage_weighting import coverage_bin_indices, validate_coverage_weighting
        validate_coverage_weighting(coverage)
        indices = coverage_bin_indices(dataset.kp, coverage["kp_bin_edges"])
        factors = np.asarray(coverage["factors"], dtype=float)
        if factors.shape[1:] != valid.shape[-2:]:
            raise ValueError("coverage weights and held-out observations use different grids")
        training_weight *= factors[indices, None]
    count_mode = loss_configuration.get("kind") == "quality_weighted_censored_counts_v1"
    if count_mode:
        scale = np.asarray(loss_configuration["channel_scale_counts"]).reshape(1, 2, 1, 1)
        tau = np.asarray(loss_configuration["model_error_fraction"]).reshape(1, 2, 1, 1)
        variance = np.square(np.asarray(dataset.standard_deviation, dtype=np.float64) / scale)
        finite = np.isfinite(variance) & (variance >= 0)
        attenuation = np.where(finite, tau**2 / (variance + tau**2), 0.0)
        training_weight *= attenuation
    elif loss_configuration.get("kind") == "quality_weighted_gaussian_v1":
        variance = np.asarray(dataset.transformed_variance, dtype=np.float64)
        model_variance = np.square(loss_configuration["model_error_std"]).reshape(1, 2, 1, 1)
        finite = np.isfinite(variance) & (variance >= 0)
        attenuation = np.where(finite, model_variance / (variance + model_variance), 0.0)
        training_weight *= attenuation
    elif loss_configuration.get("measurement_uncertainty_in_loss", False):
        variance = np.asarray(dataset.transformed_variance, dtype=np.float64)
        finite = np.isfinite(variance) & (variance >= 0)
        precision = np.where(
            finite,
            1.0 / np.maximum(variance, loss_configuration["variance_floor"]),
            0.0,
        )
        scale = np.asarray(
            loss_configuration["precision_scale"], dtype=np.float64
        ).reshape(1, 2, 1, 1)
        relative_precision = np.minimum(
            precision / scale,
            loss_configuration["maximum_relative_precision"],
        )
        training_weight *= relative_precision
    schemes["training_effective"] = training_weight

    observed = transform.transform(ensembles["observed"])
    results = {}
    for name in ("posterior_mean", "posterior_sample"):
        error = transform.transform(ensembles[name]) - observed
        results[name] = {
            scheme: _weighted_channel_rmse(error, weight)
            for scheme, weight in schemes.items()
        }
    results["weight_sums"] = {
        scheme: [float(np.sum(weight[:, channel])) for channel in range(2)]
        for scheme, weight in schemes.items()
    }
    if count_mode:
        results["training_effective_note"] = (
            "Count-variance attenuation applied to normalized-log RMSE for comparison only; "
            "not the count-space training objective. Censored zeros are scored by log-CDF during training."
        )
    return results


def plot_posterior_reconstruction_skill(skill, output, created, family="fuv"):
    schemes = ("all_valid", "auroral_quality", "training_effective")
    labels = ("all valid", "auroral quality", "training effective")
    if "training_effective_note" in skill:
        labels = ("all valid", "auroral quality", "count attenuation\n(diagnostic only)")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    positions = np.arange(len(schemes))
    width = 0.36
    for channel in range(2):
        for offset, (name, label, color) in enumerate((
            ("posterior_mean", "decoder at posterior mean", "tab:blue"),
            ("posterior_sample", "posterior samples", "tab:orange"),
        )):
            axes[channel].bar(
                positions + (offset - 0.5) * width,
                [skill[name][scheme][channel] for scheme in schemes],
                width=width, label=label, color=color
            )
        axes[channel].set(
            title=f"{FIELD_NAMES[family][channel]} held-out reconstruction",
            ylabel="normalized log10 RMSE",
            xticks=positions, xticklabels=labels
        )
        axes[channel].tick_params(axis="x", rotation=15)
        axes[channel].grid(axis="y", alpha=0.25)
        axes[channel].legend(fontsize=8)
    save_figure(
        figure, output, "posterior_reconstruction_rmse.png", created
    )


def reported_measurement_noise(dataset, family="fuv", valid=None):
    unit = "counts" if family == "fuv" else "siemens"
    if valid is None:
        valid = np.broadcast_to(dataset.common_valid, dataset.values.shape)
    summary = []
    for channel in range(2):
        values = dataset.values[:, channel][valid[:, channel]].astype(np.float64)
        uncertainty = dataset.standard_deviation[:, channel][
            valid[:, channel]
        ].astype(np.float64)
        finite = np.isfinite(values) & np.isfinite(uncertainty) & (uncertainty >= 0)
        values = values[finite]
        uncertainty = uncertainty[finite]
        observed_variance = float(np.var(values))
        mean_noise_variance = float(np.mean(uncertainty**2))
        summary.append({
            f"observed_variance_{unit}_squared": observed_variance,
            f"mean_reported_noise_variance_{unit}_squared": mean_noise_variance,
            "reported_noise_fraction_of_observed_variance": (
                mean_noise_variance / observed_variance
                if observed_variance > 0 else None
            ),
            f"simple_noise_corrected_standard_deviation_{unit}": float(np.sqrt(
                max(observed_variance - mean_noise_variance, 0.0)
            )),
        })
    return summary


def plot_posterior_prior_covariance(
    ensembles, transform, minimum_coverage, spectrum_modes,
    decomposition_seed, output, created
):
    names = tuple(POSTERIOR_PRIOR_LABELS)
    spectra = {}
    support = None
    support_count = None
    for name in names:
        transformed = transform.transform(ensembles[name])
        factor, current_support, current_count = coverage_adjusted_anomaly_factor(
            transformed, ensembles["valid"], minimum_coverage=minimum_coverage
        )
        if support is None:
            support = current_support
            support_count = current_count
        elif not np.array_equal(current_support, support):
            raise ValueError("finite covariance support differs between ensembles")
        decomposition = covariance_mode_decomposition(
            factor,
            (1, 1, factor.shape[0]),
            n_modes=spectrum_modes,
            random_seed=decomposition_seed,
            exact_max_dimension=512,
        )
        spectra[name] = decomposition

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    summaries = {}
    rank_limit = min(
        int(np.count_nonzero(support)), ensembles["observed"].shape[0] - 1
    )
    for name in names:
        values = spectra[name]["eigenvalues"]
        total_variance = spectra[name]["total_variance"]
        fraction = values / total_variance
        shown = values.size
        axes[0].semilogy(
            np.arange(1, shown + 1), fraction[:shown],
            label=POSTERIOR_PRIOR_LABELS[name]
        )
        axes[1].plot(
            np.arange(1, shown + 1), np.cumsum(fraction)[:shown],
            label=POSTERIOR_PRIOR_LABELS[name]
        )
        cumulative = np.cumsum(fraction)
        summaries[name] = {
            "rank_limit": int(rank_limit),
            "computed_mode_count": int(values.size),
            "first_mode_variance_fraction": float(fraction[0]),
            "first_8_variance_fraction": float(np.sum(fraction[:8])),
            "computed_modes_captured_variance_fraction": float(np.sum(fraction)),
            "modes_for_90_percent": (
                int(np.searchsorted(cumulative, 0.9) + 1)
                if cumulative[-1] >= 0.9 else None
            ),
            "decomposition_method": spectra[name]["method"],
        }
    axes[0].set(
        title="Mask-matched normalized-log covariance spectrum",
        xlabel="mode", ylabel="variance fraction"
    )
    axes[1].set(
        title="Cumulative normalized-log variance",
        xlabel="mode", ylabel="fraction", ylim=(0, 1.01)
    )
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    save_figure(
        figure, output, "posterior_prior_covariance_spectrum.png", created
    )
    summaries["support"] = {
        "minimum_frame_fraction": float(minimum_coverage),
        "minimum_frame_count": int(max(2, np.ceil(
            minimum_coverage * ensembles["observed"].shape[0]
        ))),
        "supported_state_elements": int(np.count_nonzero(support)),
        "possible_state_elements": int(support.size),
        "minimum_observed_count_on_support": int(np.min(support_count[support])),
        "method": "mean-centered missing residuals set to zero; each state element scaled by sqrt(valid_count - 1)",
    }
    return summaries


def plot_posterior_latent_activity(ensembles, output, created):
    mu = np.asarray(ensembles["mu"], dtype=np.float64)
    logvar = np.asarray(ensembles["logvar"], dtype=np.float64)
    posterior_latent = np.asarray(
        ensembles["posterior_latent"], dtype=np.float64
    )
    kl_by_dimension = np.mean(
        0.5 * (mu**2 + np.exp(logvar) - 1.0 - logvar), axis=0
    )
    dimensions = np.arange(1, mu.shape[1] + 1)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    axes[0].bar(dimensions, kl_by_dimension)
    axes[0].axhline(0.1, color="tab:red", linestyle="--", label="0.1 nat")
    axes[0].set(
        title="Held-out KL by latent dimension", xlabel="latent dimension",
        ylabel="nats per frame"
    )
    axes[0].legend()
    axes[1].plot(dimensions, np.std(mu, axis=0), "o-", ms=3, label="std of posterior mean")
    axes[1].plot(
        dimensions, np.std(posterior_latent, axis=0), "o-", ms=3,
        label="std of posterior samples"
    )
    axes[1].axhline(1.0, color="black", linestyle="--", label="unit prior std")
    axes[1].set(
        title="Aggregate posterior spread", xlabel="latent dimension",
        ylabel="standard deviation"
    )
    axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(alpha=0.25)
    save_figure(figure, output, "posterior_latent_activity.png", created)
    return {
        "total_mean_kl_nats_per_frame": float(np.sum(kl_by_dimension)),
        "active_dimensions_above_0_01_nat": int(np.count_nonzero(kl_by_dimension > 0.01)),
        "active_dimensions_above_0_1_nat": int(np.count_nonzero(kl_by_dimension > 0.1)),
        "active_dimensions_above_0_5_nat": int(np.count_nonzero(kl_by_dimension > 0.5)),
        "kl_by_dimension": kl_by_dimension.tolist(),
        "posterior_mean_standard_deviation_by_dimension": np.std(mu, axis=0).tolist(),
        "posterior_sample_standard_deviation_by_dimension": np.std(
            posterior_latent, axis=0
        ).tolist(),
    }


def run_posterior_prior_diagnostics(arguments, artifact, output, created):
    native = artifact.get("artifact_type") == "conductance_vae_run_v1"
    family = "conductance" if native else "fuv"
    unit = "siemens" if native else "counts"
    support_mask = np.asarray(artifact["data_provenance"]["training_support"], dtype=bool) if native else None
    device = arguments.posterior_device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model, transform, loaded_artifact = load_run_artifact(
        artifact_path(arguments.run_directory), device=device
    )
    dataset, split_name = _load_heldout_dataset(
        loaded_artifact, transform, requested_split=arguments.posterior_split
    )
    seed = int(
        artifact.get("training_configuration", {}).get("seed", 0)
        + arguments.posterior_seed_offset
    )
    ensembles = infer_posterior_prior(
        model, transform, dataset, arguments.posterior_batch_size, device, seed, support_mask=support_mask
    )
    inpainting = infer_heldout_inpainting(
        model, transform, dataset, arguments.posterior_batch_size, device,
        arguments.inpainting_patch_size, policy=arguments.inpainting_policy,
        seed=arguments.inpainting_seed, support_mask=support_mask
    )
    inpainting_metrics = inpainting_skill(ensembles, inpainting, transform, family=family)
    inpainting_examples = plot_heldout_inpainting(
        ensembles, inpainting, output, created, arguments.inpainting_examples, family=family
    )
    moments = {
        name: masked_moments(ensembles[name], ensembles["valid"])
        for name in POSTERIOR_PRIOR_LABELS
    }
    example_indices = plot_posterior_prior_examples(
        ensembles, output, created, arguments.posterior_examples, family=family
    )
    prior_sample_indices = plot_prior_sample_gallery(
        ensembles, output, created, max(arguments.posterior_examples, 3), family=family, support_mask=support_mask
    )
    plot_posterior_prior_moments(moments, output, created, family=family)
    plot_posterior_prior_distributions(ensembles, moments, output, created, family=family)
    loss_configuration = artifact["training_configuration"].get(
        "loss_configuration", {}
    )
    reconstruction_skill = posterior_reconstruction_skill(
        ensembles, dataset, transform, loss_configuration
    )
    plot_posterior_reconstruction_skill(
        reconstruction_skill, output, created, family=family
    )
    covariance = plot_posterior_prior_covariance(
        ensembles, transform, arguments.posterior_minimum_coverage,
        arguments.spectrum_modes, arguments.decomposition_seed, output, created
    )
    latent = plot_posterior_latent_activity(ensembles, output, created)
    summary = {
        "field_names": list(FIELD_NAMES[family]),
        "physical_units": list(FIELD_UNITS[family]),
        "unsupported_training_cells_excluded": int((~support_mask).sum()) if native else 0,
        "split": split_name,
        "frame_count": int(len(dataset)),
        "device": str(device),
        "seed": seed,
        "example_indices": example_indices,
        "prior_sample_indices": prior_sample_indices,
        "inpainting": {
            "selection_policy": (
                "fixed square centred on a uniformly sampled originally valid cell; edge-clamped"
                if arguments.inpainting_policy == "random" else
                f"fixed square centred on maximum valid log10({FIELD_NAMES[family][0]} + 1) + log10({FIELD_NAMES[family][1]} + 1)"
            ),
            "policy": arguments.inpainting_policy,
            "patch_seed": arguments.inpainting_seed if arguments.inpainting_policy == "random" else None,
            "patch_size": int(arguments.inpainting_patch_size),
            "bounds": inpainting["bounds"].tolist(),
            "frame_orbit": dataset.frame_orbit.tolist(),
            "frame_time": dataset.frame_time.tolist(),
            "removed_mask_sha256": hashlib.sha256(np.packbits(inpainting["removed"]).tobytes()).hexdigest(),
            "example_indices": inpainting_examples,
            "metrics": inpainting_metrics,
        },
        "physical_moments_on_identical_held_out_mask": {
            name: {
                f"mean_{unit}": moments[name]["global_mean"],
                f"standard_deviation_{unit}": moments[name][
                    "global_standard_deviation"
                ],
                f"mean_spatial_variance_{unit}_squared": moments[name][
                    "mean_spatial_variance"
                ],
                f"median_spatial_variance_{unit}_squared": moments[name][
                    "median_spatial_variance"
                ],
            }
            for name in POSTERIOR_PRIOR_LABELS
        },
        "normalized_log10_reconstruction_rmse": reconstruction_skill,
        "reported_measurement_noise": reported_measurement_noise(dataset, family=family, valid=ensembles["valid"]),
        "normalized_log10_covariance": covariance,
        "latent": latent,
        "interpretation_notes": [
            "posterior_mean is decoder(mu) and contains no posterior draw noise",
            "posterior_sample uses one seeded q(z|x) draw per held-out frame",
            "prior uses one seeded N(0,I) draw per held-out frame",
            "all physical summaries use the identical held-out observation mask",
            "masked covariance is a PSD coverage-adjusted diagnostic, not exact pairwise-complete covariance",
        ],
    }
    if getattr(model, "kp_conditioning", None) is not None:
        summary["kp_conditioning"] = dict(model.kp_conditioning)
        summary["frame_kp"] = dataset.kp.tolist()
        summary["prior_kind"] = "prior conditioned on each held-out frame's Kp; moments pool that activity mixture"
    (output / "posterior_prior_diagnostics.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


#%% Main workflow

def parse_anchor(value):
    if value is None:
        return None
    row, column = value.split(",")
    return int(row), int(column)


def run(arguments):
    if arguments.examples < 2:
        raise ValueError("examples must be at least 2")
    if arguments.modes < 1:
        raise ValueError("modes must be positive")
    if arguments.spectrum_modes < 1:
        raise ValueError("spectrum_modes must be positive")
    if arguments.posterior_examples < 1:
        raise ValueError("posterior_examples must be positive")
    if arguments.inpainting_examples < 1:
        raise ValueError("inpainting_examples must be positive")
    if arguments.inpainting_patch_size < 1:
        raise ValueError("inpainting_patch_size must be positive")
    if arguments.inpainting_seed < 0:
        raise ValueError("inpainting_seed must be nonnegative")
    if not 0 <= arguments.posterior_minimum_coverage <= 1:
        raise ValueError("posterior_minimum_coverage must be between zero and one")

    run_directory = Path(arguments.run_directory)
    output = Path(arguments.output) if arguments.output else run_directory / "figures"
    output.mkdir(parents=True, exist_ok=True)

    artifact = torch.load(
        artifact_path(run_directory), map_location="cpu", weights_only=False
    )
    if arguments.posterior_prior_only:
        created = []
        posterior_prior = run_posterior_prior_diagnostics(
            arguments, artifact, output, created
        )
        diagnostics = {
            "run_directory": str(run_directory.resolve()),
            "posterior_prior": posterior_prior,
            "created_figures": created,
            "saved_prior_anomaly_factors_loaded": False,
        }
        (output / "posterior_prior_run.json").write_text(
            json.dumps(diagnostics, indent=2) + "\n"
        )
        return diagnostics

    summary = json.loads((run_directory / "summary.json").read_text())
    ensemble_path = run_directory / "ensemble_products.npz"
    with np.load(ensemble_path) as stored:
        products = {
            name: np.asarray(stored[name])
            for name in stored.files
            if not name.endswith("_anomaly_factor")
        }

    native = artifact.get("artifact_type") == "conductance_vae_run_v1"
    families = ("conductance",) if native else ("fuv", "conductance")
    state_shape = products[f"{families[0]}_mean"].shape
    requested_anchor = parse_anchor(arguments.anchor)
    sample_count = int(summary["sample_count"])
    sample_indices = np.linspace(
        0, sample_count - 1, arguments.examples, dtype=int
    )

    created = []
    best_epoch, best_metrics = plot_loss_history(
        artifact["history"], output, created
    )
    latent_samples_available = plot_latent_samples(products, output, created)

    anchors = {}
    selected_sample_indices = {}
    moments = {}
    spectra = {}
    mode_diagnostics = {}
    for family in families:
        factor_name = f"{family}_anomaly_factor"
        with np.load(ensemble_path) as stored:
            factor = np.asarray(stored[factor_name])
        if factor.shape != (int(np.prod(state_shape)), sample_count):
            raise ValueError(
                f"{factor_name} shape does not match saved state and sample count"
            )

        mean = products[f"{family}_mean"]
        anchors[family] = requested_anchor or maximum_variance_anchor(
            factor, state_shape, channel=0
        )
        samples = reconstruct_ensemble(
            mean, factor, sample_indices=sample_indices
        )
        support_mask = products["conductance_supported"] if native else None
        if support_mask is not None:
            samples[:, :, ~support_mask] = np.nan
            mean = np.where(support_mask, mean, np.nan)
        selected_sample_indices[family] = plot_sample_mosaic(
            samples, sample_indices, family, output, created
        )
        moments[family] = {
            "mean": mean,
            "standard_deviation": np.sqrt(
                anomaly_variance(factor).reshape(state_shape)
            ),
        }
        if support_mask is not None:
            moments[family]["standard_deviation"][:, ~support_mask] = np.nan
        plot_correlation_maps(
            factor, state_shape, family, anchors[family], output, created
        )
        decomposition = covariance_mode_decomposition(
            factor,
            state_shape,
            n_modes=max(arguments.modes, arguments.spectrum_modes),
            random_seed=arguments.decomposition_seed,
        )
        displayed_mode_count = min(
            arguments.modes, len(decomposition["eigenvalues"])
        )
        eigenvalues, fraction = plot_covariance_modes(
            decomposition["eigenvalues"],
            decomposition["modes"],
            decomposition["total_variance"],
            family,
            output,
            created,
            displayed_mode_count,
        )
        spectra[family] = {
            "eigenvalues": decomposition["eigenvalues"],
            "total_variance": decomposition["total_variance"],
        }
        mode_diagnostics[family] = {
            "leading_eigenvalues": eigenvalues.tolist(),
            "leading_variance_fractions": fraction.tolist(),
            "spectrum_mode_count": int(len(decomposition["eigenvalues"])),
            "spectrum_captured_variance_fraction": decomposition[
                "captured_variance_fraction"
            ],
            "decomposition_method": decomposition["method"],
        }
        del samples, factor, decomposition
        gc.collect()

    plot_ensemble_moments(moments, output, created)
    plot_covariance_spectra(spectra, output, created)
    reconstruction_has_truth = plot_reconstruction(
        products,
        output,
        created,
        conductance_std=moments["conductance"]["standard_deviation"],
    )
    posterior_prior = None
    if arguments.posterior_prior:
        posterior_prior = run_posterior_prior_diagnostics(
            arguments, artifact, output, created
        )

    diagnostics = {
        "run_directory": str(run_directory.resolve()),
        "sample_count": sample_count,
        "best_validation_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "anchors": {name: list(anchor) for name, anchor in anchors.items()},
        "sample_indices": selected_sample_indices,
        "latent_samples_available": latent_samples_available,
        "modes": mode_diagnostics,
        "reconstruction_truth_available": reconstruction_has_truth,
        "posterior_prior": posterior_prior,
        "created_figures": created,
        "dense_covariance_materialized": False,
    }
    (output / "diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2) + "\n"
    )
    return diagnostics


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory")
    parser.add_argument("--output")
    parser.add_argument("--examples", type=int, default=6)
    parser.add_argument("--modes", type=int, default=4)
    parser.add_argument(
        "--spectrum-modes",
        type=int,
        default=64,
        help="number of leading modes retained in the scalable spectrum plot",
    )
    parser.add_argument("--decomposition-seed", type=int, default=0)
    parser.add_argument(
        "--posterior-prior",
        action="store_true",
        help=(
            "reload the held-out split and compare observations, posterior "
            "reconstructions, posterior samples, and prior samples"
        ),
    )
    parser.add_argument(
        "--posterior-prior-only",
        action="store_true",
        help=(
            "generate only held-out posterior/prior diagnostics without loading "
            "the saved prior anomaly factors"
        ),
    )
    parser.add_argument(
        "--posterior-split", choices=("test", "validation", "train")
    )
    parser.add_argument("--posterior-batch-size", type=int, default=32)
    parser.add_argument("--posterior-device", default="auto")
    parser.add_argument("--posterior-seed-offset", type=int, default=1000)
    parser.add_argument("--posterior-examples", type=int, default=2)
    parser.add_argument("--inpainting-examples", type=int, default=3)
    parser.add_argument(
        "--inpainting-policy", choices=("auroral", "random"), default="auroral",
        help="remove peak-signal patches (existing default) or uniformly sampled valid-cell patches"
    )
    parser.add_argument(
        "--inpainting-seed", type=int, default=0,
        help="independent random-patch seed; use the same seed and split for paired model comparisons"
    )
    parser.add_argument(
        "--inpainting-patch-size",
        type=int,
        default=12,
        help="side length of the square removed around peak joint auroral signal",
    )
    parser.add_argument(
        "--posterior-minimum-coverage",
        type=float,
        default=0.1,
        help="minimum held-out frame fraction for covariance state elements",
    )
    parser.add_argument(
        "--anchor",
        help="optional common CS anchor as ROW,COLUMN; default is maximum variance",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps(result, indent=2))
