"""Create reproducible diagnostics from a saved WIC/SI12 VAE run."""

#%% Imports

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icanalyzer.wic_si12.diagnostics import (
    best_validation_epoch,
    covariance_spectrum,
    joint_correlation_maps,
    leading_covariance_modes,
    maximum_variance_anchor,
    reconstruct_ensemble,
)


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

def plot_sample_mosaic(samples, family, output, created, n_examples):
    if n_examples < 2:
        raise ValueError("at least two sample examples are required")
    sample_indices = np.linspace(0, samples.shape[0] - 1, n_examples, dtype=int)
    selected = samples[sample_indices]
    names = FIELD_NAMES[family]
    units = FIELD_UNITS[family]
    figure, axes = plt.subplots(
        2, n_examples, figsize=(2.7 * n_examples, 5.5), layout="constrained"
    )

    for channel in range(2):
        signed = bool(np.any(selected[:, channel] < 0))
        limits = robust_limits(selected[:, channel], signed=signed)
        cmap = "coolwarm" if signed else "viridis"
        for column, sample_index in enumerate(sample_indices):
            image_panel(
                axes[channel, column],
                samples[sample_index, channel],
                f"{names[channel]} sample {sample_index}",
                units[channel],
                cmap=cmap,
                limits=limits,
                show_xlabel=channel == 1,
                show_ylabel=column == 0,
            )
    save_figure(figure, output, f"{family}_samples.png", created)
    return sample_indices.tolist()


def plot_ensemble_moments(products, output, created):
    fuv_mean = products["fuv_mean"]
    conductance_mean = products["conductance_mean"]
    fuv_std = np.sqrt(
        np.sum(products["fuv_anomaly_factor"]**2, axis=1)
    ).reshape(fuv_mean.shape)
    conductance_std = np.sqrt(
        np.sum(products["conductance_anomaly_factor"]**2, axis=1)
    ).reshape(conductance_mean.shape)

    fields = (
        (fuv_mean[0], fuv_std[0], "corrected WIC", "counts"),
        (fuv_mean[1], fuv_std[1], "SI12", "counts"),
        (conductance_mean[0], conductance_std[0], "Hall", "S"),
        (conductance_mean[1], conductance_std[1], "Pedersen", "S"),
    )
    figure, axes = plt.subplots(2, 4, figsize=(17, 7.5), layout="constrained")
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


def plot_covariance_modes(factor, state_shape, family, output, created, n_modes):
    eigenvalues, modes = leading_covariance_modes(
        factor, state_shape, n_modes=n_modes
    )
    names = FIELD_NAMES[family]
    fraction = eigenvalues / np.sum(factor**2)
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
    for family, values in spectra.items():
        fraction = values / values.sum()
        axes[0].semilogy(np.arange(1, len(values) + 1), fraction, label=family)
        axes[1].plot(np.arange(1, len(values) + 1), np.cumsum(fraction), label=family)
    axes[0].set(title="Covariance spectrum", xlabel="mode", ylabel="variance fraction")
    axes[1].set(title="Cumulative explained variance", xlabel="mode", ylabel="fraction")
    axes[1].set_ylim(0, 1.01)
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    save_figure(figure, output, "covariance_spectrum.png", created)


#%% Sparse reconstruction

def plot_reconstruction(products, output, created):
    prior_mean = products.get(
        "reconstruction_prior_mean", products["conductance_mean"]
    )
    posterior_mean = products["reconstruction_mean"]
    prior_std = products.get("reconstruction_prior_standard_deviation")
    if prior_std is None:
        prior_std = np.sqrt(
            np.sum(products["conductance_anomaly_factor"]**2, axis=1)
        ).reshape(prior_mean.shape)
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


#%% Main workflow

def parse_anchor(value):
    if value is None:
        return None
    row, column = value.split(",")
    return int(row), int(column)


def run(arguments):
    run_directory = Path(arguments.run_directory)
    output = Path(arguments.output) if arguments.output else run_directory / "figures"
    output.mkdir(parents=True, exist_ok=True)

    summary = json.loads((run_directory / "summary.json").read_text())
    artifact = torch.load(
        run_directory / "wic_si12_vae.pt", map_location="cpu", weights_only=False
    )
    with np.load(run_directory / "ensemble_products.npz") as stored:
        products = {name: np.asarray(stored[name]) for name in stored.files}

    fuv_samples = reconstruct_ensemble(
        products["fuv_mean"], products["fuv_anomaly_factor"]
    )
    conductance_samples = reconstruct_ensemble(
        products["conductance_mean"], products["conductance_anomaly_factor"]
    )
    state_shape = products["fuv_mean"].shape
    requested_anchor = parse_anchor(arguments.anchor)
    anchors = {
        "fuv": requested_anchor or maximum_variance_anchor(
            products["fuv_anomaly_factor"], state_shape, channel=0
        ),
        "conductance": requested_anchor or maximum_variance_anchor(
            products["conductance_anomaly_factor"], state_shape, channel=0
        ),
    }

    created = []
    best_epoch, best_metrics = plot_loss_history(
        artifact["history"], output, created
    )
    sample_indices = {
        "fuv": plot_sample_mosaic(
            fuv_samples, "fuv", output, created, arguments.examples
        ),
        "conductance": plot_sample_mosaic(
            conductance_samples, "conductance", output, created, arguments.examples
        ),
    }
    plot_ensemble_moments(products, output, created)
    latent_samples_available = plot_latent_samples(products, output, created)

    spectra = {}
    mode_diagnostics = {}
    for family in ("fuv", "conductance"):
        factor = products[f"{family}_anomaly_factor"]
        plot_correlation_maps(
            factor, state_shape, family, anchors[family], output, created
        )
        eigenvalues, fraction = plot_covariance_modes(
            factor, state_shape, family, output, created, arguments.modes
        )
        spectra[family] = covariance_spectrum(factor)
        mode_diagnostics[family] = {
            "leading_eigenvalues": eigenvalues.tolist(),
            "leading_variance_fractions": fraction.tolist(),
        }
    plot_covariance_spectra(spectra, output, created)
    reconstruction_has_truth = plot_reconstruction(products, output, created)

    diagnostics = {
        "run_directory": str(run_directory.resolve()),
        "sample_count": int(summary["sample_count"]),
        "best_validation_epoch": best_epoch,
        "best_validation_metrics": best_metrics,
        "anchors": {name: list(anchor) for name, anchor in anchors.items()},
        "sample_indices": sample_indices,
        "latent_samples_available": latent_samples_available,
        "modes": mode_diagnostics,
        "reconstruction_truth_available": reconstruction_has_truth,
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
        "--anchor",
        help="optional common CS anchor as ROW,COLUMN; default is maximum variance",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    result = run(parse_arguments())
    print(json.dumps(result, indent=2))
