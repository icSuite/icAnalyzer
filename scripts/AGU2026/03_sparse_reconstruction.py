"""Step 3: reconstruct paired conductance fields from sparse observations.

This is a covariance-conditioning demonstration, not independent model
validation. By default target fields are members of the supplied conductance
ensemble. For an independent Monte Carlo check, supply a second ensemble made
with a different sampling seed while retaining the covariance from step 2.
"""

#%% Imports

import argparse
import csv
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

from icanalyzer.wic_si12.ensemble import sparse_ensemble_reconstruction
from scripts.AGU2026._common import ensemble_archive, mapped_npz_array, masked, save_figure


#%% Sparse geometry and scoring

def sparse_sites(support, count, generator, geometry):
    candidates = np.flatnonzero(support)
    if count < 1 or count >= candidates.size:
        raise ValueError("observation count must leave supported cells unobserved")
    if geometry == "random":
        return generator.choice(candidates, count, replace=False)
    if geometry != "track":
        raise ValueError("geometry must be random or track")
    rows, columns = np.unravel_index(candidates, support.shape)
    anchor_column = int(generator.choice(candidates) % support.shape[1])
    order = np.lexsort((generator.random(candidates.size), abs(columns - anchor_column)))
    return candidates[order[:count]]


def evaluate_reconstruction(prior_mean, eigenfactor, truth, support, sites, noise, observation_std):
    field_size = support.size
    indices = np.concatenate((sites, field_size + sites))
    values = truth.reshape(-1)[indices] + noise.reshape(-1)
    posterior = sparse_ensemble_reconstruction(
        prior_mean, eigenfactor, indices, values, observation_std**2
    )
    score = support.copy().reshape(-1)
    score[sites] = False
    score = score.reshape(support.shape)
    records = []
    for channel in range(2):
        residual = posterior["mean"][channel][score] - truth[channel][score]
        prior_residual = prior_mean[channel][score] - truth[channel][score]
        lower = posterior["mean"][channel][score] - norm.ppf(0.975) * posterior["standard_deviation"][channel][score]
        upper = posterior["mean"][channel][score] + norm.ppf(0.975) * posterior["standard_deviation"][channel][score]
        records.append({
            "scored_cells": int(residual.size),
            "sse": float(residual @ residual),
            "prior_sse": float(prior_residual @ prior_residual),
            "bias_sum": float(residual.sum()),
            "covered_95": int(np.sum((truth[channel][score] >= lower) & (truth[channel][score] <= upper))),
            "negative": int(np.sum(posterior["mean"][channel][score] < 0)),
        })
    return posterior, records, values


#%% Poster figures

def plot_example(truth, prior_mean, posterior, support, sites, values, count, output, dpi):
    field_size = support.size
    observed = np.full_like(truth, np.nan)
    observed[0].flat[sites] = values[:len(sites)]
    observed[1].flat[sites] = values[len(sites):]
    figure, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
    for channel, name in enumerate(("Hall", "Pedersen")):
        vmax = max(float(np.percentile(truth[channel][support], 99.5)), 0.1)
        error = posterior["mean"][channel] - truth[channel]
        error_limit = max(float(np.percentile(abs(error[support]), 99.5)), 0.1)
        panels = (truth[channel], observed[channel], prior_mean[channel], posterior["mean"][channel], error)
        titles = ("Target member", "Sparse observations", "Prior mean", "Reconstruction", "Residual")
        for column, (panel, title) in enumerate(zip(panels, titles)):
            image = axes[channel, column].imshow(
                masked(panel, support), origin="lower",
                cmap="RdBu_r" if column == 4 else "viridis",
                vmin=-error_limit if column == 4 else 0,
                vmax=error_limit if column == 4 else vmax,
                interpolation="nearest",
            )
            axes[channel, column].set_title(f"{name}: {title}")
            axes[channel, column].set_axis_off()
            if column != 1:
                axes[channel, column].scatter(
                    sites % support.shape[1], sites // support.shape[1],
                    s=5, facecolors="none", edgecolors="cyan",
                )
            figure.colorbar(image, ax=axes[channel, column], label="S", shrink=0.72)
    figure.suptitle(f"Joint Hall–Pedersen reconstruction from {count} spatial sites")
    save_figure(figure, output, "03_sparse_reconstruction_example", dpi)
    plt.close(figure)


def plot_skill(metrics, output, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for field in ("Hall", "Pedersen"):
        selected = [row for row in metrics if row["field"] == field]
        counts = [row["sites"] for row in selected]
        axes[0].plot(counts, [row["rmse_siemens"] for row in selected], "o-", label=f"{field} reconstruction")
        axes[0].plot(counts, [row["prior_rmse_siemens"] for row in selected], "--", label=f"{field} prior")
        axes[1].plot(counts, [row["coverage_95"] for row in selected], "o-", label=field)
    axes[0].set(xlabel="Observed spatial sites", ylabel="RMSE [S]", title="Error on unobserved cells")
    axes[1].set(xlabel="Observed spatial sites", ylabel="Fraction", title="Posterior 95% coverage", ylim=(0, 1))
    axes[1].axhline(0.95, color="grey", linestyle="--")
    for axis in axes:
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    save_figure(figure, output, "03_sparse_reconstruction_skill", dpi)
    plt.close(figure)


#%% Chronological sparse-reconstruction workflow

def run(arguments):
    truth_archive = ensemble_archive(arguments.ensemble)
    covariance_path = arguments.covariance.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty sparse-reconstruction output directory")
    output.mkdir(parents=True, exist_ok=True)
    with np.load(covariance_path, allow_pickle=False) as saved:
        prior_mean = np.asarray(saved["mean"], dtype=np.float64)
        support = np.asarray(saved["support"], dtype=bool)
        eigenfactor = np.asarray(saved["eigenfactor"], dtype=np.float64)
        source_crc = int(saved["source_factor_crc32"]) if "source_factor_crc32" in saved else None
        source_size = int(saved["source_factor_file_size"]) if "source_factor_file_size" in saved else None
    if eigenfactor.shape[0] != prior_mean.size or prior_mean.shape != (2, *support.shape):
        raise ValueError("step-2 covariance arrays are inconsistent")
    retained = min(arguments.modes, eigenfactor.shape[1])
    eigenfactor = eigenfactor[:, :retained]

    with zipfile.ZipFile(truth_archive) as archive:
        factor_info = archive.getinfo("conductance_anomaly_factor.npy")
    with np.load(truth_archive, allow_pickle=False) as saved:
        truth_mean = np.asarray(saved["conductance_mean"], dtype=np.float64)
        truth_support = np.asarray(saved["conductance_supported"], dtype=bool)
    if truth_mean.shape != prior_mean.shape or not np.array_equal(truth_support, support):
        raise ValueError("truth ensemble and step-2 covariance use different grids/support")

    generator = np.random.default_rng(arguments.seed)
    with mapped_npz_array(truth_archive, "conductance_anomaly_factor", arguments.workspace) as truth_factor:
        if truth_factor.shape[0] != truth_mean.size or truth_factor.shape[1] < arguments.truth_members:
            raise ValueError("truth ensemble has too few compatible members")
        member_indices = np.sort(generator.choice(truth_factor.shape[1], arguments.truth_members, replace=False))
        truths = truth_mean[None] + np.sqrt(truth_factor.shape[1] - 1) * np.asarray(truth_factor[:, member_indices].T).reshape(arguments.truth_members, *truth_mean.shape)

    records = []
    example = None
    for member_number, (member_index, truth) in enumerate(zip(member_indices, truths)):
        maximum_count = max(arguments.observation_counts)
        all_sites = sparse_sites(support, maximum_count, generator, arguments.geometry)
        all_noise = generator.normal(0, arguments.observation_std, size=(2, maximum_count))
        for count in arguments.observation_counts:
            # Nested sites and errors make the observation-count comparison fair.
            sites = all_sites[:count]
            noise = all_noise[:, :count]
            posterior, channel_records, values = evaluate_reconstruction(
                prior_mean, eigenfactor, truth, support, sites, noise,
                arguments.observation_std,
            )
            for channel, field in enumerate(("Hall", "Pedersen")):
                records.append({
                    "member_number": member_number, "member_index": int(member_index),
                    "sites": count, "field": field, **channel_records[channel],
                })
            if member_number == 0 and count == max(arguments.observation_counts):
                example = (truth, posterior, sites, values)

    metrics = []
    for count in arguments.observation_counts:
        for field in ("Hall", "Pedersen"):
            selected = [row for row in records if row["sites"] == count and row["field"] == field]
            cells = sum(row["scored_cells"] for row in selected)
            metrics.append({
                "sites": count, "field": field, "scored_cells": cells,
                "rmse_siemens": float(np.sqrt(sum(row["sse"] for row in selected) / cells)),
                "prior_rmse_siemens": float(np.sqrt(sum(row["prior_sse"] for row in selected) / cells)),
                "bias_siemens": float(sum(row["bias_sum"] for row in selected) / cells),
                "coverage_95": float(sum(row["covered_95"] for row in selected) / cells),
                "negative_fraction": float(sum(row["negative"] for row in selected) / cells),
            })
    with (output / "03_sparse_reconstruction_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    truth, posterior, sites, values = example
    plot_example(truth, prior_mean, posterior, support, sites, values, len(sites), output, arguments.dpi)
    plot_skill(metrics, output, arguments.dpi)

    same_factor = source_crc == factor_info.CRC and source_size == factor_info.file_size
    report = {
        "purpose": "joint covariance conditioning from sparse Hall/Pedersen observations",
        "covariance": str(covariance_path),
        "truth_ensemble": str(truth_archive),
        "truth_members": member_indices.tolist(),
        "truth_independent_of_covariance_archive": not same_factor,
        "observation_counts": arguments.observation_counts,
        "observation_standard_deviation_siemens": arguments.observation_std,
        "geometry": arguments.geometry,
        "retained_joint_modes": retained,
        "metrics": metrics,
        "state_space": "physical Hall/Pedersen siemens",
        "pairing": "both conductances at each observed site condition one joint Hall/Pedersen covariance",
        "interpretation": (
            "independent Monte Carlo distribution-consistency test" if not same_factor else
            "same-ensemble distribution-consistency demonstration; not independent validation"
        ),
    }
    (output / "03_sparse_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("covariance", type=Path, help="02_conductance_covariance.npz")
    parser.add_argument("ensemble", type=Path, help="truth ensemble directory or ensemble_products.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--observation-counts", nargs="+", type=int, default=[10, 30, 100])
    parser.add_argument("--observation-std", type=float, default=0.5, help="independent Hall/Pedersen error in siemens")
    parser.add_argument("--geometry", choices=("random", "track"), default="random")
    parser.add_argument("--truth-members", type=int, default=16)
    parser.add_argument("--modes", type=int, default=100)
    parser.add_argument("--workspace", type=Path, help="temporary local directory for the extracted truth factor")
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args(argv)
    arguments.observation_counts = sorted(set(arguments.observation_counts))
    if min(arguments.observation_counts + [arguments.truth_members, arguments.modes, arguments.dpi]) < 1:
        parser.error("observation counts, truth members, modes and DPI must be positive")
    if not np.isfinite(arguments.observation_std) or arguments.observation_std <= 0:
        parser.error("--observation-std must be finite and positive")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
