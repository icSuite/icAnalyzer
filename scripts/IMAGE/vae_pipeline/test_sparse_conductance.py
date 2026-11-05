"""Compare saved conductance priors on one reusable sparse-observation benchmark.

Like legacy optimal interpolation, condition covariance eigenmodes, not the
VAE decoder. Defaults use log10(conductance + 1 S), 100 modes and 0.1 dex noise.
"""

#%% Imports

import argparse
import gc
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import norm
from threadpoolctl import threadpool_limits

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.diagnostics import covariance_mode_decomposition
from icanalyzer.wic_si12.ensemble import sparse_ensemble_reconstruction
from icanalyzer.wic_si12.physics import conductance_from_corrected_wic
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


#%% Freeze truth, spatial observations and noise once for all models

def sparse_sites(valid, count, generator, geometry):
    candidates = np.flatnonzero(valid)
    if count <= 0 or count >= candidates.size:
        raise ValueError("observation count must leave some supported sites unobserved")
    if geometry == "random":
        return generator.choice(candidates, count, replace=False)
    if geometry != "track":
        raise ValueError("geometry must be random or track")
    rows, columns = np.unravel_index(candidates, valid.shape)
    anchor = generator.choice(candidates)
    anchor_column = anchor % valid.shape[1]
    # A narrow meridional grid track; random ties avoid favouring one end.
    ordering = np.lexsort((generator.random(candidates.size), abs(columns - anchor_column)))
    return candidates[ordering[:count]]


def make_benchmark(run, arguments):
    artifact = torch.load(run / "wic_si12_vae.pt", map_location="cpu", weights_only=False)
    transform = ChannelTransform.from_dict(artifact["transform"])
    dataset, split = _load_heldout_dataset(artifact, transform, arguments.split)
    training, _ = _load_heldout_dataset(artifact, transform, "train")
    support = np.any(training.common_valid[:, 0] & (training.weight[:, 0] > 0), axis=0)
    del training
    valid = dataset.common_valid[:, 0] & support
    available = np.flatnonzero(valid.sum(axis=(1, 2)) > max(arguments.observation_counts))
    if available.size != len(dataset):
        raise ValueError("some held-out frames have too few sites; reduce observation counts")
    if arguments.max_frames is not None:
        generator = np.random.default_rng(arguments.seed)
        available = np.sort(generator.choice(available, min(arguments.max_frames, available.size), replace=False))
    converted = conductance_from_corrected_wic(dataset.values[available, 0], include_intermediates=False)
    truth = np.stack((converted["H"], converted["P"]), axis=1)
    valid = valid[available] & np.all(np.isfinite(truth), axis=1)
    generator = np.random.default_rng(arguments.seed)
    sites = np.stack([sparse_sites(mask, max(arguments.observation_counts), generator, arguments.geometry) for mask in valid])
    noise = generator.normal(size=(len(available), 2, sites.shape[1])) * arguments.observation_std
    configuration = {
        "split": split, "seed": arguments.seed, "geometry": arguments.geometry,
        "observation_counts": arguments.observation_counts,
        "observation_std": arguments.observation_std, "observation_mode": arguments.observation_mode,
        "state_space": arguments.state_space, "epsilon_siemens": 1.0,
        "source_run": str(run.resolve()), "reference_kind": "heldout_FUV_fixed_2keV_proxy",
        "selection": {key: artifact["training_configuration"].get(key) for key in ("minimum_sza_degrees", "maximum_missing_fraction", "comparison_data_version")},
        "split_orbits": {key: sorted(Path(path).name for path in paths) for key, paths in artifact["split_files"].items()},
        "propagation": json.loads((run / "summary.json").read_text())["propagation_order"]
    }
    return {
        "truth": truth, "valid": valid, "sites": sites, "noise": noise,
        "orbit": dataset.frame_orbit[available], "time": dataset.frame_time[available],
        "mlat": artifact["data_provenance"]["grid"]["mlat"],
        "mlt": artifact["data_provenance"]["grid"]["mlt"],
        "configuration": np.array(json.dumps(configuration))
    }


def check_comparable(artifact, benchmark, summary):
    configuration = json.loads(str(benchmark["configuration"]))
    splits = {key: sorted(Path(path).name for path in paths) for key, paths in artifact["split_files"].items()}
    if splits != configuration["split_orbits"]:
        raise ValueError("benchmark requires the same saved orbit splits; changing seed may change them")
    for key, expected in configuration["selection"].items():
        if artifact["training_configuration"].get(key) != expected:
            raise ValueError(f"scientific selection differs: {key}")
    for key in ("mlat", "mlt"):
        grid = np.asarray(artifact["data_provenance"]["grid"][key])
        if grid.shape != benchmark[key].shape or not np.allclose(grid, benchmark[key]):
            raise ValueError("benchmark and prior grids differ")
    if summary["propagation_order"] != configuration["propagation"]:
        raise ValueError("conductance propagation differs")
    if artifact["data_provenance"].get("electron_energy_keV") != 2.0:
        raise ValueError("benchmark requires fixed 2 keV electron propagation")


def check_benchmark(benchmark):
    configuration = json.loads(str(benchmark["configuration"]))
    truth, valid = benchmark["truth"], benchmark["valid"]
    if truth.ndim != 4 or truth.shape[1] != 2 or valid.shape != (truth.shape[0], *truth.shape[2:]):
        raise ValueError("benchmark truth and support shapes are incompatible")
    for key in ("orbit", "time"):
        if benchmark[key].shape != (len(truth),):
            raise ValueError("benchmark frame identities are incompatible")
    sites, noise = benchmark["sites"], benchmark["noise"]
    if sites.shape != (len(truth), max(configuration["observation_counts"])) or noise.shape != (len(truth), 2, sites.shape[1]):
        raise ValueError("benchmark observation shapes are incompatible")
    if not np.isfinite(configuration["observation_std"]) or configuration["observation_std"] <= 0:
        raise ValueError("benchmark observation standard deviation must be positive")
    if configuration["state_space"] not in ("log10", "physical") or configuration["epsilon_siemens"] != 1.0:
        raise ValueError("unsupported benchmark state transform")
    if np.any(~np.isfinite(noise)) or np.any(~np.isfinite(truth[np.broadcast_to(valid[:, None], truth.shape)])):
        raise ValueError("benchmark has nonfinite observations or supported truth")
    for mask, indices in zip(valid, sites):
        if indices.dtype.kind not in "iu" or np.any(indices < 0) or np.any(indices >= mask.size):
            raise ValueError("benchmark observation index outside grid")
        if np.unique(indices).size != indices.size or not mask.reshape(-1)[indices].all() or indices.size >= mask.sum():
            raise ValueError("benchmark needs unique supported observations and unobserved support")


#%% Convert prior members in batches; never retain a second full RAM ensemble

def prior_modes(run, state_space, n_modes, workspace):
    with np.load(run / "ensemble_products.npz") as archive:
        mean = archive["conductance_mean"]
        factor = archive["conductance_anomaly_factor"]
    if mean.shape[0] != 2 or factor.shape[0] != mean.size or factor.shape[1] < 2:
        raise ValueError("saved conductance factor has an incompatible shape")
    if not np.all(np.isfinite(mean)):
        raise ValueError("saved prior mean contains nonfinite values")
    sample_count = factor.shape[1]
    if state_space == "log10":
        # A disk-backed work array avoids doubling peak ensemble memory.
        log_factor = np.memmap(workspace / "log_factor.dat", mode="w+", dtype="float32", shape=factor.shape)
        for start in range(0, sample_count, 256):
            stop = min(start + 256, sample_count)
            samples = mean.reshape(-1, 1) + np.sqrt(sample_count - 1) * factor[:, start:stop]
            if np.any(samples < -1e-5) or not np.all(np.isfinite(samples)):
                raise ValueError("saved conductance samples contain negative or nonfinite values")
            # Only correct float32 factor roundoff, not substantive negatives.
            log_factor[:, start:stop] = np.log10(np.maximum(samples, 0) + 1.0)
        del factor
        factor = log_factor
        log_mean = np.empty(mean.size)
        for start in range(0, mean.size, 128):
            stop = min(start + 128, mean.size)
            log_mean[start:stop] = factor[start:stop].mean(axis=1, dtype=np.float64)
            factor[start:stop] -= log_mean[start:stop, None]
            factor[start:stop] /= np.sqrt(sample_count - 1)
        mean = log_mean.reshape(mean.shape)
    decomposition = covariance_mode_decomposition(factor, mean.shape, n_modes=n_modes, power_iterations=2)
    if decomposition["total_variance"] <= 0 or not np.all(np.isfinite(decomposition["eigenvalues"])):
        raise ValueError("prior has no usable covariance")
    eigenfactor = decomposition["modes"].reshape(len(decomposition["eigenvalues"]), -1).T * np.sqrt(decomposition["eigenvalues"])
    del factor
    gc.collect()
    return mean, eigenfactor, {
        "sample_count": sample_count, "retained_modes": eigenfactor.shape[1],
        "mode_method": decomposition["method"],
        "captured_variance_fraction": decomposition["captured_variance_fraction"]
    }


#%% Condition eigenmodes and score only unobserved spatial locations

def evaluate_case(mean, factor, truth, valid, sites, noise, configuration):
    field_size = valid.size
    channels = {"hall": [0], "pedersen": [1], "joint": [0, 1]}[configuration["observation_mode"]]
    indices = np.concatenate([sites + channel * field_size for channel in channels])
    target = np.log10(truth + 1.0) if configuration["state_space"] == "log10" else truth
    values = target.reshape(-1)[indices] + np.concatenate([noise[channel] for channel in channels])
    posterior = sparse_ensemble_reconstruction(mean, factor, indices, values, configuration["observation_std"]**2)
    analysis = posterior["mean"]
    std = posterior["standard_deviation"]
    lower, upper = analysis - norm.ppf(0.975) * std, analysis + norm.ppf(0.975) * std
    if configuration["state_space"] == "log10":
        # Back-transform the median and interval, not a claimed lognormal mean.
        analysis = 10**analysis - 1.0
        lower, upper = 10**lower - 1.0, 10**upper - 1.0
        reference = 10**mean - 1.0
    else:
        reference = mean
    prior_std = posterior["prior_standard_deviation"]
    prior_lower, prior_upper = mean - norm.ppf(0.975) * prior_std, mean + norm.ppf(0.975) * prior_std
    if configuration["state_space"] == "log10":
        prior_lower, prior_upper = 10**prior_lower - 1.0, 10**prior_upper - 1.0
    score_mask = valid.copy().reshape(-1)
    score_mask[sites] = False  # Exclude both fields at observed sites, even Hall-only.
    score_mask = score_mask.reshape(valid.shape)
    records = []
    for channel in range(2):
        residual = (analysis[channel] - truth[channel])[score_mask]
        prior_residual = (reference[channel] - truth[channel])[score_mask]
        records.append({
            "count": int(residual.size), "sse": float(residual @ residual),
            "prior_sse": float(prior_residual @ prior_residual), "error_sum": float(residual.sum()),
            "covered": int(np.sum(((truth[channel] >= lower[channel]) & (truth[channel] <= upper[channel]))[score_mask])),
            "prior_covered": int(np.sum(((truth[channel] >= prior_lower[channel]) & (truth[channel] <= prior_upper[channel]))[score_mask])),
            "negative": int(np.sum(analysis[channel][score_mask] < 0))
        })
    return records, analysis, reference, values


def score_model(mean, factor, benchmark):
    configuration = json.loads(str(benchmark["configuration"]))
    records, examples = [], []
    for count in configuration["observation_counts"]:
        for frame, truth in enumerate(benchmark["truth"]):
            sites = benchmark["sites"][frame, :count]
            scores, analysis, reference, values = evaluate_case(mean, factor, truth, benchmark["valid"][frame], sites, benchmark["noise"][frame, :, :count], configuration)
            records.append({"frame": frame, "orbit": int(benchmark["orbit"][frame]), "sites": count, "channels": scores})
            if count == configuration["observation_counts"][-1] and frame < 3:
                examples.append((frame, analysis, reference, sites, values))
    metrics = []
    for count in configuration["observation_counts"]:
        for channel, name in enumerate(("Hall", "Pedersen")):
            selected = [record["channels"][channel] for record in records if record["sites"] == count]
            totals = {key: sum(record[key] for record in selected) for key in selected[0]}
            metrics.append({
                "sites": count, "field": name, "scored_cells": totals["count"],
                "rmse_siemens": np.sqrt(totals["sse"] / totals["count"]),
                "prior_rmse_siemens": np.sqrt(totals["prior_sse"] / totals["count"]),
                "bias_siemens": totals["error_sum"] / totals["count"],
                "coverage_95": totals["covered"] / totals["count"],
                "prior_coverage_95": totals["prior_covered"] / totals["count"],
                "negative_fraction": totals["negative"] / totals["count"]
            })
    return metrics, records, examples


#%% Truth, observations, prior, reconstruction and error figures

def plot_examples(benchmark, examples, output, name):
    configuration = json.loads(str(benchmark["configuration"]))
    for frame, analysis, reference, sites, values in examples:
        truth = benchmark["truth"][frame]
        valid = benchmark["valid"][frame]
        observed = np.full_like(truth, np.nan)
        channels = {"hall": [0], "pedersen": [1], "joint": [0, 1]}[configuration["observation_mode"]]
        physical_values = 10**values - 1 if configuration["state_space"] == "log10" else values
        for number, channel in enumerate(channels):
            observed[channel].flat[sites] = physical_values[number * len(sites):(number + 1) * len(sites)]
        figure, axes = plt.subplots(2, 5, figsize=(15, 6), constrained_layout=True)
        for channel, label in enumerate(("Hall", "Pedersen")):
            vmax = max(float(np.max(truth[channel][valid])), 0.1)
            error_limit = max(float(np.max(abs(analysis[channel] - truth[channel])[valid])), 0.1)
            fields = (truth[channel], observed[channel], reference[channel], analysis[channel], analysis[channel] - truth[channel])
            for column, (title, field) in enumerate(zip(("Reference", "Sparse observations", "Prior", "Reconstruction", "Error"), fields)):
                masked = np.where(valid, field, np.nan)
                limits = (-error_limit, error_limit) if column == 4 else (0, vmax)
                image = axes[channel, column].imshow(masked, origin="lower", cmap="RdBu_r" if column == 4 else "plasma", vmin=limits[0], vmax=limits[1])
                axes[channel, column].set(title=f"{label}: {title}")
                if column == 1 and channel in channels:
                    axes[channel, column].scatter(sites % valid.shape[1], sites // valid.shape[1], c=observed[channel].flat[sites], s=20, marker="s", cmap="plasma", vmin=0, vmax=vmax)
                else:
                    axes[channel, column].scatter(sites % valid.shape[1], sites // valid.shape[1], s=5, facecolors="none", edgecolors="cyan")
                figure.colorbar(image, ax=axes[channel, column], label="S", shrink=0.7)
        figure.suptitle(f"{name}; orbit {benchmark['orbit'][frame]}; {benchmark['time'][frame]}; {len(sites)} sites")
        figure.savefig(output / f"{name}_example_{frame}.png", dpi=130)
        plt.close(figure)


def plot_skill(results, output):
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), constrained_layout=True)
    for name, result in results.items():
        for channel, field in enumerate(("Hall", "Pedersen")):
            metrics = [entry for entry in result["metrics"] if entry["field"] == field]
            counts = [entry["sites"] for entry in metrics]
            axes[channel, 0].plot(counts, [entry["rmse_siemens"] for entry in metrics], "o-", label=name)
            axes[channel, 0].plot(counts, [entry["prior_rmse_siemens"] for entry in metrics], "--", label=f"{name} prior")
            axes[channel, 1].plot(counts, [entry["coverage_95"] for entry in metrics], "o-", label=name)
            axes[channel, 1].plot(counts, [entry["prior_coverage_95"] for entry in metrics], "--", label=f"{name} prior")
    for channel, field in enumerate(("Hall", "Pedersen")):
        axes[channel, 0].set(title=f"{field}: unobserved-cell error", ylabel="RMSE [S]", xlabel="Observed spatial sites")
        axes[channel, 1].set(title=f"{field}: 95% interval coverage", ylabel="Fraction", xlabel="Observed spatial sites", ylim=(0, 1))
        axes[channel, 1].axhline(0.95, color="grey", linestyle="--")
        for axis in axes[channel]:
            axis.legend(fontsize=7)
    figure.savefig(output / "sparse_skill.png", dpi=140)
    plt.close(figure)


#%% Evaluate current or future saved runs against a frozen fixture

def run(arguments):
    output = arguments.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "metrics.json").exists():
        raise FileExistsError("choose a new output directory; existing benchmark results are preserved")
    names = [path.name for path in arguments.runs]
    if len(set(names)) != len(names):
        raise ValueError("run directory names must be unique")
    benchmark_path = arguments.benchmark
    if benchmark_path is None:
        benchmark_path = output / "benchmark.npz"
        if benchmark_path.exists():
            raise FileExistsError("reuse the existing fixture explicitly with --benchmark")
        print("Freezing benchmark from the first run's held-out split", flush=True)
        benchmark = make_benchmark(arguments.runs[0], arguments)
        np.savez_compressed(benchmark_path, **benchmark)
    with np.load(benchmark_path, allow_pickle=False) as archive:
        benchmark = {key: archive[key] for key in archive.files}
    configuration = json.loads(str(benchmark["configuration"]))
    check_benchmark(benchmark)
    fingerprint = hashlib.sha256(benchmark_path.read_bytes()).hexdigest()
    results = {}
    for path in arguments.runs:
        print(f"Evaluating {path.name}: {len(benchmark['truth'])} frames", flush=True)
        # Read metadata only: new model architectures need not be instantiated.
        artifact = torch.load(path / "wic_si12_vae.pt", map_location="cpu", weights_only=False)
        summary = json.loads((path / "summary.json").read_text())
        check_comparable(artifact, benchmark, summary)
        with tempfile.TemporaryDirectory(prefix="icanalyzer_sparse_") as temporary:
            mean, factor, mode_info = prior_modes(path, configuration["state_space"], arguments.modes, Path(temporary))
        if mean.shape != benchmark["truth"].shape[1:]:
            raise ValueError("prior and benchmark field shapes differ")
        metrics, records, examples = score_model(mean, factor, benchmark)
        # A diagonal covariance cannot update unobserved cells: its score equals
        # the prior-mean control. Store this explicitly rather than a dense matrix.
        with zipfile.ZipFile(path / "ensemble_products.npz") as archive:
            factor_info = archive.getinfo("conductance_anomaly_factor.npy")
        results[path.name] = {
            "run": str(path.resolve()), "modes": mode_info, "metrics": metrics, "per_frame": records,
            "artifact_sha256": hashlib.sha256((path / "wic_si12_vae.pt").read_bytes()).hexdigest(),
            "ensemble_factor_crc32": factor_info.CRC,
            "training_configuration": {key: value for key, value in artifact["training_configuration"].items() if key != "frame_selection"}
        }
        plot_examples(benchmark, examples, output, path.name)
        del factor, artifact
        gc.collect()
    plot_skill(results, output)
    report = {
        "benchmark": str(benchmark_path.resolve()), "benchmark_sha256": fingerprint,
        "protocol": configuration, "frame_count": len(benchmark["truth"]), "results": results,
        "limitations": [
            "Reference is FUV-derived fixed-2-keV conductance, not independent truth",
            "Existing validation/test orbits have already been inspected; not final independent validation",
            "95% coverage is marginal conditional-state coverage, not observation prediction coverage",
            "Log-space reconstructions are back-transformed medians; negatives are reported, not clipped",
            "Diagonal covariance yields the same unobserved-cell point estimates as the prior control",
            "Grid-space modes follow legacy OI mathematics, not its coefficient-space rank convention"
        ]
    }
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--benchmark", type=Path, help="reuse frozen truth/sites/noise; creation options below are then ignored")
    parser.add_argument("--modes", type=int, default=100)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--observation-counts", nargs="+", type=int, default=[10, 30, 100], help="spatial sites; joint observes both fields at each site")
    parser.add_argument("--observation-std", type=float, default=0.1, help="dex in log10 state, siemens in physical state")
    parser.add_argument("--observation-mode", choices=("hall", "pedersen", "joint"), default="joint")
    parser.add_argument("--state-space", choices=("log10", "physical"), default="log10")
    parser.add_argument("--geometry", choices=("random", "track"), default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-frames", type=int, help="seeded subset for a smoke test; omit for all held-out frames")
    arguments = parser.parse_args(argv)
    if arguments.seed < 0 or arguments.modes < 1 or not np.isfinite(arguments.observation_std) or arguments.observation_std <= 0:
        parser.error("seed must be nonnegative; modes and finite observation std must be positive")
    if min(arguments.observation_counts) < 1 or (arguments.max_frames is not None and arguments.max_frames < 1):
        parser.error("observation counts and max frames must be positive")
    arguments.observation_counts = sorted(set(arguments.observation_counts))
    return arguments


if __name__ == "__main__":
    # Small observation solves are faster with one BLAS thread and predictable
    # memory; no model inference or decoder optimization occurs here.
    with threadpool_limits(limits=1):
        report = run(parse_arguments())
    print(json.dumps({name: result["metrics"] for name, result in report["results"].items()}, indent=2))
