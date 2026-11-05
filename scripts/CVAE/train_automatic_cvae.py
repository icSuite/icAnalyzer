"""Train the existing Kp-conditional H/P spline VAE from a disk-backed cache.

Training only: no automatic ensemble sampling or covariance allocation.
Current VAE/CVAE entry points and saved runs remain unchanged.
"""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.cvae.artifacts import CONDUCTANCE_CVAE_ARTIFACT_TYPE, save_run_artifact
from icanalyzer.cvae.data import fit_kp_conditioning
from icanalyzer.cvae.loss import LEGACY_LOSS_CONFIGURATION, gaussian_uncertainty_configuration
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.training import _run_epoch, set_random_seed, train_vae, training_device
from scripts.CVAE.prepare_cvae_hp_cache import AutomaticConductanceDataset, cached_coverage_weighting, cached_precision_configuration, read_cache


def run(arguments):
    cache = arguments.cache.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    device = training_device(arguments.device)
    if output.exists():
        raise FileExistsError("choose a new run directory; existing runs are never overwritten")
    metadata = read_cache(cache)
    seed = metadata["seed"] if arguments.seed is None else arguments.seed
    include_uncertainty = arguments.measurement_uncertainty or arguments.gaussian_uncertainty
    # Divide the training mapping budget across workers, not duplicate it.
    datasets = {}
    for name in ("train", "validation", "test"):
        if not metadata["used_frames"][name]:
            datasets[name] = None
            continue
        mapping_budget = arguments.cache_mib
        if name == "train":
            mapping_budget /= max(1, arguments.num_workers)
        datasets[name] = AutomaticConductanceDataset(cache, name, mapping_budget, arguments.max_open_orbits, include_uncertainty=include_uncertainty)
    loaders = {}
    for name, dataset in datasets.items():
        if dataset is None:
            loaders[name] = None
            continue
        workers = arguments.num_workers if name == "train" else 0
        loading_options = {"num_workers": workers, "pin_memory": device == "cuda"}
        if workers:
            loading_options.update(multiprocessing_context="spawn", prefetch_factor=arguments.prefetch_factor)
        # Ordered batches and non-persistent workers preserve the old RNG sequence.
        loaders[name] = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=name == "train", **loading_options)
    transform = datasets["train"].transform
    kp_conditioning = fit_kp_conditioning(datasets["train"].kp)
    loss_configuration = dict(LEGACY_LOSS_CONFIGURATION)
    if arguments.measurement_uncertainty:
        print("Fitting bounded training precision sample...", flush=True)
        loss_configuration = cached_precision_configuration(cache, arguments.uncertainty_variance_floor, arguments.maximum_relative_precision, arguments.precision_scale_samples)
    elif arguments.gaussian_uncertainty:
        loss_configuration = gaussian_uncertainty_configuration(arguments.model_error_std)
    coverage_arrays = None
    if arguments.coverage_weighting:
        coverage, coverage_arrays = cached_coverage_weighting(cache, arguments.coverage_sigma, arguments.coverage_max_boost, arguments.coverage_kp_edges, arguments.coverage_balance)
        loss_configuration["coverage_weighting"] = coverage
        limit = "uncapped" if arguments.coverage_balance == "full" else f"{arguments.coverage_max_boost:g}"
        print(f"Coverage weighting: balance={arguments.coverage_balance}, sigma={arguments.coverage_sigma:g} pixels, maximum boost={limit}; training frames per Kp bin: {coverage['training_frames_per_bin']}.", flush=True)

    configuration = {name: getattr(arguments, name) for name in ("epochs", "batch_size", "latent_dim", "learning_rate", "beta", "spline_control_points", "spline_degree", "scheduler_factor", "scheduler_patience", "scheduler_minimum_learning_rate")}
    configuration.update({
        "seed": seed, "device": device, "input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S",
        "dataset_backend": metadata["cache_format"], "maximum_missing_fraction": metadata["maximum_missing_fraction"],
        "kp_conditioning": kp_conditioning, "loss_configuration": loss_configuration, "optimizer": "Adam", "beta_schedule": "constant",
        "learning_rate_scheduler": "ReduceLROnPlateau", "scheduler_monitor": "train_total",
        "cache_mib_per_split": arguments.cache_mib, "max_open_orbits_per_worker": arguments.max_open_orbits,
        "num_workers": arguments.num_workers, "prefetch_factor": arguments.prefetch_factor, "pin_memory": device == "cuda",
        "uncertainty_arrays_loaded": include_uncertainty,
        "training_holes": {"probability": arguments.hole_probability, "size": arguments.hole_size, "seed": seed + 3, "target_policy": "original_observed_cells_retained"} if arguments.training_holes else None,
        "frame_selection": {name: {"used": metadata["used_frames"][name]} for name in datasets},
        "support_policy": "retained training common-valid support; no synthetic background targets"
    })
    support = np.load(cache / "training_support.npy", allow_pickle=False)
    provenance = {
        "dataset_backend": metadata["cache_format"], "cache_directory": str(cache), "cache_manifest_sha256": metadata["manifest_sha256"],
        "automatic_masks": metadata["automatic_masks"], "input_product_type": "conductance_cs", "grid_id": metadata["grid_id"], "grid": metadata["grid"],
        "field_names": ["Hall", "Pedersen"], "field_units": "S", "propagation_reapplied": False,
        "electron_energy_source": "upstream native conductance retrieval, not fixed 2 keV",
        "training_support": support, "unsupported_training_cells": int((~support).sum()),
        "products": {name: [row["product"] for row in metadata["orbits"] if row["split"] == name] for name in datasets},
        "retained_frame_kp": {name: dataset.kp.copy() for name, dataset in datasets.items() if dataset is not None}
    }
    print(f"CVAE cached frames: {metadata['used_frames']}. Device: {device}. No ensembles will be generated.", flush=True)
    print(f"Training loader: {arguments.num_workers} workers, {arguments.cache_mib:g} MiB total mapping budget; uncertainty arrays: {include_uncertainty}.", flush=True)

    # Reuse the same architecture, loss, optimizer/scheduler and hole machinery.
    set_random_seed(seed)
    model = SplineConditionalVAE(
        image_shape=(46, 46), latent_dim=arguments.latent_dim,
        spline_control_points=arguments.spline_control_points,
        spline_degree=arguments.spline_degree,
        kp_conditioning=kp_conditioning,
        field_names=("Hall", "Pedersen")
    )
    output.mkdir(parents=True)
    (output / "cache_manifest.json").write_bytes((cache / "manifest.json").read_bytes())
    if coverage_arrays is not None:
        np.savez_compressed(output / "coverage_weighting.npz", **coverage_arrays)
        (output / "coverage_weighting.json").write_text(json.dumps(loss_configuration["coverage_weighting"], indent=2) + "\n")
    try:
        model, history = train_vae(
            model, loaders["train"], validation_loader=loaders["validation"], epochs=arguments.epochs,
            learning_rate=arguments.learning_rate, beta=arguments.beta, device=device, seed=seed, loss_configuration=loss_configuration,
            scheduler_factor=arguments.scheduler_factor, scheduler_patience=arguments.scheduler_patience,
            scheduler_minimum_learning_rate=arguments.scheduler_minimum_learning_rate,
            hole_probability=arguments.hole_probability if arguments.training_holes else 0.0, hole_size=arguments.hole_size
        )
        configuration["selected_epoch"] = model.best_validation_epoch or arguments.epochs
        configuration["selected_validation_total"] = model.best_validation_total
        artifact = output / "automatic_cvae.pt"
        # Preserve the trained checkpoint even if later test evaluation fails.
        save_run_artifact(artifact, model, transform, history, configuration, provenance, metadata["split_files"], artifact_type=CONDUCTANCE_CVAE_ARTIFACT_TYPE)
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        evaluation = {}
        for name in ("validation", "test"):
            if loaders[name] is not None:
                evaluation[name] = _run_epoch(model, loaders[name], optimizer=None, device=torch.device(device), beta=arguments.beta, training=False, loss_configuration=loss_configuration, progress_description=f"{name} evaluation")
        save_run_artifact(artifact, model, transform, history, configuration, provenance, metadata["split_files"], evaluation=evaluation, artifact_type=CONDUCTANCE_CVAE_ARTIFACT_TYPE)
        summary = {
            "artifact": "automatic_cvae.pt", "artifact_type": CONDUCTANCE_CVAE_ARTIFACT_TYPE, "dataset_backend": metadata["cache_format"],
            "cache_directory": str(cache), "cache_manifest_sha256": metadata["manifest_sha256"], "used_frames": metadata["used_frames"],
            "training_configuration": configuration, "held_out_evaluation": evaluation,
            "ensemble_generated": False, "postprocessing": "Use a cache-aware workflow; existing old-run plotters do not load these masks."
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {artifact}. Training complete; sampling is a separate step.", flush=True)
        return summary
    finally:
        for dataset in datasets.values():
            if dataset is not None:
                dataset.close()


#%% New entry point; existing training defaults/scientific options preserved

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, help="training RNG seed; default is the frozen cache split seed")
    for name, default in (("epochs", 1000), ("batch-size", 8), ("latent-dim", 36), ("spline-control-points", 30), ("spline-degree", 3), ("scheduler-patience", 10), ("max-open-orbits", 64), ("num-workers", 2), ("prefetch-factor", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 1e-4), ("beta", 1.0), ("scheduler-factor", 0.5), ("scheduler-minimum-learning-rate", 0.0), ("cache-mib", 512)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--training-holes", action="store_true")
    parser.add_argument("--hole-probability", type=float, default=0.5)
    parser.add_argument("--hole-size", type=int, default=12)
    parser.add_argument("--coverage-weighting", action="store_true", help="Fit smoothed training-only quality coverage and boost underrepresented pixels in the loss")
    parser.add_argument("--coverage-balance", choices=("capped", "full"), default="capped", help="Use capped compensation or the full uncapped inverse of smoothed coverage")
    parser.add_argument("--coverage-sigma", type=float, default=1.5, help="Gaussian width in grid pixels, before inversion (default: 1.5)")
    parser.add_argument("--coverage-max-boost", type=float, default=3., help="Maximum coverage multiplier in capped mode (default: 3; ignored in full mode)")
    parser.add_argument("--coverage-kp-edges", type=float, nargs="+", default=[0., 9.], help="Optional broad Kp bin edges covering 0 through 9; default fits one shared spatial map")
    uncertainty = parser.add_mutually_exclusive_group()
    uncertainty.add_argument("--measurement-uncertainty", action="store_true")
    uncertainty.add_argument("--gaussian-uncertainty", action="store_true")
    parser.add_argument("--model-error-std", nargs=2, type=float)
    parser.add_argument("--uncertainty-variance-floor", type=float)
    parser.add_argument("--maximum-relative-precision", type=float)
    parser.add_argument("--precision-scale-samples", type=int)
    arguments = parser.parse_args(argv)
    if arguments.model_error_std is not None and not arguments.gaussian_uncertainty:
        parser.error("--model-error-std requires --gaussian-uncertainty")
    arguments.model_error_std = arguments.model_error_std or [0.5, 0.5]
    for name, default in (("uncertainty_variance_floor", 1e-3), ("maximum_relative_precision", 100), ("precision_scale_samples", 2_000_000)):
        if getattr(arguments, name) is not None and not arguments.measurement_uncertainty:
            parser.error("--" + name.replace("_", "-") + " requires --measurement-uncertainty")
        if getattr(arguments, name) is None:
            setattr(arguments, name, default)
    if arguments.epochs < 1 or arguments.batch_size < 1:
        parser.error("epochs and batch size must be positive")
    if not np.isfinite(arguments.beta) or arguments.beta < 0:
        parser.error("beta must be finite and nonnegative")
    if arguments.precision_scale_samples < 1:
        parser.error("precision scale samples must be positive")
    if arguments.num_workers < 0 or arguments.prefetch_factor < 1:
        parser.error("worker count must be nonnegative and prefetch factor must be positive")
    if not np.isfinite(arguments.coverage_sigma) or arguments.coverage_sigma < 0:
        parser.error("coverage sigma must be finite and nonnegative")
    if not np.isfinite(arguments.coverage_max_boost) or arguments.coverage_max_boost < 1:
        parser.error("coverage maximum boost must be finite and at least 1")
    from icanalyzer.cvae.loss import coverage_bin_indices
    try:
        coverage_bin_indices([], arguments.coverage_kp_edges)
    except ValueError as error:
        parser.error(str(error))
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
