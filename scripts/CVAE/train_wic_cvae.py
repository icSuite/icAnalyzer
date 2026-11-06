"""Train a one-field Kp-conditional spline VAE from a WIC cache.

This initial diagnostic path uses the existing quality-weighted legacy loss.
Measurement-uncertainty and spatial-coverage experiments remain intentionally
out of scope so the scientific change is limited to the modeled field.
"""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.cvae.artifacts import WIC_CVAE_ARTIFACT_TYPE, save_run_artifact
from icanalyzer.cvae.cache import AutomaticFieldDataset
from icanalyzer.cvae.data import fit_kp_conditioning
from icanalyzer.cvae.loss import LEGACY_LOSS_CONFIGURATION
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.training import (
    _run_epoch,
    resolve_hole_device,
    set_random_seed,
    train_vae,
    training_device
)
from scripts.CVAE.prepare_wic_cvae_cache import read_wic_cache


#%% Training

def run(arguments):
    cache = arguments.cache.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    device = training_device(arguments.device)
    if output.exists():
        raise FileExistsError("choose a new WIC-CVAE run directory; existing runs are never overwritten")
    metadata = read_wic_cache(cache)
    if metadata.get("field_names") != ["wic_corrected"] or metadata.get("input_product_type") != "precipitation_cs":
        raise ValueError("expected a one-field corrected-WIC Product-2 cache")
    seed = metadata["seed"] if arguments.seed is None else arguments.seed
    hole_device = resolve_hole_device(arguments.hole_device, device)

    datasets = {}
    for name in ("train", "validation", "test"):
        if not metadata["used_frames"][name]:
            datasets[name] = None
            continue
        mapping_budget = arguments.cache_mib
        if name == "train":
            mapping_budget /= max(1, arguments.num_workers)
        datasets[name] = AutomaticFieldDataset(
            cache, name, mapping_budget, arguments.max_open_orbits,
            include_uncertainty=False, expected_cache_format=metadata["cache_format"]
        )

    loaders = {}
    for name, dataset in datasets.items():
        if dataset is None:
            loaders[name] = None
            continue
        workers = arguments.num_workers if name == "train" else 0
        options = {"num_workers": workers, "pin_memory": device == "cuda"}
        if workers:
            options.update(
                multiprocessing_context="spawn",
                prefetch_factor=arguments.prefetch_factor,
                persistent_workers=True
            )
        loaders[name] = DataLoader(dataset, batch_size=arguments.batch_size, shuffle=name == "train", **options)

    transform = datasets["train"].transform
    kp_conditioning = fit_kp_conditioning(datasets["train"].kp)
    loss_configuration = dict(LEGACY_LOSS_CONFIGURATION)
    mask_source = metadata.get("mask_source")
    if mask_source is None:
        mask_source = {"kind": "automatic", **metadata["automatic_masks"]}
    configuration = {name: getattr(arguments, name) for name in (
        "epochs", "batch_size", "latent_dim", "learning_rate", "beta",
        "spline_control_points", "spline_degree", "scheduler_factor",
        "scheduler_patience", "scheduler_minimum_learning_rate"
    )}
    configuration.update({
        "seed": seed, "device": device, "input_product_type": "precipitation_cs",
        "field_names": ["wic_corrected"], "field_units": ["counts"],
        "dataset_backend": metadata["cache_format"],
        "maximum_missing_fraction": metadata["maximum_missing_fraction"],
        "kp_conditioning": kp_conditioning, "loss_configuration": loss_configuration,
        "optimizer": "Adam", "beta_schedule": "constant",
        "learning_rate_scheduler": "ReduceLROnPlateau", "scheduler_monitor": "train_total",
        "cache_mib_per_split": arguments.cache_mib,
        "max_open_orbits_per_worker": arguments.max_open_orbits,
        "num_workers": arguments.num_workers, "prefetch_factor": arguments.prefetch_factor,
        "persistent_workers": arguments.num_workers > 0,
        "pin_memory": device == "cuda", "uncertainty_arrays_loaded": False,
        "training_holes": {
            "probability": arguments.hole_probability, "size": arguments.hole_size,
            "seed": seed + 3, "device": hole_device,
            "target_policy": "original observed cells retained"
        } if arguments.training_holes else None,
        "frame_selection": {name: {"used": metadata["used_frames"][name]} for name in datasets},
        "support_policy": f"{mask_source['kind']} mask AND native finite WIC/quality/coverage support"
    })
    support = np.load(cache / "training_support.npy", allow_pickle=False)
    if support.shape != (46, 46) or not support.any():
        raise ValueError("WIC training support must be a nonempty 46x46 mask")
    provenance = {
        "dataset_backend": metadata["cache_format"], "cache_directory": str(cache),
        "cache_manifest_sha256": metadata["manifest_sha256"],
        "mask_source": mask_source,
        "input_product_type": "precipitation_cs", "grid_id": metadata["grid_id"],
        "grid": metadata["grid"], "field_names": ["wic_corrected"],
        "field_units": ["counts"], "training_support": support,
        "unsupported_training_cells": int((~support).sum()),
        "products": {name: [row["product"] for row in metadata["orbits"] if row["split"] == name] for name in datasets},
        "retained_frame_kp": {name: dataset.kp.copy() for name, dataset in datasets.items() if dataset is not None},
        "conductance_conversion": "not applied; this artifact represents corrected WIC counts"
    }
    print(f"WIC-CVAE cached frames: {metadata['used_frames']}. Device: {device}. No conductance conversion or ensembles will be generated.", flush=True)

    set_random_seed(seed)
    model = SplineConditionalVAE(
        image_shape=(46, 46), latent_dim=arguments.latent_dim,
        spline_control_points=arguments.spline_control_points,
        spline_degree=arguments.spline_degree, kp_conditioning=kp_conditioning,
        field_names=("wic_corrected",)
    )
    output.mkdir(parents=True)
    (output / "cache_manifest.json").write_bytes((cache / "manifest.json").read_bytes())
    try:
        model, history = train_vae(
            model, loaders["train"], validation_loader=loaders["validation"],
            epochs=arguments.epochs, learning_rate=arguments.learning_rate,
            beta=arguments.beta, device=device, seed=seed,
            loss_configuration=loss_configuration,
            scheduler_factor=arguments.scheduler_factor,
            scheduler_patience=arguments.scheduler_patience,
            scheduler_minimum_learning_rate=arguments.scheduler_minimum_learning_rate,
            hole_probability=arguments.hole_probability if arguments.training_holes else 0.0,
            hole_size=arguments.hole_size, hole_device=arguments.hole_device
        )
        configuration["selected_epoch"] = model.best_validation_epoch or arguments.epochs
        configuration["selected_validation_total"] = model.best_validation_total
        artifact = output / "wic_cvae.pt"
        save_run_artifact(
            artifact, model, transform, history, configuration, provenance,
            metadata["split_files"], artifact_type=WIC_CVAE_ARTIFACT_TYPE
        )
        (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
        evaluation = {}
        for name in ("validation", "test"):
            if loaders[name] is not None:
                evaluation[name] = _run_epoch(
                    model, loaders[name], optimizer=None, device=torch.device(device),
                    beta=arguments.beta, training=False,
                    loss_configuration=loss_configuration,
                    progress_description=f"{name} evaluation"
                )
        save_run_artifact(
            artifact, model, transform, history, configuration, provenance,
            metadata["split_files"], evaluation=evaluation,
            artifact_type=WIC_CVAE_ARTIFACT_TYPE
        )
        summary = {
            "artifact": artifact.name, "artifact_type": WIC_CVAE_ARTIFACT_TYPE,
            "dataset_backend": metadata["cache_format"], "cache_directory": str(cache),
            "cache_manifest_sha256": metadata["manifest_sha256"],
            "used_frames": metadata["used_frames"],
            "training_configuration": configuration,
            "held_out_evaluation": evaluation, "ensemble_generated": False,
            "interpretation": "Kp-conditional corrected-WIC prior; no conductance conversion or uncertainty calibration"
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(f"Saved {artifact}. WIC-CVAE training complete; fixed-Kp sampling is separate.", flush=True)
        return summary
    finally:
        for dataset in datasets.values():
            if dataset is not None:
                dataset.close()


#%% CLI

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--seed", type=int)
    for name, default in (("epochs", 500), ("batch-size", 264), ("latent-dim", 36),
                          ("spline-control-points", 30), ("spline-degree", 3),
                          ("scheduler-patience", 10), ("max-open-orbits", 64),
                          ("num-workers", 2), ("prefetch-factor", 2)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 1e-4), ("beta", 0.048),
                          ("scheduler-factor", 0.5),
                          ("scheduler-minimum-learning-rate", 0.0), ("cache-mib", 512)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--training-holes", action="store_true")
    parser.add_argument("--hole-probability", type=float, default=0.5)
    parser.add_argument("--hole-size", type=int, default=12)
    parser.add_argument("--hole-device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)
    if arguments.epochs < 1 or arguments.batch_size < 1 or arguments.latent_dim < 1:
        parser.error("epochs, batch size and latent dimension must be positive")
    if not np.isfinite(arguments.beta) or arguments.beta < 0:
        parser.error("beta must be finite and nonnegative")
    if arguments.num_workers < 0 or arguments.prefetch_factor < 1:
        parser.error("worker count must be nonnegative and prefetch factor positive")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
