"""Train a conditional spline VAE from a WIC or paired WIC/SI12 cache.

One-field caches retain the historical quality-weighted loss. Paired caches
use equal per-channel reconstruction terms with separate validity and quality.
Measurement uncertainty remains outside the baseline loss in both cases.
"""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.cvae.artifacts import WIC_CVAE_ARTIFACT_TYPE, save_run_artifact
from icanalyzer.cvae.cache import (
    AutomaticFieldDataset,
    OrbitBatchSampler,
    WIC_SI12_CACHE_FORMATS,
)
from icanalyzer.cvae.conditioning import (
    CONDITIONING_PROFILES,
    align_conditioning_features,
    fit_conditioning_normalization,
    resolve_conditioning_features,
)
from icanalyzer.cvae.data import fit_kp_conditioning
from icanalyzer.cvae.hierarchical import HierarchicalSplineConditionalVAE
from icanalyzer.cvae.loss import (
    CHANNEL_BALANCED_LOSS_CONFIGURATION,
    LEGACY_LOSS_CONFIGURATION,
)
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.prior import disable_optional_native_bmm_override
from icanalyzer.cvae.training import (
    _run_epoch,
    resolve_hole_device,
    set_random_seed,
    train_vae,
    training_device
)
from scripts.CVAE.prepare_wic_cvae_cache import WIC_FIELD_MODES, read_wic_cache
from scripts.CVAE.prepare_wic_si12_cvae_cache import (
    FIELD_NAMES as WIC_SI12_FIELD_NAMES,
    read_wic_si12_cache,
)

REFERENCE_HOLE_SHAPE = (46, 46)
REFERENCE_HOLE_SIZE = 12
TRAINING_STATE_FORMAT = "wic_cvae_training_state_v1"


def resolve_training_hole_size(requested_size, image_shape):
    """Preserve the old hole's physical scale unless native pixels are explicit."""

    if requested_size is not None:
        if not isinstance(requested_size, (int, np.integer)) or requested_size < 1:
            raise ValueError("hole size must be a positive integer")
        return int(requested_size)
    linear_scale = np.sqrt(
        np.prod(image_shape) / np.prod(REFERENCE_HOLE_SHAPE)
    )
    return max(1, int(round(REFERENCE_HOLE_SIZE * linear_scale)))


#%% Training

def run(arguments):
    cache = arguments.cache.expanduser().resolve()
    output = arguments.output.expanduser().resolve()
    device = training_device(arguments.device)
    if arguments.latent_prior == "conditional-mixture":
        if arguments.architecture != "hierarchical":
            raise ValueError(
                "conditional mixture prior requires --architecture hierarchical"
            )
        if arguments.conditioning is not None:
            raise ValueError(
                "conditional mixture prior currently requires scalar Kp conditioning"
            )
        if disable_optional_native_bmm_override(
            arguments.prior_covariance_rank, device
        ):
            print(
                "Disabled PyTorch's optional Triton BMM override for the "
                "low-rank prior; using standard ATen CUDA BMM.",
                flush=True,
            )
    progress_file = output / "training_state.pt"
    if output.exists() and not arguments.resume:
        raise FileExistsError("choose a new WIC-CVAE run directory; existing runs are never overwritten")
    if arguments.resume and not progress_file.is_file():
        raise FileNotFoundError(
            f"cannot resume without the saved training state: {progress_file}"
        )
    cache_format = json.loads((cache / "manifest.json").read_text()).get(
        "cache_format"
    )
    metadata = (
        read_wic_si12_cache(cache)
        if cache_format in WIC_SI12_CACHE_FORMATS
        else read_wic_cache(cache)
    )
    supported_fields = {
        contract["field_name"] for contract in WIC_FIELD_MODES.values()
    }
    field_names = metadata.get("field_names")
    one_field = (
        isinstance(field_names, list) and len(field_names) == 1
        and field_names[0] in supported_fields
    )
    paired = field_names == list(WIC_SI12_FIELD_NAMES)
    if (
        not (one_field or paired)
        or metadata.get("input_product_type") != "precipitation_cs"
    ):
        raise ValueError("expected a supported WIC or paired WIC/SI12 cache")
    image_shape = tuple(metadata.get("image_shape", (46, 46)))
    if len(image_shape) != 2 or any(int(value) < 1 for value in image_shape):
        raise ValueError("cache image_shape must contain two positive dimensions")
    proton_corrected = paired or field_names[0] == "wic_corrected"
    seed = metadata["seed"] if arguments.seed is None else arguments.seed
    hole_device = resolve_hole_device(arguments.hole_device, device)
    hole_size = resolve_training_hole_size(arguments.hole_size, image_shape)
    if hole_size >= min(image_shape):
        raise ValueError("resolved training hole must be smaller than the image")

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
        if name == "train" and arguments.batch_order == "orbit":
            batch_sampler = OrbitBatchSampler(
                dataset, arguments.batch_size, seed=seed
            )
            loaders[name] = DataLoader(
                dataset, batch_sampler=batch_sampler, **options
            )
        else:
            loaders[name] = DataLoader(
                dataset, batch_size=arguments.batch_size,
                shuffle=name == "train", **options
            )

    transform = datasets["train"].transform
    kp_conditioning = None
    named_conditioning = None
    conditioning_evidence = None
    condition_dropped = {name: 0 for name in datasets}
    if arguments.conditioning is None:
        kp_conditioning = fit_kp_conditioning(datasets["train"].kp)
    else:
        feature_names = resolve_conditioning_features(
            arguments.conditioning_profile
        )
        conditioning_directory = arguments.conditioning.expanduser().resolve()
        manifest = None
        manifest_sha256 = None
        for name, dataset in datasets.items():
            if dataset is None:
                continue
            values, current_manifest, current_sha256 = align_conditioning_features(
                conditioning_directory,
                dataset.frame_orbit,
                dataset.source_frames,
                dataset.frame_time,
                dataset.kp,
                feature_names=feature_names,
            )
            if manifest_sha256 is not None and current_sha256 != manifest_sha256:
                raise RuntimeError("conditioning manifest changed during cache alignment")
            manifest = current_manifest
            manifest_sha256 = current_sha256
            condition_dropped[name] = dataset.attach_conditioning(values)
        named_conditioning = fit_conditioning_normalization(
            datasets["train"].conditioning,
            embedding_dim=arguments.conditioning_embedding_dim,
            feature_names=feature_names,
        )
        conditioning_evidence = {
            "directory": str(conditioning_directory),
            "manifest_sha256": manifest_sha256,
            "format": manifest["format"],
            "profile": arguments.conditioning_profile,
            "feature_names": list(feature_names),
            "missing_condition_frames_dropped": condition_dropped,
            "alignment": "exact orbit, Product-2 source-frame index and timestamp",
        }
    loss_configuration = dict(
        CHANNEL_BALANCED_LOSS_CONFIGURATION
        if paired else LEGACY_LOSS_CONFIGURATION
    )
    mask_source = metadata.get("mask_source")
    if mask_source is None:
        mask_source = {"kind": "automatic", **metadata["automatic_masks"]}
    configuration = {name: getattr(arguments, name) for name in (
        "epochs", "batch_size", "latent_dim", "learning_rate", "beta",
        "latent_samples", "conditioning_embedding_dim", "conditioning_profile",
        "spline_control_points", "spline_degree", "scheduler_factor",
        "scheduler_patience", "scheduler_minimum_learning_rate",
        "architecture", "kl_warmup_epochs", "batch_order", "latent_prior",
        "prior_components", "prior_covariance_rank", "prior_hidden_dim",
        "prior_minimum_logvar", "prior_maximum_logvar",
        "prior_maximum_factor",
    )}
    configuration.update({
        "seed": seed, "device": device, "input_product_type": "precipitation_cs",
        "image_shape": list(image_shape),
        "field_names": field_names, "field_units": metadata["field_units"],
        "wic_field": metadata.get("wic_field"),
        "dataset_backend": metadata["cache_format"],
        "maximum_missing_fraction": metadata["maximum_missing_fraction"],
        "kp_conditioning": kp_conditioning,
        "conditioning": named_conditioning,
        "loss_configuration": loss_configuration,
        "optimizer": "Adam",
        "beta_schedule": (
            "constant" if arguments.kl_warmup_epochs == 0
            else "linear_epoch_1_zero_to_configured_beta"
        ),
        "learning_rate_scheduler": "ReduceLROnPlateau", "scheduler_monitor": "train_total",
        "cache_mib_per_split": arguments.cache_mib,
        "max_open_orbits_per_worker": arguments.max_open_orbits,
        "num_workers": arguments.num_workers, "prefetch_factor": arguments.prefetch_factor,
        "persistent_workers": arguments.num_workers > 0,
        "pin_memory": device == "cuda",
        "uncertainty_arrays_cached": bool(
            metadata.get("uncertainty_arrays_cached", True)
        ),
        "uncertainty_arrays_loaded": False,
        "training_holes": {
            "probability": arguments.hole_probability, "size": hole_size,
            "requested_native_pixel_size": arguments.hole_size,
            "default_reference_size": REFERENCE_HOLE_SIZE,
            "default_reference_shape": list(REFERENCE_HOLE_SHAPE),
            "seed": seed + 3, "device": hole_device,
            "target_policy": "original observed cells retained"
        } if arguments.training_holes else None,
        "frame_selection": {
            name: {
                "cache_eligible": metadata["used_frames"][name],
                "missing_condition_dropped": condition_dropped[name],
                "used": 0 if dataset is None else len(dataset),
            }
            for name, dataset in datasets.items()
        },
        "support_policy": metadata.get(
            "mask_policy",
            f"{mask_source['kind']} mask AND native finite WIC/quality/coverage support"
        )
    })
    support = np.load(cache / "training_support.npy", allow_pickle=False)
    expected_support_shape = (
        (2, *image_shape) if paired else image_shape
    )
    if support.shape != expected_support_shape or not support.any():
        raise ValueError("training support does not match the modeled fields")
    provenance = {
        "dataset_backend": metadata["cache_format"], "cache_directory": str(cache),
        "cache_manifest_sha256": metadata["manifest_sha256"],
        "mask_source": mask_source,
        "input_product_type": "precipitation_cs", "grid_id": metadata["grid_id"],
        "grid": metadata["grid"], "field_names": field_names,
        "field_units": metadata["field_units"], "training_support": support,
        "unsupported_training_cells": int((~support).sum()),
        "sensor_support_overlap": metadata.get("sensor_support_overlap"),
        "products": {name: [row["product"] for row in metadata["orbits"] if row["split"] == name] for name in datasets},
        "retained_frame_kp": {name: dataset.kp.copy() for name, dataset in datasets.items() if dataset is not None},
        "conditioning": conditioning_evidence,
        "conductance_conversion": (
            "not applied during training; corrected WIC may be converted during sampling"
            if proton_corrected else
            "disabled for this artifact because pre-proton WIC includes proton emissions"
        )
    }
    used_frames = {
        name: 0 if dataset is None else len(dataset)
        for name, dataset in datasets.items()
    }
    condition_label = (
        "Kp" if named_conditioning is None
        else f"{arguments.conditioning_profile} named"
    )
    print(
        f"{'WIC/SI12' if paired else 'WIC'}-CVAE frames: {used_frames}. "
        f"Conditioning: {condition_label}. "
        f"Device: {device}. No ensembles will be generated.", flush=True
    )

    set_random_seed(seed)
    model_arguments = {
        "image_shape": image_shape,
        "spline_control_points": arguments.spline_control_points,
        "spline_degree": arguments.spline_degree,
        "kp_conditioning": kp_conditioning,
        "conditioning": named_conditioning,
        "quality_weight_channels": metadata.get("quality_channel_count", 1),
        "encoder_mask_policy": "union" if paired else "intersection",
        "field_names": tuple(field_names)
    }
    if arguments.architecture == "hierarchical":
        latent_prior = None
        if arguments.latent_prior == "conditional-mixture":
            latent_prior = {
                "kind": "conditional_gaussian_mixture_v1",
                "components": arguments.prior_components,
                "hidden_dim": arguments.prior_hidden_dim,
                "minimum_logvar": arguments.prior_minimum_logvar,
                "maximum_logvar": arguments.prior_maximum_logvar,
                "covariance_rank": arguments.prior_covariance_rank,
                "maximum_factor": arguments.prior_maximum_factor,
            }
        model = HierarchicalSplineConditionalVAE(
            global_latent_dim=arguments.latent_dim,
            spatial_latent_channels=arguments.spatial_latent_channels,
            latent_prior=latent_prior,
            magnetic_latitude=np.asarray(metadata["grid"]["mlat"]),
            **model_arguments
        )
        configuration.update({
            "global_latent_dim": model.global_latent_dim,
            "spatial_latent_channels": model.spatial_latent_channels,
            "spatial_latent_shape": list(model.spatial_latent_shape),
            "total_latent_dim": model.latent_dim,
        })
    else:
        model = SplineConditionalVAE(
            latent_dim=arguments.latent_dim, **model_arguments
        )
    output.mkdir(parents=True, exist_ok=arguments.resume)
    if not arguments.resume:
        (output / "cache_manifest.json").write_bytes(
            (cache / "manifest.json").read_bytes()
        )
        if arguments.conditioning is not None:
            (output / "conditioning_manifest.json").write_bytes(
                (arguments.conditioning.expanduser().resolve() / "manifest.json").read_bytes()
            )
    resume_state = None
    training_resumes = 0
    if arguments.resume:
        saved = torch.load(progress_file, map_location="cpu", weights_only=False)
        if saved.get("format") != TRAINING_STATE_FORMAT:
            raise ValueError("unsupported CVAE training-state format")
        if (
            saved.get("cache_manifest_sha256") != metadata["manifest_sha256"]
            or saved.get("training_configuration") != configuration
            or saved.get("model_configuration") != model.configuration()
        ):
            raise ValueError(
                "resume arguments, cache or model differ from training_state.pt"
            )
        resume_state = saved["training_state"]
        training_resumes = int(saved.get("training_resumes", 0)) + 1

    def save_training_state(training_state):
        temporary = progress_file.with_suffix(".pt.tmp")
        torch.save({
            "format": TRAINING_STATE_FORMAT,
            "cache_manifest_sha256": metadata["manifest_sha256"],
            "training_configuration": configuration,
            "model_configuration": model.configuration(),
            "training_resumes": training_resumes,
            "training_state": training_state,
        }, temporary)
        temporary.replace(progress_file)

    try:
        if arguments.batch_probe_batches:
            set_random_seed(seed)
            model = model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=arguments.learning_rate
            )
            probe_holes = {}
            if arguments.training_holes:
                probe_holes = {
                    "hole_probability": arguments.hole_probability,
                    "hole_size": hole_size,
                    "hole_device": hole_device,
                    "hole_generator": torch.Generator(
                        device=hole_device
                    ).manual_seed(int(seed) + 3),
                }
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            batch_sampler = getattr(loaders["train"], "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(1)
            metrics = _run_epoch(
                model, loaders["train"], optimizer=optimizer,
                device=torch.device(device), beta=arguments.beta,
                training=True, loss_configuration=loss_configuration,
                latent_samples=arguments.latent_samples,
                progress_description="GPU batch probe",
                maximum_batches=arguments.batch_probe_batches,
                **probe_holes,
            )
            if device == "cuda":
                torch.cuda.synchronize()
            probe = {
                "cache_manifest_sha256": metadata["manifest_sha256"],
                "image_shape": list(image_shape),
                "architecture": arguments.architecture,
                "batch_order": arguments.batch_order,
                "batch_size": arguments.batch_size,
                "batches": arguments.batch_probe_batches,
                "metrics": metrics,
                "cuda_peak_allocated_gib": (
                    torch.cuda.max_memory_allocated() / 1024**3
                    if device == "cuda" else None
                ),
                "cuda_peak_reserved_gib": (
                    torch.cuda.max_memory_reserved() / 1024**3
                    if device == "cuda" else None
                ),
            }
            (output / "batch_probe.json").write_text(
                json.dumps(probe, indent=2) + "\n"
            )
            print(json.dumps(probe, indent=2), flush=True)
            return probe

        model, history = train_vae(
            model, loaders["train"], validation_loader=loaders["validation"],
            epochs=arguments.epochs, learning_rate=arguments.learning_rate,
            beta=arguments.beta, device=device, seed=seed,
            loss_configuration=loss_configuration,
            scheduler_factor=arguments.scheduler_factor,
            scheduler_patience=arguments.scheduler_patience,
            scheduler_minimum_learning_rate=arguments.scheduler_minimum_learning_rate,
            hole_probability=arguments.hole_probability if arguments.training_holes else 0.0,
            hole_size=hole_size, hole_device=arguments.hole_device,
            latent_samples=arguments.latent_samples,
            kl_warmup_epochs=arguments.kl_warmup_epochs,
            resume_state=resume_state,
            checkpoint_callback=save_training_state,
        )
        configuration["training_resumes"] = training_resumes
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
                    latent_samples=arguments.latent_samples,
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
            "used_frames": used_frames,
            "training_configuration": configuration,
            "held_out_evaluation": evaluation, "ensemble_generated": False,
            "interpretation": (
                f"{condition_label}-conditional {'/'.join(field_names)} prior; "
                "no conductance conversion or uncertainty calibration"
            )
        }
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(
            f"Saved {artifact}. {'WIC/SI12' if paired else 'WIC'}-CVAE "
            "training complete; sampling is separate.", flush=True
        )
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
    parser.add_argument(
        "--conditioning", type=Path,
        help=(
            "prepared IMAGE/SuperMAG conditioning directory; when omitted, "
            "retain the historical scalar-Kp model"
        )
    )
    parser.add_argument(
        "--conditioning-profile", choices=tuple(CONDITIONING_PROFILES),
        default="all11",
        help="named feature profile used with --conditioning (default: all11)"
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument(
        "--architecture", choices=("legacy", "hierarchical"), default="legacy",
        help="decoder/latent architecture; legacy remains the compatibility default"
    )
    parser.add_argument(
        "--batch-order", choices=("frame", "orbit"), default="frame",
        help="shuffle individual frames or orbit-local batches (default: frame)"
    )
    parser.add_argument(
        "--spatial-latent-channels", type=int, default=8,
        help="6x6 spatial latent channels for the hierarchical architecture"
    )
    parser.add_argument(
        "--latent-prior",
        choices=("standard-normal", "conditional-mixture"),
        default="standard-normal",
        help=(
            "fixed standard normal or jointly learned Kp-conditioned Gaussian "
            "mixture (default: standard-normal)"
        ),
    )
    parser.add_argument("--prior-components", type=int, default=6)
    parser.add_argument("--prior-covariance-rank", type=int, default=8)
    parser.add_argument("--prior-hidden-dim", type=int, default=64)
    parser.add_argument("--prior-minimum-logvar", type=float, default=-8.0)
    parser.add_argument("--prior-maximum-logvar", type=float, default=4.0)
    parser.add_argument("--prior-maximum-factor", type=float, default=2.0)
    parser.add_argument("--seed", type=int)
    for name, default in (("epochs", 500), ("batch-size", 264), ("latent-dim", 36),
                          ("latent-samples", 1),
                          ("conditioning-embedding-dim", 32),
                          ("spline-control-points", 30), ("spline-degree", 3),
                          ("scheduler-patience", 10), ("max-open-orbits", 64),
                          ("num-workers", 2), ("prefetch-factor", 2),
                          ("kl-warmup-epochs", 0)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 1e-4), ("beta", 0.048),
                          ("scheduler-factor", 0.5),
                          ("scheduler-minimum-learning-rate", 0.0), ("cache-mib", 512)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--training-holes", action="store_true")
    parser.add_argument(
        "--resume", action="store_true",
        help="resume this output directory from training_state.pt"
    )
    parser.add_argument(
        "--batch-probe-batches", type=int, default=0,
        help="run only this many training batches and report peak GPU memory"
    )
    parser.add_argument("--hole-probability", type=float, default=0.5)
    parser.add_argument(
        "--hole-size", type=int,
        help=(
            "native-grid pixels; default preserves the historical 12/46 "
            "linear scale (48 pixels on a 184x184 grid)"
        )
    )
    parser.add_argument("--hole-device", choices=("auto", "cpu", "cuda"), default="auto")
    arguments = parser.parse_args(argv)
    if (arguments.epochs < 1 or arguments.batch_size < 1 or
            arguments.latent_dim < 1 or arguments.latent_samples < 1 or
            arguments.conditioning_embedding_dim < 1):
        parser.error(
            "epochs, batch size, latent dimensions and latent samples must be positive"
        )
    if not np.isfinite(arguments.beta) or arguments.beta < 0:
        parser.error("beta must be finite and nonnegative")
    if arguments.num_workers < 0 or arguments.prefetch_factor < 1:
        parser.error("worker count must be nonnegative and prefetch factor positive")
    if arguments.spatial_latent_channels < 0:
        parser.error("--spatial-latent-channels must be nonnegative")
    if (
        arguments.prior_components < 1 or arguments.prior_hidden_dim < 1
        or arguments.prior_covariance_rank < 0
    ):
        parser.error("prior components/hidden dimension must be positive and rank nonnegative")
    if (
        not np.isfinite(arguments.prior_minimum_logvar)
        or not np.isfinite(arguments.prior_maximum_logvar)
        or arguments.prior_minimum_logvar >= arguments.prior_maximum_logvar
        or not np.isfinite(arguments.prior_maximum_factor)
        or arguments.prior_maximum_factor <= 0
    ):
        parser.error("prior variance/factor bounds are invalid")
    if arguments.latent_prior == "conditional-mixture":
        if arguments.architecture != "hierarchical":
            parser.error("--latent-prior conditional-mixture requires --architecture hierarchical")
        if arguments.conditioning is not None:
            parser.error("--latent-prior conditional-mixture currently requires scalar Kp conditioning")
    if arguments.kl_warmup_epochs < 0:
        parser.error("--kl-warmup-epochs must be nonnegative")
    if arguments.batch_probe_batches < 0:
        parser.error("--batch-probe-batches must be nonnegative")
    if arguments.resume and arguments.batch_probe_batches:
        parser.error("--resume cannot be combined with --batch-probe-batches")
    if arguments.hole_size is not None and arguments.hole_size < 1:
        parser.error("--hole-size must be positive")
    if arguments.conditioning is None and arguments.conditioning_profile != "all11":
        parser.error("--conditioning-profile requires --conditioning")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
