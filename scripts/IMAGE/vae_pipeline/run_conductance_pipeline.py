"""Train native conductance_cs H/P spline VAEs, optionally conditioned on Kp."""

#%% Imports: reuse the model, dataset, scheduler and loss, not FUV propagation

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from icanalyzer.conductance import CONDUCTANCE_ARTIFACT_TYPE, fit_kp_conditioning, load_conductance_cs_files
from icanalyzer.wic_si12.data import fit_channel_transform, fit_precision_weighting, load_good_orbit_intervals, select_annotated_orbit_files, split_orbit_files
from icanalyzer.wic_si12.ensemble import sample_decoded_ensemble, ensemble_statistics, sparse_ensemble_reconstruction
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import LEGACY_LOSS_CONFIGURATION, gaussian_uncertainty_configuration, train_vae, evaluate_vae, save_run_artifact, load_run_artifact, set_random_seed
from scripts.IMAGE.vae_pipeline.run_wic_si12_pipeline import discover_files, make_loader, frame_selection
from scripts.manual_masking.annotated_data import load_annotations, select_annotation_files, annotation_intervals, apply_annotations
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import plot_loss_history, plot_sample_mosaic, plot_latent_samples


#%% One held-out sparse illustration, using native truth rather than WIC

def sparse_validation(dataset, statistics, support, arguments):
    truth = dataset.values[0].copy()
    valid = np.broadcast_to(dataset.common_valid[0] & support, truth.shape).copy()
    allowed = valid.copy()
    if arguments.observation_mode == "hall":
        allowed[1] = False
    elif arguments.observation_mode == "pedersen":
        allowed[0] = False
    candidates = np.flatnonzero(allowed)
    if candidates.size == 0:
        raise ValueError("selected reference has no valid training-supported observations")
    generator = np.random.default_rng(arguments.seed + 2)
    indices = np.sort(generator.choice(candidates, size=min(arguments.observations, candidates.size), replace=False))
    observations = truth.ravel()[indices]
    posterior = sparse_ensemble_reconstruction(statistics["mean"], statistics["anomaly_factor"], indices, observations, observation_error=arguments.observation_error_std**2)
    score = valid.ravel().copy()
    score[indices] = False
    error = posterior["mean"].ravel() - truth.ravel()
    prior_error = statistics["mean"].ravel() - truth.ravel()
    metrics = {
        "observation_count": len(indices), "observation_mode": arguments.observation_mode,
        "observation_error_std_siemens": arguments.observation_error_std,
        "unobserved_state_cells_scored": int(score.sum()),
        "prior_rmse_siemens": float(np.sqrt(np.mean(prior_error[score]**2))) if score.any() else None,
        "posterior_rmse_siemens": float(np.sqrt(np.mean(error[score]**2))) if score.any() else None,
        "negative_posterior_cells": int(np.count_nonzero(posterior["mean"][valid] < 0)),
        "note": "single-frame physical-space illustration; exact reference subsamples, no injected noise; not a calibrated skill benchmark"
    }
    return {"reconstruction_truth": np.where(valid, truth, np.nan), "reconstruction_mean": posterior["mean"], "reconstruction_standard_deviation": posterior["standard_deviation"], "reconstruction_prior_mean": statistics["mean"], "reconstruction_prior_standard_deviation": posterior["prior_standard_deviation"], "observation_indices": indices, "observation_values": observations, "observation_error_variance": arguments.observation_error_std**2}, metrics


#%% Load native orbits and construct the learning problem

def run(arguments):
    output = Path(arguments.output).expanduser()
    if output.exists() and any(output.iterdir()) and not arguments.reuse_artifact:
        raise FileExistsError("choose an empty output directory or explicitly --reuse-artifact")
    files = discover_files(arguments.inputs, product_name="conductance_cs")
    intervals = None
    annotation_file = arguments.annotations or arguments.hybrid_annotations
    annotations = load_annotations(annotation_file, arguments.segments if arguments.hybrid_annotations else None) if annotation_file else None
    if annotations is not None:
        files = select_annotation_files(files, annotations)
        intervals = annotation_intervals(annotations)
    elif not arguments.all_frames:
        intervals = load_good_orbit_intervals(arguments.segments)
        files = select_annotated_orbit_files(files, intervals)
    split = split_orbit_files(files, arguments.validation_fraction, arguments.test_fraction, arguments.seed)
    kp_options = {"include_kp": True} if arguments.condition_on_kp else {}
    products = {name: load_conductance_cs_files(paths, wic_source_intervals=intervals, minimum_sza=arguments.minimum_sza, **kp_options) if paths else None for name, paths in split.items()}
    if annotations is not None:
        products = {name: apply_annotations(product, annotations) for name, product in products.items()}
    device = ("cuda" if torch.cuda.is_available() else "cpu") if arguments.device == "auto" else arguments.device
    artifact_path = output / "conductance_vae.pt"
    if arguments.reuse_artifact:
        model, transform, artifact = load_run_artifact(artifact_path, device=device)
        if artifact["artifact_type"] != CONDUCTANCE_ARTIFACT_TYPE:
            raise ValueError("reuse requires a native conductance artifact, not a FUV model")
        if (model.kp_conditioning is not None) != arguments.condition_on_kp:
            raise ValueError("reuse must match the saved model's Kp conditioning option")
    else:
        transform = fit_channel_transform(products["train"]["values"], products["train"]["valid"])
    loaders = {name: make_loader(product, transform, arguments.batch_size, name == "train", arguments.maximum_missing_fraction) for name, product in products.items()}
    training_support = np.any(loaders["train"].dataset.common_valid, axis=(0, 1))
    kp_conditioning = fit_kp_conditioning(loaders["train"].dataset.kp) if arguments.condition_on_kp else None
    loss_configuration = dict(LEGACY_LOSS_CONFIGURATION)
    if arguments.measurement_uncertainty:
        loss_configuration = fit_precision_weighting(products["train"], transform, arguments.uncertainty_variance_floor, arguments.maximum_relative_precision, arguments.precision_scale_samples)
    elif arguments.gaussian_uncertainty:
        loss_configuration = gaussian_uncertainty_configuration(arguments.model_error_std)
    configuration = {name: getattr(arguments, name) for name in ("epochs", "batch_size", "latent_dim", "learning_rate", "beta", "seed", "spline_control_points", "spline_degree", "scheduler_factor", "scheduler_patience", "scheduler_minimum_learning_rate", "maximum_missing_fraction")}
    configuration.update({
        "input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S",
        "model": "SplineWICSI12VAE", "device": device, "optimizer": "Adam", "beta_schedule": "constant",
        "learning_rate_scheduler": "ReduceLROnPlateau", "scheduler_monitor": "train_total",
        "minimum_sza_degrees": arguments.minimum_sza, "loss_configuration": loss_configuration,
        "training_holes": {"probability": arguments.hole_probability, "size": arguments.hole_size, "seed": arguments.seed + 3, "target_policy": "original_observed_cells_retained", "applied_to": "training_only"} if arguments.training_holes else None,
        "frame_selection": {name: frame_selection(loader) for name, loader in loaders.items()},
        "support_policy": "retained training common-valid support; exclude never-observed cells from sampling/covariance/reconstruction, no synthetic training pixels"
    })
    provenance = {
        "input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S",
        "grid_id": products["train"]["metadata"][0]["grid_id"], "grid": products["train"]["grid"],
        "products": {name: [] if product is None else product["metadata"] for name, product in products.items()},
        "manual_frame_intervals": None if intervals is None else str(Path(arguments.segments).resolve()),
        "manual_annotations": None if annotations is None else annotations["provenance"],
        "annotation_selection": {name: product["annotation_selection"] for name, product in products.items() if product is not None} if annotations is not None else None,
        "propagation_reapplied": False, "electron_energy_source": "upstream native conductance retrieval, not fixed 2 keV",
        "training_support": training_support, "unsupported_training_cells": int((~training_support).sum())
    }
    if kp_conditioning is not None:
        configuration["kp_conditioning"] = kp_conditioning
        provenance["retained_frame_kp"] = {name: loader.dataset.kp.copy() for name, loader in loaders.items() if loader is not None}
        provenance["kp_coverage"] = {name: {"frames": len(loader.dataset), "range": [float(loader.dataset.kp.min()), float(loader.dataset.kp.max())],
                                           "quantiles_p10_p50_p90": np.quantile(loader.dataset.kp, [0.1, 0.5, 0.9]).tolist()}
                                     for name, loader in loaders.items() if loader is not None}

    # Preserve the saved settings/history when only resampling a matched run.
    if arguments.reuse_artifact:
        if artifact["data_provenance"].get("manual_annotations") != provenance["manual_annotations"]:
            raise ValueError("reuse does not match saved annotations or original frame times")
        expected_split = {name: [str(Path(path).resolve()) for path in paths] for name, paths in split.items()}
        if artifact["split_files"] != expected_split or artifact["data_provenance"]["products"] != provenance["products"] or artifact["data_provenance"]["manual_frame_intervals"] != provenance["manual_frame_intervals"]:
            raise ValueError("reuse does not match saved native data or selection")
        ignored = ("epochs", "device", "selected_epoch", "selected_validation_total")
        if any(value != artifact["training_configuration"].get(name) for name, value in configuration.items() if name not in ignored):
            raise ValueError("reuse does not match saved training settings")
        configuration, provenance = artifact["training_configuration"], artifact["data_provenance"]
        history, evaluation = artifact["history"], artifact["evaluation"]
    else:
        set_random_seed(arguments.seed)
        model = SplineWICSI12VAE(image_shape=products["train"]["values"].shape[-2:], latent_dim=arguments.latent_dim, spline_control_points=arguments.spline_control_points, spline_degree=arguments.spline_degree, kp_conditioning=kp_conditioning)
        model, history = train_vae(
            model, loaders["train"], validation_loader=loaders["validation"],
            epochs=arguments.epochs, learning_rate=arguments.learning_rate, beta=arguments.beta,
            device=device, seed=arguments.seed, loss_configuration=loss_configuration,
            scheduler_factor=arguments.scheduler_factor, scheduler_patience=arguments.scheduler_patience,
            scheduler_minimum_learning_rate=arguments.scheduler_minimum_learning_rate,
            hole_probability=arguments.hole_probability if arguments.training_holes else 0,
            hole_size=arguments.hole_size
        )
        evaluation = {name: evaluate_vae(model, loader, beta=arguments.beta, device=device, loss_configuration=loss_configuration) for name, loader in loaders.items() if name != "train" and loader is not None}
        configuration["selected_epoch"] = model.best_validation_epoch or arguments.epochs
        configuration["selected_validation_total"] = model.best_validation_total
        output.mkdir(parents=True, exist_ok=True)
        save_run_artifact(artifact_path, model, transform, history, configuration, provenance, split, evaluation=evaluation, artifact_type=CONDUCTANCE_ARTIFACT_TYPE)
        model, transform, _ = load_run_artifact(artifact_path, device=device)

    # Sample H/P directly. Unsupported zeros are masked storage placeholders.
    sample_options = {"kp": arguments.sample_kp} if kp_conditioning is not None else {}
    decoded, latent = sample_decoded_ensemble(model, transform, arguments.samples, arguments.sample_batch_size, device, arguments.seed + 1, return_latent=True, support_mask=training_support, **sample_options)
    statistics = ensemble_statistics(decoded)
    reference_split = next(name for name in ("test", "validation", "train") if loaders[name] is not None)
    reference_dataset = loaders[reference_split].dataset
    reference_statistics = statistics
    reference_draws = arguments.samples
    if kp_conditioning is not None and float(reference_dataset.kp[0]) != arguments.sample_kp:
        # The sparse illustration uses the reference image's activity, not the generation request.
        reference_draws = min(4096, arguments.samples)
        reference_samples = sample_decoded_ensemble(model, transform, reference_draws, arguments.sample_batch_size, device, arguments.seed + 1, support_mask=training_support, kp=float(reference_dataset.kp[0]))
        reference_statistics = ensemble_statistics(reference_samples)
        del reference_samples
    reconstruction, metrics = sparse_validation(reference_dataset, reference_statistics, training_support, arguments)
    ensemble_conditions = {}
    if kp_conditioning is not None:
        ensemble_conditions = {"conductance_sample_kp": np.full(arguments.samples, arguments.sample_kp, dtype=np.float32), "reconstruction_kp": float(reference_dataset.kp[0])}
        metrics.update(prior_Kp=float(reference_dataset.kp[0]), prior_sample_count=reference_draws)
    np.savez_compressed(output / "ensemble_products.npz", latent_samples=latent, conductance_mean=statistics["mean"], conductance_anomaly_factor=statistics["anomaly_factor"], conductance_supported=training_support, **ensemble_conditions, **reconstruction)
    summary = {
        "input_product_type": "conductance_cs", "artifact_type": CONDUCTANCE_ARTIFACT_TYPE,
        "field_names": ["Hall", "Pedersen"], "field_units": "S", "propagation_reapplied": False,
        "sample_count": arguments.samples, "training_configuration": configuration,
        "split_files": {name: [str(path) for path in paths] for name, paths in split.items()},
        "held_out_evaluation": evaluation, "reused_training_artifact": arguments.reuse_artifact,
        "unsupported_training_cells": provenance["unsupported_training_cells"],
        "support_policy": configuration["support_policy"],
        "sparse_reference_split": reference_split, "sparse_validation": metrics,
        "uncertainty_note": "dH/dP retained; weighting is opt-in. No measurement-noise covariance is added to the latent ensemble; this is not a noise-deconvolution claim. Posterior uncertainty is not calibrated."
    }
    if kp_conditioning is not None:
        summary["kp_coverage"] = provenance["kp_coverage"]
        summary["generation_condition"] = {"Kp": arguments.sample_kp, "within_training_range": kp_conditioning["training_min"] <= arguments.sample_kp <= kp_conditioning["training_max"]}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    created = []
    plot_loss_history(history, figures, created)
    selected = np.linspace(0, arguments.samples - 1, min(5, arguments.samples), dtype=int)
    preview = decoded[selected].copy()
    preview[:, :, ~training_support] = np.nan
    plot_sample_mosaic(preview, selected, "conductance", figures, created)
    plot_latent_samples({"latent_samples": latent}, figures, created)
    return metrics


#%% Existing defaults; Kp conditioning, uncertainty and input holes are opt-in

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--reuse-artifact", action="store_true", help="resample the matching saved conductance model; do not retrain")
    parser.add_argument("--segments", default=str(Path(__file__).resolve().parents[3] / "data" / "good_orbit_segments.csv"))
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--all-frames", "--no-csv-selection", dest="all_frames", action="store_true")
    selection.add_argument("--annotations", help="converted 46x46 JSON: completed good and masked partial frames; replaces CSV selection")
    selection.add_argument("--hybrid-annotations", help="converted 46x46 JSON overrides reviewed orbits; --segments CSV is fallback elsewhere")
    for name, default in (("epochs", 1000), ("batch-size", 8), ("latent-dim", 36), ("spline-control-points", 30), ("spline-degree", 3), ("scheduler-patience", 10), ("samples", 1000), ("sample-batch-size", 256), ("observations", 50), ("seed", 1)):
        parser.add_argument("--" + name, type=int, default=default)
    for name, default in (("learning-rate", 1e-4), ("beta", 1.0), ("scheduler-factor", 0.5), ("scheduler-minimum-learning-rate", 0.0), ("maximum-missing-fraction", 0.33), ("validation-fraction", 0.15), ("test-fraction", 0.15), ("observation-error-std", 0.5)):
        parser.add_argument("--" + name, type=float, default=default)
    parser.add_argument("--minimum-sza", type=float, help="only possible if native product supplies SZA; current copied corpus does not")
    parser.add_argument("--observation-mode", choices=("hall", "pedersen", "joint"), default="joint")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="auto")
    parser.add_argument("--training-holes", action="store_true")
    parser.add_argument("--condition-on-kp", action="store_true", help="opt-in native conditional VAE; require frame Kp for encoder and decoder")
    parser.add_argument("--sample-kp", type=float, help="requested generation Kp, required with --condition-on-kp")
    parser.add_argument("--hole-probability", type=float, default=0.5)
    parser.add_argument("--hole-size", type=int, default=12)
    uncertainty = parser.add_mutually_exclusive_group()
    uncertainty.add_argument("--measurement-uncertainty", action="store_true", help="opt-in training-median relative precision from propagated dH/dP")
    uncertainty.add_argument("--gaussian-uncertainty", action="store_true", help="opt-in normalized-log Gaussian variance plus model error")
    parser.add_argument("--model-error-std", nargs=2, type=float, metavar=("HALL", "PEDERSEN"))
    parser.add_argument("--uncertainty-variance-floor", type=float)
    parser.add_argument("--maximum-relative-precision", type=float)
    parser.add_argument("--precision-scale-samples", type=int)
    arguments = parser.parse_args(argv)
    if arguments.condition_on_kp and arguments.sample_kp is None:
        parser.error("--condition-on-kp requires an explicit --sample-kp")
    if arguments.sample_kp is not None and (not arguments.condition_on_kp or not np.isfinite(arguments.sample_kp) or not 0 <= arguments.sample_kp <= 9):
        parser.error("--sample-kp requires --condition-on-kp and a finite value in 0–9")
    if arguments.model_error_std is not None and not arguments.gaussian_uncertainty:
        parser.error("--model-error-std requires --gaussian-uncertainty")
    arguments.model_error_std = arguments.model_error_std or [0.5, 0.5]
    for name, default in (("uncertainty_variance_floor", 1e-3), ("maximum_relative_precision", 100), ("precision_scale_samples", 2_000_000)):
        if getattr(arguments, name) is not None and not arguments.measurement_uncertainty:
            parser.error("--" + name.replace("_", "-") + " requires --measurement-uncertainty")
        if getattr(arguments, name) is None:
            setattr(arguments, name, default)
    if arguments.samples < 2 or arguments.observations <= 0 or not np.isfinite(arguments.observation_error_std) or arguments.observation_error_std <= 0:
        parser.error("require samples >= 2, observations > 0 and finite positive observation error STD")
    if not np.isfinite(arguments.beta) or arguments.beta < 0:
        parser.error("beta must be finite and nonnegative")
    return arguments


if __name__ == "__main__":
    print(json.dumps(run(parse_arguments()), indent=2))
