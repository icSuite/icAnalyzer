"""Run the corrected-WIC/SI12 VAE baseline from Product-2 CS files.

This script intentionally keeps the scientific stages visible: load and split
orbits, fit the transform, train, sample, propagate corrected WIC, construct
joint covariance factors, and test one sparse conductance reconstruction.
"""

#%% Imports

import argparse
import json
from pathlib import Path
import warnings

import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.wic_si12.data import (
    ChannelTransform,
    WICSI12ArrayDataset,
    fit_channel_transform,
    fit_precision_weighting,
    load_good_orbit_intervals,
    load_precipitation_cs_files,
    select_annotated_orbit_files,
    split_orbit_files,
)
from icanalyzer.wic_si12.comparison import load_paired_files, pair_orbit_files
from scripts.manual_masking.annotated_data import load_annotations, select_annotation_files, annotation_intervals, apply_annotations
from icanalyzer.wic_si12.count_likelihood import COUNT_UNCERTAINTY_LOSS_KIND, fit_count_uncertainty_configuration
from icanalyzer.wic_si12.ensemble import (
    correct_wic_roundoff,
    ensemble_statistics,
    sample_decoded_ensemble,
    sparse_ensemble_reconstruction,
)
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.physics import conductance_from_corrected_wic
from icanalyzer.wic_si12.workflow import (
    LEGACY_LOSS_CONFIGURATION,
    GAUSSIAN_UNCERTAINTY_LOSS_KIND,
    evaluate_vae,
    gaussian_uncertainty_configuration,
    load_run_artifact,
    save_run_artifact,
    set_random_seed,
    train_vae,
)


#%% Input discovery

def discover_files(inputs, product_name="precipitation_cs"):
    files = []
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            files.extend(sorted(path.rglob("or_*.nc")))
        elif path.is_file():
            files.append(path)
        else:
            files.extend(sorted(Path().glob(value)))
    files = sorted({path.resolve() for path in files})
    if not files:
        raise FileNotFoundError(f"no {product_name} orbit files were found")
    return files


def load_split(
    paths,
    intervals,
    minimum_sza,
    comparison_pairs=None,
    data_version="current",
):
    if comparison_pairs is not None:
        pairs_by_current = {
            Path(current).resolve(): (Path(current), Path(legacy))
            for current, legacy in comparison_pairs
        }
        pairs = [pairs_by_current[Path(path).resolve()] for path in paths]
        return (
            load_paired_files(
                pairs,
                data_version=data_version,
                wic_source_intervals=intervals,
                minimum_sza=minimum_sza,
            )
            if pairs else None
        )
    return (
        load_precipitation_cs_files(
            paths,
            wic_source_intervals=intervals,
            minimum_sza=minimum_sza,
        )
        if paths else None
    )


def make_loader(
    product, transform, batch_size, shuffle, maximum_missing_fraction
):
    if product is None:
        return None
    dataset = WICSI12ArrayDataset(
        product, transform,
        maximum_missing_fraction=maximum_missing_fraction
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def frame_selection(loader):
    if loader is None:
        return {"used": 0, "dropped": 0, "dropped_indices": []}
    dataset = loader.dataset
    return {
        "used": int(dataset.selected_indices.size),
        "dropped": int(dataset.dropped_indices.size),
        "dropped_indices": dataset.dropped_indices.tolist(),
        "maximum_missing_fraction": dataset.maximum_missing_fraction,
    }


def negative_input_diagnostics(*products):
    diagnostics = {
        "policy": "clip_to_zero",
        "wic_clipped_cells": 0,
        "si12_clipped_cells": 0,
        "finite_wic_input_cells": 0,
        "finite_si12_input_cells": 0,
        "wic_valid_input_cells": 0,
        "si12_valid_input_cells": 0,
        "wic_clipped_valid_cells": 0,
        "si12_clipped_valid_cells": 0,
    }
    for product in products:
        if product is None:
            continue
        for metadata in product["metadata"]:
            diagnostics["wic_clipped_cells"] += int(
                metadata.get("negative_wic_clipped_cells", 0)
            )
            diagnostics["si12_clipped_cells"] += int(
                metadata.get("negative_si12_clipped_cells", 0)
            )
            diagnostics["finite_wic_input_cells"] += int(
                metadata.get("finite_wic_input_cells", 0)
            )
            diagnostics["finite_si12_input_cells"] += int(
                metadata.get("finite_si12_input_cells", 0)
            )
            diagnostics["wic_valid_input_cells"] += int(
                metadata.get("wic_valid_input_cells", 0)
            )
            diagnostics["si12_valid_input_cells"] += int(
                metadata.get("si12_valid_input_cells", 0)
            )
            diagnostics["wic_clipped_valid_cells"] += int(
                metadata.get("negative_wic_clipped_valid_cells", 0)
            )
            diagnostics["si12_clipped_valid_cells"] += int(
                metadata.get("negative_si12_clipped_valid_cells", 0)
            )
    total = diagnostics["wic_clipped_cells"] + diagnostics["si12_clipped_cells"]
    diagnostics["total_clipped_cells"] = total
    for channel in ("wic", "si12"):
        finite = diagnostics[f"finite_{channel}_input_cells"]
        diagnostics[f"{channel}_clipped_fraction_of_finite"] = (
            diagnostics[f"{channel}_clipped_cells"] / finite if finite else 0.0
        )
        valid = diagnostics[f"{channel}_valid_input_cells"]
        diagnostics[f"{channel}_clipped_fraction_of_valid"] = (
            diagnostics[f"{channel}_clipped_valid_cells"] / valid
            if valid else 0.0
        )
    if total:
        warnings.warn(
            "upstream VAE inputs contained negative physical counts; "
            f"clipped {diagnostics['wic_clipped_cells']} WIC cells and "
            f"{diagnostics['si12_clipped_cells']} SI12 cells to zero "
            f"({diagnostics['si12_clipped_fraction_of_finite']:.3%} of finite SI12; "
            f"{diagnostics['si12_clipped_fraction_of_valid']:.3%} of valid inputs)",
            stacklevel=2,
        )
    return diagnostics


def distribution_diagnostics(product, generated):
    diagnostics = {}
    for channel, name in enumerate(("wic_corrected", "si12")):
        observed = product["values"][:, channel][product["valid"][:, channel]]
        sampled = generated[:, channel].reshape(-1)
        diagnostics[name] = {
            "observed_mean_counts": float(np.mean(observed)),
            "observed_standard_deviation_counts": float(np.std(observed)),
            "sampled_mean_counts": float(np.mean(sampled)),
            "sampled_standard_deviation_counts": float(np.std(sampled)),
        }
    return diagnostics


def validate_reused_artifact(artifact, split, arguments, train_product):
    """Reject reuse when the saved model does not match the requested data."""

    saved_split = artifact.get("split_files", {})
    for name, paths in split.items():
        expected = [str(Path(path).resolve()) for path in paths]
        saved = [str(Path(path).resolve()) for path in saved_split.get(name, [])]
        if saved != expected:
            raise ValueError(
                f"--reuse-artifact split {name!r} does not match the saved run"
            )

    configuration = artifact["training_configuration"]
    provenance = artifact["data_provenance"]
    saved_loss_configuration = configuration.get("loss_configuration", {})
    annotation_file = arguments.annotations or arguments.hybrid_annotations
    checks = {
        "latent_dim": (configuration.get("latent_dim"), arguments.latent_dim),
        "seed": (configuration.get("seed"), arguments.seed),
        "minimum_sza_degrees": (
            configuration.get("minimum_sza_degrees"), arguments.minimum_sza
        ),
        "comparison_data_version": (
            configuration.get("comparison_data_version"), arguments.data_version
        ),
        "manual_frame_intervals": (
            provenance.get("manual_frame_intervals"),
            None if arguments.all_frames or arguments.annotations else str(Path(arguments.segments).resolve()),
        ),
        "manual_annotations": (
            provenance.get("manual_annotations"),
            None if not annotation_file else load_annotations(annotation_file, arguments.segments if arguments.hybrid_annotations else None)["provenance"]
        ),
        "legacy_comparison_directory": (
            provenance.get("legacy_comparison_directory"),
            None if arguments.legacy_comparison is None
            else str(Path(arguments.legacy_comparison).resolve()),
        ),
        "loss_quality_weight_source": (
            provenance.get("loss_quality_weight_source"),
            train_product["metadata"][0]["quality_weight_source"],
        ),
        "upstream_quality_weight_interpretation": (
            provenance.get("upstream_quality_weight_interpretation"),
            train_product["metadata"][0]["upstream_quality_weight_interpretation"],
        ),
        "measurement_uncertainty_in_loss": (
            bool(saved_loss_configuration.get("measurement_uncertainty_in_loss")),
            bool(arguments.measurement_uncertainty or arguments.gaussian_uncertainty or arguments.count_uncertainty),
        ),
        "gaussian_uncertainty": (
            saved_loss_configuration.get("kind") == GAUSSIAN_UNCERTAINTY_LOSS_KIND,
            bool(arguments.gaussian_uncertainty)
        ),
        "count_uncertainty": (
            saved_loss_configuration.get("kind") == COUNT_UNCERTAINTY_LOSS_KIND,
            bool(arguments.count_uncertainty)
        ),
    }
    if arguments.gaussian_uncertainty:
        checks["model_error_std"] = (
            saved_loss_configuration.get("model_error_std"), list(arguments.model_error_std)
        )
    if arguments.count_uncertainty:
        checks["count_transform"] = (saved_loss_configuration.get("transform"), artifact["transform"])
        checks["count_model_error_fraction"] = (
            saved_loss_configuration.get("model_error_fraction"), list(arguments.count_model_error_fraction)
        )
        retained_training = WICSI12ArrayDataset(
            train_product, ChannelTransform.from_dict(artifact["transform"]),
            maximum_missing_fraction=arguments.maximum_missing_fraction
        )
        refitted = fit_count_uncertainty_configuration(retained_training, arguments.count_model_error_fraction)
        checks["count_channel_scales"] = (saved_loss_configuration.get("channel_scale_counts"), refitted["channel_scale_counts"])
    saved_holes = configuration.get("training_holes") or {}
    checks["training_holes"] = (bool(saved_holes), bool(arguments.training_holes))
    if arguments.training_holes:
        checks["hole_probability"] = (saved_holes.get("probability"), arguments.hole_probability)
        checks["hole_size"] = (saved_holes.get("size"), arguments.hole_size)
    mismatched = [
        name for name, (saved, requested) in checks.items()
        if saved != requested
    ]
    if mismatched:
        raise ValueError(
            "--reuse-artifact does not match requested settings: "
            + ", ".join(mismatched)
        )


#%% Sparse validation

def sparse_validation(
    truth_wic,
    conductance_statistics,
    n_observations,
    error_std,
    seed,
    observation_mode="joint",
):
    truth = conductance_from_corrected_wic(truth_wic)
    truth_state = np.stack((truth["H"], truth["P"]), axis=0)
    allowed = np.ones(truth_state.shape, dtype=bool)
    if observation_mode == "hall":
        allowed[1] = False
    elif observation_mode == "pedersen":
        allowed[0] = False
    elif observation_mode != "joint":
        raise ValueError(f"unknown observation mode {observation_mode!r}")
    valid_indices = np.flatnonzero(
        (np.isfinite(truth_state) & allowed).reshape(-1)
    )
    if valid_indices.size == 0:
        raise ValueError("validation field contains no finite conductance")

    generator = np.random.default_rng(seed)
    n_observations = min(int(n_observations), valid_indices.size)
    selected = np.sort(
        generator.choice(valid_indices, size=n_observations, replace=False)
    )
    observed = truth_state.reshape(-1)[selected]
    reconstruction = sparse_ensemble_reconstruction(
        conductance_statistics["mean"],
        conductance_statistics["anomaly_factor"],
        selected,
        observed,
        observation_error=float(error_std)**2,
    )
    reconstruction.update({
        "truth": truth_state,
        "observation_indices": selected,
        "observation_values": observed,
        "observation_error_variance": float(error_std)**2,
    })

    finite = np.isfinite(truth_state) & np.isfinite(reconstruction["mean"])
    rmse = np.sqrt(np.mean((truth_state[finite] - reconstruction["mean"][finite])**2))
    prior_rmse = np.sqrt(
        np.mean((truth_state[finite] - conductance_statistics["mean"][finite])**2)
    )
    return reconstruction, {
        "observation_count": int(n_observations),
        "observation_mode": observation_mode,
        "observation_error_std_siemens": float(error_std),
        "prior_rmse_siemens": float(prior_rmse),
        "posterior_rmse_siemens": float(rmse),
    }


#%% Main workflow

def run(arguments):
    files = discover_files(arguments.inputs)
    intervals = None
    annotation_file = arguments.annotations or arguments.hybrid_annotations
    annotations = load_annotations(annotation_file, arguments.segments if arguments.hybrid_annotations else None) if annotation_file else None
    if annotations is not None:
        files = select_annotation_files(files, annotations)
        intervals = annotation_intervals(annotations)
    elif not arguments.all_frames:
        intervals = load_good_orbit_intervals(arguments.segments)
        files = select_annotated_orbit_files(files, intervals)

    comparison_pairs = None
    if arguments.legacy_comparison is not None:
        comparison_pairs = pair_orbit_files(files, arguments.legacy_comparison)
        files = [current for current, _legacy in comparison_pairs]
    elif arguments.data_version == "legacy":
        raise ValueError("--data-version legacy requires --legacy-comparison")

    split = split_orbit_files(
        files,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
    )
    train_product = load_split(
        split["train"], intervals, arguments.minimum_sza,
        comparison_pairs, arguments.data_version
    )
    validation_product = load_split(
        split["validation"], intervals, arguments.minimum_sza,
        comparison_pairs, arguments.data_version
    )
    test_product = load_split(
        split["test"], intervals, arguments.minimum_sza,
        comparison_pairs, arguments.data_version
    )
    if annotations is not None:
        train_product = apply_annotations(train_product, annotations)
        validation_product = apply_annotations(validation_product, annotations)
        test_product = apply_annotations(test_product, annotations)
    input_clipping = negative_input_diagnostics(
        train_product, validation_product, test_product
    )

    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "wic_si12_vae.pt"

    if arguments.reuse_artifact:
        if not artifact_path.is_file():
            raise FileNotFoundError(
                f"--reuse-artifact requires an existing {artifact_path}"
            )
        model, transform, artifact = load_run_artifact(artifact_path, device=device)
        validate_reused_artifact(artifact, split, arguments, train_product)
        configuration = artifact["training_configuration"]
        provenance = artifact["data_provenance"]
        evaluation = artifact.get("evaluation", {})
    else:
        transform = fit_channel_transform(
            train_product["values"], train_product["valid"]
        )
        loss_configuration = dict(LEGACY_LOSS_CONFIGURATION)
        if arguments.gaussian_uncertainty:
            loss_configuration = gaussian_uncertainty_configuration(arguments.model_error_std)
        elif arguments.measurement_uncertainty:
            loss_configuration = fit_precision_weighting(
                train_product,
                transform,
                variance_floor=arguments.uncertainty_variance_floor,
                maximum_relative_precision=(
                    arguments.maximum_relative_precision
                ),
                maximum_scale_samples=arguments.precision_scale_samples,
            )
        train_loader = make_loader(
            train_product, transform, arguments.batch_size, shuffle=True,
            maximum_missing_fraction=arguments.maximum_missing_fraction
        )
        if arguments.count_uncertainty:
            loss_configuration = fit_count_uncertainty_configuration(train_loader.dataset, arguments.count_model_error_fraction)
        validation_loader = make_loader(
            validation_product, transform, arguments.batch_size, shuffle=False,
            maximum_missing_fraction=arguments.maximum_missing_fraction
        )
        test_loader = make_loader(
            test_product, transform, arguments.batch_size, shuffle=False,
            maximum_missing_fraction=arguments.maximum_missing_fraction
        )

        set_random_seed(arguments.seed)
        model = SplineWICSI12VAE(
            image_shape=train_product["values"].shape[-2:],
            latent_dim=arguments.latent_dim,
            spline_control_points=arguments.spline_control_points,
            spline_degree=arguments.spline_degree,
        )
        model, history = train_vae(
            model,
            train_loader,
            validation_loader=validation_loader,
            epochs=arguments.epochs,
            learning_rate=arguments.learning_rate,
            beta=arguments.beta,
            device=device,
            seed=arguments.seed,
            loss_configuration=loss_configuration,
            scheduler_factor=arguments.scheduler_factor,
            scheduler_patience=arguments.scheduler_patience,
            scheduler_minimum_learning_rate=arguments.scheduler_minimum_learning_rate,
            hole_probability=arguments.hole_probability if arguments.training_holes else 0.0,
            hole_size=arguments.hole_size,
        )
        evaluation = {}
        if test_loader is not None:
            evaluation["test"] = evaluate_vae(
                model,
                test_loader,
                beta=arguments.beta,
                device=device,
                loss_configuration=loss_configuration,
            )

        provenance = {
            "grid_id": train_product["metadata"][0]["grid_id"],
            "grid": train_product["grid"],
            "products": train_product["metadata"],
            "upstream_wic_is_proton_corrected": True,
            "proton_correction_reapplied": False,
            "upstream_proton_energy_model": "hardy",
            "electron_energy_keV": 2.0,
            "input_negative_clipping": input_clipping,
            "loss_quality_weight_source": train_product["metadata"][0][
                "quality_weight_source"
            ],
            "upstream_quality_weight_interpretation": train_product["metadata"][0][
                "upstream_quality_weight_interpretation"
            ],
            "minimum_sza_degrees": arguments.minimum_sza,
            "comparison_data_version": arguments.data_version,
            "legacy_comparison_directory": (
                None if arguments.legacy_comparison is None
                else str(Path(arguments.legacy_comparison).resolve())
            ),
            "paired_orbit_count": (
                None if comparison_pairs is None else len(comparison_pairs)
            ),
            "pairing_rule": (
                None if comparison_pairs is None
                else "same orbit filename and exact product time"
            ),
            "manual_frame_intervals": (
                None if intervals is None else str(Path(arguments.segments).resolve())
            ),
            "manual_annotations": None if annotations is None else annotations["provenance"],
            "annotation_selection": {
                name: product["annotation_selection"] for name, product in
                (("train", train_product), ("validation", validation_product), ("test", test_product)) if product is not None
            } if annotations is not None else None,
            "sza_excluded_files": {
                "train": train_product.get("excluded_files", []),
                "validation": (
                    [] if validation_product is None
                    else validation_product.get("excluded_files", [])
                ),
                "test": (
                    [] if test_product is None
                    else test_product.get("excluded_files", [])
                ),
            },
        }
        configuration = {
            "epochs": arguments.epochs,
            "batch_size": arguments.batch_size,
            "learning_rate": arguments.learning_rate,
            "beta": arguments.beta,
            "latent_dim": arguments.latent_dim,
            "model": "SplineWICSI12VAE",
            "spline_control_points": arguments.spline_control_points,
            "spline_degree": arguments.spline_degree,
            "seed": arguments.seed,
            "device": device,
            "optimizer": "Adam",
            "beta_schedule": "constant",
            "learning_rate_scheduler": "ReduceLROnPlateau",
            "scheduler_monitor": "train_total",
            "scheduler_factor": arguments.scheduler_factor,
            "scheduler_patience": arguments.scheduler_patience,
            "scheduler_minimum_learning_rate": (
                arguments.scheduler_minimum_learning_rate
            ),
            "loss_configuration": loss_configuration,
            "training_holes": {
                "probability": arguments.hole_probability,
                "size": arguments.hole_size,
                "seed": arguments.seed + 3,
                "center_policy": "uniform_original_valid_pixel",
                "target_policy": "original_observed_cells_retained",
                "applied_to": "training_only"
            } if arguments.training_holes else None,
            "maximum_missing_fraction": arguments.maximum_missing_fraction,
            "minimum_sza_degrees": arguments.minimum_sza,
            "comparison_data_version": arguments.data_version,
            "selected_epoch": (
                model.best_validation_epoch
                if model.best_validation_epoch is not None else arguments.epochs
            ),
            "selected_validation_total": model.best_validation_total,
            "frame_selection": {
                "train": frame_selection(train_loader),
                "validation": frame_selection(validation_loader),
                "test": frame_selection(test_loader),
            },
        }
        save_run_artifact(
            artifact_path,
            model,
            transform,
            history,
            configuration,
            provenance,
            split,
            evaluation=evaluation,
        )
        model, transform, _artifact = load_run_artifact(
            artifact_path, device=device
        )

    decoded, latent_samples = sample_decoded_ensemble(
        model,
        transform,
        n_samples=arguments.samples,
        batch_size=arguments.sample_batch_size,
        device=device,
        seed=arguments.seed + 1,
        return_latent=True,
    )
    decoded, wic_boundary_diagnostics = correct_wic_roundoff(decoded, transform)
    fuv_statistics = ensemble_statistics(decoded)
    comparison_product = test_product or validation_product or train_product
    distribution = distribution_diagnostics(comparison_product, decoded)

    propagated = conductance_from_corrected_wic(
        decoded[:, 0], include_intermediates=False
    )
    conductance_samples = np.stack((propagated["H"], propagated["P"]), axis=1)
    conductance_statistics = ensemble_statistics(conductance_samples)

    usable_truth = np.flatnonzero(
        np.any(comparison_product["valid"][:, 0], axis=(1, 2))
    )
    if usable_truth.size == 0:
        raise ValueError("comparison split has no valid corrected-WIC frame")
    truth_index = int(usable_truth[0])
    truth_wic = comparison_product["values"][truth_index, 0].copy()
    truth_wic[~comparison_product["valid"][truth_index, 0]] = np.nan
    reconstruction, metrics = sparse_validation(
        truth_wic,
        conductance_statistics,
        arguments.observations,
        arguments.observation_error_std,
        arguments.seed + 2,
        arguments.observation_mode,
    )

    np.savez_compressed(
        output / "ensemble_products.npz",
        latent_samples=latent_samples,
        fuv_mean=fuv_statistics["mean"],
        fuv_anomaly_factor=fuv_statistics["anomaly_factor"],
        conductance_mean=conductance_statistics["mean"],
        conductance_anomaly_factor=conductance_statistics["anomaly_factor"],
        reconstruction_truth=reconstruction["truth"],
        reconstruction_prior_mean=conductance_statistics["mean"],
        reconstruction_prior_standard_deviation=reconstruction[
            "prior_standard_deviation"
        ],
        reconstruction_mean=reconstruction["mean"],
        reconstruction_standard_deviation=reconstruction["standard_deviation"],
        observation_indices=reconstruction["observation_indices"],
        observation_values=reconstruction["observation_values"],
        observation_error_variance=reconstruction["observation_error_variance"],
    )
    (output / "summary.json").write_text(
        json.dumps(
            {
                "training_configuration": configuration,
                "split_files": {
                    name: [str(path) for path in paths]
                    for name, paths in split.items()
                },
                "sample_count": int(arguments.samples),
                "input_negative_clipping": input_clipping,
                "reused_training_artifact": bool(arguments.reuse_artifact),
                "wic_nonnegative_boundary": wic_boundary_diagnostics,
                "held_out_evaluation": evaluation,
                "field_distribution_diagnostics": distribution,
                "propagation_order": propagated["propagation_order"],
                "wic_response_counts_per_mw_m2": propagated["wic_response"],
                "sparse_validation": metrics,
                "sparse_truth_frame": {
                    "orbit": int(comparison_product["orbit"][truth_index]),
                    "time": str(comparison_product["time"][truth_index]),
                    "wic_source_index": int(
                        comparison_product["source_indices"]["wic"][truth_index]
                    ),
                },
            },
            indent=2,
        ) + "\n"
    )
    return metrics


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="orbit files, directories, or globs")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--reuse-artifact",
        action="store_true",
        help=(
            "reuse output/wic_si12_vae.pt and rerun sampling, propagation, "
            "covariance, and reconstruction without training again"
        ),
    )
    parser.add_argument(
        "--segments",
        default=str(
            Path(__file__).resolve().parents[3]
            / "data" / "good_orbit_segments.csv"
        ),
        help="manual inclusive WIC-frame intervals used by default",
    )
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--annotations", help="converted 46x46 JSON: completed good and masked partial frames; replaces CSV selection")
    selection.add_argument("--hybrid-annotations", help="converted 46x46 JSON overrides reviewed orbits; --segments CSV is fallback elsewhere")
    selection.add_argument(
        "--all-frames",
        "--no-csv-selection",
        dest="all_frames",
        action="store_true",
        help=(
            "bypass good_orbit_segments.csv; technical validity and any "
            "--minimum-sza threshold still apply"
        ),
    )
    parser.add_argument(
        "--legacy-comparison",
        "--legacy-reference",
        help=(
            "legacy precipitation_IR_hardy_weighted directory; restrict to "
            "same-named orbits and exact Product-2 timestamps"
        ),
    )
    parser.add_argument(
        "--data-version",
        choices=("current", "legacy"),
        default="current",
        help=(
            "train on the current Product-2 fields or paired legacy fields; "
            "legacy requires --legacy-comparison"
        ),
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--latent-dim", type=int, default=36)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument(
        "--training-holes", action="store_true",
        help="remove artificial input-only squares during training while retaining their observed targets"
    )
    parser.add_argument("--hole-probability", type=float, default=0.5, help="probability of one hole per training image")
    parser.add_argument("--hole-size", type=int, default=12, help="side length of the training hole in CS cells")
    uncertainty_options = parser.add_mutually_exclusive_group()
    uncertainty_options.add_argument(
        "--measurement-uncertainty",
        action="store_true",
        help=(
            "weight reconstruction residuals by propagated WIC/SI12 measurement "
            "precision; disabled by default"
        ),
    )
    uncertainty_options.add_argument(
        "--gaussian-uncertainty", action="store_true",
        help="use measurement variance plus fixed model-error variance in a reference-scaled Gaussian loss"
    )
    uncertainty_options.add_argument(
        "--count-uncertainty", action="store_true",
        help="opt-in count-space Gaussian likelihood using original count errors and censored zeros"
    )
    parser.add_argument(
        "--model-error-std", nargs=2, type=float, default=None,
        metavar=("WIC", "SI12"),
        help="fixed model-error standard deviations in normalized log10 units for --gaussian-uncertainty"
    )
    parser.add_argument(
        "--count-model-error-fraction", nargs=2, type=float, default=None,
        metavar=("WIC", "SI12"),
        help="fixed count-space discrepancy as fractions of training-fitted channel RMS counts; default 0.5 0.5"
    )
    parser.add_argument(
        "--uncertainty-variance-floor",
        type=float,
        default=None,
        help="minimum normalized-log10 measurement variance",
    )
    parser.add_argument(
        "--maximum-relative-precision",
        type=float,
        default=None,
        help="cap on precision relative to its fitted training-channel median",
    )
    parser.add_argument(
        "--precision-scale-samples",
        type=int,
        default=None,
        help="maximum sampled training cells per channel for fitting precision scale",
    )
    parser.add_argument("--spline-control-points", type=int, default=30)
    parser.add_argument("--spline-degree", type=int, default=3)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument(
        "--scheduler-minimum-learning-rate", type=float, default=0.0
    )
    parser.add_argument(
        "--maximum-missing-fraction",
        type=float,
        default=0.33,
        help="drop frames above this intrinsic paired-pixel missing fraction, before SZA/manual masks",
    )
    parser.add_argument(
        "--minimum-sza",
        type=float,
        help=(
            "retain only pixels at or above this solar zenith angle in degrees; "
            "for example, --minimum-sza 105"
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.15)
    parser.add_argument("--test-fraction", type=float, default=0.15)
    parser.add_argument("--samples", type=int, default=1000)
    parser.add_argument("--sample-batch-size", type=int, default=256)
    parser.add_argument("--observations", type=int, default=50)
    parser.add_argument("--observation-error-std", type=float, default=0.5)
    parser.add_argument(
        "--observation-mode",
        default="joint",
        choices=("hall", "pedersen", "joint"),
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    arguments = parser.parse_args(argv)
    if arguments.model_error_std is not None and not arguments.gaussian_uncertainty:
        parser.error("--model-error-std requires --gaussian-uncertainty")
    if arguments.count_model_error_fraction is not None and not arguments.count_uncertainty:
        parser.error("--count-model-error-fraction requires --count-uncertainty")
    precision_options = {"uncertainty_variance_floor": 1e-3, "maximum_relative_precision": 100.0, "precision_scale_samples": 2_000_000}
    for name, default in precision_options.items():
        if getattr(arguments, name) is not None and not arguments.measurement_uncertainty:
            parser.error("--" + name.replace("_", "-") + " requires --measurement-uncertainty")
        if getattr(arguments, name) is None:
            setattr(arguments, name, default)
    arguments.model_error_std = [0.5, 0.5] if arguments.model_error_std is None else arguments.model_error_std
    arguments.count_model_error_fraction = [0.5, 0.5] if arguments.count_model_error_fraction is None else arguments.count_model_error_fraction
    for name in ("model_error_std", "count_model_error_fraction"):
        values = np.asarray(getattr(arguments, name))
        if np.any(~np.isfinite(values)) or np.any(values <= 0):
            parser.error("--" + name.replace("_", "-") + " requires finite positive values")
    return arguments


if __name__ == "__main__":
    metrics = run(parse_arguments())
    print(json.dumps(metrics, indent=2))
