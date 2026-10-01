"""Run the corrected-WIC/SI12 VAE baseline from Product-2 CS files.

This script intentionally keeps the scientific stages visible: load and split
orbits, fit the transform, train, sample, propagate corrected WIC, construct
joint covariance factors, and test one sparse conductance reconstruction.
"""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.wic_si12.data import (
    WICSI12ArrayDataset,
    fit_channel_transform,
    fit_precision_weighting,
    load_good_orbit_intervals,
    load_precipitation_cs_files,
    select_annotated_orbit_files,
    split_orbit_files,
)
from icanalyzer.wic_si12.comparison import load_paired_files, pair_orbit_files
from icanalyzer.wic_si12.ensemble import (
    ensemble_statistics,
    sample_decoded_ensemble,
    sparse_ensemble_reconstruction,
)
from icanalyzer.wic_si12.model import WICSI12VAE
from icanalyzer.wic_si12.physics import conductance_from_corrected_wic
from icanalyzer.wic_si12.workflow import (
    evaluate_vae,
    load_run_artifact,
    save_run_artifact,
    set_random_seed,
    train_vae,
)


#%% Input discovery

def discover_files(inputs):
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
        raise FileNotFoundError("no precipitation_cs orbit files were found")
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


def make_loader(product, transform, batch_size, shuffle):
    if product is None:
        return None
    dataset = WICSI12ArrayDataset(product, transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def frame_selection(loader):
    if loader is None:
        return {"used": 0, "dropped": 0, "dropped_indices": []}
    dataset = loader.dataset
    return {
        "used": int(dataset.selected_indices.size),
        "dropped": int(dataset.dropped_indices.size),
        "dropped_indices": dataset.dropped_indices.tolist(),
    }


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
    if not arguments.all_frames:
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

    transform = fit_channel_transform(
        train_product["values"], train_product["valid"]
    )
    uncertainty_weighting = fit_precision_weighting(
        train_product,
        transform,
        variance_floor=arguments.variance_floor,
        maximum_relative_precision=arguments.maximum_relative_precision,
        maximum_scale_samples=arguments.precision_scale_samples,
    )
    train_loader = make_loader(
        train_product, transform, arguments.batch_size, shuffle=True
    )
    validation_loader = make_loader(
        validation_product, transform, arguments.batch_size, shuffle=False
    )
    test_loader = make_loader(
        test_product, transform, arguments.batch_size, shuffle=False
    )

    device = arguments.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    set_random_seed(arguments.seed)
    model = WICSI12VAE(
        image_shape=train_product["values"].shape[-2:],
        latent_dim=arguments.latent_dim,
        wic_floor_transformed=float(-transform.center[0] / transform.spread[0]),
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
        loss_configuration=uncertainty_weighting,
    )
    evaluation = {}
    if test_loader is not None:
        evaluation["test"] = evaluate_vae(
            model,
            test_loader,
            beta=arguments.beta,
            device=device,
            loss_configuration=uncertainty_weighting,
        )

    output = Path(arguments.output)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {
        "grid_id": train_product["metadata"][0]["grid_id"],
        "grid": train_product["grid"],
        "products": train_product["metadata"],
        "upstream_wic_is_proton_corrected": True,
        "proton_correction_reapplied": False,
        "upstream_proton_energy_model": "hardy",
        "electron_energy_keV": 2.0,
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
            None if comparison_pairs is None else "same orbit filename and exact product time"
        ),
        "manual_frame_intervals": (
            None if intervals is None else str(Path(arguments.segments).resolve())
        ),
        "sza_excluded_files": {
            "train": train_product.get("excluded_files", []),
            "validation": (
                [] if validation_product is None
                else validation_product.get("excluded_files", [])
            ),
            "test": (
                [] if test_product is None else test_product.get("excluded_files", [])
            ),
        },
    }
    configuration = {
        "epochs": arguments.epochs,
        "batch_size": arguments.batch_size,
        "learning_rate": arguments.learning_rate,
        "beta": arguments.beta,
        "latent_dim": arguments.latent_dim,
        "seed": arguments.seed,
        "device": device,
        "optimizer": "Adam",
        "beta_schedule": "constant",
        "uncertainty_weighting": uncertainty_weighting,
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
        output / "wic_si12_vae.pt",
        model,
        transform,
        history,
        configuration,
        provenance,
        split,
        evaluation=evaluation,
    )
    model, transform, _artifact = load_run_artifact(
        output / "wic_si12_vae.pt", device=device
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
    if np.any(decoded[:, 0] < 0):
        fraction = float(np.mean(decoded[:, 0] < 0))
        raise ValueError(
            "decoded corrected WIC contains negative values "
            f"({fraction:.3%} of cells); do not clip silently"
        )
    fuv_statistics = ensemble_statistics(decoded)
    comparison_product = test_product or validation_product or train_product
    distribution = distribution_diagnostics(comparison_product, decoded)

    propagated = conductance_from_corrected_wic(decoded[:, 0])
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
        "--segments",
        default=str(
            Path(__file__).resolve().parents[3]
            / "data" / "good_orbit_segments.csv"
        ),
        help="manual inclusive WIC-frame intervals used by default",
    )
    parser.add_argument(
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
    parser.add_argument(
        "--variance-floor",
        type=float,
        default=1e-3,
        help="minimum transformed measurement variance used by the VAE loss",
    )
    parser.add_argument(
        "--maximum-relative-precision",
        type=float,
        default=100.0,
        help="cap on precision divided by its fitted training-channel scale",
    )
    parser.add_argument(
        "--precision-scale-samples",
        type=int,
        default=2_000_000,
        help="maximum candidate grid cells per channel used to fit median precision",
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
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args(argv)


if __name__ == "__main__":
    metrics = run(parse_arguments())
    print(json.dumps(metrics, indent=2))
