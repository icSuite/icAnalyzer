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
    load_precipitation_cs_files,
    split_orbit_files,
)
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


def load_split(paths):
    return load_precipitation_cs_files(paths) if paths else None


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
    split = split_orbit_files(
        files,
        validation_fraction=arguments.validation_fraction,
        test_fraction=arguments.test_fraction,
        seed=arguments.seed,
    )
    train_product = load_split(split["train"])
    validation_product = load_split(split["validation"])
    test_product = load_split(split["test"])

    transform = fit_channel_transform(
        train_product["values"], train_product["valid"]
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
    )
    evaluation = {}
    if test_loader is not None:
        evaluation["test"] = evaluate_vae(
            model, test_loader, beta=arguments.beta, device=device
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

    decoded = sample_decoded_ensemble(
        model,
        transform,
        n_samples=arguments.samples,
        batch_size=arguments.sample_batch_size,
        device=device,
        seed=arguments.seed + 1,
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

    truth_index = 0
    reconstruction, metrics = sparse_validation(
        comparison_product["values"][truth_index, 0],
        conductance_statistics,
        arguments.observations,
        arguments.observation_error_std,
        arguments.seed + 2,
        arguments.observation_mode,
    )

    np.savez_compressed(
        output / "ensemble_products.npz",
        fuv_mean=fuv_statistics["mean"],
        fuv_anomaly_factor=fuv_statistics["anomaly_factor"],
        conductance_mean=conductance_statistics["mean"],
        conductance_anomaly_factor=conductance_statistics["anomaly_factor"],
        reconstruction_mean=reconstruction["mean"],
        reconstruction_standard_deviation=reconstruction["standard_deviation"],
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
            },
            indent=2,
        ) + "\n"
    )
    return metrics


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="orbit files, directories, or globs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-3)
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
