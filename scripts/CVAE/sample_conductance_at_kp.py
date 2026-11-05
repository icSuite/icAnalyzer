"""Generate paired Hall/Pedersen CVAE fields at selected Kp values."""

#%% Imports

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icanalyzer.cvae.artifacts import load_run_artifact
from icanalyzer.cvae.ensemble import sample_decoded_ensemble, ensemble_statistics


#%% Readable physical mean and quantile maps

def plot_fields(mean, quantiles, support, kp, output, limits):
    figure, axes = plt.subplots(2, 4, figsize=(13, 6), layout="constrained")
    fields = [np.where(support[None], mean, np.nan), *quantiles]
    for channel, name in enumerate(("Hall", "Pedersen")):
        for column, (field, label) in enumerate(zip(fields, ("mean", "10th percentile", "median", "90th percentile"))):
            values = field[channel]
            im = axes[channel, column].imshow(values, origin="lower", vmin=0, vmax=limits[channel], cmap="viridis")
            axes[channel, column].set(title=f"{name}: {label}", xlabel="CS column", ylabel="CS row")
            figure.colorbar(im, ax=axes[channel, column], shrink=0.7, label="S", extend="max" if np.nanmax(values) > limits[channel] else "neither")
    figure.suptitle(f"Native conductance model prior at Kp = {kp:g}; common latent seed")
    figure.savefig(output / "conductance_at_kp.png", dpi=160)
    figure.savefig(output / "conductance_at_kp.pdf")
    plt.close(figure)


#%% Validate the saved conditional model and generate one ensemble at a time

def run(arguments):
    values = np.asarray(arguments.kp, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values < 0) | (values > 9)):
        raise ValueError("requested Kp values must be finite and lie in [0, 9]")
    if len(np.unique(values)) != len(values):
        raise ValueError("requested Kp values must be distinct")
    limits = np.asarray([arguments.hall_vmax, arguments.pedersen_vmax], dtype=float)
    if arguments.draws < 2 or arguments.threads < 1 or not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("need at least two draws, positive threads and positive display limits")
    output = Path(arguments.output).expanduser()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError("choose an empty output directory; existing results are protected")
    torch.set_num_threads(arguments.threads)
    device = ("cuda" if torch.cuda.is_available() else "cpu") if arguments.device == "auto" else arguments.device
    run_directory = Path(arguments.run_directory).expanduser()
    checkpoints = [run_directory / name for name in ("conductance_vae.pt", "automatic_cvae.pt")]
    available = [path for path in checkpoints if path.is_file()]
    if len(available) != 1:
        raise ValueError("run directory must contain exactly one conductance_vae.pt or automatic_cvae.pt")
    artifact_path = available[0]
    model, transform, artifact = load_run_artifact(artifact_path, device=device)
    if artifact.get("artifact_type") != "conductance_vae_run_v1" or getattr(model, "kp_conditioning", None) is None:
        raise ValueError("requires a saved Kp-conditional native conductance VAE")
    conditioning = dict(model.kp_conditioning)
    provenance = artifact["data_provenance"]
    support = np.asarray(provenance["training_support"], dtype=bool)
    mlat, mlt = [np.asarray(provenance["grid"][name], dtype=float) for name in ("mlat", "mlt")]
    if support.shape != tuple(model.image_shape) or not support.any() or mlat.shape != support.shape or mlt.shape != support.shape:
        raise ValueError("saved learned support and MLAT/MLT must match the model grid")
    model_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    selections = []
    for index, kp in enumerate(values):
        kp = float(kp)
        within_range = conditioning["training_min"] <= kp <= conditioning["training_max"]
        if not within_range:
            warnings.warn(f"Kp {kp:g} is outside the training range; generated fields are extrapolation", RuntimeWarning, stacklevel=2)
        destination = output / f"kp_{index:02d}_{kp:g}"
        destination.mkdir()
        print(f"Kp {kp:g}: generating {arguments.draws} paired fields", flush=True)
        # Resetting the seed for each Kp preserves identical z vectors and pairing.
        samples, latent = sample_decoded_ensemble(model, transform, arguments.draws, device=device, seed=arguments.seed, return_latent=True, support_mask=support, kp=kp)
        if not np.isfinite(samples[:, :, support]).all():
            raise ValueError("decoder produced nonfinite conductance on learned support")
        samples[:, :, ~support] = 0
        quantiles = np.quantile(samples, [0.1, 0.5, 0.9], axis=0)
        quantiles[:, :, ~support] = np.nan
        statistics = ensemble_statistics(samples)
        np.savez_compressed(
            destination / "ensemble_products.npz", latent_samples=latent,
            conductance_mean=statistics["mean"], conductance_anomaly_factor=statistics["anomaly_factor"],
            conductance_supported=support, conductance_sample_kp=np.full(arguments.draws, kp),
            conductance_q10=quantiles[0], conductance_q50=quantiles[1], conductance_q90=quantiles[2],
            mlat=mlat, mlt=mlt
        )
        metadata = {
            "Kp": kp, "within_training_range": bool(within_range), "draws": arguments.draws,
            "seed": arguments.seed, "device": device, "model_sha256": model_sha,
            "model_file": str(artifact_path.resolve()), "kp_conditioning": conditioning,
            "grid_id": provenance["grid_id"], "supported_cells": int(support.sum()),
            "field_names": ["Hall", "Pedersen"], "field_units": "S", "directory": destination.name
        }
        metadata["interpretation"] = "Fixed-Kp model prior draws; no event encoding, retrieval-noise propagation or calibrated uncertainty claim. Unsupported mean/factor zeros are storage placeholders; quantiles are unavailable there."
        (destination / "sampling.json").write_text(json.dumps(metadata, indent=2) + "\n")
        if not getattr(arguments, "no_plots", False):
            plot_fields(statistics["mean"], quantiles, support, kp, destination, (arguments.hall_vmax, arguments.pedersen_vmax))
        selections.append(metadata)
        del samples, latent, statistics, quantiles
    if hashlib.sha256(artifact_path.read_bytes()).hexdigest() != model_sha:
        raise RuntimeError("source model changed during sampling")
    (output / "sampling_summary.json").write_text(json.dumps({"selections": selections, "comparison_policy": "identical latent seed at every Kp; fixed display limits; each quantile summarizes the model distribution at that Kp"}, indent=2) + "\n")
    return selections


#%% CLI: no training or original-run output changes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--kp", type=float, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-plots", action="store_true", help="export ensemble_products.npz and provenance only")
    parser.add_argument("--hall-vmax", type=float, default=50)
    parser.add_argument("--pedersen-vmax", type=float, default=10)
    run(parser.parse_args())
