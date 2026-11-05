"""Figures 5a/b: reconstruct an entirely missing sensor using joint log PCA."""

#%% Imports

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from scripts.IMAGE.vae_pipeline.plot_wic_si12_reconstruction import log_prior, make_figure, reconstruct
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset
from scripts.IMAGE.vae_pipeline.wic_si12_coefficient_prior import coefficient_prior
from scripts.IMAGE.vae_pipeline.wic_si12_decoder_reconstruction import fit_decoder, load_decoder


#%% Keep all available pixels of one sensor; never use the missing sensor

def retained_sensor(valid, removed_channel):
    if removed_channel not in (0, 1):
        raise ValueError("removed channel must be 0 (WIC) or 1 (SI12)")
    retained = np.broadcast_to(valid, (2, *valid.shape)).copy()
    retained[removed_channel] = False
    return retained


#%% Same truth and same PCA prior for both observation cases

def run(arguments):
    run_directory = arguments.run_directory.expanduser()
    artifact = torch.load(run_directory / "wic_si12_vae.pt", map_location="cpu", weights_only=False)
    dataset, split = _load_heldout_dataset(artifact, ChannelTransform.from_dict(artifact["transform"]), arguments.split)
    if not 0 <= arguments.frame_index < len(dataset):
        raise ValueError(f"frame index must lie between 0 and {len(dataset) - 1} for the {split} split")
    truth = dataset.values[arguments.frame_index].copy()
    valid = dataset.common_valid[arguments.frame_index, 0].copy()
    identity = {"split": split, "frame_index": arguments.frame_index, "orbit": int(dataset.frame_orbit[arguments.frame_index]), "time": str(dataset.frame_time[arguments.frame_index])}
    del dataset, artifact
    prior, decoder_fit, decoder = {}, None, None
    if arguments.reconstruction_method == "decoder":
        decoder = load_decoder(run_directory, arguments.device)
    elif arguments.prior_space == "coefficients":
        prior = coefficient_prior(run_directory, arguments.modes, arguments.decode_batch_size, arguments.device)
    else:
        prior = log_prior(run_directory, arguments.modes)
    if prior:
        if truth.shape != prior["mean"].shape:
            raise ValueError("held-out image shape differs from the saved ensemble grid")
        eigenfactor = prior["patterns"].reshape(arguments.modes, -1).T
    output = arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)

    for name, removed_channel in (("figure_5a", 1), ("figure_5b", 0)):
        removed_name = ("WIC", "SI12")[removed_channel]
        retained = retained_sensor(valid, removed_channel)
        if decoder is not None:
            posterior, decoder_fit = fit_decoder(*decoder, truth, retained, arguments.observation_std, prior_weight=arguments.latent_prior_weight, restarts=arguments.latent_restarts, max_iterations=arguments.latent_max_iterations, seed=arguments.seed)
        else:
            posterior = reconstruct(prior["mean"], eigenfactor, truth, retained, arguments.observation_std)
        input_labels = tuple("Sensor removed" if channel == removed_channel else "Observed input" for channel in range(2))
        labels = {} if decoder_fit is None else {"reconstruction_label": "Decoder fit", "uncertainty_label": "Approx. SD (counts)"}
        figure = make_figure(truth, valid, retained, posterior["reconstruction"], posterior["uncertainty"], (arguments.wic_max, arguments.si12_max), (arguments.wic_uncertainty_max, arguments.si12_uncertainty_max), input_labels, **labels)
        figure.savefig(output / f"{name}.png", dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        prior_arrays = {key: prior[key] for key in ("coefficient_mean", "coefficient_patterns", "row_basis", "column_basis") if key in prior}
        if prior:
            prior_arrays.update(prior_log_mean=prior["mean"], eigenvalues=prior["eigenvalues"])
        np.savez_compressed(output / f"{name}_reconstruction.npz", truth=truth, valid=valid, retained=retained, **posterior, **prior_arrays)
        rmse = np.sqrt(np.mean((posterior["reconstruction"][:, valid] - truth[:, valid])**2, axis=1)).tolist()
        report = {
            **identity,
            "run_directory": str(run_directory.resolve()),
            "removed_sensor": removed_name,
            "observed_sensor": ("SI12", "WIC")[removed_channel],
            "observed_cells_per_sensor": retained.sum(axis=(1, 2)).tolist(),
            "reconstruction_method": arguments.reconstruction_method,
            "modes": arguments.modes if prior else None,
            "prior_space": arguments.prior_space if prior else None,
            "variance_space": "local decoder latent covariance" if decoder_fit else ("joint log10(counts+1) spline coefficients" if arguments.prior_space == "coefficients" else "joint log10(counts+1) pixels"),
            "decode_device": decoder_fit["decode_device"] if decoder_fit else prior.get("decode_device"),
            "decoder_fit": decoder_fit,
            "observation_std_dex": arguments.observation_std,
            "sample_count": int(prior["sample_count"]) if prior else None,
            "captured_variance_fraction": float(prior["explained_variance_fraction"].sum()) if prior else None,
            "rmse_counts": rmse,
            "missing_sensor_rmse_counts": rmse[removed_channel],
            "truth_maxima_counts": [float(truth[channel][valid].max()) for channel in range(2)],
            "reconstruction_maxima_counts": [float(posterior["reconstruction"][channel][valid].max()) for channel in range(2)],
            "negative_draw_cells_clipped": prior.get("negative_draw_cells_clipped"),
            "minimum_recovered_draw": prior.get("minimum_recovered_draw"),
            "negative_median_cells_clipped": posterior["negative_median_cells_clipped"],
            "uncertainty": decoder_fit["uncertainty"] if decoder_fit else "conditional marginal shifted-lognormal STD in counts; truncated PCA prior, assumed independent log-space observation errors; not calibrated"
        }
        (output / f"{name}_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output / (name + '.png')}; removed {removed_name}; {split} frame {arguments.frame_index}, orbit {identity['orbit']}.")
        print(f"WIC/SI12 RMSE (counts): {rmse}; truth/reconstruction maxima: {report['truth_maxima_counts']} / {report['reconstruction_maxima_counts']}.")
        print(f"Clipped {posterior['negative_median_cells_clipped']} negative physical predictions; fit diagnostics and assumptions recorded in companion JSON.")
        if decoder_fit is not None:
            print(f"Decoder fit converged: {decoder_fit['best_fit_converged']}; observed log RMSE: {decoder_fit['observed_log_rmse_dex']:.4f} dex; latent norm: {decoder_fit['latent_norm']:.3f}.")


#%% Choose the held-out pair and the same settings as figure 4

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="directory for figure_5a.png and figure_5b.png")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--modes", type=int, default=50)
    parser.add_argument("--reconstruction-method", choices=("covariance", "decoder"), default="covariance")
    parser.add_argument("--latent-prior-weight", type=float, default=1.0, help="decoder latent prior precision; separate from training beta")
    parser.add_argument("--latent-restarts", type=int, default=8)
    parser.add_argument("--latent-max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--prior-space", choices=("pixels", "coefficients"), default="pixels", help="PCA of saved log pixel maps or redecoded log-count spline coefficients")
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="device for decoder fitting or coefficient decoding; covariance PCA stays on CPU")
    parser.add_argument("--observation-std", type=float, default=0.1, help="assumed independent observation error STD in dex, not dwic/dsi12")
    parser.add_argument("--wic-max", type=float)
    parser.add_argument("--si12-max", type=float)
    parser.add_argument("--wic-uncertainty-max", type=float)
    parser.add_argument("--si12-uncertainty-max", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
