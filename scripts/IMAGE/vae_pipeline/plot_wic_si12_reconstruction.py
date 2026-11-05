"""Figure 4: held-out half-image reconstruction with joint log-space PCA 1--50."""

#%% Imports

import argparse
import json
from pathlib import Path
import shutil
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import linalg
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from scripts.IMAGE.vae_pipeline.plot_wic_si12_covariance import decompose_factor
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset
from scripts.IMAGE.vae_pipeline.wic_si12_coefficient_prior import coefficient_prior
from scripts.IMAGE.vae_pipeline.wic_si12_decoder_reconstruction import fit_decoder, load_decoder


#%% Build log10(physical value + 1) prior, not log10 of its physical mean

def log_prior(run_directory, n_modes=50, field_prefix="fuv"):
    if field_prefix not in ("fuv", "conductance"):
        raise ValueError("prior fields must be fuv or conductance")
    filename = Path(run_directory).expanduser() / "ensemble_products.npz"
    with np.load(filename, allow_pickle=False) as archive:
        physical_mean = archive[f"{field_prefix}_mean"]
    if physical_mean.ndim != 3 or physical_mean.shape[0] != 2 or not np.isfinite(physical_mean).all():
        raise ValueError("saved mean must be finite with shape (2, rows, columns)")
    with tempfile.TemporaryDirectory(prefix="wic-si12-log-pca-") as workspace:
        factor_path = Path(workspace) / "factor.npy"
        with zipfile.ZipFile(filename) as archive:
            with archive.open(f"{field_prefix}_anomaly_factor.npy") as source, factor_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        factor = np.load(factor_path, mmap_mode="r+", allow_pickle=False)
        if factor.ndim != 2 or factor.shape[0] != physical_mean.size or factor.shape[1] < 2:
            raise ValueError("saved anomaly factor does not match the mean")
        scale = np.sqrt(factor.shape[1] - 1)
        mean = np.empty(physical_mean.size)
        negative_count = 0
        minimum_draw = 0.0
        # Replace ONLY the temporary factor, keeping one disk-backed ensemble.
        for start in range(0, factor.shape[0], 32):
            stop = min(start + 32, factor.shape[0])
            values = np.array(factor[start:stop], dtype=np.float64)
            values *= scale
            values += physical_mean.ravel()[start:stop, None]
            if not np.isfinite(values).all():
                raise ValueError("saved prior draws contain nonfinite values")
            if field_prefix == "conductance" and values.min() < -1e-5:
                raise ValueError("saved conductance draws contain substantive negative values, not factor roundoff")
            negative_count += int(np.count_nonzero(values < 0))
            minimum_draw = min(minimum_draw, float(values.min()))
            np.maximum(values, 0, out=values)
            values += 1
            np.log10(values, out=values)
            mean[start:stop] = values.mean(axis=1)
            factor[start:stop] = (values - mean[start:stop, None]) / scale
        factor.flush()
        result = decompose_factor(factor, physical_mean.shape, n_modes=n_modes)
        del factor
    return {"mean": mean.reshape(physical_mean.shape), "negative_draw_cells_clipped": negative_count, "minimum_recovered_draw": minimum_draw, **result}


#%% Low-rank optimal interpolation; solve in PCA coefficient space

def reconstruct(mean, eigenfactor, values, retained, observation_std=0.1):
    retained = np.broadcast_to(retained, mean.shape).ravel()
    indices = np.flatnonzero(retained)
    if indices.size == 0:
        raise ValueError("no observed cells remain after masking the input")
    standard_deviation = np.broadcast_to(np.asarray(observation_std, dtype=float), mean.shape).ravel()[indices]
    if not np.isfinite(standard_deviation).all() or np.any(standard_deviation <= 0):
        raise ValueError("retained observation STD must be finite and positive (dex)")
    observations = np.asarray(values).ravel()[indices]
    if not np.isfinite(observations).all() or np.any(observations < 0):
        raise ValueError("retained observations must be finite and nonnegative")
    design = eigenfactor[indices] / standard_deviation[:, None]
    precision = np.eye(eigenfactor.shape[1]) + design.T @ design
    cholesky = linalg.cho_factor(precision, lower=True)
    residual = (np.log10(observations + 1) - mean.ravel()[indices]) / standard_deviation
    coefficients = linalg.cho_solve(cholesky, design.T @ residual)
    posterior_mean = (mean.ravel() + eigenfactor @ coefficients).reshape(mean.shape)
    conditional_factor = linalg.solve_triangular(cholesky[0], eigenfactor.T, lower=True).T
    posterior_variance = np.sum(conditional_factor**2, axis=1).reshape(mean.shape)
    # Physical median and exact shifted-lognormal marginal STD, in input units.
    median = 10**posterior_mean - 1
    log_variance = np.log(10)**2 * posterior_variance
    uncertainty = np.sqrt(np.expm1(log_variance)) * np.exp(np.log(10) * posterior_mean + 0.5 * log_variance)
    if not np.isfinite(median).all() or not np.isfinite(uncertainty).all():
        raise ValueError("posterior back-transformation overflowed; inspect the log-space reconstruction")
    return {"reconstruction": np.maximum(median, 0), "uncertainty": uncertainty, "posterior_log_mean": posterior_mean, "posterior_log_std": np.sqrt(posterior_variance), "negative_median_cells_clipped": int(np.count_nonzero(median < 0))}


def retained_half(valid, remove_half="right"):
    retained = valid.copy()
    rows, columns = retained.shape
    slices = {"left": (slice(None), slice(0, columns // 2)), "right": (slice(None), slice(columns // 2, None)), "bottom": (slice(0, rows // 2), slice(None)), "top": (slice(rows // 2, None), slice(None))}
    retained[slices[remove_half]] = False
    return retained


#%% Truth / damaged input / reconstruction / inferred uncertainty

def make_figure(truth, valid, retained, reconstruction, uncertainty, maxima=(None, None), uncertainty_maxima=(None, None), input_labels=("Half removed", "Half removed"), reconstruction_label="Reconstruction", uncertainty_label="Posterior SD (counts)", field_names=("WIC", "SI12")):
    retained = np.broadcast_to(retained, truth.shape)
    figure, axes = plt.subplots(2, 4, figsize=(10, 5))
    for channel, sensor in enumerate(field_names):
        panels = [np.where(valid, truth[channel], np.nan), np.where(retained[channel], truth[channel], np.nan), np.where(valid, reconstruction[channel], np.nan), np.where(valid, uncertainty[channel], np.nan)]
        maximum = max(float(np.nanmax(panels[0])), np.finfo(float).eps) if maxima[channel] is None else maxima[channel]
        labels = [f"{sensor} held-out", input_labels[channel], reconstruction_label, uncertainty_label]
        for column, panel in enumerate(panels):
            limit = maximum
            if column == 3:
                limit = max(float(np.nanmax(panel)), np.finfo(float).eps) if uncertainty_maxima[channel] is None else uncertainty_maxima[channel]
            if not np.isfinite(limit) or limit <= 0:
                plt.close(figure)
                raise ValueError("colour-scale maxima must be finite and positive")
            axes[channel, column].imshow(panel, origin="lower", cmap="viridis", vmin=0, vmax=limit, interpolation="nearest")
            axes[channel, column].set_axis_off()
            text_y, alignment = (1.04, "bottom") if channel == 0 else (-0.04, "top")
            axes[channel, column].text(0.5, text_y, labels[column], transform=axes[channel, column].transAxes, ha="center", va=alignment, fontsize=11, clip_on=False)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    return figure


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
    retained = retained_half(valid, arguments.remove_half)
    prior, decoder_fit = {}, None
    if arguments.reconstruction_method == "decoder":
        model, transform = load_decoder(run_directory, arguments.device)
        posterior, decoder_fit = fit_decoder(model, transform, truth, retained, arguments.observation_std, prior_weight=arguments.latent_prior_weight, restarts=arguments.latent_restarts, max_iterations=arguments.latent_max_iterations, seed=arguments.seed)
    elif arguments.prior_space == "coefficients":
        prior = coefficient_prior(run_directory, arguments.modes, arguments.decode_batch_size, arguments.device)
    else:
        prior = log_prior(run_directory, arguments.modes)
    if prior:
        if truth.shape != prior["mean"].shape:
            raise ValueError("held-out image shape differs from the saved prior grid")
        eigenfactor = prior["patterns"].reshape(arguments.modes, -1).T
        posterior = reconstruct(prior["mean"], eigenfactor, truth, retained, arguments.observation_std)
    labels = {} if decoder_fit is None else {"reconstruction_label": "Decoder fit", "uncertainty_label": "Approx. SD (counts)"}
    figure = make_figure(truth, valid, retained, posterior["reconstruction"], posterior["uncertainty"], (arguments.wic_max, arguments.si12_max), (arguments.wic_uncertainty_max, arguments.si12_uncertainty_max), **labels)
    output = arguments.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    prior_arrays = {key: prior[key] for key in ("coefficient_mean", "coefficient_patterns", "row_basis", "column_basis") if key in prior}
    if prior:
        prior_arrays.update(prior_log_mean=prior["mean"], eigenvalues=prior["eigenvalues"])
    np.savez_compressed(output.with_name(output.stem + "_reconstruction.npz"), truth=truth, valid=valid, retained=retained, **posterior, **prior_arrays)
    removed = valid & ~retained
    report = {
        **identity,
        "run_directory": str(run_directory.resolve()),
        "remove_half": arguments.remove_half,
        "reconstruction_method": arguments.reconstruction_method,
        "modes": arguments.modes if prior else None,
        "prior_space": arguments.prior_space if prior else None,
        "variance_space": "local decoder latent covariance" if decoder_fit else ("joint log10(counts+1) spline coefficients" if arguments.prior_space == "coefficients" else "joint log10(counts+1) pixels"),
        "decode_device": decoder_fit["decode_device"] if decoder_fit else prior.get("decode_device"),
        "decoder_fit": decoder_fit,
        "observation_std_dex": arguments.observation_std,
        "sample_count": int(prior["sample_count"]) if prior else None,
        "captured_variance_fraction": float(prior["explained_variance_fraction"].sum()) if prior else None,
        "negative_draw_cells_clipped": prior.get("negative_draw_cells_clipped"),
        "minimum_recovered_draw": prior.get("minimum_recovered_draw"),
        "negative_median_cells_clipped": posterior["negative_median_cells_clipped"],
        "hidden_half_rmse_counts": np.sqrt(np.mean((posterior["reconstruction"][:, removed] - truth[:, removed])**2, axis=1)).tolist(),
        "truth_maxima_counts": [float(truth[channel][valid].max()) for channel in range(2)],
        "reconstruction_maxima_counts": [float(posterior["reconstruction"][channel][valid].max()) for channel in range(2)],
        "uncertainty": decoder_fit["uncertainty"] if decoder_fit else "conditional marginal shifted-lognormal STD in counts; truncated PCA prior, assumed independent log-space observation errors; not calibrated"
    }
    output.with_name(output.stem + "_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output}; {split} frame {arguments.frame_index}, orbit {identity['orbit']}, {identity['time']}.")
    print(f"Hidden-half WIC/SI12 RMSE (counts): {report['hidden_half_rmse_counts']}; truth/reconstruction maxima: {report['truth_maxima_counts']} / {report['reconstruction_maxima_counts']}.")
    print(f"Method: {arguments.reconstruction_method}; clipped {posterior['negative_median_cells_clipped']} negative physical predictions. Fit diagnostics and uncertainty assumptions recorded in companion JSON.")
    if decoder_fit is not None:
        print(f"Decoder fit converged: {decoder_fit['best_fit_converged']}; observed log RMSE: {decoder_fit['observed_log_rmse_dex']:.4f} dex; latent norm: {decoder_fit['latent_norm']:.3f}.")


#%% User-selected example and error assumption

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--frame-index", type=int, default=0, help="zero-based index in the saved selected held-out split")
    parser.add_argument("--remove-half", choices=("left", "right", "top", "bottom"), default="right")
    parser.add_argument("--modes", type=int, default=50, help="leading joint log-space PCA modes, default 1--50")
    parser.add_argument("--reconstruction-method", choices=("covariance", "decoder"), default="covariance")
    parser.add_argument("--latent-prior-weight", type=float, default=1.0, help="decoder latent prior precision; separate from training beta")
    parser.add_argument("--latent-restarts", type=int, default=8)
    parser.add_argument("--latent-max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1, help="decoder latent initialization seed")
    parser.add_argument("--prior-space", choices=("pixels", "coefficients"), default="pixels", help="PCA of saved log pixel maps or redecoded log-count spline coefficients")
    parser.add_argument("--decode-batch-size", type=int, default=64, help="latent draws per decoder batch for coefficient PCA")
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="device for decoder fitting or coefficient decoding; covariance PCA stays on CPU")
    parser.add_argument("--observation-std", type=float, default=0.1, help="assumed independent observation error STD in dex, as in legacy OI; not dwic/dsi12")
    parser.add_argument("--wic-max", type=float, help="shared intensity maximum, default held-out WIC maximum")
    parser.add_argument("--si12-max", type=float, help="shared intensity maximum, default held-out SI12 maximum")
    parser.add_argument("--wic-uncertainty-max", type=float)
    parser.add_argument("--si12-uncertainty-max", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
