"""Figure 6: reconstruct held-out WIC/SI12 from randomly retained pixels."""

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
from scripts.IMAGE.vae_pipeline.plot_wic_si12_reconstruction import log_prior, make_figure as make_reference_figure, reconstruct
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset
from scripts.IMAGE.vae_pipeline.wic_si12_coefficient_prior import coefficient_prior


#%% Exact percentage of available pixels; one shared WIC/SI12 mask

def random_retained_mask(valid, p_remove, seed=0):
    if not np.isfinite(p_remove) or not 0 <= p_remove <= 100:
        raise ValueError("p_remove must be finite and between 0 and 100 percent")
    valid = np.asarray(valid, dtype=bool)
    if valid.ndim != 2 or not valid.any():
        raise ValueError("valid must be a two-dimensional mask with available pixels")
    indices = np.flatnonzero(valid)
    n_removed = int(np.rint(indices.size * p_remove / 100))
    removed = np.random.default_rng(seed).choice(indices, size=n_removed, replace=False)
    retained = valid.copy()
    retained.ravel()[removed] = False
    return retained


#%% Gaussian conditioning; 100% removal returns the unchanged log prior

def reconstruct_random(prior, truth, retained, observation_std):
    errors = np.asarray(observation_std)
    if errors.ndim == 0 and (not np.isfinite(errors) or errors <= 0):
        raise ValueError("observation STD must be finite and positive (dex)")
    factor = prior["patterns"].reshape(len(prior["patterns"]), -1).T
    if retained.any():
        return reconstruct(prior["mean"], factor, truth, retained, observation_std)
    log_mean = prior["mean"]
    log_variance = np.sum(factor**2, axis=1).reshape(log_mean.shape)
    median = 10**log_mean - 1
    variance_ln = np.log(10)**2 * log_variance
    uncertainty = np.sqrt(np.expm1(variance_ln)) * np.exp(np.log(10) * log_mean + 0.5 * variance_ln)
    if not np.isfinite(median).all() or not np.isfinite(uncertainty).all():
        raise ValueError("prior back-transformation overflowed; inspect the log prior")
    return {"reconstruction": np.maximum(median, 0), "uncertainty": uncertainty, "posterior_log_mean": log_mean.copy(), "posterior_log_std": np.sqrt(log_variance), "negative_median_cells_clipped": int(np.count_nonzero(median < 0))}


#%% Keep reference gaps; display predictions and uncertainty on the full grid

def make_figure(truth, valid, retained, reconstruction, uncertainty, maxima=(None, None), uncertainty_maxima=(None, None), input_labels=("Random input", "Random input"), **labels):
    figure = make_reference_figure(truth, valid, retained, reconstruction, uncertainty, maxima, uncertainty_maxima, input_labels, **labels)
    for channel in range(2):
        figure.axes[4 * channel + 2].images[0].set_data(reconstruction[channel])
        sd_image = figure.axes[4 * channel + 3].images[0]
        sd_image.set_data(uncertainty[channel])
        if uncertainty_maxima[channel] is None:
            sd_image.set_clim(0, max(float(np.max(uncertainty[channel])), np.finfo(float).eps))
    return figure


#%% Load the selected FUV frame and its joint log-space prior

def run(arguments):
    if not np.isfinite(arguments.p_remove) or not 0 <= arguments.p_remove <= 100:
        raise ValueError("--p-remove must be between 0 and 100 percent")
    if not np.isfinite(arguments.observation_std) or arguments.observation_std <= 0:
        raise ValueError("--observation-std must be finite and positive (dex)")
    if arguments.dpi <= 0 or arguments.seed < 0:
        raise ValueError("DPI must be positive and seed must be nonnegative")
    run_directory = arguments.run_directory.expanduser()
    output = arguments.output.expanduser()
    filenames = ("figure_6.png", "figure_6_reconstruction.npz", "figure_6_reconstruction.json")
    artifact = torch.load(run_directory / "wic_si12_vae.pt", map_location="cpu", weights_only=False)
    dataset, split = _load_heldout_dataset(artifact, ChannelTransform.from_dict(artifact["transform"]), arguments.split)
    if not 0 <= arguments.frame_index < len(dataset):
        raise ValueError(f"frame index must lie between 0 and {len(dataset) - 1} for the {split} split")
    truth = dataset.values[arguments.frame_index].copy()
    valid = dataset.common_valid[arguments.frame_index, 0].copy()
    identity = {"split": split, "frame_index": arguments.frame_index, "orbit": int(dataset.frame_orbit[arguments.frame_index]), "time": str(dataset.frame_time[arguments.frame_index])}
    del artifact, dataset
    if arguments.prior_space == "coefficients":
        prior = coefficient_prior(run_directory, arguments.modes, arguments.decode_batch_size, arguments.device)
    else:
        prior = log_prior(run_directory, arguments.modes, field_prefix="fuv")
    if truth.shape != prior["mean"].shape:
        raise ValueError("held-out WIC/SI12 shape differs from the saved prior grid")
    retained = random_retained_mask(valid, arguments.p_remove, arguments.seed)
    posterior = reconstruct_random(prior, truth, retained, arguments.observation_std)
    prior_only = not retained.any()
    label = f"{arguments.p_remove:g}% removed"
    figure = make_figure(truth, valid, retained, posterior["reconstruction"], posterior["uncertainty"], (arguments.wic_max, arguments.si12_max), (arguments.wic_uncertainty_max, arguments.si12_uncertainty_max), (label, label), reconstruction_label="Prior only" if prior_only else "Reconstruction", uncertainty_label="Prior SD (counts)" if prior_only else "Posterior SD (counts)")
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / filenames[0], dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    prior_arrays = {key: prior[key] for key in ("coefficient_mean", "coefficient_patterns", "row_basis", "column_basis") if key in prior}
    np.savez_compressed(output / filenames[1], truth=truth, valid=valid, retained=retained, **posterior, prior_log_mean=prior["mean"], eigenvalues=prior["eigenvalues"], field_names=np.array(["WIC", "SI12"]), **prior_arrays)
    removed = valid & ~retained
    missing_rmse = [float(np.sqrt(np.mean((posterior["reconstruction"][channel][removed] - truth[channel][removed])**2))) if removed.any() else None for channel in range(2)]
    report = {
        **identity,
        "run_directory": str(run_directory.resolve()),
        "field_names": ["WIC", "SI12"],
        "units": "counts",
        "geometry": "uniform random removal without replacement; same spatial mask for both sensors",
        "p_remove": arguments.p_remove,
        "p_remove_definition": "percentage of available valid spatial pixels, rounded to the nearest integer pixel count; existing gaps are excluded",
        "actual_percent_removed": float(100 * removed.sum() / valid.sum()),
        "available_spatial_pixels": int(valid.sum()),
        "removed_spatial_pixels": int(removed.sum()),
        "observed_spatial_pixels": int(retained.sum()),
        "seed": arguments.seed,
        "prior_only": prior_only,
        "prior_space": arguments.prior_space,
        "variance_space": "joint log10(counts+1) spline coefficients" if arguments.prior_space == "coefficients" else "joint log10(counts+1) pixels",
        "modes": arguments.modes,
        "sample_count": int(prior["sample_count"]),
        "captured_variance_fraction": float(prior["explained_variance_fraction"].sum()),
        "observation_std_dex": arguments.observation_std,
        "observation_noise_added": False,
        "missing_rmse_counts": missing_rmse,
        "truth_maxima_counts": [float(truth[channel][valid].max()) for channel in range(2)],
        "reconstruction_maxima_counts": [float(posterior["reconstruction"][channel][valid].max()) for channel in range(2)],
        "full_grid_reconstruction_maxima_counts": [float(posterior["reconstruction"][channel].max()) for channel in range(2)],
        "display_masks": "original/sampled inputs retain missing-data masks; reconstruction and posterior SD displayed on the full grid; RMSE still excludes original missing pixels",
        "negative_median_cells_clipped": posterior["negative_median_cells_clipped"],
        "negative_draw_cells_clipped": prior.get("negative_draw_cells_clipped"),
        "minimum_recovered_draw": prior.get("minimum_recovered_draw"),
        "reference_kind": "held-out corrected WIC and SI12 inputs; no conductance conversion",
        "uncertainty": "formal shifted-lognormal STD in counts under truncated log-PCA covariance; assumed independent log errors, not propagated dwic/dsi12 or calibrated model error"
    }
    (output / filenames[2]).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output / filenames[0]}; {split} frame {arguments.frame_index}, orbit {identity['orbit']}, {identity['time']}.")
    print(f"Retained {retained.sum()}/{valid.sum()} available pixels in each sensor; {report['actual_percent_removed']:.3f}% removed; seed {arguments.seed}.")
    print(f"Missing WIC/SI12 RMSE (counts): {missing_rmse}." + (" No observations: prior-only estimate." if prior_only else ""))
    return report


#%% Percentage and seed; coefficient mode reproduces the legacy G Q G.T update

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="directory for figure_6.png and companion NPZ/JSON")
    parser.add_argument("--p-remove", type=float, required=True, help="0--100 percent of valid pixels to remove, identically in WIC and SI12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--modes", type=int, default=50)
    parser.add_argument("--prior-space", choices=("pixels", "coefficients"), default="pixels", help="pixels matches default figure 4; coefficients uses legacy-style spline covariance")
    parser.add_argument("--decode-batch-size", type=int, default=64)
    parser.add_argument("--device", choices=("cpu", "cuda", "auto"), default="cpu", help="coefficient decoding only")
    parser.add_argument("--observation-std", type=float, default=0.1, help="assumed independent observation STD in dex, not counts")
    parser.add_argument("--wic-max", type=float)
    parser.add_argument("--si12-max", type=float)
    parser.add_argument("--wic-uncertainty-max", type=float)
    parser.add_argument("--si12-uncertainty-max", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args(argv)
    if not np.isfinite(arguments.p_remove) or not 0 <= arguments.p_remove <= 100:
        parser.error("--p-remove must be between 0 and 100 percent")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
