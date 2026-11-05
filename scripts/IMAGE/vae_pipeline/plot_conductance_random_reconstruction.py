"""Figure 6: reconstruct Hall/Pedersen from randomly retained image pixels.

Uses figure 4's log-space pixel-PCA prior and Gaussian conditioning. Only
the observation geometry changes; this is not legacy coefficient-space PCA.
"""

#%% Imports

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.IMAGE.vae_pipeline.plot_conductance_reconstruction import load_conductance_frame, validate_conductance_propagation, observation_error_mode, observation_errors, add_observation_error_arguments
from scripts.IMAGE.vae_pipeline.plot_wic_si12_reconstruction import log_prior, make_figure
from scripts.IMAGE.vae_pipeline.plot_wic_si12_random_reconstruction import random_retained_mask, reconstruct_random


#%% Same held-out reference and 2x4 layout as figure 4

def run(arguments):
    if not np.isfinite(arguments.p_remove) or not 0 <= arguments.p_remove <= 100:
        raise ValueError("--p-remove must be between 0 and 100 percent")
    if arguments.dpi <= 0 or arguments.seed < 0:
        raise ValueError("DPI must be positive and seed must be nonnegative")
    run_directory = arguments.run_directory.expanduser()
    output = arguments.output.expanduser()
    filenames = ("figure_6.png", "figure_6_reconstruction.npz", "figure_6_reconstruction.json")
    if any((output / filename).exists() for filename in filenames):
        raise FileExistsError("figure 6 already exists; choose another output directory to preserve it")
    native = bool(validate_conductance_propagation(run_directory))
    error_mode = observation_error_mode(arguments, native)
    if error_mode == "propagated":
        truth, valid, identity, standard_deviation = load_conductance_frame(run_directory, arguments.split, arguments.frame_index, include_uncertainty=True)
    else:
        truth, valid, identity = load_conductance_frame(run_directory, arguments.split, arguments.frame_index)
        standard_deviation = None
    prior = log_prior(run_directory, arguments.modes, field_prefix="conductance")
    if truth.shape != prior["mean"].shape:
        raise ValueError("held-out conductance shape differs from the saved prior grid")
    retained = random_retained_mask(valid, arguments.p_remove, arguments.seed)
    errors, error_report = observation_errors(truth, standard_deviation, retained, arguments, error_mode)
    posterior = reconstruct_random(prior, truth, retained, errors)
    prior_only = not retained.any()
    label = f"{arguments.p_remove:g}% removed"
    figure = make_figure(truth, valid, retained, posterior["reconstruction"], posterior["uncertainty"], (arguments.hall_max, arguments.pedersen_max), (arguments.hall_uncertainty_max, arguments.pedersen_uncertainty_max), (label, label), reconstruction_label="Prior only" if prior_only else "Reconstruction", uncertainty_label="Prior SD (S)" if prior_only else "Posterior SD (S)", field_names=("Hall", "Pedersen"))
    output.mkdir(parents=True, exist_ok=True)
    figure.savefig(output / filenames[0], dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    measurement_arrays = {} if standard_deviation is None else {"input_standard_deviation_siemens": standard_deviation}
    np.savez_compressed(output / filenames[1], truth=truth, valid=valid, retained=retained, observation_std_dex=errors, **measurement_arrays, **posterior, prior_log_mean=prior["mean"], eigenvalues=prior["eigenvalues"], field_names=np.array(["Hall", "Pedersen"]))
    removed = valid & ~retained
    missing_rmse = [float(np.sqrt(np.mean((posterior["reconstruction"][channel][removed] - truth[channel][removed])**2))) if removed.any() else None for channel in range(2)]
    report = {
        **identity,
        "run_directory": str(run_directory.resolve()),
        "field_names": ["Hall", "Pedersen"],
        "units": "siemens",
        "geometry": "uniform random removal without replacement; same spatial mask for both fields",
        "p_remove": arguments.p_remove,
        "p_remove_definition": "percentage of available valid spatial pixels, rounded to the nearest integer pixel count; existing gaps are excluded",
        "actual_percent_removed": float(100 * removed.sum() / valid.sum()),
        "available_spatial_pixels": int(valid.sum()),
        "removed_spatial_pixels": int(removed.sum()),
        "observed_spatial_pixels": int(retained.sum()),
        "seed": arguments.seed,
        "prior_only": prior_only,
        "prior_space": "pixels",
        "state_space": "log10(conductance_siemens + 1)",
        "modes": arguments.modes,
        "sample_count": int(prior["sample_count"]),
        "captured_variance_fraction": float(prior["explained_variance_fraction"].sum()),
        **error_report,
        "missing_rmse_siemens": missing_rmse,
        "truth_maxima_siemens": [float(truth[channel][valid].max()) for channel in range(2)],
        "reconstruction_maxima_siemens": [float(posterior["reconstruction"][channel][valid].max()) for channel in range(2)],
        "negative_median_cells_clipped": posterior["negative_median_cells_clipped"],
        "negative_draw_cells_clipped": prior["negative_draw_cells_clipped"],
        "minimum_recovered_draw": prior["minimum_recovered_draw"],
        "reference_kind": "held-out native conductance_cs retrieval, not independent truth" if native else "held-out corrected grid-WIC propagated at fixed 2 keV; NOT conductance_cs or independent conductance truth",
        "limitation": "only geometry changes versus figure 4; pixel PCA is not legacy coefficient PCA; native H/P are not forced proportional" if native else "only geometry changes versus figure 4; pixel PCA is not legacy coefficient PCA; WIC-derived fixed-energy Hall/Pedersen are proportional",
        "uncertainty": "formal shifted-lognormal STD in siemens under truncated log-PCA covariance; independent observation errors with first-order native dH/dP log propagation when selected; no calibrated model error"
    }
    (output / filenames[2]).write_text(json.dumps(report, indent=2) + "\n")
    print(f"Saved {output / filenames[0]}; {identity['split']} frame {identity['frame_index']}, orbit {identity['orbit']}, {identity['time']}.")
    print(f"Retained {retained.sum()}/{valid.sum()} available pixels in each field; {report['actual_percent_removed']:.3f}% removed; seed {arguments.seed}.")
    print(f"Missing Hall/Pedersen RMSE (S): {missing_rmse}." + (" No observations: prior-only estimate." if prior_only else ""))
    return report


#%% Percentage and random seed; all other defaults match conductance figure 4

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="directory for figure_6.png and companion NPZ/JSON")
    parser.add_argument("--p-remove", type=float, required=True, help="0--100 percent of valid spatial pixels to remove, identically in both fields")
    parser.add_argument("--seed", type=int, default=0, help="random mask seed")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--modes", type=int, default=50, help="joint log-space pixel PCA mode count, same default as figure 4")
    add_observation_error_arguments(parser)
    parser.add_argument("--hall-max", type=float)
    parser.add_argument("--pedersen-max", type=float)
    parser.add_argument("--hall-uncertainty-max", type=float)
    parser.add_argument("--pedersen-uncertainty-max", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args(argv)
    if not np.isfinite(arguments.p_remove) or not 0 <= arguments.p_remove <= 100:
        parser.error("--p-remove must be between 0 and 100 percent")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
