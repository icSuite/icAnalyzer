"""Figure 4: held-out Hall/Pedersen half-image reconstruction from log covariance."""

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
from icanalyzer.wic_si12.physics import conductance_from_corrected_wic, PROPAGATION_ORDER, FROZEN_WIC_RESPONSE_AT_2_KEV
from scripts.IMAGE.vae_pipeline.plot_wic_si12_reconstruction import log_prior, make_figure, reconstruct, retained_half
from scripts.IMAGE.vae_pipeline.plot_wic_si12_sensor_reconstruction import retained_sensor
from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import _load_heldout_dataset


#%% Native held-out frames or the reviewed FUV-derived conductance reference

def validate_conductance_propagation(run_directory):
    summary = json.loads((run_directory / "summary.json").read_text())
    if summary.get("input_product_type") == "conductance_cs":
        if summary.get("field_names") != ["Hall", "Pedersen"] or summary.get("field_units") != "S" or summary.get("propagation_reapplied") is not False:
            raise ValueError("native conductance run must declare Hall/Pedersen in S without repropagation")
        return True
    if summary.get("propagation_order") != PROPAGATION_ORDER or not np.isclose(summary.get("wic_response_counts_per_mw_m2", np.nan), FROZEN_WIC_RESPONSE_AT_2_KEV, rtol=0, atol=1e-12):
        raise ValueError("saved conductance prior must use the reviewed grid-WIC fixed-2-keV propagation")
    return False


def load_conductance_frame(run_directory, split, frame_index, include_uncertainty=False):
    native = validate_conductance_propagation(run_directory)
    artifact = torch.load(run_directory / ("conductance_vae.pt" if native else "wic_si12_vae.pt"), map_location="cpu", weights_only=False)
    if native != (artifact.get("artifact_type") == "conductance_vae_run_v1"):
        raise ValueError("run summary and artifact disagree on native conductance versus FUV")
    if artifact.get("model_configuration", {}).get("kp_conditioning") is not None:
        raise ValueError("this covariance reconstruction requires an unconditional run; fixed-Kp ensembles must be matched to the reference activity explicitly")
    dataset, split = _load_heldout_dataset(artifact, ChannelTransform.from_dict(artifact["transform"]), split)
    if not 0 <= frame_index < len(dataset):
        raise ValueError(f"frame index must lie between 0 and {len(dataset) - 1} for the {split} split")
    if native:
        truth = dataset.values[frame_index].copy()
    else:
        converted = conductance_from_corrected_wic(dataset.values[frame_index, 0], include_intermediates=False)
        truth = np.stack((converted["H"], converted["P"]))
    valid = dataset.common_valid[frame_index, 0].copy() & np.all(np.isfinite(truth), axis=0)
    if native:
        valid &= artifact["data_provenance"]["training_support"]
    if not valid.any():
        raise ValueError("selected held-out frame has no common valid conductance pixels")
    identity = {"split": split, "frame_index": frame_index, "orbit": int(dataset.frame_orbit[frame_index]), "time": str(dataset.frame_time[frame_index])}
    if include_uncertainty:
        standard_deviation = dataset.standard_deviation[frame_index].copy() if native else None
        return truth, valid, identity, standard_deviation
    return truth, valid, identity


#%% Per-observation errors; native dH/dP are in S, the likelihood is in dex

def observation_error_mode(arguments, native):
    mode = arguments.observation_errors
    if mode == "auto":
        mode = "fixed" if arguments.observation_std is not None or not native else "propagated"
    if mode == "propagated" and not native:
        raise ValueError("propagated dH/dP require a native conductance_cs run")
    if mode == "propagated" and arguments.observation_std is not None:
        raise ValueError("--observation-std is only for fixed errors; omit it to use dH/dP")
    if arguments.observation_std is not None and (not np.isfinite(arguments.observation_std) or arguments.observation_std <= 0):
        raise ValueError("--observation-std must be finite and positive (dex)")
    if not np.isfinite(arguments.minimum_observation_std) or arguments.minimum_observation_std < 0:
        raise ValueError("--minimum-observation-std must be finite and nonnegative (S)")
    if mode == "fixed" and arguments.minimum_observation_std != 0:
        raise ValueError("--minimum-observation-std applies only to propagated dH/dP")
    return mode


def observation_errors(truth, standard_deviation, retained, arguments, mode):
    observed = np.broadcast_to(retained, truth.shape)
    errors = np.full(truth.shape, np.nan, dtype=float)
    floor_count = 0
    fixed_std = 0.1 if arguments.observation_std is None else arguments.observation_std
    if mode == "fixed":
        errors[observed] = fixed_std
    else:
        # Validate and transform ONLY the retained measurements: hidden errors
        # must not affect conditioning. No auroral-quality weighting is used.
        measured = np.asarray(standard_deviation, dtype=float)[observed]
        if not np.isfinite(measured).all() or np.any(measured < 0):
            raise ValueError("retained dH/dP must be finite and nonnegative (S)")
        floor_count = int(np.count_nonzero(measured < arguments.minimum_observation_std))
        measured = np.maximum(measured, arguments.minimum_observation_std)
        if np.any(measured == 0):
            raise ValueError("retained dH/dP contain zeros; set --minimum-observation-std to an explicit positive floor in S")
        # First-order propagation for y = log10(conductance + 1 S).
        errors[observed] = measured / (np.log(10) * (truth[observed] + 1))
    report = {
        "observation_error_source": "native_dH_dP" if mode == "propagated" else "fixed_log_std",
        "observation_std_dex": None if mode == "propagated" else fixed_std,
        "minimum_observation_std_siemens": arguments.minimum_observation_std if mode == "propagated" else None,
        "observation_std_floored_cells": floor_count,
        "observation_error_covariance": "diagonal; spatial and Hall/Pedersen measurement-error correlations unavailable",
        "observation_error_transform": "first-order sigma_log10 = sigma_S / (ln(10) * (conductance_S + 1))" if mode == "propagated" else "constant dex STD",
        "model_error_included": False,
        "observation_noise_added": False
    }
    return errors, report


def add_observation_error_arguments(parser):
    parser.add_argument("--observation-errors", choices=("auto", "propagated", "fixed"), default="auto", help="native default: dH/dP; FUV-derived default: fixed dex errors")
    parser.add_argument("--observation-std", type=float, help="fixed observation STD in dex; supplying this explicitly selects fixed errors in auto mode (legacy default 0.1)")
    parser.add_argument("--minimum-observation-std", type=float, default=0, help="optional lower bound for retained dH/dP in S; default 0, no clipping")


#%% Joint log10(conductance + 1 S) covariance; no decoder/coefficient PCA

def run(arguments, case="half"):
    if case not in ("half", "fields"):
        raise ValueError("conductance case must be half or fields")
    run_directory = arguments.run_directory.expanduser()
    native = validate_conductance_propagation(run_directory)
    error_mode = observation_error_mode(arguments, native)
    if error_mode == "propagated":
        truth, valid, identity, standard_deviation = load_conductance_frame(run_directory, arguments.split, arguments.frame_index, include_uncertainty=True)
    else:
        truth, valid, identity = load_conductance_frame(run_directory, arguments.split, arguments.frame_index)
        standard_deviation = None
    prior = log_prior(run_directory, arguments.modes, field_prefix="conductance")
    if truth.shape != prior["mean"].shape:
        raise ValueError("held-out conductance shape differs from the saved prior grid")
    eigenfactor = prior["patterns"].reshape(arguments.modes, -1).T
    output = arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    cases = [("figure_4", None)] if case == "half" else [("figure_5a", 1), ("figure_5b", 0)]
    field_names = ("Hall", "Pedersen")

    for name, removed_channel in cases:
        if case == "half":
            retained = retained_half(valid, arguments.remove_half)
            input_labels = ("Half removed", "Half removed")
        else:
            retained = retained_sensor(valid, removed_channel)
            input_labels = tuple("Field removed" if channel == removed_channel else "Observed input" for channel in range(2))
        errors, error_report = observation_errors(truth, standard_deviation, retained, arguments, error_mode)
        posterior = reconstruct(prior["mean"], eigenfactor, truth, retained, errors)
        figure = make_figure(truth, valid, retained, posterior["reconstruction"], posterior["uncertainty"], (arguments.hall_max, arguments.pedersen_max), (arguments.hall_uncertainty_max, arguments.pedersen_uncertainty_max), input_labels, uncertainty_label="Posterior SD (S)", field_names=field_names)
        figure.savefig(output / f"{name}.png", dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
        plt.close(figure)
        measurement_arrays = {} if standard_deviation is None else {"input_standard_deviation_siemens": standard_deviation}
        np.savez_compressed(output / f"{name}_reconstruction.npz", truth=truth, valid=valid, retained=retained, observation_std_dex=errors, **measurement_arrays, **posterior, prior_log_mean=prior["mean"], eigenvalues=prior["eigenvalues"], field_names=np.array(field_names))
        observed = np.broadcast_to(retained, truth.shape)
        missing = np.broadcast_to(valid, truth.shape) & ~observed
        rmse = [float(np.sqrt(np.mean((posterior["reconstruction"][channel][valid] - truth[channel][valid])**2))) for channel in range(2)]
        missing_rmse = [float(np.sqrt(np.mean((posterior["reconstruction"][channel][missing[channel]] - truth[channel][missing[channel]])**2))) if missing[channel].any() else None for channel in range(2)]
        report = {
            **identity,
            "run_directory": str(run_directory.resolve()),
            "field_names": list(field_names),
            "units": "siemens",
            "case": case,
            "removed_field": None if removed_channel is None else field_names[removed_channel],
            "remove_half": arguments.remove_half if case == "half" else None,
            "observed_cells_per_field": observed.sum(axis=(1, 2)).tolist(),
            "modes": arguments.modes,
            "state_space": "log10(conductance_siemens + 1)",
            **error_report,
            "sample_count": int(prior["sample_count"]),
            "captured_variance_fraction": float(prior["explained_variance_fraction"].sum()),
            "rmse_siemens": rmse,
            "missing_rmse_siemens": missing_rmse,
            "truth_maxima_siemens": [float(truth[channel][valid].max()) for channel in range(2)],
            "reconstruction_maxima_siemens": [float(posterior["reconstruction"][channel][valid].max()) for channel in range(2)],
            "negative_draw_cells_clipped": prior["negative_draw_cells_clipped"],
            "minimum_recovered_draw": prior["minimum_recovered_draw"],
            "negative_median_cells_clipped": posterior["negative_median_cells_clipped"],
            "reference_kind": "held-out native conductance_cs retrieval, not independent conductance truth" if native else "held-out grid-corrected WIC propagated at fixed 2 keV; NOT conductance_cs or independent conductance truth",
            "electron_energy_keV": None if native else 2.0,
            "propagation_order": "upstream_native_conductance_cs_no_repropagation" if native else PROPAGATION_ORDER,
            "limitation": "native retrieved fields, restricted to retained training support; no fixed-energy proportionality imposed by icAnalyzer" if native else "fixed energy makes Hall and Pedersen proportional; whole-field recovery is largely redundant",
            "uncertainty": "formal conditional shifted-lognormal STD in siemens; independent observation errors, first-order log propagation for native dH/dP when selected; no calibrated model error"
        }
        (output / f"{name}_reconstruction.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Saved {output / (name + '.png')}; {identity['split']} frame {identity['frame_index']}, orbit {identity['orbit']}, {identity['time']}.")
        print(f"Hall/Pedersen missing RMSE (S): {missing_rmse}; truth/reconstruction maxima: {report['truth_maxima_siemens']} / {report['reconstruction_maxima_siemens']}.")
        print(f"Observation errors: {error_report['observation_error_source']}; floored {error_report['observation_std_floored_cells']} retained cells.")
    print("Reference is native conductance_cs, not independent truth; posterior uncertainty is not calibrated." if native else "Reference is fixed-2-keV WIC-derived conductance, not conductance_cs. Hall/Pedersen are proportional; posterior uncertainty is not calibrated.")


#%% Same frame index as FUV figures; all maps and output uncertainty in siemens

def parse_arguments(argv=None, description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="directory for figure PNGs and companion NPZ/JSON files")
    parser.add_argument("--split", choices=("test", "validation"), default="test")
    parser.add_argument("--frame-index", type=int, default=0, help="zero-based saved held-out frame index; matches the browser for this run")
    parser.add_argument("--remove-half", choices=("left", "right", "top", "bottom"), default="right", help="figure4 only")
    parser.add_argument("--modes", type=int, default=50)
    add_observation_error_arguments(parser)
    parser.add_argument("--hall-max", type=float)
    parser.add_argument("--pedersen-max", type=float)
    parser.add_argument("--hall-uncertainty-max", type=float)
    parser.add_argument("--pedersen-uncertainty-max", type=float)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
