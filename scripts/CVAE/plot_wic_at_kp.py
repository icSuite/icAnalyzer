"""Create polar WIC/SI12/Fang plots from conditional ensembles."""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np

from icanalyzer.cvae.summaries import summarize_wic_anomaly_factor
from scripts.CVAE.sample_wic_at_kp import (
    plot_conductance,
    plot_wic_exceedance_probability,
    plot_wic_medoid,
    plot_wic,
    plot_wic_sample_mosaic,
)


#%% Plot saved summaries without sampling again

def _sensor_mosaic_members(saved, prefix):
    """Load sensor mosaic members or reconstruct them from saved anomalies."""

    samples_name = f"{prefix}_mosaic_samples"
    indices_name = f"{prefix}_mosaic_sample_indices"
    if {samples_name, indices_name}.issubset(saved):
        return saved[samples_name], saved[indices_name]

    mean = np.asarray(saved[f"{prefix}_mean"], dtype=np.float64)
    anomaly_factor = np.asarray(saved[f"{prefix}_anomaly_factor"])
    if (
        mean.ndim != 2 or anomaly_factor.ndim != 2
        or anomaly_factor.shape[0] != mean.size
        or anomaly_factor.shape[1] < 2
    ):
        raise ValueError(
            f"saved {prefix} mean and anomaly factor cannot reconstruct samples"
        )
    count = min(36, anomaly_factor.shape[1])
    indices = np.arange(count, dtype=np.int32)
    flattened = (
        mean.reshape(-1, 1)
        + anomaly_factor[:, :count] * np.sqrt(anomaly_factor.shape[1] - 1)
    )
    return flattened.T.reshape(count, *mean.shape), indices


def _wic_mosaic_members(saved):
    """Backward-compatible WIC mosaic reconstruction helper."""

    return _sensor_mosaic_members(saved, "wic")


def _sensor_distribution_summary(
    saved, prefix, support, mlat, mlt, threshold
):
    """Load one sensor summary or derive it from a saved anomaly factor."""

    required = {
        f"{prefix}_medoid", f"{prefix}_medoid_sample_index",
        f"{prefix}_exceedance_probability",
        f"{prefix}_exceedance_threshold",
    }
    if required.issubset(saved) and np.isclose(
        float(saved[f"{prefix}_exceedance_threshold"]), threshold
    ):
        return {
            "medoid": saved[f"{prefix}_medoid"],
            "medoid_index": int(saved[f"{prefix}_medoid_sample_index"]),
            "exceedance_probability": saved[
                f"{prefix}_exceedance_probability"
            ],
            "exceedance_threshold": float(
                saved[f"{prefix}_exceedance_threshold"]
            ),
        }
    return summarize_wic_anomaly_factor(
        saved[f"{prefix}_mean"], saved[f"{prefix}_anomaly_factor"],
        support, mlat, mlt, threshold=threshold
    )


def _wic_distribution_summary(saved, support, mlat, mlt, threshold):
    """Backward-compatible WIC distribution-summary helper."""

    return _sensor_distribution_summary(
        saved, "wic", support, mlat, mlt, threshold
    )


def run(arguments):
    limits = np.asarray(
        [arguments.wic_vmax, arguments.hall_vmax, arguments.pedersen_vmax,
         arguments.wic_exceedance_threshold, arguments.si12_vmax,
         arguments.si12_exceedance_threshold],
        dtype=float
    )
    if (
        not np.isfinite(limits).all()
        or np.any(limits[[0, 1, 2, 4]] <= 0)
        or np.any(limits[[3, 5]] < 0)
    ):
        raise ValueError(
            "plot limits must be finite, with positive maxima and "
            "nonnegative threshold"
        )
    root = arguments.sample_directory.expanduser().resolve()
    summary_path = root / "sampling_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError("sampling_summary.json is absent; wait for sampling to complete")
    summary = json.loads(summary_path.read_text())
    selections = summary.get("selections")
    if not isinstance(selections, list) or not selections:
        raise ValueError("sampling summary has no condition selections")

    plotted = []
    for selection in selections:
        destination = root / selection["directory"]
        title = selection.get("label")
        if title is None:
            title = f"Kp {selection['Kp']:g}"
        product_path = destination / "ensemble_products.npz"
        if not product_path.is_file():
            raise FileNotFoundError(f"missing saved ensemble product: {product_path}")
        with np.load(product_path) as saved:
            field_name = (
                str(saved["wic_field_name"])
                if "wic_field_name" in saved else "wic_corrected"
            )
            field_label = (
                "Corrected WIC"
                if field_name == "wic_corrected" else "Pre-proton WIC"
            )
            mlat = saved["mlat"]
            mlt = saved["mlt"]
            wic_quantiles = np.stack(
                [saved["wic_q10"], saved["wic_q50"], saved["wic_q90"]]
            )
            plot_wic(
                saved["wic_mean"], wic_quantiles,
                saved["wic_supported"].astype(bool), mlat, mlt,
                title, destination, arguments.wic_vmax, field_label
            )
            mosaic_samples, mosaic_indices = _wic_mosaic_members(saved)
            plot_wic_sample_mosaic(
                mosaic_samples, mosaic_indices,
                saved["wic_supported"].astype(bool), mlat, mlt,
                title, destination, arguments.wic_vmax, field_label
            )
            distribution_summary = _wic_distribution_summary(
                saved, saved["wic_supported"].astype(bool), mlat, mlt,
                arguments.wic_exceedance_threshold
            )
            plot_wic_medoid(
                distribution_summary["medoid"],
                distribution_summary["medoid_index"],
                saved["wic_supported"].astype(bool), mlat, mlt,
                title, destination, arguments.wic_vmax, field_label
            )
            plot_wic_exceedance_probability(
                distribution_summary["exceedance_probability"],
                distribution_summary["exceedance_threshold"],
                saved["wic_supported"].astype(bool), mlat, mlt,
                title, destination, field_label
            )
            if "si12_mean" in saved:
                si12_support = saved["si12_supported"].astype(bool)
                si12_quantiles = np.stack([
                    saved["si12_q10"], saved["si12_q50"],
                    saved["si12_q90"]
                ])
                plot_wic(
                    saved["si12_mean"], si12_quantiles,
                    si12_support, mlat, mlt, title, destination,
                    arguments.si12_vmax, "SI12", "si12"
                )
                si12_mosaic, si12_indices = _sensor_mosaic_members(
                    saved, "si12"
                )
                plot_wic_sample_mosaic(
                    si12_mosaic, si12_indices, si12_support,
                    mlat, mlt, title, destination,
                    arguments.si12_vmax, "SI12", "si12"
                )
                si12_summary = _sensor_distribution_summary(
                    saved, "si12", si12_support, mlat, mlt,
                    arguments.si12_exceedance_threshold
                )
                plot_wic_medoid(
                    si12_summary["medoid"], si12_summary["medoid_index"],
                    si12_support, mlat, mlt, title, destination,
                    arguments.si12_vmax, "SI12", "si12"
                )
                plot_wic_exceedance_probability(
                    si12_summary["exceedance_probability"],
                    si12_summary["exceedance_threshold"],
                    si12_support, mlat, mlt, title, destination,
                    "SI12", "si12"
                )
            if "conductance_mean" in saved:
                conductance_quantiles = np.stack([
                    saved["conductance_q10"], saved["conductance_q50"],
                    saved["conductance_q90"]
                ])
                plot_conductance(
                    saved["conductance_mean"], conductance_quantiles,
                    saved["conductance_supported"].astype(bool), mlat, mlt,
                    title, destination,
                    (arguments.hall_vmax, arguments.pedersen_vmax)
                )
        plotted.append(destination.name)
        print(f"Plotted {destination.name}", flush=True)
    return plotted


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_directory", type=Path)
    parser.add_argument("--wic-vmax", type=float, default=1000.0)
    parser.add_argument("--wic-exceedance-threshold", type=float, default=100.0)
    parser.add_argument("--si12-vmax", type=float, default=50.0)
    parser.add_argument("--si12-exceedance-threshold", type=float, default=10.0)
    parser.add_argument("--hall-vmax", type=float, default=24.0)
    parser.add_argument("--pedersen-vmax", type=float, default=12.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
