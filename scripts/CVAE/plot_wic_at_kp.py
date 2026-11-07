"""Create polar WIC/Fang plots from completed conditional ensembles."""

#%% Imports

import argparse
import json
from pathlib import Path

import numpy as np

from scripts.CVAE.sample_wic_at_kp import plot_conductance, plot_wic


#%% Plot saved summaries without sampling again

def run(arguments):
    limits = np.asarray(
        [arguments.wic_vmax, arguments.hall_vmax, arguments.pedersen_vmax],
        dtype=float
    )
    if not np.isfinite(limits).all() or np.any(limits <= 0):
        raise ValueError("plot limits must be positive and finite")
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
        product_path = destination / "ensemble_products.npz"
        if not product_path.is_file():
            raise FileNotFoundError(f"missing saved ensemble product: {product_path}")
        with np.load(product_path) as saved:
            mlat = saved["mlat"]
            mlt = saved["mlt"]
            wic_quantiles = np.stack(
                [saved["wic_q10"], saved["wic_q50"], saved["wic_q90"]]
            )
            conductance_quantiles = np.stack([
                saved["conductance_q10"], saved["conductance_q50"],
                saved["conductance_q90"]
            ])
            plot_wic(
                saved["wic_mean"], wic_quantiles,
                saved["wic_supported"].astype(bool), mlat, mlt,
                selection.get("label", f"Kp {selection['Kp']:g}"),
                destination, arguments.wic_vmax
            )
            plot_conductance(
                saved["conductance_mean"], conductance_quantiles,
                saved["conductance_supported"].astype(bool), mlat, mlt,
                selection.get("label", f"Kp {selection['Kp']:g}"), destination,
                (arguments.hall_vmax, arguments.pedersen_vmax)
            )
        plotted.append(destination.name)
        print(f"Plotted {destination.name}", flush=True)
    return plotted


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sample_directory", type=Path)
    parser.add_argument("--wic-vmax", type=float, default=1000.0)
    parser.add_argument("--hall-vmax", type=float, default=24.0)
    parser.add_argument("--pedersen-vmax", type=float, default=12.0)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
