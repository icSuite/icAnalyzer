"""Step 2: summarize paired Hall/Pedersen conductance covariance.

The state vector is always ``[Hall pixels, Pedersen pixels]`` for the same
ensemble member. The script produces the physical mean, leading joint PCA
patterns scaled to one standard deviation, and anchor teleconnections without
forming the dense state-by-state covariance matrix.
"""

#%% Imports

import argparse
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from icanalyzer.wic_si12.diagnostics import anomaly_variance, covariance_mode_decomposition
from scripts.AGU2026._common import load_ensemble_grid, mapped_npz_array, masked, save_figure
from scripts.IMAGE.vae_pipeline.plot_wic_si12_teleconnections import teleconnection_maps


#%% Joint modes and anchor selection

def orient_modes(modes, support):
    """Choose a reproducible sign; PCA signs have no physical orientation."""

    modes = np.asarray(modes).copy()
    for number in range(len(modes)):
        if np.sum(modes[number, :, support]) < 0:
            modes[number] *= -1
    return modes


def automatic_anchor(factor, state_shape, support):
    """Choose the supported cell with maximum joint H/P variance."""

    variance = anomaly_variance(factor).reshape(state_shape)
    joint = np.where(support, variance.sum(axis=0), -np.inf)
    return int(np.argmax(joint))


#%% Poster figures

def plot_mean(mean, support, output, dpi):
    figure, axes = plt.subplots(1, 2, figsize=(8.2, 3.8), constrained_layout=True)
    for channel, name in enumerate(("Hall", "Pedersen")):
        image = axes[channel].imshow(masked(mean[channel], support), origin="lower", cmap="viridis", vmin=0, interpolation="nearest")
        axes[channel].set_title(f"Mean {name} conductance")
        axes[channel].set_axis_off()
        figure.colorbar(image, ax=axes[channel], label="S", shrink=0.82)
    save_figure(figure, output, "02_conductance_mean", dpi)
    plt.close(figure)


def plot_modes(patterns, fractions, support, output, dpi):
    count = len(patterns)
    figure, axes = plt.subplots(2, count, figsize=(3.6 * count, 6.2), constrained_layout=True, squeeze=False)
    for channel, name in enumerate(("Hall", "Pedersen")):
        limit = max(float(np.nanpercentile(abs(patterns[:, channel, support]), 99.5)), np.finfo(float).eps)
        for mode in range(count):
            image = axes[channel, mode].imshow(
                masked(patterns[mode, channel], support), origin="lower",
                cmap="RdBu_r", vmin=-limit, vmax=limit, interpolation="nearest",
            )
            axes[channel, mode].set_title(f"{name} PC{mode + 1}\n{100 * fractions[mode]:.1f}% variance")
            axes[channel, mode].set_axis_off()
        figure.colorbar(image, ax=axes[channel], label="1σ conductance anomaly [S]", shrink=0.78)
    figure.suptitle("Joint Hall–Pedersen PCA; mode sign is arbitrary")
    save_figure(figure, output, "02_joint_pca_modes", dpi)
    plt.close(figure)


def plot_teleconnections(correlations, anchor, support, output, dpi):
    row, column = anchor
    figure, axes = plt.subplots(2, 2, figsize=(8.2, 7.3), constrained_layout=True)
    for source, source_name in enumerate(("Hall", "Pedersen")):
        for target, target_name in enumerate(("Hall", "Pedersen")):
            image = axes[target, source].imshow(
                masked(correlations[source, target], support), origin="lower",
                cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest",
            )
            axes[target, source].plot(column, row, "o", mfc="none", mec="black", mew=1.5)
            axes[target, source].set_title(f"anchor {source_name} → {target_name}")
            axes[target, source].set_axis_off()
    figure.colorbar(image, ax=axes, label="Pearson correlation", shrink=0.82)
    figure.suptitle(f"Conductance teleconnections from row {row}, column {column}")
    save_figure(figure, output, "02_conductance_teleconnections", dpi)
    plt.close(figure)


#%% Chronological covariance workflow

def run(arguments):
    grid = load_ensemble_grid(arguments.ensemble)
    output = arguments.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("choose an empty covariance output directory")
    output.mkdir(parents=True, exist_ok=True)
    workspace = None if arguments.workspace is None else arguments.workspace.expanduser().resolve()

    with zipfile.ZipFile(grid["archive"]) as archive:
        factor_info = archive.getinfo("conductance_anomaly_factor.npy")
    with mapped_npz_array(grid["archive"], "conductance_anomaly_factor", workspace) as factor:
        expected = (grid["mean"].size,)
        if factor.ndim != 2 or factor.shape[0] != expected[0] or factor.shape[1] < 2:
            raise ValueError("conductance anomaly factor is incompatible with the saved mean")
        decomposition = covariance_mode_decomposition(
            factor, grid["mean"].shape, n_modes=arguments.modes,
            random_seed=arguments.seed, power_iterations=arguments.power_iterations,
        )
        modes = orient_modes(decomposition["modes"], grid["support"])
        patterns = modes * np.sqrt(decomposition["eigenvalues"])[:, None, None, None]
        fractions = decomposition["eigenvalues"] / decomposition["total_variance"]
        grid_id = arguments.grid_id
        if grid_id is None:
            grid_id = automatic_anchor(factor, grid["mean"].shape, grid["support"])
        if not 0 <= grid_id < grid["support"].size or not grid["support"].ravel()[grid_id]:
            raise ValueError("teleconnection grid ID must identify a supported spatial cell")
        connections = teleconnection_maps(
            factor, grid["mean"].shape, grid_id, batch_size=arguments.teleconnection_batch_size
        )
        sample_count = int(factor.shape[1])

    row, column = connections["anchor_row_column"]
    eigenfactor = modes.reshape(len(modes), -1).T * np.sqrt(decomposition["eigenvalues"])
    np.savez_compressed(
        output / "02_conductance_covariance.npz",
        mean=grid["mean"], support=grid["support"], mlat=grid["mlat"], mlt=grid["mlt"],
        eigenvalues=decomposition["eigenvalues"], modes=modes,
        eigenfactor=eigenfactor.astype(np.float32),
        explained_variance_fraction=fractions,
        total_variance=np.array(decomposition["total_variance"]),
        sample_count=np.array(sample_count),
        anchor_grid_id=np.array(grid_id), anchor_row_column=np.array((row, column)),
        anchor_mlat=np.array(grid["mlat"][row, column]),
        anchor_mlt=np.array(grid["mlt"][row, column]),
        correlations=connections["correlations"],
        anchor_correlation=connections["anchor_correlation"],
        source_factor_crc32=np.array(factor_info.CRC),
        source_factor_file_size=np.array(factor_info.file_size),
    )
    plot_mean(grid["mean"], grid["support"], output, arguments.dpi)
    plot_modes(patterns[:arguments.plot_modes], fractions[:arguments.plot_modes], grid["support"], output, arguments.dpi)
    plot_teleconnections(connections["correlations"], (row, column), grid["support"], output, arguments.dpi)

    report = {
        "purpose": "joint physical Hall/Pedersen covariance summary",
        "ensemble_archive": str(grid["archive"]),
        "source_factor_crc32": factor_info.CRC,
        "source_factor_file_size": factor_info.file_size,
        "sample_count": sample_count,
        "state_shape": list(grid["mean"].shape),
        "retained_modes": len(modes),
        "displayed_modes": arguments.plot_modes,
        "decomposition": decomposition["method"],
        "captured_variance_fraction": decomposition["captured_variance_fraction"],
        "leading_explained_variance_fraction": fractions[:arguments.plot_modes].tolist(),
        "anchor": {
            "selection": "maximum supported joint Hall/Pedersen variance" if arguments.grid_id is None else "user grid ID",
            "grid_id": grid_id, "row": int(row), "column": int(column),
            "mlat": float(grid["mlat"][row, column]), "mlt": float(grid["mlt"][row, column]),
        },
        "state_space": "physical siemens",
        "pairing": "Hall and Pedersen from each identical conductance realization remain one joint state",
        "dense_covariance_materialized": False,
    }
    (output / "02_conductance_covariance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("sample_count", "retained_modes", "captured_variance_fraction", "anchor")}, indent=2))
    return report


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ensemble", type=Path, help="sampling condition directory or ensemble_products.npz")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--modes", type=int, default=100, help="joint covariance modes saved for sparse reconstruction")
    parser.add_argument("--plot-modes", type=int, default=3)
    parser.add_argument("--grid-id", type=int, help="optional row-major teleconnection anchor; default is maximum joint variance")
    parser.add_argument("--workspace", type=Path, help="temporary local directory with room for the extracted anomaly factor")
    parser.add_argument("--teleconnection-batch-size", type=int, default=64)
    parser.add_argument("--power-iterations", type=int, default=2)
    parser.add_argument("--seed", type=int, default=12104)
    parser.add_argument("--dpi", type=int, default=300)
    arguments = parser.parse_args(argv)
    if min(arguments.modes, arguments.plot_modes, arguments.teleconnection_batch_size, arguments.dpi) < 1:
        parser.error("mode, batch and DPI settings must be positive")
    if arguments.plot_modes > arguments.modes:
        parser.error("--plot-modes cannot exceed --modes")
    if arguments.power_iterations < 0:
        parser.error("--power-iterations cannot be negative")
    return arguments


if __name__ == "__main__":
    run(parse_arguments())
