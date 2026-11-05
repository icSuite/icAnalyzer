"""Figures 3a--c: single-anchor correlations and joint explained variance."""

#%% Imports

import argparse
from pathlib import Path
import shutil
import tempfile
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


#%% Covariance connections from the saved ensemble factor

def teleconnection_maps(factor, state_shape, grid_id, batch_size=64):
    """Use both channel anchors at a row-major spatial ID; no dense covariance."""

    if len(state_shape) != 3 or state_shape[0] != 2:
        raise ValueError("state shape must be (2, rows, columns)")
    field_size = int(np.prod(state_shape[1:]))
    if not 0 <= grid_id < field_size:
        raise ValueError(f"grid ID must lie between 0 and {field_size - 1}")
    if factor.ndim != 2 or factor.shape[0] != 2 * field_size or factor.shape[1] < 2:
        raise ValueError("saved anomaly factor does not match the grid")
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    anchors = np.asarray(factor[[grid_id, field_size + grid_id]], dtype=np.float64)
    anchor_covariance = anchors @ anchors.T
    anchor_variance = np.diag(anchor_covariance)
    if not np.isfinite(anchors).all() or np.any(anchor_variance <= 0):
        raise ValueError("chosen location has zero/nonfinite variance in either field; choose another grid ID")
    anchor_std = np.sqrt(anchor_variance)
    anchor_correlation = anchor_covariance / np.outer(anchor_std, anchor_std)
    anchor_correlation = np.clip(anchor_correlation, -1, 1)
    # Standardized predictors avoid unequal field-scale conditioning problems.
    # A pseudoinverse handles perfectly redundant anchors without double counting.
    inverse = np.linalg.pinv(anchor_correlation, rcond=1e-12, hermitian=True)
    correlations = np.full((2 * field_size, 2), np.nan)
    jointly_explained = np.full(2 * field_size, np.nan)
    for start in range(0, factor.shape[0], batch_size):
        stop = min(start + batch_size, factor.shape[0])
        block = np.asarray(factor[start:stop], dtype=np.float64)
        if not np.isfinite(block).all():
            raise ValueError("saved anomaly factor contains nonfinite values")
        variance = np.einsum("ij,ij->i", block, block)
        covariance = block @ anchors.T
        scale = np.sqrt(variance[:, None]) * anchor_std
        r = np.full(covariance.shape, np.nan)
        np.divide(covariance, scale, out=r, where=scale > 0)
        r = np.clip(r, -1, 1)
        correlations[start:stop] = r
        # R^2 = c.T inv(C_anchor) c / Var(target) = r.T inv(R_anchor) r.
        jointly_explained[start:stop] = np.clip(np.einsum("ij,jk,ik->i", r, inverse, r), 0, 1)
    return {
        "correlations": correlations.T.reshape(2, *state_shape),
        "joint_explained_variance": jointly_explained.reshape(state_shape),
        "anchor_correlation": anchor_correlation,
        "grid_id": grid_id,
        "anchor_row_column": np.array(divmod(grid_id, state_shape[2])),
        "sample_count": factor.shape[1]
    }


def calculate_teleconnections(run_directory, grid_id, batch_size=64, field_prefix="fuv"):
    if field_prefix not in ("fuv", "conductance"):
        raise ValueError("teleconnection fields must be fuv or conductance")
    filename = Path(run_directory).expanduser() / "ensemble_products.npz"
    with np.load(filename, allow_pickle=False) as archive:
        state_shape = archive[f"{field_prefix}_mean"].shape
        support = archive["conductance_supported"] if field_prefix == "conductance" and "conductance_supported" in archive else None
    if len(state_shape) != 3 or state_shape[0] != 2 or not 0 <= grid_id < np.prod(state_shape[1:]):
        raise ValueError("grid ID must index a cell of the saved two-channel spatial grid")
    if support is not None:
        if support.shape != state_shape[1:] or support.dtype != bool:
            raise ValueError("conductance support must be a boolean spatial-grid mask")
        if not support.ravel()[grid_id]:
            raise ValueError("chosen grid ID is outside the supported training domain; choose another location")
    with tempfile.TemporaryDirectory(prefix="wic-si12-teleconnections-") as workspace:
        factor_path = Path(workspace) / "factor.npy"
        with zipfile.ZipFile(filename) as archive:
            with archive.open(f"{field_prefix}_anomaly_factor.npy") as source, factor_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        factor = np.load(factor_path, mmap_mode="r", allow_pickle=False)
        result = teleconnection_maps(factor, state_shape, grid_id, batch_size)
        del factor
    if support is not None:
        result["correlations"][:, :, ~support] = np.nan
        result["joint_explained_variance"][:, ~support] = np.nan
        result["conductance_supported"] = support
    return result


#%% Matching WIC/SI12 panels and fixed colour scales

def make_figure(maps, anchor_row_column, joint=False, colorbar=True, field_names=("WIC", "SI12")):
    figure, axes = plt.subplots(2, 1, figsize=(4.5, 8.5), layout="constrained")
    row, column = anchor_row_column
    for channel, sensor in enumerate(field_names):
        image = axes[channel].imshow(maps[channel], origin="lower", cmap="viridis" if joint else "RdBu_r", vmin=0 if joint else -1, vmax=1, interpolation="nearest")
        axes[channel].plot(column, row, marker="o", markersize=6, markerfacecolor="none", markeredgecolor="black", markeredgewidth=1.5)
        axes[channel].set_axis_off()
        text_y = 1.02 if channel == 0 else -0.02
        alignment = "bottom" if channel == 0 else "top"
        axes[channel].text(0.5, text_y, sensor, transform=axes[channel].transAxes, ha="center", va=alignment, fontsize=12)
    if colorbar:
        label = "Jointly explained variance" if joint else "Correlation"
        figure.colorbar(image, ax=axes, shrink=0.8, label=label)
    return figure


def run(arguments, field_prefix="fuv", field_names=("WIC", "SI12")):
    result = calculate_teleconnections(arguments.run_directory, arguments.grid_id, arguments.batch_size, field_prefix)
    output = arguments.output.expanduser()
    output.mkdir(parents=True, exist_ok=True)
    maps = [result["correlations"][0], result["correlations"][1], result["joint_explained_variance"]]
    for name, values in zip(("figure_3a", "figure_3b", "figure_3c"), maps):
        figure = make_figure(values, result["anchor_row_column"], joint=name == "figure_3c", colorbar=not arguments.no_colorbar, field_names=field_names)
        figure.savefig(output / f"{name}.png", dpi=arguments.dpi, bbox_inches="tight")
        plt.close(figure)
    np.savez_compressed(output / "figure_3_teleconnections.npz", field_names=np.array(field_names), **result)
    print(f"Saved figure_3a/b/c.png in {output}; grid ID {arguments.grid_id}, row/column {tuple(result['anchor_row_column'])}; {result['sample_count']} stored draws.")
    return result


#%% Command-line settings

def argument_parser(description=__doc__):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--grid-id", type=int, required=True, help="zero-based row-major spatial ID (0--2115 for a 46x46 grid), without a channel offset")
    parser.add_argument("--output", type=Path, required=True, help="directory for figure_3a.png, figure_3b.png and figure_3c.png")
    parser.add_argument("--batch-size", type=int, default=64, help="state rows per factor batch")
    parser.add_argument("--no-colorbar", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser


def parse_arguments(argv=None, description=__doc__):
    return argument_parser(description).parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
