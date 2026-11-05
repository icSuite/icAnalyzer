"""Figure 2: saved WIC/SI12 mean and four joint, one-sigma covariance patterns."""

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
from scipy import linalg
from scipy.linalg.blas import dsyrk


#%% Joint covariance and exact leading PCA components

def decompose_factor(factor, state_shape, batch_size=1024, n_modes=4):
    """Accumulate C = A A.T in sample batches; never form sample-by-sample C."""

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if factor.ndim != 2 or factor.shape[0] != np.prod(state_shape) or factor.shape[1] < 2:
        raise ValueError("anomaly factor must match the grid and contain at least two samples")
    if not 1 <= n_modes <= min(factor.shape[0], factor.shape[1] - 1):
        raise ValueError("requested PCA mode count exceeds the ensemble/grid rank limit")
    covariance = np.zeros((factor.shape[0], factor.shape[0]), dtype=np.float64, order="F")
    for start in range(0, factor.shape[1], batch_size):
        block = np.array(factor[:, start:start + batch_size], dtype=np.float64, order="F")
        if not np.isfinite(block).all():
            raise ValueError("saved anomaly factor contains nonfinite values")
        # Symmetric BLAS update stores only the lower triangle, in place.
        covariance = dsyrk(alpha=1, a=block, beta=1, c=covariance, lower=1, overwrite_c=1)
    total_variance = float(np.trace(covariance))
    if total_variance <= 0:
        raise ValueError("saved ensemble has no variance")

    first = covariance.shape[0] - n_modes
    eigenvalues, eigenvectors = linalg.eigh(covariance, lower=True, subset_by_index=(first, covariance.shape[0] - 1), overwrite_a=True, check_finite=False)
    eigenvalues = np.maximum(eigenvalues[::-1], 0)
    eigenvectors = eigenvectors[:, ::-1].T.copy()
    # Fix each arbitrary PCA sign jointly, never separately by channel.
    pivots = np.argmax(np.abs(eigenvectors), axis=1)
    eigenvectors *= np.where(eigenvectors[np.arange(n_modes), pivots] < 0, -1, 1)[:, None]
    patterns = eigenvectors * np.sqrt(eigenvalues)[:, None]
    return {
        "patterns": patterns.reshape(n_modes, *state_shape),
        "eigenvalues": eigenvalues,
        "explained_variance_fraction": eigenvalues / total_variance,
        "total_variance": total_variance,
        "sample_count": factor.shape[1]
    }


def calculate_pca(run_directory, batch_size=1024):
    """Unpack only A to temporary disk, then memory-map it for batched access."""

    filename = Path(run_directory).expanduser() / "ensemble_products.npz"
    with np.load(filename, allow_pickle=False) as archive:
        mean = archive["fuv_mean"]
    if mean.ndim != 3 or mean.shape[0] != 2 or not np.isfinite(mean).all():
        raise ValueError("FUV mean must be finite with shape (2, rows, columns)")
    with tempfile.TemporaryDirectory(prefix="wic-si12-pca-") as workspace:
        factor_path = Path(workspace) / "factor.npy"
        with zipfile.ZipFile(filename) as archive:
            with archive.open("fuv_anomaly_factor.npy") as source, factor_path.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        factor = np.load(factor_path, mmap_mode="r", allow_pickle=False)
        result = decompose_factor(factor, mean.shape, batch_size)
        del factor
    return {"mean": mean, **result}


#%% Axes-free 2x5 presentation layout

def make_figure(mean, patterns, explained_variance_fraction, mean_maxima=(None, None), mode_maxima=(None, None)):
    figure, axes = plt.subplots(2, 5, figsize=(12.5, 5))
    for channel in range(2):
        mean_max = mean_maxima[channel]
        mode_max = mode_maxima[channel]
        mean_max = max(float(mean[channel].max()), np.finfo(float).eps) if mean_max is None else mean_max
        mode_max = max(float(np.abs(patterns[:, channel]).max()), np.finfo(float).eps) if mode_max is None else mode_max
        if not np.isfinite([mean_max, mode_max]).all() or min(mean_max, mode_max) <= 0:
            plt.close(figure)
            raise ValueError("colour-scale maxima must be finite and positive")
        axes[channel, 0].imshow(mean[channel], origin="lower", cmap="viridis", vmin=0, vmax=mean_max, interpolation="nearest")
        for mode in range(4):
            axes[channel, mode + 1].imshow(patterns[mode, channel], origin="lower", cmap="RdBu_r", vmin=-mode_max, vmax=mode_max, interpolation="nearest")
    for channel, sensor in enumerate(("WIC", "SI12")):
        labels = [f"{sensor} ensemble mean"]
        labels += [f"PC {mode + 1} : {100 * fraction:.2f}%" for mode, fraction in enumerate(explained_variance_fraction)]
        for column, label in enumerate(labels):
            axis = axes[channel, column]
            axis.set_axis_off()
            text_y = 1.04 if channel == 0 else -0.04
            alignment = "bottom" if channel == 0 else "top"
            axis.text(0.5, text_y, label, transform=axis.transAxes, ha="center", va=alignment, fontsize=12, clip_on=False)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.02, hspace=0.02)
    return figure


def run(arguments):
    result = calculate_pca(arguments.run_directory, arguments.batch_size)
    figure = make_figure(result["mean"], result["patterns"], result["explained_variance_fraction"], (arguments.wic_mean_max, arguments.si12_mean_max), (arguments.wic_mode_max, arguments.si12_mode_max))
    output = arguments.output.expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=arguments.dpi, bbox_inches="tight", pad_inches=0)
    plt.close(figure)
    data_path = output.with_name(output.stem + "_pca.npz")
    np.savez_compressed(data_path, **result)
    fractions = 100 * result["explained_variance_fraction"]
    print(f"Saved {output} and {data_path}; {result['sample_count']} draws, joint raw-count covariance.")
    print("Variance explained by PC1-PC4 (%): " + ", ".join(f"{value:.2f}" for value in fractions))


#%% Command-line settings

def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1024, help="samples per covariance accumulation batch")
    parser.add_argument("--wic-mean-max", type=float)
    parser.add_argument("--si12-mean-max", type=float)
    parser.add_argument("--wic-mode-max", type=float, help="symmetric +/- counts limit shared by all four WIC PCA panels")
    parser.add_argument("--si12-mode-max", type=float, help="symmetric +/- counts limit shared by all four SI12 PCA panels")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


if __name__ == "__main__":
    run(parse_arguments())
