"""Numerical diagnostics for saved WIC/SI12 ensemble runs."""

#%% Imports

import numpy as np
from scipy import linalg


#%% Ensemble reconstruction

def reconstruct_ensemble(mean, anomaly_factor, sample_indices=None):
    """Recover all or selected saved members from a mean and anomaly factor."""

    mean = np.asarray(mean, dtype=float)
    anomaly_factor = np.asarray(anomaly_factor)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[0] != mean.size:
        raise ValueError("anomaly_factor shape is incompatible with mean")
    sample_count = anomaly_factor.shape[1]
    if sample_count < 2:
        raise ValueError("anomaly_factor must contain at least two samples")

    if sample_indices is None:
        sample_indices = np.arange(sample_count)
    sample_indices = np.asarray(sample_indices, dtype=int)
    if sample_indices.ndim != 1:
        raise ValueError("sample_indices must be one-dimensional")
    if np.any(sample_indices < 0) or np.any(sample_indices >= sample_count):
        raise ValueError("sample index lies outside the saved ensemble")

    selected = anomaly_factor[:, sample_indices].T
    flattened = mean.reshape(1, -1) + np.sqrt(sample_count - 1) * selected
    return flattened.reshape(sample_indices.size, *mean.shape)


def anomaly_variance(anomaly_factor):
    """Return row variances without allocating ``anomaly_factor**2``."""

    anomaly_factor = np.asarray(anomaly_factor)
    if anomaly_factor.ndim != 2:
        raise ValueError("anomaly_factor must be a matrix")
    return np.einsum(
        "ij,ij->i", anomaly_factor, anomaly_factor,
        dtype=np.float64, optimize=True
    )


#%% Covariance and correlation

def joint_correlation_maps(
    anomaly_factor,
    state_shape,
    anchor_row,
    anchor_column,
):
    """Return the four two-field correlation maps at one spatial anchor."""

    state_shape = tuple(int(value) for value in state_shape)
    if len(state_shape) != 3 or state_shape[0] != 2:
        raise ValueError("state_shape must be (2, rows, columns)")
    _, rows, columns = state_shape
    if not 0 <= anchor_row < rows or not 0 <= anchor_column < columns:
        raise ValueError("anchor lies outside the state grid")

    anomaly_factor = np.asarray(anomaly_factor)
    if anomaly_factor.shape[0] != int(np.prod(state_shape)):
        raise ValueError("anomaly_factor shape is incompatible with state_shape")

    field_size = rows * columns
    variance = anomaly_variance(anomaly_factor)
    maps = np.empty((2, 2, rows, columns), dtype=float)
    for source_channel in range(2):
        anchor = source_channel * field_size + anchor_row * columns + anchor_column
        covariance = anomaly_factor @ anomaly_factor[anchor]
        scale = np.sqrt(variance[anchor] * variance)
        correlation = np.zeros_like(covariance)
        np.divide(covariance, scale, out=correlation, where=scale > 0)
        maps[source_channel] = np.clip(
            correlation.reshape(state_shape), -1.0, 1.0
        )
    return maps


def maximum_variance_anchor(anomaly_factor, state_shape, channel=0):
    """Return the row and column of maximum variance in one field."""

    state_shape = tuple(int(value) for value in state_shape)
    if len(state_shape) != 3 or state_shape[0] != 2:
        raise ValueError("state_shape must be (2, rows, columns)")
    if channel not in (0, 1):
        raise ValueError("channel must be 0 or 1")

    variance = anomaly_variance(anomaly_factor)
    variance = variance.reshape(state_shape)
    return tuple(int(value) for value in np.unravel_index(
        np.argmax(variance[channel]), state_shape[1:]
    ))


#%% Low-rank modes

def covariance_spectrum(anomaly_factor):
    """Return the exact spectrum using the smaller covariance orientation."""

    anomaly_factor = np.asarray(anomaly_factor)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[1] < 2:
        raise ValueError("anomaly_factor must be a state-by-sample matrix")
    if anomaly_factor.shape[0] <= anomaly_factor.shape[1]:
        gram = anomaly_factor @ anomaly_factor.T
    else:
        gram = anomaly_factor.T @ anomaly_factor
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    tolerance = max(float(eigenvalues[0]), 1.0) * 1e-12
    return np.clip(eigenvalues[eigenvalues > tolerance], 0.0, None)


def covariance_mode_decomposition(
    anomaly_factor,
    state_shape,
    n_modes=4,
    random_seed=0,
    exact_max_dimension=512,
    oversampling=12,
    power_iterations=1,
):
    """Return exact small or randomized large covariance modes.

    The randomized path never forms a sample-by-sample or state-by-state
    covariance matrix. Its memory is linear in the sample count.
    """

    anomaly_factor = np.asarray(anomaly_factor)
    state_shape = tuple(int(value) for value in state_shape)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[0] != np.prod(state_shape):
        raise ValueError("anomaly_factor shape is incompatible with state_shape")
    rank_limit = min(anomaly_factor.shape[0], anomaly_factor.shape[1] - 1)
    n_modes = min(int(n_modes), rank_limit)
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")
    for start in range(0, anomaly_factor.shape[0], 256):
        if not np.isfinite(anomaly_factor[start:start + 256]).all():
            raise ValueError("anomaly_factor contains nonfinite values")

    total_variance = float(np.sum(anomaly_variance(anomaly_factor)))
    smaller_dimension = min(anomaly_factor.shape)
    if smaller_dimension <= int(exact_max_dimension):
        if anomaly_factor.shape[0] <= anomaly_factor.shape[1]:
            covariance = anomaly_factor @ anomaly_factor.T
            first = covariance.shape[0] - n_modes
            eigenvalues, vectors = linalg.eigh(
                covariance,
                subset_by_index=(first, covariance.shape[0] - 1),
                check_finite=False,
            )
            modes = vectors
        else:
            covariance = anomaly_factor.T @ anomaly_factor
            first = covariance.shape[0] - n_modes
            eigenvalues, vectors = linalg.eigh(
                covariance,
                subset_by_index=(first, covariance.shape[0] - 1),
                check_finite=False,
            )
            modes = anomaly_factor @ vectors
            modes /= np.sqrt(
                np.maximum(eigenvalues, np.finfo(eigenvalues.dtype).eps)
            )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.clip(eigenvalues[order], 0.0, None)
        modes = modes[:, order].T
        method = "exact_smaller_gram"
    else:
        projected_dimension = min(
            rank_limit, n_modes + max(int(oversampling), 2)
        )
        dtype = np.float32 if anomaly_factor.dtype.itemsize <= 4 else np.float64
        generator = np.random.default_rng(random_seed)
        random_projection = generator.standard_normal(
            (anomaly_factor.shape[1], projected_dimension), dtype=dtype
        )
        projected = anomaly_factor @ random_projection
        for _ in range(max(int(power_iterations), 0)):
            basis, _triangular = np.linalg.qr(projected, mode="reduced")
            projected = anomaly_factor @ (anomaly_factor.T @ basis)
        basis, _triangular = np.linalg.qr(projected, mode="reduced")
        compressed = basis.T @ anomaly_factor
        left, singular_values, _right = linalg.svd(
            compressed,
            full_matrices=False,
            overwrite_a=True,
            check_finite=False,
        )
        eigenvalues = np.square(singular_values[:n_modes])
        modes = (basis @ left[:, :n_modes]).T
        method = "randomized_svd"

    return {
        "eigenvalues": eigenvalues,
        "modes": modes.reshape(n_modes, *state_shape),
        "total_variance": total_variance,
        "captured_variance_fraction": float(np.sum(eigenvalues) / total_variance),
        "method": method,
    }


def leading_covariance_modes(anomaly_factor, state_shape, n_modes=4):
    """Return leading covariance eigenvalues and state-space eigenmodes."""

    decomposition = covariance_mode_decomposition(
        anomaly_factor, state_shape, n_modes=n_modes
    )
    return decomposition["eigenvalues"], decomposition["modes"]


#%% Training history

def best_validation_epoch(history):
    """Return the one-based epoch and metrics at minimum validation total loss."""

    candidates = [record for record in history if "validation" in record]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda record: record["validation"]["total"])
    return int(best["epoch"]), dict(best["validation"])
