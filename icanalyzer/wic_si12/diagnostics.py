"""Numerical diagnostics for saved WIC/SI12 ensemble runs."""

#%% Imports

import numpy as np
from scipy import linalg


#%% Ensemble reconstruction

def reconstruct_ensemble(mean, anomaly_factor):
    """Recover the exact saved ensemble from its mean and anomaly factor."""

    mean = np.asarray(mean, dtype=float)
    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[0] != mean.size:
        raise ValueError("anomaly_factor shape is incompatible with mean")
    sample_count = anomaly_factor.shape[1]
    if sample_count < 2:
        raise ValueError("anomaly_factor must contain at least two samples")

    flattened = mean.reshape(1, -1)
    flattened = flattened + np.sqrt(sample_count - 1) * anomaly_factor.T
    return flattened.reshape(sample_count, *mean.shape)


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

    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    if anomaly_factor.shape[0] != int(np.prod(state_shape)):
        raise ValueError("anomaly_factor shape is incompatible with state_shape")

    field_size = rows * columns
    variance = np.sum(anomaly_factor**2, axis=1)
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

    variance = np.sum(np.asarray(anomaly_factor, dtype=float)**2, axis=1)
    variance = variance.reshape(state_shape)
    return tuple(int(value) for value in np.unravel_index(
        np.argmax(variance[channel]), state_shape[1:]
    ))


#%% Low-rank modes

def covariance_spectrum(anomaly_factor):
    """Return descending nonzero covariance eigenvalues without dense state covariance."""

    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[1] < 2:
        raise ValueError("anomaly_factor must be a state-by-sample matrix")
    gram = anomaly_factor.T @ anomaly_factor
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    tolerance = max(float(eigenvalues[0]), 1.0) * 1e-12
    return np.clip(eigenvalues[eigenvalues > tolerance], 0.0, None)


def leading_covariance_modes(anomaly_factor, state_shape, n_modes=4):
    """Return leading covariance eigenvalues and state-space eigenmodes."""

    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    state_shape = tuple(int(value) for value in state_shape)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[0] != np.prod(state_shape):
        raise ValueError("anomaly_factor shape is incompatible with state_shape")
    n_modes = min(int(n_modes), anomaly_factor.shape[1] - 1)
    if n_modes <= 0:
        raise ValueError("n_modes must be positive")

    gram = anomaly_factor.T @ anomaly_factor
    first = gram.shape[0] - n_modes
    eigenvalues, right_vectors = linalg.eigh(
        gram,
        subset_by_index=(first, gram.shape[0] - 1),
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    right_vectors = right_vectors[:, order]

    modes = anomaly_factor @ right_vectors
    modes = modes / np.sqrt(np.maximum(eigenvalues, np.finfo(float).eps))
    modes = modes.T.reshape(n_modes, *state_shape)
    return eigenvalues, modes


#%% Training history

def best_validation_epoch(history):
    """Return the one-based epoch and metrics at minimum validation total loss."""

    candidates = [record for record in history if "validation" in record]
    if not candidates:
        return None, None
    best = min(candidates, key=lambda record: record["validation"]["total"])
    return int(best["epoch"]), dict(best["validation"])
