"""Latent sampling, ensemble statistics, and sparse reconstruction."""

#%% Imports

import numpy as np
import torch


#%% Latent sampling

def sample_decoded_ensemble(
    model,
    transform,
    n_samples,
    batch_size=256,
    device="cpu",
    seed=0,
):
    """Decode exactly ``n_samples`` paired corrected-WIC/SI12 fields."""

    if n_samples < 2:
        raise ValueError("at least two latent samples are required")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    device = torch.device(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    model = model.to(device)
    model.eval()

    transformed_batches = []
    with torch.no_grad():
        for start in range(0, n_samples, batch_size):
            current_size = min(batch_size, n_samples - start)
            latent = torch.randn(
                current_size,
                model.latent_dim,
                generator=generator,
                device=device,
            )
            transformed_batches.append(model.decode(latent).cpu())

    transformed = torch.cat(transformed_batches, dim=0)
    physical = transform.inverse(transformed).numpy()
    if physical.shape[0] != n_samples:
        raise RuntimeError("latent sampler returned the wrong sample count")
    return physical.astype(np.float32, copy=False)


#%% Ensemble moments

def ensemble_statistics(samples):
    """Return a mean field and normalized anomaly factor.

    If ``samples`` has shape ``(sample, channel, y, x)``, the anomaly factor
    has shape ``(channel*y*x, sample)`` and satisfies ``cov = A @ A.T``.
    """

    samples = np.asarray(samples, dtype=float)
    if samples.ndim < 2 or samples.shape[0] < 2:
        raise ValueError("samples must contain at least two ensemble members")
    if np.any(~np.isfinite(samples)):
        raise ValueError("ensemble statistics require finite samples")

    mean = samples.mean(axis=0)
    flattened = samples.reshape(samples.shape[0], -1)
    anomaly_factor = (flattened - flattened.mean(axis=0)).T
    anomaly_factor = anomaly_factor / np.sqrt(samples.shape[0] - 1)
    return {
        "mean": mean,
        "anomaly_factor": anomaly_factor,
        "state_shape": samples.shape[1:],
        "sample_count": samples.shape[0],
        "variance": np.sum(anomaly_factor**2, axis=1).reshape(samples.shape[1:]),
    }


def covariance_blocks(anomaly_factor, field_size):
    """Materialize four covariance blocks for a two-field joint state."""

    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    if anomaly_factor.shape[0] != 2 * field_size:
        raise ValueError("anomaly_factor does not contain two equal fields")
    first = anomaly_factor[:field_size]
    second = anomaly_factor[field_size:]
    return {
        "11": first @ first.T,
        "12": first @ second.T,
        "21": second @ first.T,
        "22": second @ second.T,
    }


#%% Sparse reconstruction

def _observation_covariance(observation_error, n_observations):
    error = np.asarray(observation_error, dtype=float)
    if error.ndim == 0:
        if error <= 0:
            raise ValueError("observation variance must be positive")
        return np.eye(n_observations) * float(error)
    if error.ndim == 1:
        if error.shape != (n_observations,) or np.any(error <= 0):
            raise ValueError("observation variances must be positive and aligned")
        return np.diag(error)
    if error.shape != (n_observations, n_observations):
        raise ValueError("observation covariance has the wrong shape")
    if not np.allclose(error, error.T):
        raise ValueError("observation covariance must be symmetric")
    return error


def sparse_ensemble_reconstruction(
    prior_mean,
    anomaly_factor,
    observation_indices,
    observation_values,
    observation_error,
):
    """Condition a low-rank joint ensemble on sparse direct observations."""

    prior_mean = np.asarray(prior_mean, dtype=float)
    state_shape = prior_mean.shape
    mean_flat = prior_mean.reshape(-1)
    anomaly_factor = np.asarray(anomaly_factor, dtype=float)
    if anomaly_factor.ndim != 2 or anomaly_factor.shape[0] != mean_flat.size:
        raise ValueError("anomaly_factor shape is incompatible with prior_mean")

    indices = np.asarray(observation_indices, dtype=int)
    values = np.asarray(observation_values, dtype=float)
    if indices.ndim != 1 or values.shape != indices.shape:
        raise ValueError("observation indices and values must be aligned vectors")
    if np.any(indices < 0) or np.any(indices >= mean_flat.size):
        raise ValueError("observation index lies outside the joint state")
    if len(np.unique(indices)) != indices.size:
        raise ValueError("observation indices must be unique")

    prior_variance = np.sum(anomaly_factor**2, axis=1)
    if indices.size == 0:
        return {
            "mean": prior_mean.copy(),
            "standard_deviation": np.sqrt(prior_variance).reshape(state_shape),
            "prior_standard_deviation": np.sqrt(prior_variance).reshape(state_shape),
        }

    error_covariance = _observation_covariance(observation_error, indices.size)
    observed_anomalies = anomaly_factor[indices]
    innovation_covariance = (
        observed_anomalies @ observed_anomalies.T + error_covariance
    )
    innovation = values - mean_flat[indices]
    innovation_weights = np.linalg.solve(innovation_covariance, innovation)

    cross_covariance = anomaly_factor @ observed_anomalies.T
    posterior_mean = mean_flat + cross_covariance @ innovation_weights

    solved_cross = np.linalg.solve(innovation_covariance, cross_covariance.T).T
    variance_reduction = np.sum(cross_covariance * solved_cross, axis=1)
    posterior_variance = np.clip(prior_variance - variance_reduction, 0.0, None)

    return {
        "mean": posterior_mean.reshape(state_shape),
        "standard_deviation": np.sqrt(posterior_variance).reshape(state_shape),
        "prior_standard_deviation": np.sqrt(prior_variance).reshape(state_shape),
    }
