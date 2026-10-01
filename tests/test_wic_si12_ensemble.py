import numpy as np
import pytest

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.ensemble import (
    correct_wic_roundoff,
    covariance_blocks,
    ensemble_statistics,
    nonnegative_wic_floor,
    sample_decoded_ensemble,
    sparse_ensemble_reconstruction,
)
from icanalyzer.wic_si12.model import WICSI12VAE


def test_prior_sampling_preserves_pairing_and_nonnegative_wic():
    transform = ChannelTransform(scale=[100, 20], center=[0.5, 0], spread=[2, 1])
    floor = float(-transform.center[0] / transform.spread[0])
    model = WICSI12VAE(latent_dim=2, wic_floor_transformed=floor)
    samples, latent = sample_decoded_ensemble(
        model, transform, n_samples=5, batch_size=2, seed=4, return_latent=True
    )
    assert samples.shape == (5, 2, 46, 46)
    assert latent.shape == (5, 2)
    assert np.all(samples[:, 0] >= 0)


def test_float32_wic_boundary_is_safe_and_only_roundoff_is_corrected():
    transform = ChannelTransform(
        scale=[21.64110565185547, 1.7304844856262207],
        center=[0.9756972789764404, 1.1206520795822144],
        spread=[1.518889307975769, 1.1444892883300781],
    )
    floor = nonnegative_wic_floor(transform)
    transformed = np.zeros((1, 2, 1, 1), dtype=np.float32)
    transformed[:, 0] = floor
    assert transform.inverse(transformed)[0, 0, 0, 0] >= 0

    samples = np.zeros((1, 2, 1, 2), dtype=np.float32)
    samples[0, 0, 0] = [-1.2899104e-6, 2.0]
    corrected, diagnostics = correct_wic_roundoff(samples, transform)
    np.testing.assert_array_equal(corrected[0, 0, 0], [0.0, 2.0])
    assert diagnostics["roundoff_corrected_cell_count"] == 1
    assert samples[0, 0, 0, 0] < 0


def test_materially_negative_wic_is_not_hidden_as_roundoff():
    transform = ChannelTransform(scale=[20, 2], center=[1, 0], spread=[1.5, 1])
    samples = np.zeros((1, 2, 1, 1), dtype=np.float32)
    samples[0, 0, 0, 0] = -0.01

    with pytest.raises(ValueError, match="materially negative"):
        correct_wic_roundoff(samples, transform)


def test_ensemble_factor_and_cross_block_transpose():
    generator = np.random.default_rng(7)
    samples = generator.normal(size=(20, 2, 3, 3))
    statistics = ensemble_statistics(samples)
    dense = statistics["anomaly_factor"] @ statistics["anomaly_factor"].T
    expected = np.cov(samples.reshape(20, -1), rowvar=False)
    np.testing.assert_allclose(dense, expected)

    blocks = covariance_blocks(statistics["anomaly_factor"], field_size=9)
    np.testing.assert_allclose(blocks["12"], blocks["21"].T)


def test_no_observations_returns_prior():
    mean = np.arange(6, dtype=float).reshape(2, 3)
    anomaly = np.eye(6)
    result = sparse_ensemble_reconstruction(mean, anomaly, [], [], 1.0)
    np.testing.assert_array_equal(result["mean"], mean)
    np.testing.assert_allclose(result["standard_deviation"], 1.0)


def test_sparse_update_moves_cross_correlated_unobserved_state():
    mean = np.zeros(2)
    anomaly = np.array([[1.0, 0.0], [0.8, 0.6]])
    result = sparse_ensemble_reconstruction(
        mean,
        anomaly,
        observation_indices=[0],
        observation_values=[2.0],
        observation_error=1e-6,
    )
    assert result["mean"][0] > 1.99
    assert result["mean"][1] > 1.5
    assert np.all(result["standard_deviation"] <= result["prior_standard_deviation"])


def test_full_near_exact_observations_recover_state():
    mean = np.zeros(3)
    anomaly = np.eye(3)
    truth = np.array([1.0, -2.0, 3.0])
    result = sparse_ensemble_reconstruction(
        mean,
        anomaly,
        observation_indices=[0, 1, 2],
        observation_values=truth,
        observation_error=1e-12,
    )
    np.testing.assert_allclose(result["mean"], truth, atol=1e-10)


def test_larger_observation_error_weakens_update():
    mean = np.zeros(2)
    anomaly = np.array([[1.0, 0.0], [0.8, 0.6]])
    precise = sparse_ensemble_reconstruction(mean, anomaly, [0], [2.0], 0.01)
    noisy = sparse_ensemble_reconstruction(mean, anomaly, [0], [2.0], 100.0)
    assert precise["mean"][0] > noisy["mean"][0]
    assert precise["mean"][1] > noisy["mean"][1]
