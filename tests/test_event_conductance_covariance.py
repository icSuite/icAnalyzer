import numpy as np
import torch

from scripts.IMAGE.vae_pipeline.validate_event_conductance_covariance import (
    decode_event, physical_covariance, covariance_checks, anchor_maps
)


class LinearDecoder:
    def eval(self):
        return self

    def decode(self, latent):
        first, second = latent[:, 0], latent[:, 1]
        return torch.stack((first, 2 * first + second, -second, first - 2 * second), dim=1).reshape(-1, 2, 1, 2)


class IdentityTransform:
    def inverse(self, values):
        return values


def test_event_covariance_matches_linear_gaussian_and_preserves_HP_cross_blocks():
    noise = np.random.default_rng(7).normal(size=(50000, 2)).astype(np.float32)
    support = np.ones((1, 2), dtype=bool)
    samples, latent = decode_event(LinearDecoder(), IdentityTransform(), np.array([2., -1.]), np.log([0.25, 4.]), noise, support)
    covariance, factor, maps, indices = physical_covariance(samples, support)
    jacobian = np.array([[1, 0], [2, 1], [0, -1], [1, -2]])
    expected = jacobian @ np.diag([0.25, 4.]) @ jacobian.T
    np.testing.assert_allclose(covariance, expected, atol=0.15)
    np.testing.assert_allclose(maps[0].ravel(), jacobian @ np.array([2., -1.]), atol=0.04)
    np.testing.assert_allclose(covariance, np.cov(samples.reshape(len(samples), -1), rowvar=False), atol=1e-12)
    np.testing.assert_allclose(maps[1].ravel()**2, np.diag(covariance), atol=1e-12)
    assert covariance[0, 3] > 0
    assert covariance[1, 2] < 0
    checks = covariance_checks(covariance, factor)
    assert checks["minimum_probe_quadratic_form"] >= 0
    assert checks["quadratic_form_factor_relative_error"] < 1e-12


def test_unsupported_placeholder_values_are_excluded_from_joint_state():
    samples = np.arange(24, dtype=float).reshape(3, 2, 2, 2)
    support = np.array([[True, False], [True, False]])
    samples[:, :, ~support] = np.nan
    covariance, factor, maps, indices = physical_covariance(samples, support)
    np.testing.assert_array_equal(indices, [0, 2, 4, 6])
    assert covariance.shape == (4, 4)
    assert np.isnan(maps[0][:, ~support]).all()
    np.testing.assert_allclose(covariance, np.cov(samples[:, :, support].reshape(3, 4), rowvar=False))
    correlations = anchor_maps(covariance, maps, support, (0, 0))
    assert np.isnan(correlations[:, :, ~support]).all()
    np.testing.assert_allclose(correlations[:, :, support], 1)


def test_nonlinear_decoding_can_give_covariance_rank_exceeding_latent_dimension():
    latent = np.linspace(-2, 2, 1000)
    samples = np.stack((latent, latent**2), axis=1).reshape(-1, 2, 1, 1)
    covariance, _, _, _ = physical_covariance(samples, np.ones((1, 1), dtype=bool))
    assert np.linalg.matrix_rank(covariance) == 2
