import numpy as np

from scripts.IMAGE.vae_pipeline.validate_conductance_prior import (
    aggregate_posterior, cell_weights, orbit_weights, physical_summary,
    saved_members, weighted_correlation
)


def test_aggregate_posterior_includes_conditional_variance_and_orbit_mass():
    mu = np.array([[0., 0.], [2., 4.], [2., 4.]])
    logvar = np.log(np.full_like(mu, 0.5))
    weights = orbit_weights(np.array([1, 2, 2]))
    mean, covariance = aggregate_posterior(mu, logvar, weights)
    np.testing.assert_allclose(weights, [0.5, 0.25, 0.25])
    np.testing.assert_allclose(mean, [1, 2])
    np.testing.assert_allclose(covariance, [[1.5, 2], [2, 4.5]])


def test_mask_quality_and_area_are_identical_for_each_field_family():
    valid = np.array([[[True, False]], [[True, True]]])
    area = np.array([[1., 3.]])
    quality = np.array([[[1., 100.]], [[1., 0.]]])
    np.testing.assert_allclose(cell_weights(valid, area), [[[1, 0]], [[0.25, 0.75]]])
    np.testing.assert_allclose(cell_weights(valid, area, quality), [[[1, 0]], [[1, 0]]])
    fields = np.array([[[[1., 1e9]], [[2., 1e9]]], [[[3., 100.]], [[6., 200.]]]])
    summary, rows = physical_summary(fields, valid, quality, area, np.array([0.5, 0.5]), np.array([1, 2]), "quality")
    assert summary["Hall"]["mean_S"] == 2
    assert summary["Pedersen"]["mean_S"] == 4
    assert len(rows) == 4


def test_quality_zero_frames_do_not_bias_orbit_balancing():
    fields = np.ones((3, 2, 1, 1))
    fields[2] = 3
    valid = np.ones((3, 1, 1), dtype=bool)
    quality = np.array([[[1]], [[0]], [[1]]])
    summary, _ = physical_summary(fields, valid, quality, np.ones((1, 1)), orbit_weights([1, 1, 2]), np.array([1, 1, 2]), "quality")
    assert summary["Hall"]["mean_S"] == 2


def test_pairwise_correlation_ignores_missing_values():
    x = np.array([1., 2., np.nan, 4.])
    y = np.array([2., 4., 1e9, 8.])
    assert np.isclose(weighted_correlation(x, y, np.ones(4)), 1)


def test_saved_members_reconstruct_entire_paired_fields_in_both_storage_orders(tmp_path):
    samples = np.arange(40, dtype=float).reshape(5, 2, 2, 2)
    mean = samples.mean(axis=0)
    factor = (samples.reshape(5, -1).T - mean.ravel()[:, None]) / np.sqrt(4)
    for fortran in (False, True):
        values = np.asfortranarray(factor) if fortran else factor
        np.savez_compressed(tmp_path / "ensemble_products.npz", conductance_mean=mean, conductance_anomaly_factor=values)
        np.testing.assert_allclose(saved_members(tmp_path, [4, 0, 2]), samples[[4, 0, 2]])
