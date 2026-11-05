"""Streaming covariance must agree with direct sample covariance."""

import numpy as np

from scripts.IMAGE.vae_pipeline.validate_event_correlation_convergence import correlations, update_moments


def test_streaming_rows_match_direct_covariance_with_offset_and_constant_state():
    fields = np.random.default_rng(35).normal(size=(123, 6)) + 10000
    fields[:, 5] = 7
    anchors = np.array([1, 3])
    count = 0
    mean, variance_sum = np.zeros(6), np.zeros(6)
    cross_sum = np.zeros((2, 6))
    for block in np.array_split(fields, 7):
        count = update_moments(count, mean, variance_sum, cross_sum, block, anchors)
    covariance = np.cov(fields, rowvar=False)
    variance = variance_sum / (count - 1)
    rows = cross_sum / (count - 1)
    np.testing.assert_allclose(mean, fields.mean(axis=0), rtol=1e-12)
    np.testing.assert_allclose(variance, np.diag(covariance), atol=2e-12)
    np.testing.assert_allclose(rows, covariance[anchors], atol=2e-12)
    sampled = correlations(variance, rows, anchors)
    assert np.isnan(sampled[:, 5]).all()
    np.testing.assert_allclose(sampled[np.arange(2), anchors], 1)
