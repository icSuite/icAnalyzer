"""Known paired-member reconstruction, support and covariance ordering."""

import numpy as np

from scripts.IMAGE.vae_pipeline.load_conductance_ensemble import joint_moments, recover_members


def test_saved_factor_recovers_samples_and_joint_moments():
    physical = np.array([[[[1., 3., 0.]], [[2., 1., 0.]]],
                         [[[3., 1., 0.]], [[6., 2., 0.]]],
                         [[[2., 2., 0.]], [[4., 4., 0.]]]])
    support = np.array([[True, True, False]])
    saved_mean = physical.mean(axis=0)
    factor = (physical.reshape(3, -1) - saved_mean.ravel()).T / np.sqrt(2)
    samples = recover_members(saved_mean, factor, support)
    np.testing.assert_allclose(samples[:, :, support], physical[:, :, support])
    assert np.isnan(samples[:, :, ~support]).all()
    mean, covariance, indices = joint_moments(samples, support, block_size=2)
    np.testing.assert_allclose(mean[:, support], saved_mean[:, support])
    np.testing.assert_allclose(covariance, np.cov(physical[:, :, support].reshape(3, -1), rowvar=False))
    np.testing.assert_array_equal(indices, [0, 1, 3, 4])
    assert covariance[0, 2] == 2.0
