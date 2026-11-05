"""Focused numerical checks for cached CVAE postprocessing, no training."""

import numpy as np
import pytest
import torch

from test_automatic_cvae_loading import make_cache
from scripts.CVAE.prepare_cvae_hp_cache import AutomaticConductanceDataset
from scripts.IMAGE.vae_pipeline.postprocess_automatic_cvae import HeldoutFrames, parse_arguments, streaming_statistics


@pytest.mark.parametrize("log_space", [False, True])
def test_streaming_covariance_matches_direct_joint_covariance(tmp_path, log_space):
    samples = np.random.default_rng(4).uniform(0, 100, (23, 2, 3, 3)).astype(np.float32)
    samples[:, :, 0, 0] = 0.0  # Unsupported zero placeholders retain zero variance.
    filename = tmp_path / "samples.npy"
    np.save(filename, samples)
    mean, covariance = streaming_statistics(filename, batch_size=7, log_space=log_space)
    reference = samples.astype(np.float64)
    if log_space:
        reference = np.log10(reference + 1)
    np.testing.assert_allclose(mean, reference.mean(axis=0), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(covariance, np.cov(reference.reshape(23, -1), rowvar=False), rtol=1e-12, atol=1e-12)
    assert covariance[0, 0] == 0


def test_selected_frames_preserve_masks_quality_errors_and_identity(tmp_path):
    dataset = AutomaticConductanceDataset(make_cache(tmp_path), "train")
    support = np.ones((46, 46), dtype=bool)
    support[3, 3] = False
    try:
        indices = np.array([5, 1])
        selected = HeldoutFrames(dataset, indices, support)
        original = dataset[5]
        assert np.array_equal(selected.indices, indices)
        assert selected.orbits.tolist() == [1, 0]
        np.testing.assert_array_equal(selected.values[0], original["physical_target"].numpy())
        np.testing.assert_array_equal(selected.standard_deviation[0], original["physical_standard_deviation"].numpy())
        assert not selected.common_valid[0, 0, 3, 3]
        assert np.all(selected.encoder_input[0, :, 3, 3] == 0)
        assert torch.equal(selected[0]["encoder_input"], torch.from_numpy(selected.encoder_input[0]))
        np.testing.assert_array_equal(selected.weight[0, :, 4, 4], original["weight"].numpy()[:, 4, 4])
    finally:
        dataset.close()


def test_sampling_requires_explicit_kp_but_diagnostics_do_not():
    with pytest.raises(SystemExit):
        parse_arguments(["run", "--output", "new_output"])
    assert parse_arguments(["run", "--output", "new_output", "--diagnostics-only"]).kp is None
    assert parse_arguments(["run", "--output", "new_output", "--kp", "3"]).kp == 3
