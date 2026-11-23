import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/CVAE/diagnose_native_cvae_resolution.py"
)
SPEC = importlib.util.spec_from_file_location("native_resolution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_sensor_metrics_separate_coarse_and_fine_reconstruction():
    coarse = np.arange(16, dtype=float).reshape(4, 4)
    coarse = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
    pattern = np.tile(np.array([[-1.0, -0.5, 0.5, 1.0]] * 4), (4, 4))
    observed = coarse + pattern
    reconstructed = coarse + 0.5 * pattern

    metrics = MODULE.sensor_metrics(
        observed, reconstructed, np.ones_like(observed, dtype=bool)
    )

    assert metrics["coarse_normalized_rmse"] == 0
    np.testing.assert_allclose(metrics["detail_correlation"], 1)
    np.testing.assert_allclose(metrics["detail_amplitude_ratio"], 0.5)


def test_standard_normal_posterior_has_zero_kl():
    zeros = np.zeros((3, 5))
    np.testing.assert_array_equal(MODULE.kl_per_dimension(zeros, zeros), zeros)
