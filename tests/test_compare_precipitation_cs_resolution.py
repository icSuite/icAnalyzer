import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/CVAE/compare_precipitation_cs_resolution.py"
)
SPEC = importlib.util.spec_from_file_location("compare_cs_resolution", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_nested_block_mean_and_comparison_metrics():
    coarse = np.arange(6, dtype=float).reshape(2, 3)
    fine = np.repeat(np.repeat(coarse, 4, axis=0), 4, axis=1)
    subcell_pattern = np.tile(
        np.array([[-1.0, -0.5, 0.5, 1.0]] * 4), (2, 3)
    )
    fine += subcell_pattern

    metrics = MODULE.comparison_metrics(coarse, fine)

    np.testing.assert_allclose(metrics["fine_block"], coarse)
    np.testing.assert_allclose(metrics["fine_residual"], subcell_pattern)
    assert metrics["coarse_correlation"] == 1.0
    assert metrics["coarse_normalized_rmse"] == 0.0
    assert metrics["detail_variance_fraction"] > 0
