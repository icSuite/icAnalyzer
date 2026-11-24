import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/CVAE/sample_empirical_latent_at_kp.py"
)
SPEC = importlib.util.spec_from_file_location("empirical_latent", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_bank_selection_is_bounded_and_spans_each_kp_pool():
    kp = np.array([0, 0, 0, 1.667, 2, 2, 2.333, 4])

    selected, available = MODULE.select_bank_indices(
        kp, targets=[0, 2], half_width=0.34, maximum_per_target=2
    )

    np.testing.assert_array_equal(selected, [0, 2, 3, 6])
    assert available == {"0.0": 3, "2.0": 4}


def test_empirical_mean_sampling_preserves_complete_joint_rows():
    mu = np.arange(20, dtype=np.float32).reshape(5, 4)
    logvar = np.zeros_like(mu)
    pool = np.array([1, 3, 4])

    latent, source_rows = MODULE.choose_latents(
        mu, logvar, pool, draws=3, rng=np.random.default_rng(4),
        posterior_sample=False,
    )

    np.testing.assert_array_equal(latent, mu[source_rows])
    assert set(source_rows) == set(pool)
