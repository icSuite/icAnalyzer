import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[1]
    / "scripts/CVAE/analyze_empirical_latent_modes.py"
)
SPEC = importlib.util.spec_from_file_location("empirical_modes", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_orbit_heldout_split_has_no_group_leakage():
    orbits = np.repeat(np.arange(12), 5)

    train, test = MODULE.orbit_heldout_split(orbits, test_fraction=0.25, seed=3)

    assert set(orbits[train]).isdisjoint(orbits[test])
    assert sorted(np.r_[train, test]) == list(range(len(orbits)))


def test_retained_pca_components_reaches_requested_variance():
    explained = np.array([0.50, 0.30, 0.10, 0.05])

    assert MODULE.retained_pca_components(explained, 0.79) == 2
    assert MODULE.retained_pca_components(explained, 0.85) == 3


def test_bic_recovers_two_well_separated_latent_modes():
    rng = np.random.default_rng(8)
    first = rng.normal(-4, 0.35, size=(160, 3))
    second = rng.normal(4, 0.35, size=(160, 3))
    train = np.r_[first[:120], second[:120]]
    test = np.r_[first[120:], second[120:]]

    records, _models = MODULE.fit_mixture_candidates(
        train, test, maximum_components=4, covariance_type="diag",
        regularization=1e-4, initializations=3, seed=2,
    )
    selected = MODULE.select_candidate(records, minimum_component_fraction=0.03)

    assert records[selected]["components"] == 2


def test_inverse_pca_centers_and_representatives_preserve_rows():
    rng = np.random.default_rng(5)
    values = rng.normal(size=(40, 6))
    pca = MODULE.PCA(n_components=4, random_state=1).fit(values)
    centers = pca.transform(values[:2])[:, :3]
    restored = MODULE.inverse_pca_centers(pca, 3, centers)
    expected = pca.mean_[None] + centers @ pca.components_[:3]
    np.testing.assert_allclose(restored, expected, rtol=1e-6, atol=1e-6)

    responsibility = np.array([[.9, .1], [.8, .2], [.1, .9], [.2, .8]])
    labels = np.array([0, 0, 1, 1])
    np.testing.assert_array_equal(
        MODULE.representative_rows(responsibility, labels), [0, 2]
    )
