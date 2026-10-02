import importlib.util
from pathlib import Path

import numpy as np
import torch

from icanalyzer.wic_si12.diagnostics import (
    best_validation_epoch,
    covariance_mode_decomposition,
    covariance_spectrum,
    joint_correlation_maps,
    leading_covariance_modes,
    reconstruct_ensemble,
)
from icanalyzer.wic_si12.ensemble import ensemble_statistics


SCRIPT = (
    Path(__file__).parents[1] / "scripts" / "IMAGE" / "vae_pipeline"
    / "visualize_wic_si12_run.py"
)
SPEC = importlib.util.spec_from_file_location("visualize_wic_si12_run", SCRIPT)
VISUALIZE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VISUALIZE)


def synthetic_ensemble(n_samples=12, shape=(2, 3, 3)):
    generator = np.random.default_rng(8)
    return generator.normal(size=(n_samples, *shape))


def test_reconstructs_exact_samples_from_low_rank_factor():
    samples = synthetic_ensemble()
    statistics = ensemble_statistics(samples)
    recovered = reconstruct_ensemble(
        statistics["mean"], statistics["anomaly_factor"]
    )
    np.testing.assert_allclose(recovered, samples)

    selected = reconstruct_ensemble(
        statistics["mean"], statistics["anomaly_factor"], [1, 7]
    )
    np.testing.assert_allclose(selected, samples[[1, 7]])


def test_cross_field_correlation_matches_direct_calculation():
    samples = synthetic_ensemble()
    statistics = ensemble_statistics(samples)
    maps = joint_correlation_maps(
        statistics["anomaly_factor"], (2, 3, 3), 1, 2
    )

    flattened = samples.reshape(samples.shape[0], -1)
    expected = np.corrcoef(flattened, rowvar=False)
    first_anchor = 1 * 3 + 2
    second_anchor = 9 + first_anchor
    np.testing.assert_allclose(
        maps[0].reshape(2, -1), expected[first_anchor].reshape(2, -1)
    )
    np.testing.assert_allclose(
        maps[1].reshape(2, -1), expected[second_anchor].reshape(2, -1)
    )


def test_low_rank_spectrum_and_modes_match_dense_covariance():
    samples = synthetic_ensemble()
    statistics = ensemble_statistics(samples)
    factor = statistics["anomaly_factor"]
    expected = np.linalg.eigvalsh(factor @ factor.T)[::-1]
    expected = expected[expected > expected[0] * 1e-12]

    spectrum = covariance_spectrum(factor)
    np.testing.assert_allclose(spectrum, expected)
    eigenvalues, modes = leading_covariance_modes(factor, (2, 3, 3), n_modes=3)
    np.testing.assert_allclose(eigenvalues, expected[:3])
    np.testing.assert_allclose(
        modes.reshape(3, -1) @ modes.reshape(3, -1).T,
        np.eye(3),
        atol=1e-10,
    )


def test_randomized_modes_avoid_sample_gram_and_recover_low_rank_field():
    generator = np.random.default_rng(2)
    left, _ = np.linalg.qr(generator.normal(size=(18, 3)))
    right, _ = np.linalg.qr(generator.normal(size=(600, 3)))
    singular_values = np.array([4.0, 2.0, 1.0])
    factor = (left * singular_values) @ right.T

    decomposition = covariance_mode_decomposition(
        factor,
        (2, 3, 3),
        n_modes=3,
        exact_max_dimension=4,
        power_iterations=1,
    )

    assert decomposition["method"] == "randomized_svd"
    np.testing.assert_allclose(
        decomposition["eigenvalues"], singular_values**2, rtol=1e-10
    )
    np.testing.assert_allclose(
        decomposition["modes"].reshape(3, -1)
        @ decomposition["modes"].reshape(3, -1).T,
        np.eye(3),
        atol=1e-10,
    )


def test_best_validation_epoch():
    history = [
        {"epoch": 1, "validation": {"total": 2.0}},
        {"epoch": 2, "validation": {"total": 1.0}},
        {"epoch": 3, "validation": {"total": 1.5}},
    ]
    epoch, metrics = best_validation_epoch(history)
    assert epoch == 2
    assert metrics["total"] == 1.0


def test_plot_limits_ignore_nonfinite_pixels():
    values = np.array([-4.0, 1.0, np.nan, np.inf])
    lower, upper = VISUALIZE.robust_limits(values, signed=True)
    assert np.isfinite([lower, upper]).all()
    assert lower < -3.0
    assert upper > 3.0


def test_headless_visualization_smoke(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    fuv = synthetic_ensemble()
    conductance = np.abs(synthetic_ensemble())
    fuv_statistics = ensemble_statistics(fuv)
    conductance_statistics = ensemble_statistics(conductance)
    history = [
        {
            "epoch": 1,
            "train": {"total": 1.0, "reconstruction": 0.9, "kl": 10.0},
            "validation": {"total": 1.1, "reconstruction": 1.0, "kl": 10.0},
        },
        {
            "epoch": 2,
            "train": {"total": 0.8, "reconstruction": 0.7, "kl": 10.0},
            "validation": {"total": 0.9, "reconstruction": 0.8, "kl": 10.0},
        },
    ]
    torch.save({"history": history}, run / "wic_si12_vae.pt")
    (run / "summary.json").write_text('{"sample_count": 12}\n')
    np.savez_compressed(
        run / "ensemble_products.npz",
        fuv_mean=fuv_statistics["mean"],
        fuv_anomaly_factor=fuv_statistics["anomaly_factor"],
        latent_samples=np.random.default_rng(4).normal(size=(12, 4)),
        conductance_mean=conductance_statistics["mean"],
        conductance_anomaly_factor=conductance_statistics["anomaly_factor"],
        reconstruction_mean=conductance_statistics["mean"],
        reconstruction_standard_deviation=np.sqrt(
            conductance_statistics["variance"]
        ),
    )

    arguments = VISUALIZE.parse_arguments([
        str(run), "--examples", "2", "--modes", "2"
    ])
    diagnostics = VISUALIZE.run(arguments)
    assert diagnostics["best_validation_epoch"] == 2
    assert diagnostics["dense_covariance_materialized"] is False
    assert diagnostics["latent_samples_available"] is True
    for filename in diagnostics["created_figures"]:
        assert (run / "figures" / filename).is_file()
