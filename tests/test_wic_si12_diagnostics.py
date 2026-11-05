import importlib.util
from pathlib import Path

import numpy as np
import pytest
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


def test_masked_moments_do_not_treat_missing_pixels_as_zero():
    samples = np.array([
        [[[1.0, 10.0]], [[2.0, 20.0]]],
        [[[3.0, 30.0]], [[4.0, 40.0]]],
        [[[5.0, 50.0]], [[6.0, 60.0]]],
    ])
    valid = np.array([
        [[[True, False]]],
        [[[True, True]]],
        [[[False, True]]],
    ])

    moments = VISUALIZE.masked_moments(samples, valid)

    np.testing.assert_allclose(moments["mean"][0, 0], [2.0, 40.0])
    np.testing.assert_allclose(moments["mean"][1, 0], [3.0, 50.0])
    np.testing.assert_allclose(
        moments["global_mean"], [21.0, 26.5]
    )
    np.testing.assert_array_equal(moments["count"], 2)


def test_coverage_adjusted_factor_preserves_available_sample_variances():
    samples = np.array([
        [[[1.0, 10.0]], [[2.0, 20.0]]],
        [[[3.0, 30.0]], [[4.0, 40.0]]],
        [[[5.0, 50.0]], [[6.0, 60.0]]],
        [[[7.0, 70.0]], [[8.0, 80.0]]],
    ])
    valid = np.ones((4, 1, 1, 2), dtype=bool)
    valid[0, 0, 0, 1] = False

    factor, support, count = VISUALIZE.coverage_adjusted_anomaly_factor(
        samples, valid, minimum_coverage=0.5
    )
    diagonal = np.diag(factor @ factor.T)
    flat = samples.reshape(4, -1)
    flat_valid = np.broadcast_to(valid, samples.shape).reshape(4, -1)
    expected = np.array([
        np.var(flat[:, index][flat_valid[:, index]], ddof=1)
        for index in np.flatnonzero(support)
    ])

    np.testing.assert_allclose(diagonal, expected)
    np.testing.assert_array_equal(count, [4, 3, 4, 3])


def test_auroral_occlusion_removes_fixed_patch_around_joint_peak():
    values = np.zeros((2, 2, 5, 5), dtype=np.float32)
    values[0, :, 4, 4] = [100.0, 10.0]
    values[1, :, 0, 0] = [80.0, 8.0]
    valid = np.ones((2, 1, 5, 5), dtype=bool)

    input_valid, bounds, removed_signal = VISUALIZE.auroral_occlusion_masks(
        values, valid, patch_size=3
    )

    np.testing.assert_array_equal(bounds, [[2, 5, 2, 5], [0, 3, 0, 3]])
    np.testing.assert_array_equal(np.sum(~input_valid, axis=(1, 2, 3)), [9, 9])
    assert not input_valid[0, 0, 4, 4]
    assert not input_valid[1, 0, 0, 0]
    assert np.all(np.isfinite(removed_signal))


def test_random_occlusion_is_seeded_brightness_independent_and_preserves_gaps():
    values = np.zeros((16, 2, 8, 8), dtype=np.float32)
    valid = np.ones((16, 1, 8, 8), dtype=bool)
    valid[:, :, :2, :2] = False
    original_valid = valid.copy()
    global_state = np.random.get_state()
    first, bounds, _ = VISUALIZE.inpainting_occlusion_masks(values, valid, 3, "random", 7)
    different_values = values.copy()
    different_values[:, :, 7, 7] = 1e6
    repeated, repeated_bounds, _ = VISUALIZE.inpainting_occlusion_masks(
        different_values, valid, 3, "random", 7
    )
    _, other_bounds, _ = VISUALIZE.inpainting_occlusion_masks(values, valid, 3, "random", 8)
    np.testing.assert_array_equal(first, repeated)
    np.testing.assert_array_equal(bounds, repeated_bounds)
    assert not np.array_equal(bounds, other_bounds)
    np.testing.assert_array_equal(np.random.get_state()[1], global_state[1])
    np.testing.assert_array_equal(valid, original_valid)
    assert not np.any(first & ~valid)
    assert np.all(np.sum(valid & ~first, axis=(1, 2, 3)) > 0)
    assert np.all(np.sum(first, axis=(1, 2, 3)) > 0)
    assert np.all(bounds[:, 1] - bounds[:, 0] == 3)
    assert np.all(bounds[:, 3] - bounds[:, 2] == 3)
    with pytest.raises(ValueError, match="removes all observed input"):
        VISUALIZE.inpainting_occlusion_masks(values, valid, 8, "random", 7)


def test_random_inpainting_masks_all_inputs_and_is_batch_size_independent():
    from types import SimpleNamespace

    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.inputs = []

        def encode(self, encoder_input, mask):
            self.inputs.append((encoder_input.clone(), mask.clone()))
            mu = torch.zeros(encoder_input.shape[0], 1)
            return mu, mu

        def decode(self, mu):
            return torch.zeros(mu.shape[0], 2, 8, 8)

    class Dataset(SimpleNamespace):
        def __len__(self):
            return self.values.shape[0]

    dataset = Dataset(
        values=np.ones((4, 2, 8, 8), dtype=np.float32),
        target=np.ones((4, 2, 8, 8), dtype=np.float32),
        weight=np.ones((4, 1, 8, 8), dtype=np.float32),
        common_valid=np.ones((4, 1, 8, 8), dtype=bool)
    )
    transform = SimpleNamespace(inverse=lambda value: value)
    model = RecordingModel()
    first = VISUALIZE.infer_heldout_inpainting(model, transform, dataset, 3, "cpu", 3, "random", 7)
    second = VISUALIZE.infer_heldout_inpainting(RecordingModel(), transform, dataset, 1, "cpu", 3, "random", 7)
    np.testing.assert_array_equal(first["removed"], second["removed"])
    np.testing.assert_array_equal(first["bounds"], second["bounds"])
    for encoder_input, mask in model.inputs:
        assert torch.equal(encoder_input, mask.expand_as(encoder_input).float())
    assert np.all(dataset.target == 1) and np.all(dataset.common_valid)
    ensembles = {"observed": dataset.values, "posterior_mean": dataset.values}
    metrics = VISUALIZE.inpainting_skill(ensembles, first, SimpleNamespace(transform=lambda value: value))
    np.testing.assert_array_equal(metrics["per_frame_removed_cell_count"], np.full((4, 2), 9))
    np.testing.assert_array_equal(metrics["per_frame_removed_squared_error_counts"], np.full((4, 2), 9))
    np.testing.assert_array_equal(metrics["inpainted_removed_region_physical_rmse_counts"], [1, 1])


def test_random_inpainting_cli_retains_auroral_default():
    default = VISUALIZE.parse_arguments(["saved_run"])
    assert default.inpainting_policy == "auroral"
    random = VISUALIZE.parse_arguments(["saved_run", "--inpainting-policy", "random", "--inpainting-seed", "7"])
    assert random.inpainting_policy == "random" and random.inpainting_seed == 7


def test_weighted_channel_rmse_ignores_zero_weight_and_nonfinite_errors():
    error = np.array([
        [[[1.0, np.nan]], [[2.0, 4.0]]],
        [[[3.0, 100.0]], [[6.0, 8.0]]],
    ])
    weights = np.array([
        [[[1.0, 0.0]], [[1.0, 0.0]]],
        [[[1.0, 0.0]], [[1.0, 0.0]]],
    ])

    rmse = VISUALIZE._weighted_channel_rmse(error, weights)

    np.testing.assert_allclose(rmse, [np.sqrt(5.0), np.sqrt(20.0)])


def test_gaussian_reconstruction_diagnostics_use_reference_attenuation():
    from types import SimpleNamespace
    from icanalyzer.wic_si12.data import ChannelTransform
    values = np.zeros((1, 2, 1, 2))
    dataset = SimpleNamespace(
        weight=np.ones((1, 1, 1, 2)),
        transformed_variance=np.array([[[[0.0, 3.0]], [[3.0, 0.0]]]])
    )
    ensembles = {
        "observed": values,
        "posterior_mean": values,
        "posterior_sample": values,
        "valid": np.ones_like(values, dtype=bool)
    }
    configuration = {
        "kind": "quality_weighted_gaussian_v1",
        "measurement_uncertainty_in_loss": True,
        "model_error_std": [1.0, 1.0]
    }
    result = VISUALIZE.posterior_reconstruction_skill(
        ensembles, dataset, ChannelTransform([1, 1], [0, 0], [1, 1]), configuration
    )
    np.testing.assert_allclose(result["weight_sums"]["training_effective"], [1.25, 1.25])


def test_count_reconstruction_diagnostics_do_not_claim_log_rmse_is_the_count_objective():
    from types import SimpleNamespace
    from icanalyzer.wic_si12.data import ChannelTransform
    values = np.zeros((1, 2, 1, 2))
    dataset = SimpleNamespace(weight=np.ones((1, 1, 1, 2)), standard_deviation=np.array([[[[0.0, 3.0]], [[3.0, 0.0]]]]))
    ensembles = {"observed": values, "posterior_mean": values, "posterior_sample": values, "valid": np.ones_like(values, dtype=bool)}
    config = {"kind": "quality_weighted_censored_counts_v1", "measurement_uncertainty_in_loss": True,
              "channel_scale_counts": [1, 1], "model_error_fraction": [1, 1]}
    result = VISUALIZE.posterior_reconstruction_skill(ensembles, dataset, ChannelTransform([1, 1], [0, 0], [1, 1]), config)
    np.testing.assert_allclose(result["weight_sums"]["training_effective"], [1.1, 1.1])
    assert "not the count-space training objective" in result["training_effective_note"]


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
