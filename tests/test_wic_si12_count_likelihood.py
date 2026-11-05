"""Reference calculations and end-to-end wiring for the opt-in count loss."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.special import log_ndtr
from scipy.optimize import minimize_scalar
import torch
from torch.utils.data import DataLoader

from icanalyzer.wic_si12.count_likelihood import (
    COUNT_UNCERTAINTY_LOSS_KIND,
    count_space_reconstruction_loss,
    fit_count_uncertainty_configuration,
    validate_count_configuration
)
from icanalyzer.wic_si12.data import ChannelTransform, WICSI12ArrayDataset
from icanalyzer.wic_si12.model import SplineWICSI12VAE, vae_loss
from icanalyzer.wic_si12.workflow import evaluate_vae, load_run_artifact, save_run_artifact, train_vae


SCRIPT = Path(__file__).parents[1] / "scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py"
SPEC = importlib.util.spec_from_file_location("count_pipeline_test", SCRIPT)
PIPELINE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PIPELINE)


def configuration(scales=(10.0, 2.0), fractions=(0.5, 0.5)):
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    return {
        "kind": COUNT_UNCERTAINTY_LOSS_KIND,
        "measurement_uncertainty_in_loss": True,
        "channel_scale_counts": list(scales),
        "model_error_fraction": list(fractions),
        "transform": transform.to_dict()
    }


def loss(mean, observed, sigma, config=None, weight=None, mask=None):
    mean = torch.as_tensor(mean, dtype=torch.float64).reshape(1, 2, 1, -1)
    observed = torch.as_tensor(observed, dtype=torch.float64).reshape_as(mean)
    sigma = torch.as_tensor(sigma, dtype=torch.float64).reshape_as(mean)
    weight = torch.ones_like(mean[:, :1]) if weight is None else weight
    mask = torch.ones_like(weight, dtype=torch.bool) if mask is None else mask
    decoded = torch.log10(mean + 1.0).requires_grad_()
    value = count_space_reconstruction_loss(decoded, observed, sigma, mask, weight, config or configuration())
    return value, decoded


def test_positive_count_likelihood_matches_reference_gaussian_arithmetic():
    mean = np.array([8.0, 4.0])
    observed = np.array([10.0, 2.0])
    sigma = np.array([3.0, 1.0])
    scale = np.array([10.0, 2.0])
    tau2 = np.array([0.25, 0.25])
    variance = (sigma / scale)**2 + tau2
    reference = np.sum(tau2 * (((mean - observed) / scale)**2 / variance + np.log(variance / tau2))) * 2.0 / (1 + 1e-8)
    actual, _ = loss(mean, observed, sigma)
    np.testing.assert_allclose(actual.item(), reference, rtol=1e-12)


def test_zero_counts_use_censored_probability_not_zero_residual():
    actual, decoded = loss([0.0, 0.0], [0.0, 0.0], [0.0, 0.0])
    np.testing.assert_allclose(actual.item(), 2 * np.log(2) / (1 + 1e-8))
    actual.backward()
    assert torch.all(decoded.grad > 0)
    assert torch.isfinite(decoded.grad).all()


def test_count_units_rescale_mean_sigma_and_channel_scale_together():
    original, _ = loss([8, 4], [10, 2], [3, 1])
    rescaled, _ = loss([800, 4], [1000, 2], [300, 1], configuration(scales=(1000, 2)))
    np.testing.assert_allclose(original.item(), rescaled.item(), rtol=1e-12)


def test_count_errors_are_not_log_delta_method_errors():
    actual, _ = loss([1, 1], [0, 0], [100, 100], configuration(scales=(1, 1)))
    expected = -2 * 0.25 * log_ndtr(-1 / np.sqrt(10000.25)) * 4 / (1 + 1e-8)
    np.testing.assert_allclose(actual.item(), expected, rtol=1e-12)


def test_censored_tail_is_stable_at_one_hundred_sigma():
    actual, decoded = loss([50, 50], [0, 0], [0, 0], configuration(scales=(1, 1)))
    np.testing.assert_allclose(actual.item(), -2 * log_ndtr(-100) / (1 + 1e-8), rtol=1e-12)
    actual.backward()
    assert torch.isfinite(decoded.grad).all() and torch.all(decoded.grad > 0)


def test_unclipped_preclipping_mean_has_gradient_below_physical_zero():
    actual, decoded = loss([-0.5, -0.5], [2, 2], [1, 1])
    actual.backward()
    assert torch.isfinite(decoded.grad).all() and torch.all(decoded.grad < 0)
    transform = ChannelTransform.from_dict(configuration()["transform"])
    assert torch.all(transform.inverse(decoded) == 0)


def test_likelihood_gradient_matches_finite_differences():
    decoded = torch.tensor([[[[0.4, -0.1]], [[0.7, 0.2]]]], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([[[[2.0, 0.0]], [[0.0, 1.0]]]], dtype=torch.float64)
    sigma = torch.ones_like(observed)
    mask = torch.ones(1, 1, 1, 2, dtype=torch.bool)
    weight = torch.ones(1, 1, 1, 2, dtype=torch.float64)
    assert torch.autograd.gradcheck(
        lambda values: count_space_reconstruction_loss(values, observed, sigma, mask, weight, configuration()), (decoded,)
    )


def test_synthetic_censoring_recovers_mean_instead_of_clipped_sample_bias():
    # This is a Gaussian CS-level simulation, NOT a model of native WIC
    # clipping followed by averaging. It checks the likelihood's stated model.
    generator = np.random.default_rng(7)
    clipped = np.maximum(generator.normal(3.0, 4.0, 20000), 0)
    values = torch.tensor(np.broadcast_to(clipped, (1, 2, 1, len(clipped))).copy(), dtype=torch.float64)
    sigma = torch.full_like(values, 4.0)
    mask = torch.ones_like(values[:, :1], dtype=torch.bool)
    weight = torch.ones_like(values[:, :1])
    config = configuration(scales=(4, 4), fractions=(0.001, 0.001))

    def objective(mean):
        decoded = torch.full_like(values, np.log10(mean + 1))
        result = count_space_reconstruction_loss(decoded, values, sigma, mask, weight, config)
        return result.item() / (0.001**2 * values.numel())

    fitted = minimize_scalar(objective, bounds=(0.0, 8.0), method="bounded")
    assert fitted.success
    assert abs(fitted.x - 3.0) < 0.1
    assert clipped.mean() - 3.0 > 0.4


def test_float32_tail_has_finite_training_gradient():
    decoded = torch.full((1, 2, 1, 1), float(np.log10(51)), requires_grad=True)
    actual = count_space_reconstruction_loss(decoded, torch.zeros_like(decoded), torch.zeros_like(decoded),
                                              torch.ones(1, 1, 1, 1).bool(), torch.ones(1, 1, 1, 1), configuration(scales=(1, 1)))
    actual.backward()
    assert torch.isfinite(actual) and torch.isfinite(decoded.grad).all()
    assert torch.all(decoded.grad > 0)


def test_missing_and_zero_quality_cells_do_not_contribute_or_produce_nan_gradients():
    decoded = torch.tensor([[[[0.0, float("nan"), 1e6]], [[0.0, float("nan"), 1e6]]]], requires_grad=True)
    observed = torch.tensor([[[[1.0, float("nan"), float("nan")]], [[1.0, float("nan"), float("nan")]]]])
    mask = torch.tensor([[[[True, False, True]]]])
    weight = torch.tensor([[[[1.0, float("nan"), 0.0]]]])
    actual = count_space_reconstruction_loss(decoded, observed, observed, mask, weight, configuration())
    assert torch.isfinite(actual)
    actual.backward()
    assert torch.isfinite(decoded.grad).all() and torch.all(decoded.grad[..., 1:] == 0)


@pytest.mark.parametrize("bad_field", ["observed", "sigma", "decoded"])
def test_invalid_weighted_cells_raise(bad_field):
    arrays = {name: torch.ones(1, 2, 1, 1) for name in ("observed", "sigma", "decoded")}
    arrays[bad_field][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="invalid data"):
        count_space_reconstruction_loss(arrays["decoded"], arrays["observed"], arrays["sigma"], torch.ones(1, 1, 1, 1).bool(), torch.ones(1, 1, 1, 1), configuration())


@pytest.mark.parametrize("name,value", [("channel_scale_counts", [0, 1]), ("model_error_fraction", [-1, 1]), ("channel_scale_counts", [1])])
def test_count_configuration_rejects_invalid_scales(name, value):
    config = configuration()
    config[name] = value
    with pytest.raises(ValueError, match=name):
        validate_count_configuration(config)


def test_count_loss_retains_literal_summed_kl_and_beta():
    decoded = torch.zeros(2, 2, 1, 1)
    mu = torch.ones(2, 3)
    logvar = torch.zeros_like(mu)
    losses = vae_loss(decoded, decoded, torch.ones(2, 1, 1, 1).bool(), torch.ones(2, 1, 1, 1), mu, logvar,
                      beta=1.6, count_observed=torch.ones_like(decoded), count_standard_deviation=torch.ones_like(decoded), count_configuration=configuration())
    assert losses["kl"].item() == 3.0
    np.testing.assert_allclose(losses["total"].item(), losses["reconstruction"].item() + 1.6 * 3)
    with pytest.raises(ValueError, match="cannot also use"):
        vae_loss(decoded, decoded, torch.ones(2, 1, 1, 1).bool(), torch.ones(2, 1, 1, 1), mu, logvar,
                 transformed_variance=torch.ones_like(decoded), count_configuration=configuration())


def training_dataset():
    values = np.full((3, 2, 46, 46), 2.0, dtype=np.float32)
    values[:, 0] *= 10
    values[0, :, 10:22, 10:22] = 0
    values[2] = 1e6  # Entirely rejected frame must not influence the count scale.
    valid = np.ones_like(values, dtype=bool)
    valid[2] = False
    transform = ChannelTransform.from_dict(configuration()["transform"])
    product = {"values": values, "standard_deviation": np.ones_like(values), "valid": valid, "quality": np.ones_like(values)}
    return WICSI12ArrayDataset(product, transform)


def test_training_only_scales_use_retained_frames_and_physical_batch_errors():
    dataset = training_dataset()
    config = fit_count_uncertainty_configuration(dataset)
    np.testing.assert_allclose(config["channel_scale_counts"], np.sqrt(np.mean(dataset.values.astype(float)**2, axis=(0, 2, 3))))
    np.testing.assert_array_equal(dataset[0]["physical_standard_deviation"], dataset.standard_deviation[0])
    assert dataset[0]["encoder_input"].shape == (3, 46, 46)
    assert len(dataset) == 2


def test_count_training_with_holes_and_artifact_reload(tmp_path):
    dataset = training_dataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    config = fit_count_uncertainty_configuration(dataset)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = SplineWICSI12VAE()
        model, history = train_vae(model, loader, validation_loader=loader, epochs=1, beta=1.6,
                                   loss_configuration=config, hole_probability=1.0, hole_size=12)
        assert all(np.isfinite(list(history[0]["train"].values())))
        path = tmp_path / "count_model.pt"
        save_run_artifact(path, model, dataset.transform, history, {"loss_configuration": config}, {}, {})
        restored, transform, artifact = load_run_artifact(path)
        assert restored.configuration() == model.configuration()
        assert restored.loss_configuration == config
        assert artifact["training_configuration"]["loss_configuration"] == config
        assert transform.to_dict() == dataset.transform.to_dict()
        torch.manual_seed(3)
        original_evaluation = evaluate_vae(model, loader, beta=1.6, loss_configuration=config)
        torch.manual_seed(3)
        assert evaluate_vae(restored, loader, beta=1.6) == original_evaluation
    finally:
        torch.set_num_threads(previous_threads)


def test_count_cli_is_opt_in_and_options_cannot_be_silently_ignored():
    default = PIPELINE.parse_arguments(["data", "--output", "run"])
    assert not default.count_uncertainty and not default.gaussian_uncertainty and not default.measurement_uncertainty
    assert default.latent_dim == 36 and default.spline_control_points == 30
    opted = PIPELINE.parse_arguments(["data", "--output", "run", "--count-uncertainty", "--count-model-error-fraction", "0.2", "0.3"])
    assert opted.count_model_error_fraction == [0.2, 0.3]
    for options in (["--model-error-std", "0.5", "0.5"], ["--count-model-error-fraction", "0.5", "0.5"], ["--count-uncertainty", "--gaussian-uncertainty"],
                    ["--maximum-relative-precision", "100"], ["--uncertainty-variance-floor", "0.001"], ["--precision-scale-samples", "1000"],
                    ["--count-uncertainty", "--count-model-error-fraction", "0", "1"]):
        with pytest.raises(SystemExit):
            PIPELINE.parse_arguments(["data", "--output", "run", *options])


def test_count_artifact_reuse_rejects_changed_likelihood_or_training_scales():
    dataset = training_dataset()
    config = fit_count_uncertainty_configuration(dataset)
    arguments = PIPELINE.parse_arguments(["data", "--output", "run", "--count-uncertainty"])
    product = {"values": dataset.values, "standard_deviation": dataset.standard_deviation,
               "valid": dataset.valid, "quality": dataset.quality,
               "metadata": [{"quality_weight_source": "auroral_quality", "upstream_quality_weight_interpretation": "1-dgweight"}]}
    artifact = {
        "transform": dataset.transform.to_dict(), "split_files": {"train": [], "validation": [], "test": []},
        "training_configuration": {"latent_dim": 36, "seed": 1, "minimum_sza_degrees": None, "comparison_data_version": "current", "loss_configuration": config},
        "data_provenance": {"manual_frame_intervals": str(Path(arguments.segments).resolve()), "legacy_comparison_directory": None,
                            "loss_quality_weight_source": "auroral_quality", "upstream_quality_weight_interpretation": "1-dgweight"}
    }
    PIPELINE.validate_reused_artifact(artifact, artifact["split_files"], arguments, product)
    arguments.count_model_error_fraction = [0.2, 0.5]
    with pytest.raises(ValueError, match="count_model_error_fraction"):
        PIPELINE.validate_reused_artifact(artifact, artifact["split_files"], arguments, product)
    arguments.count_model_error_fraction = [0.5, 0.5]
    product["values"] = product["values"] * 2
    with pytest.raises(ValueError, match="count_channel_scales"):
        PIPELINE.validate_reused_artifact(artifact, artifact["split_files"], arguments, product)
