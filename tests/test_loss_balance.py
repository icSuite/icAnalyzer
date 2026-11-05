"""Loss/gradient fidelity and non-mutating audit tests."""
import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.count_likelihood import fit_count_uncertainty_configuration
from icanalyzer.wic_si12.data import ChannelTransform, WICSI12ArrayDataset
from icanalyzer.wic_si12.model import SplineWICSI12VAE, vae_loss
from icanalyzer.wic_si12.workflow import LEGACY_LOSS_CONFIGURATION, gaussian_uncertainty_configuration
from scripts.IMAGE.vae_pipeline.diagnose_loss_balance import (
    audit_model, fit_bright_threshold, gradient_difference, gradient_statistics, loss_cells, observation_groups, parse_arguments, plot_audit
)


def small_dataset():
    generator = np.random.default_rng(9)
    values = generator.uniform(0, 100, (2, 2, 8, 8)).astype(np.float32)
    values[:, 1] /= 20
    values[:, :, :2] = 0
    transform = ChannelTransform([1, 1], [1, .4], [.8, .4], kind=ChannelTransform.LOG10_KIND)
    return WICSI12ArrayDataset({"values": values, "standard_deviation": np.ones_like(values),
                               "valid": np.ones_like(values, dtype=bool), "quality": np.ones_like(values)}, transform)


@pytest.mark.parametrize("kind", ["legacy", "gaussian", "count"])
def test_decomposed_loss_and_gradients_match_production(kind):
    dataset = small_dataset()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    decoded = (batch["target"] + .3).detach().requires_grad_()
    mu = torch.zeros(2, 3)
    arguments = {}
    if kind == "legacy":
        config = LEGACY_LOSS_CONFIGURATION
    elif kind == "gaussian":
        config = gaussian_uncertainty_configuration()
        arguments = {"transformed_variance": batch["variance"], "model_error_std": [.5, .5]}
    else:
        config = fit_count_uncertainty_configuration(dataset)
        arguments = {"count_configuration": config, "count_observed": batch["physical_target"],
                     "count_standard_deviation": batch["physical_standard_deviation"]}
    reference = vae_loss(decoded, batch["target"], batch["mask"], batch["weight"], mu, mu, **arguments)["reconstruction"]
    actual = loss_cells(decoded, batch, config).sum()
    torch.testing.assert_close(actual, reference)
    a = torch.autograd.grad(actual, decoded, retain_graph=True)[0]
    b = torch.autograd.grad(reference, decoded)[0]
    torch.testing.assert_close(a, b)


def test_zero_quality_nonfinite_cells_are_safe_in_loss_and_gradients():
    dataset = small_dataset()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    batch["weight"][:, :, 0, 0] = 0
    batch["physical_target"][:, :, 0, 0] = float("nan")
    batch["physical_standard_deviation"][:, :, 0, 0] = float("nan")
    decoded = batch["target"].clone()
    decoded[:, :, 0, 0] = float("nan")
    decoded.requires_grad_()
    result = loss_cells(decoded, batch, fit_count_uncertainty_configuration(dataset)).sum()
    gradient = torch.autograd.grad(result, decoded)[0]
    assert torch.isfinite(result) and torch.isfinite(gradient).all()
    assert torch.all(gradient[:, :, 0, 0] == 0)


def test_observation_groups_partition_supported_pixels_once():
    dataset = small_dataset()
    dataset.weight[0, :, 1, 1] = 0
    groups = observation_groups(dataset.values, dataset.weight, fit_bright_threshold(dataset))
    np.testing.assert_array_equal(groups.sum(axis=2), np.broadcast_to(dataset.weight > 0, dataset.values.shape))


def test_fixed_reference_is_the_expected_temperature_multiplier():
    dataset = small_dataset()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    decoded = (batch["target"] + .2).detach().requires_grad_()
    config = fit_count_uncertainty_configuration(dataset, (.1, .1))
    actual = loss_cells(decoded, batch, config, fixed_reference=True)
    native = loss_cells(decoded, batch, config)
    torch.testing.assert_close(actual, native * 25)
    a = torch.autograd.grad(actual.sum(), decoded, retain_graph=True)[0]
    b = torch.autograd.grad(native.sum(), decoded)[0]
    torch.testing.assert_close(a, b * 25)


def test_shared_encoder_norms_include_cancellation():
    result = gradient_statistics([torch.tensor([3., 0.])], [torch.tensor([-3., 4.])], [torch.tensor([0., 2.])])
    np.testing.assert_allclose(result[:4], [3, 5, 4, 2])
    np.testing.assert_allclose(result[4:], [1, -.6])


def test_gradient_difference_detects_direction_changes_with_equal_norm():
    result = gradient_difference([torch.tensor([3., 4.])], [torch.tensor([-3., 4.])])
    np.testing.assert_allclose(result, [1.2, .28])


def test_audit_does_not_mutate_model_and_is_seed_reproducible():
    dataset = small_dataset()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=6)
        model.eval()
        state = {key: value.clone() for key, value in model.state_dict().items()}
        config = fit_count_uncertainty_configuration(dataset)
        kwargs = dict(model=model, dataset=dataset, configuration=LEGACY_LOSS_CONFIGURATION,
                      count_configuration=config, indices=[0, 1], batch_size=2, seed=4, beta=1.6,
                      bright_threshold=fit_bright_threshold(dataset))
        first, hash1 = audit_model(**kwargs)
        second, hash2 = audit_model(**kwargs)
        assert first == second and hash1 == hash2
        for key, value in model.state_dict().items():
            assert torch.equal(value, state[key])
        assert all(parameter.grad is None for parameter in model.parameters())
        for result in first.values():
            assert np.isclose(np.sum(result["pixel_gradient_share"]), 1)
        assert first["native"] == first["native_no_measurements"] or first["native"]["pixel_gradient_l1"] == first["native_no_measurements"]["pixel_gradient_l1"]
    finally:
        torch.set_num_threads(previous_threads)


def test_cli_rejects_invalid_batch_and_frame_limits():
    for options in (["--batch-size", "0"], ["--max-frames", "0"]):
        with pytest.raises(SystemExit):
            parse_arguments(["model", "--output", "output", *options])


def test_plots_use_consistent_group_colors_and_write_all_figures(tmp_path, monkeypatch):
    import matplotlib.pyplot as plt
    figures = []
    original_subplots = plt.subplots

    def capture_figure(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plt, "subplots", capture_figure)
    scenarios = {"native": {"pixel_gradient_share": [[.1, .1, .2, .1], [.1, .1, .2, .1]]},
                 "native_no_measurements": {"mean_relative_encoder_gradient_change_on_measurement_removal": .01}}
    for fraction in (.03, .1, .5):
        scenarios[f"count_f{fraction:g}_measured"] = {"reconstruction_to_betaKL_norm_ratio": 2.0}
        scenarios[f"count_f{fraction:g}_no_measurements"] = {"reconstruction_to_betaKL_norm_ratio": 3.0,
                                                             "mean_relative_encoder_gradient_change_on_measurement_removal": .2}
        if fraction != .5:
            scenarios[f"count_f{fraction:g}_fixed_reference"] = {"reconstruction_to_betaKL_norm_ratio": 50.0}
    plot_audit({"first": {"scenarios": scenarios}, "second": {"scenarios": scenarios}}, tmp_path)
    for axis in figures[0].axes:
        colors = [patch.get_facecolor() for patch in axis.patches]
        assert colors[:4] == colors[4:]
    for name in ("native_gradient_budget.png", "allowance_and_KL_sensitivity.png", "measurement_gradient_change.png"):
        assert (tmp_path / name).is_file()
