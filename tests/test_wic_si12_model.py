import numpy as np
import torch
from torch.utils.data import DataLoader

import icanalyzer.wic_si12.workflow as workflow
from icanalyzer.wic_si12.data import (
    ChannelTransform,
    WICSI12ArrayDataset,
    fit_precision_weighting,
)
from icanalyzer.wic_si12.model import (
    PartialConv2d,
    SplineWICSI12VAE,
    WICSI12VAE,
    vae_loss,
)
from icanalyzer.wic_si12.workflow import (
    LEGACY_LOSS_CONFIGURATION,
    evaluate_vae,
    gaussian_uncertainty_configuration,
    load_run_artifact,
    save_run_artifact,
    train_vae,
)


FIXED_WEIGHTING = LEGACY_LOSS_CONFIGURATION


def test_training_holes_are_shared_reproducible_and_do_not_change_source():
    encoder_input = torch.ones(4, 3, 8, 8)
    mask = torch.ones(4, 1, 8, 8, dtype=torch.bool)
    rng_state = torch.random.get_rng_state().clone()
    damaged, input_mask = workflow.add_training_holes(
        encoder_input, mask, 1.0, 3, torch.Generator().manual_seed(4)
    )
    repeated, repeated_mask = workflow.add_training_holes(
        encoder_input, mask, 1.0, 3, torch.Generator().manual_seed(4)
    )
    assert torch.equal(damaged, repeated)
    assert torch.equal(input_mask, repeated_mask)
    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert torch.all((~input_mask).sum(dim=(1, 2, 3)) == 9)
    assert torch.equal(damaged, input_mask.expand_as(damaged).float())
    assert torch.all(encoder_input == 1)
    assert torch.all(mask)
    unchanged, unchanged_mask = workflow.add_training_holes(encoder_input, mask, 0.0, 3, None)
    assert unchanged is encoder_input and unchanged_mask is mask


def test_training_holes_keep_some_original_observations():
    encoder_input = torch.ones(1, 3, 8, 8)
    mask = torch.zeros(1, 1, 8, 8, dtype=torch.bool)
    mask[0, 0, 4, 4] = True
    _, input_mask = workflow.add_training_holes(
        encoder_input, mask, 1.0, 3, torch.Generator().manual_seed(4)
    )
    assert torch.equal(input_mask, mask)


def test_training_holes_change_encoder_only_and_leave_evaluation_clean(monkeypatch):
    class RecordingModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.level = torch.nn.Parameter(torch.tensor(0.0))

        def forward(self, encoder_input, mask):
            self.input = encoder_input.clone()
            self.mask = mask.clone()
            reconstructed = self.level.expand_as(encoder_input[:, :2])
            latent = torch.zeros(encoder_input.shape[0], 1)
            return reconstructed, latent, latent

    batch = {
        "encoder_input": torch.ones(2, 3, 8, 8),
        "target": torch.ones(2, 2, 8, 8),
        "mask": torch.ones(2, 1, 8, 8, dtype=torch.bool),
        "weight": torch.ones(2, 1, 8, 8)
    }
    original_loss = workflow.vae_loss

    def checked_loss(reconstructed, target, mask, weight, *args, **kwargs):
        assert torch.equal(target, batch["target"])
        assert torch.equal(mask, batch["mask"])
        assert torch.equal(weight, batch["weight"])
        return original_loss(reconstructed, target, mask, weight, *args, **kwargs)

    monkeypatch.setattr(workflow, "vae_loss", checked_loss)
    model = RecordingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    for training in (True, False):
        workflow._run_epoch(
            model, [batch], optimizer, "cpu", 0.0, training,
            loss_configuration=FIXED_WEIGHTING, hole_probability=1.0,
            hole_size=3, hole_generator=torch.Generator().manual_seed(4)
        )
        if training:
            assert torch.all((~model.mask).sum(dim=(1, 2, 3)) == 9)
            assert torch.equal(model.input, model.mask.expand_as(model.input).float())
        else:
            assert torch.equal(model.input, batch["encoder_input"])
            assert torch.equal(model.mask, batch["mask"])


def synthetic_product(n_samples=6):
    generator = np.random.default_rng(3)
    values = np.empty((n_samples, 2, 46, 46), dtype=np.float32)
    values[:, 0] = generator.uniform(0, 500, size=(n_samples, 46, 46))
    values[:, 1] = generator.normal(20, 5, size=(n_samples, 46, 46))
    return {
        "values": values,
        "standard_deviation": np.ones_like(values),
        "valid": np.ones_like(values, dtype=bool),
        "quality": np.ones_like(values),
    }


def test_model_forward_and_consistent_loss_reduction():
    model = WICSI12VAE(latent_dim=4, wic_floor_transformed=-0.25)
    encoder_input = torch.randn(2, 4, 46, 46)
    reconstructed, mu, logvar = model(encoder_input)
    assert reconstructed.shape == (2, 2, 46, 46)
    assert mu.shape == logvar.shape == (2, 4)
    assert torch.all(reconstructed[:, 0] >= -0.25)

    target = torch.zeros_like(reconstructed)
    mask = torch.ones(2, 1, 46, 46, dtype=torch.bool)
    weight = torch.ones(2, 1, 46, 46)
    mask[:, :, 0, 0] = False
    weight[:, :, 0, 0] = torch.nan
    losses = vae_loss(
        reconstructed, target, mask, weight, mu, logvar, beta=0.1
    )
    assert set(losses) == {"total", "reconstruction", "kl"}
    assert torch.isfinite(losses["total"])


def test_partial_convolution_ignores_values_outside_mask():
    layer = PartialConv2d(
        1, 1, kernel_size=3, padding=1, bias=False, return_mask=True
    )
    with torch.no_grad():
        layer.weight.fill_(1.0)
    mask = torch.ones(1, 1, 5, 5)
    mask[:, :, 2, 2] = 0.0
    first = torch.ones(1, 1, 5, 5)
    second = first.clone()
    second[:, :, 2, 2] = 1e6

    first_output, first_mask = layer(first, mask)
    second_output, second_mask = layer(second, mask)

    torch.testing.assert_close(first_output, second_output)
    torch.testing.assert_close(first_mask, second_mask)


def test_spline_model_uses_partial_mask_and_projects_to_image_grid():
    model = SplineWICSI12VAE(
        image_shape=(46, 46), latent_dim=4, spline_control_points=30
    )
    values = torch.randn(2, 3, 46, 46)
    valid = torch.ones(2, 1, 46, 46, dtype=torch.bool)
    valid[:, :, :4, :4] = False

    reconstructed, mu, logvar = model(values, valid)
    coefficients = model.decode_coefficients(torch.zeros(2, 4))

    assert reconstructed.shape == (2, 2, 46, 46)
    assert coefficients.shape == (2, 2, 30, 30)
    assert mu.shape == logvar.shape == (2, 4)
    assert model.configuration()["quality_weight_in_encoder"] is True
    assert model.configuration()["encoder_channels"] == [3, 16, 32, 64]
    assert model.configuration()["decoder_seed_shape"] == [9, 9]
    torch.testing.assert_close(
        model.row_basis.sum(dim=1), torch.ones(46), rtol=1e-5, atol=1e-6
    )
    torch.testing.assert_close(
        model.column_basis.sum(dim=1), torch.ones(46), rtol=1e-5, atol=1e-6
    )
    constant_coefficients = torch.full((1, 2, 30, 30), 2.5)
    constant_field = torch.einsum(
        "yi,bcij,xj->bcyx",
        model.row_basis,
        constant_coefficients,
        model.column_basis,
    )
    torch.testing.assert_close(
        constant_field, torch.full((1, 2, 46, 46), 2.5),
        rtol=1e-5, atol=1e-5
    )


def test_spline_decoder_coefficient_axes_match_legacy(monkeypatch):
    model = SplineWICSI12VAE(
        image_shape=(7, 9), latent_dim=2, spline_control_points=5
    )
    coefficients = torch.zeros(1, 2, 5, 5)
    coefficients[0, 0, 1, 3] = 1.0
    monkeypatch.setattr(
        model, "decode_coefficients", lambda latent: coefficients
    )

    reconstructed = model.decode(torch.zeros(1, 2))
    expected = torch.einsum(
        "y,x->yx", model.row_basis[:, 3], model.column_basis[:, 1]
    )

    torch.testing.assert_close(reconstructed[0, 0], expected)
    torch.testing.assert_close(reconstructed[0, 1], torch.zeros(7, 9))


def test_default_latent_and_spline_dimensions_match_legacy():
    model = SplineWICSI12VAE()

    assert model.latent_dim == 36
    assert model.spline_control_points == 30
    assert model.spline_degree == 3


def test_loss_matches_literal_legacy_formula():
    reconstructed = torch.tensor([
        [[[1.0, 2.0]], [[3.0, 4.0]]],
        [[[2.0, 1.0]], [[4.0, 3.0]]],
    ])
    target = torch.zeros_like(reconstructed)
    mask = torch.tensor([[[[True, True]]], [[[True, False]]]])
    weight = torch.tensor([[[[1.0, 0.5]]], [[[0.25, 1.0]]]])
    mu = torch.ones(2, 3)
    logvar = torch.zeros_like(mu)

    losses = vae_loss(
        reconstructed, target, mask, weight, mu, logvar, beta=1.0
    )
    masked_weight = weight * mask
    expected_reconstruction = (
        torch.sum(masked_weight * (reconstructed - target)**2)
        / (masked_weight.sum() + 1e-8)
        * (2 * 1 * 2)
    )
    expected_kl = -0.5 * torch.sum(
        1 + logvar - mu.pow(2) - logvar.exp()
    )

    torch.testing.assert_close(losses["reconstruction"], expected_reconstruction)
    torch.testing.assert_close(losses["kl"], expected_kl)
    torch.testing.assert_close(
        losses["total"], expected_reconstruction + expected_kl
    )


def test_uniform_weight_scale_cancels_as_in_legacy_loss():
    reconstructed = torch.ones(1, 2, 1, 1)
    target = torch.zeros_like(reconstructed)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    mu = torch.zeros(1, 1)
    logvar = torch.zeros_like(mu)

    full = vae_loss(
        reconstructed, target, mask, torch.ones(1, 1, 1, 1),
        mu, logvar, beta=0.0
    )
    half = vae_loss(
        reconstructed, target, mask, torch.full((1, 1, 1, 1), 0.5),
        mu, logvar, beta=0.0
    )

    torch.testing.assert_close(full["reconstruction"], half["reconstruction"])


def test_measurement_precision_option_preserves_legacy_loss_scale():
    reconstructed = torch.tensor([[[[1.0]], [[2.0]]]])
    target = torch.zeros_like(reconstructed)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    weight = torch.ones(1, 1, 1, 1)
    mu = torch.zeros(1, 1)
    logvar = torch.zeros_like(mu)

    legacy = vae_loss(
        reconstructed, target, mask, weight, mu, logvar, beta=0.0
    )
    uncertainty_weighted = vae_loss(
        reconstructed,
        target,
        mask,
        weight,
        mu,
        logvar,
        beta=0.0,
        transformed_variance=torch.ones_like(reconstructed),
        precision_scale=(1.0, 1.0),
    )

    torch.testing.assert_close(
        uncertainty_weighted["reconstruction"], legacy["reconstruction"]
    )


def test_measurement_precision_changes_channel_residual_weight():
    reconstructed = torch.tensor([[[[1.0]], [[2.0]]]])
    target = torch.zeros_like(reconstructed)
    mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
    weight = torch.ones(1, 1, 1, 1)
    mu = torch.zeros(1, 1)
    logvar = torch.zeros_like(mu)
    variance = torch.tensor([[[[0.25]], [[1.0]]]])

    losses = vae_loss(
        reconstructed,
        target,
        mask,
        weight,
        mu,
        logvar,
        beta=0.0,
        transformed_variance=variance,
        precision_scale=(1.0, 1.0),
        variance_floor=1e-6,
    )

    torch.testing.assert_close(losses["reconstruction"], torch.tensor(6.4))


def test_evaluation_accepts_opt_in_measurement_uncertainty_loss():
    product = synthetic_product(n_samples=2)
    transform = ChannelTransform(scale=[100, 20], center=[0, 0], spread=[1, 1])
    dataset = WICSI12ArrayDataset(product, transform)
    loader = DataLoader(dataset, batch_size=2, shuffle=False)
    weighting = fit_precision_weighting(
        product, transform, maximum_scale_samples=10_000
    )
    model = SplineWICSI12VAE(latent_dim=2)

    evaluation = evaluate_vae(
        model,
        loader,
        beta=0.0,
        loss_configuration=weighting,
    )

    assert np.isfinite(evaluation["total"])


def test_gaussian_uncertainty_matches_analytic_loss_and_bounded_gradient():
    reconstructed = torch.tensor([[[[1.0, 2.0]], [[3.0, 4.0]]]], requires_grad=True)
    target = torch.zeros_like(reconstructed)
    mask = torch.tensor([[[[True, False]]]])
    weight = torch.ones(1, 1, 1, 2)
    mu = torch.zeros(1, 1)
    variance = torch.tensor([[[[0.0, float("nan")]], [[3.0, float("nan")]]]])
    result = vae_loss(reconstructed, target, mask, weight, mu, mu, beta=0,
                      transformed_variance=variance, model_error_std=(1.0, 1.0))
    expected = 4 * (1.0 + 9.0 / 4.0 + np.log(4.0))
    torch.testing.assert_close(result["reconstruction"], torch.tensor(expected, dtype=torch.float32))
    result["total"].backward()
    torch.testing.assert_close(reconstructed.grad, torch.tensor([[[[8.0, 0.0]], [[6.0, 0.0]]]]))


def test_gaussian_zero_measurement_variance_preserves_legacy_for_both_channels():
    recon = torch.randn(2, 2, 3, 3)
    target = torch.zeros_like(recon)
    mask = torch.ones(2, 1, 3, 3, dtype=torch.bool)
    weight = torch.rand(2, 1, 3, 3)
    mu = torch.ones(2, 2)
    logvar = torch.zeros_like(mu)
    legacy = vae_loss(recon, target, mask, weight, mu, logvar, beta=1.6)
    gaussian = vae_loss(recon, target, mask, weight, mu, logvar, beta=1.6,
                        transformed_variance=torch.zeros_like(recon), model_error_std=(0.1, 2.0))
    for name in legacy:
        torch.testing.assert_close(gaussian[name], legacy[name])


def test_gaussian_configuration_validation_and_training_reload(tmp_path):
    for values in ((0, 1), (-1, 1), (np.nan, 1), (1,)):
        try:
            gaussian_uncertainty_configuration(values)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid model-error standard deviation was accepted")
    configuration = gaussian_uncertainty_configuration((0.5, 0.75))
    dataset = WICSI12ArrayDataset(synthetic_product(2), ChannelTransform([100, 20], [0, 0], [1, 1]))
    loader = DataLoader(dataset, batch_size=2)
    model, history = train_vae(SplineWICSI12VAE(latent_dim=2), loader, epochs=1,
                               loss_configuration=configuration)
    filename = tmp_path / "gaussian.pt"
    save_run_artifact(filename, model, dataset.transform, history,
                      {"loss_configuration": configuration}, {}, {})
    restored, _, _ = load_run_artifact(filename)
    assert restored.loss_configuration == configuration
    evaluation = evaluate_vae(restored, loader, loss_configuration=configuration)
    assert np.isfinite(evaluation["total"])


def test_one_epoch_and_artifact_reload(tmp_path, capsys):
    product = synthetic_product()
    product["valid"][0] = False
    product["quality"][0] = np.nan
    transform = ChannelTransform(scale=[100, 20], center=[0, 0], spread=[1, 1])
    dataset = WICSI12ArrayDataset(product, transform)
    np.testing.assert_array_equal(dataset.selected_indices, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(dataset.dropped_indices, [0])
    assert dataset[0]["encoder_input"].shape == (3, 46, 46)
    assert dataset[0]["mask"].shape == (1, 46, 46)
    assert dataset[0]["weight"].shape == (1, 46, 46)
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    model = SplineWICSI12VAE(latent_dim=3)
    model, history = train_vae(
        model,
        loader,
        epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        seed=5,
        loss_configuration=FIXED_WEIGHTING,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["train"]["total"])
    assert model.best_validation_epoch is None
    progress_output = capsys.readouterr()
    assert "Training VAE on cpu" in progress_output.out
    assert "VAE epochs" in progress_output.err and "Epoch 1/1 train" in progress_output.err
    assert "train_loss=" in progress_output.err and "lr=" in progress_output.err
    evaluation = evaluate_vae(model, loader, beta=1e-4)
    assert np.isfinite(evaluation["total"])

    artifact = tmp_path / "run.pt"
    save_run_artifact(
        artifact,
        model,
        transform,
        history,
        training_configuration={
            "epochs": 1,
            "seed": 5,
            "loss_configuration": model.loss_configuration,
        },
        data_provenance={"grid_id": "image_apex_130km_46x46_v1"},
        split_files={"train": ["or_0001.nc"], "validation": [], "test": []},
        evaluation={"test": evaluation},
    )
    reloaded, reloaded_transform, metadata = load_run_artifact(artifact)
    assert reloaded.configuration() == model.configuration()
    assert metadata["artifact_type"] == "wic_si12_vae_run_v1"
    assert np.isfinite(metadata["evaluation"]["test"]["total"])
    assert reloaded_transform.to_dict() == transform.to_dict()
    assert reloaded.loss_configuration == model.loss_configuration


def test_dataset_adds_common_weight_channel_and_legacy_missing_limit():
    product = synthetic_product(n_samples=3)
    product["quality"][:, 0] = 0.75
    product["quality"][:, 1] = 0.5
    flattened = product["valid"].reshape(3, 2, -1)
    flattened[0, :, :698] = False
    flattened[1, :, :699] = False
    transform = ChannelTransform(scale=[100, 20], center=[0, 0], spread=[1, 1])

    dataset = WICSI12ArrayDataset(product, transform)

    np.testing.assert_array_equal(dataset.selected_indices, [0, 2])
    np.testing.assert_array_equal(dataset.dropped_indices, [1])
    assert dataset.missing_fraction[0] <= 0.33
    item = dataset[0]
    np.testing.assert_allclose(item["encoder_input"][2], item["weight"][0])
    np.testing.assert_allclose(item["weight"][0][item["mask"][0]], 0.5)


def test_training_restores_lowest_validation_loss_checkpoint(monkeypatch, capsys):
    model = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        model.weight.zero_()

    def fake_epoch(
        model,
        loader,
        optimizer,
        device,
        beta,
        training,
        loss_configuration=None,
        progress_description=None,
    ):
        if training:
            with torch.no_grad():
                model.weight.add_(1.0)
        value = float(model.weight.item())
        total = value if value == 1.0 else value + 10.0
        return {"total": total, "reconstruction": total, "kl": 0.0}

    monkeypatch.setattr(workflow, "_run_epoch", fake_epoch)
    model, history = train_vae(
        model,
        train_loader=[None],
        validation_loader=[None],
        epochs=3,
        learning_rate=1e-2,
        scheduler_factor=0.5,
        scheduler_patience=0,
        loss_configuration=FIXED_WEIGHTING,
    )

    assert len(history) == 3
    assert model.best_validation_epoch == 1
    assert model.best_validation_total == 1.0
    assert model.weight.item() == 1.0
    assert history[0]["learning_rate"] == 1e-2
    assert history[1]["learning_rate"] == 5e-3
    assert history[2]["learning_rate"] == 2.5e-3
    assert "val_loss=" in capsys.readouterr().err


def test_training_requires_explicit_loss_configuration():
    model = WICSI12VAE(latent_dim=2)
    try:
        train_vae(model, train_loader=[None], epochs=1)
    except ValueError as error:
        assert "loss_configuration is required" in str(error)
    else:
        raise AssertionError("training accepted an implicit uncertainty loss")


def test_plateau_scheduler_monitors_training_loss_not_validation(monkeypatch):
    model = torch.nn.Linear(1, 1, bias=False)
    train_totals = iter((1.0, 2.0, 3.0))
    validation_totals = iter((3.0, 2.0, 1.0))

    def fake_epoch(
        model,
        loader,
        optimizer,
        device,
        beta,
        training,
        loss_configuration=None,
        progress_description=None,
    ):
        total = next(train_totals if training else validation_totals)
        return {"total": total, "reconstruction": total, "kl": 0.0}

    monkeypatch.setattr(workflow, "_run_epoch", fake_epoch)
    _model, history = train_vae(
        model,
        train_loader=[None],
        validation_loader=[None],
        epochs=3,
        learning_rate=1e-2,
        scheduler_factor=0.5,
        scheduler_patience=0,
        loss_configuration=FIXED_WEIGHTING,
    )

    assert history[1]["learning_rate"] == 5e-3
    assert history[2]["learning_rate"] == 2.5e-3
