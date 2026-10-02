import numpy as np
import torch
from torch.utils.data import DataLoader

import icanalyzer.wic_si12.workflow as workflow
from icanalyzer.wic_si12.data import (
    ChannelTransform,
    WICSI12ArrayDataset,
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
    load_run_artifact,
    save_run_artifact,
    train_vae,
)


FIXED_WEIGHTING = LEGACY_LOSS_CONFIGURATION


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


def test_one_epoch_and_artifact_reload(tmp_path):
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


def test_training_restores_lowest_validation_loss_checkpoint(monkeypatch):
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
