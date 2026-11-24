"""Native-grid hierarchical encoder, latent groups and spline decoder."""

import numpy as np
import pytest
import torch
from torch import nn

from icanalyzer.cvae.artifacts import load_run_artifact, save_run_artifact
from icanalyzer.cvae.cache import OrbitBatchSampler
from icanalyzer.cvae.data import ChannelTransform
from icanalyzer.cvae.hierarchical import HierarchicalSplineConditionalVAE
from icanalyzer.cvae.ensemble import sample_decoded_ensemble
from icanalyzer.cvae.loss import vae_loss
from icanalyzer.cvae.training import kl_beta_for_epoch, train_vae


KP_CONDITION = {
    "center": 3.0, "scale": 1.0,
    "training_min": 0.0, "training_max": 9.0,
    "embedding_dim": 8,
}
NAMED_CONDITION = {
    "kind": "named_standardization_v1",
    "feature_names": ["sml", "by", "bz", "tilt"],
    "center": [0.0, 0.0, 0.0, 0.0],
    "scale": [100.0, 5.0, 5.0, 10.0],
    "training_min": [-1000.0, -20.0, -20.0, -35.0],
    "training_max": [100.0, 20.0, 20.0, 35.0],
    "embedding_dim": 12,
}


def small_model(magnetic_latitude=None):
    return HierarchicalSplineConditionalVAE(
        image_shape=(64, 64), global_latent_dim=4,
        spatial_latent_channels=2, spline_control_points=16,
        spline_degree=3, quality_weight_channels=2,
        encoder_mask_policy="union", kp_conditioning=KP_CONDITION,
        field_names=("wic_corrected", "si12"),
        magnetic_latitude=magnetic_latitude,
    )


def test_native_grid_has_planned_scales_and_latent_shapes():
    model = HierarchicalSplineConditionalVAE(
        image_shape=(184, 184), global_latent_dim=36,
        spatial_latent_channels=8, spline_control_points=92,
        quality_weight_channels=2, kp_conditioning=KP_CONDITION,
        field_names=("wic_corrected", "si12"),
    )

    assert model.encoder_stage_shapes == (
        (92, 92), (46, 46), (23, 23), (12, 12), (6, 6)
    )
    assert model.decoder_shapes == ((12, 12), (23, 23), (46, 46), (92, 92))
    assert model.spatial_latent_shape == (8, 6, 6)
    assert model.latent_dim == 36 + 8 * 6 * 6
    assert model.latent_groups == {
        "global": (0, 36), "spatial": (36, 324)
    }


def test_forward_uses_spatial_latent_and_separate_spline_heads():
    torch.manual_seed(4)
    model = small_model()
    values = torch.randn(2, 4, 64, 64)
    valid = torch.ones(2, 2, 64, 64, dtype=torch.bool)
    valid[:, 0, :8, :8] = False

    reconstructed, mu, logvar = model(values, valid, kp=torch.tensor([1., 5.]))
    coefficients = model.decode_coefficients(
        torch.zeros(2, model.latent_dim), kp=torch.tensor([1., 5.])
    )

    assert reconstructed.shape == (2, 2, 64, 64)
    assert coefficients.shape == (2, 2, 16, 16)
    assert mu.shape == logvar.shape == (2, 12)
    assert model.sensor_heads[0] is not model.sensor_heads[1]
    reconstructed.sum().backward()
    assert model.spatial_projection.weight.grad.abs().sum() > 0
    assert model.encoder_blocks[0].film.weight.grad.abs().sum() > 0


def test_masked_encoder_ignores_unobserved_values():
    torch.manual_seed(5)
    model = small_model().eval()
    valid = torch.ones(1, 2, 64, 64, dtype=torch.bool)
    valid[:, :, 10:30, 12:28] = False
    first = torch.randn(1, 4, 64, 64)
    second = first.clone()
    second[:, :, 10:30, 12:28] = 1e6

    first_mu, first_logvar = model.encode(first, valid, kp=3)
    second_mu, second_logvar = model.encode(second, valid, kp=3)

    torch.testing.assert_close(first_mu, second_mu)
    torch.testing.assert_close(first_logvar, second_logvar)


def test_latent_group_kl_is_reported_separately():
    model = small_model()
    target = torch.zeros(1, 2, 2, 2)
    reconstructed = torch.zeros_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    weight = torch.ones_like(target)
    mu = torch.cat((torch.ones(1, 4), torch.full((1, 8), 2.0)), dim=1)
    logvar = torch.zeros_like(mu)

    losses = vae_loss(
        reconstructed, target, mask, weight, mu, logvar,
        channel_balanced=True, latent_groups=model.latent_groups
    )

    torch.testing.assert_close(losses["kl_global"], torch.tensor(2.0))
    torch.testing.assert_close(losses["kl_spatial"], torch.tensor(16.0))
    torch.testing.assert_close(
        losses["kl"], losses["kl_global"] + losses["kl_spatial"]
    )


def test_kl_warmup_is_deterministic_and_reaches_configured_beta():
    assert kl_beta_for_epoch(0.048, 1, 50) == 0.0
    assert kl_beta_for_epoch(0.048, 50, 50) == 0.048
    assert kl_beta_for_epoch(0.048, 80, 50) == 0.048
    assert kl_beta_for_epoch(0.048, 1, 0) == 0.048


def test_training_resumes_from_last_completed_epoch(monkeypatch):
    import icanalyzer.cvae.training as training

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.zeros(()))

    fail_once = {"enabled": True}

    def run_epoch(model, loader, optimizer, device, beta, training, **kwargs):
        if training:
            with torch.no_grad():
                model.weight.add_(1)
            if fail_once["enabled"] and model.weight.item() == 2:
                fail_once["enabled"] = False
                raise RuntimeError("simulated interruption")
        value = float(10 - model.weight.item())
        return {"total": value, "reconstruction": value, "kl": 0.0}

    monkeypatch.setattr(training, "_run_epoch", run_epoch)
    checkpoints = []
    arguments = dict(
        train_loader=[None], validation_loader=[None], epochs=3,
        loss_configuration={
            "kind": "legacy_weighted_mse_v1",
            "reconstruction_reduction": "global_weight_sum_times_state_size",
            "kl_reduction": "sum_batch_and_latent",
            "measurement_uncertainty_in_loss": False,
        },
        checkpoint_callback=checkpoints.append,
    )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        train_vae(TinyModel(), **arguments)

    assert checkpoints[-1]["completed_epoch"] == 1
    resumed, history = train_vae(
        TinyModel(), resume_state=checkpoints[-1], **arguments
    )

    assert [row["epoch"] for row in history] == [1, 2, 3]
    assert resumed.weight.item() == 3
    assert resumed.best_validation_epoch == 3


def test_orbit_batch_sampler_is_complete_local_and_epoch_deterministic():
    class ShardedDataset:
        shard_ids = np.repeat([0, 1, 2], [3, 5, 4])

        def __len__(self):
            return len(self.shard_ids)

    dataset = ShardedDataset()
    sampler = OrbitBatchSampler(dataset, batch_size=4, seed=7)

    sampler.set_epoch(1)
    first = list(sampler)
    sampler.set_epoch(1)
    repeated = list(sampler)
    sampler.set_epoch(2)
    second = list(sampler)

    assert first == repeated
    assert first != second
    flattened = [index for batch in first for index in batch]
    assert sorted(flattened) == list(range(12))
    assert all(len(batch) == 4 for batch in first)
    assert all(len(set(dataset.shard_ids[batch])) <= 2 for batch in first)
    for shard in (0, 1, 2):
        positions = [index for index in flattened if dataset.shard_ids[index] == shard]
        assert positions == sorted(positions)


def test_hierarchical_artifact_roundtrip_preserves_coordinates(tmp_path):
    latitude = np.broadcast_to(
        np.linspace(50, 90, 64, dtype=np.float32)[:, None], (64, 64)
    ).copy()
    model = small_model(latitude).eval()
    transform = ChannelTransform(
        [1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND
    )
    artifact = tmp_path / "hierarchical.pt"
    save_run_artifact(
        artifact, model, transform, history=[], training_configuration={},
        data_provenance={}, split_files={"train": [], "validation": [], "test": []}
    )

    restored, _, saved = load_run_artifact(artifact)
    latent = torch.zeros(2, model.latent_dim)

    assert isinstance(restored, HierarchicalSplineConditionalVAE)
    assert saved["model_configuration"]["spatial_latent_shape"] == [2, 2, 2]
    torch.testing.assert_close(restored.coordinates, model.coordinates)
    torch.testing.assert_close(
        restored.decode(latent, kp=3), model.decode(latent, kp=3)
    )


def test_joint_conditional_prior_roundtrip_and_automatic_sampling(tmp_path):
    model = HierarchicalSplineConditionalVAE(
        image_shape=(64, 64), global_latent_dim=4,
        spatial_latent_channels=2, spline_control_points=16,
        spline_degree=3, quality_weight_channels=2,
        encoder_mask_policy="union", kp_conditioning=KP_CONDITION,
        latent_prior={
            "kind": "conditional_gaussian_mixture_v1",
            "components": 3, "hidden_dim": 8, "covariance_rank": 2,
        },
        field_names=("wic_corrected", "si12"),
    ).eval()
    with torch.no_grad():
        model.latent_prior.mean_head.bias.fill_(2.5)
        model.latent_prior.logvar_head.bias.fill_(-8.0)
        model.latent_prior.factor_head.weight.zero_()
        model.latent_prior.factor_head.bias.zero_()
    transform = ChannelTransform(
        [1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND
    )
    artifact = tmp_path / "joint-prior.pt"
    save_run_artifact(
        artifact, model, transform, history=[], training_configuration={},
        data_provenance={}, split_files={"train": [], "validation": [], "test": []}
    )

    restored, restored_transform, saved = load_run_artifact(artifact)
    assert saved["model_configuration"]["latent_prior"]["components"] == 3
    assert restored.latent_prior.covariance_rank == 2
    samples, latent = sample_decoded_ensemble(
        restored, restored_transform, 5, batch_size=5, seed=19,
        kp=2, return_latent=True,
    )

    assert samples.shape == (5, 2, 64, 64)
    assert latent.shape == (5, model.latent_dim)
    assert latent.mean() > 2.0


def test_named_conditioning_and_ensemble_sampling_use_total_latent():
    model = HierarchicalSplineConditionalVAE(
        image_shape=(64, 64), global_latent_dim=4,
        spatial_latent_channels=2, spline_control_points=16,
        quality_weight_channels=2, conditioning=NAMED_CONDITION,
        field_names=("wic_corrected", "si12"),
    ).eval()
    transform = ChannelTransform(
        [1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND
    )
    condition = np.asarray([-300.0, 3.0, -4.0, 5.0], dtype=np.float32)

    samples, latent = sample_decoded_ensemble(
        model, transform, 3, batch_size=2, seed=7,
        conditioning=condition, return_latent=True
    )

    assert samples.shape == (3, 2, 64, 64)
    assert latent.shape == (3, model.latent_dim)
    first = model.decode(torch.zeros(1, model.latent_dim), conditioning=condition)
    changed = model.decode(
        torch.zeros(1, model.latent_dim),
        conditioning=np.asarray([-50.0, -3.0, 4.0, -5.0], dtype=np.float32)
    )
    assert not torch.equal(first, changed)
