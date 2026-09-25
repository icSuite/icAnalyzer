import numpy as np
import torch
from torch.utils.data import DataLoader

from icanalyzer.wic_si12.data import (
    ChannelTransform,
    WICSI12ArrayDataset,
)
from icanalyzer.wic_si12.model import WICSI12VAE, vae_loss
from icanalyzer.wic_si12.workflow import (
    evaluate_vae,
    load_run_artifact,
    save_run_artifact,
    train_vae,
)


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
    valid = torch.ones_like(reconstructed, dtype=torch.bool)
    quality = torch.ones_like(reconstructed)
    variance = torch.ones_like(reconstructed)
    valid[:, :, 0, 0] = False
    quality[:, :, 0, 0] = torch.nan
    variance[:, :, 0, 0] = torch.nan
    losses = vae_loss(
        reconstructed, target, valid, quality, variance, mu, logvar, beta=0.1
    )
    assert set(losses) == {"total", "reconstruction", "kl"}
    assert torch.isfinite(losses["total"])


def test_one_epoch_and_artifact_reload(tmp_path):
    product = synthetic_product()
    product["valid"][0] = False
    product["quality"][0] = np.nan
    transform = ChannelTransform(scale=[100, 20], center=[0, 0], spread=[1, 1])
    dataset = WICSI12ArrayDataset(product, transform)
    np.testing.assert_array_equal(dataset.selected_indices, [1, 2, 3, 4, 5])
    np.testing.assert_array_equal(dataset.dropped_indices, [0])
    loader = DataLoader(dataset, batch_size=3, shuffle=False)
    wic_floor = float(-transform.center[0] / transform.spread[0])
    model = WICSI12VAE(latent_dim=3, wic_floor_transformed=wic_floor)
    model, history = train_vae(
        model,
        loader,
        epochs=1,
        learning_rate=1e-3,
        beta=1e-4,
        seed=5,
    )
    assert len(history) == 1
    assert np.isfinite(history[0]["train"]["total"])
    evaluation = evaluate_vae(model, loader, beta=1e-4)
    assert np.isfinite(evaluation["total"])

    artifact = tmp_path / "run.pt"
    save_run_artifact(
        artifact,
        model,
        transform,
        history,
        training_configuration={"epochs": 1, "seed": 5},
        data_provenance={"grid_id": "image_apex_130km_46x46_v1"},
        split_files={"train": ["or_0001.nc"], "validation": [], "test": []},
        evaluation={"test": evaluation},
    )
    reloaded, reloaded_transform, metadata = load_run_artifact(artifact)
    assert reloaded.configuration() == model.configuration()
    assert metadata["artifact_type"] == "wic_si12_vae_run_v1"
    assert np.isfinite(metadata["evaluation"]["test"]["total"])
    assert reloaded_transform.to_dict() == transform.to_dict()
