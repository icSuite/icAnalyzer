"""Decoder MAP/uncertainty against a linear reference; masking and CLI checks."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import save_run_artifact
from scripts.IMAGE.vae_pipeline import plot_wic_si12_reconstruction as half_script
from scripts.IMAGE.vae_pipeline import plot_wic_si12_sensor_reconstruction as sensor_script
from scripts.IMAGE.vae_pipeline.wic_si12_decoder_reconstruction import fit_decoder, load_decoder


class LinearDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_shape = (2, 3)
        self.latent_dim = 2
        self.weight = nn.Parameter(torch.tensor(np.random.default_rng(3).normal(size=(12, 2)) * 0.1))
        self.bias = nn.Parameter(torch.ones(12, dtype=torch.float64))

    def decode(self, latent):
        return (latent @ self.weight.T + self.bias).reshape(-1, 2, 2, 3)


@pytest.mark.parametrize("mask_kind", ["half", "WIC", "SI12"])
def test_linear_decoder_matches_analytic_latent_MAP_and_uncertainty_without_hidden_leakage(mask_kind):
    model = LinearDecoder()
    transform = ChannelTransform([1, 1], [0.2, 0.1], [0.7, 1.3], kind=ChannelTransform.LOG10_KIND)
    spread = np.repeat(transform.spread, 6)
    center = np.repeat(transform.center, 6)
    B = model.weight.detach().numpy() * spread[:, None]
    mean = model.bias.detach().numpy() * spread + center
    truth = (10**(mean + B @ np.array([0.8, -0.6])) - 1).reshape(2, 2, 3)
    valid = np.ones((2, 3), dtype=bool)
    retained = np.broadcast_to(half_script.retained_half(valid), truth.shape) if mask_kind == "half" else sensor_script.retained_sensor(valid, 1 if mask_kind == "WIC" else 0)
    indices = np.flatnonzero(retained)
    observation_std, prior_weight = 0.1, 1.4
    precision = prior_weight * np.eye(2) + B[indices].T @ B[indices] / observation_std**2
    latent_covariance = np.linalg.inv(precision)
    expected_latent = latent_covariance @ B[indices].T @ (np.log10(truth.ravel()[indices] + 1) - mean[indices]) / observation_std**2
    expected_mean = mean + B @ expected_latent
    expected_variance = np.einsum("ij,jk,ik->i", B, latent_covariance, B)
    weights_before = model.weight.detach().numpy().copy()
    posterior, diagnostics = fit_decoder(model, transform, truth, retained, observation_std, prior_weight, restarts=3, max_iterations=100, seed=4)
    np.testing.assert_allclose(posterior["latent_map"], expected_latent, atol=1e-6)
    np.testing.assert_allclose(posterior["latent_covariance"], latent_covariance, atol=1e-12)
    np.testing.assert_allclose(posterior["posterior_log_mean"].ravel(), expected_mean, atol=1e-7)
    np.testing.assert_allclose(posterior["posterior_log_std"].ravel()**2, expected_variance, atol=1e-12)
    natural_variance = np.log(10)**2 * expected_variance
    expected_std = np.sqrt(np.expm1(natural_variance)) * np.exp(np.log(10) * expected_mean + 0.5 * natural_variance)
    np.testing.assert_allclose(posterior["uncertainty"].ravel(), expected_std, rtol=1e-6)
    np.testing.assert_array_equal(model.weight.detach().numpy(), weights_before)
    corrupted = truth.copy()
    corrupted[~retained] = np.nan
    repeated, repeated_diagnostics = fit_decoder(model, transform, corrupted, retained, observation_std, prior_weight, restarts=3, max_iterations=100, seed=4)
    for key in ("latent_map", "latent_covariance", "reconstruction", "uncertainty"):
        np.testing.assert_array_equal(posterior[key], repeated[key])
    assert diagnostics == repeated_diagnostics
    assert diagnostics["best_fit_converged"] and diagnostics["restarts"] == 3


def test_local_uncertainty_contracts_with_more_accurate_observations():
    model = LinearDecoder()
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    truth = np.full((2, 2, 3), 9.0)
    retained = np.ones_like(truth, dtype=bool)
    precise, _ = fit_decoder(model, transform, truth, retained, observation_std=0.1, restarts=1)
    noisy, _ = fit_decoder(model, transform, truth, retained, observation_std=0.5, restarts=1)
    assert np.all(precise["posterior_log_std"] < noisy["posterior_log_std"])


@pytest.mark.parametrize("settings", [{"observation_std": 0}, {"prior_weight": 0}, {"restarts": 0}, {"max_iterations": 0}])
def test_invalid_fit_settings_are_rejected(settings):
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    with pytest.raises(ValueError):
        fit_decoder(LinearDecoder(), transform, np.ones((2, 2, 3)), np.ones((2, 3), dtype=bool), **settings)


@pytest.mark.parametrize("script", [half_script, sensor_script])
def test_decoder_CLI_uses_saved_model_not_ensemble_or_encoder_and_saves_latent_arrays(tmp_path, monkeypatch, script):
    torch.manual_seed(8)
    model = SplineWICSI12VAE(image_shape=(8, 10), latent_dim=3, spline_control_points=4).eval()
    transform = ChannelTransform([1, 1], [1.0, 0.7], [0.3, 0.2], kind=ChannelTransform.LOG10_KIND)
    artifact_path = tmp_path / "wic_si12_vae.pt"
    save_run_artifact(artifact_path, model, transform, [], {}, {}, {"test": ["unused.nc"]})
    source = artifact_path.read_bytes()
    with torch.no_grad():
        truth = transform.inverse(model.decode(torch.tensor([[0.2, -0.4, 0.1]]))).numpy()
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    dataset = Dataset(values=truth, common_valid=np.ones((1, 1, 8, 10), dtype=bool), frame_orbit=np.array([267]), frame_time=np.array(["2000-08-31"]))
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    def forbidden(*args):
        pytest.fail("decoder fitting must not use PCA or an encoder")
    monkeypatch.setattr(script, "log_prior", forbidden)
    monkeypatch.setattr(script, "coefficient_prior", forbidden)
    monkeypatch.setattr(SplineWICSI12VAE, "encode", forbidden)
    calls = []
    def counted_decoder(*args):
        calls.append(1)
        return load_decoder(*args)
    monkeypatch.setattr(script, "load_decoder", counted_decoder)
    output = tmp_path / "decoder_figures"
    target = output / "figure_4.png" if script is half_script else output
    script.run(script.parse_arguments([str(tmp_path), "--reconstruction-method", "decoder", "--latent-restarts", "2", "--latent-max-iterations", "80", "--seed", "4", "--output", str(target), "--dpi", "30"]))
    assert calls == [1] and artifact_path.read_bytes() == source
    names = ["figure_4"] if script is half_script else ["figure_5a", "figure_5b"]
    for name in names:
        assert (output / f"{name}.png").is_file()
        report = json.loads((output / f"{name}_reconstruction.json").read_text())
        assert report["reconstruction_method"] == "decoder" and report["modes"] is None
        assert report["sample_count"] is None and report["decoder_fit"]["seed"] == 4
        assert "local Gauss-Newton" in report["uncertainty"]
        with np.load(output / f"{name}_reconstruction.npz", allow_pickle=False) as archive:
            assert archive["latent_map"].shape == (3,) and archive["latent_covariance"].shape == (3, 3)
            assert "eigenvalues" not in archive and "prior_log_mean" not in archive
            np.testing.assert_array_equal(archive["truth"], truth[0])


def test_load_decoder_rejects_non_log_transform(tmp_path):
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=4)
    save_run_artifact(tmp_path / "wic_si12_vae.pt", model, ChannelTransform([1, 1], [0, 0], [1, 1]), [], {}, {}, {})
    with pytest.raises(ValueError, match="log10 model"):
        load_decoder(tmp_path)
