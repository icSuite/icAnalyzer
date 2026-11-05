"""Coefficient PCA units/projection, legacy OI parity, and both figure CLIs."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
from scipy import linalg
import torch

from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import save_run_artifact
from scripts.IMAGE.vae_pipeline import plot_wic_si12_reconstruction as half_script
from scripts.IMAGE.vae_pipeline import plot_wic_si12_sensor_reconstruction as sensor_script
from scripts.IMAGE.vae_pipeline.wic_si12_coefficient_prior import coefficient_prior


@pytest.fixture
def coefficient_run(tmp_path):
    torch.manual_seed(6)
    model = SplineWICSI12VAE(image_shape=(8, 10), latent_dim=3, spline_control_points=4).eval()
    with torch.no_grad():
        model.spline_conv.bias[0] = -1
    transform = ChannelTransform([1, 1], [0.05, 1.3], [0.7, 1.3], kind=ChannelTransform.LOG10_KIND)
    latent = np.random.default_rng(4).normal(size=(37, 3)).astype(np.float32)
    save_run_artifact(tmp_path / "wic_si12_vae.pt", model, transform, [], {}, {}, {"test": ["unused.nc"]})
    # No physical anomaly factor: coefficient mode must use ONLY saved latents.
    np.savez_compressed(tmp_path / "ensemble_products.npz", latent_samples=latent, fuv_mean=np.ones((2, 8, 10)))
    with torch.inference_mode():
        normalized = model.decode_coefficients(torch.from_numpy(latent)).numpy().astype(float)
        log_maps = model.decode(torch.from_numpy(latent)).numpy().astype(float)
    coefficients = normalized * transform.spread[None, :, None, None] + transform.center[None, :, None, None]
    log_maps = log_maps * transform.spread[None, :, None, None] + transform.center[None, :, None, None]
    return tmp_path, model, transform, coefficients, log_maps


def test_coefficient_pca_matches_direct_covariance_and_restores_log_count_units(coefficient_run):
    directory, model, transform, coefficients, log_maps = coefficient_run
    source = {name: (directory / name).read_bytes() for name in ("wic_si12_vae.pt", "ensemble_products.npz")}
    prior = coefficient_prior(directory, n_modes=5, batch_size=8)
    np.testing.assert_allclose(prior["coefficient_mean"], coefficients.mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(prior["mean"], log_maps.mean(axis=0), atol=2e-7)
    expected_covariance = np.cov(coefficients.reshape(37, -1), rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(expected_covariance)
    np.testing.assert_allclose(prior["eigenvalues"], eigenvalues[::-1][:5], rtol=2e-5, atol=1e-12)
    coefficient_factor = prior["coefficient_patterns"].reshape(5, -1).T
    expected_factor = eigenvectors[:, -5:] * np.sqrt(np.maximum(eigenvalues[-5:], 0))
    # Decoder convolutions are float32 and differ slightly with batch shape.
    np.testing.assert_allclose(coefficient_factor @ coefficient_factor.T, expected_factor @ expected_factor.T, rtol=3e-4, atol=2e-9)
    # Negative log coefficients/maps are retained, not clipped through counts.
    assert np.any(prior["mean"] < 0)
    assert prior["sample_count"] == 37 and prior["decode_device"] == "cpu"
    for name, data in source.items():
        assert (directory / name).read_bytes() == data


def test_batches_use_every_saved_latent_and_preserve_projected_prior(coefficient_run):
    directory, _, _, _, _ = coefficient_run
    small = coefficient_prior(directory, n_modes=5, batch_size=7)
    large = coefficient_prior(directory, n_modes=5, batch_size=64)
    np.testing.assert_allclose(small["mean"], large["mean"], atol=1e-7)
    np.testing.assert_allclose(small["eigenvalues"], large["eigenvalues"], rtol=2e-5, atol=1e-12)
    small_factor = small["patterns"].reshape(5, -1).T
    large_factor = large["patterns"].reshape(5, -1).T
    np.testing.assert_allclose(small_factor @ small_factor.T, large_factor @ large_factor.T, atol=3e-10)


def test_projected_coefficient_update_matches_legacy_G_Q_G_transpose(coefficient_run):
    directory, _, _, _, _ = coefficient_run
    prior = coefficient_prior(directory, n_modes=5, batch_size=8)
    G = np.einsum("xi,yj->yxij", prior["column_basis"], prior["row_basis"]).reshape(80, 16)
    G_joint = linalg.block_diag(G, G)
    coefficient_factor = prior["coefficient_patterns"].reshape(5, -1).T
    Q = coefficient_factor @ coefficient_factor.T
    projected_factor = prior["patterns"].reshape(5, -1).T
    np.testing.assert_allclose(projected_factor, G_joint @ coefficient_factor, atol=1e-14)
    truth = np.maximum(10**prior["mean"] - 1, 0)
    retained = half_script.retained_half(np.ones((8, 10), dtype=bool))
    posterior = half_script.reconstruct(prior["mean"], projected_factor, truth, retained, 0.1)
    indices = np.flatnonzero(np.broadcast_to(retained, truth.shape))
    G_observed = G_joint[indices]
    S = G_observed @ Q @ G_observed.T + np.eye(len(indices)) * 0.1**2
    K = linalg.solve(S, (Q @ G_observed.T).T, assume_a="pos").T
    mean_coefficients = prior["coefficient_mean"].ravel()
    residual = np.log10(truth.ravel()[indices] + 1) - G_observed @ mean_coefficients
    expected_mean = G_joint @ (mean_coefficients + K @ residual)
    posterior_Q = Q - K @ G_observed @ Q
    expected_variance = np.diag(G_joint @ posterior_Q @ G_joint.T)
    np.testing.assert_allclose(posterior["posterior_log_mean"].ravel(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(posterior["posterior_log_std"].ravel()**2, expected_variance, atol=1e-12)
    truth[:, ~retained] = np.nan
    hidden = half_script.reconstruct(prior["mean"], projected_factor, truth, retained)
    np.testing.assert_array_equal(hidden["reconstruction"], posterior["reconstruction"])


@pytest.mark.parametrize("script", [half_script, sensor_script])
def test_both_coefficient_variations_save_arrays_and_provenance(coefficient_run, monkeypatch, script):
    directory, _, _, _, log_maps = coefficient_run
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    truth = np.maximum(10**log_maps[:1] - 1, 0)
    dataset = Dataset(values=truth, common_valid=np.ones((1, 1, 8, 10), dtype=bool), frame_orbit=np.array([150]), frame_time=np.array(["2000-01-01"]))
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    def forbidden_pixel_prior(*args):
        pytest.fail("coefficient mode must not calculate PCA from physical pixel draws")
    monkeypatch.setattr(script, "log_prior", forbidden_pixel_prior)
    calls = []
    def counted_prior(*args):
        calls.append(1)
        return coefficient_prior(*args)
    monkeypatch.setattr(script, "coefficient_prior", counted_prior)
    output = directory / "coefficient_figures"
    target = output / "figure_4.png" if script is half_script else output
    arguments = script.parse_arguments([str(directory), "--prior-space", "coefficients", "--modes", "5", "--decode-batch-size", "8", "--output", str(target), "--dpi", "30"])
    script.run(arguments)
    assert calls == [1]
    names = ["figure_4"] if script is half_script else ["figure_5a", "figure_5b"]
    for name in names:
        assert (output / f"{name}.png").is_file()
        report = json.loads((output / f"{name}_reconstruction.json").read_text())
        assert report["prior_space"] == "coefficients" and report["decode_device"] == "cpu"
        assert report["negative_draw_cells_clipped"] is None
        assert "spline coefficients" in report["variance_space"]
        with np.load(output / f"{name}_reconstruction.npz") as archive:
            np.testing.assert_array_equal(archive["truth"], truth[0])
            assert archive["coefficient_patterns"].shape == (5, 2, 4, 4)
            if script is sensor_script:
                removed_channel = 1 if name.endswith("a") else 0
                assert not archive["retained"][removed_channel].any()


@pytest.mark.parametrize("settings", [{"batch_size": 0}, {"n_modes": 0}, {"n_modes": 33}])
def test_invalid_coefficient_settings_fail_clearly(coefficient_run, settings):
    directory, _, _, _, _ = coefficient_run
    with pytest.raises(ValueError):
        coefficient_prior(directory, **settings)


def test_non_log_transform_is_rejected(coefficient_run):
    directory, model, _, _, _ = coefficient_run
    transform = ChannelTransform([1, 1], [0, 0], [1, 1])
    save_run_artifact(directory / "wic_si12_vae.pt", model, transform, [], {}, {}, {})
    with pytest.raises(ValueError, match="log10 spline VAE"):
        coefficient_prior(directory, n_modes=5)
