"""Log-prior PCA, coefficient-space OI parity, and no hidden-half leakage."""

from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from scripts.IMAGE.vae_pipeline import plot_wic_si12_reconstruction as script


def saved_prior(directory):
    logs = 1 + 0.15 * np.random.default_rng(5).normal(size=(80, 2, 2, 4))
    samples = 10**logs - 1
    mean = samples.mean(axis=0)
    factor = np.ascontiguousarray((samples.reshape(80, -1) - mean.ravel()).T / np.sqrt(79))
    np.savez_compressed(directory / "ensemble_products.npz", fuv_mean=mean, fuv_anomaly_factor=factor)
    return logs, samples


def test_log_prior_recomputes_mean_and_covariance_without_changing_archive(tmp_path):
    logs, _ = saved_prior(tmp_path)
    source = (tmp_path / "ensemble_products.npz").read_bytes()
    prior = script.log_prior(tmp_path, n_modes=5)
    np.testing.assert_allclose(prior["mean"], logs.mean(axis=0), atol=1e-12)
    expected = np.linalg.eigvalsh(np.cov(logs.reshape(80, -1), rowvar=False))[::-1][:5]
    np.testing.assert_allclose(prior["eigenvalues"], expected, rtol=1e-12)
    assert (tmp_path / "ensemble_products.npz").read_bytes() == source


def test_reconstruction_matches_dense_OI_and_does_not_use_removed_values(tmp_path):
    _, samples = saved_prior(tmp_path)
    prior = script.log_prior(tmp_path, n_modes=5)
    factor = prior["patterns"].reshape(5, -1).T
    retained = script.retained_half(np.ones((2, 4), dtype=bool))
    result = script.reconstruct(prior["mean"], factor, samples[0], retained, 0.1)
    indices = np.flatnonzero(np.broadcast_to(retained, (2, 2, 4)))
    covariance = factor @ factor.T
    innovation_covariance = covariance[np.ix_(indices, indices)] + np.eye(indices.size) * 0.1**2
    residual = np.log10(samples[0].ravel()[indices] + 1) - prior["mean"].ravel()[indices]
    expected_mean = prior["mean"].ravel() + covariance[:, indices] @ np.linalg.solve(innovation_covariance, residual)
    expected_covariance = covariance - covariance[:, indices] @ np.linalg.solve(innovation_covariance, covariance[indices])
    np.testing.assert_allclose(result["posterior_log_mean"].ravel(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(result["posterior_log_std"].ravel()**2, np.diag(expected_covariance), atol=1e-12)
    corrupted = samples[0].copy()
    corrupted[:, ~retained] = np.nan
    repeated = script.reconstruct(prior["mean"], factor, corrupted, retained)
    np.testing.assert_array_equal(result["reconstruction"], repeated["reconstruction"])
    np.testing.assert_array_equal(result["uncertainty"], repeated["uncertainty"])
    log_variance = np.log(10)**2 * np.diag(expected_covariance)
    expected_std = np.sqrt(np.expm1(log_variance)) * np.exp(np.log(10) * expected_mean + log_variance / 2)
    np.testing.assert_allclose(result["uncertainty"].ravel(), expected_std, atol=1e-12)


@pytest.mark.parametrize("half", ["left", "right", "top", "bottom"])
def test_removes_exactly_half_and_preserves_original_missing_cells(half):
    valid = np.ones((4, 6), dtype=bool)
    retained = script.retained_half(valid, half)
    assert retained.sum() == valid.size // 2
    valid[0, 0] = False
    assert not script.retained_half(valid, half)[0, 0]


def test_panels_truth_input_reconstruction_uncertainty_and_shared_scales():
    truth = np.arange(16).reshape(2, 2, 4)
    valid = np.ones((2, 4), dtype=bool)
    retained = script.retained_half(valid)
    figure = script.make_figure(truth, valid, retained, truth + 1, np.ones_like(truth))
    assert len(figure.axes) == 8 and all(not axis.axison for axis in figure.axes)
    for row in range(2):
        assert np.isnan(figure.axes[row * 4 + 1].images[0].get_array().filled(np.nan)[:, 2:]).all()
        assert all(figure.axes[row * 4 + column].images[0].get_clim() == figure.axes[row * 4].images[0].get_clim() for column in (1, 2))
        np.testing.assert_array_equal(figure.axes[row * 4 + 3].images[0].get_array(), 1)
    plt.close(figure)


def test_cli_loads_only_heldout_truth_and_saves_outputs(tmp_path, monkeypatch):
    _, samples = saved_prior(tmp_path)
    artifact = {"transform": {}, "split_files": {"test": ["test.nc"]}}
    torch.save(artifact, tmp_path / "wic_si12_vae.pt")
    class FakeDataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    dataset = FakeDataset(values=samples[:1], common_valid=np.ones((1, 1, 2, 4), dtype=bool), frame_orbit=np.array([150]), frame_time=np.array(["2000-01-01"]))
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda settings: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    output = tmp_path / "figure_4.png"
    script.run(script.parse_arguments([str(tmp_path), "--output", str(output), "--modes", "5", "--dpi", "30"]))
    assert output.is_file() and (tmp_path / "figure_4_reconstruction.json").is_file()
    with np.load(tmp_path / "figure_4_reconstruction.npz") as archive:
        assert archive["reconstruction"].shape == (2, 2, 4)
