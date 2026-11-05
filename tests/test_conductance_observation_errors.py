"""Native retrieval-error propagation and heteroscedastic Gaussian conditioning."""

import json

import numpy as np
import pytest
import torch

from scripts.IMAGE.vae_pipeline import plot_conductance_reconstruction as script
from scripts.IMAGE.vae_pipeline.plot_wic_si12_reconstruction import reconstruct


def options(tmp_path, *flags):
    return script.parse_arguments([str(tmp_path), "--output", str(tmp_path / "plots"), *flags])


def test_native_default_and_explicit_fixed_override(tmp_path):
    assert script.observation_error_mode(options(tmp_path), True) == "propagated"
    assert script.observation_error_mode(options(tmp_path), False) == "fixed"
    assert script.observation_error_mode(options(tmp_path, "--observation-std", ".1"), True) == "fixed"
    with pytest.raises(ValueError, match="omit it"):
        script.observation_error_mode(options(tmp_path, "--observation-errors", "propagated", "--observation-std", ".1"), True)
    with pytest.raises(ValueError, match="native"):
        script.observation_error_mode(options(tmp_path, "--observation-errors", "propagated"), False)


def test_dH_dP_log_errors_and_floor_use_only_retained_observations(tmp_path):
    truth = np.full((2, 2, 3), 9.0)
    std = np.stack((np.full((2, 3), 1.0), np.full((2, 3), 2.0)))
    retained = np.ones_like(truth, dtype=bool)
    retained[:, :, 2] = False
    original = std.copy()
    errors, report = script.observation_errors(truth, std, retained, options(tmp_path), "propagated")
    np.testing.assert_allclose(errors[retained], std[retained] / (10 * np.log(10)))
    assert report["observation_error_source"] == "native_dH_dP"
    assert report["observation_std_floored_cells"] == 0
    std[~retained] = np.nan
    hidden_changed, _ = script.observation_errors(truth, std, retained, options(tmp_path), "propagated")
    np.testing.assert_array_equal(hidden_changed, errors)
    std[0, 0, 0] = 0
    with pytest.raises(ValueError, match="contain zeros"):
        script.observation_errors(truth, std, retained, options(tmp_path), "propagated")
    errors, report = script.observation_errors(truth, std, retained, options(tmp_path, "--minimum-observation-std", "1.5"), "propagated")
    assert report["observation_std_floored_cells"] == retained[0].sum()
    assert errors[0, 0, 0] == pytest.approx(1.5 / (10 * np.log(10)))
    np.testing.assert_array_equal(original[1], np.full((2, 3), 2.0))


@pytest.mark.parametrize("bad", ["-1", "nan", "inf"])
def test_invalid_physical_error_floor_rejected(tmp_path, bad):
    with pytest.raises(ValueError, match="nonnegative"):
        script.observation_error_mode(options(tmp_path, "--minimum-observation-std", bad), True)


def test_heteroscedastic_update_matches_dense_Gaussian_and_ignores_hidden_errors():
    rng = np.random.default_rng(9)
    mean = np.ones((2, 2, 3))
    factor = rng.normal(size=(mean.size, 4)) * .1
    truth = np.full_like(mean, 12)
    retained = np.ones_like(mean, dtype=bool)
    retained[1] = False
    errors = np.linspace(.02, .3, mean.size).reshape(mean.shape)
    indices = np.flatnonzero(retained)
    covariance = factor @ factor.T
    innovation = covariance[np.ix_(indices, indices)] + np.diag(errors.ravel()[indices]**2)
    residual = np.log10(truth.ravel()[indices] + 1) - mean.ravel()[indices]
    expected_mean = mean.ravel() + covariance[:, indices] @ np.linalg.solve(innovation, residual)
    expected_cov = covariance - covariance[:, indices] @ np.linalg.solve(innovation, covariance[indices])
    result = reconstruct(mean, factor, truth, retained, errors)
    np.testing.assert_allclose(result["posterior_log_mean"].ravel(), expected_mean, atol=1e-12)
    np.testing.assert_allclose(result["posterior_log_std"].ravel()**2, np.diag(expected_cov), atol=1e-12)
    truth[~retained] = np.nan
    errors[~retained] = np.nan
    repeated = reconstruct(mean, factor, truth, retained, errors)
    np.testing.assert_array_equal(result["posterior_log_mean"], repeated["posterior_log_mean"])
    np.testing.assert_array_equal(result["posterior_log_std"], repeated["posterior_log_std"])
    scalar = reconstruct(mean, factor, truth, retained, .1)
    constant_map = reconstruct(mean, factor, truth, retained, np.full(mean.shape, .1))
    np.testing.assert_array_equal(scalar["posterior_log_mean"], constant_map["posterior_log_mean"])


def test_high_uncertainty_observation_has_less_influence():
    mean = np.ones((2, 1, 1))
    factor = np.eye(2) * .2
    truth = np.full_like(mean, 99)
    errors = np.array([.01, 1.0]).reshape(mean.shape)
    result = reconstruct(mean, factor, truth, np.ones_like(mean, bool), errors)
    assert result["posterior_log_mean"][0, 0, 0] > 1.99
    assert result["posterior_log_mean"][1, 0, 0] < 1.04
    assert result["posterior_log_std"][0, 0, 0] < result["posterior_log_std"][1, 0, 0]


@pytest.mark.parametrize("case", ["half", "fields"])
def test_native_figures_save_actual_errors_and_preserve_artifact(tmp_path, monkeypatch, case):
    from types import SimpleNamespace
    logs = 1 + .1 * np.random.default_rng(3).normal(size=(20, 2, 2, 4))
    samples = 10**logs - 1
    mean = samples.mean(axis=0)
    factor = (samples.reshape(20, -1) - mean.ravel()).T / np.sqrt(19)
    np.savez_compressed(tmp_path / "ensemble_products.npz", conductance_mean=mean, conductance_anomaly_factor=factor)
    artifact = {"artifact_type": "conductance_vae_run_v1", "transform": {}, "data_provenance": {"training_support": np.ones((2, 4), bool)}}
    torch.save(artifact, tmp_path / "conductance_vae.pt")
    (tmp_path / "summary.json").write_text(json.dumps({"input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S", "propagation_reapplied": False}))
    class Dataset(SimpleNamespace):
        def __len__(self): return len(self.values)
    std = np.stack((np.full((2, 4), .4), np.full((2, 4), .8)))[None]
    dataset = Dataset(values=samples[:1], standard_deviation=std.copy(), common_valid=np.ones((1, 1, 2, 4), bool), frame_orbit=np.array([209]), frame_time=np.array(["2000-07-28"]))
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda configuration: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    before = (tmp_path / "conductance_vae.pt").read_bytes()
    script.run(options(tmp_path, "--modes", "3", "--dpi", "20"), case=case)
    names = ["figure_4"] if case == "half" else ["figure_5a", "figure_5b"]
    for name in names:
        report = json.loads((tmp_path / "plots" / f"{name}_reconstruction.json").read_text())
        assert report["observation_error_source"] == "native_dH_dP"
        assert report["observation_std_dex"] is None and not report["model_error_included"]
        with np.load(tmp_path / "plots" / f"{name}_reconstruction.npz") as saved:
            retained = np.broadcast_to(saved["retained"], samples[0].shape)
            expected = std[0][retained] / (np.log(10) * (samples[0][retained] + 1))
            np.testing.assert_allclose(saved["observation_std_dex"][retained], expected)
            assert np.isnan(saved["observation_std_dex"][~retained]).all()
            np.testing.assert_array_equal(saved["input_standard_deviation_siemens"], std[0])
    assert (tmp_path / "conductance_vae.pt").read_bytes() == before
    np.testing.assert_array_equal(dataset.standard_deviation, std)
