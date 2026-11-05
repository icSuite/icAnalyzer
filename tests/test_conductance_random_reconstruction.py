"""Figure 6: exact seeded masks, zero-observation prior and saved provenance."""

import json

import numpy as np
import pytest

from scripts.IMAGE.vae_pipeline import plot_conductance_random_reconstruction as script


@pytest.mark.parametrize("percent", [0, 37.5, 98, 100])
def test_mask_removes_requested_fraction_of_only_available_pixels(percent):
    valid = np.ones((5, 8), dtype=bool)
    valid[:, 0] = False
    original = valid.copy()
    mask = script.random_retained_mask(valid, percent, seed=8)
    assert np.count_nonzero(valid & ~mask) == int(np.rint(valid.sum() * percent / 100))
    assert not mask[:, 0].any()
    np.testing.assert_array_equal(valid, original)
    np.testing.assert_array_equal(mask, script.random_retained_mask(valid, percent, seed=8))
    if percent == 0:
        np.testing.assert_array_equal(mask, valid)
    if percent == 100:
        assert not mask.any()


@pytest.mark.parametrize("percent", [-1, 101, np.nan, np.inf])
def test_invalid_percentage_rejected(percent):
    with pytest.raises(ValueError, match="between 0 and 100"):
        script.random_retained_mask(np.ones((3, 4), dtype=bool), percent)
    with pytest.raises(SystemExit):
        script.parse_arguments(["unused", "--output", "unused", "--p-remove", str(percent)])


def test_random_observations_and_prior_only_do_not_leak_missing_truth():
    rng = np.random.default_rng(3)
    prior = {"mean": np.full((2, 3, 4), 0.5), "patterns": rng.normal(size=(3, 2, 3, 4)) * 0.02}
    truth = np.full((2, 3, 4), 4.0)
    retained = script.random_retained_mask(np.ones((3, 4), bool), 75, 5)
    result = script.reconstruct_random(prior, truth, retained, 0.1)
    corrupted = truth.copy()
    corrupted[:, ~retained] = np.nan
    other = script.reconstruct_random(prior, corrupted, retained, 0.1)
    np.testing.assert_array_equal(result["reconstruction"], other["reconstruction"])
    np.testing.assert_array_equal(result["uncertainty"], other["uncertainty"])
    no_observations = np.zeros_like(retained)
    prior_only = script.reconstruct_random(prior, np.full_like(truth, np.nan), no_observations, 0.1)
    np.testing.assert_array_equal(prior_only["posterior_log_mean"], prior["mean"])
    np.testing.assert_allclose(prior_only["posterior_log_std"]**2, np.sum(prior["patterns"]**2, axis=0))
    np.testing.assert_allclose(prior_only["reconstruction"], 10**prior["mean"] - 1)
    assert np.isfinite(prior_only["uncertainty"]).all()


@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("percent", [0, 75, 100])
def test_figure6_CLI_saves_masks_units_identity_and_prior_endpoint(tmp_path, monkeypatch, native, percent):
    logs = 0.6 + 0.08 * np.random.default_rng(7).normal(size=(30, 2, 2, 4))
    samples = 10**logs - 1
    mean = samples.mean(axis=0)
    factor = (samples.reshape(30, -1) - mean.ravel()).T / np.sqrt(29)
    source = tmp_path / "ensemble_products.npz"
    np.savez_compressed(source, conductance_mean=mean, conductance_anomaly_factor=factor)
    original = source.read_bytes()
    valid = np.ones((2, 4), bool)
    valid[0, 0] = False
    identity = {"split": "test", "frame_index": 210, "orbit": 267, "time": "2000-08-31 22:07:14"}
    def load_frame(directory, split, index, include_uncertainty=False):
        result = (samples[0].copy(), valid.copy(), identity.copy())
        return (*result, np.full_like(samples[0], 0.4)) if include_uncertainty else result
    monkeypatch.setattr(script, "load_conductance_frame", load_frame)
    monkeypatch.setattr(script, "validate_conductance_propagation", lambda directory: native)
    make_figure = script.make_figure
    def check_panels(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        assert len(figure.axes) == 8 and all(not axis.axison for axis in figure.axes)
        assert figure.axes[0].texts[0].get_text() == "Hall held-out"
        assert figure.axes[4].texts[0].get_text() == "Pedersen held-out"
        if percent == 100:
            assert figure.axes[2].texts[0].get_text() == "Prior only"
            assert figure.axes[3].texts[0].get_text() == "Prior SD (S)"
        return figure
    monkeypatch.setattr(script, "make_figure", check_panels)
    output = tmp_path / "figure6"
    arguments = script.parse_arguments([str(tmp_path), "--output", str(output), "--p-remove", str(percent), "--frame-index", "210", "--modes", "3", "--seed", "4", "--dpi", "20"])
    report = script.run(arguments)
    assert report["frame_index"] == 210 and report["seed"] == 4
    assert report["units"] == "siemens" and report["prior_space"] == "pixels"
    assert report["observation_error_source"] == ("native_dH_dP" if native else "fixed_log_std")
    assert report["prior_only"] == (percent == 100)
    assert report["removed_spatial_pixels"] == int(np.rint(valid.sum() * percent / 100))
    assert ("held-out native conductance_cs" in report["reference_kind"]) == native
    if percent == 0:
        assert report["missing_rmse_siemens"] == [None, None]
    assert (output / "figure_6.png").exists()
    assert json.loads((output / "figure_6_reconstruction.json").read_text())["orbit"] == 267
    with np.load(output / "figure_6_reconstruction.npz") as data:
        np.testing.assert_array_equal(data["valid"], valid)
        assert data["retained"].ndim == 2  # same spatial mask for both fields
        assert np.isfinite(data["reconstruction"]).all()
    assert source.read_bytes() == original and not script.plt.get_fignums()
    with pytest.raises(FileExistsError, match="preserve"):
        script.run(arguments)
