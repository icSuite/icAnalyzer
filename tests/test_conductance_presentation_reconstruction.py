"""Conductance archive moments, matched frame mapping, masks/units and CLI."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.physics import conductance_from_corrected_wic, PROPAGATION_ORDER, FROZEN_WIC_RESPONSE_AT_2_KEV
from scripts.IMAGE.vae_pipeline import plot_conductance_reconstruction as script


def save_conductance_samples(directory, samples):
    mean = samples.mean(axis=0)
    factor = np.ascontiguousarray((samples.reshape(len(samples), -1) - mean.ravel()).T / np.sqrt(len(samples) - 1))
    np.savez_compressed(directory / "ensemble_products.npz", conductance_mean=mean, conductance_anomaly_factor=factor, fuv_mean=np.zeros_like(mean), fuv_anomaly_factor=np.zeros_like(factor))


@pytest.fixture
def run_data(tmp_path, monkeypatch):
    logs = 1 + np.random.default_rng(5).normal(size=(80, 2, 2, 4)) * 0.15
    save_conductance_samples(tmp_path, 10**logs - 1)
    torch.save({"transform": {}}, tmp_path / "wic_si12_vae.pt")
    summary = {"propagation_order": PROPAGATION_ORDER, "wic_response_counts_per_mw_m2": FROZEN_WIC_RESPONSE_AT_2_KEV}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    values = np.ones((2, 2, 2, 4))
    values[1, 0] = np.arange(8).reshape(2, 4) * 100 + 20
    dataset = Dataset(values=values, common_valid=np.ones((2, 1, 2, 4), dtype=bool), frame_orbit=np.array([150, 267]), frame_time=np.array(["2000-01-01", "2000-08-31"]))
    dataset.common_valid[1, 0, 0, 0] = False
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda settings: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    return tmp_path, dataset, logs


def test_log_prior_reads_conductance_not_FUV_and_matches_direct_covariance(run_data):
    directory, _, logs = run_data
    original = (directory / "ensemble_products.npz").read_bytes()
    prior = script.log_prior(directory, n_modes=5, field_prefix="conductance")
    np.testing.assert_allclose(prior["mean"], logs.mean(axis=0), atol=1e-12)
    expected = np.linalg.eigvalsh(np.cov(logs.reshape(80, -1), rowvar=False))[::-1][:5]
    np.testing.assert_allclose(prior["eigenvalues"], expected, rtol=1e-12)
    assert (directory / "ensemble_products.npz").read_bytes() == original


def test_frame_id_preserves_FUV_numbering_and_applies_matching_fixed_energy_conversion(run_data):
    directory, dataset, _ = run_data
    truth, valid, identity = script.load_conductance_frame(directory, "test", 1)
    expected = conductance_from_corrected_wic(dataset.values[1, 0], include_intermediates=False)
    np.testing.assert_array_equal(truth, np.stack((expected["H"], expected["P"])))
    np.testing.assert_array_equal(valid, dataset.common_valid[1, 0])
    assert identity == {"split": "test", "frame_index": 1, "orbit": 267, "time": "2000-08-31"}
    assert np.allclose(truth[0] / truth[1], truth[0, 0, 0] / truth[1, 0, 0])
    with pytest.raises(ValueError, match="frame index"):
        script.load_conductance_frame(directory, "test", 2)


@pytest.mark.parametrize("case", ["half", "fields"])
def test_conductance_CLI_uses_one_prior_and_preserves_masks_units_and_provenance(run_data, monkeypatch, case):
    directory, _, _ = run_data
    calls = []
    original_prior = script.log_prior
    def counted_prior(*args, **kwargs):
        calls.append(kwargs["field_prefix"])
        return original_prior(*args, **kwargs)
    monkeypatch.setattr(script, "log_prior", counted_prior)
    original_reconstruct = script.reconstruct
    def checked_reconstruction(mean, factor, truth, retained, std):
        result = original_reconstruct(mean, factor, truth, retained, std)
        corrupted = truth.copy()
        corrupted[~np.broadcast_to(retained, truth.shape)] = np.nan
        repeated = original_reconstruct(mean, factor, corrupted, retained, std)
        np.testing.assert_array_equal(result["reconstruction"], repeated["reconstruction"])
        np.testing.assert_array_equal(result["uncertainty"], repeated["uncertainty"])
        return result
    monkeypatch.setattr(script, "reconstruct", checked_reconstruction)
    original_figure = script.make_figure
    def checked_figure(*args, **kwargs):
        figure = original_figure(*args, **kwargs)
        assert len(figure.axes) == 8 and all(not axis.axison for axis in figure.axes)
        assert figure.axes[0].texts[0].get_text() == "Hall held-out"
        assert figure.axes[4].texts[0].get_text() == "Pedersen held-out"
        assert figure.axes[3].texts[0].get_text() == "Posterior SD (S)"
        return figure
    monkeypatch.setattr(script, "make_figure", checked_figure)
    source = (directory / "ensemble_products.npz").read_bytes()
    output = directory / "conductance_figures"
    arguments = script.parse_arguments([str(directory), "--output", str(output), "--frame-index", "1", "--modes", "5", "--dpi", "30"])
    script.run(arguments, case=case)
    assert calls == ["conductance"]
    names = ["figure_4"] if case == "half" else ["figure_5a", "figure_5b"]
    for name in names:
        assert (output / f"{name}.png").is_file()
        report = json.loads((output / f"{name}_reconstruction.json").read_text())
        assert report["frame_index"] == 1 and report["orbit"] == 267
        assert report["units"] == "siemens" and report["electron_energy_keV"] == 2
        assert "NOT conductance_cs" in report["reference_kind"]
        with np.load(output / f"{name}_reconstruction.npz", allow_pickle=False) as archive:
            assert archive["truth"].shape == (2, 2, 4)
            assert archive["field_names"].tolist() == ["Hall", "Pedersen"]
            assert not archive["valid"][0, 0]
            if case == "fields":
                removed = 1 if name.endswith("a") else 0
                assert not archive["retained"][removed].any()
                assert report["removed_field"] == ("Hall", "Pedersen")[removed]
                assert report["missing_rmse_siemens"][1 - removed] is None
            else:
                assert not archive["retained"][:, 2:].any()
    assert (directory / "ensemble_products.npz").read_bytes() == source


@pytest.mark.parametrize("key,value", [("propagation_order", "different"), ("wic_response_counts_per_mw_m2", 999)])
def test_incompatible_propagation_is_rejected(run_data, key, value):
    directory, _, _ = run_data
    path = directory / "summary.json"
    summary = json.loads(path.read_text())
    summary[key] = value
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="fixed-2-keV propagation"):
        script.load_conductance_frame(directory, "test", 1)


@pytest.mark.parametrize("negative,accepted", [(-1e-7, True), (-0.01, False)])
def test_only_roundoff_negative_conductance_is_clipped(tmp_path, negative, accepted):
    samples = np.ones((20, 2, 2, 4)) * 3
    samples[0, 0, 0, 0] = negative
    save_conductance_samples(tmp_path, samples)
    if accepted:
        prior = script.log_prior(tmp_path, n_modes=1, field_prefix="conductance")
        assert prior["negative_draw_cells_clipped"] == 1
        assert prior["minimum_recovered_draw"] < 0
    else:
        with pytest.raises(ValueError, match="substantive negative"):
            script.log_prior(tmp_path, n_modes=1, field_prefix="conductance")
