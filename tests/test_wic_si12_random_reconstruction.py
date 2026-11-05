"""Figure 6 uses WIC/SI12 counts, with pixel or legacy-style coefficient prior."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scripts.IMAGE.vae_pipeline import plot_wic_si12_random_reconstruction as script


@pytest.fixture
def fuv_run(tmp_path, monkeypatch):
    logs = 0.6 + 0.08 * np.random.default_rng(7).normal(size=(30, 2, 2, 4))
    samples = 10**logs - 1
    mean = samples.mean(axis=0)
    factor = (samples.reshape(30, -1) - mean.ravel()).T / np.sqrt(29)
    np.savez_compressed(tmp_path / "ensemble_products.npz", fuv_mean=mean, fuv_anomaly_factor=factor, conductance_mean=mean * 100, conductance_anomaly_factor=factor * 100)
    torch.save({"transform": {}}, tmp_path / "wic_si12_vae.pt")
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    dataset = Dataset(values=samples[:2], common_valid=np.ones((2, 1, 2, 4), bool), frame_orbit=np.array([150, 267]), frame_time=np.array(["2000-01-01", "2000-08-31"]))
    dataset.common_valid[1, 0, 0, 0] = False
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda settings: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    return tmp_path, dataset, logs


@pytest.mark.parametrize("percent", [0, 75, 100])
def test_FUV_endpoints_masks_identity_units_and_source_are_preserved(fuv_run, monkeypatch, percent):
    directory, dataset, logs = fuv_run
    source = (directory / "ensemble_products.npz").read_bytes()
    original_values = dataset.values.copy()
    make_figure = script.make_figure
    def check_labels(*args, **kwargs):
        figure = make_figure(*args, **kwargs)
        assert figure.axes[0].texts[0].get_text() == "WIC held-out"
        assert figure.axes[4].texts[0].get_text() == "SI12 held-out"
        assert "counts" in figure.axes[3].texts[0].get_text()
        return figure
    monkeypatch.setattr(script, "make_figure", check_labels)
    output = directory / "figure6"
    args = script.parse_arguments([str(directory), "--output", str(output), "--p-remove", str(percent), "--frame-index", "1", "--modes", "3", "--dpi", "20"])
    report = script.run(args)
    assert report["field_names"] == ["WIC", "SI12"] and report["units"] == "counts"
    assert report["frame_index"] == 1 and report["orbit"] == 267
    assert report["prior_only"] == (percent == 100)
    assert report["prior_space"] == "pixels"
    assert (output / "figure_6.png").exists()
    assert json.loads((output / "figure_6_reconstruction.json").read_text())["reference_kind"].endswith("no conductance conversion")
    with np.load(output / "figure_6_reconstruction.npz") as data:
        assert data["field_names"].tolist() == ["WIC", "SI12"]
        np.testing.assert_array_equal(data["truth"], dataset.values[1])
        np.testing.assert_allclose(data["prior_log_mean"], logs.mean(axis=0), atol=1e-12)
        assert not data["retained"][0, 0]
    np.testing.assert_array_equal(dataset.values, original_values)
    assert (directory / "ensemble_products.npz").read_bytes() == source
    (output / "figure_6.png").write_bytes(b"old plot")
    (output / "keep.txt").write_text("unrelated output")
    args.seed = 1
    rerun = script.run(args)
    assert rerun["seed"] == 1
    assert (output / "figure_6.png").read_bytes().startswith(b"\x89PNG")
    assert json.loads((output / "figure_6_reconstruction.json").read_text())["seed"] == 1
    assert (output / "keep.txt").read_text() == "unrelated output"


def test_coefficient_option_uses_existing_prior_without_pixel_PCA(fuv_run, monkeypatch):
    directory, _, _ = fuv_run
    prior = script.log_prior(directory, 3)
    prior["coefficient_mean"] = np.ones((2, 2, 2))
    calls = []
    def coefficients(*args):
        calls.append(args)
        return prior
    monkeypatch.setattr(script, "coefficient_prior", coefficients)
    monkeypatch.setattr(script, "log_prior", lambda *args, **kwargs: pytest.fail("pixel prior must not be used"))
    output = directory / "coefficients"
    args = script.parse_arguments([str(directory), "--output", str(output), "--p-remove", "75", "--frame-index", "1", "--modes", "3", "--prior-space", "coefficients", "--dpi", "20"])
    report = script.run(args)
    assert calls == [(directory, 3, 64, "cpu")]
    assert report["prior_space"] == "coefficients" and "spline coefficients" in report["variance_space"]
    with np.load(output / "figure_6_reconstruction.npz") as data:
        np.testing.assert_array_equal(data["coefficient_mean"], prior["coefficient_mean"])


def test_out_of_range_frame_rejected_without_output(fuv_run):
    directory, _, _ = fuv_run
    output = directory / "bad_frame"
    args = script.parse_arguments([str(directory), "--output", str(output), "--p-remove", "50", "--frame-index", "2"])
    with pytest.raises(ValueError, match="frame index"):
        script.run(args)
    assert not output.exists()


def test_prediction_and_SD_show_full_grid_while_reference_gaps_stay_blank():
    truth = np.arange(16, dtype=float).reshape(2, 2, 4)
    valid = np.ones((2, 4), dtype=bool)
    valid[0, 0] = False
    retained = valid.copy()
    retained[:, 2:] = False
    reconstruction = truth + 10
    reconstruction[:, 0, 0] = 5000
    uncertainty = np.ones_like(truth)
    uncertainty[:, 0, 0] = 200
    figure = script.make_figure(truth, valid, retained, reconstruction, uncertainty)
    for channel in range(2):
        axes = figure.axes[4 * channel:4 * channel + 4]
        assert np.ma.getmaskarray(axes[0].images[0].get_array())[0, 0]
        assert np.ma.getmaskarray(axes[1].images[0].get_array())[:, 2:].all()
        assert not np.ma.getmaskarray(axes[2].images[0].get_array()).any()
        assert not np.ma.getmaskarray(axes[3].images[0].get_array()).any()
        np.testing.assert_array_equal(axes[2].images[0].get_array(), reconstruction[channel])
        np.testing.assert_array_equal(axes[3].images[0].get_array(), uncertainty[channel])
        assert axes[3].images[0].get_clim() == (0, 200)
    script.plt.close(figure)
