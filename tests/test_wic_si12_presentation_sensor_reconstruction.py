"""Whole-sensor masking, no hidden-channel leakage, and shared-prior CLI."""

import json
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch

from scripts.IMAGE.vae_pipeline import plot_wic_si12_sensor_reconstruction as script


@pytest.mark.parametrize("removed_channel", [0, 1])
def test_whole_sensor_mask_and_no_hidden_channel_leakage(removed_channel):
    valid = np.ones((2, 3), dtype=bool)
    valid[0, 0] = False
    retained = script.retained_sensor(valid, removed_channel)
    assert not retained[removed_channel].any()
    np.testing.assert_array_equal(retained[1 - removed_channel], valid)
    mean = np.ones((2, 2, 3))
    factor = np.random.default_rng(3).normal(size=(12, 4)) * 0.1
    truth = np.arange(12).reshape(2, 2, 3).astype(float)
    result = script.reconstruct(mean, factor, truth, retained)
    truth[removed_channel] = np.nan
    hidden = script.reconstruct(mean, factor, truth, retained)
    for key in ("reconstruction", "posterior_log_mean", "posterior_log_std", "uncertainty"):
        np.testing.assert_array_equal(hidden[key], result[key])
    labels = tuple("Sensor removed" if channel == removed_channel else "Observed input" for channel in range(2))
    figure = script.make_figure(np.zeros_like(truth), valid, retained, result["reconstruction"], result["uncertainty"], input_labels=labels)
    assert np.ma.getmaskarray(figure.axes[removed_channel * 4 + 1].images[0].get_array()).all()
    assert not np.ma.getmaskarray(figure.axes[(1 - removed_channel) * 4 + 1].images[0].get_array())[valid].any()
    assert figure.axes[removed_channel * 4 + 1].texts[0].get_text() == "Sensor removed"
    plt.close(figure)


def test_cli_generates_both_cases_from_one_prior_and_identical_truth(tmp_path, monkeypatch):
    artifact = {"transform": {}}
    torch.save(artifact, tmp_path / "wic_si12_vae.pt")
    class FakeDataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    truth = np.arange(12).reshape(2, 2, 3).astype(float)
    dataset = FakeDataset(values=truth[None], common_valid=np.ones((1, 1, 2, 3), dtype=bool), frame_orbit=np.array([150]), frame_time=np.array(["2000-01-01"]))
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda settings: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", lambda artifact, transform, split: (dataset, split))
    calls = []
    def prior(run_directory, n_modes):
        calls.append(n_modes)
        return {"mean": np.ones((2, 2, 3)), "patterns": np.random.default_rng(3).normal(size=(4, 2, 2, 3)) * 0.1, "eigenvalues": np.ones(4), "explained_variance_fraction": np.ones(4) / 4, "sample_count": 50, "negative_draw_cells_clipped": 0, "minimum_recovered_draw": 0}
    monkeypatch.setattr(script, "log_prior", prior)
    script.run(script.parse_arguments([str(tmp_path), "--output", str(tmp_path), "--modes", "4", "--dpi", "30"]))
    assert calls == [4]
    for letter, removed_channel in (("a", 1), ("b", 0)):
        name = f"figure_5{letter}"
        assert (tmp_path / f"{name}.png").is_file()
        report = json.loads((tmp_path / f"{name}_reconstruction.json").read_text())
        assert report["removed_sensor"] == ("WIC", "SI12")[removed_channel]
        assert report["observed_cells_per_sensor"][removed_channel] == 0
        with np.load(tmp_path / f"{name}_reconstruction.npz") as archive:
            np.testing.assert_array_equal(archive["truth"], truth)
            assert not archive["retained"][removed_channel].any()
