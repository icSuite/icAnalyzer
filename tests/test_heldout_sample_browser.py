"""Browsing preserves saved frame IDs, masks, source data and physical units."""

import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.physics import conductance_from_corrected_wic, PROPAGATION_ORDER, FROZEN_WIC_RESPONSE_AT_2_KEV
from scripts.IMAGE.vae_pipeline import plot_heldout_samples as script


@pytest.fixture
def saved_split(tmp_path, monkeypatch):
    class Dataset(SimpleNamespace):
        def __len__(self):
            return len(self.values)
    dataset = Dataset(values=np.arange(64).reshape(4, 2, 2, 4).astype(float), common_valid=np.ones((4, 1, 2, 4), dtype=bool), frame_orbit=np.array([10, 20, 30, 40]), frame_time=np.array(["2000-01-01", "2000-01-02", "2000-01-03", "2000-01-04"]))
    dataset.common_valid[1, 0, 0, 0] = False
    torch.save({"transform": {}}, tmp_path / "wic_si12_vae.pt")
    (tmp_path / "summary.json").write_text(json.dumps({"propagation_order": PROPAGATION_ORDER, "wic_response_counts_per_mw_m2": FROZEN_WIC_RESPONSE_AT_2_KEV}))
    calls = []
    def load(artifact, transform, split):
        calls.append(split)
        return dataset, split
    monkeypatch.setattr(script.ChannelTransform, "from_dict", lambda settings: None)
    monkeypatch.setattr(script, "_load_heldout_dataset", load)
    return tmp_path, dataset, calls


@pytest.mark.parametrize("fields", ["conductance", "fuv"])
def test_browser_loads_once_preserves_full_split_ids_and_masks(saved_split, monkeypatch, fields):
    directory, dataset, calls = saved_split
    original = dataset.values.copy()
    artifact = (directory / "wic_si12_vae.pt").read_bytes()
    plotted = []
    make_figure = script.make_figure
    def capture(values, valid, names, units, identity, maxima):
        plotted.append(values.copy())
        figure, limits = make_figure(values, valid, names, units, identity, maxima)
        assert len(figure.axes) == 4  # two maps and two colourbars
        assert not figure.axes[0].axison and not figure.axes[1].axison
        if "frame 1 " in identity:
            assert np.ma.getmaskarray(figure.axes[0].images[0].get_array())[0, 0]
        assert units == ("S" if fields == "conductance" else "counts")
        return figure, limits
    monkeypatch.setattr(script, "make_figure", capture)
    output = directory / "browser"
    arguments = script.parse_arguments([str(directory), "--fields", fields, "--output", str(output), "--start", "1", "--stop", "3", "--maxima", "10", "20", "--dpi", "30"])
    report = script.run(arguments)
    assert report == {"split": "test", "count": 2, "total_frames": 4, "fields": fields}
    assert calls == ["test"]
    with (output / "frames.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [int(row["frame_index"]) for row in rows] == [1, 2]
    assert [int(row["orbit"]) for row in rows] == [20, 30]
    assert int(rows[0]["valid_cells"]) == 7
    assert (output / "test_frame_000001.png").exists()
    assert not (output / "test_frame_000000.png").exists()
    assert 'loading="lazy"' in (output / "index.html").read_text()
    expected = dataset.values[1]
    if fields == "conductance":
        converted = conductance_from_corrected_wic(expected[0], include_intermediates=False)
        expected = np.stack((converted["H"], converted["P"]))
    np.testing.assert_array_equal(plotted[0], expected)
    np.testing.assert_array_equal(dataset.values, original)
    assert (directory / "wic_si12_vae.pt").read_bytes() == artifact
    assert not list(script.plt.get_fignums())


def test_default_exports_all_validation_frames(saved_split):
    directory, _, calls = saved_split
    output = directory / "all"
    report = script.run(script.parse_arguments([str(directory), "--output", str(output), "--split", "validation", "--dpi", "20"]))
    assert report["count"] == 4 and report["fields"] == "conductance"
    assert calls == ["validation"]
    assert len(list(output.glob("validation_frame_*.png"))) == 4
    assert "Independent per-frame colour maxima" in (output / "index.html").read_text()


@pytest.mark.parametrize("options", [["--start", "-1"], ["--start", "2", "--stop", "2"], ["--stop", "9"], ["--maxima", "nan", "10"], ["--dpi", "0"]])
def test_invalid_settings_do_not_create_output(saved_split, options):
    directory, _, _ = saved_split
    output = directory / "invalid"
    with pytest.raises(ValueError):
        script.run(script.parse_arguments([str(directory), "--output", str(output), *options]))
    assert not output.exists()


def test_existing_output_is_preserved(saved_split):
    directory, _, calls = saved_split
    output = directory / "existing"
    output.mkdir()
    (output / "keep.txt").write_text("preserve")
    with pytest.raises(FileExistsError, match="preserved"):
        script.run(script.parse_arguments([str(directory), "--output", str(output)]))
    assert (output / "keep.txt").read_text() == "preserve" and not calls
