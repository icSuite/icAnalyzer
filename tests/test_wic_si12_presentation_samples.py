"""Selected saved draws and the requested clean two-by-five presentation layout."""

import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.IMAGE.vae_pipeline.plot_wic_si12_samples import load_selected_samples, make_figure, parse_arguments, run


def saved_ensemble(directory, fortran_order=False):
    samples = np.arange(12 * 2 * 3 * 4, dtype=float).reshape(12, 2, 3, 4)
    mean = samples.mean(axis=0)
    factor = (samples.reshape(12, -1) - mean.ravel()).T / np.sqrt(11)
    factor = np.asfortranarray(factor) if fortran_order else np.ascontiguousarray(factor)
    np.savez_compressed(directory / "ensemble_products.npz", fuv_mean=mean, fuv_anomaly_factor=factor)
    return samples


@pytest.mark.parametrize("fortran_order", [False, True])
def test_selected_draws_preserve_indices_order_and_duplicates(tmp_path, fortran_order):
    original = saved_ensemble(tmp_path, fortran_order)
    indices = [11, 0, 5, 5, 2]
    recovered, count = load_selected_samples(tmp_path, indices)
    np.testing.assert_allclose(recovered, original[indices], atol=1e-12)
    assert count == 12


@pytest.mark.parametrize("indices", [[0, 1], [-1, 0, 1, 2, 3], [0, 1, 2, 3, 12]])
def test_invalid_indices_are_rejected(tmp_path, indices):
    saved_ensemble(tmp_path)
    with pytest.raises(ValueError, match="five|between"):
        load_selected_samples(tmp_path, indices)


def test_layout_channels_axes_and_shared_scales(tmp_path):
    samples = saved_ensemble(tmp_path)[:5]
    figure = make_figure(samples, wic_max=300, si12_max=400)
    assert len(figure.axes) == 10
    for row in range(2):
        for column in range(5):
            axis = figure.axes[row * 5 + column]
            assert not axis.axison
            assert axis.get_title() == axis.get_xlabel() == axis.get_ylabel() == ""
            np.testing.assert_array_equal(axis.images[0].get_array(), samples[column, row])
            assert axis.images[0].get_clim() == (0, (300, 400)[row])
    plt.close(figure)


def test_cli_saves_only_the_requested_figure(tmp_path):
    saved_ensemble(tmp_path)
    output = tmp_path / "figures" / "figure_1.png"
    arguments = parse_arguments([str(tmp_path), "--indices", "0", "1", "2", "3", "4", "--output", str(output), "--dpi", "30"])
    run(arguments)
    assert output.is_file()


def test_stream_read_size_is_bounded(tmp_path, monkeypatch):
    saved_ensemble(tmp_path)
    original_open = zipfile.ZipFile.open
    sizes = []

    def monitored_open(archive, name, *args, **kwargs):
        stream = original_open(archive, name, *args, **kwargs)
        if name == "fuv_anomaly_factor.npy":
            original_read = stream.read
            def monitored_read(size=-1):
                sizes.append(size)
                return original_read(size)
            stream.read = monitored_read
        return stream

    monkeypatch.setattr(zipfile.ZipFile, "open", monitored_open)
    load_selected_samples(tmp_path, [0, 1, 2, 3, 4])
    assert sizes and all(0 < size <= 8 * 1024 * 1024 for size in sizes)
