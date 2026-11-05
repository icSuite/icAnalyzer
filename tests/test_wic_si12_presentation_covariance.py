"""Check exact joint PCA, one-sigma amplitudes and the presentation layout."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.IMAGE.vae_pipeline.plot_wic_si12_covariance import calculate_pca, decompose_factor, make_figure, parse_arguments, run


def saved_ensemble(directory):
    samples = np.random.default_rng(8).normal(size=(18, 2, 2, 3))
    samples[:, 1] = 0.3 * samples[:, 0] + 0.1 * samples[:, 1]
    samples += 5
    mean = samples.mean(axis=0)
    factor = np.ascontiguousarray((samples.reshape(18, -1) - mean.ravel()).T / np.sqrt(17))
    np.savez_compressed(directory / "ensemble_products.npz", fuv_mean=mean, fuv_anomaly_factor=factor)
    return samples, factor


def test_exact_joint_modes_match_full_covariance_across_batches(tmp_path):
    samples, factor = saved_ensemble(tmp_path)
    result = calculate_pca(tmp_path, batch_size=3)
    covariance = np.cov(samples.reshape(18, -1), rowvar=False)
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1][:4]
    np.testing.assert_allclose(result["eigenvalues"], eigenvalues, rtol=1e-12)
    patterns = result["patterns"].reshape(4, -1)
    np.testing.assert_allclose(np.sum(patterns**2, axis=1), eigenvalues, rtol=1e-12)
    np.testing.assert_allclose(covariance @ patterns.T, patterns.T * eigenvalues, atol=1e-12)
    np.testing.assert_allclose(result["explained_variance_fraction"], eigenvalues / np.trace(covariance))
    assert np.any(np.abs(result["patterns"][:, 1]) > 0)
    np.testing.assert_allclose(result["patterns"], decompose_factor(factor, (2, 2, 3), batch_size=18)["patterns"], atol=1e-12)
    assert not list(tmp_path.glob("factor.npy"))


def test_mean_and_signed_patterns_in_correct_panels(tmp_path):
    saved_ensemble(tmp_path)
    result = calculate_pca(tmp_path)
    figure = make_figure(result["mean"], result["patterns"], result["explained_variance_fraction"])
    assert len(figure.axes) == 10
    for row in range(2):
        for column in range(5):
            axis = figure.axes[row * 5 + column]
            assert not axis.axison and axis.get_title() == axis.get_xlabel() == axis.get_ylabel() == ""
            expected_label = f"{('WIC', 'SI12')[row]} ensemble mean" if column == 0 else f"PC {column} : {100 * result['explained_variance_fraction'][column - 1]:.2f}%"
            assert len(axis.texts) == 1 and axis.texts[0].get_text() == expected_label
            assert axis.texts[0].get_position() == (0.5, 1.04 if row == 0 else -0.04)
            assert axis.texts[0].get_verticalalignment() == ("bottom" if row == 0 else "top")
            assert not axis.texts[0].get_clip_on()
            values = result["mean"][row] if column == 0 else result["patterns"][column - 1, row]
            np.testing.assert_array_equal(axis.images[0].get_array(), values)
            if column > 0:
                low, high = axis.images[0].get_clim()
                assert low == -high
                assert axis.images[0].get_clim() == figure.axes[row * 5 + 1].images[0].get_clim()
    plt.close(figure)


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size_is_rejected(batch_size):
    with pytest.raises(ValueError, match="batch size"):
        decompose_factor(np.zeros((6, 8)), (2, 1, 3), batch_size)


def test_no_variance_is_rejected():
    with pytest.raises(ValueError, match="no variance"):
        decompose_factor(np.zeros((6, 8)), (2, 1, 3))


def test_cli_saves_figure_and_compact_pca_data(tmp_path):
    saved_ensemble(tmp_path)
    output = tmp_path / "figure_2.png"
    run(parse_arguments([str(tmp_path), "--output", str(output), "--dpi", "30"]))
    assert output.is_file()
    with np.load(tmp_path / "figure_2_pca.npz") as archive:
        assert archive["patterns"].shape == (4, 2, 2, 3)
        assert archive["sample_count"] == 18
