"""Single-anchor Pearson correlation and scale-invariant joint linear R-squared."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from scripts.IMAGE.vae_pipeline.plot_wic_si12_teleconnections import make_figure, parse_arguments, run, teleconnection_maps


def test_correlations_and_joint_variance_match_covariance_and_regression():
    values = np.random.default_rng(12).normal(size=(6, 100))
    values[1] = 0.7 * values[0] + 0.4 * values[3] + 0.2 * values[1]
    values -= values.mean(axis=1, keepdims=True)
    factor = values / np.sqrt(99)
    result = teleconnection_maps(factor, (2, 1, 3), 0, batch_size=2)
    covariance = factor @ factor.T
    reference = np.corrcoef(values)[:, [0, 3]].T.reshape(2, 2, 1, 3)
    np.testing.assert_allclose(result["correlations"], reference, atol=1e-12)
    anchor_covariance = covariance[np.ix_([0, 3], [0, 3])]
    cross = covariance[:, [0, 3]]
    expected = np.einsum("ij,jk,ik->i", cross, np.linalg.inv(anchor_covariance), cross) / np.diag(covariance)
    np.testing.assert_allclose(result["joint_explained_variance"].ravel(), expected, atol=1e-12)
    predicted = np.linalg.lstsq(values[[0, 3]].T, values.T, rcond=None)[0]
    explained = np.sum((values[[0, 3]].T @ predicted)**2, axis=0) / np.sum(values**2, axis=1)
    np.testing.assert_allclose(result["joint_explained_variance"].ravel(), explained, atol=1e-12)
    assert np.all(result["joint_explained_variance"].ravel() >= np.max(reference.reshape(2, -1)**2, axis=0) - 1e-12)
    rescaled = factor.copy()
    rescaled[3:] *= 1e-5
    scaled = teleconnection_maps(rescaled, (2, 1, 3), 0)
    np.testing.assert_allclose(scaled["joint_explained_variance"], result["joint_explained_variance"], atol=1e-12)


def test_redundant_anchors_and_constant_targets():
    factor = np.random.default_rng(1).normal(size=(6, 30))
    factor[3] = -4 * factor[0]
    factor[2] = 0
    result = teleconnection_maps(factor, (2, 1, 3), 0)
    np.testing.assert_allclose(result["joint_explained_variance"].ravel(), result["correlations"][0].ravel()**2, atol=1e-12)
    assert np.isnan(result["joint_explained_variance"].ravel()[2])
    factor[0] = 0
    with pytest.raises(ValueError, match="zero/nonfinite"):
        teleconnection_maps(factor, (2, 1, 3), 0)


@pytest.mark.parametrize("grid_id", [-1, 3])
def test_invalid_grid_id(grid_id):
    with pytest.raises(ValueError, match="grid ID"):
        teleconnection_maps(np.ones((6, 10)), (2, 1, 3), grid_id)


def test_layout_limits_marker_and_labels():
    for joint in (False, True):
        maps = np.arange(12).reshape(2, 2, 3) / 12
        figure = make_figure(maps, (1, 2), joint, colorbar=False)
        assert len(figure.axes) == 2
        for channel, axis in enumerate(figure.axes):
            assert not axis.axison
            np.testing.assert_array_equal(axis.images[0].get_array(), maps[channel])
            assert axis.images[0].get_clim() == ((0 if joint else -1), 1)
            assert axis.texts[0].get_text() == ("WIC", "SI12")[channel]
            assert axis.lines[0].get_xdata()[0] == 2 and axis.lines[0].get_ydata()[0] == 1
        plt.close(figure)


def test_cli_writes_named_figures_and_data(tmp_path):
    factor = np.random.default_rng(0).normal(size=(12, 20)).astype(np.float32)
    np.savez_compressed(tmp_path / "ensemble_products.npz", fuv_mean=np.ones((2, 2, 3)), fuv_anomaly_factor=factor)
    output = tmp_path / "figures"
    run(parse_arguments([str(tmp_path), "--grid-id", "4", "--output", str(output), "--dpi", "30"]))
    assert all((output / f"figure_3{letter}.png").is_file() for letter in "abc")
    with np.load(output / "figure_3_teleconnections.npz") as archive:
        np.testing.assert_array_equal(archive["anchor_row_column"], [1, 1])
        assert archive["correlations"].shape == (2, 2, 2, 3)
