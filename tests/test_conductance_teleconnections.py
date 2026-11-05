"""Conductance figure3 channel selection, support, provenance and FUV parity."""

import json

import matplotlib.pyplot as plt
import numpy as np
import pytest

from icanalyzer.wic_si12.physics import PROPAGATION_ORDER, FROZEN_WIC_RESPONSE_AT_2_KEV
from scripts.IMAGE.vae_pipeline import plot_conductance_teleconnections as script
from scripts.IMAGE.vae_pipeline.plot_wic_si12_teleconnections import calculate_teleconnections, make_figure, teleconnection_maps


@pytest.mark.parametrize("native", [False, True])
def test_conductance_CLI_uses_full_conductance_factor_and_preserves_sources(tmp_path, native):
    factor = np.random.default_rng(4).normal(size=(12, 35)).astype(np.float32)
    support = np.ones((2, 3), bool)
    support[0, 0] = False
    factor[[0, 6]] = 0
    arrays = {"conductance_mean": np.ones((2, 2, 3)), "conductance_anomaly_factor": factor, "fuv_mean": np.ones((2, 2, 3)), "fuv_anomaly_factor": np.ones_like(factor)}
    if native:
        arrays["conductance_supported"] = support
    np.savez_compressed(tmp_path / "ensemble_products.npz", **arrays)
    summary = {"input_product_type": "conductance_cs", "field_names": ["Hall", "Pedersen"], "field_units": "S", "propagation_reapplied": False} if native else {"propagation_order": PROPAGATION_ORDER, "wic_response_counts_per_mw_m2": FROZEN_WIC_RESPONSE_AT_2_KEV}
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    source = (tmp_path / "ensemble_products.npz").read_bytes()
    output = tmp_path / "plots"
    arguments = script.parse_arguments([str(tmp_path), "--grid-id", "4", "--output", str(output), "--dpi", "20", "--no-colorbar"])
    report = script.run(arguments)
    reference = teleconnection_maps(factor, (2, 2, 3), 4)
    with np.load(output / "figure_3_teleconnections.npz") as result:
        np.testing.assert_allclose(result["correlations"], reference["correlations"], equal_nan=True)
        np.testing.assert_allclose(result["joint_explained_variance"], reference["joint_explained_variance"], equal_nan=True)
        assert result["field_names"].tolist() == ["Hall", "Pedersen"]
    assert report["sample_count"] == 35 and report["anchor_row_column"] == [1, 1]
    assert ("native" in report["reference_kind"]) == native
    assert not report["measurement_uncertainty_used"]
    assert all((output / f"figure_3{letter}.png").exists() for letter in "abc")
    assert not plt.get_fignums() and (tmp_path / "ensemble_products.npz").read_bytes() == source
    with pytest.raises(FileExistsError, match="overwrite"):
        script.run(arguments)
    sentinel = output / "unrelated.txt"
    sentinel.write_text("preserve unrelated files")
    changed = script.parse_arguments([str(tmp_path), "--grid-id", "5", "--output", str(output), "--dpi", "20", "--overwrite"])
    updated = script.run(changed)
    assert updated["grid_id"] == 5
    assert json.loads((output / "figure_3_teleconnections.json").read_text())["grid_id"] == 5
    with np.load(output / "figure_3_teleconnections.npz") as replaced:
        assert int(replaced["grid_id"]) == 5
    assert sentinel.read_text() == "preserve unrelated files"
    assert (tmp_path / "ensemble_products.npz").read_bytes() == source


def test_unsupported_anchor_rejected_and_targets_masked(tmp_path):
    factor = np.random.default_rng(2).normal(size=(12, 10))
    support = np.ones((2, 3), bool)
    support[0, 0] = False
    np.savez_compressed(tmp_path / "ensemble_products.npz", conductance_mean=np.ones((2, 2, 3)), conductance_anomaly_factor=factor, conductance_supported=support)
    with pytest.raises(ValueError, match="supported training domain"):
        calculate_teleconnections(tmp_path, 0, field_prefix="conductance")
    result = calculate_teleconnections(tmp_path, 4, field_prefix="conductance")
    assert np.isnan(result["correlations"][:, :, 0, 0]).all()
    assert np.isnan(result["joint_explained_variance"][:, 0, 0]).all()


@pytest.mark.parametrize("joint", [False, True])
def test_Hall_Pedersen_panels_keep_signed_or_joint_scales(joint):
    figure = make_figure(np.zeros((2, 2, 3)), (1, 1), joint=joint, colorbar=False, field_names=("Hall", "Pedersen"))
    for channel, axis in enumerate(figure.axes):
        assert axis.texts[0].get_text() == ("Hall", "Pedersen")[channel]
        assert not axis.axison
        assert axis.images[0].get_clim() == ((0 if joint else -1), 1)
        assert axis.lines[0].get_xdata()[0] == 1
    plt.close(figure)
