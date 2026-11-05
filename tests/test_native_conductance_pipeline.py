"""Native H/P contract, unchanged VAE, supported covariance and plotting parity."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer import conductance as data
from icanalyzer.wic_si12.data import ChannelTransform
from icanalyzer.wic_si12.ensemble import sample_decoded_ensemble
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.IMAGE.vae_pipeline import run_conductance_pipeline as pipeline
from scripts.IMAGE.vae_pipeline import run_wic_si12_pipeline as fuv_pipeline
from scripts.IMAGE.vae_pipeline import plot_conductance_reconstruction as plotter
from scripts.IMAGE.vae_pipeline import plot_heldout_samples as browser
from scripts.IMAGE.vae_pipeline import visualize_wic_si12_run as visualize


class NativeProduct:
    product_type = "conductance_cs"
    grid_id = "image_apex_130km_46x46_v1"
    schema_version = 2
    software_version = "fixture"
    shape = (3, 46, 46)

    def __init__(self, orbit=85):
        frames, rows, cols = np.indices(self.shape)
        H = (orbit / 100 + frames + rows / 10 + cols / 30).astype(np.float32)
        self.arrays = {"H": H, "P": H * (1.2 + rows / 100), "dH": np.full(self.shape, 0.3), "dP": np.full(self.shape, 0.4), "coverage": np.full(self.shape, 0.7), "uncertainty_coverage": np.full(self.shape, 0.6), "conductance_valid": np.ones(self.shape, dtype=bool), "conductance_uncertainty_valid": np.ones(self.shape, dtype=bool), "method_quality_weight": np.full(self.shape, 0.2)}
        for name, array in self.arrays.items():
            array[:, 0, 0] = False if array.dtype == bool else np.nan
            units = "S" if name in ("H", "P", "dH", "dP") else (None if array.dtype == bool else "1")
            setattr(self, name, SimpleNamespace(dimensions=("time", "dim1", "dim2"), units=units))
        self.time = np.array([f"2000-01-01T00:0{i}:00" for i in range(3)])
        self.mlat = np.indices((46, 46))[0].astype(float)
        self.mlt = np.indices((46, 46))[1].astype(float)
        self.source_indices = {name: np.array([10, 12, 15]) for name in ("wic", "si12", "si13")}
        self.attrs = {"conductance_model": "robinson", "method_quality_weight_method": "WIC auroral-quality field defined as 1 minus fuvpy dgweight", "proton_energy_model": "hardy", "proton_flux_source": "SI12", "nonlinear_ordering": "detector conductance binned afterwards"}
        self.reads = []

    def read(self, name, selection):
        self.reads.append((name, selection))
        return self.arrays[name][selection]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_native_channels_uncertainties_quality_masks_and_source_annotation():
    product = NativeProduct()
    result = data.load_conductance_cs("or_0085.nc", lambda path: product, wic_source_interval=(11, 15))
    np.testing.assert_array_equal(result["values"][:, 0], product.arrays["H"][1:])
    np.testing.assert_array_equal(result["values"][:, 1], product.arrays["P"][1:].astype(np.float32))
    np.testing.assert_allclose(result["standard_deviation"][:, 0, 1, 1], [0.3, 0.3])
    np.testing.assert_allclose(result["quality"][:, :, 1, 1], 0.14)
    assert not result["valid"][:, :, 0, 0].any() and (result["quality"][:, :, 0, 0] == 0).all()
    assert result["source_indices"]["wic"].tolist() == [12, 15]
    assert all(isinstance(selection, slice) and selection.start == 1 for name, selection in product.reads)


@pytest.mark.parametrize("fault", ["type", "grid", "units", "weight_orientation", "weight_range", "sza"])
def test_native_contract_rejects_ambiguous_or_incompatible_inputs(fault):
    product = NativeProduct()
    kwargs = {}
    if fault == "type": product.product_type = "precipitation_cs"
    if fault == "grid": product.grid_id = "wrong-grid"
    if fault == "units": product.H.units = "counts"
    if fault == "weight_orientation": product.attrs["method_quality_weight_method"] = "background fit"
    if fault == "weight_range": product.arrays["method_quality_weight"][:, 1, 1] = 2
    if fault == "sza": kwargs["minimum_sza"] = 105
    with pytest.raises(ValueError):
        data.load_conductance_cs("or_0085.nc", lambda path: product, **kwargs)


def test_native_uncertainty_support_and_negative_clipping_do_not_modify_source():
    product = NativeProduct()
    product.arrays["H"][0, 1, 1] = -3
    product.arrays["dP"][0, 1, 2] = -1
    product.arrays["uncertainty_coverage"][0, 1, 3] = 0
    with pytest.warns(UserWarning, match="negative H/P"):
        result = data.load_conductance_cs("or_0085.nc", lambda path: product)
    assert result["values"][0, 0, 1, 1] == 0
    assert result["metadata"]["negative_H_P_clipped_cells"] == [1, 0]
    assert not result["valid"][0, :, 1, 2:4].any()
    assert product.arrays["H"][0, 1, 1] == -3


def test_unsupported_sampling_is_masked_before_exponentiation():
    class Model(torch.nn.Module):
        image_shape = (2, 3)
        latent_dim = 36
        def decode(self, latent):
            output = torch.ones((len(latent), 2, 2, 3))
            output[:, :, 0, 0] = 1e6
            return output
    mask = np.ones((2, 3), dtype=bool)
    mask[0, 0] = False
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    samples = sample_decoded_ensemble(Model(), transform, 4, support_mask=mask)
    assert np.isfinite(samples).all() and (samples[:, :, 0, 0] == 0).all()
    assert (samples[:, :, mask] == 9).all()
    decoded = torch.ones((2, 2, 2, 3))
    decoded[:, :, 0, 0] = 1e6
    physical = visualize.physical_decode(decoded, transform, mask)
    assert np.isfinite(physical).all() and (physical[:, :, 0, 0] == 0).all()
    assert (physical[:, :, mask] == 9).all()


@pytest.fixture
def fixture_files(tmp_path, monkeypatch):
    paths = [tmp_path / f"or_{orbit:04d}.nc" for orbit in (85, 89, 487)]
    for path in paths:
        path.touch()
    loader = data.load_conductance_cs
    def materialize(path, product_loader=None, wic_source_interval=None, minimum_sza=None):
        orbit = int(Path(path).stem[3:])
        return loader(path, lambda filename: NativeProduct(orbit), wic_source_interval, minimum_sza)
    monkeypatch.setattr(data, "load_conductance_cs", materialize)
    return paths


@pytest.mark.parametrize("uncertainty", [[], ["--measurement-uncertainty"], ["--gaussian-uncertainty"]])
def test_end_to_end_native_artifacts_and_matched_resampling(fixture_files, tmp_path, uncertainty):
    output = tmp_path / "run"
    options = [*[str(path) for path in fixture_files], "--output", str(output), "--no-csv-selection", "--epochs", "1", "--samples", "8", "--device", "cpu", *uncertainty]
    arguments = pipeline.parse_arguments(options)
    metrics = pipeline.run(arguments)
    model, transform, artifact = load_run_artifact(output / "conductance_vae.pt")
    assert artifact["artifact_type"] == data.CONDUCTANCE_ARTIFACT_TYPE
    assert model.configuration()["latent_dim"] == 36 and model.configuration()["spline_control_points"] == 30
    assert artifact["training_configuration"]["field_names"] == ["Hall", "Pedersen"]
    assert artifact["training_configuration"]["loss_configuration"]["measurement_uncertainty_in_loss"] == bool(uncertainty)
    assert artifact["data_provenance"]["unsupported_training_cells"] == 1
    assert transform.kind == ChannelTransform.LOG10_KIND
    with np.load(output / "ensemble_products.npz") as stored:
        assert "fuv_mean" not in stored.files
        assert not stored["conductance_supported"][0, 0]
        assert (stored["conductance_anomaly_factor"].reshape(2, 46, 46, 8)[:, 0, 0] == 0).all()
        assert stored["latent_samples"].shape == (8, 36)
    checksum = (output / "conductance_vae.pt").read_bytes()
    assert pipeline.run(pipeline.parse_arguments([*options, "--reuse-artifact"])) == metrics
    assert (output / "conductance_vae.pt").read_bytes() == checksum
    with pytest.raises(FileExistsError):
        pipeline.run(arguments)
    with pytest.raises(ValueError, match="training settings"):
        pipeline.run(pipeline.parse_arguments([*options, "--reuse-artifact", "--beta", "1.6"]))
    assert (output / "figures" / "loss_history.png").exists()
    fixture_files[0].touch()
    with pytest.raises(ValueError, match="saved native data"):
        pipeline.run(pipeline.parse_arguments([*options, "--reuse-artifact"]))


def test_native_reference_and_browser_never_reconvert_H_as_WIC(fixture_files, tmp_path, monkeypatch):
    output = tmp_path / "run"
    pipeline.run(pipeline.parse_arguments([*[str(path) for path in fixture_files], "--output", str(output), "--no-csv-selection", "--epochs", "1", "--samples", "8", "--device", "cpu"]))
    def forbidden(*args, **kwargs):
        raise AssertionError("native H/P must never be propagated from WIC again")
    monkeypatch.setattr(plotter, "conductance_from_corrected_wic", forbidden)
    monkeypatch.setattr(browser, "conductance_from_corrected_wic", forbidden)
    truth, valid, identity = plotter.load_conductance_frame(output, "test", 1)
    summary = json.loads((output / "summary.json").read_text())
    source = data.load_conductance_cs_files(summary["split_files"]["test"])
    np.testing.assert_array_equal(truth, source["values"][1])
    assert identity["frame_index"] == 1 and not valid[0, 0]
    assert not np.allclose(truth[0, 1:], truth[1, 1:] * 0.81)
    plotter.run(plotter.parse_arguments([str(output), "--output", str(tmp_path / "recon"), "--frame-index", "1", "--modes", "2", "--dpi", "20"]), case="fields")
    reconstruction_report = json.loads((tmp_path / "recon" / "figure_5a_reconstruction.json").read_text())
    assert reconstruction_report["electron_energy_keV"] is None
    assert reconstruction_report["reference_kind"].startswith("held-out native conductance_cs")
    result = browser.run(browser.parse_arguments([str(output), "--output", str(tmp_path / "browse"), "--start", "1", "--stop", "2", "--dpi", "20"]))
    assert result["count"] == 1
    assert "Common Hall/Pedersen" in (tmp_path / "browse" / "index.html").read_text()
    with pytest.raises(ValueError, match="do not contain WIC"):
        browser.run(browser.parse_arguments([str(output), "--fields", "fuv", "--output", str(tmp_path / "no_fuv")]))


def test_general_visualizer_handles_native_heldout_and_prior_figures(fixture_files, tmp_path, monkeypatch):
    output = tmp_path / "run"
    pipeline.run(pipeline.parse_arguments([*[str(path) for path in fixture_files], "--output", str(output), "--no-csv-selection", "--epochs", "1", "--samples", "8", "--device", "cpu"]))
    model_bytes = (output / "conductance_vae.pt").read_bytes()
    labels = []
    image_panel = visualize.image_panel

    def record_panel(axis, values, title, colorbar_label, **kwargs):
        labels.append((title, colorbar_label))
        return image_panel(axis, values, title, colorbar_label, **kwargs)

    monkeypatch.setattr(visualize, "image_panel", record_panel)
    options = [str(output), "--output", str(tmp_path / "diagnostics"), "--posterior-prior-only", "--posterior-device", "cpu", "--spectrum-modes", "2", "--posterior-examples", "1", "--inpainting-examples", "1"]
    # Held-out diagnostics must not decompress the saved 50k-draw factors.
    with monkeypatch.context() as patch:
        patch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("held-out-only loaded saved factors"))
        report = visualize.run(visualize.parse_arguments(options))
    diagnostics = report["posterior_prior"]
    assert report["saved_prior_anomaly_factors_loaded"] is False
    assert diagnostics["field_names"] == ["Hall", "Pedersen"]
    assert diagnostics["physical_units"] == ["S", "S"]
    assert diagnostics["unsupported_training_cells_excluded"] == 1
    assert diagnostics["frame_count"] == 3
    assert "inpainted_removed_region_physical_rmse_siemens" in diagnostics["inpainting"]["metrics"]
    assert "mean_siemens" in diagnostics["physical_moments_on_identical_held_out_mask"]["observed"]
    assert any("Hall" in title and unit == "S" for title, unit in labels)
    assert any("Pedersen" in title and unit == "S" for title, unit in labels)
    assert not any("WIC" in title or "SI12" in title or unit == "counts" for title, unit in labels)
    assert {"posterior_prior_examples.png", "heldout_inpainting_examples.png"} <= set(report["created_figures"])
    for filename in report["created_figures"]:
        assert (tmp_path / "diagnostics" / filename).is_file()
    prior = visualize.run(visualize.parse_arguments([str(output), "--output", str(tmp_path / "prior"), "--examples", "2", "--modes", "2", "--spectrum-modes", "2"]))
    assert set(prior["modes"]) == {"conductance"}
    assert prior["dense_covariance_materialized"] is False
    assert (output / "conductance_vae.pt").read_bytes() == model_bytes


def test_native_defaults_match_current_VAE_not_interactive_architecture():
    old = fuv_pipeline.parse_arguments(["unused", "--output", "unused"])
    new = pipeline.parse_arguments(["unused", "--output", "unused"])
    for name in ("epochs", "batch_size", "latent_dim", "learning_rate", "beta", "spline_control_points", "spline_degree", "scheduler_factor", "scheduler_patience", "maximum_missing_fraction", "seed", "training_holes", "measurement_uncertainty"):
        assert getattr(new, name) == getattr(old, name), name
