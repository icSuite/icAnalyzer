"""Optional conditions must stay aligned and leave old conductance runs intact."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer import conductance
from icanalyzer.wic_si12.data import ChannelTransform, WICSI12ArrayDataset
from icanalyzer.wic_si12.ensemble import sample_decoded_ensemble
from icanalyzer.wic_si12.model import SplineWICSI12VAE
from icanalyzer.wic_si12.workflow import load_run_artifact
from scripts.manual_masking import annotated_data
from scripts.IMAGE.vae_pipeline import run_conductance_pipeline as pipeline
from scripts.IMAGE.vae_pipeline import visualize_wic_si12_run as visualize
from scripts.manual_masking.annotation_store import encode_mask
from test_native_conductance_pipeline import NativeProduct


CONDITION = {"center": 3., "scale": 1., "training_min": 1., "training_max": 5., "embedding_dim": 8}


def conditional_product(orbit=85):
    product = NativeProduct(orbit)
    product.arrays["Kp"] = np.array([1., 3., 5.], dtype=np.float32)
    product.Kp = SimpleNamespace(dimensions=("time",), units="1")
    product.time = np.array(["2000-01-01T00:00:00", "2000-01-01T03:00:00", "2000-01-01T06:00:00"], dtype="datetime64[us]")
    product.Kp_interval_start = product.time.copy()
    product.kp_provenance = {"source": "fixture", "status": "def"}
    return product


def test_optional_native_kp_source_slicing_and_default_metadata():
    product = conditional_product()
    ordinary = conductance.load_conductance_cs("or_0085.nc", lambda path: product)
    assert "kp" not in ordinary and "kp_provenance" not in ordinary["metadata"]
    selected = conductance.load_conductance_cs("or_0085.nc", lambda path: product, (11, 15), include_kp=True)
    np.testing.assert_array_equal(selected["kp"], [3, 5])
    np.testing.assert_array_equal(selected["kp_interval_start"], product.time[1:])
    assert selected["metadata"]["kp_provenance"]["source"] == "fixture"
    product.Kp = product.arrays["Kp"]
    product.variable_attrs = {"Kp": {"units": "1"}}
    eager = conductance.load_conductance_cs("or_0085.nc", lambda path: product, include_kp=True)
    np.testing.assert_array_equal(eager["kp"], [1, 3, 5])
    product.Kp_interval_start = product.time + np.timedelta64(1, "s")
    with pytest.raises(ValueError, match="interval"):
        conductance.load_conductance_cs("or_0085.nc", lambda path: product, include_kp=True)


def test_kp_alignment_through_json_selection_then_intrinsic_frame_rejection():
    shape = (8, 8)
    values = np.ones((4, 2, *shape), dtype=np.float32)
    times = np.arange(4) * np.timedelta64(3, "h") + np.datetime64("2000-01-01", "us")
    grid = {"mlat": np.zeros(shape), "mlt": np.zeros(shape)}
    product = {"values": values, "standard_deviation": values.copy(), "valid": np.ones_like(values, bool),
               "quality": values.copy(), "frame_missing_fraction": np.array([0, .8, 0, 0]),
               "time": times, "orbit": np.full(4, 85), "grid": grid, "kp": np.array([1, 2, 3, 4]),
               "kp_interval_start": times.copy(), "metadata": []}
    record = {"n_frames": 4, "usable_start": 1, "usable_stop": 3, "full_good_start": 1,
              "full_good_stop": 3, "review_status": "complete", "target_annotated_runs": encode_mask(np.ones(shape, bool)), "keyframes": {}}
    selected = annotated_data.apply_annotations(product, {"records": {85: record}, "source_times": {85: times}, "grid": grid})
    np.testing.assert_array_equal(selected["kp"], [2, 3, 4])
    np.testing.assert_array_equal(selected["kp_interval_start"], times[1:])
    transform = ChannelTransform([1, 1], [0, 0], [1, 1])
    dataset = WICSI12ArrayDataset(selected, transform)
    np.testing.assert_array_equal(dataset.kp, [3, 4])
    assert dataset[0]["kp"].item() == 3
    normalization = conductance.fit_kp_conditioning(dataset.kp)
    assert normalization["center"] == 3.5 and normalization["scale"] == .5


@pytest.mark.parametrize("kp", [None, float("nan"), -1., 10., [1.]])
def test_conditional_decoder_requires_explicit_finite_aligned_kp(kp):
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=4, spline_control_points=4, kp_conditioning=CONDITION)
    with pytest.raises(ValueError, match="Kp"):
        model.decode(torch.zeros(2, 4), kp=kp)


def test_condition_changes_decoder_and_sampling_preserves_identical_latents():
    torch.manual_seed(7)
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=4, spline_control_points=4, kp_conditioning=CONDITION).eval()
    transform = ChannelTransform([1, 1], [1, 1], [.1, .1], kind=ChannelTransform.LOG10_KIND)
    first, z1 = sample_decoded_ensemble(model, transform, 8, seed=5, kp=1, return_latent=True)
    second, z2 = sample_decoded_ensemble(model, transform, 8, seed=5, kp=5, return_latent=True)
    np.testing.assert_array_equal(z1, z2)
    assert not np.array_equal(first, second)
    values = torch.ones(2, 3, 8, 8)
    mask = torch.ones(2, 1, 8, 8, dtype=torch.bool)
    mu1, _ = model.encode(values, mask, kp=1)
    mu5, _ = model.encode(values, mask, kp=5)
    assert not torch.equal(mu1, mu5)
    model.decode(torch.zeros(2, 4), kp=3).sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in model.kp_embedding.parameters()) > 0


def test_unconditional_config_and_explicit_condition_rejection():
    torch.manual_seed(8)
    first = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=4, spline_control_points=4)
    torch.manual_seed(8)
    second = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=4, spline_control_points=4, kp_conditioning=None)
    assert "kp_conditioning" not in first.configuration()
    assert first.configuration() == second.configuration()
    assert all(torch.equal(value, second.state_dict()[name]) for name, value in first.state_dict().items())
    with pytest.raises(ValueError, match="unconditional"):
        first.decode(torch.zeros(2, 4), kp=3)
    with pytest.raises(ValueError, match="unconditional"):
        sample_decoded_ensemble(first, ChannelTransform([1, 1], [0, 0], [1, 1]), 4, kp=3)


def test_native_conditional_training_roundtrip_reuse_and_heldout_diagnostics(tmp_path, monkeypatch):
    paths = [tmp_path / f"or_{orbit:04d}.nc" for orbit in (85, 89, 487)]
    for path in paths:
        path.touch()
    original_loader = conductance.load_conductance_cs
    def materialize(path, product_loader=None, wic_source_interval=None, minimum_sza=None, include_kp=False):
        return original_loader(path, lambda _: conditional_product(int(Path(path).stem[3:])), wic_source_interval, minimum_sza, include_kp)
    monkeypatch.setattr(conductance, "load_conductance_cs", materialize)
    output = tmp_path / "run"
    options = [*[str(path) for path in paths], "--output", str(output), "--all-frames", "--epochs", "1", "--samples", "8", "--device", "cpu", "--condition-on-kp", "--sample-kp", "2"]
    metrics = pipeline.run(pipeline.parse_arguments(options))
    model, transform, artifact = load_run_artifact(output / "conductance_vae.pt")
    assert model.kp_conditioning["center"] == 3
    assert model.kp_conditioning == artifact["training_configuration"]["kp_conditioning"]
    np.testing.assert_array_equal(artifact["data_provenance"]["retained_frame_kp"]["train"], [1, 3, 5])
    assert metrics["prior_Kp"] == 1 and metrics["prior_sample_count"] == 8
    dataset, _ = visualize._load_heldout_dataset(artifact, transform, requested_split="test")
    arrays = visualize.infer_posterior_prior(model, transform, dataset, 2, "cpu", 1)
    np.testing.assert_array_equal(arrays["kp"], [1, 3, 5])
    inpainting = visualize.infer_heldout_inpainting(model, transform, dataset, 2, "cpu", 3)
    assert np.isfinite(inpainting["reconstruction"]).all()
    diagnostic_options = [str(output), "--output", str(tmp_path / "diagnostics"), "--posterior-prior-only", "--posterior-device", "cpu", "--spectrum-modes", "2", "--posterior-examples", "1", "--inpainting-examples", "1"]
    diagnostic_report = visualize.run(visualize.parse_arguments(diagnostic_options))["posterior_prior"]
    assert "conditioned" in diagnostic_report["prior_kind"]
    assert diagnostic_report["frame_kp"] == [1, 3, 5]
    from scripts.IMAGE.vae_pipeline.plot_conductance_reconstruction import load_conductance_frame
    with pytest.raises(ValueError, match="unconditional"):
        load_conductance_frame(output, "test", 0)
    changed_artifact = dict(artifact)
    changed_artifact["data_provenance"] = dict(artifact["data_provenance"])
    changed_artifact["data_provenance"]["retained_frame_kp"] = {"test": np.array([2, 3, 5])}
    with pytest.raises(ValueError, match="retained-frame Kp"):
        visualize._load_heldout_dataset(changed_artifact, transform, requested_split="test")
    with np.load(output / "ensemble_products.npz") as saved:
        np.testing.assert_array_equal(saved["conductance_sample_kp"], np.full(8, 2))
        assert saved["reconstruction_kp"].item() == 1
    checksum = (output / "conductance_vae.pt").read_bytes()
    pipeline.run(pipeline.parse_arguments([*options, "--reuse-artifact", "--sample-kp", "4"]))
    assert (output / "conductance_vae.pt").read_bytes() == checksum
    with pytest.raises(ValueError, match="conditioning option"):
        pipeline.run(pipeline.parse_arguments([*[str(path) for path in paths], "--output", str(output), "--all-frames", "--reuse-artifact"]))
    for wrong in (["--condition-on-kp"], ["--sample-kp", "3"], ["--condition-on-kp", "--sample-kp", "nan"]):
        with pytest.raises(SystemExit):
            pipeline.parse_arguments(["dummy", "--output", "dummy", *wrong])
