"""Fixed-Kp generation preserves paired members and protects original artifacts."""

from argparse import Namespace
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.CVAE import sample_conductance_at_kp as sampling


def setup_sampling(tmp_path, monkeypatch):
    run_directory = tmp_path / "model"
    run_directory.mkdir()
    (run_directory / "conductance_vae.pt").write_bytes(b"unchanged conditional artifact")
    support = np.array([[True, True, False]])
    model = SimpleNamespace(kp_conditioning={"center": 2., "scale": 1., "training_min": 1., "training_max": 5., "embedding_dim": 8}, image_shape=(1, 3))
    artifact = {"artifact_type": "conductance_vae_run_v1", "data_provenance": {"training_support": support, "grid": {"mlat": np.full((1, 3), 70.), "mlt": np.arange(3.)[None]}, "grid_id": "test-grid"}}
    monkeypatch.setattr(sampling, "load_run_artifact", lambda *args, **kwargs: (model, None, artifact))
    monkeypatch.setattr(sampling, "plot_fields", lambda *args: None)
    calls = []
    def sample(model, transform, count, **kwargs):
        latent = np.random.default_rng(kwargs["seed"]).normal(size=(count, 2)).astype(np.float32)
        first = latent[:, 0, None, None] + 20 + kwargs["kp"]
        second = 2 * first + latent[:, 1, None, None]
        fields = np.concatenate([np.repeat(first, 3, axis=2)[:, None], np.repeat(second, 3, axis=2)[:, None]], axis=1)
        fields[:, :, ~support] = 0
        calls.append((kwargs["kp"], fields.copy(), latent.copy()))
        return fields, latent
    monkeypatch.setattr(sampling, "sample_decoded_ensemble", sample)
    arguments = Namespace(run_directory=run_directory, output=tmp_path / "samples", kp=[2., 4.], draws=16, seed=71, threads=1, device="cpu", hall_vmax=50., pedersen_vmax=10.)
    return arguments, model, artifact, calls


def test_common_latents_pairing_explicit_kp_and_unsupported_quantiles(tmp_path, monkeypatch):
    arguments, _, _, calls = setup_sampling(tmp_path, monkeypatch)
    results = sampling.run(arguments)
    np.testing.assert_array_equal(calls[0][2], calls[1][2])
    for metadata, (_, fields, latent) in zip(results, calls):
        with np.load(arguments.output / metadata["directory"] / "ensemble_products.npz") as saved:
            recovered = (saved["conductance_anomaly_factor"].T * np.sqrt(arguments.draws - 1) + saved["conductance_mean"].ravel()).reshape(fields.shape)
            np.testing.assert_allclose(recovered, fields, atol=1e-6)
            np.testing.assert_array_equal(saved["latent_samples"], latent)
            np.testing.assert_array_equal(saved["conductance_sample_kp"], metadata["Kp"])
            assert np.isnan(saved["conductance_q90"][:, ~saved["conductance_supported"]]).all()
            np.testing.assert_allclose(saved["conductance_q50"][:, :,:2], np.quantile(fields, 0.5, axis=0)[:, :,:2])
        assert metadata["model_sha256"] == hashlib.sha256((arguments.run_directory / "conductance_vae.pt").read_bytes()).hexdigest()
        assert metadata["within_training_range"]
    assert len(json.loads((arguments.output / "sampling_summary.json").read_text())["selections"]) == 2


@pytest.mark.parametrize("kp", [[-1], [10], [float("nan")], [float("inf")], [2, 2], []])
def test_invalid_kp_rejected_without_output(tmp_path, monkeypatch, kp):
    arguments, _, _, calls = setup_sampling(tmp_path, monkeypatch)
    arguments.kp = kp
    with pytest.raises(ValueError):
        sampling.run(arguments)
    assert not arguments.output.exists() and not calls


def test_model_type_output_protection_and_extrapolation_warning(tmp_path, monkeypatch):
    arguments, model, artifact, calls = setup_sampling(tmp_path, monkeypatch)
    model.kp_conditioning = None
    with pytest.raises(ValueError, match="conditional native"):
        sampling.run(arguments)
    model.kp_conditioning = {"center": 2., "scale": 1., "training_min": 1., "training_max": 5., "embedding_dim": 8}
    artifact["artifact_type"] = "wic_si12_vae_run_v1"
    with pytest.raises(ValueError, match="conditional native"):
        sampling.run(arguments)
    artifact["artifact_type"] = "conductance_vae_run_v1"
    arguments.output.mkdir()
    protected = arguments.output / "existing.txt"
    protected.write_text("keep this")
    with pytest.raises(FileExistsError):
        sampling.run(arguments)
    assert protected.read_text() == "keep this" and not calls
    protected.unlink()
    arguments.kp = [6.]
    with pytest.warns(RuntimeWarning, match="extrapolation"):
        results = sampling.run(arguments)
    assert not results[0]["within_training_range"]


def test_real_tiny_conditional_artifact_sampling_and_figures(tmp_path):
    import torch
    from icanalyzer.cvae.artifacts import save_run_artifact
    from icanalyzer.cvae.data import ChannelTransform
    from icanalyzer.cvae.model import SplineConditionalVAE

    torch.manual_seed(2)
    conditioning = {"center": 3., "scale": 1., "training_min": 1., "training_max": 5., "embedding_dim": 8}
    model = SplineConditionalVAE(
        image_shape=(8, 8), latent_dim=4, spline_control_points=4,
        spline_degree=2, kp_conditioning=conditioning,
        field_names=("Hall", "Pedersen")
    )
    transform = ChannelTransform([1, 1], [1, 1], [0.1, 0.1], kind=ChannelTransform.LOG10_KIND)
    support = np.ones((8, 8), dtype=bool)
    support[0] = False
    provenance = {"training_support": support, "grid_id": "tiny", "grid": {"mlat": np.full((8, 8), 70.), "mlt": np.zeros((8, 8))}}
    run_directory = tmp_path / "model"
    path = run_directory / "conductance_vae.pt"
    save_run_artifact(path, model, transform, [], {}, provenance, {}, artifact_type="conductance_vae_run_v1")
    before = path.read_bytes()
    arguments = Namespace(run_directory=run_directory, output=tmp_path / "samples", kp=[2., 4.], draws=8, seed=71, threads=1, device="cpu", hall_vmax=50., pedersen_vmax=10.)
    results = sampling.run(arguments)
    assert path.read_bytes() == before
    with np.load(arguments.output / results[0]["directory"] / "ensemble_products.npz") as first, np.load(arguments.output / results[1]["directory"] / "ensemble_products.npz") as second:
        np.testing.assert_array_equal(first["latent_samples"], second["latent_samples"])
        assert not np.array_equal(first["conductance_mean"], second["conductance_mean"])
        assert np.isnan(first["conductance_q10"][:, ~support]).all()
    for result in results:
        assert (arguments.output / result["directory"] / "conductance_at_kp.png").stat().st_size > 0
        assert (arguments.output / result["directory"] / "conductance_at_kp.pdf").stat().st_size > 0
