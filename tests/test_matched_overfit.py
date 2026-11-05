"""Same-frame deterministic overfit: no sampling, source mutation or default sweep."""
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.model import SplineWICSI12VAE
from scripts.IMAGE.vae_pipeline import diagnose_latent_capacity as overfit
from scripts.IMAGE.vae_pipeline.diagnose_reconstruction_bottleneck import selection_record
from tests.test_loss_balance import small_dataset


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def matched_source(tmp_path, monkeypatch):
    dataset = small_dataset()
    dataset.frame_orbit = np.array([85, 86])
    dataset.frame_time = np.array(["2001-01-01", "2001-01-02"])
    torch.manual_seed(4)
    source = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=5).eval()
    artifact = {"model_configuration": source.configuration(), "transform": {"kind": dataset.transform.kind},
                "split_files": {"train": ["85.nc", "86.nc"]},
                "training_configuration": {"loss_configuration": {"kind": "legacy_weighted_mse_v1"}}}
    monkeypatch.setattr(overfit, "load_run_artifact", lambda *args, **kwargs: (source, dataset.transform, artifact))
    monkeypatch.setattr(overfit, "_load_heldout_dataset", lambda *args, **kwargs: (dataset, "train"))
    run = tmp_path / "source"
    run.mkdir()
    (run / "wic_si12_vae.pt").write_bytes(b"source model must not change")
    baseline = tmp_path / "e04"
    baseline.mkdir()
    selection = selection_record(dataset, artifact, "train", 2, 1)
    (baseline / "selection.json").write_text(json.dumps({"splits": {"train": selection}}))
    source_hash = hashlib.sha256((run / "wic_si12_vae.pt").read_bytes()).hexdigest()
    (baseline / "metrics.json").write_text(json.dumps({"artifact_sha256": source_hash, "splits": {"train": {"selection": selection}}}))
    np.savez(baseline / "comparisons.npz", train_target=dataset.target, train_mask=dataset.common_valid,
             train_weight=dataset.weight, train_optimized_latent=dataset.target + .1, train_free_spline=dataset.target + .01)
    return dataset, source, run, baseline, selection


def test_matched_run_preserves_source_uses_exact_frames_and_saves_best_checkpoint(matched_source, tmp_path):
    dataset, source, run, baseline, selection = matched_source
    state = {name: value.clone() for name, value in source.state_dict().items()}
    arguments = overfit.parse_arguments([str(run), "--selection", str(baseline / "selection.json"),
                                        "--output", str(tmp_path / "result"), "--epochs", "2", "--batch-size", "2", "--examples", "1", "--device", "cpu"])
    assert arguments.latent_dimensions is None
    report = overfit.run(arguments)
    assert report["frame_count"] == 2 and report["selection"] == selection
    assert report["selected_dataset_indices"] == selection["indices"]
    assert len(report["results"]) == 1 and report["results"][0]["latent_dim"] == 3
    assert not report["results"][0]["architecture_changed"]
    assert "no holes" in report["training"] and "beta=0" in report["training"]
    assert "Optimized latent (oracle)" in report["baselines"]
    history = report["history"]["Overfit latent 3"]
    assert len(history) == 3
    result = report["results"][0]
    assert result["legacy_reconstruction"] == pytest.approx(min(h["eval_mode_subset_loss"] for h in history), rel=1e-7)
    assert history[result["best_epoch"]]["eval_mode_subset_loss"] == pytest.approx(result["legacy_reconstruction"], rel=1e-7)
    for name, value in source.state_dict().items():
        assert torch.equal(value, state[name])
    assert all(parameter.grad is None for parameter in source.parameters())
    assert (run / "wic_si12_vae.pt").read_bytes() == b"source model must not change"
    output = tmp_path / "result"
    for filename in ("overfit_comparison.png", "overfit_convergence.png", "overfit_reconstruction_examples.png",
                     "overfit_comparisons.npz", "selection.json", "latent_capacity_sweep.json", "deterministic_overfit_latent3.pt"):
        assert (output / filename).is_file()
    checkpoint = torch.load(output / "deterministic_overfit_latent3.pt", weights_only=False)
    restored = SplineWICSI12VAE.from_configuration(checkpoint["model_configuration"])
    restored.load_state_dict(checkpoint["model_state"])
    prediction = overfit.predict_fields(restored, torch.utils.data.DataLoader(dataset, batch_size=2), "cpu")
    with np.load(output / "overfit_comparisons.npz") as arrays:
        np.testing.assert_allclose(prediction, arrays["Overfit_latent_3"], rtol=1e-6, atol=1e-6)
    assert json.loads((output / "latent_capacity_sweep.json").read_text()) == report


def test_no_sampling_and_full_input_preserved_in_deterministic_training(monkeypatch):
    dataset = small_dataset()
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=5)
    expected = [dataset[index]["encoder_input"] for index in range(len(dataset))]
    original_encode = model.encode

    def checked_encode(values, mask):
        for frame in values:
            assert any(torch.equal(frame.cpu(), full) for full in expected)
        assert torch.all(mask)
        return original_encode(values, mask)

    def forbidden(*args, **kwargs):
        pytest.fail("stochastic forward/reparameterization must not be used")

    monkeypatch.setattr(model, "encode", checked_encode)
    monkeypatch.setattr(model, "forward", forbidden)
    monkeypatch.setattr(model, "reparameterize", forbidden)
    arguments = SimpleNamespace(batch_size=2, seed=1, epochs=2, learning_rate=1e-4)
    overfit.fit_subset(model, dataset, dataset.target, dataset.common_valid, dataset.weight, arguments, "cpu")


def test_best_eval_checkpoint_is_restored_even_if_last_epoch_worsens(monkeypatch):
    dataset = small_dataset()
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=5)
    original_predict = overfit.predict_fields
    states = []

    def captured_predict(model, loader, device):
        prediction = original_predict(model, loader, device)
        states.append({name: value.clone() for name, value in model.state_dict().items()})
        return prediction

    scores = iter([3., 1., 2., 1.])
    monkeypatch.setattr(overfit, "predict_fields", captured_predict)
    monkeypatch.setattr(overfit, "reconstruction_metrics", lambda *args: {"legacy_reconstruction": next(scores)})
    arguments = SimpleNamespace(batch_size=2, seed=1, epochs=2, learning_rate=1e-4)
    _, _, epoch, _ = overfit.fit_subset(model, dataset, dataset.target, dataset.common_valid, dataset.weight, arguments, "cpu")
    assert epoch == 1
    for name, value in model.state_dict().items():
        assert torch.equal(value, states[1][name])


@pytest.mark.parametrize("tamper", ["source", "weight"])
def test_mismatched_oracle_baselines_rejected_before_training(matched_source, tmp_path, monkeypatch, tamper):
    dataset, source, run, baseline, selection = matched_source
    if tamper == "source":
        report = json.loads((baseline / "metrics.json").read_text())
        report["artifact_sha256"] = "different checkpoint"
        (baseline / "metrics.json").write_text(json.dumps(report))
    else:
        np.savez(baseline / "comparisons.npz", train_target=dataset.target, train_mask=dataset.common_valid,
                 train_weight=dataset.weight * .5, train_optimized_latent=dataset.target, train_free_spline=dataset.target)
    monkeypatch.setattr(overfit, "fit_subset", lambda *args: pytest.fail("mismatched baselines must not be trained"))
    args = overfit.parse_arguments([str(run), "--selection", str(baseline / "selection.json"), "--output", str(tmp_path / "failed"), "--device", "cpu"])
    with pytest.raises(ValueError, match="differs"):
        overfit.run(args)


def test_selection_free_run_defaults_to_source_dimension_and_refits_floor(matched_source, tmp_path):
    _, _, run, _, _ = matched_source
    args = overfit.parse_arguments([str(run), "--output", str(tmp_path / "standalone"), "--frame-count", "2", "--epochs", "1", "--batch-size", "2", "--examples", "1", "--device", "cpu"])
    report = overfit.run(args)
    assert [result["latent_dim"] for result in report["results"]] == [3]
    assert "Free spline" in report["baselines"]
    assert "Optimized latent (oracle)" not in report["baselines"]


def test_output_overwrite_guard(tmp_path):
    (tmp_path / "user.txt").write_text("keep")
    with pytest.raises(FileExistsError):
        overfit.run(SimpleNamespace(run_directory="run", output=str(tmp_path)))
    assert (tmp_path / "user.txt").read_text() == "keep"


def test_cli_rejects_invalid_settings_and_does_not_default_to_latent_sweep():
    assert overfit.parse_arguments(["run"]).latent_dimensions is None
    assert overfit.parse_arguments(["run"]).decoder_feature_size is None
    assert overfit.parse_arguments(["run", "--decoder-feature-size", "30"]).decoder_feature_size == 30
    for options in (["--epochs", "0"], ["--latent-dimensions", "36", "36"], ["--latent-dimensions", "0"],
                    ["--learning-rate", "nan"], ["--seed", "-1"], ["--examples", "0"], ["--decoder-feature-size", "19"]):
        with pytest.raises(SystemExit):
            overfit.parse_arguments(["run", *options])


def test_decoder_resolution_changes_only_spatial_processing_and_roundtrips():
    models = []
    for size in (18, 30):
        torch.manual_seed(1)
        models.append(SplineWICSI12VAE(decoder_feature_size=size).eval())
    legacy, finer = models
    assert legacy.encoded_shape == finer.encoded_shape == (64, 6, 6)
    assert sum(p.numel() for p in legacy.parameters()) == sum(p.numel() for p in finer.parameters()) == 298490
    for name, value in legacy.state_dict().items():
        assert torch.equal(value, finer.state_dict()[name])
    assert "decoder_feature_size" not in legacy.configuration()
    assert finer.configuration()["decoder_feature_size"] == 30

    # Check where resolution changes, not just the final image shape.
    latent = torch.randn(2, 36)
    for model, size in zip(models, (18, 30)):
        seen = []
        hook = model.dec_conv2.register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].shape[-2:]))
        field = model.decode(latent)
        hook.remove()
        assert seen == [(size, size)]
        assert field.shape == (2, 2, 46, 46)
        field.square().mean().backward()
        assert torch.isfinite(model.fc_decode.weight.grad).all()
        assert torch.isfinite(model.spline_conv.weight.grad).all()
        restored = SplineWICSI12VAE.from_configuration(model.configuration()).eval()
        restored.load_state_dict(model.state_dict())
        torch.testing.assert_close(restored.decode(latent), field, rtol=0, atol=0)
    features = torch.randn(2, 16, 30, 30)
    torch.testing.assert_close(finer.spline_resize(features), features, rtol=0, atol=0)


def test_opt_in_decoder_run_preserves_source_basis_and_checkpoint_setting(matched_source, tmp_path):
    dataset, source, run, baseline, selection = matched_source
    state = {name: value.clone() for name, value in source.state_dict().items()}
    output = tmp_path / "decoder30"
    args = overfit.parse_arguments([str(run), "--selection", str(baseline / "selection.json"),
                                   "--output", str(output), "--decoder-feature-size", "30", "--epochs", "1",
                                   "--batch-size", "2", "--examples", "1", "--device", "cpu"])
    report = overfit.run(args)
    result = report["results"][0]
    assert report["selection"] == selection and result["latent_dim"] == source.latent_dim
    assert result["decoder_feature_size"] == 30 and result["architecture_changed"]
    assert result["parameter_count"] == sum(p.numel() for p in source.parameters())
    checkpoint = torch.load(output / "deterministic_overfit_latent3_decoder30.pt", weights_only=False)
    restored = SplineWICSI12VAE.from_configuration(checkpoint["model_configuration"])
    restored.load_state_dict(checkpoint["model_state"])
    assert restored.decoder_feature_size == 30
    assert torch.equal(restored.row_basis, source.row_basis) and torch.equal(restored.column_basis, source.column_basis)
    prediction = overfit.predict_fields(restored, torch.utils.data.DataLoader(dataset, batch_size=2), "cpu")
    with np.load(output / "overfit_comparisons.npz") as arrays:
        np.testing.assert_allclose(prediction, arrays["Overfit_latent_3,_decoder_30x30"], rtol=1e-6, atol=1e-6)
    for name, value in source.state_dict().items():
        assert torch.equal(value, state[name])
    assert (run / "wic_si12_vae.pt").read_bytes() == b"source model must not change"


@pytest.mark.parametrize("tamper", [None, "epochs", "weight", "decoder"])
def test_matched_control_reuse_and_mismatch_guards(matched_source, tmp_path, monkeypatch, tamper):
    _, _, run, baseline, _ = matched_source
    monkeypatch.setattr(overfit, "plot_results", lambda *args: None)
    common = [str(run), "--selection", str(baseline / "selection.json"), "--epochs", "1",
              "--batch-size", "2", "--device", "cpu"]
    control_output = tmp_path / "control"
    control = overfit.run(overfit.parse_arguments([*common, "--output", str(control_output)]))
    if tamper in ("epochs", "decoder"):
        if tamper == "epochs":
            control["epochs"] = 500
        else:
            control["results"][0]["model_configuration"]["decoder_feature_size"] = 30
        (control_output / "latent_capacity_sweep.json").write_text(json.dumps(control))
    elif tamper == "weight":
        with np.load(control_output / "overfit_comparisons.npz") as arrays:
            contents = {key: arrays[key].copy() for key in arrays.files}
        contents["weight"] *= .5
        np.savez(control_output / "overfit_comparisons.npz", **contents)
    candidate = overfit.parse_arguments([*common, "--output", str(tmp_path / "candidate"),
                                        "--decoder-feature-size", "30", "--overfit-control", str(control_output)])
    if tamper is not None:
        monkeypatch.setattr(overfit, "fit_subset", lambda *args: pytest.fail("mismatched controls must not be trained"))
        with pytest.raises(ValueError, match="matched overfit control"):
            overfit.run(candidate)
    else:
        report = overfit.run(candidate)
        label = "Matched overfit latent 3, decoder 18x18"
        assert report["baselines"][label]["legacy_reconstruction"] == control["results"][0]["legacy_reconstruction"]
        assert report["overfit_control"]["model_configuration"] == control["results"][0]["model_configuration"]
        with np.load(control_output / "overfit_comparisons.npz") as old, np.load(tmp_path / "candidate" / "overfit_comparisons.npz") as new:
            np.testing.assert_array_equal(new[label.replace(" ", "_")], old["Overfit_latent_3"])
