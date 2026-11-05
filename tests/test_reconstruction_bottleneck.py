"""Matched bottleneck fits, legacy arithmetic, provenance and frozen-model tests."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.model import SplineWICSI12VAE, _spline_basis, vae_loss
from scripts.IMAGE.vae_pipeline import diagnose_reconstruction_bottleneck as diagnostic
from tests.test_loss_balance import small_dataset


@pytest.fixture(autouse=True)
def single_torch_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_legacy_objective_matches_production_with_unequal_frame_weight_mass():
    dataset = small_dataset()
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))
    batch["weight"][0] *= .1
    decoded = (batch["target"] + .3).requires_grad_()
    mu = torch.zeros(2, 3)
    reference = vae_loss(decoded, batch["target"], batch["mask"], batch["weight"], mu, mu)["reconstruction"]
    actual = diagnostic.legacy_reconstruction(decoded, batch["target"], batch["mask"], batch["weight"])
    torch.testing.assert_close(actual, reference)
    a = torch.autograd.grad(actual, decoded, retain_graph=True)[0]
    b = torch.autograd.grad(reference, decoded)[0]
    torch.testing.assert_close(a, b)
    metrics = diagnostic.reconstruction_metrics(decoded.detach().numpy(), batch["target"].numpy(), batch["mask"].numpy(), batch["weight"].numpy())
    assert metrics["legacy_reconstruction"] == pytest.approx(float(reference), rel=1e-6)


def test_free_spline_matches_exact_saved_axis_order_on_nonsquare_grid():
    row = _spline_basis(11, 5, 3).double().numpy()
    column = _spline_basis(13, 5, 3).double().numpy()
    coefficients = np.random.default_rng(2).normal(size=(2, 2, 5, 5))
    target = torch.einsum("xi,bcij,yj->bcyx", torch.tensor(column), torch.tensor(coefficients), torch.tensor(row)).numpy()
    q = np.random.default_rng(3).uniform(.1, 1, (2, 1, 11, 13))
    fitted, controls, ranks = diagnostic.fit_free_splines(target, np.ones_like(q, dtype=bool), q, row, column, 1e-10)
    np.testing.assert_allclose(fitted, target, atol=1e-12)
    np.testing.assert_allclose(controls, coefficients, atol=1e-11)
    assert ranks == [25, 25]


def test_spline_fit_ignores_masked_and_zero_quality_targets():
    row = _spline_basis(8, 5, 3).double().numpy()
    target = np.ones((1, 2, 8, 8))
    q = np.ones((1, 1, 8, 8))
    mask = np.ones_like(q, dtype=bool)
    q[:, :, 0, 0] = 0
    mask[:, :, 0, 1] = False
    clean, _, _ = diagnostic.fit_free_splines(target, mask, q, row, row, 1e-10)
    target[:, :, 0, :2] = np.nan
    fitted, _, _ = diagnostic.fit_free_splines(target, mask, q, row, row, 1e-10)
    np.testing.assert_allclose(fitted, clean, atol=1e-12)
    q[:] = 0
    with pytest.raises(ValueError, match="positive common quality"):
        diagnostic.fit_free_splines(target, mask, q, row, row, 1e-10)


def test_selection_balances_orbits_is_reproducible_and_has_no_duplicates():
    orbits = np.repeat([85, 86, 89], [5, 7, 6])
    indices = diagnostic.select_frames(orbits, 6, seed=4)
    np.testing.assert_array_equal(indices, diagnostic.select_frames(orbits, 6, seed=4))
    assert len(np.unique(indices)) == 6
    np.testing.assert_array_equal(np.unique(orbits[indices], return_counts=True)[1], [2, 2, 2])
    np.testing.assert_array_equal(diagnostic.select_frames(orbits, len(orbits), seed=1), np.arange(len(orbits)))
    with pytest.raises(ValueError):
        diagnostic.select_frames(orbits, len(orbits) + 1, seed=1)


def test_saved_selection_checks_frame_identity_and_actual_data():
    dataset = small_dataset()
    dataset.frame_orbit = np.array([85, 86])
    dataset.frame_time = np.array(["2001-01-01", "2001-01-02"])
    artifact = {"split_files": {"train": ["85.nc", "86.nc"]}, "transform": {}, "training_configuration": {}}
    record = diagnostic.selection_record(dataset, artifact, "train", 2, 1)
    assert diagnostic.selection_record(dataset, artifact, "train", 99, 22, saved=record) == record
    changed = deepcopy(record)
    changed["orbit"][0] = 88
    with pytest.raises(ValueError, match="identities"):
        diagnostic.selection_record(dataset, artifact, "train", 2, 1, saved=changed)
    dataset.weight[0, 0, 1, 1] *= .5
    with pytest.raises(ValueError, match="weights have changed"):
        diagnostic.selection_record(dataset, artifact, "train", 2, 1, saved=record)


class LinearFrozenDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 2
        self.scale = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)

    def decode(self, z):
        return (z * self.scale)[:, :, None, None].expand(-1, -1, 2, 3)


def test_latent_oracle_improves_retains_best_and_is_deterministic():
    model = LinearFrozenDecoder().eval()
    target = torch.tensor([[1., -2.], [.5, 1.5]])[:, :, None, None].expand(-1, -1, 2, 3)
    batch = {"target": target, "mask": torch.ones(2, 1, 2, 3, dtype=torch.bool), "weight": torch.ones(2, 1, 2, 3)}
    kwargs = dict(model=model, batch=batch, encoded_mu=torch.zeros(2, 2), steps=100, starts=3, learning_rate=.1, seed=5, batch_size=1)
    first = diagnostic.optimize_latents(**kwargs)
    second = diagnostic.optimize_latents(**kwargs)
    assert diagnostic.legacy_reconstruction(first[0], **batch) < .005
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a, b)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert float(model.scale) == 1
    q = batch["weight"].sum((1, 2, 3))
    chosen, _ = diagnostic.reconstruction_terms(first[0], **batch)
    best = np.minimum(first[3].min(axis=(0, 1)), first[3][0, 0])
    np.testing.assert_allclose(chosen.numpy() / q.numpy() * 12, best, atol=1e-7)


def test_real_spline_latent_fit_never_worsens_encoder_or_changes_buffers():
    torch.manual_seed(2)
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=5).eval().requires_grad_(False)
    batch = next(iter(torch.utils.data.DataLoader(small_dataset(), batch_size=2)))
    state = {name: value.clone() for name, value in model.state_dict().items()}
    with torch.no_grad():
        mu, _ = model.encode(batch["encoder_input"], batch["mask"])
        original = model.decode(mu)
    fitted, _, _, _ = diagnostic.optimize_latents(model, batch, mu, steps=5, starts=2, learning_rate=.03, seed=1, batch_size=2)
    before, _ = diagnostic.reconstruction_terms(original, batch["target"], batch["mask"], batch["weight"])
    after, _ = diagnostic.reconstruction_terms(fitted, batch["target"], batch["mask"], batch["weight"])
    assert torch.all(after <= before)
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    assert all(parameter.grad is None for parameter in model.parameters())


def test_cli_rejects_unsafe_or_invalid_settings():
    for extra in (["--steps", "0"], ["--starts", "0"], ["--learning-rate", "nan"], ["--splits", "train", "train"], ["--spline-rcond", "1"]):
        with pytest.raises(SystemExit):
            diagnostic.parse_arguments(["run", "--output", "output", *extra])


def test_run_refuses_to_overwrite_existing_output(tmp_path):
    (tmp_path / "keep.txt").write_text("user output")
    with pytest.raises(FileExistsError):
        diagnostic.run(SimpleNamespace(output=tmp_path))
    assert (tmp_path / "keep.txt").read_text() == "user output"


def test_full_workflow_writes_matched_metrics_arrays_plots_and_reusable_selection(tmp_path, monkeypatch):
    dataset = small_dataset()
    dataset.frame_orbit = np.array([85, 86])
    dataset.frame_time = np.array(["2001-01-01", "2001-01-02"])
    torch.manual_seed(3)
    model = SplineWICSI12VAE(image_shape=(8, 8), latent_dim=3, spline_control_points=5).eval()
    state = {name: value.clone() for name, value in model.state_dict().items()}
    transform = dataset.transform
    artifact = {"split_files": {"train": ["85.nc"], "validation": ["86.nc"]},
                "model_configuration": model.configuration(), "transform": {"kind": transform.kind},
                "training_configuration": {"loss_configuration": {"kind": "legacy_weighted_mse_v1"}}}
    monkeypatch.setattr(diagnostic, "load_run_artifact", lambda *args, **kwargs: (model, transform, artifact))
    monkeypatch.setattr(diagnostic, "_load_heldout_dataset", lambda *args, requested_split: (dataset, requested_split))
    run_path = tmp_path / "run"
    run_path.mkdir()
    (run_path / "wic_si12_vae.pt").write_bytes(b"unchanged test artifact")
    arguments = diagnostic.parse_arguments([str(run_path), "--output", str(tmp_path / "first"), "--frame-count", "2",
                                           "--steps", "2", "--starts", "2", "--examples", "1", "--device", "cpu"])
    report = diagnostic.run(arguments)
    for name in ("metrics.json", "selection.json", "comparisons.npz", "reconstruction_rmse.png",
                 "convergence_train.png", "convergence_validation.png", "reconstruction_examples_train.png", "reconstruction_examples_validation.png"):
        assert (arguments.output / name).is_file()
    assert json.loads((arguments.output / "metrics.json").read_text()) == report
    for name, value in model.state_dict().items():
        assert torch.equal(value, state[name])
    selection = arguments.output / "selection.json"
    arguments.output = tmp_path / "second"
    arguments.selection = selection
    arguments.seed = 77
    arguments.frame_count = 99
    reused = diagnostic.run(arguments)
    for split in arguments.splits:
        assert reused["splits"][split]["selection"] == report["splits"][split]["selection"]
