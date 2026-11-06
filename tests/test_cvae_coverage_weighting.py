"""Coverage fitting and loss checks, without scientific model training."""

import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icanalyzer.cvae.artifacts import load_run_artifact, save_run_artifact
from icanalyzer.cvae.data import ChannelTransform
from icanalyzer.cvae.loss import LEGACY_LOSS_CONFIGURATION, coverage_bin_indices, smooth_coverage_weighting, vae_loss, validate_coverage_weighting
from icanalyzer.cvae.model import SplineConditionalVAE
from icanalyzer.cvae.training import _run_epoch
from scripts.CVAE.prepare_cvae_hp_cache import CACHE_FORMAT, cached_coverage_weighting
from scripts.CVAE.train_automatic_cvae import parse_arguments


def test_uniform_coverage_has_no_boundary_boost_and_unsupported_cells_are_not_filled():
    coverage = np.full((1, 7, 7), 10.)
    coverage[0, 3, 3] = 0
    configuration, smoothed = smooth_coverage_weighting(coverage)
    np.testing.assert_allclose(smoothed, 10.)
    np.testing.assert_array_equal(configuration["factors"], np.ones_like(coverage))


def test_smoothing_precedes_inverse_and_boost_is_capped():
    coverage = np.full((1, 9, 9), 100.)
    coverage[0, 4, 4] = 1
    raw, _ = smooth_coverage_weighting(coverage, sigma_pixels=0)
    smooth, values = smooth_coverage_weighting(coverage, sigma_pixels=1.5)
    raw_factor, smooth_factor = np.asarray(raw["factors"]), np.asarray(smooth["factors"])
    assert raw_factor[0, 4, 4] == 3
    assert 1 < smooth_factor[0, 4, 4] < 1.2
    np.testing.assert_allclose(smooth_factor, np.clip(values.max() / values, 1, 3), rtol=1e-6)


def test_full_balance_removes_cap_and_equalizes_supported_unsmoothed_coverage():
    coverage = np.array([[[100., 1., 0.]]])
    configuration, _ = smooth_coverage_weighting(coverage, sigma_pixels=0, balance_mode="full")
    assert configuration["balance_mode"] == "full"
    assert configuration["maximum_boost"] is None
    factors = np.asarray(configuration["factors"])
    np.testing.assert_allclose(factors, [[[1, 100, 1]]])
    np.testing.assert_allclose((coverage * factors)[coverage > 0], 100.)


def test_full_balance_inverts_smoothed_coverage_and_preserves_unsupported_cells():
    coverage = np.ones((1, 9, 9))
    coverage[:, :, :4] = 100
    coverage[0, 0, 0] = 0
    configuration, smoothed = smooth_coverage_weighting(coverage, balance_mode="full")
    factors = np.asarray(configuration["factors"])
    supported = coverage > 0
    np.testing.assert_allclose((smoothed * factors)[supported], configuration["reference_coverage"][0], rtol=1e-6)
    assert factors.max() > 3
    assert factors[0, 0, 0] == 1
    broken = dict(configuration, maximum_boost=3)
    with pytest.raises(ValueError, match="no maximum"):
        validate_coverage_weighting(broken)


def test_kp_boundaries_include_zero_and_nine_and_edges_go_to_the_upper_bin():
    np.testing.assert_array_equal(coverage_bin_indices([0, 1.999, 2, 4, 6, 9], [0, 2, 4, 6, 9]), [0, 0, 1, 2, 3, 3])


def test_noninteger_maximum_boost_allows_float32_rounding():
    configuration, _ = smooth_coverage_weighting(np.array([[[100., 1.]]]), sigma_pixels=0, maximum_boost=1.1)
    validate_coverage_weighting(configuration)
    np.testing.assert_allclose(configuration["factors"], [[[1, 1.1]]])


@pytest.mark.parametrize("kwargs", [{"sigma_pixels": -1}, {"sigma_pixels": np.nan}, {"maximum_boost": .5}, {"maximum_boost": np.inf}, {"kp_bin_edges": [0, 2, 2, 9]}])
def test_invalid_coverage_settings_fail(kwargs):
    with pytest.raises(ValueError):
        smooth_coverage_weighting(np.ones((1, 3, 3)), **kwargs)


def test_empty_training_kp_bin_requires_broader_bins():
    with pytest.raises(ValueError, match="broader bins"):
        smooth_coverage_weighting(np.zeros((1, 3, 3)))


def test_only_eligible_training_quality_fits_the_maps(tmp_path):
    shard = tmp_path / "train"
    shard.mkdir()
    physical = np.ones((4, 5, 3, 3), dtype=np.float32)
    physical[:, 4] = np.array([1, 2, 3, 1e6])[:, None, None]
    physical[0, 4, 0, 1] = np.nan
    physical[0, 4, 0, 2] = -1
    valid = np.ones((4, 3, 3), dtype=bool)
    valid[:, 0, 0] = False
    np.save(shard / "physical.npy", physical)
    np.save(shard / "valid.npy", valid)
    np.savez(shard / "frames.npz", eligible=[True, True, True, False], kp=[0, 2, 9, 1])
    # Held-out paths deliberately do not exist: opening either would fail.
    metadata = dict(cache_format=CACHE_FORMAT, used_frames=dict(train=3), orbits=[
        dict(split="train", shard="train"), dict(split="validation", shard="do_not_open"), dict(split="test", shard="do_not_open")])
    (tmp_path / "manifest.json").write_text(json.dumps(metadata))
    configuration, arrays = cached_coverage_weighting(tmp_path, sigma_pixels=0, kp_bin_edges=[0, 2, 9])
    assert configuration["training_frames_per_bin"] == [1, 2]
    expected = np.ones((2, 3, 3))
    expected[1] = 5
    expected[:, 0, 0] = 0
    expected[0, 0, 1:3] = 0
    np.testing.assert_array_equal(arrays["quality_sum"], expected)
    np.testing.assert_array_equal(arrays["factors"][:, 0, 0], 1)


def loss_batch():
    return dict(encoder_input=torch.arange(18, dtype=torch.float32).reshape(3, 3, 1, 2),
        target=torch.zeros(3, 2, 1, 2), weight=torch.full((3, 1, 1, 2), .5),
        mask=torch.tensor([[[[True, True]]], [[[True, True]]], [[[True, False]]]]), kp=torch.tensor([0., 2., 9.]))


def test_multi_sample_loss_averages_reconstruction_and_counts_kl_once():
    target = torch.zeros(2, 1, 2, 2)
    mask = torch.ones(2, 1, 2, 2, dtype=torch.bool)
    weight = torch.ones(2, 1, 2, 2)
    mu = torch.tensor([[0.2, -0.1], [0.3, 0.4]])
    logvar = torch.tensor([[0.1, -0.2], [0.0, 0.3]])
    first = torch.full_like(target, 0.5)
    second = torch.full_like(target, 1.5)
    beta = 0.4

    individual = [
        vae_loss(sample, target, mask, weight, mu, logvar, beta=beta)
        for sample in (first, second)
    ]
    combined = vae_loss(
        torch.stack((first, second), dim=1),
        target, mask, weight, mu, logvar, beta=beta
    )
    for name in ("total", "reconstruction", "kl"):
        expected = torch.mean(torch.stack([loss[name] for loss in individual]))
        torch.testing.assert_close(combined[name], expected)


class FixedModel(torch.nn.Module):
    def forward(self, values, mask):
        self.inputs = values.clone(), mask.clone()
        prediction = torch.tensor([1., 2.]).expand(len(values), 2, 1, 2)
        latent = torch.zeros(len(values), 1)
        return prediction, latent, latent


def test_epoch_weights_correct_kp_map_without_changing_encoder_or_missing_mask():
    batch, model = loss_batch(), FixedModel()
    coverage = dict(kp_bin_edges=[0, 2, 9], maximum_boost=3, factors=[[[1, 3]], [[3, 1]]])
    configuration = dict(LEGACY_LOSS_CONFIGURATION, coverage_weighting=coverage)
    metrics = _run_epoch(model, [batch], None, "cpu", 1., False, loss_configuration=configuration)
    expected_weight = batch["weight"] * torch.tensor([1., 3., 3., 1., 3., 1.]).reshape(3, 1, 1, 2)
    prediction, mu, logvar = model(batch["encoder_input"], batch["mask"])
    expected = vae_loss(prediction, batch["target"], batch["mask"], expected_weight, mu, logvar)
    assert metrics["reconstruction"] == pytest.approx(float(expected["reconstruction"]) / 3)
    assert torch.equal(model.inputs[0], batch["encoder_input"])
    assert torch.equal(model.inputs[1], batch["mask"])
    prediction = prediction.clone().requires_grad_()
    vae_loss(prediction, batch["target"], batch["mask"], expected_weight, mu, logvar)["total"].backward()
    assert torch.equal(prediction.grad[2, :, :, 1], torch.zeros(2, 1))


def test_unit_coverage_map_preserves_existing_loss_exactly():
    coverage = dict(kp_bin_edges=[0, 9], maximum_boost=3, factors=[[[1, 1]]])
    default = _run_epoch(FixedModel(), [loss_batch()], None, "cpu", 1., False, loss_configuration=LEGACY_LOSS_CONFIGURATION)
    weighted = _run_epoch(FixedModel(), [loss_batch()], None, "cpu", 1., False, loss_configuration=dict(LEGACY_LOSS_CONFIGURATION, coverage_weighting=coverage))
    assert weighted == default


def test_decimal_kp_edge_uses_identical_precision_when_fitting_and_evaluating():
    batch = loss_batch()
    batch["kp"] = torch.tensor([2.333, 0., 9.])
    # The float32 observation is slightly below the decimal edge, not on it.
    np.testing.assert_array_equal(coverage_bin_indices(batch["kp"].numpy(), [0, 2.333, 9]), [0, 0, 1])
    coverage = dict(kp_bin_edges=[0, 2.333, 9], maximum_boost=3, factors=[[[1, 3]], [[3, 1]]])
    actual = _run_epoch(FixedModel(), [batch], None, "cpu", 1., False, loss_configuration=dict(LEGACY_LOSS_CONFIGURATION, coverage_weighting=coverage))
    prediction, mu, logvar = FixedModel()(batch["encoder_input"], batch["mask"])
    weight = batch["weight"] * torch.tensor([1., 3., 1., 3., 3., 1.]).reshape(3, 1, 1, 2)
    expected = vae_loss(prediction, batch["target"], batch["mask"], weight, mu, logvar)
    assert actual["reconstruction"] == pytest.approx(float(expected["reconstruction"]) / 3)


@pytest.mark.parametrize("balance_mode", ["capped", "full"])
def test_loaded_model_retains_the_frozen_coverage_loss(tmp_path, balance_mode):
    coverage, _ = smooth_coverage_weighting(np.ones((1, 46, 46)), balance_mode=balance_mode)
    configuration = dict(LEGACY_LOSS_CONFIGURATION, coverage_weighting=coverage)
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    model = SplineConditionalVAE(
        image_shape=(46, 46), latent_dim=2, spline_control_points=5,
        field_names=("Hall", "Pedersen")
    )
    filename = tmp_path / "model.pt"
    save_run_artifact(filename, model, transform, [], dict(loss_configuration=configuration), {}, {}, artifact_type="conductance_vae_run_v1")
    loaded, _, _ = load_run_artifact(filename)
    assert loaded.loss_configuration == configuration


def test_heldout_training_metric_uses_coverage_but_neutral_metrics_do_not():
    from scripts.IMAGE.vae_pipeline.visualize_wic_si12_run import posterior_reconstruction_skill
    shape = (2, 2, 1, 2)
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    observed = np.zeros(shape)
    reconstructed = np.tile([0., 99.], (2, 2, 1, 1))
    ensembles = dict(valid=np.ones(shape, dtype=bool), observed=observed, posterior_mean=reconstructed, posterior_sample=reconstructed)
    dataset = SimpleNamespace(weight=np.ones((2, 1, 1, 2)), kp=np.array([0., 9.]))
    coverage = dict(kp_bin_edges=[0, 9], maximum_boost=3, factors=[[[1, 3]]])
    configuration = dict(LEGACY_LOSS_CONFIGURATION, coverage_weighting=coverage)
    result = posterior_reconstruction_skill(ensembles, dataset, transform, configuration)
    np.testing.assert_allclose(result["posterior_mean"]["auroral_quality"], np.sqrt(2))
    np.testing.assert_allclose(result["posterior_mean"]["training_effective"], np.sqrt(3))


def test_cli_weighting_is_opt_in_and_rejects_invalid_edges():
    assert not parse_arguments(["cache", "--output", "run"]).coverage_weighting
    args = parse_arguments(["cache", "--output", "run", "--coverage-weighting", "--coverage-kp-edges", "0", "2", "4", "6", "9"])
    assert args.coverage_kp_edges == [0, 2, 4, 6, 9]
    assert args.coverage_balance == "capped"
    assert parse_arguments(["cache", "--output", "run", "--coverage-weighting", "--coverage-balance", "full"]).coverage_balance == "full"
    with pytest.raises(SystemExit):
        parse_arguments(["cache", "--output", "run", "--coverage-kp-edges", "1", "9"])


def test_python_settings_select_separate_weighted_output_and_allow_overrides(tmp_path, monkeypatch):
    import scripts.IMAGE.step_1_two_stage_cvae as step
    monkeypatch.setattr(step, "run", lambda arguments: arguments)
    assert step.main([]).output == step.OUTPUT
    monkeypatch.setattr(step, "COVERAGE_WEIGHTING", True)
    weighted = step.main([])
    assert weighted.coverage_weighting
    assert weighted.coverage_balance == "full"
    assert weighted.output == step.OUTPUT.with_name(step.OUTPUT.name + "_coverage_full")
    capped = step.main(["--coverage-balance", "capped"])
    assert capped.output == step.OUTPUT.with_name(step.OUTPUT.name + "_coverage")
    assert step.main(["--output", str(step.OUTPUT)]).output == step.OUTPUT
    monkeypatch.setattr(step, "COVERAGE_WEIGHTING", False)
    assert step.main(["--coverage-weighting"]).output == weighted.output
    custom = step.main(["--output", str(tmp_path / "comparison"), "--coverage-max-boost", "2"])
    assert custom.output == tmp_path / "comparison"
    assert custom.coverage_max_boost == 2


@pytest.mark.parametrize("balance_mode", ["capped", "full"])
def test_trainer_saves_fitted_maps_and_restores_them_for_evaluation(tmp_path, monkeypatch, balance_mode):
    import scripts.CVAE.train_automatic_cvae as trainer

    cache = tmp_path / "cache"
    cache.mkdir()
    rows = []
    for orbit, split in enumerate(("train", "validation", "test")):
        shard = cache / split
        shard.mkdir()
        physical = np.ones((2, 5, 46, 46), dtype=np.float32)
        physical[:, 4] = .5
        np.save(shard / "physical.npy", physical)
        np.save(shard / "valid.npy", np.ones((2, 46, 46), dtype=bool))
        np.savez(shard / "frames.npz", eligible=[True, True], kp=[1., 3.], source_frame=[0, 1], time=np.arange(2).astype("datetime64[us]"))
        rows.append(dict(split=split, shard=split, orbit=orbit, prepared_frames=2, used_frames=2, product={}))
    transform = ChannelTransform([1, 1], [0, 0], [1, 1], kind=ChannelTransform.LOG10_KIND)
    metadata = dict(cache_format=CACHE_FORMAT, transform=transform.to_dict(), orbits=rows, seed=1,
        used_frames=dict(train=2, validation=2, test=2), maximum_missing_fraction=.95,
        automatic_masks={}, grid_id="synthetic", grid={}, split_files={name:[] for name in ("train", "validation", "test")})
    (cache / "manifest.json").write_text(json.dumps(metadata))
    np.save(cache / "training_support.npy", np.ones((46, 46), dtype=bool))

    def skip_scientific_training(model, *args, **kwargs):
        assert kwargs["loss_configuration"]["coverage_weighting"]["training_frames_per_bin"] == [2]
        assert kwargs["latent_samples"] == 3
        model.best_validation_epoch = None
        model.best_validation_total = None
        return model, []

    monkeypatch.setattr(trainer, "train_vae", skip_scientific_training)
    output = tmp_path / "run"
    args = parse_arguments([str(cache), "--output", str(output), "--device", "cpu", "--num-workers", "0", "--batch-size", "2", "--latent-dim", "2", "--latent-samples", "3", "--spline-control-points", "5", "--coverage-weighting", "--coverage-balance", balance_mode])
    summary = trainer.run(args)
    loaded, _, _ = load_run_artifact(output / "automatic_cvae.pt")
    saved = json.loads((output / "coverage_weighting.json").read_text())
    assert saved["balance_mode"] == balance_mode
    assert saved == loaded.loss_configuration["coverage_weighting"]
    assert (output / "coverage_weighting.npz").exists()
    assert summary["training_configuration"]["latent_samples"] == 3
    assert summary["held_out_evaluation"]["test"]["reconstruction"] >= 0
