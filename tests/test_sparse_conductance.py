import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from icanalyzer.wic_si12.ensemble import ensemble_statistics


SCRIPT = Path(__file__).parents[1] / "scripts/IMAGE/vae_pipeline/test_sparse_conductance.py"
SPEC = importlib.util.spec_from_file_location("sparse_conductance", SCRIPT)
SPARSE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SPARSE)


def configuration(state_space="physical", mode="joint"):
    return {"state_space": state_space, "observation_mode": mode, "observation_std": 0.1}


def test_random_sites_are_reproducible_unique_supported_and_leave_truth():
    valid = np.ones((4, 4), dtype=bool)
    valid[0] = False
    first = SPARSE.sparse_sites(valid, 4, np.random.default_rng(4), "random")
    second = SPARSE.sparse_sites(valid, 4, np.random.default_rng(4), "random")
    np.testing.assert_array_equal(first, second)
    assert len(np.unique(first)) == 4
    assert valid.flat[first].all()
    assert not np.array_equal(first, SPARSE.sparse_sites(valid, 4, np.random.default_rng(5), "random"))
    with pytest.raises(ValueError, match="unobserved"):
        SPARSE.sparse_sites(valid, 12, np.random.default_rng(4), "random")


def test_track_stays_in_one_column_when_there_are_enough_sites():
    sites = SPARSE.sparse_sites(np.ones((8, 8), dtype=bool), 6, np.random.default_rng(0), "track")
    assert len(np.unique(sites % 8)) == 1


@pytest.mark.parametrize("mode,channels", [("joint", [0, 1]), ("hall", [0]), ("pedersen", [1])])
def test_eigenmode_update_matches_dense_legacy_oi_and_adds_noise(mode, channels):
    generator = np.random.default_rng(3)
    mean = np.ones((2, 2, 2))
    factor = generator.normal(size=(8, 3)) * 0.2
    truth = mean + generator.normal(size=mean.shape) * 0.3
    sites = np.array([0, 2])
    noise = np.array([[0.1, -0.1], [0.2, -0.2]])
    scores, analysis, prior, values = SPARSE.evaluate_case(mean, factor, truth, np.ones((2, 2), bool), sites, noise, configuration(mode=mode))
    indices = np.concatenate([sites + channel * 4 for channel in channels])
    covariance = factor @ factor.T
    expected_values = truth.reshape(-1)[indices] + np.concatenate([noise[channel] for channel in channels])
    expected = mean.reshape(-1) + covariance[:, indices] @ np.linalg.solve(covariance[np.ix_(indices, indices)] + np.eye(len(indices)) * 0.01, expected_values - mean.reshape(-1)[indices])
    np.testing.assert_allclose(analysis.reshape(-1), expected)
    np.testing.assert_allclose(values, expected_values)
    np.testing.assert_allclose(prior, mean)
    assert [score["count"] for score in scores] == [2, 2]


def test_scoring_excludes_both_fields_at_observed_sites_and_natural_gaps():
    mean = np.ones((2, 2, 2))
    truth = mean.copy()
    truth[:, 0, 0] = 1000
    truth[:, 1, 1] = np.nan
    valid = np.ones((2, 2), bool)
    valid[1, 1] = False
    scores, _, _, _ = SPARSE.evaluate_case(mean, np.zeros((8, 1)), truth, valid, np.array([0]), np.zeros((2, 1)), configuration(mode="hall"))
    assert [score["count"] for score in scores] == [2, 2]
    assert [score["sse"] for score in scores] == [0, 0]


def test_log_transform_is_not_a_physical_covariance_transform():
    physical = np.arange(8).reshape(2, 2, 2) + 1.0
    mean = np.log10(physical + 1)
    factor = np.ones((8, 1)) * 0.05
    scores, analysis, prior, values = SPARSE.evaluate_case(mean, factor, physical, np.ones((2, 2), bool), np.array([0]), np.zeros((2, 1)), configuration("log10"))
    np.testing.assert_allclose(analysis, physical)
    np.testing.assert_allclose(prior, physical)
    np.testing.assert_allclose(values, np.log10(physical[:, 0, 0] + 1))
    assert all(score["covered"] == 3 for score in scores)


def test_negative_posterior_is_reported_not_clipped():
    mean = np.ones((2, 2, 2)) * 0.1
    scores, analysis, _, _ = SPARSE.evaluate_case(mean, np.ones((8, 1)), mean, np.ones((2, 2), bool), np.array([0]), np.ones((2, 1)) * -1, configuration())
    assert np.all(analysis < 0)
    assert all(score["negative"] == 3 for score in scores)


def test_prior_log_modes_match_transformed_sample_covariance(tmp_path):
    samples = np.random.default_rng(4).lognormal(size=(20, 2, 2, 2)).astype("float32")
    statistics = ensemble_statistics(samples)
    np.savez(tmp_path / "ensemble_products.npz", conductance_mean=statistics["mean"], conductance_anomaly_factor=statistics["anomaly_factor"])
    mean, factor, information = SPARSE.prior_modes(tmp_path, "log10", 8, tmp_path)
    log_samples = np.log10(samples + 1)
    np.testing.assert_allclose(mean, log_samples.mean(axis=0), atol=1e-7)
    np.testing.assert_allclose(factor @ factor.T, np.cov(log_samples.reshape(20, -1).T), atol=1e-7)
    assert information["retained_modes"] == 8
    assert information["captured_variance_fraction"] == pytest.approx(1, abs=1e-6)


def synthetic_benchmark():
    protocol = {
        **configuration(), "split": "validation", "seed": 0, "geometry": "random",
        "observation_counts": [1, 2], "epsilon_siemens": 1.0,
        "split_orbits": {"train": ["or_1.nc"], "validation": ["or_2.nc"]},
        "selection": {"minimum_sza_degrees": None}, "propagation": "fixed"
    }
    return {
        "truth": np.ones((2, 2, 2, 2)), "valid": np.ones((2, 2, 2), bool),
        "sites": np.array([[0, 1], [1, 2]]), "noise": np.zeros((2, 2, 2)),
        "orbit": np.array([2, 2]), "time": np.array(["a", "b"]),
        "mlat": np.ones((2, 2)), "mlt": np.ones((2, 2)),
        "configuration": np.array(json.dumps(protocol))
    }


def test_benchmark_validation_and_aggregate_scores():
    benchmark = synthetic_benchmark()
    SPARSE.check_benchmark(benchmark)
    metrics, records, examples = SPARSE.score_model(np.zeros((2, 2, 2)), np.zeros((8, 1)), benchmark)
    assert len(metrics) == 4 and len(records) == 4 and len(examples) == 2
    assert [entry["scored_cells"] for entry in metrics] == [6, 6, 4, 4]
    assert all(entry["rmse_siemens"] == 1 for entry in metrics)
    benchmark["sites"][0] = [0, 0]
    with pytest.raises(ValueError, match="unique"):
        SPARSE.check_benchmark(benchmark)


def test_comparability_rejects_split_leakage_grid_or_selection_changes():
    benchmark = synthetic_benchmark()
    artifact = {
        "split_files": {"train": ["/a/or_1.nc"], "validation": ["/a/or_2.nc"]},
        "training_configuration": {"minimum_sza_degrees": None},
        "data_provenance": {"grid": {"mlat": benchmark["mlat"], "mlt": benchmark["mlt"]}, "electron_energy_keV": 2.0}
    }
    SPARSE.check_comparable(artifact, benchmark, {"propagation_order": "fixed"})
    artifact["split_files"]["train"].append("/a/or_2.nc")
    with pytest.raises(ValueError, match="orbit splits"):
        SPARSE.check_comparable(artifact, benchmark, {"propagation_order": "fixed"})
    artifact["split_files"]["train"].pop()
    artifact["training_configuration"]["minimum_sza_degrees"] = 105
    with pytest.raises(ValueError, match="selection"):
        SPARSE.check_comparable(artifact, benchmark, {"propagation_order": "fixed"})


def test_cli_defaults_and_reuse_fixture_plotting(tmp_path):
    arguments = SPARSE.parse_arguments(["example", "--output", str(tmp_path)])
    assert arguments.modes == 100 and arguments.state_space == "log10"
    assert arguments.observation_counts == [10, 30, 100]
    benchmark = synthetic_benchmark()
    np.savez(tmp_path / "benchmark.npz", **benchmark)
    with np.load(tmp_path / "benchmark.npz") as archive:
        reloaded = {key: archive[key] for key in archive.files}
    SPARSE.check_benchmark(reloaded)
    metrics, _, examples = SPARSE.score_model(np.ones((2, 2, 2)), np.ones((8, 1)) * 0.01, reloaded)
    SPARSE.plot_examples(reloaded, examples, tmp_path, "example")
    SPARSE.plot_skill({"example": {"metrics": metrics}}, tmp_path)
    assert (tmp_path / "sparse_skill.png").exists()
    assert len(list(tmp_path.glob("example_example_*.png"))) == 2


def test_reuse_runs_without_instantiating_a_decoder_and_preserves_fixture(tmp_path):
    benchmark = synthetic_benchmark()
    fixture = tmp_path / "benchmark.npz"
    np.savez(fixture, **benchmark)
    frozen_bytes = fixture.read_bytes()
    model_run = tmp_path / "future_model"
    model_run.mkdir()
    artifact = {
        "model_configuration": {"model": "not_a_supported_vae_architecture"},
        "split_files": {"train": ["/a/or_1.nc"], "validation": ["/a/or_2.nc"]},
        "training_configuration": {"minimum_sza_degrees": None},
        "data_provenance": {"grid": {"mlat": benchmark["mlat"], "mlt": benchmark["mlt"]}, "electron_energy_keV": 2.0}
    }
    torch.save(artifact, model_run / "wic_si12_vae.pt")
    (model_run / "summary.json").write_text(json.dumps({"propagation_order": "fixed"}))
    samples = np.random.default_rng(7).lognormal(size=(10, 2, 2, 2)).astype("float32")
    statistics = ensemble_statistics(samples)
    np.savez(model_run / "ensemble_products.npz", conductance_mean=statistics["mean"], conductance_anomaly_factor=statistics["anomaly_factor"])
    arguments = SPARSE.parse_arguments([str(model_run), "--benchmark", str(fixture), "--output", str(tmp_path / "output"), "--modes", "4"])
    report = SPARSE.run(arguments)
    assert report["frame_count"] == 2
    assert report["protocol"]["state_space"] == "physical"  # Fixture wins over log10 CLI default.
    assert report["results"]["future_model"]["modes"]["retained_modes"] == 4
    assert fixture.read_bytes() == frozen_bytes
    assert (arguments.output / "metrics.json").exists()
    with pytest.raises(FileExistsError, match="preserved"):
        SPARSE.run(arguments)
