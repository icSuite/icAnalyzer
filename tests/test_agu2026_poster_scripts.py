import argparse
import importlib.util
from pathlib import Path

import numpy as np

from icanalyzer.wic_si12.ensemble import ensemble_statistics


ROOT = Path(__file__).parents[1]


def load_script(name):
    path = ROOT / "scripts" / "AGU2026" / name
    spec = importlib.util.spec_from_file_location(name.replace(".py", ""), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


VALIDATION = load_script("01_model_validation.py")
COVARIANCE = load_script("02_conductance_covariance.py")
SPARSE = load_script("03_sparse_reconstruction.py")


def synthetic_ensemble(path):
    generator = np.random.default_rng(4)
    samples = generator.normal(size=(12, 2, 3, 4)).astype(np.float32)
    samples[:, 0] += 5
    samples[:, 1] += 3
    statistics = ensemble_statistics(samples)
    np.savez_compressed(
        path,
        conductance_mean=statistics["mean"],
        conductance_anomaly_factor=statistics["anomaly_factor"],
        conductance_supported=np.ones((3, 4), dtype=bool),
        conductance_sample_kp=np.full(12, 2, dtype=np.float32),
        mlat=np.linspace(50, 90, 12).reshape(3, 4),
        mlt=np.linspace(0, 24, 12, endpoint=False).reshape(3, 4),
    )


def test_validation_metrics_are_exact_for_exact_reconstruction():
    observed = np.arange(24, dtype=float).reshape(1, 2, 3, 4)
    valid = np.ones_like(observed, dtype=bool)
    metrics = VALIDATION.frame_metrics(observed, observed, valid)
    assert metrics[0][0]["rmse"] == 0
    assert metrics[0][1]["nrmse"] == 0
    assert metrics[0][0]["correlation"] == 1


def test_validation_selects_ten_distinct_performance_quantiles():
    records = [
        [{"nrmse": float(value)}, {"nrmse": float(value + 1)}]
        for value in range(20)
    ]
    selected, scores, quantiles = VALIDATION.select_representatives(records)
    assert len(np.unique(selected)) == 10
    np.testing.assert_allclose(quantiles, np.arange(0.05, 1, 0.1))
    assert np.all(np.diff(scores[selected]) > 0)


def test_validation_representatives_respect_coverage_eligibility():
    records = [
        [{"nrmse": float(value)}, {"nrmse": float(value + 1)}]
        for value in range(30)
    ]
    eligible = np.arange(30) >= 10
    selected, _scores, _quantiles = VALIDATION.select_representatives(
        records, eligible=eligible
    )
    assert eligible[selected].all()
    assert selected.min() >= 10


def test_covariance_then_sparse_reconstruction_smoke(tmp_path):
    ensemble = tmp_path / "ensemble_products.npz"
    synthetic_ensemble(ensemble)
    covariance_output = tmp_path / "covariance"
    report = COVARIANCE.run(argparse.Namespace(
        ensemble=ensemble, output=covariance_output, modes=4, plot_modes=2,
        grid_id=None, workspace=None, teleconnection_batch_size=3,
        power_iterations=1, seed=4, dpi=40,
    ))
    assert report["sample_count"] == 12
    assert (covariance_output / "02_conductance_covariance.npz").exists()
    assert (covariance_output / "02_joint_pca_modes.png").exists()

    sparse_output = tmp_path / "sparse"
    sparse_report = SPARSE.run(argparse.Namespace(
        covariance=covariance_output / "02_conductance_covariance.npz",
        ensemble=ensemble, output=sparse_output, observation_counts=[1, 3],
        observation_std=0.2, geometry="random", truth_members=3, modes=4,
        workspace=None, seed=4, dpi=40,
    ))
    assert sparse_report["truth_independent_of_covariance_archive"] is False
    assert len(sparse_report["metrics"]) == 4
    assert (sparse_output / "03_sparse_reconstruction_example.png").exists()
