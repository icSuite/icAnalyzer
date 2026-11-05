"""Descriptive event-location and reconstruction ranks from saved diagnostics."""

#%% Inputs and training-only latent geometry

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


def empirical_percentile(reference, value):
    return float(100 * np.mean(reference <= value))


def nearest_other_orbit(query, train, train_orbits, query_orbits=None):
    """Avoid adjacent images from the same training orbit defining its neighbors."""
    result = np.empty(len(query))
    for start in range(0, len(query), 128):
        stop = min(start + 128, len(query))
        distances = cdist(query[start:stop], train)
        if query_orbits is not None:
            distances[query_orbits[start:stop, None] == train_orbits[None]] = np.inf
        result[start:stop] = distances.min(axis=1)
    return result


def run(arguments):
    with np.load(arguments.prior / "train_latent.npz") as saved:
        train_mu = saved["mu"].astype(float)
        train_orbits = saved["orbit"]
    with np.load(arguments.prior / "test_latent.npz") as saved:
        test_mu = saved["mu"].astype(float)
        test_orbits = saved["orbit"]
    center = train_mu.mean(axis=0)
    covariance = np.cov(train_mu, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues.min() <= 0:
        raise ValueError("training-center covariance must be positive definite")
    whitening = eigenvectors / np.sqrt(eigenvalues)
    train_white = (train_mu - center) @ whitening
    test_white = (test_mu - center) @ whitening
    train_distance2 = np.sum(train_white**2, axis=1)
    test_distance2 = np.sum(test_white**2, axis=1)
    train_nearest = nearest_other_orbit(train_white, train_white, train_orbits, train_orbits)
    test_nearest = nearest_other_orbit(test_white, train_white, train_orbits)

    # Deterministic decoder(mu) reconstruction on the same observed supported cells.
    with np.load(arguments.prior / "inference.npz") as saved:
        observed, reconstruction, valid = saved["observed"], saved["posterior_mean"], saved["valid"]
        np.testing.assert_array_equal(saved["frame_orbit"], test_orbits)
        np.testing.assert_allclose(saved["mu"], test_mu, atol=1e-7)
    squared_error = np.where(valid, (reconstruction - observed)**2, 0)
    rmse = np.sqrt(squared_error.sum(axis=(2, 3)) / valid.sum(axis=(2, 3)))
    log_error = np.where(valid, (np.log10(reconstruction + 1) - np.log10(observed + 1))**2, 0)
    log_rmse = np.sqrt(log_error.sum(axis=(2, 3)) / valid.sum(axis=(2, 3)))
    selections = json.loads((arguments.events / "event_covariance_test.json").read_text())["events"]
    rows = []
    for event in selections:
        index, orbit = event["frame_index"], event["orbit"]
        if int(test_orbits[index]) != orbit:
            raise ValueError("event index and cached test orbit differ")
        with np.load(arguments.events / f"event_{orbit:04d}.npz") as saved:
            np.testing.assert_allclose(saved["mu"], test_mu[index], rtol=0, atol=1e-6)
        rows.append({
            "orbit": orbit, "frame_index": index,
            "latent_distance2": float(test_distance2[index]),
            "latent_distance_train_percentile": empirical_percentile(train_distance2, test_distance2[index]),
            "latent_distance_test_percentile": empirical_percentile(test_distance2, test_distance2[index]),
            "nearest_other_orbit_train_percentile": empirical_percentile(train_nearest, test_nearest[index]),
            "decoder_mu_RMSE_S_Hall_Pedersen": rmse[index].tolist(),
            "reconstruction_test_percentile_Hall_Pedersen": [empirical_percentile(rmse[:, channel], rmse[index, channel]) for channel in (0, 1)],
            "log_reconstruction_test_percentile_Hall_Pedersen": [empirical_percentile(log_rmse[:, channel], log_rmse[index, channel]) for channel in (0, 1)]
        })
    result = {
        "training_frames": len(train_mu), "training_contributing_orbits": int(len(np.unique(train_orbits))),
        "test_frames": len(test_mu), "test_orbits": int(len(np.unique(test_orbits))),
        "training_mu_covariance_eigenrange": eigenvalues[[0, -1]].tolist(), "events": rows,
        "source_cache": str(arguments.prior), "event_cache": str(arguments.events),
        "interpretation": [
            "Training-only frame-weighted moments whiten encoder means; squared distance is descriptive, not latent density or an OOD probability.",
            "Training nearest-neighbor reference excludes the entire query orbit; test orbits were never used in training.",
            "Reconstruction percentiles compare decoder(mu) physical or log errors against all retained test images; lower is better.",
            "Correlated frames, finite training coverage and encoder information loss limit outlier conclusions. No formal rejection threshold adopted.",
            "Uses unchanged saved diagnostic arrays; does not retrain a conditional model or add images to the training population."
        ]
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args())
