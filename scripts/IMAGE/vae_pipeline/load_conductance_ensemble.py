"""Interactive example: saved paired H/P members, ensemble moments and CS grid.

Run from the repository root in the tConductance environment, or run the #%%
sections in an editor. Change run_directory to inspect another native run.
All 50000 recovered float64 fields occupy about 1.7 GB; covariance uses only
learned-support cells. Saved float32 anomalies recover members up to roundoff.
"""

#%% Imports and run selection

from pathlib import Path
import hashlib
import json

import numpy as np

from icreader import load
from icanalyzer.wic_si12.workflow import load_run_artifact


run_directory = Path("~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid").expanduser()
ensemble_directory = run_directory              # Or one sample_conductance_at_kp.py output subfolder


#%% Recover the paired physical members from their saved anomaly representation

def recover_members(saved_mean, anomaly_factor, support):
    """Return (member, channel, row, column), channels Hall then Pedersen in S."""
    count = anomaly_factor.shape[1]
    if count < 2 or anomaly_factor.shape[0] != saved_mean.size or saved_mean.shape != (2, *support.shape):
        raise ValueError("saved mean, paired anomaly factor and support do not align")
    # Factor columns equal (member - saved_mean) / sqrt(count - 1).
    fields = np.array(anomaly_factor.T, dtype=np.float64, order="C", copy=True)
    fields *= np.sqrt(count - 1)
    fields += saved_mean.ravel()[None]
    samples = fields.reshape(count, *saved_mean.shape)
    samples[:, :, ~support] = np.nan
    return samples


def joint_moments(samples, support, block_size=2048):
    """Mean maps and unbiased covariance; supported Hall cells precede Pedersen."""
    count = len(samples)
    if count < 2 or not support.any() or samples.shape[1:] != (2, *support.shape):
        raise ValueError("need at least two paired fields on nonempty aligned support")
    state_flat_indices = np.flatnonzero(np.broadcast_to(support, samples.shape[1:]).ravel())
    mean = np.full(samples.shape[1:], np.nan)
    mean[:, support] = samples[:, :, support].mean(axis=0, dtype=np.float64)
    state_mean = mean.ravel()[state_flat_indices]
    covariance = np.zeros((len(state_mean), len(state_mean)), dtype=np.float64)
    # A few thousand members at a time keep the covariance working array small.
    for start in range(0, count, block_size):
        fields = samples[start:start + block_size].reshape(-1, mean.size)[:, state_flat_indices]
        if not np.isfinite(fields).all():
            raise ValueError("nonfinite sample on learned support")
        anomalies = fields - state_mean
        covariance += anomalies.T @ anomalies
    covariance /= count - 1
    return mean, covariance, state_flat_indices


#%% Load once, calculate moments, and load the verified matching grid

if __name__ == "__main__":
    model, transform, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
    if artifact["artifact_type"] != "conductance_vae_run_v1":
        raise ValueError("this example requires a native Hall/Pedersen conductance run")
    metadata_path = ensemble_directory / "sampling.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text())
        if metadata["model_sha256"] != hashlib.sha256((run_directory / "conductance_vae.pt").read_bytes()).hexdigest():
            raise ValueError("sample directory belongs to a different saved model")
    with np.load(ensemble_directory / "ensemble_products.npz") as saved:
        saved_mean = saved["conductance_mean"]
        anomaly_factor = saved["conductance_anomaly_factor"]
        support = saved["conductance_supported"].astype(bool)
        sample_kp = saved["conductance_sample_kp"] if "conductance_sample_kp" in saved.files else None
    samples = recover_members(saved_mean, anomaly_factor, support)
    del anomaly_factor
    Hall = samples[:, 0]                         # (50000, 46, 46), S
    Pedersen = samples[:, 1]                     # Same member order as Hall
    print(f"Loaded {len(samples)} paired members on {support.sum()} supported cells", flush=True)

    mean, covariance, state_flat_indices = joint_moments(samples, support)
    Hall_mean, Pedersen_mean = mean
    cells = int(support.sum())
    covariance_HH = covariance[:cells, :cells]    # S²
    covariance_HP = covariance[:cells, cells:]    # H/P cross covariance, S²
    covariance_PP = covariance[cells:, cells:]   # S²

    # The artifact stores MLAT/MLT. The source file also supplies the CSgrid object.
    mlat = np.asarray(artifact["data_provenance"]["grid"]["mlat"])
    mlt = np.asarray(artifact["data_provenance"]["grid"]["mlt"])
    with load(artifact["split_files"]["train"][0]) as source:
        grid = source.grid
        np.testing.assert_allclose(source.mlat, mlat)
        np.testing.assert_allclose(source.mlt, mlt)
        if source.grid_id != artifact["data_provenance"]["grid_id"]:
            raise ValueError("artifact and source grid IDs differ")
    np.testing.assert_array_equal(support, artifact["data_provenance"]["training_support"])
    assert grid.shape == support.shape

    # Map covariance rows back to spatial coordinates in the same state order.
    channel_index, row_index, column_index = np.unravel_index(state_flat_indices, mean.shape)
    state_mlat, state_mlt = mlat[row_index, column_index], mlt[row_index, column_index]
    print(f"Mean shape {mean.shape}; joint covariance shape {covariance.shape}; grid shape {grid.shape}")
    print("Use Hall[member], Pedersen[member], Hall_mean, Pedersen_mean and grid interactively.")
