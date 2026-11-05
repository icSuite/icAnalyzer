#%%

import numpy as np
import matplotlib.pyplot as plt
from lompe.utils.conductance import hardy
from polplot import Polarplot

from pathlib import Path
from icreader import load
import os
os.chdir('/home/bing/Dropbox/work/code/repos/icAnalyzer/')
from icanalyzer.wic_si12.workflow import load_run_artifact

#run_directory = Path("~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_smoothed_b16_holes_hybrid").expanduser()
#run_directory = Path("~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_automatic_cvae_b16_holes").expanduser()
#run_directory = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/cvae_runs/conductance_two_stage_valid5_20261015').expanduser()
run_directory = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/cvae_runs/conductance_two_stage_valid5_20261015_coverage_full').expanduser()
run_directory = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/cvae_runs/conductance_model2_fuvpy_bs2d_valid5').expanduser()
run_directory = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/cvae_runs/conductance_model2_fuvpy_bs2d_valid5_cHardy').expanduser()
run_directory = Path('/home/bing/Dropbox/work/code/repos/icAnalyzer/figures/cvae_runs/conductance_manual_fuvpy_bs2d_valid5_cHardy').expanduser()

output = Path('/home/bing/Dropbox/work/conference/Dynamit_2026/figures/automatic_m2_bs2d')
output = Path('/home/bing/Dropbox/work/conference/Dynamit_2026/figures/automatic_m2_bs2d_cHardy')
output = Path('/home/bing/Dropbox/work/conference/Dynamit_2026/figures/manual_m2_bs2d_cHardy')
output.mkdir(exist_ok=True)

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

try:
    _, _, artifact = load_run_artifact(run_directory / "conductance_vae.pt", device="cpu")
except:
    _, _, artifact = load_run_artifact(run_directory / "automatic_cvae.pt", device="cpu")
if artifact["artifact_type"] != "conductance_vae_run_v1":
    raise ValueError("this example requires a native Hall/Pedersen conductance run")

samples = []
for kp in range(7):
    with np.load(run_directory / f"conductance_kp_0_6/kp_{kp:02d}_{kp:01d}/ensemble_products.npz") as saved:
        saved_mean = saved["conductance_mean"]
        anomaly_factor = saved["conductance_anomaly_factor"]
        support = saved["conductance_supported"].astype(bool)
    samples.append(recover_members(saved_mean, anomaly_factor, support))
del anomaly_factor


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

#%%

MLT, MLAT = np.meshgrid(np.arange(0, 24 + .1, .1), np.arange(45, 90+.1, .1))

#%%

for q in [0.1, 0.5, 0.9]:

    fig, axs = plt.subplots(4, 7, figsize=(14, 8))
    paxs = np.array([Polarplot(ax) for ax in axs.flatten()]).reshape(axs.shape)
    
    vmin = 0
    
    for i, kp in enumerate(np.arange(0, 7)):
        hall = np.nanquantile(samples[i][:, 0, :, :], q, axis=0)
        pedersen = np.nanquantile(samples[i][:, 1, :, :], q, axis=0)
        vmaxh = np.nanmax(hall)
        vmaxp = np.nanmax(pedersen)
        paxs[0, i].contourf(mlat, mlt, hall, vmin=vmin, vmax=vmaxh)
        paxs[1, i].contourf(mlat, mlt, pedersen, vmin=vmin, vmax=vmaxp)
        axs[0, i].set_title(np.round(vmaxh, 2))
        axs[1, i].set_title(np.round(vmaxp, 2))    
        
        hall, pedersen = hardy(MLAT, MLT, kp)
        vmaxh = np.nanmax(hall)
        vmaxp = np.nanmax(pedersen)
        paxs[2, i].contourf(MLAT, MLT, hall, vmin=vmin, vmax=vmaxh)
        paxs[3, i].contourf(MLAT, MLT, pedersen, vmin=vmin, vmax=vmaxp)
        axs[2, i].set_title(np.round(vmaxh, 2))
        axs[3, i].set_title(np.round(vmaxp, 2))
        
    output_file = output / f'q_{int(q*100)}.png'
    plt.savefig(output_file, bbox_inches='tight', dpi=150)
    plt.close('all')
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    