# Loading Saved Conductance Ensembles

Last reviewed: 2026-10-12
Status: Full hybrid ensemble and matching grid loaded; moments verified.

## Selected run and entry point

User requests using `conductance_smoothed_b16_holes_hybrid`, the largest native
selection among the compared annotated/CSV/hybrid runs. Saved summary confirms
6881/1203/1117 train/validation/test images, beta1.6/latent36, selected epoch207.
Support2077 cells on46x46;39 unavailable. This is an analysis-target choice,
not scientific acceptance of its uncertainty. Earlier conditional correlation
and outlier results concern `_annotated`, not this run.

`scripts/IMAGE/vae_pipeline/load_conductance_ensemble.py` is the scientist-facing
interactive example. README owns the command. It defaults to `_hybrid`; edit
`run_directory` for another native H/P model. It loads the saved model artifact
and `ensemble_products.npz`, recovers the paired physical members, computes
sample moments, and reads the matching source grid using icReader. No source
files, weights, model outputs or annotations are modified.

## Array conventions

Saved mean M has shape(2,46,46), normalized anomalies A have shape(4232,50000).
State order is flattened Hall followed by flattened Pedersen, C order.
Recover member i as `M.ravel() + sqrt(N-1)*A[:,i]`; reshape to(2,46,46).
This preserves actual paired members up to float32 anomaly-storage roundoff.
Full recovered float64 array shape(50000,2,46,46) occupies about1.7GB.

Example globals: `samples`, `Hall`, `Pedersen`, `mean`, `Hall_mean`,
`Pedersen_mean`, `covariance`, its HH/HP/PP blocks, `support`,
`state_flat_indices`, `model`, `transform`, `artifact`, `grid`, `mlat`, `mlt`.
Mean maps have NaN outside support. Covariance excludes these unavailable
locations; joint shape4154x4154, units S². Supported Hall states precede
supported Pedersen states, with row-major cell order. State channel/row/column
and magnetic coordinates are exposed for mapping covariance rows to locations.
The HP block retains cross-channel dependence.

Artifact stores MLAT/MLT arrays. icReader reconstructs the complete CSgrid
from a saved source NetCDF; grid ID and coordinates are checked against the
artifact. Grid is native magnetic geometry, not an event-specific geographic
grid ready for Lompe without coordinate conversion.

## Verification

Executed the example on all50000 saved hybrid members. Mean differs from saved
M by at most9.07e-8 S (factor-storage roundoff); covariance is symmetric to
printed precision, trace19125.2034 S², no zero-variance supported states.
A12-state block matches direct `numpy.cov` at1e-12 tolerance. Source grid
ID `image_apex_130km_46x46_v1`, shape and all MLAT/MLT coordinates match.
One focused synthetic test verifies sample recovery, support exclusion, state
ordering, unbiased normalization and a known H/P cross covariance.

Small verification report:
`figures/validation/native_hybrid_ensemble_loading_20261012/verification.json`.
Full samples and covariance are calculated in memory; no duplicate large
artifacts are stored by this example. Scientific checks have now been rerun
and are owned by [[Hybrid Conductance VAE Validation]].
