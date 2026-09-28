# Handoff - Latest

Last updated: 2026-09-28
Repository snapshot: `vae/wic-si12` at `61acb57`; `main` remains at `7c71591`

## Outcome

The first coherent WIC/SI12-to-conductance VAE baseline is implemented and
committed. The old mixed worktree was preserved on
`archive/pre-wic-si12-vae-2026-09-25`; development now lives on
`vae/wic-si12`.

The canonical command is documented in
`scripts/IMAGE/vae_pipeline/README.md` and implemented by
`scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`.

The workflow:

```text
icreader.load(precipitation_cs)
  -> corrected-WIC/SI12 tensors, masks, uncertainty, grid, provenance
  -> orbit-held-out mask-aware VAE training and artifact reload
  -> paired prior latent samples and WIC/SI12 anomaly factor
  -> corrected WIC only, fixed 2 keV electron propagation
  -> joint Hall/Pedersen anomaly factor
  -> Hall-only, Pedersen-only, or joint sparse Gaussian update
```

Corrected WIC is already proton corrected upstream using the Hardy energy
model. icAnalyzer does not call a proton-correction routine and sampled SI12 is
not subtracted a second time. Downstream electron mean energy is fixed at 2
keV in this baseline.

## Evidence

- icAnalyzer commit: `5f6de08 Implement WIC SI12 VAE baseline`.
- Focused tests: 18 passed.
- Syntax: 56 Python files parsed.
- Live-contract smoke: three generated Product-2 orbits passed through current
  icBuilder and icReader code, one training epoch, held-out evaluation,
  artifact reload, eight latent samples, physical propagation, covariance,
  and sparse reconstruction.
- Repeated CPU smoke metrics were identical at the same seed.
- A real-data smoke on orbits 1720, 0813, and 1734 used all 30 frames and
  completed one training epoch, held-out evaluation, artifact reload, 16
  paired samples, conductance propagation, and joint sparse reconstruction.
- The canonical runner now applies archived manual good intervals by default,
  mapped through Product-2 `wic_source_index` before array reads. An annotated
  smoke on orbits 85, 86, and 89 selected 121 frames, dropped one frame with no
  valid support, and completed end to end.
- Upstream review found `wic_corrected`, `dwic_corrected`,
  `si13_corrected`, `dsi13_corrected`, `si12`, and `dsi12`, each in counts on
  `(time, 46, 46)`. The earlier synthetic contract fixture used SI12 and
  constant-2-keV proton provenance; the copied production files do not.
- Compatibility review found `hall()` and `ped()` already broadcast arbitrary
  leading dimensions and reproduced the frozen 2 keV reference fixture.

The smoke demonstrates execution and interface coherence only. Its one-epoch
prior samples had strongly collapsed variance and are not a scientific result.

## First full annotated run

The user ran 20 epochs on CUDA with latent dimension 32, beta 0.001, and 1,000
prior samples. The orbit split was 111/23/23, with 7,556/1,365/807 usable
train/validation/test frames.

Validation loss was best at epoch 10 (`0.2352`) rather than the saved epoch 20
(`0.2412`), while final train loss was `0.2025`. Prior samples remained
underdispersed: corrected-WIC standard deviation was 149 versus 306 observed,
and SI12 was 4.53 versus 7.65 counts. Means were closer, especially corrected
WIC (129 versus 124 counts).

For one held-out frame and 50 random joint observations, posterior RMSE fell
from 1.400 S to 0.587 S. This is encouraging but not yet a validation result.
The physical-space Gaussian posterior reached -0.213 S, revealing that sparse
reconstruction needs an explicit positivity-preserving state or method.

## Important boundaries

- The contract changes are committed as icBuilder `38b5afb` and icReader
  `3a3b73f`. The copied files report producer version `38b5afb+worktree`.
- No icPhysics code was changed. The local adapter freezes the current
  `fWm(2.0) == 477.5` response, then uses public `hall()` and `ped()`. Because
  `fWm` is private, the requested public grid-independent batched propagation
  boundary remains a separate approval and compatibility task.
- Propagation occurs after CS gridding. Existing `conductance_cs` instead
  averages detector-space nonlinear conductance, so equality between the two
  products is neither expected nor a valid regression criterion.
- Measurement uncertainty is used for loss weighting and observation error;
  latent covariance remains a separately named geophysical component.
- Generated corrected WIC is nonnegative by decoder construction. The workflow
  still rejects any negative physical sample rather than clipping silently.

## Production-corpus audit

The copied directory now contains 1,378 `precipitation_cs` files, 358,439
frames, and about 31 GiB. All files passed a header audit for product type,
schema, grid, dimensions, count units, and the six coordinated camera fields.

All 1,378 files declare `proton_energy_model="hardy"`. The user confirmed on
2026-09-28 that this is intentional. The loader contract now requires Hardy
upstream proton correction while retaining fixed 2 keV electron propagation
downstream.

A stratified 32-orbit value audit otherwise supports the implementation:
7,011 of 7,960 frames had jointly valid cells, corrected WIC was nonnegative,
uncertainties were nonnegative, and SI12 contained valid negative
background-subtracted counts requiring the signed transform.

The manual segment CSV contains 173 accepted orbit intervals; 157 are present
in the copied Product-2 corpus. They map to 10,680 frames, of which 9,728 have
valid weighted support. Sixteen accepted orbits have no copied Product-2 file.
The selected core arrays are estimated at about 0.7 GiB before transformed
training arrays and model state, making the annotated workflow practical on a
normal workstation.

Next:

1. restore/save the best validation checkpoint and add early stopping before
   extending training;
2. measure prior-sample calibration and covariance convergence across seeds;
3. evaluate many held-out fields and Hall-only, Pedersen-only, and joint
   realistic sparse geometries;
4. compare prior-mean and independent-channel baselines; and
5. implement a positivity-preserving reconstruction formulation without
   silent clipping.

Annotation conversion is also open. The annotation JSON contains 53 orbit
records and 469 valid keyframes, but the old 36x36 source grid files are absent
at their recorded path. Recover those coordinates before writing the agreed
non-destructive Cubed-Sphere converter; do not rescale pixel indices.

## Portfolio impact

- Central update needed: Yes
- Changes: The first full annotated baseline completed. It learns useful
  covariance and substantially improved one sparse reconstruction, but the
  best validation epoch was 10, samples were underdispersed, and the linear
  posterior allowed negative conductance. A
  public icPhysics propagation boundary and coordinate-based annotation
  conversion remain explicit follow-up decisions.
- Priority/deadline: unchanged. The December 2026 paper target remains the
  latest recorded target; the AGU abstract outcome is still unknown.
