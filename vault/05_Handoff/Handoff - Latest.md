# Handoff - Latest

Last updated: 2026-10-02
Repository snapshot: `vae/wic-si12` implementation at `9a723b3`; `main` remains at `7c71591`

## Outcome

The WIC/SI12-to-conductance pipeline now uses the restored legacy-lineage
spline/partial-convolution VAE. The old mixed worktree was preserved on
`archive/pre-wic-si12-vae-2026-09-25`; development now lives on
`vae/wic-si12`.

The canonical command is documented in
`scripts/IMAGE/vae_pipeline/README.md` and implemented by
`scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`.

Saved-run diagnostics are implemented by
`scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py`. The script reads an
existing run without retraining and writes loss, sample, moment, correlation,
mode, spectrum, and sparse-reconstruction figures plus `diagnostics.json`.

The workflow:

```text
icreader.load(precipitation_cs)
  -> corrected-WIC/SI12 tensors, masks, uncertainty, grid, provenance
  -> zero clipping and training-only standardized log10(value + 1)
  -> orbit-held-out partial-convolution/spline VAE and artifact reload
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

- Architecture restoration: `e923875 Restore spline partial-convolution VAE`.
  The canonical model uses three propagated-mask partial convolutions, a
  30-by-30 tensor-product cubic-spline decoder on the 46-by-46 grid, and the
  old defaults: latent 36, batch 8, learning rate `1e-4`, `beta=1`, seed 1,
  and 1,000 epochs. `ReduceLROnPlateau` uses training total loss, factor 0.5,
  and patience 10; validation independently selects the saved checkpoint, and
  the visualizer plots the recorded learning-rate history.
- Legacy-semantics restoration: `fcf4c90` restores the literal global weighted
  reconstruction reduction and summed KL, the corrected common weight as the
  third encoder channel, and the 33% common-missing frame limit. Measurement
  precision is absent from this baseline loss. Latent dimension 36 and the
  30x30 degree-3 spline remain unchanged. SI13 validity no longer removes
  WIC/SI12 training pixels. Follow-up `9a723b3` measures the 33% limit before
  optional SZA masking, as in the legacy preprocessing order.
- Verification after `9a723b3`: 54 tests pass, including a literal legacy-loss
  formula regression and the exact 33% cutoff boundary. A real one-epoch
  Product-2 smoke completed artifact reload, sampling, propagation, covariance,
  and sparse reconstruction with reconstruction `2115.04` and KL `3.92` rather
  than collapsed KL. Across the full annotated split, the cutoff retains 4,048
  train, 488 validation, and 1,102 test frames. A separate real three-orbit
  `SZA >= 105°` smoke also completed with the intrinsic 33% cutoff.
- The old `wic_si12_spline_v1` artifact remains invalid: it selected epoch 73
  with validation KL `0.0010` under the nonlegacy loss and is posterior-collapsed.
- New models clip finite negative WIC/SI12 inputs before log transformation,
  emit a warning, and save clipped/finite/technically-valid counts. The complete
  manual-selection audit found zero negative WIC cells and 3,482,898 negative
  SI12 cells among 14,030,013 finite cells (24.82%). With `SZA >= 105`,
  390,848 of 2,718,083 technically valid SI12 cells were negative (14.38%). This is
  an unresolved upstream processing issue.
- Current verification: all 53 tests under `tests/` pass and all Python files
  parse. The new real-data smoke and full-split selection audit supersede the
  earlier execution-only checks for the canonical loss.
- Large-ensemble repair: `209e9c6` makes diagnostics selective and sequential,
  removes squared-factor temporaries, and uses deterministic randomized SVD
  instead of a sample-by-sample Gram matrix. All twelve figures from the real
  50,000-sample, 3.1-GiB archive completed in 31.18 seconds at 2,308,412 KiB
  peak resident memory with eight spectrum modes; they captured 63.10% of FUV
  and 62.69% of conductance variance. New runs preallocate batched decoded
  output, omit unused propagation intermediates, and retain float32 factors.
  Sampling, propagation, and saved factors remain linear in sample count, so
  this does not yet make 500,000 full image-space samples workstation-safe.
- Nonnegative-boundary recovery: `ad44da0 Recover VAE runs from WIC boundary
  roundoff`. The reported 0.003% negative sample fraction was 66 values, all
  exactly `-1.2899104e-6` counts, caused by float32 inversion of the
  mathematical zero boundary. The corrected sampler maps only values within
  an explicit roundoff tolerance to zero and records diagnostics; substantive
  negatives still fail. This path now applies to old signed-asinh artifacts;
  new log-transform artifacts clamp both decoded physical channels to zero.
- `--reuse-artifact` now validates split, seed, latent dimension, SZA/manual
  selection, data version, legacy directory, and corrected quality-weight
  provenance before skipping training. A reused-artifact end-to-end smoke and
  a fresh-training smoke both passed. The focused suite now has 42 tests.
- Weight-orientation correction: `2e5ff4d Invert background fit weights for
  VAE`. Current Product 2 now loads
  `(1 - method_quality_weight) * channel_coverage`; the legacy adapter loads
  `1 - w`. Both transformations and the upstream background-fit meaning are
  preserved in product metadata, run artifacts, and comparison reports.
- In a deterministic 64-file Product-2 sample, all 138,394 exact zero-weight
  cells had `method_valid=True`. The validity mask therefore preserves these
  likely-auroral pixels and the inversion restores their training weight.
- Verification after the correction: 40 focused tests, 61 parsed Python
  files, a two-orbit direct comparison, and matched three-orbit current and
  legacy end-to-end smokes through training, reload, latent sampling,
  propagation, covariance construction, and sparse reconstruction.
- All real-data VAE artifacts predating `2e5ff4d` used the reversed weight
  orientation and require retraining. Physical-field mean maps in the earlier
  direct comparison are unchanged; quality-weight statistics require rerun.
- Superseded uncertainty experiment: `23afaeb` added precision weighting and a
  nonlegacy loss reduction. Commit `fcf4c90` removes both from the canonical
  baseline. The helper remains available only for a future explicit ablation;
  no current baseline artifact should claim uncertainty-weighted training.
- Full SZA-support audit: across all 1,378 current files and 358,439 frames,
  only 1,483 of 2,116 grid cells ever satisfy `SZA >= 105°`. Columns 33--45
  have zero eligible targets; column 32 has only 2,047 cell-times. The current
  decoder still emits all 46x46 cells and ensemble statistics retain them, so
  the unsupported region is neural-network extrapolation rather than learned
  local covariance.
- Paired comparison commit: `c6bedbf Add paired legacy rerun comparison`.
  The legacy adapter uses Product-2 selection/SZA, exact common timestamps, and
  now-inverted legacy combined weight `1 - w`. The runner trains `current` or
  `legacy` on the same common orbit split. The streaming comparison emits JSON
  field/support/weight statistics and spatial mean/difference maps.
- Corpus identity audit: 1,371 common orbit filenames and 326,739 exact common
  timestamps; 243,282 common frames retain at least one `SZA >= 105°` cell.
- Real orbit-85 verification retained 112 exact timestamps. Both sides
  completed independent one-epoch VAE runs through artifact reload, sampling,
  conductance propagation, covariance, and sparse reconstruction. The direct
  report and figure were also generated. The focused suite has 34 passing
  tests and 61 Python files parse.
- Empty-orbit fix: `9fd9957 Skip SZA-empty orbit files`. The first full launch
  found that an empty indexed icReader camera read had shape `(0,1,1)` rather
  than `(0,46,46)`. Such orbit files are now skipped before camera reads and
  recorded by split in provenance. A real orbit-113 check, a six-orbit
  end-to-end smoke containing that empty orbit, and all 31 tests passed.
- SZA-only commit: `6e8a3db Support scalable SZA-only VAE selection`.
- `--no-csv-selection --minimum-sza 105` bypasses the CSV while retaining
  technical validity checks. SZA-empty frames are now discarded before camera
  fields are read. The full corpus retains 261,773 of 358,439 frames (73.0%),
  19.6% of grid cells, and excludes all frames from 193 files. A three-orbit
  SZA-only run completed end to end; the full focused suite has 30 tests.
- SZA-filter commit: `52817c6 Add configurable SZA filtering to VAE input`.
- `--minimum-sza 105` retains finite pixels at `SZA >= 105°`; 6,701 of
  10,680 CSV-selected frames retain weighted support and 11.8% of spatial
  cells remain over the selected corpus. A three-orbit end-to-end smoke
  completed and preserved the threshold in configuration and provenance.
- The current runner still does not apply `partial_image_annotations.json`
  masks. In the 42 completed annotations overlapping the CSV selection, 2,872
  of 4,158 selected frames are partial or bad under the newer classification.
  A full-good-only baseline is recorded as a deferred comparison; the user
  chose to isolate the SZA-filter effect first.
- The earlier CSV-plus-SZA checkpoint had 29 passing tests.
- Diagnostics commit: `1504b1e Add WIC SI12 VAE diagnostics`.
- Syntax: 59 Python files parsed at the diagnostics checkpoint.
- The existing 1,000-sample full run generated ten figures without a dense
  state covariance. Its leading four modes explain 55.1% of FUV variance and
  53.6% of conductance variance.
- A fresh three-orbit, two-epoch smoke saved latent vectors, sparse truth and
  observations, restored and recorded the best validation checkpoint, and
  generated all twelve available figures. Visual inspection caught and fixed
  nonfinite-pixel error scaling before the checkpoint.
- icAnalyzer commit: `5f6de08 Implement WIC SI12 VAE baseline`.
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

## Historical first full annotated run

The user ran 20 epochs on CUDA with latent dimension 32, beta 0.001, and 1,000
prior samples. The orbit split was 111/23/23, with 7,556/1,365/807 usable
train/validation/test frames.

Validation loss was best at epoch 10 (`0.2352`) rather than the saved epoch 20
(`0.2412`), while final train loss was `0.2025`. That historical artifact still
contains the epoch-20 model; current training restores the best validation
state before evaluation, sampling, and saving. Prior samples remained
underdispersed: corrected-WIC standard deviation was 149 versus 306 observed,
and SI12 was 4.53 versus 7.65 counts. Means were closer, especially corrected
WIC (129 versus 124 counts).

For one held-out frame and 50 random joint observations, posterior RMSE fell
from 1.400 S to 0.587 S. This is encouraging but not yet a validation result.
The physical-space Gaussian posterior reached -0.213 S, revealing that sparse
reconstruction needs an explicit positivity-preserving state or method.

This artifact used the simplified convolutional architecture, signed-asinh
transform, reversed quality weight, and pre-revision loss. It is not a valid
comparison result for the restored model.

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
- Measurement uncertainty remains available for observation-error models but
  is not used by the restored legacy training loss; latent covariance remains
  a separately named geophysical component.
- New log-transform artifacts clip decoded WIC and SI12 at physical zero.
  Input clipping is warned and recorded. Old signed-asinh artifacts retain the
  narrow WIC roundoff recovery and reject substantive negative WIC.

## Production-corpus audit

The copied directory now contains 1,378 `precipitation_cs` files, 358,439
frames, and about 31 GiB. All files passed a header audit for product type,
schema, grid, dimensions, count units, and the six coordinated camera fields.

All 1,378 files declare `proton_energy_model="hardy"`. The user confirmed on
2026-09-28 that this is intentional. The loader contract now requires Hardy
upstream proton correction while retaining fixed 2 keV electron propagation
downstream.

The complete manual-selection value audit supersedes the earlier stratified
sample for sign policy: WIC has no negative finite cells, while 24.82% of
finite SI12 cells are negative. These values are now clipped and reported;
their upstream cause remains unresolved.

The manual segment CSV contains 173 accepted orbit intervals; 157 are present
in the copied Product-2 corpus. They map to 10,680 frames, of which 9,728 have
valid weighted support. Sixteen accepted orbits have no copied Product-2 file.
The selected core arrays are estimated at about 0.7 GiB before transformed
training arrays and model state, making the annotated workflow practical on a
normal workstation.

Next:

1. identify and fix the upstream source of negative SI12; treat zero-clipped
   runs as experiments until that is resolved;
2. if 500,000 samples are required, redesign the remaining linear-size
   sampling, propagation, and image-space factor storage around streaming or
   the spline coefficient representation; 50,000-sample plotting is repaired;
3. train the restored legacy-semantics baseline on current data and compare KL,
   held-out physical residuals, and sample dispersion with the legacy run;
4. choose a scientifically explicit nightside support/domain or revise the SZA
   threshold; if retaining `SZA >= 105°`, exclude permanently unsupported
   cells from ensemble statistics, physical propagation, visualization, and
   reconstruction before full paired training;
5. then test uncertainty maps as auxiliary encoder inputs and implement clear
   downstream separation of latent/background covariance from measurement/
   observation covariance; do not decode uncertainty fields;
6. measure prior-sample calibration and covariance convergence across seeds;
7. evaluate many held-out fields and Hall-only, Pedersen-only, and joint
   realistic sparse geometries;
8. compare prior-mean and independent-channel baselines;
9. implement a positivity-preserving reconstruction formulation without
   silent clipping.

Annotation conversion is also open. The annotation JSON contains 53 orbit
records and 469 valid keyframes, but the old 36x36 source grid files are absent
at their recorded path. Recover those coordinates before writing the agreed
non-destructive Cubed-Sphere converter; do not rescale pixel indices.

## Portfolio impact

- Central update needed: Yes
- Changes: The legacy-lineage spline/partial-convolution VAE is restored and
  verified end to end. A complete audit exposed negative SI12 in 24.82% of
  finite manually selected cells; icAnalyzer now clips and reports it, but the
  upstream cause requires investigation. The controlled legacy/rerun
  comparison is executable, but the full
  SZA-only audit shows that columns 33--45 have no training targets while the
  current VAE and covariance path still emits and uses them. A supported-domain
  decision and implementation must precede scientific interpretation or full
  paired SZA-only training. The VAE baseline now restores the legacy objective,
  quality encoder channel, and frame sparsity limit while preserving corrected
  weight orientation and Product-2 fields. A fresh full training comparison is
  now the next model step. Upstream background-fit
  weights are now correctly inverted for auroral-interest training; all
  pre-correction real-data VAE artifacts require retraining. The
  public icPhysics propagation boundary and coordinate-based annotation
  conversion remain explicit follow-up decisions.
- Priority/deadline: unchanged. The December 2026 paper target remains the
  latest recorded target; the AGU abstract outcome is still unknown.
