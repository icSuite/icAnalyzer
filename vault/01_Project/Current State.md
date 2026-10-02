# Current State

Last reviewed: 2026-10-02
Repository snapshot: `vae/wic-si12` implementation at `209e9c6`; `main` remains at `7c71591`
Worktree: clean before this memory checkpoint

## Current position

The canonical development path is now a paired corrected-WIC/SI12 VAE on the
46x46 Cubed-Sphere Product-2 grid. Upstream Product 2 proton-corrects WIC with
the Hardy energy model. Corrected WIC is then propagated at fixed 2 keV
electron energy to Hall and Pedersen conductance; it is never proton corrected
again in icAnalyzer.

The repository's prior mixed state was preserved in three commits on
`archive/pre-wic-si12-vae-2026-09-25`. The dedicated branch starts from that
checkpoint. `main` and `origin/main` remain untouched at `7c71591`.

## Implemented baseline

Commit `5f6de08` adds:

- `icanalyzer.wic_si12.data`: strict live Product-2 contract validation,
  eager field/source/grid materialization, orbit splits, channel transforms,
  variance propagation, masks, quality weighting, and explicit unusable-frame
  accounting. Manual good intervals are applied through `wic_source_index`
  before the camera arrays are materialized;
- `icanalyzer.wic_si12.model`: the VAE implementations and consistently
  reduced reconstruction/KL loss;
- `icanalyzer.wic_si12.physics`: the reviewed fixed-2-keV local adapter with a
  frozen WIC-response fixture and no second proton correction;
- `icanalyzer.wic_si12.ensemble`: exact-count latent sampling, low-rank joint
  covariance factors, covariance blocks, and ensemble-space sparse updates;
- `icanalyzer.wic_si12.workflow`: reproducible training, held-out evaluation,
  and complete save/reload artifacts; and
- `scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`: the canonical linear
  workflow from orbit discovery through sparse reconstruction.

The sparse validation supports Hall-only, Pedersen-only, and joint direct
observations with explicit observation-error covariance. Detector uncertainty
weights reconstruction but is not silently mixed into the latent geophysical
covariance.

## Restored legacy-lineage architecture

Commit `e923875` corrects the architectural divergence in the first rewrite.
The canonical runner now uses:

- a three-layer partial-convolution encoder with propagated validity masks;
- corrected WIC and SI12 as the only encoder science channels; the continuous
  quality weight remains out of the encoder for this experiment;
- a 30-by-30 tensor-product cubic-spline coefficient grid projected to the
  46-by-46 Product-2 grid;
- a training-only standardized `log10(value + 1)` transform;
- latent dimension 36, batch size 8, learning rate `1e-4`, constant `beta=1`,
  seed 1, and 1,000 epochs by default; and
- `ReduceLROnPlateau` with factor 0.5 and patience 10, monitoring training
  total loss as in the old script and recording learning rate in every epoch;
  validation loss independently selects the saved checkpoint.

The 46-by-46 grid's `xi` and `eta` axes are uniformly spaced with equal extent,
so the normalized tensor-product basis is equivalent to evaluating the old
affine-scaled CS-coordinate basis. Tests verify basis partition of unity,
constant-field reproduction, mask invariance, saved-artifact reload, and the
scheduler. The prior convolutional artifacts remain readable but are not the
canonical architecture.

### Legacy-objective fidelity gap

A direct code audit on 2026-10-02 found that the restored architecture still
does not reproduce the legacy objective. The legacy script multiplies its
weighted reconstruction ratio by `36*36*2` and sums KL across the batch. The
current implementation averages reconstruction across sample/channel and
averages KL across the batch. With batch size 8 and otherwise uniform weights,
`beta=1` therefore gives KL 648 times more influence relative to reconstruction
than the legacy script.

This is not only theoretical. The saved `wic_si12_spline_v1` checkpoint chose
epoch 73 with validation reconstruction `1.5538` and KL `0.0010`. The near-zero
KL shows posterior collapse: the encoded posterior is effectively the unit
Gaussian and the decoder cannot carry useful frame-specific information.
This objective mismatch is the leading explanation for the degraded samples
and supersedes the earlier plan to sweep around `beta=1`.

Other code-level departures remain: the legacy quality map is the third
encoder channel; the current experiment deliberately omits it; the current
loss adds uncertainty precision, inverted background-fit quality, and channel
coverage; sparse frames are normalized per sample/channel instead of
contributing in proportion to global valid support; the current workflow uses
orbit-held-out training and the best validation checkpoint; and the 46x46
model retains 30x30 spline coefficients versus 30x30 on the legacy 36x36
grid. These need controlled ablations after repairing the objective scale.

Finite negative WIC and SI12 inputs are clipped to zero before the log
transform. The runner warns and saves finite, technically valid, and clipped counts
as upstream-problem provenance. The inverse transform clips decoded physical
WIC and SI12 below zero to zero.

## Verification

- 51 tests under `tests/` pass in the current environment.
- All 61 Python files under `icanalyzer/`, `scripts/`, and `tests/` parse.
- A fresh two-epoch run on real orbits 85, 86, and 89 verified the restored
  model through best-checkpoint selection, artifact reload, eight latent
  samples, propagation, low-rank covariance, sparse reconstruction, and all
  twelve figures. Reusing the spline artifact and a separate SZA-105 run also
  completed.
- A three-orbit fixture generated by the live icBuilder Product-2/CS code was
  read through live icReader and completed one VAE epoch, held-out evaluation,
  artifact save/reload, eight paired latent samples, fixed-energy propagation,
  joint covariance construction, and sparse reconstruction.
- Repeating that CPU smoke with the same seed reproduced its metrics exactly.
- After the Hardy correction was confirmed, three real production orbits
  (`1720`, `0813`, and `1734`) completed a one-epoch end-to-end smoke with 30
  usable frames, held-out evaluation, artifact reload, 16 latent samples,
  fixed-2-keV electron propagation, and joint sparse reconstruction.
- The archived manual-selection semantics were ported and verified on orbits
  85, 86, and 89. Their inclusive source intervals selected 121 frames; 120
  had valid weighted support and completed an end-to-end smoke.
- The live upstream focused suites passed during contract review: 7 icBuilder
  detector-CS tests, 26 icReader detector-first tests, 17 icPhysics tests, and
  30 relevant icBuilder physics-path tests.

The smoke runs are deliberately tiny. The one-epoch real model showed strongly
collapsed prior-sample variance, so its numerical errors do not measure
scientific skill.

## Historical first full annotated baseline

The user completed a 20-epoch CUDA run on the default manual selection and
1,000 prior samples. All 157 available annotated orbits were split by orbit:
111 train, 23 validation, and 23 test. After invalid-support filtering, the
splits contained 7,556, 1,365, and 807 frames respectively.

Verified diagnostics:

- validation total loss reached its minimum at epoch 10 (`0.2352`) and rose to
  `0.2412` by epoch 20 while train loss continued down to `0.2025`; this is
  mild overfitting and the saved epoch-20 state is not the best checkpoint;
- held-out test total loss was `0.3007` with reconstruction loss `0.2600`;
- sampled corrected-WIC mean was close to observed (`129.3` versus `124.3`
  counts), while sampled standard deviation was low (`149.2` versus `305.6`);
- sampled SI12 mean/std were `2.94/4.53` counts versus observed `3.72/7.65`,
  also showing underdispersion;
- 50 joint sparse observations reduced RMSE from `1.400` S to `0.587` S and
  reduced posterior standard deviation everywhere in this example; and
- the physical-space Gaussian update produced a negative posterior minimum
  (`-0.213` S), so positivity must be handled explicitly before treating the
  reconstruction as physical.

This run used the simplified convolutional model, signed-asinh transform,
reversed background-fit weight, and pre-revision loss. It is retained only as
historical execution evidence and is not a valid result for the restored
architecture or current scientific target.

## Reproducible diagnostics

Commit `1504b1e` adds a headless visualization workflow for both the existing
full run and future artifacts. It reconstructs the saved ensembles exactly
from their low-rank anomaly factors and produces decoded FUV samples,
propagated conductance samples, moments, joint-channel correlation maps,
leading covariance modes, covariance spectra, loss history, and sparse-update
summaries without constructing a dense 4232-by-4232 covariance.

The existing full run produced ten figures. Its leading four covariance modes
account for 55.1% of FUV variance and 53.6% of conductance variance. The
maximum-variance anchors are CS `(row, column) = (10, 13)` for corrected WIC
and `(27, 7)` for Hall. These are descriptive diagnostics of this run, not yet
validated physical conclusions.

New runs additionally retain the sampled latent vectors, sparse-validation
truth, observation indices and values, prior uncertainty, and truth-frame
identity. They restore the minimum-validation-loss model before evaluation,
sampling, and artifact saving. A three-orbit, two-epoch smoke verified all of
these fields and generated twelve figures, including latent-draw and
truth/error panels. Older runs remain visualizable, but missing latent draws
and validation truth cannot be recovered after the fact.

### Large-ensemble diagnostics

Commit `209e9c6` resolves the diagnostic crash on the existing 50,000-sample
archive. The former visualizer eagerly retained both full reconstructed
ensembles and both float64 anomaly factors, then formed a
50,000-by-50,000 Gram matrix requiring 18.6 GiB before eigensolver workspace.

The visualizer now reconstructs only requested mosaic members, calculates
variance without a full squared-factor temporary, processes FUV and
conductance sequentially, and uses deterministic randomized SVD when neither
covariance orientation is small. On the real 3.1-GiB archive, all twelve
figures completed in 31.18 seconds with 2,308,412 KiB peak resident memory
using eight retained spectrum modes. Those eight modes captured 63.10% of FUV
variance and 62.69% of conductance variance.

New runs preallocate batch-decoded samples, avoid concatenating a second full
decoded ensemble, omit unused propagation intermediates, and save float32
latent samples and anomaly factors. A fresh real-orbit smoke verified float32
`(4232, 8)` FUV and conductance factors and generated all figures. Sampling,
physical propagation, and factor storage still grow linearly with ensemble
size: unlike the old coefficient-space workflow, the current path has not yet
made a 500,000-sample run workstation-safe.

## Experimental SZA selection

Commit `52817c6` adds the optional `--minimum-sza` input threshold while
preserving the current CSV/source-index frame selection. The filter is
inclusive and pixel-level: finite cells with `SZA >= threshold` remain
eligible, lower or nonfinite-SZA cells become invalid with zero loss weight,
and frames with no remaining weighted pixels are dropped. The unfiltered
baseline remains the default when the option is omitted.

For `--minimum-sza 105`, a full read-only corpus audit found that 6,701 of the
10,680 CSV-selected frames retain weighted support. Across those selected
frames, 11.8% of spatial grid cells remain. A three-orbit, two-epoch
end-to-end smoke completed and recorded `105.0` in both data provenance and
training configuration. This verifies mechanics, not model quality.

Commit `6e8a3db` adds the explicit `--no-csv-selection` alias for the existing
`--all-frames` bypass and makes SZA-only loading practical at the file-read
boundary: frames with no qualifying SZA cell are removed before the six camera
fields are materialized. A three-orbit SZA-only smoke completed end to end.

Across the complete 1,378-file corpus, `SZA >= 105°` alone retains 261,773 of
358,439 frames (73.0%) and 19.6% of all grid cells. There are 193 orbit files
with no qualifying frame. This experiment is therefore far larger than the
CSV-selected run even though its pixel support is sparse; training time and
memory should be monitored during the first full attempt.

A complete per-location audit on 2026-09-29 exposed a stronger limitation:
only 1,483 of 2,116 grid locations have any `SZA >= 105°` target in the whole
corpus. Columns 33 through 45 have zero eligible targets in all 358,439 frames;
column 32 has only 2,047 eligible cell-times. The mask-aware loss gives these
permanently unsupported outputs zero direct reconstruction gradient, but the
decoder and prior sampler still emit full 46x46 fields. Current ensemble and
conductance statistics also retain those outputs. They are structured neural-
network extrapolation from shared parameters, not learned local distributions,
and must not be interpreted as measured global covariance.

The first full launch exposed an empty-index edge case in the 193 orbit files
with no qualifying SZA frame: icReader returned `(0,1,1)` camera fields for an
empty selection. Commit `9fd9957` fixes this by skipping those files before
camera reads and recording them by split as `sza_excluded_files`. A real-file
check with empty orbit 113 and a six-orbit end-to-end smoke passed; the focused
suite now has 31 tests.

The cleaner full-good-only experiment is recorded as deferred: completed JSON
annotations provide about 1,378 fully good source frames across 34 orbits,
without requiring conversion of the 36x36 partial-frame masks.

The reason for that proposed comparison is concrete: among the 42 completed
JSON annotations that overlap the old CSV selection, the CSV selects 1,286
fully good, 2,822 partial, and 50 bad frames under the newer classification.
The current VAE runner does not apply the JSON keyframe masks, so partial-frame
bad regions remain eligible except where upstream validity and coverage happen
to remove them. The SZA experiment intentionally leaves this selection method
unchanged so its effect can be isolated.

## Paired legacy/rerun comparison

Commit `c6bedbf` adds a dedicated adapter for the legacy
`precipitation_IR_hardy_weighted` products without changing icReader,
icBuilder, or icPhysics. Product 2 remains the authority for manual orbit
selection and SZA. Current and legacy frames are paired only when the orbit
filename and product timestamp match exactly. The same Product-2 SZA mask is
then applied to both versions. The upstream legacy `w` and current
`method_quality_weight` measure background-fit conformity, not scientific
confidence. Legacy training therefore uses `1 - w`; current training uses
`(1 - method_quality_weight)` multiplied by channel coverage.

The canonical VAE runner accepts `--legacy-comparison` and
`--data-version current|legacy`. Both matched runs derive their orbit split
from the same common current-file list and seed. A separate streaming script,
`compare_legacy_current_inputs.py`, reports value, uncertainty, quality, valid
support, correlation, RMSE, and mean-difference statistics and writes spatial
mean/difference maps one orbit at a time.

The copied corpora contain 1,371 common orbit filenames and 326,739 exact
common product timestamps. With SZA-only `SZA >= 105°`, 243,282 common frames
retain at least one eligible cell. An orbit-85 check retained 112 paired
timestamps; both the current and legacy sides completed one epoch, artifact
reload, latent sampling, propagation, covariance construction, and sparse
reconstruction. This is execution evidence only. The direct comparison is a
net pipeline comparison, not an isolated fuvpy experiment, because matching,
coregistration, detector-to-CS reduction, uncertainty, and validity handling
also changed.

## Absolute measurement-uncertainty loss

Commit `23afaeb` implements the first agreed uncertainty revision. `dwic` and
`dsi12` are propagated through the fitted channel transform as before, but
their precision is no longer normalized independently within every image.
The training split now fits one median transformed-space precision scale for
WIC and one for SI12. The loss divides weighted squared error by valid support
per sample/channel rather than by the weight sum, preserving absolute
uncertainty and quality differences between frames.

The transformed variance floor defaults to `1e-3`, relative precision is
capped at `100`, and at most two million candidate cells per channel are used
for deterministic scale fitting. Fitted scales, sampling diagnostics, caps,
and loss reduction are stored under `uncertainty_weighting` in the run
artifact. Training now requires an explicit uncertainty-loss configuration so
an old artifact cannot be silently evaluated with new loss semantics.

Focused tests verify that fourfold variance gives one-quarter reconstruction
weight, fitted channel scales balance their typical precision, and uniform
quality differences are no longer cancelled. One-orbit current and legacy
smokes and a three-orbit train/validation/test smoke completed end to end. Raw
losses are not comparable with older runs or directly between current and
legacy runs fitted with different transforms/scales. The restored `beta=1`
default reproduces the old architecture experiment but has not been calibrated
for the revised loss.

## Corrected background-fit weight orientation

Commit `2e5ff4d` corrects a scientifically reversed loss weight. Upstream zero
means that the background method could not fit the data; those cells commonly
contain aurora. Upstream one means that the background fit explains the data.
icAnalyzer now converts both current and legacy upstream weights to the
auroral-interest orientation with `1 - w` during ingestion, before applying
channel coverage and measurement precision.

A deterministic 64-file Product-2 audit covered 35,337,200 cells. All 138,394
exact zero-weight cells had `method_valid=True`, so the existing validity mask
does not remove them and the inversion restores their loss contribution. The
focused suite has 40 passing tests, all 61 Python files parse, a two-orbit
direct comparison completed, and matched three-orbit current and legacy runs
completed training, reload, latent sampling, conductance propagation,
covariance construction, and sparse reconstruction.

All real-data VAE artifacts produced before `2e5ff4d` used the opposite
quality orientation and are not scientifically valid auroral-model results;
they must be retrained. Existing physical-field mean maps from the direct
legacy/rerun comparison are unaffected, but its quality-weight statistics
must be regenerated.

## Nonnegative boundaries and old-artifact recovery

New log-transform models clamp both decoded physical channels to zero during
inverse transformation. Commit `ad44da0` remains relevant only to older
signed-asinh convolutional artifacts: their roundoff-scale negative WIC is
corrected with explicit diagnostics while materially negative WIC still fails.
`--reuse-artifact` supports both model versions and validates the saved split
and scientific selection before rerunning downstream stages.

## Production corpus and blockers

- The coordinated product changes are committed as icBuilder `38b5afb` and
  icReader `3a3b73f` on their `modular_pipeline` branches.
- The copied corpus under `precipitation_cs/image_apex_130km_46x46_v1/IR_hardy`
  contains 1,378 files, 358,439 frames, and about 31 GiB. All headers expose
  the six required camera fields with the expected shape and count units.
- Every file declares `proton_energy_model="hardy"`. The user confirmed on
  2026-09-28 that this is the intended upstream correction. The icAnalyzer
  loader now requires Hardy provenance; fixed 2 keV applies only to downstream
  electron propagation.
- Across all 157 manually selected files, corrected WIC had no negative finite
  cells. SI12 had 3,482,898 negative cells among 14,030,013 finite cells
  (24.82%). With `SZA >= 105`, 390,848 of 2,718,083 technically valid SI12 cells
  remained negative (14.38%). This is a substantial upstream issue; the VAE
  now clips and reports it rather than treating negative counts as valid.
- The segment CSV contains 173 accepted orbit intervals. The copied corpus has
  157 of those orbits, mapping to 10,680 Product-2 frames; 9,728 frames have
  valid weighted support. The selected core arrays are estimated at about 0.7
  GiB before transformed training arrays and model state.
- No icPhysics source was changed. Existing `hall()` and `ped()` broadcast over
  ensembles, but WIC response currently comes from private `fWm`. The desired
  public, grid-independent batched propagation boundary remains open and must
  be coordinated with icBuilder compatibility tests before implementation.
- The annotation document remains valid with 53 orbit records, 49 complete
  and four in progress, and 469 keyframes. The recorded source data directory
  containing the old 36x36 grid is absent. Coordinate-based reprojection is
  therefore blocked; index scaling is not an acceptable substitute.

## Next actionable work

1. Trace and correct the upstream source of the widespread negative SI12
   counts. Until then, treat zero-clipped VAE runs as controlled experiments,
   not final scientific products.
2. If 500,000 samples are scientifically required, redesign sampling,
   propagation, and covariance storage around coefficient-space or streamed
   statistics; the 50,000-sample visualizer is now verified, but full factors
   still scale linearly with ensemble size.
3. Add a legacy-compatible objective experiment on the current data and test
   an effective beta near `1e-3`; do not treat `beta=1` as legacy-equivalent.
   Compare reconstruction, KL, held-out physical residuals, and prior-sample
   dispersion before changing the encoder or data selection.
4. Then ablate uncertainty precision, per-frame/channel loss reduction, and
   the quality-map encoder channel separately so their effects are identifiable.
5. Regenerate the full legacy/rerun quality comparison and retrain any VAE
   intended for interpretation; pre-`2e5ff4d` artifacts used reversed weights.
6. Before running the full paired SZA-only VAEs, decide whether the state is
   the explicitly supported nightside domain or whether the SZA threshold must
   change. Mask permanently unsupported decoder output out of sampling,
   propagation, covariance, visualization, and reconstruction if the
   nightside-only formulation is retained.
7. After the revised loss is evaluated, test uncertainty maps as auxiliary
   encoder inputs. Separately preserve latent/background covariance and
   measurement/observation covariance as named downstream quantities. Do not
   make `dwic` and `dsi12` decoder outputs.
8. Evaluate prior dispersion and sparse reconstruction across many held-out
   frames, seeds, and Hall-only/Pedersen-only/joint geometries.
9. Choose an explicit positivity-preserving reconstruction state or method;
   do not silently clip negative posterior conductance.
10. Decide whether early stopping is useful now that the best validation state
   is restored automatically.
11. Recover the legacy annotation source grid and implement the previously
   agreed non-destructive coordinate conversion before applying annotations.
12. Decide separately whether to authorize a public icPhysics propagation API.

The December 2026 paper target remains the latest recorded publication target.
The AGU 2026 abstract outcome, target journal, co-authors, and review standard
remain unknown.
