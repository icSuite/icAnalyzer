# WIC/SI12 VAE Pipeline Plan

Last reviewed: 2026-09-29
Status: End-to-end baseline, saved-run diagnostics, and paired legacy/rerun
comparison implemented; full comparison runs and publication validation remain

## Implementation checkpoint

Implemented on `vae/wic-si12` at `5f6de08`:

- strict `precipitation_cs` ingestion for corrected WIC, corrected SI13,
  SI12, their uncertainties, support, source identity, grid, and provenance;
- orbit-level train/validation/test splitting and explicit reporting of frames
  excluded for having no valid weighted pixels;
- training-only signed-asinh channel transforms and propagated transformed
  measurement variance;
- a mask-aware 46x46 convolutional VAE with consistently per-sample reduced
  reconstruction and KL losses;
- an architectural nonnegative corrected-WIC decoder constraint;
- reloadable run artifacts containing model, transform, histories, held-out
  metrics, split files, coordinates, provenance, optimizer settings, and seed;
- reproducible paired latent sampling and low-rank physical-count covariance;
- a local fixed-2-keV electron-propagation adapter that never repeats proton
  correction and preserves ensemble dimensions;
- joint Hall/Pedersen anomaly factors and ensemble-space sparse reconstruction
  with Hall-only, Pedersen-only, or joint observation selection; and
- a canonical command-line workflow and focused regression tests.

Verification on 2026-09-25: 18 focused tests passed; 56 Python files parsed;
and a generated three-orbit Product-2 fixture ran through live icBuilder and
icReader code, one training epoch, artifact reload, eight latent samples,
conductance propagation, and sparse reconstruction. Repeating the CPU smoke
with the same seed reproduced the metrics exactly. This proves interface and
execution coherence, not model skill; the one-epoch fixture scores have no
scientific interpretation.

On 2026-09-28, after the user confirmed Hardy upstream proton correction, a
real-data smoke used orbits 1720, 0813, and 1734. All 30 frames were usable;
the workflow trained for one epoch, reloaded the artifact, drew 16 paired
samples in uneven batches, propagated conductance, and completed a joint
sparse update. The generated variance was strongly collapsed, as expected for
this minimal run, so the result validates execution rather than model skill.

The archive's `good_orbit_segments.csv` selection is now part of the canonical
runner. Inclusive manual intervals are mapped through each file's
`wic_source_index` before large variables are read. In the copied corpus, 157
of 173 annotated orbits are present, yielding 10,680 selected frames and 9,728
frames with valid weighted support. A three-orbit annotation smoke selected
the exact intended source bounds and completed end to end.

The first full annotated run used 20 epochs, latent dimension 32, beta 0.001,
and 1,000 prior samples on CUDA. Validation loss was best at epoch 10 and then
rose slightly while train loss kept falling. The epoch-20 prior matched the
corrected-WIC mean reasonably but underestimated WIC and SI12 dispersion. One
50-observation joint reconstruction reduced RMSE from 1.400 S to 0.587 S, but
the unconstrained physical-space update produced values down to -0.213 S.
The implementation now restores the validation-best checkpoint. Before
treating reconstructions as physical, define a positivity-preserving
formulation. Evaluate reconstruction over many held-out frames rather than
interpreting this single example.

Commit `1504b1e` completes the initial analysis layer. The runner now restores
the lowest-validation-loss state and future artifacts retain latent draws,
sparse truth and observations, prior uncertainty, and truth-frame identity.
The standalone headless visualizer produces decoded and propagated sample
mosaics, latent diagnostics when stored, ensemble moments, joint-channel
correlation maps, low-rank covariance modes and spectra, and sparse prior /
posterior / truth panels. It remains compatible with the first full run, while
explicitly reporting that its unsaved latent draws and truth are unavailable.
The validation-best restoration is therefore complete; early stopping remains
an optional efficiency improvement rather than a correctness blocker.

Commit `c6bedbf` implements the legacy/rerun comparison requested after the
SZA work. It intersects same-named orbit files, aligns exact product times,
and applies Product-2 manual selection and SZA to both sides. The legacy
adapter maps `wic_corrected`, `si12`, their uncertainties, and combined robust
weight `w` into the VAE contract. The current path keeps its existing validity,
coverage, and method-quality semantics. The VAE runner can train either side
on the same orbit split; a streaming direct comparison reports fields,
uncertainties, quality weights, support, and spatial mean/difference maps.
Treat its result as a net old-versus-new product-pipeline change, not as an
isolated fuvpy effect.

Still open:

- Deferred clean-selection experiment proposed on 2026-09-29: use only the
  `full_good_start:full_good_stop` ranges from completed
  `partial_image_annotations.json` records. This avoids applying unreprojected
  36x36 masks and would provide about 1,378 source frames across 34 orbits.
  The user initially chose an exposed SZA threshold with the current CSV
  selection, then requested an SZA-only experiment without the CSV.

- The SZA experiment was extended to permit SZA-only selection. Use
  `--no-csv-selection --minimum-sza 105`; ordinary finite-value, uncertainty,
  method-validity, coverage, and positive-weight checks still apply. Empty-SZA
  frames are removed before camera fields are read. The full corpus retains
  261,773 frames and is therefore a substantially larger training experiment
  than the manually selected baseline.

  A later full-grid audit found that this is not a globally supported state:
  1,483 of 2,116 locations are ever eligible, columns 33--45 are never eligible,
  and column 32 has only 2,047 eligible cell-times in 358,439 frames. The
  current full-grid decoder emits finite values there through shared learned
  parameters, but no local target constrains them. Do not use those outputs in
  covariance or propagation. Choose an explicit supported-domain state or
  revise the SZA selection before the full paired VAE experiment.

- The Product-2 changes have landed as icBuilder `38b5afb` and icReader
  `3a3b73f`. A copied production corpus contains 1,378 files and 358,439
  frames. The user confirmed on 2026-09-28 that its Hardy proton-energy model
  is intentional. The strict loader now requires that provenance.
- The fixed-energy adapter must currently use private `icphysics.image.fWm`
  for the WIC response. No icPhysics code was changed. A public,
  grid-independent batched boundary remains a separately approved upstream
  task after compatibility review.
- The legacy 36x36 files that define the annotation source grid are not
  available at their recorded local path. Reprojecting the 469 masks by pixel
  scaling would violate the coordinate-independent annotation decision, so
  conversion remains blocked until the source cubed-sphere coordinates can be
  recovered.
- Prior-sample calibration, covariance convergence, observation geometries,
  independent-channel baselines, and publication-scale held-out results remain
  to be run on the copied corpus.

Production-corpus audit on 2026-09-28: all 1,378 headers had the six VAE
fields with the expected dimensions and count units. A stratified 32-orbit
sample contained 7,960 frames, of which 7,011 had at least one jointly valid
cell. Corrected WIC was nonnegative; SI12 legitimately included negative
background-subtracted counts, supporting the signed transform.

## Objective

Build a reproducible pipeline that:

1. trains a joint VAE on paired IMAGE WIC and SI12 global fields;
2. samples the latent distribution and analyzes WIC/SI12 spatial covariance;
3. propagates the corrected-WIC component of each paired sample nonlinearly to
   Hall and Pedersen conductance;
4. estimates the joint Hall/Pedersen covariance; and
5. reconstructs conductance from sparse Hall and/or Pedersen observations.

The first milestone is an unconditional baseline on one clearly versioned
dataset. Conditional generation and publication-scale evaluation follow only
after that baseline is verified.

## Confirmed baseline assumptions

- Upstream proton correction uses the Hardy proton-energy model recorded in
  Product 2.
- Mean electron energy is spatially and temporally constant at 2 keV during
  downstream propagation.
- Proton and electron energies are not sampled in the first ensemble. The
  resulting covariance measures variation learned from paired corrected-WIC/
  SI12 fields after deterministic Hardy proton correction and under fixed
  downstream electron energy.
- Detector measurement uncertainty remains separate from latent geophysical
  variability. Use it in data weighting and reconstruction observation error;
  do not silently add it to the learned prior covariance.
- WIC and SI12 samples remain paired through every processing stage.
- The first sparse reconstruction uses the propagated conductance ensemble,
  not latent optimization through the VAE.

The fixed 2 keV electron assumption is a deliberate baseline simplification,
not a claim that electron precipitation energy is constant. Electron-energy
models and physical-model uncertainty can be introduced later as controlled
sensitivity experiments.

## Repository and API boundaries

```text
fuvpy
  -> icBuilder fuv_detector
  -> icBuilder precipitation_detector (Product 2)
  -> icBuilder precipitation_cs
  -> icReader.load()
  -> icAnalyzer corrected-WIC/SI12 VAE
  -> reviewed fixed-E0 propagation at E0 = 2 keV
  -> Hall/Pedersen ensemble
  -> joint covariance and sparse reconstruction
```

`icAnalyzer` must not import `fuvpy` or `icBuilder`. `icBuilder` owns camera
preprocessing, coregistration, geometry, gridding, and product provenance.
`icReader` owns product loading and schema validation. `icPhysics` owns the
array-to-array physical conversion.

## Authoritative VAE input product

The user has coordinated an upstream Product-2 contract change in icBuilder.
Product 2 will include:

- proton-corrected WIC and its uncertainty;
- proton-corrected SI13 and its uncertainty;
- SI12 and its uncertainty; and
- the existing precipitation fields, support, grid, and provenance.

These observation fields will be carried into `precipitation_cs`, which is the
authoritative fixed-grid product ingested by icAnalyzer through
`icreader.load()`. A separate `fuv_cs` product is not required for this plan.

This is a confirmed interface decision but remains pending live-code and
schema verification after the icBuilder and icReader changes land. The reader
must expose the new Product-2 CS fields without icAnalyzer importing icBuilder.

The VAE channels are specifically `wic_corrected` and `si12`. Use those names
in code and artifacts even when the shorter phrase “WIC/SI12 VAE” is used in
discussion. Corrected SI13 is retained for diagnostics and later comparisons,
but it is not a baseline VAE channel.

**Critical invariant:** `wic_corrected` has already undergone SI12-based proton
correction in icBuilder. icAnalyzer must never call `proton_correct_images()`
on this field or subtract a proton contribution again. The Hardy proton-energy
model belongs to the upstream Product-2 provenance.

## icPhysics compatibility gate

`ped()` and `hall()` already accept NumPy arrays. The only conversion relevant
inside icAnalyzer is from the already proton-corrected WIC sample to electron
energy flux for an explicitly supplied electron mean energy, followed by Hall
and Pedersen conductance. SI12 is not used to proton-correct the sample again.

This observation does **not** authorize an icPhysics change. icBuilder already
depends on the current public functions, and its detector, precipitation, and
conductance products have numerical parity tests tied to them. Before changing
icPhysics:

1. enumerate every icBuilder and icAnalyzer call site;
2. freeze representative 2 keV input/output fixtures from the current
   icBuilder calculation;
3. determine whether the existing public API can be safely composed without a
   new function;
4. run the icPhysics and icBuilder test suites against any candidate change;
   and
5. request a separate approval for an API change after reporting numerical and
   pipeline impact.

If the existing API proves insufficient, a possible small API—subject to that
review rather than accepted by this plan—would have semantics equivalent to:

```python
precipitation_from_mean_energy(
    wic_corrected,
    dwic_corrected,
    electron_energy,
    electron_energy_uncertainty,
)
```

It would broadcast scalar energies across arrays and return `E0`, `dE0`, `Fe`,
`dFe`, and `varE0Fe`, without knowing about files, grids, Kp, fuvpy, or the
VAE. Do not implement this candidate merely because it appears in this plan.

The intended scientific composition, whether achieved through existing calls
or a separately approved API, is:

```python
precipitation = reviewed_precipitation_adapter(
    wic_corrected,
    dwic_corrected,
    electron_energy=2.0,
    electron_energy_uncertainty=0.0,
)

hall_sample = hall(precipitation["E0"], precipitation["Fe"])
pedersen_sample = ped(precipitation["E0"], precipitation["Fe"])
```

For latent-ensemble covariance, the energy values stay fixed and only the
paired decoded corrected-WIC/SI12 fields vary. Analytic
measurement-uncertainty outputs are stored separately and do not define the
ensemble anomalies.

Under this baseline, propagated Hall and Pedersen depend directly on sampled
`wic_corrected` and fixed electron energy. Sampled SI12 does not enter the
conversion a second time. SI12 remains part of the learned joint FUV state and
its covariance with corrected WIC, but it has no additional independent effect
on conductance unless a future experiment instead samples uncorrected WIC and
repeats proton correction.

## Phase 0: preserve the current repository state

1. Create `archive/pre-wic-si12-vae-2026-09-25` from the current dirty
   worktree before committing.
2. Make coherent preservation commits for:
   - annotation and project memory;
   - the experimental WIC/SI12 loader/script and overview changes; and
   - repository hygiene.
3. Stop tracking Python bytecode and ignore generated caches.
4. Do not commit `figures/ncso_new/` or the local literature PDF as ordinary
   source files.
5. Create `vae/wic-si12` from the preserved checkpoint. Leave `main` at the
   reviewed `7c71591` commit until integration is intentional.

The copied `step_1_vae_wic_s12.py` is preserved as an experiment, not adopted
as the new canonical implementation.

## Phase 1: establish the fixed-grid data contract

1. Wait for the coordinated Product-2 update to add corrected WIC, corrected
   SI13, SI12, and their uncertainties to detector and CS products.
2. Update and verify `PrecipitationCS` schema support in `icreader.load()`.
3. Verify the corrected-WIC provenance and assert that icAnalyzer does not
   apply proton correction again.
4. Verify constant-field reproduction, area-integral behavior, coverage, and
   propagated detector variance on representative frames.
5. Record exact upstream commits and processing configuration in every file,
   including the Hardy proton-energy model used for correction.
6. Convert the existing annotation geometry non-destructively from 36x36
   index coordinates to Cubed-Sphere coordinates, then rasterize on 46x46.

Exit criterion: icAnalyzer can load one orbit through `icreader.load()` with
corrected WIC, corrected SI13, SI12, their uncertainties, masks/weights,
orbit/time identity, and grid/provenance, with no silently dropped frames and
no duplicate proton correction.

## Phase 2: audit and verify physical propagation

1. Make no icPhysics code change initially.
2. Trace the current icBuilder production of corrected WIC, its electron-flux
   conversion, and `robinson_conductance()` and freeze 2 keV reference cases.
3. Determine whether existing public functions can support scalar, 46x46,
   time-batched, and ensemble-batched inputs without semantic changes.
4. Define finite, negative-count, zero-count, masked, and NaN behavior from the
   current pipeline rather than inventing new behavior in icAnalyzer.
5. If a public API gap remains, prepare a minimal icPhysics proposal with
   focused icPhysics tests, frozen icBuilder parity tests, and an icAnalyzer
   batched-array test. Do not implement it without separate approval.

Exit criterion: an approved, numerically parity-tested path converts a
corrected-WIC ensemble of shape `(sample, 46, 46)` to paired Hall/Pedersen
fields of the same shape at fixed 2 keV electron energy without changing
existing icBuilder results or applying proton correction twice.

## Phase 3: build the processed VAE dataset

Create a versioned artifact containing:

- corrected WIC, SI12, and optional corrected SI13 fields;
- per-channel measurement variance;
- binary validity masks and continuous quality weights as distinct concepts;
- orbit, time, source identity, and grid metadata;
- train/validation/test assignment by orbit; and
- upstream product identifiers and repository commits.

Do not use the legacy first-10,000-frame split. Do not use
`log10(value + 1)` until the new background-subtracted value range is
inspected; negative values may require a linear standardization or signed
transform.

Exit criterion: saved split counts, shapes, units, finite fractions, grid
identity, and annotation alignment reproduce on reload.

## Phase 4: train a defensible WIC/SI12 VAE

Use a two-channel, mask-aware model with dynamically derived grid shape.
Retain the readable spline decoder only if its 46x46 basis is verified.

The saved run artifact must contain:

- model state;
- architecture and latent dimension;
- normalization/transform parameters;
- grid definition;
- data-product provenance and split identifiers;
- optimizer, learning-rate, beta/KL schedule, and random seeds; and
- separate reconstruction and KL histories.

Reduce reconstruction and KL terms consistently per sample so changing batch
size does not change the effective beta.

Verification gates:

- tiny-subset overfit smoke test;
- orbit-held-out reconstruction metrics;
- reload reproduces the same outputs;
- prior samples and aggregate-posterior samples are compared with held-out
  WIC/SI12 distributions; and
- missing-data masks do not become a generated physical channel.

## Phase 5: sample and analyze WIC/SI12 covariance

1. Draw paired latent samples in reproducible chunks.
2. Decode physical WIC and SI12 fields, applying only the documented inverse
   transform.
3. Preserve sample ordering and pairing.
4. Store the ensemble mean and centered anomaly factor. Materialize dense
   covariance blocks only when needed.
5. Distinguish coefficient covariance, transformed-field covariance, and
   physical-count covariance in names and metadata.

Required checks include symmetry, positive-semidefinite tolerance,
cross-block transpose, bounded correlations, and convergence of means,
variances, and leading subspaces across sample counts and seeds.

## Phase 6: propagate samples and form conductance covariance

For every latent sample:

1. take the decoded `wic_corrected` field without further proton correction;
2. infer electron energy flux from corrected WIC at 2 keV electron energy;
3. calculate paired Hall and Pedersen conductance fields; and
4. retain the same sample index through all outputs.

Construct the joint state as `[Hall, Pedersen]`. Save physical-Siemens moments
for interpretation and, if used for assimilation, a separately named
transformed positive state. Verify that the Hall-to-Pedersen and
Pedersen-to-Hall covariance blocks are transposes.

## Phase 7: sparse conductance reconstruction

Use the propagated ensemble mean and anomaly factor directly in an ensemble-
space Gaussian update. Support separate Hall and Pedersen observation masks
and an explicit observation-error covariance `R`.

Required experiments:

- Hall-only observations;
- Pedersen-only observations;
- joint observations;
- several realistic sparse geometries; and
- prior-mean and independent-channel baselines.

Required invariants:

- no observations returns the prior mean;
- near-zero-error full observations recover the state;
- larger observation error weakens the update;
- Hall-only observations affect Pedersen only through learned cross-covariance;
- posterior uncertainty does not increase from a valid observation update;
  and
- final scores use held-out orbits.

## First end-to-end milestone

One representative orbit must run reproducibly through:

```text
icreader.load(precipitation_cs)
  -> processed WIC/SI12 tensors
  -> trained/reloaded VAE
  -> paired latent samples
  -> WIC/SI12 covariance diagnostics
  -> fixed-2-keV propagation
  -> joint Hall/Pedersen ensemble
  -> sparse reconstruction with posterior uncertainty
```

Only after this succeeds should the workflow scale to the full corpus or add
conditional variables, nonconstant energy models, or sampled physical-model
uncertainty.

## Deferred decisions

- the VAE input transform after inspecting the new count distributions;
- latent dimension and spline/control-grid resolution;
- whether SI13 becomes an auxiliary VAE channel later;
- the final representation and storage of large covariance products;
- conditional variables for the later conditional VAE; and
- the realistic proton/electron energy model and its uncertainty distribution
  after the fixed-2-keV baseline.
- whether the fixed-energy precipitation conversion can use the existing
  public icPhysics API unchanged or needs a separately approved addition.
