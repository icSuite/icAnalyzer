# Decision Log

## 2026-11-09 — Use designed IMF/SML sampling for the hybrid4 climatology

Do not use empirical training-row resampling for the scientific IMF comparison;
that would mix the observed joint condition distribution into a controlled
climatology. Cross four IMF directions with three SML activity bands:
`[-100,0)`, `[-300,-100)` and `[-1000,-300)` nT. Draw SML, selected IMF
component magnitude from2--10 nT and clock offset within plus/minus30 degrees
independently and uniformly; fix dipole tilt at0 degrees. Use common latent
draws and uniform ranks across groups. Hold Kp at3 only in the downstream
Zhang--Paxton/Fang conversion rather than recovering it from observed rows.
Treat instantaneous SML as an electrojet-activity class, not a substorm label;
the disturbed band may mix substorms, storms, convection and recovery phases.
Retain empirical-row sampling only as an explicitly labelled diagnostic.

## 2026-11-06 — Generalize one CVAE pipeline rather than fork a Kp copy

Do not rename the supported `scripts/CVAE/` workflow to `CVAE_kp` or duplicate
its cache, model, loss, training and sampling machinery. Kp will remain the
one-feature compatibility case of a future named conditioning vector. Before
changing the model, prepare a source-independent, frame-aligned SuperMAG table
and quantify coverage and predictive importance. Generated conditioning data
belongs outside the repository; current Kp artifacts and samplers remain
unchanged until the useful parameter set has been selected.

## 2026-11-05 — Use generic WIC and explicit H/P cache preparation names

The canonical WIC commands are `prepare_wic_cvae_cache.py` and
`train_wic_cvae.py`; Hall/Pedersen preparation is
`prepare_cvae_hp_cache.py`. WIC preparation is one implementation with mutually
exclusive manual-annotation and frozen-automatic-mask inputs; do not duplicate
the cache pipeline by selection source. Record generic mask provenance and keep
old automatic WIC caches readable. Hall/Pedersen preparation remains explicit
about its field pair and current automatic-mask input. This prepares the manual
WIC baseline but does not authorize or imply a full cache/training run.

## 2026-11-05 — Extract current CVAE core from `icanalyzer/wic_si12`

The supported Hall/Pedersen and WIC Kp-CVAEs share a field-agnostic core under
`icanalyzer/cvae/`: cache, data transform/split, spline conditional model, loss,
training, artifact and ensemble modules. New models explicitly identify their
fields. Preserve read compatibility for pre-rename spline artifacts and repair
historically mislabeled conductance fields at load time without rewriting saved
files. Do not carry legacy-only WIC/SI12 loaders, count likelihood,
comparison/diagnostic/physics code, unconditional model, covariance blocks or
sparse reconstruction into the current package. Keep the old package isolated
until its remaining script consumers are classified rather than deleting it
prematurely.

## 2026-11-05 — Gather supported automatic CVAEs under scripts/CVAE

The supported automatic-CVAE script boundary consists of Hall/Pedersen
prepare/train/sample and WIC prepare/train/sample/plot. Move those seven entries
to `scripts/CVAE/` and keep reusable model, cache, transform, runtime and
ensemble machinery under the importable `icanalyzer` package. Do not duplicate
library code in the scripts directory. Leave the generic automatic postprocessor
with the older WIC/SI12 visualization and diagnostic family because it is not
required by the fixed-Kp sampling workflows. Existing caches, runs, samples and
figures remain unchanged.

## 2026-11-05 — Remove the temporal Bad veto from automatic masking

The user rejects the hidden third-network complexity in the nominal Model1 +
Model2 policy. Remove the five-frame temporal context builder, Bad-veto CNN and
training, its checkpoint/threshold/hash contract, stored Bad scores and veto
inference. Model1 is one single-frame whole-frame acceptance CNN; a nonempty
frame above its frozen cutoff is released, otherwise Model2 supplies the pixel
mask. Existing masks and CVAE caches made with the veto remain immutable
historical artifacts and must not be presented as acceptance-only outputs. Any
new comparison uses fresh mask/cache/run paths. The old acceptance checkpoint
and Model2 lineage remain compatible; no retraining or data generation was
performed during the cleanup.

## 2026-10-15 — Choose a fully balanced CVAE comparison

After quantifying the3x cap's limited correction, the user agrees to testing a
fully balanced model against the current unweighted run. Preserve original
per-observation quality and validity masks; coverage compensation belongs in
the reconstruction loss, before its weight-sum normalization. The KL formula
and encoder input values remain unchanged. This selects the next experiment,
not a scientifically validated production model. Full-balance mode is now
implemented with uncapped inverse smoothed coverage; no new training is
launched. [[CVAE Coverage Weighting]] owns
the current capped implementation and measured cap effects.

## 2026-10-15 — Optional smoothed coverage compensation in CVAE loss

User authorizes implementation following the discussion of weaker dayside
coverage. Fit quality-weighted coverage from eligible training frames, smooth
before inversion, retain original quality/masks, and cap the compensation.
Implemented experiment defaults are sigma1.5 pixels and maximum boost3, with a
shared spatial map or optional broad Kp bins. Freeze maps in model provenance
and reuse them for evaluation. Preserve the running baseline and require a
separate output for the comparison. This authorizes the functionality, not an
automatic long training run or adoption as a scientifically validated remedy.
[[CVAE Coverage Weighting]] owns the algorithm and verification.

## 2026-10-13 — Isolated automatic-mask CVAE, user-run training

User authorizes only steps1–3 for CVAE, explicitly reserving the actual run for
the user and skipping the standalone verification campaign. Add separate
preparation/training entry points; import the existing native Kp-CVAE model,
loss and trainer unchanged. Freeze automatic masks in an on-disk cache with
source/time/grid/split provenance, preserve intrinsic validity/cutoff and the
normalization-before-intrinsic-cut order. No manual merge, mask cleanup or
new uncertainty weighting/defaults. Bound per-split mappings and batch reads;
never automatically generate ensembles after training. New checkpoint naming
prevents old plots from silently ignoring cached masks. Existing commands and
artifacts remain untouched; cached plotting/covariance is deferred to step4.
Small data-loading checks are allowed; no training launched. Current State
owns evidence/limitations; README owns user commands.

## 2026-10-13 — Accept automatic-mask errors to expand VAE/CVAE data

Confirmed user statement: some automatic masks are wrong, and increased mask
error is acceptable when it increases the data entering VAE/CVAE. Imperfect
examples do not block corpus integration or require another mask-optimization
loop. No new numeric error tolerance, threshold change, shape cleanup or manual
override is authorized. Current NPZ masks remain Boolean validity selectors,
intersected with upstream validity, not a new confidence weight or loss term.
The subsequent scoped CVAE cache/training commands are now implemented;
actual training is reserved for the user. Other workflow scope remains unchanged.
Current State owns the verified automatic-corpus format and memory evidence.

## 2026-10-13 - Keep automatic WIC masks separate from manual annotations

Confirmed user instruction: the weight0.1 U-Net may later generate raw masks
for every frame in the named precipitation_cs source directory. Save them in
data/training_data/, with examples in figures/training_data/. Do not merge or
override manual reviews in those outputs; any combination belongs to later VAE
code. Do not clean the saved mask shapes. The user requested script creation
first and corrected an unauthorized training launch; neither script completed
during that preparation. Subsequent live automatic outputs are documented in
Current State, without implying an agent-authorized training run.

## 2026-10-13 - Explicit shape prior for WIC mask predictions

Confirmed user requirements: no enclosed rejected islands, and at most one
accepted region. An accepted island in rejected surroundings is allowed.
User authorizes testing these rules in the training loss; fully rejected
frames remain valid empty masks. [[WIC Topology Loss Benchmark]] owns the
verified full-grid target contract, missing-data separation and experiment.
The training penalty is explicit and raw outputs remain inspectable; no
silent cleanup, annotation adoption or VAE training follows automatically.
The completed benchmark reduces shape violations but degrades fully rejected
mask agreement; it does not establish the model as the VAE mask source.

## 2026-10-12 - Transfer annotation classes and lines across the full native grid

Confirmed user instruction:36x36 to46x46 conversion transfers frame class and
drawn lines, without marking the added area unreviewed. Full frames retain all
finite native pixels; rejected frames retain none; partial line sides apply
across the entire grid and hold forward within each partial interval. Remove
the unintended old-grid clipping while preserving later native edits and all
class/time/keyframe boundaries. [[Full Grid Annotation Transfer]] owns repair,
backup and verification. Prior clipped-mask benchmarks are historical.

## 2026-10-12 — Implement optional Kp conditioning; compare means with Hardy

**Confirmed by the user:** implement Kp conditioning while preserving all
existing functionality. Scientific aim is to see whether ensemble means at
fixed Kp look more like Hardy. User will annotate more data over time; no
deadline or completeness guarantee is stated. The expectation that thin data
may worsen teleconnections is a concern/hypothesis, not an established result.

Use one conditional encoder/decoder sharing all training data, optional flags,
explicit generation Kp and a separate trained artifact. Preserve old runs and
defaults. Software verification is complete; scientific agreement/spatial
acceptance awaits a full conditional training comparison. Detailed owner:
[[Kp-Conditional Conductance VAE Proposal]].

## 2026-10-12 — Use the hybrid native run for conductance analysis

**Confirmed by the user:** use `_hybrid`, which includes the largest data
selection among these native runs. Live artifact
`conductance_smoothed_b16_holes_hybrid` confirms6881 retained training images,
versus3754 for `_annotated`; beta1.6/latent36, selected epoch207. This is the
target for subsequent conductance diagnostics and the interactive loading
example. Earlier annotated-run results retain their scope and do not establish
hybrid uncertainty quality. No retraining or publication-model acceptance.

Detailed owner: [[Loading Saved Conductance Ensembles]].

## 2026-10-12 — Conditional spatial structure requires reasonable teleconnections

**Confirmed by the user:** conditional teleconnections must look physically
reasonable before assuming that the event-conditioned distribution captures
conductance spatial structure. Numerical covariance generation and sampling
convergence are insufficient acceptance criteria. Visual plausibility is a
necessary screen; independent physical validation remains required.

Both ensembles use the same restricted-population VAE. Event representation
diagnostics flag orbit 99 but not 106; data coverage remains a possible cause,
not an established fix. No retraining or alternative inference method adopted.
Detailed owner: [[Event-Conditioned Conductance Covariance Test]].

## 2026-10-12 — Event-conditioned conductance covariance as a Lompe use case

**Confirmed by the user:** encoding an available event image can provide the
starting point for event-specific conductance covariance. Use a model trained
on matching inputs; sample its encoder posterior and decode full paired fields.

**Boundary:** this model-implied conditional covariance is distinct from the
unconditional population covariance and the proposed training-encoding mixture.
Physical uncertainty calibration remains unverified; no sampler/prior adoption
or Lompe inversion was implemented in this clarification. Detailed owner:
[[Native Conductance Prior Validation]].

## 2026-10-09 — Native-grid annotation without silently extending old reviews

User requests46x46 native CS annotation and the same edited JSON in the FUV
pipeline. Update the existing GUI, retaining schema2 and per-orbit source-file
time axes. Align opened converted reviews by timestamps, never by old ordinal;
preserve cached masks, dropped-keyframe propagation and the reviewed footprint,
with the original review archived. Whole-grid review is an explicit action,
not a consequence of loading larger images. New native records cover the new
grid, use native CS indices and icReader CF times; CSV seeds use WIC source IDs.
Keep JSON-only/hybrid/default CSV policies and the model/loss unchanged in both
pipelines. Original36x36 JSON remains untouched. Current State owns verification
and saved-run fingerprint caveat; README owns usage.

## 2026-10-09 — Whole-orbit JSON precedence with CSV fallback

User approves combining the shorter-period pixel annotations with the older
CSV. Add explicit --hybrid-annotations, keeping JSON-only --annotations and
the original CSV default unchanged. Completed JSON reviews override the
whole orbit, including bad/unmatched-frame exclusions; no_usable reviews
exclude whole orbits. CSV applies only elsewhere, including unfinished JSON
reviews. Never union accepted frames inside a completed review. Retain CSV
source-index selection and zero-quality-frame semantics, upstream masks,
intrinsic cutoff and model/loss. Save/recheck both annotation fingerprints and
original timestamps; held-out figures use the same priority. Expanding the
orbit set changes the seeded splits, so scientific comparisons need matched
held-out evaluation. Current State owns verification and corpus counts.

## 2026-10-09 — Opt-in completed annotations for paired VAE training

User requests fully good frames and masked partial frames from the converted
annotations. Add --annotations to both runners without changing CSV defaults,
architecture or loss. JSON replaces CSV selection; completed reviews only.
Match original conductance timestamps (within1 second), never assume new frame
ordinals match old ones. Apply shared manual masks to normalization, encoder
and targets; hold an earlier mask only within its own partial interval.
Outside original spatial support is unannotated, not trustworthy background.
Preserve the intrinsic33% technical cutoff before manual masking. Save JSON
and original-time fingerprints and reload the same selection for held-out
figures. Fresh training is required; no model improvement/adoption is inferred.
Current State owns verification; README owns usage.

## 2026-10-09 — Simple explicit spherical conversion of manual annotations

User requires conversion through spherical latitude/longitude rather than
assuming identical CS grids, and requests one readable script without a
large test/framework exercise. Implement a separate-file converter for old
pixel geometry and masks. Each old/target grid supplies its own projection;
the stored angular frame is magnetic latitude and15*MLT longitude, not
geographic. Target masks are evaluated by inverse mapping so transformed
curvature does not silently become a straight chord. Preserve source geometry,
mask and frame IDs, add timestamps, mark extra target-domain cells unannotated.
Do not change VAE/GUI loaders, temporal indexing or existing annotations in
this task. Current State owns real-file verification and next integration step.

## 2026-10-07 — Conductance versions of teleconnection figure3

User authorizes conductance figure3 using the existing three definitions:
Hall and Pedersen anchor Pearson maps, and unsigned joint explained variance
R^2 at the same flattened spatial location. Preserve physical-ensemble
covariance/full sample count, not the log-space reconstruction covariance or
measurement-error covariance. Reuse the shared numerical calculation and
layout without changing the FUV default. Native training support is excluded;
unsupported anchors rejected; existing conductance outputs preserved. Native
and fixed-2-keV proxy provenance remain distinct. No retraining or new
uncertainty/prior-model adoption. Current State owns verification.

## 2026-10-07 — Use native retrieval uncertainties in conductance reconstruction

User requires conductance_cs uncertainty to enter the reconstruction
likelihood. Figures4/5/6 default to selected-frame dH/dP, propagated to
log10(Sigma+1) by the first-order derivative; Gaussian conditioning uses
diagonal per-observation variance, not method-quality weights. Explicit
--observation-std retains fixed-dex comparisons; --observation-errors propagated
requires native input. No floor by default; optional --minimum-observation-std
is in S and recorded. Zero retained errors require an explicit positive floor.
No training/loss change or retrieval-error correlations are inferred. Real
frame660 actual-error and1 S-floor tests both worsen hidden-region skill;
neither floor choice nor calibrated uncertainty is adopted. Current State owns
verification and metrics.

Last reviewed: 2026-10-09

## 2026-07-26 - Use the standard project vault

**Decision:** Store project memory in the repository-root `vault/`, with root
`AGENTS.md` as the machine entry point.

**Rationale:** The user wants one predictable project-memory path across every
repository and automatic continuity between project and central sessions.

The vault uses automatic milestone checkpoints and a structured
`Portfolio impact` handoff. This documentation change does not imply
scientific, model, or publication progress.

## 2026-07-26 - Preserve the publication concept

**Confirmed context:** The project aims to create a VAE-based empirical model
of IMAGE-derived height-integrated Hall and Pedersen conductance. Spatial
covariance is especially important for the user's broader
ionospheric-electrodynamics data-assimilation work.

This records the user's stated research direction. It does not select a
specific script, covariance formulation, validation result, or final abstract
claim.

## 2026-07-27 - Preserve the working VAE lineage

**Working direction confirmed by the user:** Treat
`scripts/IMAGE/doing_it_manually_log_spline.py` as the strongest architectural
starting point, while retaining the following lineage for later
implementation:

1. keep the log-space spline VAE and joint coefficient covariance;
2. keep the control-space optimal-interpolation formulation;
3. discard or repair the current latent-assimilation experiment;
4. restore the Hall-only, Pedersen-only, and joint-observation experiments
   from `scripts/Cleaned_VAE_final.py`;
5. restore adaptive rank selection and posterior uncertainty;
6. add orbit-separated quantitative validation and an independent baseline.

This is a working lineage, not yet a declaration that either script is
canonical or that its scientific results are validated. Audit the remaining
`scripts/IMAGE/` experiments before changing the implementation.

## 2026-07-27 - Establish the primary log-spline script

**Decision:** Use `scripts/IMAGE/primary_log_spline_vae.py`, copied from
`doing_it_manually_log_spline.py`, as the working implementation. Preserve the
source experiment unchanged.

The primary script:

- normalizes Hall-Pedersen cross-covariance using variances from the matching
  Hall and Pedersen marginal blocks;
- performs latent assimilation in normalized log10 space, including
  first-order transformation of physical observation errors;
- converts assimilated spline fields back to non-negative physical
  conductance.

**Scope decision confirmed by the user:** Do not pursue the iterative-PCA
baseline. Defer the independent-versus-joint observation experiments and the
larger validation/uncertainty/reproducibility work for a later task.

## 2026-07-27 - Sequence development around data quality and a conditional VAE

**Direction confirmed by the user:** The next development phase has four
linked objectives:

1. overhaul the training data through manual quality labeling, while retaining
   partially useful images by marking and ignoring only their bad regions;
2. study and revise the current VAE formulation until its design choices are
   theoretically defensible;
3. develop a conditional VAE driven by geomagnetic-activity descriptors, so
   users can request conditions and generate a conditional conductance field
   and covariance;
4. split the monolithic workflow into data preparation, model training,
   initial analysis, and advanced comparison stages.

**Dependency decision:** Establish the annotation/data contract and modular
pipeline first, then build a defensible unconditional baseline, then extend it
to a conditional VAE. Do not change data selection and model class
simultaneously when a controlled comparison is possible.

The exact condition variables, annotation interface, processed-data format,
and model architecture remain to be selected from inspected data and
scientific requirements.

## 2026-07-27 - Use resumable interval and keyframe-mask annotation

**Implementation decision:** Represent each reviewed orbit with a single
usable envelope and an optional fully good core. Frames between those
boundaries are full, partial, or bad without requiring a separate label for
every image.

For partial regions, store sparse 36x36 valid-mask keyframes and their lasso
vertices. Reuse the nearest keyframe outside the keyframe span and interpolate
between keyframes through signed-distance fields. Require a keyframe in every
partial interval before an orbit can be marked complete.

Keep `data/good_orbit_segments.csv` read-only and seed unambiguous historical
good intervals from it. Store new annotations, explicit review states, and the
resume cursor in a separate schema-versioned JSON file, saved atomically after
each interaction. Rasterization into a processed training dataset remains a
separate pipeline stage.

## 2026-07-27 - Hold masks forward instead of interpolating

**Decision confirmed by the user:** A mask keyframe becomes active on its
frame and remains unchanged on subsequent frames until another keyframe
explicitly replaces it. Do not interpolate mask shape between keyframes.

Every partial interval must therefore have its initial mask on its first
frame. This supersedes the temporal-interpolation part of the earlier
annotation decision while retaining its orbit boundaries, sparse keyframes,
separate JSON persistence, and resumable cursor.

## 2026-07-27 - Support lasso and additive half-plane mask geometry

**Implementation direction requested by the user:** Retain lasso masks for
irregular regions, but support the common case in which a straight diagonal
separates a bad half-image from a valid half-image.

The line tool stores two endpoints and the selected valid side for every cut.
Multiple lines combine by intersection, so each new valid half-plane further
trims the retained region. The user may flip or undo the newest line because
line direction should not require mental side calculation. The resulting
36x36 mask is stored alongside the geometry.

Both geometry types use the same forward-held keyframe semantics.
Dividing-line mode is the default for new keyframes; an existing lasso reopens
in lasso mode. Lasso remains the fallback for concave or disconnected valid
regions that an intersection of half-planes cannot represent.

## 2026-07-28 - Retain IMAGE global snapshots

**Decision confirmed by the user:** Global auroral snapshots are essential to
the project because the intended VAE must represent global spatial covariance.
Retain IMAGE as the primary observational geometry.

GUVI Level 2B and Level 3 products contain valuable retrieved electron mean
energy and energy flux, but their orbital swaths are not simultaneous global
states. Evaluate GUVI as an external calibration, validation, or benchmark
source rather than replacing IMAGE.

**Proposed implementation consequence:** Compare Zhang-Paxton first with
OVATION Prime, which supplies both average energy and energy flux and has
public implementations. Evaluate ASHLEY-A as a newer spectral alternative only
after confirming code access and batch-operability. No alternative may require
manual or fragile per-frame oval registration.

## 2026-08-18 - Make annotations independent of the analysis grid

**Decision confirmed by the user:** Replace the 36x36-index-dependent
annotation representation with geometry in cubed-sphere coordinates. A saved
annotation must be rasterizable on grids with different density or spatial
extent.

Write a one-time, non-destructive converter for the existing annotation
dataset. Transform dividing-line endpoints from old array coordinates into
cubed-sphere coordinates and retain their valid half-plane. The two keyframes
using the older single-line representation may be discarded if supporting
them adds unnecessary complexity.

## 2026-09-25 - Start WIC/SI12 propagation with fixed 2 keV energies

Superseded for proton energy by the 2026-09-28 Hardy decision below. Fixed
2 keV electron propagation remains active.

**Decision confirmed by the user:** The first WIC/SI12 VAE propagation
baseline assumes a spatially and temporally constant proton mean energy of
2 keV and electron mean energy of 2 keV.

Keep both energies fixed across latent samples. This makes the first joint
Hall/Pedersen covariance attributable to learned corrected-WIC variability
under an explicit physical baseline. Do not treat the assumption as the final
precipitation model. Keep detector measurement uncertainty and future
physical-model uncertainty separate from the latent ensemble so their
contributions can be evaluated independently.

## 2026-09-25 - Ingest corrected FUV fields from Product 2 CS

**Decision confirmed by the user:** icBuilder Product 2 will include corrected
WIC, corrected SI13, SI12, and their uncertainties. These fields will be
carried into `precipitation_cs`, which icAnalyzer will ingest through
icReader. Do not create a separate `fuv_cs` product for this workflow.

The WIC field is already proton corrected upstream. icAnalyzer must not apply
proton correction again. The selected proton-energy model belongs to the
upstream correction and its product provenance; downstream sample
propagation starts from corrected WIC and uses the fixed 2 keV electron energy
to obtain electron energy flux and conductance.

## 2026-09-25 - Keep fixed-energy propagation local pending a public API review

**Implementation decision:** Do not modify icPhysics or icBuilder as part of
the icAnalyzer VAE baseline. Compose the existing batched `hall()` and `ped()`
functions behind a small icAnalyzer compatibility adapter and freeze the
current 2 keV WIC response (`477.5 counts/(mW m^-2)`) in a parity test.

The adapter accepts already proton-corrected WIC, rejects finite negative
counts, preserves NaNs and arbitrary leading ensemble dimensions, and records
that the nonlinear conversion occurs after CS gridding. It does not imply that
the missing public, grid-independent WIC-to-precipitation boundary is resolved:
the current response helper is private, so an icPhysics API remains a separate
reviewed change requiring upstream parity evidence and approval.

Store covariance as an ensemble mean plus normalized anomaly factor by
default. This representation is exactly sufficient for the implemented sparse
Gaussian update and avoids materializing a dense 4232-by-4232 matrix.

## 2026-09-28 - Use Hardy for upstream proton correction

**Decision confirmed by the user:** The production Product-2 corpus is
supposed to use the Hardy proton-energy model. This supersedes the earlier
constant-2-keV proton assumption.

Keep the downstream electron-energy propagation fixed at 2 keV for the first
VAE baseline. Product-2 `wic_corrected` has already undergone SI12-based proton
correction using Hardy energy, so icAnalyzer must still never repeat proton
correction. Require `proton_energy_model="hardy"` in the ingestion contract and
carry that provenance into saved run artifacts.

## 2026-09-28 - Apply manual intervals through WIC source identity

**Implementation decision:** Reuse the manually accepted inclusive intervals
from `data/good_orbit_segments.csv`, but do not assume regenerated Product-2
array positions equal historical frame numbers. Map `good_start:good_stop`
through each file's `wic_source_index` and read only those positions.

Manual selection is enabled by default in the canonical runner. `--all-frames`
is an explicit diagnostic override, not the production training default. This
reduces the copied corpus from 358,439 frames to 10,680 mapped frames before
materialization and preserves source-frame identity when upstream gating has
removed frames.

## 2026-09-28 - Select the minimum-validation-loss VAE state

**Implementation decision:** Restore the model state with minimum validation
total loss before held-out evaluation, latent sampling, physical propagation,
and artifact saving. Retain the full training history and record the selected
epoch and validation loss in the artifact configuration.

This prevents a later overfit epoch from silently defining the ensemble, as
happened in the first 20-epoch artifact where epoch 10 validated better than
the saved epoch 20. Early stopping may later reduce compute, but it is not
required for correct checkpoint selection.

## 2026-09-29 - Compare legacy and rerun products on Product-2 identity

**Implementation decision accepted by the user:** Compare the legacy
`precipitation_IR_hardy_weighted` data with the rerun Product-2 data on
same-named orbit files and exact common product timestamps. Product 2 owns the
manual source-frame selection and the SZA mask for both sides. Use the legacy
combined robust weight `w` for the legacy VAE and retain the current Product-2
quality/coverage weighting for the rerun VAE.

The direct-pairing decision remains active, but the stated weight orientation
was corrected on 2026-10-01: both upstream weights must be inverted.

Use the same common orbit list, split seed, and model settings for matched VAE
runs. Accompany model comparisons with direct paired field, uncertainty,
quality, and support diagnostics. Describe the result as a net product-pipeline
change rather than a pure fuvpy effect because matching, coregistration,
gridding, validity, and uncertainty propagation also changed.

## 2026-09-30 - Sequence VAE uncertainty modeling through the loss first

**Decision confirmed by the user:** Implement absolute measurement-uncertainty
weighting in the reconstruction loss before experimenting with uncertainty as
encoder context or downstream covariance composition.

The current VAE already propagates `dwic` and `dsi12` into transformed-space
variance, but normalizes precision independently within every sample and
channel and divides by the weight sum. This retains spatially relative weights
while largely cancelling absolute uncertainty differences between frames.

Implement the first revision with one training-set precision scale per WIC and
SI12 channel, explicit uncertainty floors/caps, and valid-support reduction so
globally uncertain frames contribute less than precise frames. Keep corrected
WIC and SI12 as the only decoder outputs and recalibrate the KL balance.

After that baseline is evaluated:

1. test transformed/log `dwic` and `dsi12` as auxiliary encoder inputs;
2. keep latent/background covariance and measurement/observation covariance
   separate downstream, combining them only for an explicitly defined
   predictive-observation quantity; and
3. do not make measurement uncertainty additional decoder outputs in the
   baseline, because that would entangle observing-system uncertainty with the
   intended geophysical latent covariance.

**Implementation checkpoint:** Commit `23afaeb` implements the first step with
training-only median precision scales, transformed-variance floors, relative-
precision caps, and valid-support reduction. The fitted configuration is
mandatory for training/evaluation and is saved in the artifact. The old beta
value is not automatically accepted as calibrated under the new loss.

**Superseded reduction:** That valid-support reduction changed the literal
legacy reconstruction/KL scale and was removed from the baseline.

## 2026-10-02 - Make measurement uncertainty an explicit option

**Decision confirmed by the user:** Measurement uncertainty should be
available to the VAE reconstruction likelihood but remain disabled by default.
Do not add uncertainty as decoder outputs or silently change the baseline.

When enabled, transform `dwic` and `dsi12` into normalized-log10 variance, fit
one median precision scale per channel on the training split, and apply an
explicit variance floor and relative-precision cap. Multiply this precision by
the auroral quality weight. Preserve the legacy loss scale by dividing the
channel-expanded effective weight sum by the channel count; retain the summed
KL reduction. Save the complete fitted loss configuration in the artifact.

The canonical CLI exposes `--measurement-uncertainty`; the interactive script
uses `USE_MEASUREMENT_UNCERTAINTY = False`. A real one-orbit smoke completed
the full pipeline with the option enabled, and all 57 focused tests pass.

## 2026-10-01 - Invert upstream background-fit weights for VAE training

**Scientific correction confirmed by the user:** Upstream zero weight means
that the background method could not fit the data, and those data tend to be
aurora. The weight is therefore not a confidence score for VAE training.

At ingestion, use `1 - method_quality_weight` for current Product 2 and
`1 - w` for the matched legacy product. Apply per-channel coverage afterward
for current data. Keep technical validity independent from this continuous
weight. Record the transformation in comparison reports and run provenance.

Real-data VAE artifacts made before this correction optimized the reversed
scientific target and must be retrained. Physical camera-field mean maps from
the direct legacy/rerun comparison remain usable, but its quality statistics
must be regenerated.

**Superseded on 2026-10-02 for current Product 2.** The premise applies to raw
fuvpy/legacy Tukey weights, but icBuilder already performs that inversion when
creating Product 2. Current Product 2 must use `method_quality_weight` directly;
only legacy `w` should be inverted.

## 2026-10-02 - Use Product-2 auroral-quality weight directly

Live source and tests confirm the complete transformation: fuvpy Product 1
stores Tukey `dgweight`; icBuilder computes and documents Product-2
`method_quality_weight = 1 - dgweight`; CS reduction carries that field forward.

Use `method_quality_weight * channel_coverage` for current Product 2. Use
`1 - w` for the legacy product, which retains the original Tukey orientation.
Do not apply a second inversion to Product 2.

**Superseded on 2026-10-02 for the copied corpus.** This conclusion inspected
current icBuilder source, not the producer revision and stored values. The
copied files report `38b5afb+worktree`; committed `38b5afb` wrote the
uninverted product of fuvpy `dgweight`s, the CS metadata records the method as
`unrecorded`, and the stored values are strongly anti-correlated with auroral
brightness. Direct loading therefore emphasizes background in this corpus.
Whether to regenerate the product or expose an explicit old-corpus inversion
option remains to be decided. Future Product-2 orientation must be established
from explicit provenance rather than inferred from the field name.

**Historical implementation checkpoint:** The canonical and exploratory
current-data loaders were changed to direct loading under this now-superseded
decision. They remain wrong for the copied corpus; the focused tests encoded
the assumed contract and therefore did not detect the data-version mismatch.
The separate `wic_source_index` selection correction remains valid.

## 2026-10-01 - Restore the spline VAE lineage and clip negative inputs

**Decision confirmed by the user:** Replace the simplified convolutional
baseline with the architecture inherited from
`scripts/IMAGE/doing_it_manually_log_spline.py`: a partial-convolution encoder,
a tensor-product cubic-spline decoder, standardized `log10(value + 1)` input,
legacy hyperparameter defaults, and a `ReduceLROnPlateau` scheduler.

For the first controlled run, keep the continuous auroral-interest quality
weight out of the encoder. Continue using it, channel coverage, and propagated
`dwic`/`dsi12` precision in the reconstruction loss. This isolates the restored
architecture before testing quality or uncertainty maps as encoder context.

Clip finite negative corrected-WIC and SI12 counts to zero at the ingestion
boundary and record the counts in warnings and artifacts. Also clip decoded
physical fields at zero after the inverse log transform. The complete manual-
selection audit found zero negative WIC cells but 3,482,898 negative SI12 cells
among 14,030,013 finite cells, so negative SI12 is an upstream processing issue
to report and investigate rather than a reason to retain the signed transform.

## 2026-10-03 - Require explicit Product-2 auroral-weight provenance

The replacement Product-2 CS corpus is
`IR_hardy_smoothed_wic0p8_si120_si131p2`. It explicitly defines
`method_quality_weight` as WIC auroral quality formed from `1 - dgweight`; a
9.22-million-cell value audit independently confirms positive association with
auroral signal.

Use this field directly, multiplied by channel coverage. Reject Product-2
files whose metadata does not explicitly contain the auroral-quality and
`1 minus` orientation instead of inferring semantics from the variable name.
Do not reuse VAE artifacts trained from the old ambiguous `IR_hardy` corpus.

## 2026-10-03 - Artificial training holes are opt-in

The user authorized artificial input-only holes as a separate experiment.
Hide the same patch in WIC, SI12, and the quality encoder input, retaining
original observed targets and loss weights inside it. Keep validation/test
inputs clean and preserve default training, architecture, and loss semantics.
Save augmentation settings and use an independent seeded generator. The user
is separately testing beta 0.5; defer uncertainty encoder inputs and decoder
outputs until other options have been assessed.

## 2026-10-03 - Maintain one systematic VAE experiment register

The user requires a persistent test plan that identifies which parameter or
model aspect is tested, why, and the settings chosen after each conclusion.
`vault/02_Algorithm/VAE Test Plan.md` owns that live queue and settings ledger.
Close each experiment with evidence, limitations, adopt/reject/inconclusive,
and the exact settings carried forward. Proposed thresholds and architecture
changes are not accepted merely because they appear in the plan. Keep the
scientific background and historical pipeline roadmap linked, not as competing
execution queues. This decision does not authorize new model changes or runs.

## 2026-10-03 - Conductance covariance is the primary evaluation target

The user reaffirmed that the objective is a conductance model with covariance
for data assimilation. Legacy sparse reconstruction used covariance eigenmodes,
not the decoder. Encoder inpainting is secondary; its improvement does not
establish improved prior covariance. The current anomaly-factor sparse update
already conditions the covariance without calling the decoder.

Proposed, awaiting agreement: bring E08 forward to compare the existing hole
and no-hole conductance priors using sparse assimilation on unobserved supported
cells, including uncertainty calibration, before promotion or revised-loss E03.
No new implementation, run, default change or setting adoption is authorized
by this clarification. The live test plan records the distinction and proposal.

## 2026-10-03 - Reusable covariance-based sparse reconstruction benchmark

The user considers input holes useful for learning spatial relationships and
authorized a reusable sparse-reconstruction test informed by the legacy code.
Freeze held-out truth proxies, support, observations and noise; reuse them
across existing and future models. Condition covariance eigenmodes rather than
the decoder, and score unobserved sites with point errors and interval coverage.

The implemented defaults use log10(conductance + 1 S), 100 grid-space modes
and 0.1 dex synthetic observation noise. These are explicit benchmark settings,
not changes to the VAE or adopted scientific optima. The first comparison has
useful point skill but inadequate interval coverage; E08 owns its evidence.
No model setting promotion, uncertainty adjustment or upstream edit follows
automatically. Rank/calibration follow-ups remain proposed.

## 2026-10-04 - Opt-in count-space uncertainty experiment

The user authorized implementation and testing of the proposed count-space
likelihood. Keep the log10/quality encoder, partial convolutions, 6x6 features,
latent 36, 30x30 splines, scheduler and legacy default loss unchanged. Use the
original count errors and explicit retained-training RMS channel scales, with
censored probabilities for observed zeros. Preserve physical-output zero
clipping; the likelihood operates on a pre-clipping corrected-count mean.

This is an approximate CS-level Gaussian observation model, not a raw Poisson
or exact native-WIC clipping/averaging model. The initial discrepancy fractions
0.5/0.5 are experimental, not an adopted optimum. Numerical and real-data smoke
tests pass; scientific promotion requires the matched held-out and frozen sparse
conductance comparisons recorded in E06 of the live test plan. Full training
has not been run. Inactive uncertainty settings now fail instead of silently
using another loss. No upstream modifications are authorized by this experiment.

## 2026-10-05 - Table uncertainty integration; investigate other bottlenecks

The user requested pausing uncertainty integration to avoid a loss-formulation
rabbit hole and focus on other potential model bottlenecks. Defer E03/E06
follow-ups, hybrid-loss implementation and uncertainty encoder channels; retain
their evidence and existing optional code. This is a pause, not a conclusion
that measurement uncertainty is uninformative.

E04's matched free-spline, optimized-latent and encoder-mean reconstruction
diagnostic is the proposed next step, using the corrected corpus and legacy
loss. Deterministic subset overfitting can follow if training remains unresolved.
Conductance covariance/sparse assimilation remains the scientific endpoint.
No architecture change, new training run or default promotion is authorized
by this prioritization. The live test plan owns execution order and closeout.

## 2026-10-05 - Implement matched E04 fitting oracles, without model changes

The user authorized the encoder / optimized-latent / free-spline diagnostic.
Keep the saved model, legacy loss, transform, masks and quality unchanged;
optimize only latent vectors or directly fitted coefficients. Oracle columns
use the targets and must not be presented as deployable held-out reconstruction
or covariance assimilation. Freeze exact frame identities/data hashes for reuse,
inspect multi-start convergence and preserve the encoder fit as a fallback.
Deterministic subset retraining and architecture changes are not included.
The first real smoke verifies execution; it does not identify the dominant
bottleneck or promote any setting. E04 owns the full comparison and closeout.

## 2026-10-05 - Matched deterministic subset overfit, not a new VAE run

The user authorized updating the overfit test after clarification: no separate
beta-zero/holes pipeline run is needed. Reuse E04's exact training frames,
transform, masks, weights and source model/basis; train a separate from-scratch
diagnostic model with unchanged architecture/latent size, beta 0, no sampling
and no artificial holes. Do not default to the old 36/72/144 capacity sweep.
Restore the best eval-mode TRAINING-subset checkpoint and compare with the
matched source encoder, optimized latent and free-spline oracles. Protect the
original model and outputs; explicitly mark saved overfit models as diagnostic.
These results test fitting ability, not held-out skill or prior covariance.
No production beta/default, capacity change or uncertainty integration is adopted.

## 2026-10-05 - Free-spline fitting is not a production VAE target

The user explicitly distinguishes the diagnostic from the scientific model:
direct independently fitted splines overfit the observations and that degree
of fitting is not desired in the actual VAE. Deterministic beta-zero subset
overfitting probes whether the framework can represent/follow the observations
without KL or latent-sampling pressure; it is not an adopted training recipe
or a demand that production reconstructions approach the free-spline floor.
Judge production changes by generalization and useful conductance covariance
for sparse assimilation, not proximity to this training-error comparator.
No new run, architecture change or default change is authorized here.

## 2026-10-05 - Opt-in matched decoder spatial-resolution diagnostic

The user authorized implementing/verifying the proposed decoder spatial-stage
test without changing production defaults. Keep latent 36, encoder 6x6,
30x30 cubic spline controls, parameter count, exact E04 training frames,
weights/transform and deterministic fitting protocol fixed. Compare second
convolution features at 30x30 against the existing 18x18 control; reuse the
completed 2,000-epoch control in matched plots with provenance/data guards.
Saved candidate configuration must retain the spatial setting despite unchanged
parameter shapes, and old/default models must load and compute unchanged.
This authorizes the diagnostic option and short verification smokes, not a
production-resolution adoption, uncertainty change or upstream edit. The test
plan owns full-run results and any proposed later covariance evaluation.

## 2026-10-07 - Fix only unsupported coefficients in the user's new2 script

User copied step_1_vae_wic_s12_new.py to step_1_vae_wic_s12_new2.py and
authorized the hard inactive-coefficient mask, not targeted regularization.
Determine support from actual original training images before padding; fix
spline directions that affect no observed training pixel to0 inside the decoder.
Preserve latent/encoder/spline dimensions, existing padding, weight convention,
uncertainty configuration, loss and training settings for this single-aspect
comparison. The mask must cover both output channels, training and latent draws.
This is not evidence of calibrated covariance outside the observed footprint,
nor a promise that weakly supported coefficients are fixed. Next is the user's
run/review; weighted regularization remains a separate pending choice.

## 2026-10-07 - Presentation covariance summary layout

User accepted the ensemble mean in column 1 and four joint WIC/SI12 PCA
patterns in columns 2--5, each scaled by sqrt(eigenvalue) to represent a
one-standard-deviation departure in counts. Use WIC above SI12 and an
axes/title-free 2x5 layout; PCA colours are signed about zero, not positive
sample images or mean-plus-pattern. Use all saved draws from the canonical
smoothed beta-1.6 hole run. Preserve raw-count covariance without silently
normalizing the channels; larger WIC variance dominates this particular PCA.
Subsequent presentation update: add sensor ensemble mean / PC-number-and-
joint-explained-variance text above row 1 and below row 2; axes remain off.

## 2026-10-07 - Single and joint input teleconnection summaries

User authorized figure3a WIC-anchor and figure3b SI12-anchor signed
correlations, plus figure3c fraction of variance explained jointly by both
at the same selected spatial cell. The combined quantity is multiple linear
R-squared, bounded0--1 with no unique sign, not an average of signed maps.
No actual observation values or noise model are supplied; no assimilation
is performed. Use a spatial-only zero-based flattened grid ID for both
anchors and the full saved ensemble. Standardize anchor covariance before
pseudoinversion to handle different count scales and redundant predictors.

## 2026-10-07 - Held-out half-image covariance reconstruction figure

User requested2x4 WIC/SI12 truth, half removed, reconstruction and inferred
uncertainty, using log-space PCA1--50 rather than the VAE decoder. Use held-out
test/validation only and full saved prior draws, transform each draw before
computing log moments/modes; removed truth must not enter OI. Implementation
defaults are right half and assumed0.1 dex independent observation STD (legacy
OI), both exposed; this is not adoption of a calibrated error model or measured
dwic/dsi12. Display physical conditional median and formal marginal STD in
counts, with explicit clipping/assumptions recorded. Report model outliers,
not just display-limited maps. First real frame has severe WIC extrapolation;
no successful general reconstruction or calibrated uncertainty is established.

## 2026-10-07 - Entire-sensor reconstruction comparison

User requested figure5a with SI12 entirely absent and figure5b with WIC
entirely absent, otherwise matching figure4's covariance-only log-space
PCA1--50 reconstruction and2x4 panel layout. Reuse one prior and the same
held-out pair for both; missing-sensor truth must not enter conditioning,
nor be represented as zero observations. Keep exposed0.1 dex error assumption
and formal posterior-STD definition unchanged; report outliers/statistics.
No model/decoder change or uncertainty calibration is implied by these figures.

## 2026-10-07 - Add coefficient-space covariance reconstruction variants

User authorized coefficient-space variations of presentation figures4/5.
Implement as an opt-in prior-space setting in the existing scripts, preserving
pixel-PCA defaults and the same held-out selection, masks, errors and layout.
Use the stored latent draws with the saved model, restore training-standardized
coefficients to raw log-count units before joint coefficient PCA, then project
retained eigenfactors through the saved spline bases for legacy G Q G.T OI.
Do not clip physical fields before constructing this coefficient prior or
generate new draws. Keep50 modes for matched comparison;100 remains available
to match archived mode count. Report coefficient explained variance as such,
and retain coefficient prior arrays/provenance for inspection. This authorizes
the variant and verification, not a new model adoption or uncertainty claim;
the first real half-image example still fails despite lower WIC RMSE.

## 2026-10-07 - Add decoder-only latent-fit reconstruction comparison

User authorized decoder-based figure4/5 variants after the coefficient-PCA
comparison. Fit a latent vector through the frozen saved decoder using only
retained observations, not an encoder or hidden sensor/half; decode both full
fields. Keep covariance as default and existing masks/layout/selection.
Assume0.1 dex independent log errors and a standard-normal latent prior with
exposed precision separate from training beta. Seeded multi-start fitting
and explicit convergence reporting are required;2,000 default iterations
address the500-iteration limit seen in the real half-image verification.
For the fourth column use explicitly labelled local Gauss-Newton/Laplace
latent covariance propagated through the decoder, not a claim of full
posterior/calibrated prediction uncertainty. Save latent parameters and
restart diagnostics. No retraining, loss change, model adoption or uncertainty
integration into training is authorized by this comparison. Frame210 peak
underfitting persists despite removing huge covariance extrapolations.

## 2026-10-07 - Conductance presentation reconstruction variants

User requested Hall/Pedersen versions of the held-out figures using log-space
covariance reconstruction, not a conductance decoder. Implementation preserves
the saved FUV frame index and converts corrected grid-WIC with the SAME reviewed
fixed-2-keV Robinson adapter as the saved conductance ensemble. This baseline
reference choice was explicitly disclosed: NOT native conductance_cs or
independent conductance truth, and not equivalent to upstream detector-first
conductance gridding. No upstream/physics/model/training change is authorized.
Use joint log10(S+1) pixel PCA,50 modes/assumed0.1 dex default, exposed rank/
error/half/scales and2x4 physical median/formal SD(S) figures.5a removes
Pedersen,5b Hall. At fixed energy H/P are proportional; whole-field recovery
must not be claimed as independent cross-field skill. Formal uncertainty is
not measured dH/dP or calibrated model error. Real frame210 still underfits
peaks; this implements the comparison, not adoption of a validated model.

## 2026-10-07 - Direct native conductance_cs VAE alternative

User authorized a direct Hall/Pedersen pipeline preserving current VAE
architecture/loss/default settings, with uncertainty opt-in. Reuse model,
dataset and training; load H/P/dH/dP and declared auroral quality via icReader.
No extra weight inversion, conductance conversion, imposed energy or upstream
changes. Distinct native artifact/units prevent FUV misinterpretation. Preserve
orbit-held-out/source-index selection/33% cutoff. Current native files lack
SZA: refuse rather than invent/misalign a filter. Export domain determined
ONLY from retained training inputs, masked before exponentiation/covariance,
with explicitly tagged storage placeholders, not fake background training
targets. Weakly observed corners remain separate. Tiny operational verification
does not establish full native model skill/calibration or scientific adoption.

## 2026-10-14 — Relax automatic CVAE intrinsic frame coverage

Confirmed by user: require at least5% intrinsic valid Hall/Pedersen coverage
before automatic masking, to retain usable partial-footprint frames. This
corresponds to allowing95% missing, replacing the67%-valid preparation default.
Implemented for future automatic conductance caches with explicit legacy0.33
maximum-missing option. Existing cache/checkpoint remain at the historical
rule; no full rebuild/training requested or started. Positive masked-quality
and pixel-validity requirements remain. Real cache smoke and loader checks
pass; [[WIC Conductance Browsing Diagnostic]] owns evidence and reproduction.

## 2026-10-15 — Model 1 accepts all-accepted manual Partial masks

User confirms that a manual Partial frame with all pixels accepted must count
as Accepted for the whole-frame Model1 task. Figure reference uses all observed
WIC pixels accepted and preserves the saved annotation label. Future Model1
training must follow that effective target; existing fitted scores remain a
historical trial. Model2 is the frozen pixel-mask U-net. Ownership and exact
counts: [[WIC Conservative Full Frame Gate Plan]].

## 2026-10-15 — Try Model 1 full releases with Model 2 fallback

After inspecting accepted-Bad examples, user considers them less problematic
than expected and authorizes trying the two-model approach. Use latest retrained
Model1 acceptance+veto with frozen cutoffs; release all available pixels only
on passes, keep current Model2 U-net masks otherwise. Export separate full-corpus
masks and a new5%-valid CVAE cache; preserve previous products/runs. Visual
approval permits the experiment despite known manual-label errors; it does not
relabel saved Bad annotations or establish a physical-noise guarantee. New CVAE
training remains the user's next step with a ready matching command. Evidence,
small fixed-policy coverage gain and artifact paths: [[WIC Conservative Full
Frame Gate Plan]].
