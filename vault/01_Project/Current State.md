# Current State

Last reviewed: 2026-09-25
Repository snapshot: `main` at `7c71591`, aligned with `origin/main`
Worktree: nine tracked files are modified, comprising annotation progress, two
exploratory script changes, a generated Python cache, repository guidance, and
project-memory updates. Relevant untracked paths include generated overview
figures, a local literature directory, the WIC/SI12 experiment, and an
algorithm note; preserve all of it.

## Current position

**Confirmed by the user:** The intended project is an empirical model of
height-integrated Hall and Pedersen conductance using IMAGE-derived data and a
variational autoencoder. The spatial covariance representation is especially
important for later ionospheric-electrodynamics data assimilation.

The user has described this as the quickest project to turn into a
publication. The AGU 2026 abstract deadline was 2026-08-05, but the repository
does not record whether an abstract was submitted or accepted. The associated
paper remains targeted for December 2026 in the latest confirmed project
context.

**Confirmed desired direction, 2026-09-25:** update `icAnalyzer` for the
regenerated upstream dataset and develop a WIC/SI12 VAE that supports latent
sampling, WIC/SI12 covariance analysis, nonlinear propagation of sampled
corrected WIC to Hall/Pedersen, joint conductance covariance, and sparse
conductance reconstruction. The user's proposed preservation commit and
dedicated VAE branch remain to be confirmed after technical review.

## Verified repository evidence

- `main` and the locally recorded `origin/main` are both at `7c71591`, a
  2026-07-28 checkpoint that committed the primary log-spline working copy,
  loader correction, annotation implementation, initial annotations, and
  project-memory updates.
- The tracked worktree now has nine modified files: `AGENTS.md`, the
  annotation JSON, `scripts/IMAGE/helpers.py`, its tracked bytecode cache,
  `scripts/make_overview_images.py`, and four project-memory files.
- All 44 Python files under `scripts/` passed a read-only AST parse on
  2026-09-25. No scientific pipeline was run.
- The annotation JSON loaded under schema version 1. Boundary/completion
  checks succeeded for all 53 represented orbit records, and all 469 stored
  masks decoded with valid fractions matching their recorded values.
- The `scripts/IMAGE/` experiments form a progression from iterative PCA and
  direct-pixel VAE covariance, through direct-pixel manual reconstruction, to
  physical-space and then log-space spline VAE reconstruction.
- `scripts/IMAGE/doing_it_manually_log_spline.py` is the strongest
  architectural candidate: it learns normalized log10 conductance through
  spline control points and constructs joint Hall-Pedersen covariance in
  coefficient space.
- Its primary coefficient-space OI is internally consistent, but the later
  latent-assimilation experiment mixes raw physical observations with
  normalized log-space model output and does not convert its final log-space
  field back to conductance.
- Cross-channel teleconnection plots in the manual variants normalize
  cross-covariance blocks as if their diagonals were variances and are not
  trustworthy.
- `scripts/IMAGE/doing_it_manually.py` contains recoverable independent-versus-
  joint OI comparisons and per-image reconstruction-error machinery that the
  spline variants dropped.
- `scripts/IMAGE/primary_log_spline_vae.py` is now the working copy of the
  log-spline experiment. The original experiment remains unchanged.
- The primary script fixes cross-channel covariance normalization and
  reformulates latent assimilation in normalized log10 units with
  physical-unit, non-negative outputs.
- Synthetic tests in the `tConductance` environment verified the
  cross-correlation calculation against `numpy.corrcoef` and verified that a
  differentiable mock decoder converges to the expected physical Hall and
  Pedersen fields.
- `scripts/IMAGE/gen_cov_iterative_PCA.py`,
  `scripts/IMAGE/gen_cov_iterative_VAE.py`, and
  `scripts/IMAGE/linear_reconstruction.py` unpack five values from
  `helpers.load_data`, which now returns six, so their main blocks fail at data
  loading without repair.
- Many scripts reference `icReader` classes or interfaces.
- No maintained automated test suite or CI workflow was found.
- On 2026-07-26, a read-only AST parse succeeded for all 39 Python files under
  `scripts/`.
- The live modular product boundary is `icBuilder` products loaded through
  `icreader.load()`, with array physics supplied by `icPhysics`; `icAnalyzer`
  should not import `fuvpy` or `icBuilder` directly.
- The current fixed CS grid is `image_apex_130km_46x46_v1`.
  `precipitation_cs` and `conductance_cs` already exist, with icBuilder writers
  and icReader dispatch/classes.
- **Confirmed upstream contract decision, pending implementation
  verification:** Product 2 will include corrected WIC, corrected SI13, SI12,
  and their uncertainties, and carry those fields into `precipitation_cs` for
  icAnalyzer. A separate `fuv_cs` product is not required.
- The `precipitation_cs.wic_corrected` field has already undergone proton
  correction in icBuilder. icAnalyzer must not call proton correction on it
  again. The 2 keV proton-energy assumption belongs to upstream Product-2
  provenance; the 2 keV electron-energy assumption is used when propagating
  decoded corrected-WIC samples to conductance.
- The untracked `step_1_vae_wic_s12.py` is nearly a copy of the conductance
  VAE with only its loader changed. Its downstream names, transformations,
  covariance analysis, and sparse reconstruction still treat WIC/SI12 as
  Hall/Pedersen, and it contains no propagation to conductance.
- `icPhysics` can already calculate Hall/Pedersen conductance for arrays, but
  the reusable WIC/SI12 precipitation boundary needed for a 46x46 ensemble is
  not yet confirmed through its public API. Do not change icPhysics until all
  icBuilder call sites and parity tests have been reviewed; first determine
  whether existing functions can be composed unchanged.
- The first propagation baseline now fixes both proton and electron mean
  energy at 2 keV. Energies are not sampled in the first ensemble; detector
  uncertainty remains separate from learned WIC/SI12 variability.

## Working lineage

- **Current primary:** log-space spline VAE, joint coefficient covariance, and
  control-space OI in `scripts/IMAGE/primary_log_spline_vae.py`.
- **Proposed successor:** a clean WIC/SI12 workflow built against modular
  46x46 products, with propagation of sampled corrected WIC and ensemble-space
  joint conductance reconstruction. Do not treat the copied WIC/SI12 script as
  this workflow.
- **Completed:** correct cross-channel correlation normalization and
  unit-consistent latent assimilation.
- **Deferred:** restore Hall-only, Pedersen-only, and joint-observation
  experiments from the direct-pixel predecessors.
- **Deferred:** add adaptive rank selection, posterior uncertainty,
  orbit-separated quantitative validation, metrics, saved model/configuration
  state, and reproducible figure provenance.
- **Skipped by user decision:** iterative PCA as a baseline.

## Development roadmap

The WIC/SI12 direction supersedes starting new work directly from the legacy
36x36 loader. The proposed sequence, pending confirmation of the preservation
and branch plan, is:

1. **Stable data and physics contracts.** Verify the updated Product-2 and
   `precipitation_cs` schemas expose corrected WIC, corrected SI13, SI12, and
   their uncertainties through icReader. Audit the existing icPhysics and
   icBuilder path before proposing any API change; preserve numerical parity
   and never repeat proton correction in icAnalyzer.
2. **Processed dataset.** Preserve WIC, SI12, their measurement uncertainties,
   binary validity, continuous quality weights, orbit/time, grid identity,
   split assignment, and exact upstream repository/product provenance.
3. **Defensible unconditional VAE.** Use orbit-separated validation, a
   two-channel mask-aware loss with consistent per-sample scaling, and a
   reloadable artifact containing normalization, grid, configuration, and
   split identifiers.
4. **Paired sampling and propagation.** Decode paired corrected-WIC/SI12
   samples and propagate the corrected-WIC component through the reviewed
   nonlinear conversion before estimating Hall/Pedersen statistics.
5. **Covariance and reconstruction.** Keep transformed/count/physical spaces
   explicit, retain a low-rank ensemble anomaly representation, and validate
   joint sparse conductance reconstruction with explicit observation errors
   on held-out orbits.

The earlier broader roadmap remains relevant after this baseline, especially
for a conditional VAE and publication-scale evaluation:

The next phase should be sequenced as follows:

1. **Annotation and data contract.** Replace binary whole-image rejection with
   an image-level quality class plus a spatial validity mask so useful regions
   of partially corrupted images can train the VAE. Keep binary validity
   distinct from continuous measurement/confidence weights.
2. **Modular data pipeline.** Create separate stages for importing,
   annotating/cleaning, visualizing, and saving a versioned processed dataset;
   training and saving a VAE; loading it for initial covariance and
   reconstruction analysis; and advanced comparisons with other conductance
   models.
3. **Defensible unconditional VAE.** Review the loss scaling, KL treatment,
   architecture, missing-data mechanism, optimization, and validation before
   treating the current model as a scientific baseline.
4. **Conditional VAE.** Condition the encoder and decoder on selected
   geomagnetic-activity descriptors. For requested conditions, sample the
   conditional generative distribution to estimate a representative field and
   covariance. Do not assume that decoding `z=0` is the statistical mode
   without validation.
5. **Advanced evaluation.** Compare against other conductance models and
   complete the previously deferred cross-channel experiments, adaptive rank,
   posterior uncertainty, orbit-separated metrics, and reproducible artifacts.

Exact activity descriptors and storage formats are not yet selected.

## Partial-image annotation implementation

The user currently reviews 6x6 overview pages and assigns at most one good
sequence per orbit. There are 11,098 overview PNGs under the external `ncso/`
directory and roughly 400,000 source images, so per-image or per-pixel manual
annotation is infeasible.

Inspection of `orbit_0169_072.png` and the preceding page shows a common
pattern: the auroral structure remains useful while an edge-connected bad
region grows gradually from the lower image boundary.

The first annotation implementation is now under
`scripts/IMAGE/vae_pipeline/`:

- `annotate_partial_images.py` displays live WIC, Hall, Pedersen, or weight
  arrays in a clickable 6x6 orbit view;
- four boundaries encode `bad | partial | fully good | partial | bad`;
- a double-click opens an enlarged four-channel editor in which a lasso
  selects the valid region;
- each sparse mask becomes active at its keyframe and is held unchanged until
  a later keyframe replaces it; there is no temporal mask interpolation;
- the existing segment CSV is read only as a seed and is never overwritten;
- annotations and the exact orbit/page/frame cursor are atomically autosaved
  in a separate, versioned JSON document;
- ordinary launch resumes the saved cursor, while `--start-orbit` and
  `--start-frame` permit an explicit restart point.

Headless verification in `tConductance` loaded and rendered real orbit 169,
saved a mask keyframe, completed the orbit, and recovered its exact cursor
from a new JSON load. Store tests covered mask RLE, temporal propagation,
two-sided partial-interval validation, atomic round trips, no-usable reset,
and the corrected orbit-169 seed.

The first desktop trial exposed confusing boundary terminology: the user
reasonably interpreted `usable start` as the start of a new partial section,
but it is the first usable frame of the entire orbit and therefore cannot
follow an existing fully good start at frame zero. The controls now say FIRST
and LAST explicitly, provide a targeted explanation for that invalid action,
and instruct the orbit-169 workflow to extend only LAST usable. The corrected
case was verified headlessly; it still needs another desktop trial.

The second trial exposed a separate editor-launch bug. The backend check
mistook Qt5Agg/TkAgg for the non-GUI Agg backend because their names contain
`agg`, so the new editor figure was created but never explicitly shown. The
check now distinguishes exact Agg, raises Qt/Tk editor windows to the
foreground, and reports construction failures in the main status panel.

The user also rejected temporal mask interpolation. Current semantics are
forward-held keyframes: the first partial frame must have a mask, which remains
active until another keyframe replaces it. For orbit 169 with usable stop 233,
frame 52 must have the initial mask; a later keyframe changes the mask from
that later frame onward.

The mask editor now supports both irregular lassos and additive half-plane
cuts. Each new dividing line intersects its valid side with all earlier valid
sides, producing a convex retained region. The newest line is cyan, older
lines are white, and controls flip or undo the last line or clear all cuts.
All endpoints, side choices, and the raster mask are stored together.
Real-data headless round trips verified old single-line compatibility,
multi-line intersection, flip-last, undo-last, clear, JSON save, fresh load,
identical masks, and restored geometry. Dividing-line mode is the default for
new keyframes; existing saved lassos still reopen in lasso mode.

A later attempt to add a keyframe at frame 58 exposed two avoidable line-input
restrictions: the divider accepted only press-drag-release and only on the WIC
panel. It now accepts dragging or two endpoint clicks on any of the four
aligned panels. A regression using the live trial state verified a drag on
Hall, two clicks on Pedersen, and persistence of the new frame-58 keyframe.

The user changed `scripts/IMAGE/helpers.py` so a missing good start defaults to
frame zero. They report that this increases loaded data from 3,664 to 11,136
images, and they increased the primary training subset from 3,000 to 10,000,
with a substantial speed cost. These counts and training behavior are
user-reported rather than independently regenerated here.

The production annotation file is `data/partial_image_annotations.json`.
Read-only inspection on 2026-09-25 found 53 represented orbit records: 49
complete and four in progress, with the cursor at orbit 167. The file contains
469 mask keyframes across 50 orbits and was last saved on 2026-08-19. Compared
with `HEAD`, 46 orbit records changed and 45 moved from in progress to
complete. This is substantial annotation progress, but the number of orbits
discoverable from the external overview/data directories was not enumerated,
so it is not evidence that the full annotation campaign is nearly complete.

### TODO: grid-independent annotation format

The current schema stores 36x36 raster masks and dividing-line coordinates in
array-index space. Replace this with a grid-independent format that stores
annotation geometry in cubed-sphere coordinates, together with enough grid
metadata to reproduce a raster mask on any chosen grid.

Write a one-time converter from the existing annotation file to the new
format. Dividing-line endpoints should be mapped from the old pixel grid into
cubed-sphere coordinates and evaluated as infinite half-planes on the new
grid. The two keyframes using the older single-line geometry may be discarded
if supporting them is not worthwhile.

Inspection on 2026-08-18 resolved the 15 keyframes without a `geometry` field:
nine are all valid, two are all invalid, and the remaining four retain lasso
vertices. They can therefore be migrated exactly; there are no partial
keyframes represented only by an irrecoverable raster mask.

## WIC/S12 input exploration

The dirty worktree adds `load_data_wic` to `scripts/IMAGE/helpers.py`. It loads
WIC and S12 means and standard deviations, weights, times, orbit identifiers,
and the grid using the same good-segment selection as the conductance loader.
The existing `scripts/IMAGE/step_1_vae_wic_s12.py` consumes this helper, ignores
the two returned standard-deviation arrays, and passes WIC and S12 means into
variables and model paths still labelled Hall and Pedersen. Static inspection
therefore shows an active WIC/S12 experiment, not a verified conductance-model
pipeline or scientific result.

`scripts/make_overview_images.py` has also been redirected from `conductance`
to `conductance_new` and to new overview-output directories. These path changes
were inspected but not executed. The helper still catches all per-orbit
exceptions silently, and the overview script remains a hard-coded interactive
workflow; both require provenance and failure checks before their outputs are
used scientifically.

## Kp-dependent electron mean energy

The user added `literature/zhang_paxton_2008.pdf`. Review confirms that the
paper provides full Kp-dependent `E0` and energy-flux maps as functions of
magnetic latitude and MLT.

Figure 8(b) is instead a flux-thresholded global average derived from the full
model. A frame-constant value taken from that curve is suitable as a simple
baseline, but it discards large spatial variations and should not be the
preferred replacement without validation.

The user reports through a colleague that Fasil has implemented Zhang-Paxton
and found that the raw model can misplace the auroral oval in magnetic
latitude. Fasil reportedly normalizes latitude so that the model contributes
primarily MLT dependence while magnetic-latitude information comes from the
images. The exact method is unknown.

The automatic pipeline cannot accommodate manual oval registration or a
fragile per-frame oval fit. Fasil's method is viable only if it is fully
automatic and robust at dataset scale. Coordination must first determine
whether he detects and rescales each oval automatically or instead removes the
model's latitude dependence without per-frame registration.

See `vault/02_Algorithm/Zhang-Paxton Kp Auroral Model.md` for the model,
pipeline compatibility, limitations, and an appendix-formula caution.

A direct audit of the public `habtie-phys/auroraloval` repository found a
plausible but partial third-party transcription. Its energy-flux coefficients,
Epstein form, hemispheric-power interpolation, and example morphology agree
with the paper, and a 2026 *Scientific Data* paper cites it for oval-boundary
validation. However, it implements only `Q` and threshold boundaries, not
electron mean energy `E0`; it is unlicensed and contains untested endpoint,
southern-hemisphere, NumPy-compatibility, packaging, and path failures. It
must not be used unchanged as the production implementation.

## Source geometry and alternative-model boundary

**Decision confirmed by the user:** Global auroral snapshots are important to
the project. IMAGE remains the primary observational source because the VAE is
intended to learn global spatial covariance. GUVI's retrieved mean-energy and
energy-flux products are orbital swaths rather than simultaneous global
states, so GUVI should be evaluated as an external calibration or validation
source rather than as a replacement dataset.

Zhang-Paxton is not the only empirical precipitation model. OVATION Prime is
the strongest immediately practical comparison because it provides global
average-energy and energy-flux patterns, has public implementations, and uses
solar-wind/IMF inputs that should be recoverable for the IMAGE interval.
ASHLEY-A is a newer spectral alternative based on 19 DMSP energy channels, but
its code availability and batch-operability remain unknown. Newer FTA,
PrecipNet, and SSUSI/XGBoost models improve selected morphology or energy-flux
tasks but do not directly supply the required mean-energy and energy-flux pair.

## User-reported primary run

On 2026-07-27, the user ran the new primary script and reported:

- average training loss decreased from about 397 at epoch 1 to about 125 at
  epoch 352;
- it then remained near 125 through epoch 1000;
- the behavior and resulting reconstruction were very similar to
  `doing_it_manually_log_spline.py`.

The similarity is expected because the primary copy did not change model
training or coefficient-space OI. Static inspection indicates that
`ReduceLROnPlateau` can repeatedly halve the learning rate from `1e-4` to an
effectively frozen value near `1e-8`. The run has not yet been instrumented to
separate weighted reconstruction loss from KL divergence, so posterior
collapse, reconstruction limitation, and ordinary convergence have not been
distinguished.

## Unknown

- whether the spline basis and 36-dimensional latent representation have been
  quantitatively tuned or validated;
- whether weight `w` is the correct quantity for training weights and how
  product uncertainty should define the OI observation-error covariance;
- which generated figures have known inputs and configuration;
- the intended abstract claim, final figures, co-authors, and target journal;
- whether the AGU abstract was submitted or accepted;
- the intended scientific role and validation criteria for the WIC/S12
  experiment and the `conductance_new` products.

## Risks

- The current scripts mix exploratory model, covariance, assimilation, and
  plotting code in monolithic executable files.
- No current IMAGE workflow saves its model, normalization statistics,
  configuration, split definition, metrics, or figure provenance.
- Sequential first-10,000-frame training and hand-picked later indices do not
  establish orbit-independent generalization.
- Several scripts change the process working directory, use hard-coded local
  paths and global decoder state, or silently discard data-loading exceptions.
- The AGU abstract deadline has passed, and submission/acceptance status is not
  recorded in the repository.

## Next action

Follow `vault/02_Algorithm/WIC-SI12 VAE Pipeline Plan.md`.

First confirm and execute the curated repository checkpoint and dedicated VAE
branch. Then verify the updated `precipitation_cs`/icReader contract and the
fixed-energy propagation compatibility gate. Do not change icPhysics unless an
existing-API composition proves insufficient and a separately reviewed change
passes frozen icBuilder parity tests. Verify the first propagation using
upstream proton-corrected WIC and fixed 2 keV electron energy before training
at scale.

Continue annotation from orbit 167 as bounded background work. Convert the
36x36 geometry non-destructively before rasterizing it on the 46x46 grid.
