# Current State

Last reviewed: 2026-07-27
Repository snapshot: `main` at `f0a139a`, aligned with `origin/main`
Worktree: dirty with the current IMAGE development, project-memory updates,
and the user's loader/cache changes; preserve all of it

## Current position

**Confirmed by the user:** The intended project is an empirical model of
height-integrated Hall and Pedersen conductance using IMAGE-derived data and a
variational autoencoder. The spatial covariance representation is especially
important for later ionospheric-electrodynamics data assimilation.

The user considers this the quickest project to turn into a publication. The
AGU 2026 abstract is due 2026-08-05, and the associated paper is targeted for
December 2026.

## Verified repository evidence

- `main` tracks `origin/main`; both are at `f0a139a`, committed on 2026-07-27
  as `Checkpoint after prolonged pause of project`.
- The checkpoint incorporated the previously dirty research scripts, selected
  data and figures, and the project vault. The worktree was clean before the
  2026-07-27 memory update.
- All 43 Python files under `scripts/` pass a read-only AST parse as of the
  annotation-tool checkpoint.
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

## Working lineage

- **Current primary:** log-space spline VAE, joint coefficient covariance, and
  control-space OI in `scripts/IMAGE/primary_log_spline_vae.py`.
- **Completed:** correct cross-channel correlation normalization and
  unit-consistent latent assimilation.
- **Deferred:** restore Hall-only, Pedersen-only, and joint-observation
  experiments from the direct-pixel predecessors.
- **Deferred:** add adaptive rank selection, posterior uncertainty,
  orbit-separated quantitative validation, metrics, saved model/configuration
  state, and reproducible figure provenance.
- **Skipped by user decision:** iterative PCA as a baseline.

## Development roadmap

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

Read-only inspection confirms that the user's current trial is saving to
`/tmp/icAnalyzer_annotation_trial.json`: it contains two orbit records, and
orbit 169 has fully good `0:51`, usable stop 233, and saved keyframes. Because
`/tmp` is not durable across cleanup or reboot, this file should be copied to
the default project annotation path before it becomes production annotation
work.

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
- the intended abstract claim, final figures, co-authors, and target journal.

## Risks

- The current scripts mix exploratory model, covariance, assimilation, and
  plotting code in monolithic executable files.
- No current IMAGE workflow saves its model, normalization statistics,
  configuration, split definition, metrics, or figure provenance.
- Sequential first-10,000-frame training and hand-picked later indices do not
  establish orbit-independent generalization.
- Several scripts change the process working directory, use hard-coded local
  paths and global decoder state, or silently discard data-loading exceptions.
- Open-ended model experimentation could consume the short abstract window.

## Next action

Have the user try the new dividing-line editor and confirm its orientation,
side-flip behavior, and interaction speed. Before continuing substantive
annotation, migrate the successfully saved `/tmp` trial JSON to the durable
default project annotation path so a cleanup or reboot cannot remove it.

After the interface is accepted, finish the processed-data contract: combine
manual validity with finite-value validity and continuous `w`; preserve
provenance, orbit/time, condition variables, split assignment, and
normalization metadata; and select the versioned processed-data format. Then
scaffold the remaining data, training, initial-analysis, and
advanced-analysis stages. The deferred independent-versus-joint experiments
and validation/uncertainty package remain required later.
