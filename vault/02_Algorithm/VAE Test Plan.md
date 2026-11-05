# WIC/SI12 VAE test plan and settings ledger

Last reviewed: 2026-10-05
Status: Live experiment register; future tests and adoption criteria are proposed

## Purpose and next action

**Confirmed scientific endpoint, reaffirmed by the user on 2026-10-03:** create
a conductance prior with covariance for data assimilation. Sparse reconstruction
uses covariance/eigenmodes, not the encoder or decoder to infer a field from
observations. The decoder generates prior samples; encoder inpainting is a
secondary diagnostic, not the scientific acceptance criterion.
This document owns the test order, results, and settings carried forward.
Update it after each completed experiment; do not leave decisions in conversation.

**User clarification, 2026-10-05:** approaching the direct free-spline fit is
NOT a production-VAE objective. Independent spline fits overfit the observations;
they are a representation/fitting-capacity comparator only. The deterministic
beta-zero subset test deliberately removes regularization and sampling to probe
fitting ability, not to select the desired production reconstruction detail.
Production improvements must preserve useful geophysical covariance and pass
held-out E08 evaluation; lower training error alone is not an adoption gate.

**Latest action:** E05's full matched decoder-stage diagnostic is complete:
same 32 training frames/2,000 epochs/deterministic beta-zero protocol, latent 36.
30x30 features give quality-log RMSE 0.2048/0.2793 versus 18x18's
0.2167/0.2958 (WIC/SI12), improving 5.49%/5.59% without extra parameters.
Best epoch 1962; last-100 best-loss gain 1.50%, not proven convergence.
This is a modest fitting benefit, not a dominant-bottleneck explanation or
evidence for improved prior covariance. Retain latent 36. These are training errors,
not held-out skill or a demonstration of improved covariance.
E08 remains the scientific endpoint; uncertainty screens are not promoted.
**User direction, 2026-10-05:** table uncertainty integration and investigate
other model bottlenecks first; avoid continuing the loss-formulation rabbit hole.
E03/E06 follow-ups, the proposed hybrid loss and E07 uncertainty inputs are
deferred, not rejected. Preserve their evidence and existing optional code.
**Next action (proposed, approval needed):** test this stage in the REGULARIZED
full-corpus VAE against `wic_si12_smoothed_b16_holes`: same saved corpus/split,
seed 1, beta 1.6, legacy loss, holes p=0.5/size=12, 500-epoch budget/scheduler,
latent 36, encoder/spline dimensions and sampling count. Change only decoder
feature stage; compare held-out reconstruction, prior moments/covariance and
frozen E08 sparse skill/calibration. Do not prescribe 2,000 production epochs
from the subset test. The canonical runner needs an opt-in flag/reuse check
first; only the diagnostic currently exposes this option. No new code/run or
production adoption is authorized by this review. Production defaults unchanged.
Covariance calibration/rank sensitivity remains open; E08 remains the endpoint.
Preserve reference runs and CLI defaults; no new training/setting adoption is
authorized by these screens.

## Working reference and settings carried forward

The reference is `wic_si12_smoothed_b16`, not a scientifically validated final model.
Run paths below are relative to `~/Dropbox/work/data/IMAGE_FUV/vae_runs/`.
The code snapshot is `vae/wic-si12` at `be31406` plus the current dirty worktree;
HEAD alone does not identify the implementation used in a run.

| Aspect | Working setting | Reason / decision status |
| --- | --- | --- |
| Corpus | `precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2` | Explicit, verified auroral-weight orientation; old corpus excluded |
| Selection | Existing CSV intervals; no SZA filter; intrinsic missing fraction <= 0.33 | Match completed smoothed controls; selection is not being swept |
| Orbit split | Saved reference lists; 4,033 / 487 / 1,101 usable train / validation / test frames | Prevent data changes from masquerading as model improvements |
| Inputs / targets | WIC, SI12, quality to encoder; WIC and SI12 targets | Preserve restored legacy lineage |
| Transform / sign | Training-fitted standardized `log10(value + 1)`; clip and report negative counts | Fixed for these comparisons; upstream sign issue remains open |
| Encoder / latent / spline | Partial convolution, 6x6 final features; latent 36; 30x30 cubic controls | Preserve until a diagnostic supports an explicitly approved change |
| Training | Adam, batch 8, LR 1e-4, up to 1,000 epochs; plateau factor 0.5, patience 10 on training total | Same optimization protocol; save minimum-validation-total state |
| Beta | 1.6 for the working reference | E01 did not establish a better overall replacement; CLI default is still 1 |
| Loss | Literal legacy quality-weighted MSE and summed KL; no measurement precision | E03 log and E06 count uncertainty screens not promoted |
| Artificial holes | Off in reference | E02 screening improves inpainting; overall adoption unresolved |
| Seed / samples | Seed 1 for paired screening; 50,000 prior samples for full-run comparison | Keep prior draw count/seed identical; this is not a convergence guarantee |
| Propagation | Hardy proton correction already upstream; electron energy 2 keV downstream | No repeated correction, energy sweep, or upstream code changes |

All changes below apply to experimental runs first. An adopted setting updates
this table and the decision log after user confirmation. CLI defaults change
only by explicit agreement; a setting can be rejected without deleting its code.

## Common comparison and closeout rules

- Change one aspect at a time; keep corpus, orbit lists, selection, transform,
  architecture, batch size, training budget, and sampling fixed unless named.
- Save exact command, code/diff identity, dataset provenance, seeds, run path,
  selected epoch, and comparison control. Flag confounded runs instead of ranking them.
- Use validation for prospective choices. The existing test set has already been
  inspected and is exploratory, not an untouched final assessment. Reserve new
  orbits or a new independent corpus for final scientific confirmation.
- Screen with seed 1. Repeat promising control/candidate pairs with seeds 2 and
  3, keeping orbit lists fixed: changing `--seed` alone also changes the split
  in the present runner, so fixed-split repeats need explicit support first.
- Judge WIC and SI12 separately: auroral-quality normalized-log RMSE, all-valid
  RMSE, physical mean/bias/std, and identical-frame full-input and inpainting figures.
- Report removed-region and retained-region errors separately. Use the same
  saved patches in both models; natural gaps have no known truth.
- Always check prior moments, spatial/cross-channel covariance and spectra,
  and KL/active dimensions. KL is a diagnostic, not a quantity to maximize.
  Do not compare total losses across different objectives as a skill score.
- The primary scientific gate is covariance-based sparse conductance skill
  and uncertainty calibration on unobserved, originally supported cells.
  Inpainting improvements or variance-explained ratios alone cannot pass it.

**Proposed adoption gate:** the test's target improves on validation in both
channels across paired repeats, without a material regression in full-input
reconstruction, mean bias, prior spread, or covariance. Before a run, record
numerical tolerances for its target and safeguards. If a concrete initial rule
is needed, propose >= 10% target-error reduction and <= 5% regression in each
error safeguard; these are decision tolerances, not established scientific laws.
Mixed tradeoffs require a user decision. One-seed evidence remains provisional.

Every closeout records: **result; interpretation; adopt / reject / inconclusive;
settings carried forward; user confirmation; next test.** A smoke pass means
the option executes, not that the scientific hypothesis succeeded.

## Execution queue

| ID | Aspect / question | Status | Decision sought |
| --- | --- | --- | --- |
| E01 | Beta: is KL pressure the main limitation? | Screening reviewed | Keep 1.6 reference; lower beta not promoted |
| E02 | Input holes: can the model learn missing-region reconstruction? | Bright/random-patch screen complete; setting decision pending | Holes on/off; probability and patch size |
| E08 | Conductance covariance and eigenmode/factor sparse assimilation | Reusable test implemented; initial validation screen complete | Comparable point skill; resolve interval undercoverage |
| E03 | Revised uncertainty weighting: can errors inform training without concentration/bias? | Screen complete; further integration deferred by user 2026-10-05 | Revisit after other bottlenecks; no setting promoted |
| E04 | Spline vs learned decoder vs encoder vs training limitation | Three-way, epoch-budget and latent-36/72 screens reviewed; 72 improves only 1–2% | Keep 36; do not interpret free-spline gap as production failure |
| E05 | Targeted architecture/optimization change, selected by E04 | Full decoder-stage diagnostic reviewed: ~5.5% training RMSE gain; regularized test proposed | Does fitting benefit transfer to held-out/prior covariance skill? |
| E06 | Observation errors near zero: is log-space approximation misleading? | Full run/audit complete; further integration deferred by user 2026-10-05 | Preserve evidence; no fraction sweep or hybrid-loss implementation now |
| E07 | Uncertainty encoder channels: does reliability context help? | Deferred by user; not implemented | Three vs five encoder inputs |

## E01 — Beta screening: reviewed, not a final calibration

Only beta changed; same corpus, seed 1, orbit split, and training settings.
Errors below are test-set exploratory auroral-quality normalized-log RMSE.
Prior std uses identical held-out support and 1,101 diagnostic prior draws.

| Run | Beta | Selected epoch | WIC / SI12 RMSE | KL nats/frame | Prior WIC / SI12 std, counts |
| --- | --- | --- | --- | --- | --- |
| `wic_si12_smoothed_b16` | 1.6 | 167 | 0.469 / 0.591 | 29.36 | 243 / 5.41 |
| `wic_si12_smoothed_b1` | 1.0 | 167 | 0.459 / 0.589 | 40.03 | 224 / 4.91 |
| `wic_si12_smoothed_b05` | 0.5 | 86 | 0.451 / 0.567 | 57.55 | 189 / 4.12 |

Observed std is 332 / 7.24 counts. Beta 0.5 improves reconstruction about 4%
but worsens prior underdispersion; removed-patch RMSE is mixed, 0.711 / 0.882
versus 0.718 / 0.816 at beta 1.6. Beta is active, but KL pressure alone does
not explain the reconstruction limitation. This does not diagnose its cause.
**Disposition:** lower beta not promoted; retain 1.6 as comparison reference,
not a claimed optimum. No further beta-only sweep until another aspect changes.
This continuation policy is proposed; user confirmation is pending.
Evidence: each run's `summary.json` and `figures/posterior_prior_diagnostics.json`.

## E02 — Artificial input holes

**Question/why:** does training on deliberately missing observations improve
inpainting rather than merely returning a smooth template?
**Change:** enable `--training-holes --hole-probability 0.5 --hole-size 12`.
Hide all three input channels and the encoder mask together; retain original
observed targets and quality loss weights. Validation remains unaugmented.
**Verified candidate/control:** `wic_si12_smoothed_b16_holes` against
`wic_si12_smoothed_b16`. Both use beta 1.6, seed 1, the same orbit lists,
usable frames, transform, model configuration and data provenance, legacy loss,
and 50,000 pipeline samples. Candidate uses probability 0.5, size 12, RNG seed 4.
Budgets differ (500 vs 1,000 epochs); selected states are epochs 165 vs 167.
The control's best checkpoint within its first 500 epochs is also 167, so
limiting its selection to the candidate budget would retain the same artifact.
**Measure:** fixed 12x12 bright-patch held-out test and a fixed randomly placed
patch test; compare full-input, removed-region, and retained-region RMSE and figures.
Both bright- and random-patch evaluation are implemented in the existing visualizer.
**Decision:** promote holes only if missing-region skill improves without the
common safeguards failing. If mixed, retain off; sweep probability OR size next,
not both.

### E02 screening result — 2026-10-03

Both saved figures use frames 262/637/779 and the same bright-patch policy.
Aggregate scores below cover all 1,101 test frames, not just those examples;
removed support is identical (153,978 cells/channel).

| Metric | No holes: WIC / SI12 | Holes: WIC / SI12 | Interpretation |
| --- | --- | --- | --- |
| Removed-region physical RMSE, counts | 626.22 / 11.21 | 548.80 / 9.51 | Improves 12.4% / 15.2% |
| Removed-region normalized-log RMSE | 0.718 / 0.816 | 0.686 / 0.708 | Improves 4.4% / 13.2% |
| Retained-region normalized-log RMSE | 0.709 / 0.658 | 0.707 / 0.647 | Slight improvement |
| Full-input auroral normalized-log RMSE | 0.469 / 0.591 | 0.467 / 0.612 | WIC unchanged; SI12 worsens 3.6% |
| Prior std, counts (1,101 diagnostic draws) | 242.83 / 5.41 | 247.54 / 4.93 | WIC spread closer; SI12 further below observed 7.24 |
| Posterior mean, counts | 132.79 / 3.26 | 135.15 / 3.08 | Both biases worsen; observed 131.89 / 3.75 |

KL is 27.57 vs 29.36 nats/frame; 35 vs 36 dimensions exceed 0.1 nat.
Prior first-eight-mode variance fraction is nearly unchanged (61.70% vs
61.48%; observed 45.95%); this is not proof of calibrated spatial covariance.
**Conclusion:** metrics support the user's visual impression of better
inpainting, with a modest full-input SI12 cost and poorer SI12 prior spread.
**Disposition:** positive inpainting screen; inconclusive for overall adoption.
**Settings carried forward:** retain holes p=0.5/size=12 as an experimental
candidate; leave the reference and CLI default off. User promotion decision
pending. No architecture, loss, or beta change was made.
**Remaining check for final calibration:** paired training-seed repeats with
fixed splits still need support. Bright/random-patch validation is reviewed below.
No prospective tolerances were agreed before this completed screen; do not
retroactively describe the proposed gate as a passed preregistered test.
**Evidence:** both run artifacts, histories, `summary.json`, and
`figures/posterior_prior_diagnostics.json`; both example figures inspected.

### E02 bright-patch validation — 2026-10-03

The user ran the existing visualizer with `--posterior-split validation` into
each run's `figures_validation/` directory. Both reports confirm 487 frames,
seed 1001, 12x12 peak-signal patches, identical removed support of 68,411
cells/channel and examples 175/273/205. Figures and JSON were inspected;
these are different images from the earlier 1,101-frame test-set screen.

| Validation metric | No holes: WIC / SI12 | Holes: WIC / SI12 | Interpretation |
| --- | --- | --- | --- |
| Removed-region physical RMSE, counts | 776.02 / 11.26 | 714.85 / 9.83 | Improves 7.9% / 12.8% |
| Removed-region normalized-log RMSE | 0.683 / 0.719 | 0.664 / 0.643 | Improves 2.8% / 10.6% |
| Retained-region normalized-log RMSE | 0.688 / 0.629 | 0.672 / 0.606 | Improves 2.4% / 3.6% |
| Full-input auroral normalized-log RMSE | 0.449 / 0.529 | 0.449 / 0.542 | WIC unchanged; SI12 worsens 2.5% |
| Full-input all-valid normalized-log RMSE | 0.678 / 0.612 | 0.672 / 0.602 | Slight improvement; weighting matters |
| Prior std, counts (487 diagnostic draws) | 239.93 / 5.52 | 243.91 / 5.04 | SI12 underdispersion worsens; observed 418.47 / 6.87 |
| Posterior mean, counts | 137.90 / 3.14 | 140.02 / 2.93 | Both biases worsen; observed 130.57 / 3.38 |

Mean KL is 25.86 vs 27.38 nats/frame. Prior first-eight-mode variance fractions
are similar, 61.83% vs 61.61% (observed 55.78%), not a covariance calibration pass.
**Conclusion:** bright-region inpainting gain is reproduced on both evaluated
splits; it is not just an attractive example figure. WIC gain is modest and
SI12 full-input fidelity/prior spread still trade off against inpainting.
**Disposition/settings carried forward:** promising candidate p=0.5/size=12;
overall adoption still inconclusive. Keep the off reference and CLI defaults
unchanged pending the E02 setting decision and training-seed replication.
The follow-up random-patch implementation and result are recorded below.
**Evidence:** both `figures_validation/posterior_prior_diagnostics.json` files
and `heldout_inpainting_examples.png` figures. Validation was already used for
checkpoint selection; this is not a fresh independent scientific assessment.

### E02 random-patch validation — implemented and executed 2026-10-03

The existing visualizer now supports `--inpainting-policy random` and an
independent `--inpainting-seed` (default 0). Default remains peak-signal
`auroral`. Centres are uniform over original valid cells, with edge-clamped
squares; intensity/model seed do not determine placement. One patch per frame;
reject complete removal instead of silently altering the evaluated sample.
Shared input/mask removal preserves original targets and natural-gap exclusion.

Seed 0 was run on both saved models, 487 validation frames across 17 usable
orbits, with identical frame identities, bounds, per-frame counts, and removed
mask SHA256 `1f022f61ab5d1dcba75f4846ed78e60e48a6db500369d13190021aceef616d51`.
The JSON saves those checks and per-frame error sums for paired analysis.

| Random-patch seed 0 metric | No holes: WIC / SI12 | Holes: WIC / SI12 | Improvement |
| --- | --- | --- | --- |
| Removed-region physical RMSE, counts | 416.47 / 4.90 | 338.06 / 4.53 | 18.8% / 7.6% |
| Removed-region normalized-log RMSE | 0.723 / 0.644 | 0.718 / 0.613 | 0.7% / 4.9% |
| Retained-region normalized-log RMSE | 0.685 / 0.622 | 0.674 / 0.605 | 1.6% / 2.7% |

Removed support is 66,859 cells/channel. Additional patch seeds 1 and 2 give
physical-error improvements of 4.65%/9.31% and 13.02%/11.92%; these vary patch
placement, not training seeds. A paired orbit bootstrap (2,000 draws, seed
2026) gives seed-0 physical-improvement 95% intervals of -2.89..21.93% WIC
and 5.25..11.98% SI12. Normalized-log intervals are -1.15..2.52% and
2.54..6.96%. WIC's aggregate physical gain is sensitive to orbit/intensity;
it is not a demonstrated uniform relative-error improvement.

**Conclusion/disposition:** positive inpainting screen on bright and random
patches; SI12 improvement is more consistent. Overall covariance-model
promotion remains inconclusive because the prior/fidelity tradeoffs persist.
**Earlier continuation proposal, not adopted:** holes p=0.5/size=12 as E03's
experimental control. The user's scientific-objective clarification motivates
the proposed E08 comparison first; no hole setting is promoted on inpainting
evidence alone. No defaults or existing model/artifact were changed.
Final calibration needs independent
assessment and training-seed replication rather than more same-model pictures.
**Verification:** 73 tests pass, including deterministic intensity-independent
patches, changed-seed sensitivity, natural gaps, shared-channel removal,
batch-size independence, retained targets and per-frame metric arithmetic.
Eight diagnostic figures per model generated without loading 50,000-sample
prior factors. Extra seed checks did not write or overwrite existing results.
**Outputs:** `figures/vae_runs/e02_random_validation_2026-10-03/` (ignored
generated material), model subfolders and `comparison.json`. The README has
the paired command to reproduce into each run's `figures_validation_random/`.

## E03 — Revised uncertainty loss

**Question/why:** does bounded measurement-error weighting improve inference
without the original experiment's precision concentration and mean bias?
**Change:** `--gaussian-uncertainty --model-error-std 0.5 0.5`, instead of
`--measurement-uncertainty`. Keep the chosen E02 hole setting identical in
candidate and control; beta and architecture also remain fixed.
The fixed allowances are in standardized-log units, not detector errors.
This tests the whole revised objective, not measurement information alone:
attenuating residuals also changes their balance against KL. If promoted,
beta must be calibrated in a separate comparison under that objective.
**Measure:** common skill metrics plus WIC/SI12 effective weight mass, residuals
by intensity/uncertainty, zero-clipped subgroup errors, and latent/prior behaviour.
**Existing result:** original `wic_si12_smoothed_u_b16` has RMSE 0.804 / 0.942
versus 0.469 / 0.591 in the no-uncertainty control; original weighting is not
the preferred model. Corrected revised-loss full runs were evaluated below;
they remain worse for WIC and sparse conductance inference.
**Decision:** retain legacy loss unless the revised candidate passes. If it is
promising, test one common allowance at a time (0.25, 0.5, 1.0); do not also
retune beta or separate channel allowances in the same comparison.
**Settings recommendation:** retain legacy loss for current comparisons;
do not adopt Gaussian allowances 0.5/0.5. No default or baseline change.

### E03 run-configuration audit — 2026-10-04

Historical audit: superseded by the corrected reruns below. These same paths
now contain Gaussian-loss artifacts, so this paragraph no longer describes
their current configuration.

The user reported `wic_si12_smoothed_b16_mes` and
`wic_si12_smoothed_b16_holes_mes` complete. Both `wic_si12_vae.pt` and
`summary.json` record `kind=legacy_weighted_mse_v1` and
`measurement_uncertainty_in_loss=false`, with no model-error allowance saved.
They share the reference corpus/splits/transform/model and have 500 epochs;
selected epochs are 204 and 165, respectively. They are additional legacy-loss
runs, not evidence for or against uncertainty inclusion. Preserve them.

At that audit, live runner inspection and a parse-only reproduction confirmed that supplying
`--model-error-std 0.5 0.5` alone leaves `gaussian_uncertainty=false` and its
values unused. `--gaussian-uncertainty` is required to select the new loss;
the CLI then did not reject this inactive option. The actual user launch
command is not available, so omitted activation is inferred, not confirmed.
Next: confirm the command and use both flags with a new output directory.
Retrofitting an artifact cannot turn completed legacy training into this test.
No retraining, new figures/benchmark run, default or code change was made in
this audit. The subsequent user-authorized E06 implementation now includes
the inactive-option guard; unmatched uncertainty settings are rejected.

### E03 corrected reruns — evaluated 2026-10-04

Both `_mes` artifacts and summaries now record `quality_weighted_gaussian_v1`,
uncertainty enabled, and model-error standard deviations 0.5/0.5 in standardized
log10 units. No architecture, transform, split, selection, corpus provenance or
seed changed against the corresponding controls. All runs have 50,000 prior
members. Hole pairs share p=0.5/size=12 and a 500-epoch budget (selected epochs
165 legacy / 128 Gaussian). No-hole Gaussian has 250 epochs and selects 191,
against the 1,000-epoch control's 167; its first-250 minimum is also 167.
This bounds the saved-control budget issue, but does not rule out a later
Gaussian improvement. One training seed only; no architecture/size change.

Evaluation uses 487 common validation frames. Existing validation figures are
newer than the replacement artifacts; fresh CPU posterior inference reproduces
their neutral, quality and training-weighted RMSE. Both revised runs have
identical bright-patch bounds/identities/mask SHA256, verified against reloaded
data. Old controls' artifacts still match E08's saved SHA256 identifiers.

| Run | WIC/SI12 auroral normalized-log RMSE | Reconstructed WIC/SI12 mean [counts] | Hall/Pedersen sparse RMSE [S], 100 sites | Hall/Pedersen 95% coverage |
| --- | --- | --- | --- | --- |
| Legacy | 0.449 / 0.529 | 137.9 / 3.14 | 0.713 / 0.884 | 59.0% / 58.3% |
| Legacy + holes | 0.449 / 0.542 | 140.0 / 2.93 | 0.711 / 0.882 | 58.9% / 58.1% |
| Gaussian 0.5/0.5 | 0.668 / 0.546 | 198.9 / 3.42 | 0.798 / 0.985 | 39.9% / 37.7% |
| Gaussian + holes | 0.680 / 0.536 | 212.8 / 3.62 | 0.794 / 0.980 | 38.5% / 36.4% |

Observed held-out means are 130.6 / 3.38 counts. Revised WIC auroral RMSE
increases 49–52%; SI12 changes +3.1% without holes / -1.3% with holes. WIC
all-valid RMSE is 1.179/1.219 rather than 0.678/0.672. Validation KL decreases
from 27.38/25.86 to 16.88/17.09 nats/frame. KL is diagnostic, not a target;
changing residual weights also changes their balance with beta.

Bright-patch physical WIC RMSE is almost unchanged (776.0 -> 774.4 counts
without holes; 714.8 -> 718.4 with holes). SI12 is unchanged without holes
(11.26 -> 11.30), improves about 4.8% with holes (9.83 -> 9.35). These mixed
large-signal physical scores do not negate WIC low-count bias or worse log
errors. No new random-patch screen was run in this checkpoint.

The **same frozen E08 fixture** and SHA256 are reused: joint 10/30/100 sites,
0.1 dex synthetic noise, 100 log-conductance grid-space modes. Both revised
priors worsen sparse RMSE at all observation counts: about 14–15% at 10,
11–13% at 30 and 11–12% at 100. Modes retain 98.89%/99.02% of variance.
Prior-only 95% coverage drops to about 36–40% from 86%, so the calibration
regression is not solely caused by conditioning. FUV-derived reference fields
are noisy fixed-energy proxies, not independent true conductance; these scores
do not establish the quality of an underlying noise-free physical model.

**Weight audit:** validation effective mass is 50.74% WIC / 49.26% SI12: the old
between-channel imbalance is largely resolved. But zero-WIC cells have 8.42%
of baseline quality mass and only 0.0782% of revised WIC effective mass, over
100-fold less relative weight. SI12 zero-cell shares change 12.60% -> 5.38%.
WIC's brightest-decile shares change 62.4% -> 72.4%. These are weight sums,
not exact gradients. The strong within-channel suppression is a plausible
mechanism for WIC bias, not a demonstrated causal attribution.

**Disposition:** current allowance 0.5/0.5 is not recommended for adoption.
Measurement uncertainty itself is not rejected: log-variance approximation,
clipped/zero targets, latent noise-free versus noisy-proxy interpretation and
likelihood/model discrepancy need separate examination. Proposed next: E06
low/zero-count likelihood diagnostic before another training sweep; retain
log encoder inputs and architecture. No reformulation or new training was
implemented. Hole + legacy loss remains a proposed experimental control,
not a default promotion.

**Evidence/outputs:** `figures/vae_runs/e03_mes_validation_2026-10-04/` contains
`comparison.json`, four-model `sparse_skill.png`, and `sparse/metrics.json`
with the new two-model benchmark figures and full provenance. Original run
figures and controls were not overwritten. The user replaced the previous
inactive `_mes` artifacts with these corrected runs. No model, CLI/default or
upstream source changes were made by this evaluation.

## E04 — Locate the reconstruction bottleneck before changing capacity

**Current focus, 2026-10-05:** the user requested other bottlenecks take priority
over uncertainty, then authorized implementation. The first three comparisons
are implemented and verified in `diagnose_reconstruction_bottleneck.py`.
The fourth, deterministic subset retraining, is now implemented after a separate
user request. No architecture change or final capacity conclusion follows.

Freeze 32 training and 32 validation frames across orbits, selected by a saved
seeded rule on the corrected corpus; use the reference transform/weights/masks.
Compare full input first, with the same weighted reconstruction objective:

1. Freely fit 30x30 spline coefficients in normalized-log space: estimates the
   representation's approximation floor on observed support.
2. Optimize latent vectors through the frozen decoder, from encoder means and
   several starts: estimates accessible learned-decoder fitting ability.
3. Decode encoder means once: exposes any additional inference gap.
4. Separately overfit the same training subset with deterministic `decode(mu)`,
   beta 0, and no latent sampling: checks learning/optimization under an easier task.

These are target-informed oracle fits, not deployable held-out reconstructions.
Use convergence curves and multiple starts; failure to optimize z is not proof
that no suitable z exists. Diagnostic beta 0 is not a new production setting.
If free splines fit poorly, investigate the spline/data floor. If splines fit
well but optimized z does not, investigate learned decoder capacity/training.
If optimized z fits well but encoder means do not, investigate encoder/inference.
If deterministic overfitting succeeds but the VAE does not, isolate stochastic
training/regularization next. Results on the old weight-invalid corpus are not
substitutes. `diagnose_latent_capacity.py` covers only a deterministic subset
capacity sweep; it is not the matched three-way fitting comparison.

### E04 implementation and smoke — 2026-10-05

**Defaults / controls:** 32 training and 32 validation frames sampled by seeded
orbit-balanced round robin; seed 1, Adam latent LR 0.03, 1,000 steps, three
starts (encoder mean plus two prior-normal initializations), batch 8, automatic
device. Reload the saved corpus/selection/transform and use literal legacy
reconstruction reduction. The diagnostic does not optimize model weights,
change beta/architecture/defaults, retrain a subset, integrate uncertainties,
or load the 50,000-sample prior. Starts are precomputed independently of batch
partition; each frame retains its best joint-channel fit across all steps/starts
and the encoder baseline. No latent prior or KL restriction is imposed on the
oracle; latent norms and convergence are reported to make that limit visible.

Free fitting uses double-precision, rank-revealing QR with `rcond=1e-10`,
the exact saved row/column bases and `column[x,i]*C[c,i,j]*row[y,j]` order.
No ridge penalty is added. The numerical floor is quality-weighted, not an
all-valid or gap-reconstruction bound. Examples mask zero-quality/invalid cells;
all-valid RMSE is separately recorded. Display is log10(counts+1), floored at
zero for display only; metrics use unclipped standardized-log predictions.

**Real smoke:** `wic_si12_smoothed_b16_holes`, CPU, two train frames from
orbits 108/177 and two validation frames from 174/283, 100 steps, two starts.
Quality-weighted standardized-log RMSE:

| Split | Encoder WIC / SI12 | Optimized latent WIC / SI12 | Free spline WIC / SI12 |
| --- | --- | --- | --- |
| Train | 0.3680 / 0.4703 | 0.3426 / 0.4155 | 0.1375 / 0.2251 |
| Validation | 0.5430 / 0.5599 | 0.4718 / 0.4743 | 0.1751 / 0.2315 |

This verifies execution and readable comparisons, NOT the dominant bottleneck:
four frames and 100-step fitting are insufficient, and prior-start curves are
still improving. The full 32+32-frame diagnostic with convergence inspection
is next; no capacity setting is adopted. The saved model SHA256 is unchanged
(`7dd1efaf...02b8d1`), and every parameter/buffer is checked unchanged in memory.

**Verification / reuse:** 129 tests pass, including legacy loss/gradient
equivalence, nonsquare-grid coefficient-axis orientation, ignored masked and
zero-quality targets, deterministic best-retained latent fitting, unchanged
parameters/buffers, orbit-balanced selection, selection/data drift rejection,
overwrite refusal and complete figure/report output. A real validation-only
selection-reuse smoke also passed. Exact frame/data hashes are saved in
`selection.json`; use `--selection` for matching later checkpoints. The README
contains the runnable command. Authoritative smoke outputs:
`figures/vae_runs/e04_bottleneck_verified_smoke_2026-10-05/`.

### E04 full three-way run — reviewed 2026-10-05

**Confirmed:** the user's run under `figures/vae_runs/e04_bottleneck_holes/`
matches the default protocol: 32 train frames across 32 orbits, 32 validation
frames across 17 orbits, seed 1, CUDA, three starts, 1,000 Adam steps, LR 0.03,
batch 8. Control is `wic_si12_smoothed_b16_holes`; artifact SHA256 matches
`7dd1efaf...02b8d1`, script SHA256 is
`376a620afa92af38b597250aadf585eb2c76d6a3a8f532248bf54c0ece1b3807`
and matches live source. Recomputed weighted RMSE from `comparisons.npz`
agrees with the report. Model weights/buffers and artifact are unchanged.

| Split | Encoder WIC / SI12 RMSE | Optimized latent WIC / SI12 RMSE | Free spline WIC / SI12 RMSE |
| --- | --- | --- | --- |
| Train | 0.3366 / 0.4115 | 0.3050 / 0.3786 | 0.1222 / 0.2016 |
| Validation | 0.4602 / 0.5390 | 0.3727 / 0.4268 | 0.1307 / 0.2093 |

Errors are quality-weighted standardized-log RMSE on identical observed
support. Validation latent optimization improves 19.03% / 20.81%; training
improves 9.39% / 7.98%. Joint legacy loss is 1196.07 -> 1000.45 -> 235.15
on train, and 2125.67 -> 1358.68 -> 257.69 on validation. Oracle latent fitting
closes only 20.36% of the encoder-to-spline squared-loss gap on train and
41.06% on validation; the larger residual gap remains in the learned mapping.

**Convergence / inference limits:** best retained joint loss improves only
0.00323% train / 0.11344% validation in the final 100 steps. Largest per-frame
last-window gains are 0.0672% / 0.9141%. This supports achieved local plateaus,
not a proof of global optimality. Prior starts beat encoder initialization in
5/32 train and 21/32 validation frames. Optimized latent norm medians are
8.403 / 9.559, versus encoder means 4.390 / 4.077. 30/32 train and 26/32
validation fits exceed the standard-normal 36-D prior's 95% radius (7.1413).
Thus the oracle's improvement is partly obtained in atypical prior regions;
do not call all of this an encoder bug or promise the same prior-sampling skill.

**Interpretation (inferred):** the existing spline grid can represent much
more observed detail than this saved learned decoder generates. There is an
encoder/inference/regularization gap, but encoder correction alone cannot
explain the full error. Investigate learned-decoder/latent mapping capacity OR
how it was trained; this does not isolate latent dimension, decoder architecture,
KL, stochastic optimization, or generalization individually. Free splines have
far more independent coefficients and can fit measurement noise; their floor
is not a geophysical covariance target. All-valid latent RMSE is also mixed:
validation WIC 0.6734 -> 0.6587, SI12 0.5880 -> 0.5900. No covariance improvement
is established by this target-informed diagnostic.

**Disposition / next proposed test:** keep architecture, production beta 1.6,
legacy loss and uncertainty deferral unchanged. Run the planned deterministic
overfit on the SAME 32 training images, using the saved transform/weights and
architecture, beta 0, no sampling and no artificial holes, in a separate model
copy. If the spline gap substantially closes, training/regularization merits
attention before architecture. Otherwise capacity or optimization remains open;
failure of one optimizer run is not proof that the architecture cannot fit.
The subsequently user-authorized `diagnose_latent_capacity.py` update below
now supports exact-selection reuse. Do not sweep latent dimensions yet.
No production setting or architecture is promoted by this diagnostic.
Test plan, Current State and Handoff are checkpointed; user artifacts/figures
are preserved and no source/model/default/upstream changes were made.

### E04 matched deterministic overfit — implemented 2026-10-05

**Authorization:** after clarifying that no new beta-zero/holes pipeline run
is required, the user requested updating the test script. Updated
`diagnose_latent_capacity.py` uses `--selection` from the completed E04 folder,
including its sibling metrics/arrays. It checks source model SHA256, exact
training-frame identities, selected target/mask/quality hashes and paired arrays
before training. Source encoder is recomputed; latent/spline oracles are reused.

**Protocol:** initialize a separate diagnostic model from scratch (seed 1),
using unchanged saved architecture and exact saved spline basis. Default latent
size is the source's 36, not the previous automatic 36/72/144 sweep. Fit only
the same 32 training frames, saved transform/masks/common quality, literal
legacy reconstruction, deterministic `decode(mu)`, beta 0, no latent sampling,
no input holes, Adam LR 1e-4/batch 8/500 epochs, no scheduler. Optional explicit
dimension sweeps remain available but are not part of this experiment.

Eval-mode full-subset loss is recorded every epoch alongside train-mode batch
mean; the best eval-mode TRAINING-subset checkpoint (including epoch zero)
is restored and saved, not merely the last epoch. BatchNorm/reduction differences
between those curves must not be mistaken for stochastic sampling or different
data. Output diagnostic model is `deterministic_overfit_latent36.pt`, not a
production VAE replacement. Parameters/buffers and source artifact are verified
unchanged. No uncertainty, physics or production-training code is changed.

**Verification:** 137 tests pass. New tests cover exact frame/baseline matching,
unchanged latent size/source model, no stochastic forward or removed inputs,
best-checkpoint restoration after worsening, serialized model/prediction
roundtrip, rejected mismatched baselines, overwrite protection and CLI guards.
A real CPU smoke used all 32 original E04 training frames for TWO epochs,
best epoch 2; the source hash remains `7dd1efaf...02b8d1`, architecture unchanged.
Original source encoder CPU RMSE is 0.336625/0.411478 (within 7e-6 of E04 CUDA),
with exact reused oracles. Two epochs verify execution only, not fitting capacity.
Smoke folder: `figures/vae_runs/e04_matched_overfit_smoke_2026-10-05/`.

**Execution:** the user completed the full 500-epoch matched diagnostic; reviewed
results follow below. Compare convergence and the source/oracle/free-spline gap.
Plots and exact results are in `overfit_comparison.png`,
`overfit_reconstruction_examples.png`, `overfit_convergence.png`,
`latent_capacity_sweep.json` and compact `overfit_comparisons.npz`.
No beta/default/capacity setting is adopted. This is training-set fitting ability,
not validation generalization or a calibrated prior/covariance. Finite training
failure is not a proof of an architectural limit. Uncertainty remains deferred.

### E04 matched 500-epoch run — reviewed 2026-10-05

**Confirmed:** `figures/vae_runs/e04_matched_overfit_holes/` uses the exact
32 E04 training frames and paired oracle arrays. CUDA, seed 1, Adam LR 1e-4,
batch 8, 500 epochs, unchanged 36-latent/6x6 encoder/30x30 spline architecture;
deterministic from-scratch fitting, beta 0, no sampling/holes/scheduler.
Source artifact SHA256 remains `7dd1efaf...02b8d1`; recorded script SHA256
`3ea0d91a5aee3bcd5c1213d6ba01e4178073dbece6c57345f744176d767ec4bb`
matches live source. Selected targets/masks/weights and selection manifest
match E04 exactly. Recomputed array metrics match the JSON. Reloaded best
checkpoint is epoch 496; CPU quality RMSE agrees with CUDA within 8e-6
(maximum prediction difference 0.00129 standardized units).

| Method, same TRAINING frames | WIC / SI12 quality-log RMSE | Joint legacy reconstruction loss |
| --- | --- | --- |
| Source encoder mean | 0.3366 / 0.4115 | 1196.07 |
| Optimized latent oracle | 0.3050 / 0.3786 | 1000.45 |
| Deterministic overfit, latent 36 | 0.2763 / 0.3530 | 850.43 |
| Free-spline oracle | 0.1222 / 0.2016 | 235.15 |

**Convergence:** best loss at epochs 100/200/300/400/450/500 is
1562.80/1177.84/1015.79/914.30/879.27/850.43. Best retained loss falls
6.99% in the final 100 epochs and 3.28% in the final 50; minimum is epoch
496, not a demonstrated plateau. Last epoch loss is 874.29, so use the
restored best model, not the final state. Train/eval curves remain close.

**Interpretation (inferred):** unchanged architecture fits these training
frames better than both source encoder and frozen-decoder latent oracle:
quality RMSE improves 17.92%/14.21% relative to the source. However, 64.03%
of the original source-to-free-spline squared-loss gap remains, and the
overfit examples are still smoother than the observed/free-spline fields.
The finite, still-improving run cannot establish a capacity limit. Subset
training, removed KL, removed sampling and removed holes change together;
this does not isolate beta as the explanation. All-valid errors are mixed:
source 0.6302/0.5681 versus overfit 0.5918/0.5778. Better quality-weighted
fitting is not uniformly better pixel fidelity. Free splines may fit counting
noise and are not an ideal prior/covariance target.

**Disposition / next proposed test:** inconclusive on architectural capacity;
no production settings adopted. Before a latent-size/encoder change, run the
same diagnostic from scratch with 2,000 epochs in a separate output folder,
changing only `--epochs`, then inspect convergence and the spline gap. This
needs user approval; no run is launched by this result review. Source VAE,
figures, defaults and upstream repositories are unchanged. Uncertainty stays
deferred and E08 remains the scientific adoption gate. Test plan, Current State
and Handoff are checkpointed; no central portfolio fields changed.

### E04 matched 2,000-epoch run — reviewed 2026-10-05

**Confirmed:** the user completed the longer run in
`figures/vae_runs/e04_matched_overfit_holes_2000/`. It changes only epoch budget
relative to the 500-epoch protocol: same 32 training frames/target/mask/quality,
source artifact, script hash, saved transform/basis, latent 36/architecture,
CUDA/seed 1, Adam 1e-4/batch 8, no sampling/KL/holes/scheduler. Exact E04 arrays
and recomputed metrics match. Source SHA remains `7dd1efaf...02b8d1`, script
SHA remains `3ea0d91a...67ec4bb`. Diagnostic checkpoint restores epoch 1966;
CPU reload quality RMSE agrees within 1e-6 (maximum CPU/CUDA prediction
difference 0.00146 standardized units). Initial losses are identical; CUDA
training paths are not bitwise identical (best at 500 is 848.25 versus the
earlier run's 850.43), despite the same seed and script.

| Method, same TRAINING frames | WIC / SI12 quality-log RMSE | Joint legacy reconstruction loss |
| --- | --- | --- |
| Source encoder mean | 0.3366 / 0.4115 | 1196.07 |
| Deterministic overfit, 500 epochs | 0.2763 / 0.3530 | 850.43 |
| Deterministic overfit, 2,000 epochs | 0.2167 / 0.2958 | 569.19 |
| Free-spline diagnostic comparator | 0.1222 / 0.2016 | 235.15 |

**Convergence/result:** best loss at epochs 500/1000/1500/1800/1900/2000 is
848.25/690.56/613.96/586.07/574.56/569.19. Best epoch 1966; final-epoch
loss 581.58. Best loss declines 7.29% in the last 500 epochs, 2.88% in the
last 200 and 0.93% in the last 100: approaching a slow-improvement regime,
not proven convergence. Compared with the separate 500-epoch run, quality
RMSE improves 21.55%/16.20% and joint loss 33.07%; versus the source encoder,
quality RMSE improves 35.61%/28.10%. All-valid RMSE is 0.4582/0.4894, now
better in both channels than the source's 0.6302/0.5681. Inspected examples
show more spatial detail than the shorter overfit, but remain smooth.

**Interpretation/disposition:** the earlier finite-training result was
materially optimization-budget limited. Architecture can fit better than
the saved VAE; neither KL alone, latent dimension nor generalization is
isolated by this subset experiment. Do not prescribe 2,000 production epochs
or seek the noise-fitting free-spline floor. No production setting is adopted.

**Next proposed test, approval needed:** explicit latent-72 diagnostic with
the same 32 frames, seed, loss, 2,000 epochs and other controls. Existing
`--latent-dimensions 72` supports it; use a new output folder. This changes
latent dimension and associated fully connected parameter counts, so it tests
latent/mapping capacity jointly, not an intrinsic dimensionality estimate.
Inspect each candidate's convergence before interpreting error differences.
Only a meaningful diagnostic benefit motivates a later regularized full-corpus
VAE comparison, whose adoption gate remains held-out E08 covariance skill.
No new training, model/default changes or upstream edits were made during
review. Uncertainty stays deferred. Test plan, Current State and Handoff
are checkpointed; no central portfolio fields changed.

### E04 latent 72 versus 36 — reviewed 2026-10-05

**Confirmed:** the user's completed run is
`figures/vae_runs/e04_matched_overfit_holes_latent72_2000/`. Same exact selection,
targets, masks, weights and oracle arrays as the latent-36 2,000-epoch control;
same source/script SHA, Adam 1e-4/batch 8, CUDA/seed 1, deterministic beta 0,
no holes/sampling/scheduler. Only latent dimension and associated linear layers
change; encoder features, decoder spatial stages and 30x30 spline basis remain
fixed. Recomputed metrics agree; reloaded checkpoint epoch 2000 has CPU quality
RMSE within 2e-6 of CUDA (max prediction difference 0.00170 standardized units).
Source/artifacts/plots are preserved; no source or production default changes.

| Same 32 TRAINING frames, 2,000 epochs | Latent 36 | Latent 72 |
| --- | --- | --- |
| Quality-log RMSE, WIC / SI12 | 0.21674 / 0.29584 | 0.21427 / 0.29124 |
| All-valid RMSE, WIC / SI12 | 0.45824 / 0.48940 | 0.44668 / 0.47643 |
| Best joint legacy loss | 569.19 | 553.27 |
| Best epoch | 1966 | 2000 |
| Parameter count | 298,490 | 547,394 |
| Best-loss gain, final 100 / 500 epochs | 0.93% / 7.29% | 1.27% / 7.21% |

Quality RMSE improves 1.14%/1.55%, joint loss 2.80%, all-valid RMSE
2.52%/2.65%, with 83.39% more parameters. Lower per-frame quality errors
occur on 24/32 WIC and 30/32 SI12 frames; median gains 1.39%/2.14%.
Benefits are small rather than dominated by one frame. Inspected examples
and convergence resemble latent 36. Neither fit has proven converged; same
seed across differently shaped networks is not identical initialization.
No repeated seeds, held-out candidate evaluation or covariance test was run.

**Interpretation/recommendation:** larger latent/mapping capacity gives only a
small improvement in this screen, not compelling evidence that latent 36 is
the dominant bottleneck. Do not promote 72 or launch 144/a full VAE solely on
this result. Retain production latent 36 and other settings; not a universal
rejection of 72. Matching independently fitted splines remains NOT the goal.

**Next proposed E05 test, approval needed:** examine decoder spatial-stage
resolution instead of another latent sweep. Live code makes second-stage
features at 18x18 (`9x9 -> up2 x2 -> dec_conv2`), enlarges via adaptive average
pool to 30x30, then runs the coefficient convolution. Hypothesis: this coarse
feature stage limits learned spatial detail even though spline controls are
30x30. Compare an opt-in `up2` to 30x30 BEFORE `dec_conv2`; the subsequent
30x30 resize is identity. Keep latent 36, encoder 6x6, spline 30x30/degree 3,
parameter count, seed, inputs and matched 2,000-epoch protocol fixed. This is
an architectural experiment, not a confirmed bug or an instruction to fit
counting noise. Option/verification need implementation first. A later regularized
full-corpus/E08 comparison is required before any production adoption.
No new training or implementation is authorized by this result review.
Test plan, Current State and Handoff checkpointed; portfolio fields unchanged.

## E05 — Decoder spatial-stage resolution: implemented 2026-10-05

**Authorization:** after the latent-72 screen and request for a command, the
user explicitly approved implementing/verifying an opt-in decoder-resolution
test without changing production defaults. This is not adoption of 30x30
features or approval to fit measurement noise.

**Changed aspect:** `--decoder-feature-size 30` in `diagnose_latent_capacity.py`
sets the second nearest-neighbor upsample to 30x30 BEFORE `dec_conv2`, replacing
the 18x18 feature stage followed by enlargement to 30x30. The final adaptive
pool is identity at 30x30. Latent 36, encoder (64,6,6), spline basis/control
grid 30x30/degree 3, 298,490 parameters and all layer-weight shapes stay fixed.
Same-seed construction gives identical initial weights/buffers in both variants;
only spatial computation changes. Production runner exposes no new setting.
Absent the flag, the saved architecture (legacy 18x18) is preserved.

The shared model constructor/config loader records `decoder_feature_size` only
for nondefault models; old default configurations and their computation remain
unchanged. Candidate checkpoints save this setting despite identical weight
shapes, preventing silent reload as the old stage. Candidate filename is
`deterministic_overfit_latent36_decoder30.pt` and labels include decoder size.

**Matched evaluation:** `--overfit-control
figures/vae_runs/e04_matched_overfit_holes_2000` reads the completed control,
not retraining it. Exact source/frame/target/mask/quality identities, epoch
budget, seed, LR, batch size, deterministic recipe and control architecture
must match. Predictions' recomputed loss must agree with the control report.
Control appears in error/examples plots and as a horizontal convergence line;
JSON records its report hash/device/config/best epoch/metrics. Different latent
size, budget, data or decoder control is rejected before candidate training.
Original model/buffers/artifact and existing output folders are protected.

**Verification:** full suite 143 passed. Added coverage for unchanged parameter
count, identical seeded initial states, actual convolution-input resolution,
30x30 resize identity, finite gradients, default/opt-in configuration reload,
serialized prediction roundtrip, CLI validation and matched-control reuse/
mismatch guards. Real CPU smokes on all original 32 E04 training frames used
two epochs each: default control under
`figures/vae_runs/e05_decoder18_smoke_2026-10-05/`, candidate/control-comparison
under `figures/vae_runs/e05_decoder30_smoke_2026-10-05/`. Candidate selects
epoch 2, loss 19211.48, quality RMSE 1.48912/1.52384; control 1.48655/1.52365.
These two-epoch errors establish execution only, not improvement or a capacity
conclusion. Exact CPU checkpoint/prediction roundtrip passes, figures inspected,
source SHA remains `7dd1efaf...02b8d1`. Diagnostic script SHA is
`fc399f8faec5c8d1edbcf26bab3403095b411c296a626e5b8550056a9d113ad8`.

**Execution:** the user completed the full 2,000-epoch diagnostic; reviewed
closeout follows. No production setting, uncertainty mode or upstream change
is adopted by this diagnostic. Implementation checkpoint: README, test plan,
Current State, Decision Log and Handoff.

### E05 full decoder-stage comparison — reviewed 2026-10-05

**Confirmed:** `figures/vae_runs/e05_matched_overfit_decoder30_2000/` uses
CUDA/seed 1, Adam 1e-4/batch 8, 2,000 epochs, latent 36, deterministic
beta 0/no sampling/holes/scheduler. Only decoder spatial processing changes;
298,490 parameters, encoder/spline dimensions and exact matched 32 frames
stay fixed. Source SHA `7dd1efaf...02b8d1` and diagnostic script SHA
`fc399f8f...d113ad8` match live files. Completed 18x18 control report SHA
`3dddf065c40cc219351c400052e9eb1abc4a0c82a011e2bf19e0f6a1caba0edb`
matches; exact data, oracle and reused-control prediction arrays are verified.
Recomputed losses/RMSE match all reports. Reloaded candidate checkpoint epoch
1962 reproduces CUDA quality RMSE within 4e-6 (maximum prediction difference
0.00165 standardized units). Generated figures inspected; artifacts unchanged.

| Same TRAINING frames, 2,000 epochs | Decoder 18x18 | Decoder 30x30 |
| --- | --- | --- |
| Quality-log RMSE, WIC / SI12 | 0.21674 / 0.29584 | 0.20483 / 0.27931 |
| All-valid RMSE, WIC / SI12 | 0.45824 / 0.48940 | 0.45330 / 0.47670 |
| Joint legacy reconstruction loss | 569.19 | 507.72 |
| Best epoch | 1966 | 1962 |
| Parameter count | 298,490 | 298,490 |

Quality RMSE improves 5.49%/5.59%, all-valid RMSE 1.08%/2.59%, joint loss
10.80%. Per-frame quality errors improve on 29/32 WIC and 31/32 SI12 frames;
median gains 3.75%/4.53%. Examples show modest extra detail, not a major
reconstruction change. Candidate best loss at 500/1000/1500/1800/1900/2000
epochs is 803.89/642.52/557.53/528.43/515.43/507.72. Final loss is 512.47;
best loss still improves 8.93% in the last 500 and 1.50% in the last 100.
Neither fit has demonstrated convergence; same initial weights do not make
the distinct spatial computations/optimization trajectories identical.

**Interpretation/disposition:** modest, widespread fitting benefit without
more parameters supports screening this stage in the actual VAE. It does not
establish the main cause of poor source reconstructions, a confirmed legacy
bug, optimal resolution, or improved geophysical covariance. No stage/default
is promoted. This single-seed training-subset result is not held-out evidence;
approaching the direct-spline noise-fitting comparator is NOT the objective.

**Next proposed test, approval needed:** regularized full-corpus 30x30-stage
candidate matched to the completed 18x18 `wic_si12_smoothed_b16_holes` control:
beta 1.6, seed 1, same orbit split/transform/data, legacy loss, holes 0.5/12,
500 epochs and original scheduler/checkpoint policy, latent 36 and 50,000
samples. Compare held-out fidelity, prior statistics/spatial covariance and
frozen E08 validation fixture before adoption. The runner still lacks the
opt-in flag; implementation must carry the setting through construction,
provenance and artifact-reuse validation, with unchanged default 18.
No implementation/full run is authorized or launched by this result review.
Uncertainty stays deferred. Test plan, Current State and Handoff updated;
no central portfolio fields changed.

## E06 — Count-space uncertainty: implemented and initially screened

**Authorization:** the user requested implementation and testing of the proposed
count-space likelihood. This is an opt-in experiment, not a promoted default.

**Changed aspect:** only the observation loss. Use original count uncertainties,
quality-weighted training-only RMS channel scales, fixed model discrepancy of
0.5 scale units per channel, Gaussian density for positive counts and a stable
left-censored Gaussian probability for zeros. Keep log10 encoder inputs, the
quality channel, latent 36, 6x6 encoder features, 30x30 splines, scheduler,
quality/missing-data policy and physical-output zero clipping unchanged.
Implementation/units/assumptions are documented in the pipeline
[README](../../scripts/IMAGE/vae_pipeline/README.md#opt-in-count-space-uncertainty-e06).

**Important limitations:** WIC was clipped on native pixels before averaging;
this CS-level Gaussian/censoring model is approximate, including for positive
averages. It is not a raw Poisson model and ignores correlated measurement
errors. The likelihood uses a signed pre-clipping inverse mean down to -1
count; exported fields remain nonnegative. Counts normalized by training RMS
are not the legacy standardized-log residuals. Fixed beta therefore does not
imply matched reconstruction/KL pressure across these loss modes.

**Verified implementation tests (2026-10-04):** 108 maintained tests pass on
CPU. New checks cover analytic Gaussian and log-CDF references, original count
errors, unit-rescaling invariance, finite gradients at/below zero and at a
100-sigma tail (float32/float64), finite-difference gradients, missing/zero-quality
support, retained-training-only scales, unchanged summed KL, opt-in CLI guards,
holes, artifact reload and rejection of changed reuse settings/scales. A
Gaussian clipping simulation recovers its generating mean while naive averaging
of clipped samples is biased; it does NOT validate the native-WIC forward model.

Real smoothed orbits 85/89/212 retain 20/6/32 train/validation/test frames under
the existing manual selection and 33% cutoff. Two CPU epochs, beta 1.6, seed 1,
holes 0.5/12 and 16 prior draws complete training, held-out evaluation, exact
artifact-reuse metrics, propagation, covariance and sparse reconstruction.
Training count scales are 250.1403 WIC / 2.8727 SI12, with physical discrepancy
125.0701 / 1.4364 counts. Twenty verified PNGs include held-out inpainting.
The reusable eigenmode sparse benchmark also completes on two validation frames,
four modes and 2/10 sites. Outputs: `figures/vae_runs/e06_count_smoke_2026-10-04/`
and `figures/vae_runs/e06_count_sparse_smoke_2026-10-04/`.
These tiny runs check mechanics, NOT scientific skill. CUDA was unavailable.

**Original controlled-run proposal:** smoothed corpus, CSV selection, no SZA filter, seed 1,
beta 1.6, batch 8, learning rate 1e-4, 500 epochs, holes 0.5/12, latent 36,
30x30 cubic controls, 50,000 prior draws. Use a fresh output such as
`wic_si12_smoothed_b16_holes_count`, adding `--count-uncertainty
--count-model-error-fraction 0.5 0.5` and omitting both prior uncertainty flags.
This matches the 500-epoch hole controls, not the 250-epoch no-hole `_mes` run.
Do not overwrite or retrain controls. The user subsequently completed the
250-epoch run evaluated below; the original 500-epoch proposal was not followed.

Score the same held-out neutral/quality reconstruction and inpainting metrics,
zero/intensity strata and latent activity. Reuse E08's frozen 487-frame fixture
for unobserved-cell conductance RMSE, bias and 95% coverage. The visualizer's
count-attenuated log-RMSE is only a comparison diagnostic, not the censored NLL.
Do not select a winner from absolute loss values. Adopt/reject/inconclusive and
settings carried forward require these results and the user's decision.

### E06 full run — evaluated 2026-10-04

**Verified configuration:** `wic_si12_smoothed_b16_holes_cmef` genuinely uses
`quality_weighted_censored_counts_v1`, fractions 0.5/0.5, beta 1.6, seed 1,
holes 0.5/12, 50,000 prior draws, and the unchanged architecture, transform,
complete data provenance, orbit split and 4,033/487/1,101 usable frames.
It trained 250 epochs and selects 131, versus the hole controls' 500 epochs
and selected 165 legacy / 128 log-Gaussian. Both controls' first-250 minima
are their final selected epochs, bounding the saved-checkpoint budget issue;
improvement from a longer count run is not ruled out. One training seed only.
Artifact SHA256: `1da0921acd29af158337c2f37a29498e77b1f550b9ebea16f25bbf4fec29be50`.

Fresh CPU validation reproduces the user's CUDA reconstruction RMSE within
8e-6 absolute. New/log-Gaussian bright-patch identities/bounds/mask match
SHA256 `9d9ca5f2833c092836cc65d1f9449906ebebcdc42670e5c34bf32a2087a53498`.
The same E08 fixture (SHA256 recorded below), all 487 validation frames, joint
10/30/100 sites, 0.1 dex noise and 100 modes are reused. The new modes retain
99.198% of log-conductance variance. Both old artifact hashes remain unchanged.

| Hole-trained loss | WIC / SI12 auroral normalized-log RMSE | Reconstructed WIC / SI12 mean [counts] | Hall / Pedersen sparse RMSE [S], 100 sites | Hall / Pedersen 95% coverage |
| --- | --- | --- | --- | --- |
| Legacy | 0.449 / 0.542 | 140.0 / 2.93 | 0.711 / 0.882 | 58.9% / 58.1% |
| Log Gaussian 0.5/0.5 | 0.680 / 0.536 | 212.8 / 3.62 | 0.794 / 0.980 | 38.5% / 36.4% |
| Count censored 0.5/0.5 | 0.595 / 0.703 | 203.3 / 2.63 | 0.774 / 0.958 | 50.1% / 48.8% |

Observed all-valid means remain 130.6 / 3.38 counts. Relative to the legacy
hole control, count auroral RMSE worsens 32.6% WIC / 29.7% SI12; all-valid
RMSE is 1.084/0.914 instead of 0.672/0.602. Sparse errors worsen 9.2–9.5%
at 10 sites, 7.2–7.6% at 30 and 8.6–8.9% at 100. Against log-Gaussian,
count sparse errors improve 2.3–4.5%, with better coverage, but do not recover
the legacy control. Prior-only coverage is 70.7% Hall / 69.4% Pedersen,
versus about 86% for legacy. These are noisy fixed-energy proxy references,
not independently measured physical truth.

**Mixed secondary results:** bright-patch WIC physical RMSE improves 3.7%
(714.85 -> 688.20 counts), while SI12 worsens 1.0% (9.826 -> 9.928).
Bright-patch log RMSE worsens 49.5% WIC / 31.6% SI12. Prior std increases
from 243.9/5.04 to 303.7/6.04 counts, closer to observed 418.5/6.87, but
prior WIC mean shifts from 126.6 to 194.8. More physical dispersion does not
ensure better covariance or interval calibration. Mean KL is 16.57 versus
25.86 nats/frame; 36 versus 35 dimensions exceed 0.1 nat. No complete collapse.

**Physical allowance audit:** fitted training scales are 1,253.099 WIC /
21.6899 SI12 counts, so f=0.5 gives discrepancy std 626.549 / 10.845 counts.
Quality-weighted validation measurement-sigma medians are only 20.126 / 0.830
counts (90th percentiles 35.818 / 1.362). Count-density attenuation retains
99.8155% / 99.1056% of common quality mass. Measurement variance therefore
has little influence at these settings. This diagnostic is NOT the gradient
weight for censored zeros, whose loss is a CDF. Do not attribute the model's
regression to measurement errors on this evidence.

**Intensity audit:** on positive-quality zero-WIC targets, the deterministic
posterior's quality-weighted predicted mean rises from 7.83 to 49.09 counts.
For positive WIC <=50, observed mean is 14.86 and predicted means are
57.85 legacy / 133.27 count. Conversely, WIC >50 physical RMSE improves
1,035.79 -> 944.30 counts. This is a trade from faint/background fidelity
toward brighter physical-count fitting, not an across-the-board improvement.
It is consistent with a changed count/censoring objective, not proof of a
specific causal mechanism or an architectural bug.

**Disposition:** do not promote this count setting; preserve the legacy hole
run as the experimental control and retain defaults. Measurement information
itself is not rejected. The subsequently authorized gradient audit below
examines channel/intensity balance and discrepancy scaling without retraining.
No new setting or architecture change is adopted. Independent calibration/seed
checks remain open.

**Outputs:** `figures/vae_runs/e06_count_validation_2026-10-04/` contains fresh
eight-figure posterior validation, the new sparse benchmark report/examples,
`sparse_skill.png` comparing the three hole-trained losses, and
`reconstruction_comparison.png`. Existing user figures/models/data were not
overwritten. Benchmark/artifact hashes were verified before comparison.

### E06 saved-model loss-gradient audit — completed 2026-10-04

**Authorization:** the user said "TLDR. Just do it" after the proposed audit.
Scope is diagnostics only; no retraining, architecture/default or upstream changes.
Reusable script: `scripts/IMAGE/vae_pipeline/diagnose_loss_balance.py`.
All 487 validation frames, matching saved splits/transforms/architectures/data,
CPU, batch 8, noise seed 1001, beta 1.6. Each model uses the same seeded
posterior-noise sequence. Each batch's counterfactual losses share its frozen
prediction graph. Model weights/buffers remain unchanged and artifact SHA256s
match before/after. No 50,000-sample ensemble is loaded.

Observation strata: zero; faint positive <=50 WIC / <=1 SI12 counts; middle;
bright >= quality-weighted positive 90th percentile (2179.064 / 33.508 counts).
Gradients are computed with respect to standardized-log decoder outputs (L1
budget), and the same shared encoder parameters for reconstruction and beta-KL
(L2 norms). Values below are native losses; shares use BOTH fields as denominator.

| Hole-trained model | WIC / SI12 output-gradient share | Zero+faint WIC share | Bright WIC share | Encoder-gradient change removing measured variance | Recon / beta-KL mean encoder norm |
| --- | --- | --- | --- | --- | --- |
| Legacy | 44.53% / 55.47% | 15.29% | 2.27% | 0% | 5.18 |
| Log-Gaussian | 40.08% / 59.92% | 4.79% | 3.58% | 176.81% | 5.33 |
| Count f=0.5 | 77.69% / 22.31% | 1.60% | 37.86% | 0.90% | 6.01 |

**Within-model check:** on the frozen LEGACY predictions, changing only the
objective to count f=0.5 shifts WIC share from 44.53% to 79.23%, zero+faint
WIC from 15.29% to 0.45%, and bright WIC from 2.27% to 38.32%. Thus the
loss formulation itself strongly changes the learning emphasis; this is not
merely a comparison of different fitted images. RMS normalization balances
units, not gradients: inverse-log differentiation, residual magnitudes and
censoring still matter. This supports an objective-driven tradeoff, not proof
of the entire training-time cause or a validated replacement objective.

**Smaller fractions, frozen count model:** at f=0.5 / 0.1 / 0.03, removing
measured variance changes the encoder gradient by 0.90% / 15.57% / 88.63%.
Measurement information becomes influential, but SI12's output-gradient share
falls 22.31% -> 18.09% -> 9.82%, and recon/beta-KL norm ratio falls
6.01 -> 5.07 -> 2.95. Smaller fractions alone do NOT isolate uncertainty
information or repair faint-WIC emphasis. Holding the count loss reference
at 0.5 squared instead yields ratios 6.01 / 126.64 / 819.16; this demonstrates
the coupled temperature effect, not a recommended training setting.

**Limitations:** eval-mode, full-input stochastic posterior audit, not exact
train-mode/holes optimization. Norms are means of 61 per-batch norms, not the
norm of a mean gradient. Encoder Jacobians differ across trained models;
counterfactual comparisons within one model are strongest. Counterfactual
fixed references are diagnostic only. No sparse-skill improvement is claimed.

**Disposition / future option (deferred by user 2026-10-05):** retain the legacy hole run as
control. Before another uncertainty run, define an objective retaining its
faint-signal and SI12 fidelity, with discrepancy variance separate from loss
temperature. One possible controlled test is a bounded uncertainty addition
to legacy log-MSE rather than a full count-loss replacement; mixing strength,
channel safeguards and physical discrepancy scales need agreement first.
Evaluate any candidate on the existing E08 fixture; no candidate is implemented
or promoted by this audit. Do not implement this hybrid-loss proposal while
uncertainty integration is tabled; E04 is the current next diagnostic.

**Verification / outputs:** 119 tests pass, including production-loss and
gradient equivalence for all three objectives, finite masked gradients,
direction-sensitive measurement removal, deterministic non-mutation, and
consistent plot-stratum colors. Report/script/model/frame hashes, per-batch
statistics and three figures are under
`figures/vae_runs/e06_loss_audit_2026-10-04/`. The README documents the command.
Earlier smoke/intermediate audit folders are not the authoritative checkpoint.

## E05/E07 — Conditional next steps; do not run everything

- **E05:** the authorized decoder spatial-stage diagnostic is complete. A
  regularized full-VAE comparison is proposed next; its runner opt-in and run
  need approval. Do not also change latent, encoder or spline dimensions.
- **E07:** add uncertainty as encoder context only after earlier results. Keep
  output targets unchanged and isolate input conditioning from loss changes.

## E08 — Conductance covariance and sparse assimilation

**Question:** which existing prior, `wic_si12_smoothed_b16` or
`wic_si12_smoothed_b16_holes`, better supports sparse conductance assimilation?
The user authorized implementation. The initial comparison is complete;
broader geometry, noise, rank and sample-convergence sweeps remain proposed.

Use the saved generated/propagated ensembles to obtain mean and covariance.
Do not encode held-out images or optimize latent vectors during assimilation.
The existing `sparse_ensemble_reconstruction` conditions using the anomaly
factor A, with C = A A.T, and does not call the decoder. An eigenfactor
U sqrt(Lambda) represents the same covariance when all nonzero modes are
retained; truncation changes the prior. Matching the legacy solver also requires
matching rank, covariance and regularization, not just calling both eigenmode fits.

Compare identical sparse observations, noise, support and geometries across
many held-out frames; include random and track-like layouts and Hall-only,
Pedersen-only and joint observations. Score originally observed cells excluded
from assimilation: posterior RMSE/bias and uncertainty coverage, against prior-
mean and diagonal-covariance controls. Check sample-count and retained-mode
sensitivity, unsupported regions and negative posterior conductance explicitly.
The current pipeline example is one frame with exact subsampled values and
assigned observation variance, not a many-frame noisy benchmark; its aggregate
RMSE also includes assimilated locations. Do not substitute that for this test.

Held-out FUV-derived conductance is a fixed-2-keV proxy, not independent ground
truth. Fixed energy constrains Hall/Pedersen dependence. Keep geophysical prior
covariance distinct from measurement/observation covariance R. Improved
inpainting, by itself, neither validates nor invalidates the sampled covariance.

### E08 implementation and first validation screen — 2026-10-03

Entry point: `scripts/IMAGE/vae_pipeline/test_sparse_conductance.py`; commands
and the future-model reuse contract are in the pipeline README. Freeze one
`benchmark.npz` and reuse it with `--benchmark`: reference fields, supported
mask, frame identities, sparse locations and injected noise do not change.
Reject differing saved splits, scientific selection, grids or propagation.
Read model metadata only; no decoder is instantiated or called. Next-model
architectures may differ while retaining the ensemble/provenance interface.

This screen uses all 487 validation frames, seed 0, joint Hall/Pedersen
observations at 10/30/100 nested random spatial sites, Gaussian noise 0.1 dex,
and `log10(conductance + 1 S)` covariance with 100 leading grid-space modes.
The modes retain 98.65%/98.73% of total variance for no-hole/hole priors.
The mathematics follows legacy eigenmode OI, but grid-space rank is not the
legacy coefficient-space rank. No claim of exact legacy numerical equivalence.
Scores exclude observed spatial sites from both fields and exclude natural
gaps and permanently unsupported training locations. Prior medians provide
the no-assimilation control; diagonal covariance cannot improve these unseen
locations. Detector errors are not mixed into prior covariance or synthetic R.

| Spatial sites | Hall RMSE, no holes / holes [S] | Pedersen RMSE, no holes / holes [S] |
| --- | --- | --- |
| 10 | 1.048 / 1.051 | 1.296 / 1.299 |
| 30 | 0.891 / 0.891 | 1.105 / 1.104 |
| 100 | 0.713 / 0.711 | 0.884 / 0.882 |

At 100 sites the prior-only RMSE is about 1.23/1.51 S: conditioning improves
point reconstruction about 42%. Hole/no-hole differences stay below 0.3%.
**Interpretation:** both priors contain useful spatial covariance; this screen
does not establish a hole-training covariance advantage, despite E02 gains.
It does not show that holes are harmful or should be removed.

**Calibration risk:** nominal 95% intervals cover only 58–59% at 100 sites,
versus 75–76% at 10 sites and about 86% without conditioning. Negative
back-transformed predictions at 100 sites affect roughly 9.5–9.7% of Hall
and 8.6–8.7% of Pedersen scored cells; they are reported, not clipped.
This is marginal coverage against noisy FUV-derived proxy fields, not
independent physical truth. Cause is unresolved: truncation, prior mismatch,
likelihood/model discrepancy and proxy measurement error require separation.
Do not claim calibrated assimilation uncertainties from this screen.

**Verification:** 86 maintained tests pass, including dense legacy-equation
reference agreement, all observation modes, fixed sites/noise, original-support
scoring, log-sample covariance, reporting negatives, unknown-architecture
metadata-only reuse and preserving existing results. The real 487-frame
`--benchmark` reuse reproduces the hole model's metrics exactly.
Large factors are handled sequentially, with batched log transformation into
a temporary disk-backed factor and randomized SVD; no sample-square Gram matrix.
**Outputs:** `figures/vae_runs/e08_sparse_validation_2026-10-03/`: frozen fixture,
`metrics.json`, `sparse_skill.png` and three 100-site examples per model.
Fixture SHA256: `955294b8124706f9625c64ab1aee3b13caa7d733d52d95a4011ec7e831a48d39`.
**Disposition/settings:** initial point-skill screen useful; covariance
calibration remains inadequate. No retraining, upstream changes or default
promotion. Proposed next check: retained-rank sensitivity and a self-consistency
test with truth drawn from the specified prior, before diagnosing a model bug
or selecting an uncertainty adjustment. Independent orbits/training seeds and
track/single-field tests remain open; no broad sweep has been launched.

## Reusable experiment closeout

- ID/date/status and run/control paths:
- Question; one changed aspect; verified fixed settings/code/splits:
- Target metric and numerical adoption/safeguard tolerances agreed before run:
- Commands; implementation/evaluation readiness; seeds; selected epochs:
- Results and figure paths (WIC/SI12 separately; validation vs exploratory test):
- Interpretation and limitations:
- Disposition: adopt / reject / inconclusive; user confirmation:
- Exact settings carried forward and next test:

Method background lives in [VAE uncertainty - next experiments](<VAE uncertainty - next experiments.md>).
Historical implementation scope lives in [WIC-SI12 VAE Pipeline Plan](<WIC-SI12 VAE Pipeline Plan.md>).
