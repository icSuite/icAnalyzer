# WIC class and temporal mask benchmark

Last reviewed: 2026-10-13
Status: Class-head, temporal and mask-only ablation evaluated; no model adoption
Repository: vae/wic-si12 at be31406 with existing worktree changes preserved

## Scope and frozen comparison

User returned to the queued improvements after the full-grid correction:
a three-class frame output alongside the mask, then five-frame image context.
[[Full Grid Annotation Transfer]] owns the corrected labels and
[[WIC Mask Propagation Benchmark]] owns the headless baseline.

Dataset: figures/validation/wic_mask_unet_fullgrid_20261012/.
Annotation SHA256:
c272e0b57c36e020ebeede12aa0cfbfb28fd06eaaf6adda10b9cf70db5331895.
All 17612 nonempty frames and the existing 9333/3438/4841 split retained.
Training/validation/test contain 35/13/25 disjoint orbits and orbit-start days.
The held-out set contains 846 full, 2412 rejected and 1583 partial frames.
All finite WIC pixels supervise; partial keyframes hold forward within their
partial interval. No original-grid border restriction.

Four maps: corrected WIC, method_quality_weight, SZA, DZA, with their finite
indicators. No dwic_corrected, redundant coverage, Hall/Pedersen, ratio,
energy/flux, manual mask or absolute date/orbit predictors. Quality is input
only; supervised pixels score equally. Current CS support still embeds
upstream retrieval validity, so this is not intrinsic detector-WIC validation.

## Models and context

Same small U-Net encoder/decoder as baseline. A global-average-pooled bottleneck
feeds a three-class linear head. Per-image cross entropy, weight 1, is added
to the existing equally scored BCE and Dice mask loss. Class/orbit/segment
sample balancing, AdamW, batch64, seed12026 and train-center normalization
are unchanged. Best epoch and pixel threshold use validation only.

The temporal model stacks native offsets −2, −1, 0, +1, +2 within an orbit.
Native unlabelled/empty exposures are observations, never target features.
Every intervening adjacent time gap must be ≤300 s; missing neighbors have
zero normalized maps and finite flags. Relative time offsets/300 are constant
input maps. Context is offline and includes future frames. There are 19287
native frames, 17234 complete windows, and 1209 additional unlabelled/empty
exposures. Typical cadence122 s; the largest admitted two-step offset491 s.

Eight single-frame channels versus45 temporal channels. New first-layer
weights outside the current eight channels start at zero; all other weights
and the shuffled training order match the single-frame experiment initially.
Parameters: baseline30441, class head30540, five frames33204.
Class head selects epoch2/10, threshold0.65; temporal selects12/20, threshold0.45.
Early stopping uses joint validation loss; the two tasks may compete.

## Held-out results

Mask errors below are mean frame disagreement within each orbit, then equal
orbit means within class. They are not percent of annotation labor.

| Model | Class balanced accuracy | Full mask error | Rejected mask error | Partial mask error |
|---|---:|---:|---:|---:|
| Baseline; mask-fraction class proxy |55.6%|29.2%|27.3%|23.5%|
| Single-frame joint class head |53.0%|24.8%|39.0%|28.1%|
| Five-frame joint class head |59.6%|27.5%|15.2%|23.8%|
| Single-frame class-gated mask |same head|24.7%|35.3%|28.2%|
| Five-frame class-gated mask |same head|28.7%|11.2%|26.2%|

Five-frame accuracy69.3%, macro-F1 60.2%; accepted/rejected/partial recall
23.2%/83.7%/72.0%, precision90.3%/80.6%/53.8%. Its gain mainly identifies
rejected images; accepted recall remains low. Baseline accuracy59.0%;
single-frame head55.4%. Frame metrics give frames equal weight and are
distinct from orbit-balanced mask metrics.

Temporal raw rejected-mask improvement12.11 percentage points, paired
orbit-start-day bootstrap95% interval6.20–19.27. Partial change−0.34 pp,
interval−4.14–4.01: no improvement established. Single-frame head worsens
rejected error11.70 pp and partial error4.61 pp in this experiment.
Temporal equal-class raw disagreement22.2% versus baseline26.7%.

Partial useful pixels discarded rise17.5%→31.8%; rejected pixels retained
fall31.1%→16.0%. Thus similar partial correction area conceals a substantial
tradeoff. Forcing the mask to follow the predicted class raises useful loss
to36.8%. Class-gating rule is fixed: predicted full keeps all finite WIC,
bad rejects all, partial retains the raw predicted mask; no test tuning.
Raw predictions are retained separately.

## Five-frame mask-only ablation (2026-10-13)

User authorized the proposed ablation. Remove the class output and cross
entropy, retaining all three classes' constant/propagated mask targets. Same
frozen dataset, context, split, normalization, sample weights and optimizer.
Initialize then discard the class layer before optimizer construction: all
initial mask tensors, first mask outputs and 9333 shuffled training indices
match the joint temporal run. The baseline's older shuffle stream differs,
so only the direct joint comparison isolates class-task removal this tightly.

33105 parameters; no classifier state, optimizer parameters or class outputs.
Best16/24 epochs selected by mask-only validation loss, threshold0.425 selected
by the unchanged validation mask criterion. No quality loss weighting.

| Raw mask model | Accepted error | Rejected error | Partial error |
|---|---:|---:|---:|
| Single-frame baseline |29.2%|27.3%|23.5%|
| Joint five-frame |27.5%|15.2%|23.8%|
| Mask-only five-frame |22.6%|27.7%|25.9%|

Equal-class orbit-average disagreement25.4% versus22.2% joint and26.7%
baseline. The accepted point estimate improves, with day-bootstrap intervals
including no gain. Rejected and partial errors worsen versus joint:12.47 pp
(95%6.11–19.83) and2.12 pp (0.22–3.71). Partial worsening versus baseline2.46 pp
has interval−1.76–6.10, so no partial benefit is established.

Partial useful pixels discarded27.8% versus31.8% joint and17.5% baseline;
rejected pixels retained19.2% versus16.0% joint and31.1% baseline. Removing
the class task changes this balance but does not solve useful-pixel loss.
Mask-fraction class proxy has66.3% accuracy/60.1% balanced accuracy; this is
not a trained class output and no class-gating is applied.

Outputs:figures/validation/wic_temporal_mask_only_20261013/ with model/,
experiment_manifest.json, comparison.json, paired_frame_scores.csv,
mask_only_comparison.png, mask_only_same_30.png and
mask_only_partial_same_30.png. Both 30-frame lists are exactly the baseline
choices, frozen by SHA256. README owns reproduction commands.

11 focused model/ablation tests pass; changed scripts parse. Initialization,
context/splits/normalization/selection and paired score checks pass; comparison
and both example figures inspected. GUI, live annotation JSON, all three
reference checkpoints and the previous comparison report hashes unchanged.
This is one seed on a reused holdout; early selection uses mask-only loss here,
while the joint model used joint loss. No adoption or measured time saving.

## Interpretation and next action

Confirmed: the single-frame joint head did not improve this baseline; short
context improves rejected recognition but does not improve partial masks.
Neither automatic review nor measured editing-time savings is established.
This is one seed on an already inspected holdout, so follow-up is exploratory.

The mask-only follow-up above did not improve partial masks. Keep the existing
baseline for comparison and retain all experimental checkpoints. Proposed
next step is review of the fixed example sheets before choosing another
model or target representation; no additional training is queued. Predictions
remain drafts, with no changes to live annotations or GUI.

## Artifacts and verification

figures/validation/wic_class_temporal_20261012/ contains context/, class_head/,
class_head_temporal/, comparison.json, paired_frame_scores.csv,
improvement_comparison.png, mask_improvements_same_30.png and verification.json.
The 30 examples are exactly the baseline's choices, not selected by new scores.
Per-model outputs include raw probabilities/masks, frame class outputs,
separate gated masks, summaries, splits, histories and example sheets.
scripts/IMAGE/vae_pipeline/README.md owns reproduction commands.

27 focused tests pass; four experiment scripts parse. Independent checks
reproduce native neighbor/time/gap alignment, allowed inputs, disjoint splits,
training-only normalization, selected epochs/thresholds, raw/gated metrics and
all paired bootstrap intervals. Both comparison figures visually inspected.
Live annotation JSON, annotate_partial_images.py and corrected baseline
checkpoint hashes match pre-experiment guards. Source/scientific weights
untouched. Current State, Handoff and classifier plan checkpoint this result.

