# Manual annotation policy and blind audit

Last reviewed: 2026-10-14  
Repository: `vae/wic-si12` at `be31406`, existing dirty worktree preserved.

## Confirmed user policy

The user usually selects a conservative accepted region with one dividing line,
occasionally two for simultaneous day/night problems. Geometric/workload limits
mean visibly reasonable pixels can be rejected because they lie near bad pixels
or cannot be conveniently isolated. Labels therefore represent the user's
conservative selection policy, not independently verified pixel contamination.

Lines are chosen for reuse on subsequent frames. On otherwise well-behaved,
changing sequences the user may draw a new keyframe only every18th frame.
Rejected/partial/full sequences can switch abruptly. A mostly stable partial
boundary may briefly be replaced by a heavily masked frame and then return.
These abrupt changes must remain possible in any proposed temporal model.

The current tool copies a mask unchanged until the next keyframe or class
boundary, without interpolation. It shows36 frames per overview page(6×6);
the new native images and stored audit masks are46×46 pixels. The earlier legacy
images were36×36. One/few line masks are intersections of accepted half-planes:
convex regions clipped by the grid, without interior holes/disconnected islands.

## New independent review evidence

All eight selected audit orbits are reviewed: seven complete,247 no-usable,
2053 frames. Existing main annotations/production U-Net/automatic masks/cache
remain unchanged. Evaluated only selected audit records against the frozen
automatic masks, with exact source/annotation/mask timestamp and grid checks,
source-size/mtime verification and production model/mask hashes.

Thirty new keyframes:24 line sets(18 one-line,5 two-line,1 three-line),4 all-reject
and2 all-keep. Independent read-only review reproduces all24 drawn-line raster
masks exactly. Stored semantic frame classes are736 bad/1178 partial/139 full;
actual full-grid mask content is917 rejected/823 partial/313 full because some
keyframes within partial intervals specify all-zero/all-one masks. Use effective
mask content when measuring full/partial/rejected handling. No lasso in this audit.

| Available finite-WIC pixels retained, MLAT>=60 | Manual audit | Frozen U-Net |
| --- | ---: | ---: |
| Dayside06<=MLT<18 | 39.58% | 37.33% |
| Nightside | 54.41% | 58.24% |

The new audit does not reproduce greater aggregate U-Net dayside acceptance.
Nevertheless, on dayside the U-Net keeps19.16% of manually rejected pixels and
discards34.92% of manually accepted pixels. Corresponding nightside rates are
31.77% and19.57%. These are pixel-weighted disagreements with the stated policy,
not measured physical-noise leakage or independent-frame confidence intervals.
Eight stratified random orbits are a pilot, not proof of corpus-wide calibration.

## Direct versus propagated supervision

Of1178 semantic-partial frames,30 are explicit keyframes and1148 propagated.
Median age since keyframe is24 frames,max236. For geometrically nonconstant
partial masks, the24 drawn keyframes show43.87% useful dayside pixels discarded;
propagated nonconstant partial masks show43.17%. Nightside rates25.90/25.48%.
Failure therefore exists on directly drawn frames; propagation alone cannot
explain it. These groups have different frame populations and small direct sample
counts; this is not a causal test or a precision comparison.

Current training equalizes class/orbit/mask-segment influence, but propagated
masks remain per-frame targets. Existing class-head/five-frame experiments and
input-channel study have not established improved partial-mask prediction;
do not describe temporal context or a structured decoder as a demonstrated fix.

## Proposed direction, not adopted

First distinguish imitation of conservative manual selection from physical
pixel-quality detection. The current labels directly supervise the former;
they cannot reliably identify every physically useful pixel or contaminated
pixel. Compare newly drawn keyframes separately from held-forward labels.

A proposed structured model predicts effective reject/partial/full state and,
for partial frames, one/few accepted line sides. Temporal consistency would
apply within stable regimes, with change-aware reset/override for bad bursts
and state switches. Blanket smoothing is inconsistent with user policy.
The user subsequently authorized the initial single-frame state/count/three-line
trial; [[WIC Dividing Line Model Benchmark]] owns its completed three-seed result,
which is worse and not adopted. Interpolation, label reweighting and temporal
modeling remain untested; no production model/mask/VAE input was replaced.

## Reproduction and artifacts

`scripts/automatic_masking/evaluate_unet_audit.py` consumes the saved selection,
separate audit JSON and production masks; no inference/training. Choose a fresh
output directory with`--output` when rerunning. Reviewed final outputs:
`figures/validation/wic_blind_audit_20261014_final/annotations_snapshot.json`,
`summary.json`, `frame_scores.csv`, `audit_examples.png`. Examples use the middle
stored keyframe per orbit, or middle rejected frame for247; not cherry-picked.
PNG inspected. Independent CSV sums, keep/reject identities and effective-mask
partition totals agree for all-grid/day/night regions. Earlier unsuffixed output
is a draft preceding effective-mask and drawn-keyframe stratification.

[[Automatic Mask Data Coverage]] owns the prior full-corpus and reviewed-corpus
coverage comparison and original selection protocol.
