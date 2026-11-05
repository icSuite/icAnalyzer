# WIC topology loss benchmark

Last reviewed: 2026-10-13
Status: Both trials evaluated; shape improves but rejected-frame masks regress
Repository: vae/wic-si12 at be31406 with existing worktree changes preserved

## User requirements and scope

User requires no enclosed rejected islands and at most one accepted connected
region. An accepted island surrounded by rejection is allowed; an empty mask
must remain possible. They ask to test an explicit training penalty, rather
than silently applying cleanup. No VAE run or live annotation change authorized.

Use original corrected single-frame U-Net as baseline:
figures/validation/wic_mask_unet_fullgrid_20261012/. Same 17612 frames,
9333/3438/4841 split, labels, initialization, shuffled training order,
normalization, optimizer and four WIC/quality/SZA/DZA inputs. Quality remains
input only; dwic_corrected, coverage and forbidden retrieval quantities absent.
[[WIC Class and Temporal Mask Benchmark]] owns prior class/time experiments.

## Verified shape and missing-data contract

All 17612 raw geometric targets satisfy the prior: at most one8-connected
accepted region and zero enclosed4-connected rejected holes. Clipping to finite
WIC fabricates multiple components in41 targets and holes in1545 targets.
There are149 raw all-reject and158 raw all-keep partial targets; do not force
exactly one accepted component.

Loss applies to the full predicted probability mask, never probability times
finite-WIC domain. Pixel BCE/Dice and accuracy remain on finite WIC only.
Unobserved cells get a geometric prior but are never turned into observations.
Observable shape diagnostics permit unknown cells to bridge foreground, count
only regions containing observed accepted cells, and treat rejected components
touching unknown cells or the outer image border as open boundaries.

## Penalty and implementation

topology.py uses GUDHI CubicalComplex with filtration1-p. Detached
probabilities determine critical birth/death cells; PyTorch differentiates
the corresponding persistence values. Coface indices are Fortran ordered,
so the Torch image is transposed before flattening. Sum squared persistence
of ALL finite H0 bars (extra accepted components) and H1 bars (holes).
Essential H0 is exempt: one region allowed, never forced to exist.
This is a piecewise differentiable persistence prior, not a threshold guarantee.

Per-image loss is BCE + Dice + lambda * topology. Trainer default lambda0
preserves prior behaviour and does not import GUDHI. GUDHI3.13.0 installed
only in /tmp/icAnalyzer_topology_20261013, leaving the project environment
unchanged. README owns reproduction commands and result artifacts.
For the tConductance Python 3.10 environment, Mamba cannot resolve GUDHI
3.13.0 with the existing PyQt/Qt and Boost/ICU dependencies. The PyPI
Python 3.10 wheel imported successfully in this environment from /tmp;
the standalone trainer documents a binary-only, no-dependencies pip install.

20 focused tests pass: empty/full/single/diagonal shapes, nonsquare critical
indices, region/hole gradient descent, missing-data separation, random binary
PH versus discrete topology, and existing mask-model tests. Independent
finite differences match all40 coordinates of a5x8 field (error7.6e-11).
64-frame CPU timing: original forward/loss/backward0.094s, topology0.206s;
persistence-only0.163s. Dependency and extra CPU cost are feasible.

## Frozen experiment protocol

figures/validation/wic_mask_topology_20261013/ contains experiment_manifest.json
and pretraining_audit.json. Candidate weights0.1 and1.0 are fixed in advance;
seed12026, epochs limit40, patience8, no class head or temporal context.
Best epoch uses candidate validation loss; binary mask threshold uses common
class-wise orbit-average validation pixel disagreement.

Select positive lambda by the lowest common validation pixel disagreement,
saving selection.json before test inspection. Baseline is a reference, not
a positive penalty candidate; selection does not imply adoption. Training
may generate candidate test arrays, but test scores/examples remain unread
until selection. Candidate stdout is redirected and only epoch lines read.

Evaluation compares raw masks only: per-class orbit-average correction,
partial useful/rejected pixel errors, full-grid and observable shape violations,
paired orbit-start-day bootstrap changes and baseline-fixed30-frame sheets.
No hole filling or component filtering applied; prior checkpoints retained.
One seed on a previously inspected holdout is exploratory.

## Results and interpretation

Both trials completed with frozen inputs/splits. Validation selected lambda0.1
before inspecting candidate test predictions: common class-wise orbit-average
pixel disagreement was11.79% for baseline,10.34% for lambda0.1 and12.58% for
lambda1.0. Lambda0.1 used epoch5/13 and threshold0.50; baseline threshold0.65.
Stronger lambda1.0 produced fewer full-grid validation violations but worse
pixel agreement, so the predeclared rule did not select it.

The selected raw model was evaluated on4841 frames in25 unseen orbits:

| Orbit-average disagreement | Baseline | Topology0.1 |
| --- | ---: | ---: |
| Fully accepted, useful pixels discarded |29.2%|21.1%|
| Fully rejected, rejected pixels retained |27.3%|38.0%|
| Partial, all supervised pixels |23.5%|22.1%|
| Partial, useful pixels discarded |17.5%|17.1%|
| Partial, rejected pixels retained |31.1%|25.2%|

Paired improvement in pixel disagreement (baseline minus topology), with
orbit-start-day bootstrap95% intervals: fully accepted+8.16pp [-1.46,+16.67],
fully rejected-10.71pp [-14.89,-7.11], partial+1.37pp [-0.80,+3.44].
The rejected-frame regression is clear; accepted/partial improvements are
uncertain across days.

Full-grid raw predictions with either violation fell1750→779 frames; the
unknown-aware observable count fell598→143 (12.4%→3.0%). Observable frames
with multiple accepted regions fell391→79 and enclosed rejected holes245→65.
The target count is zero, so the loss improves topology but does not enforce it.
Same30-frame and partial30-frame sheets were visually inspected: some masks
gain smooth, connected regions and substantial pixel agreement, while others
retain excessive accepted area, especially rejected examples. These sheets
are baseline-selected examples, not a random performance estimate.

topology_comparison.json/PNG, topology_same_30.png,
topology_partial_same_30.png and paired_frame_scores.csv record the comparison.
Independent NumPy/SciPy calculations reproduce per-class pixel disagreement
and full-grid component/hole counts. verification.json confirms validation
prediction hashes and unchanged GUI, live annotation and prior model hashes.
Focused tests passed20/20; no postprocessing or live prediction application.

Conclusion for this experiment: the explicit shape loss is feasible and
reduces violations, but lambda0.1 is not suitable as a direct replacement for
the baseline because fully rejected frames degrade materially and violations
remain. No VAE integration or editing-time benefit is established. A proposed
next bounded comparison is explicit geometric projection of baseline and
topology outputs, measured for pixel cost and residual violations before
choosing any mask source. This proposal is not an adopted processing rule.

## Next actionable state

If an automatic mask is needed for the VAE, first decide how fully rejected
frames will be handled and test a geometry-enforcing projection on held-out
frames. The topology model remains an experimental candidate. GUI, live
annotations and saved scientific models retain their original hashes.

## Primary references

- [Explicit topological loss via persistent homology](https://arxiv.org/abs/1910.01877)
- [GUDHI cubical-complex/coface reference](https://gudhi.inria.fr/python/latest/cubical_complex_ref.html)

