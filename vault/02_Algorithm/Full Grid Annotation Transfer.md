# Full-grid annotation transfer

Last reviewed: 2026-10-12
Status: Conversion/live annotations corrected; mask and quality analyses rebuilt
Repository: vae/wic-si12 at be31406, existing worktree edits preserved

## Confirmed user contract

Transfer frame classes and drawn lines from36x36 to46x46. The user rejects the
added outside-old-grid unreviewed area. Fully accepted retains every finite
native pixel; rejected retains none. Partial masks apply transferred line sides
across the whole grid. Each keyframe holds forward within its partial interval.
Lassos remain bounded polygons; nonfinite observations are not data targets.

## Correction and preservation

The converter had clipped labels at the old grid edge, an unintended semantic
change affecting every class. That restriction is removed. The shared geometry
helper evaluates original line sides on inverse-mapped target centres without
clipping at source pixel0/35; new native lines use native coordinates.

The safe live repair expands53 inherited domains and four nested archives.
All91 domains now cover2116 cells.464 live masks expand, adding96321 kept
keyframe cells. Previously labelled pixels,classes,boundaries,time axes,keyframe
IDs,geometry,human timestamps and38 already-full native records are unchanged.
Synthetic166 boundaries survive;169/170 remain no_usable with no live keyframes.
Archival drafts are not restored. Original36x36 JSON remains unchanged.

Exact backup/corrected JSON/report:
figures/validation/annotation_conversion_fullgrid_20261012/.
Before SHA:ad947fc3c474c98528b55b2ca92c73683816c3a446a00c89128d32827ba58a68.
After SHA:c272e0b57c36e020ebeede12aa0cfbfb28fd06eaaf6adda10b9cf70db5331895.
Atomic compare-before-write applied the correction. README owns commands.
Restart an open GUI to reload corrected masks. Its loader also repairs old
clipped converted records in memory, so reopening an earlier saved file does
not reintroduce the old boundary.

22 tests and independent live/archival/geometry/propagation audits pass.
Real orbit85 editor rendering/reopening/double flip agrees exactly with masks.
Fresh conversion of53 original orbits/469 keyframes succeeds.

## Affected analysis

Rebuilt WIC dataset:17612 nonempty frames (2109 full,6671 bad,8832 partial),
plus3263 empty frames. Every domain is finite WIC across the whole46x46 grid.
Adds2527409 supervised cells, including715656 kept partial cells; earlier
labels retained. All4841 test targets/domains, including1583 partial frames,
are unchanged because those reviews were already native.

Same inputs,normalization,weights and9333/3438/4841 split; corrected U-Net at
figures/validation/wic_mask_unet_fullgrid_20261012/. Best16/24 epochs,threshold0.65.
Disagreement29.2% full,27.3% rejected,23.5% partial versus19.3%/36.5%/23.9%
before. This fixes supervision without establishing automatic handling or time
savings. [[WIC Mask Propagation Benchmark]] owns the model comparison.

Rebuilt quality histogram/scatter:figures/annotation_quality_fullgrid_20261012/.
Same populations; mean-quality medians0.0874/0.0752/0.1122/0.0822 for full,
rejected,partial masked/unmasked. Prior plots/models are historical snapshots
using the superseded restriction.

Future annotation-selected conductance/FUV inputs use corrected masks.
Scientific weights and saved selections remain unchanged; the earlier
current-selection Kp comparison is a frozen pre-correction snapshot.
Class-head/temporal ideas remain queued. No automatic masks/new GUI workflow.
