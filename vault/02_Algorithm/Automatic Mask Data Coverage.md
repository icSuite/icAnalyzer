# Automatic mask data coverage

Last reviewed: 2026-10-14  
Repository: `vae/wic-si12` at `be31406`, with existing uncommitted work preserved.

## Question and evidence

User reports that visual comparisons of the completed automatic-mask CVAE with
Hardy show clear Kp-dependent auroral structure, alongside noisy dayside output.
This is user-reported visual evidence, not a quantitative model-skill or
calibration result. The requested first diagnostic tests whether the U-Net
excluded all dayside pixels.

`scripts/automatic_masking/plot_data_coverage.py` counts each saved Boolean
keep decision across all 1,378 orbit files / 358,439 frames. It separately
counts the frozen cache's common-valid H/P mask over the 123,874 eligible
training frames, and checks which of those pixels have positive loss weights.
Each pixel's count is the number of frames retaining that location. These are
coverage counts, not conductance values or counts of independent observations.

Inputs are the `automatic_masks_e5c009e6713b` corpus and the cache recorded by
the user-trained `conductance_automatic_cvae_b16_holes` run. Cache and mask
manifest hashes, individual mask hashes recorded by the cache, frame counts,
mask-subset ordering and exact agreement with `training_support.npy` are checked.
Processing uses one orbit at a time and bounded cached batches; source models,
masks, caches and training outputs are untouched.

## Result

Dayside here means **06 <= MLT < 18**. This spatial definition is not an SZA
or actual illumination selection. Across the full grid, raw predictions retain
all 2,116 cells at least once; the training cache retains 2,078 cells, with 38
unsupported cells. Of the 1,058 dayside cells, 1,041 have training coverage.

For the **488 dayside cells at magnetic latitude >= 60 degrees**:

| Count per pixel | Raw U-Net / all source frames | Eligible training frames |
| --- | ---: | ---: |
| Cells with coverage | 488 / 488 | 488 / 488 |
| Minimum | 50,833 | 6,925 |
| Median | 177,247 | 62,406 |
| Maximum | 246,657 | 105,882 |

All 128,759,898 retained training pixel-frame pairs also have positive cached
loss weight, including the 27,653,218 pairs in the high-latitude dayside region.
This checks positivity, not the magnitude of their relative loss influence.

**Conclusion:** blanket dayside exclusion by the U-Net is ruled out. Coverage
is lower on the dayside, especially toward lower latitudes near noon, but it
is substantial throughout the high-latitude dayside. This does not establish
that noisy measurements passed through or explain the generated noise.
Proposed next diagnostic: inspect retained dayside source measurements and
their quality/weight distributions, with Kp and coverage held in view.

## Artifacts and reproduction

Run from the repository root:

```bash
python scripts/automatic_masking/plot_data_coverage.py
```

Use `--output` to choose a fresh directory when repeating. Final reviewed
artifacts are in `figures/validation/wic_coverage_20261014_final/`:
`data_coverage.png`, `data_coverage.pdf`, `coverage_counts.npz`, and `summary.json`.
Noon is above, midnight below, dawn left and dusk right. Each panel has its own
linear count scale; light grey denotes zero coverage. PNG visually inspected.
Earlier `_20261014` / `_v2` directories are superseded plotting drafts.

## Frame selection and matched manual comparison (2026-10-14)

Follow-up addresses the ambiguity introduced by comparing all raw predictions
with a filtered training subset. The orbit split is70/15/15 percent for
training/validation/test, not one third for training. Training orbits contain
253714 frames;186461 have at least one common-valid masked pixel with positive
quality. The intrinsic-missing-fraction<=0.33 cut leaves123874 training frames.
The other retained splits contain24485 validation and25554 test frames.
Unmatched timestamps contribute zero exclusions. The frame missing fraction is
computed **before** the U-Net mask, from upstream nominal/uncertainty validity.

`icanalyzer/conductance.py` requires both nominal and uncertainty validity,
finite H/P and finite nonnegative dH/dP, finite quality, and positive finite
nominal/uncertainty coverage. Uncertainty validity remains required even when
uncertainty weighting is disabled. No explicit SZA cut applies to this cache.
These checks establish usable retrievals; they do not establish noise-free data.

On the **identical123874 retained training frames**, the fraction of all
high-latitude pixel-frame locations retained is:

| MLAT>=60 region | U-Net mask alone | U-Net AND upstream validity |
| --- | ---: | ---: |
| Dayside06–18MLT | 46.81% | 45.75% |
| Nightside | 87.98% | 85.77% |

Thus the dayside deficit within the retained training population is already
present in U-Net predictions. The additional pixel validity intersection makes
a much smaller change. This isolates the pixel intersection on retained frames;
it does not attribute the spatial effects of whole-frame selection separately.

The current manual dataset exactly matches the annotation JSON/checkpoint
snapshotc272e0b57c36...:17612 reviewed nonempty frames, including2109 full,
6671 bad,8832 partial (partial masks include held-forward keyframes). Comparing
manual targets and U-Net predictions on **the same frames and finite-WIC
domains**, the retained fractions of available high-latitude observations are:

| MLAT>=60 region | Manual masks | U-Net masks |
| --- | ---: | ---: |
| Dayside | 34.93% | 34.90% |
| Nightside | 58.97% | 65.53% |

Manual annotations also favor the nightside. Equal aggregate dayside acceptance
does not mean identical accepted pixels or absence of noise leakage. This is
the manual supervision corpus, including U-Net training/validation/test rows;
it is not an independent accuracy benchmark or an exact reconstruction of the
historical hybrid CVAE corpus, which also uses CSV fallback.

`scripts/automatic_masking/compare_manual_coverage.py` creates
`figures/validation/wic_manual_coverage_20261014/`: `manual_vs_unet.png/pdf`
(manual left/automatic right, counts above and fractions of available WIC
below), `training_validity_loss.png` (mask-only left/mask AND validity right on
identical training frames/shared count scale), `counts.npz`, and `summary.json`.
Time alignment and frame counts pass; both PNGs visually inspected and opened
in the user's viewer. No source/mask/cache/model changes.

## Full-corpus retention using the same available-WIC denominator (2026-10-14)

A direct one-orbit/batched read of all1378 original precipitation files and
their saved automatic masks verifies native timestamp alignment, unchanged
source sizes/mtimes, and358439 total frames. For06<=MLT<18 andMLAT>=60,
the U-Net retains61081883 of128230427 finite corrected-WIC pixel-frame pairs:
**47.6345%**. This is comparable in denominator/region definition to the34.9002%
computed on the17612 manually reviewed frames, but the frame populations differ.
It does not establish a greater false-acceptance rate on unreviewed frames.
The earlier training-mask percentages46.81/45.75% divide by *all* pixel-frame
locations in eligible training frames, including missing WIC, and therefore
must not be equated with these percentages of available observations.
Machine-readable evidence:
`figures/validation/wic_coverage_20261014_final/full_corpus_dayside_retention.json`.

## Nightside calculation with identical definitions (2026-10-14)

Requested before drawing conclusions about increased dayside acceptance.
Nightside isMLT<06 orMLT>=18, stillMLAT>=60. Denominator is finite corrected-WIC
pixel-frame pairs, with the same per-orbit source/time/frame verification as above.

| Mask and population | Retained / available nightside pairs | Percent |
| --- | ---: | ---: |
| Manual,17612 reviewed frames | 3962550 / 6719401 | 58.9718% |
| U-Net, same reviewed frames | 4403344 / 6719401 | 65.5318% |
| U-Net, all358439 frames | 92752788 / 120755261 | 76.8106% |

Results saved in
`figures/validation/wic_coverage_20261014_final/nightside_retention.json`.
No noise diagnosis or model/annotation decision follows from these counts.

## Generalization diagnostic from saved masks (2026-10-14)

User proposes population non-representativeness, U-Net overfitting, excessive
compression/framework problems, and task unsuitability as competing hypotheses.
These remain hypotheses, not adopted diagnoses or architecture decisions.

Split the existing matched manual/automatic masks by the exact9333/3438/4841
training/validation/test IDs. Independently reconstructed the standalone training
script's orbit/day boundaries and verified exact ID agreement and disjoint test
orbits. No inference or training rerun. Region remainsMLAT>=60, dayside06–18MLT;
nonfinite WIC excluded. Pixel-weighted fraction of manually rejected observations
that the U-Net keeps:

| Region | Training | Validation | Held-out test |
| --- | ---: | ---: | ---: |
| Dayside | 9.33% | 11.70% | 26.98% |
| Nightside | 20.09% | 21.96% | 53.80% |

These aggregate rates are not the previous orbit-averaged class metrics; class
and image-condition mixtures differ across splits. They show poor agreement
with manual labels on unseen later orbits, but do not distinguish ordinary
overfitting from distribution shift, label drift or class-composition effects.
The full-corpus acceptance rate alone cannot measure false acceptance without
labels. Manual rejection is a label proxy, not proof of physically noisy pixels.

The currentSmallUNet reduces46→23→11 spatially and restores full resolution with
skip connections at both upper levels, with30441 learned parameters/eight input
channels. This provides no direct evidence that bottleneck compression is the
cause. The earlier topology trial's rejected-frame tradeoff remains relevant;
architecture/input/loss/threshold causes are unseparated. The independent input
channel study is underway and production masks remain protected.

Evidence: `figures/validation/wic_manual_coverage_20261014/by_split_errors.json`.
Proposed next discrimination: a small blind manual audit of unreviewed orbits
and comparison of false-keep errors under matched image conditions, before
adopting another architecture or interpreting generated dayside noise.

## Selected blind audit orbits (2026-10-14)

User asks which and how many orbits to review. Proposed pilot is eight whole
orbits:247,749,838,1016,1144,1489,1679,1809, spanning2000-08-19 to2003-03-04.
Choose one seeded random orbit from each of eight equal-count strata of sorted
eligible IDs(seed20261014). Pool899: exclude every orbit with a current or audit
JSON record and any historical CSV annotation; require a present conductance
file, nonempty WIC and nonempty nominal conductance. Selection never uses
U-Net predictions. Source times/Kp/frame counts are inspected only after random
selection; no outcome-driven sample substitution. Reviews are now complete;
[[Manual Annotation Policy and Blind Audit]] owns evaluation and interpretation.

The user's `data/unet_audit_20261014.json` already exists as a separate copy,
with no additional orbit records at selection time. Continue annotations there,
then compare only newly reviewed audit orbits with frozen production masks.
`data/unet_audit_orbits_20261014.json` records selected dates/counts/Kp, exclusions,
seed, and existing annotation/model fingerprints. Use the existing annotator's
`--annotations data/unet_audit_20261014.json --start-orbit NUMBER`. Mark complete
or no-usable per selected orbit; completion auto-advances sequentially, so
restart at the next selected ID rather than following that next orbit. Audit
labels remain outside training. The initial single-frame line-model experiment
has subsequently completed; [[WIC Dividing Line Model Benchmark]] owns its
negative result. No GUI or production model/mask changes.
