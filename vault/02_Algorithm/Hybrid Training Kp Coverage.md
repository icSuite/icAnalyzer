# Hybrid Training Kp Coverage

Last reviewed: 2026-10-12
Status: Exact retained-frame Kp recovered from original source products.

## Question and provenance

User reports plotting mean and pointwise0.1/0.5/0.9 quantile maps of50000
hybrid members against Lompe Hardy conductance for Kp0–6. Their reported peak
ranges are Hall4–24/Pedersen3–13 S for Hardy and Hall0.5–21/Pedersen1–6 S across
VAE quantile maps. These are user-reported exploratory comparisons, not an
independently reproduced matched-activity/model-physics test.

The native VAE checkpoint, summary, ensemble and saved diagnostic JSONs do
not contain per-frame Kp labels or a Kp-range summary. Source NetCDFs do contain
`Kp` and `Kp_interval_start`, with GFZ definitive-index provenance
(DOI10.5880/Kp.0001, original acquisition2026-08-05). The checkpoint retains
source paths, annotation fingerprints and frame-selection settings, so the
exact retained populations can be reloaded and joined to source Kp by exact
UTC image time.

Entry point `scripts/IMAGE/vae_pipeline/audit_run_kp.py`; README owns command.
No interpolation/nearest-time matching is introduced. Model/source evidence
is unchanged before/after. All9201 retained train/validation/test frames match
source rows; no missing Kp. Original run artifacts are not modified.

## Results

| Split | Frames | Contributing orbits | Kp min–max | p10 | Median | p90 | Frames Kp≥5 |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| Train | 6881 | 100 | 0.667–6.667 | 1.667 | 3.333 | 5.667 | 19.78% |
| Validation | 1203 | 21 | 0.667–6.667 | 0.667 | 2.667 | 3.667 | 4.66% |
| Test | 1117 | 19 | 0.667–6.667 | 1.333 | 3.000 | 5.667 | 20.23% |

Training integer-bin counts: Kp<1:39;1–<2:704;2–<3:1707;3–<4:1885;
4–<5:1185;5–<6:914;6–<7:447. No retained frames with Kp≥7. Thus the
training selection includes substantial active conditions and is not confined
to quiet Kp. It also does not cover Kp0 or the strongest storm indices.
Validation is less active than training/test by these descriptive summaries;
this alone does not establish a checkpoint-selection bias.

Frame counts are temporally correlated. Distinct three-hour intervals:
179 train,33 validation,32 test; interval-weighted summaries are also saved.
Intervals can span orbit boundaries/splits, so counts are not independent
draws or guaranteed disjoint activity contexts. Primary numbers above weight
retained frames equally, not training loss quality/pixel weights.

## Interpretation and next comparison

The VAE encoder uses H/P and quality, not Kp. Unconditional generated members
have no assigned Kp. Conductance quantiles therefore describe a model-implied
mixed-activity distribution; a0.9 conductance quantile does not correspond to
Kp at its90th percentile. The maximum of a pointwise quantile map also differs
from a quantile of member-wise field maxima.

Training coverage alone cannot explain or validate the user-reported Pedersen
contrast with Hardy. A useful next comparison matches the source retrieval,
VAE and Hardy spatial support/coordinates, auroral versus solar components and
summary definition, then accounts for the actual training Kp distribution.
Activity-stratified retrieval/reconstruction checks could distinguish a
training-population effect from model compression or conversion differences.
No new Kp-conditioned generator, retraining or scientific acceptance decision.

Artifacts:
`figures/validation/hybrid_conductance_validation_20261012/kp/kp_coverage.json`
and `retained_frame_kp.csv`. CSV records split-local frame index, orbit, UTC,
Kp interval, original source row and file for every retained image. One focused
test verifies exact matching and rejects nearby/duplicate timestamps.
