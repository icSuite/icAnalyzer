# WIC masks and CVAE conductance browsing diagnostic

Last reviewed: 2026-10-14  
Status: Same90 examples verified with Hall/Pedersen; near-noon coverage
compared with full training; noise cause unresolved. Production unchanged.

## User observations and question

User reviews the90-frame browsing set and reports good apparent performance
on accepted/partial images, with rejected images the main mismatch. Their
visual estimate is10/30 rejected, with other frames fully/partially accepted.
Some manual-rejected frames look usable and are recovered by the U-net.
User reports high acceptance scores outside the masked area, and finds no
clear explanation of noisy CVAE dayside fields from these WIC examples.
These are user observations from the sample, not a full-corpus validation or
confirmed physical-noise diagnosis. Conservative manual selection is still
not equivalent to physical pixel-quality ground truth.

Requested next check: add Hall/Pedersen columns and verify what the current
automatic-data CVAE actually ingests. No classifier change/training requested.

## Saved CVAE input verified

Run is `conductance_automatic_cvae_b16_holes` under IMAGE_FUV/cvae_runs;
cache is `automatic_cvae_e5c009e6713` under IMAGE_FUV/vae_cache. Saved checkpoint
training configuration/data provenance and identical live/run cache manifests
confirm native upstream `conductance_cs`, fields Hall/Pedersen in S. Upstream
model is Robinson; H/P are overlap means of detector-space conductance, not
recalculated here from gridded WIC using a fixed electron energy. Propagation
is not reapplied by the CVAE workflow.

Training receives channel-standardized log10(H+1), log10(P+1), a shared
quality/coverage-weight map, and per-frame Kp conditioning. Invalid targets
are zeroed. Automatic common-valid mask is p≥0.475 AND intrinsic common H/P
validity (valid retrieval and uncertainties, finite nonnegative uncertainties,
finite quality, positive conductance/uncertainty coverage). Frame inclusion
also requires some accepted positive-weight pixels and intrinsic missing
fraction≤0.33. Pixel-mask acceptance alone does not establish frame inclusion.
Artificial holes are training augmentation, not a replacement validity policy.

## Revised figures and exact alignment

`figures/validation/wic_probability_conductance_20261014/` contains30 PNGs,
three10-page class PDFs and a combined30-page PDF, retaining exactly the
prior90 identities and ordering (30 per class). Columns: WIC, manual mask,
current U-net mask, raw acceptance score, Hall, Pedersen. Prior four columns
retain the same scales/coordinates and production0.475 threshold.

H/P show the full intrinsically valid upstream fields BEFORE automatic masking,
in physical S with common asinh(S/1) display and pooled99th-percentile maximum
36.117598S. White outlines mark the automatic conductance pixel mask, including
on frames later excluded. Row labels explicitly show CVAE train/validation/test
or exclusion. Gray H/P cells are intrinsically invalid. These are observed
conductance products, not CVAE reconstructions or generated samples.

Native conductance timestamps matched exactly to WIC examples, with grid and
source-stat checks. Reused actual `masked_orbit` preparation logic, independently
checked source values/validity/quality against every corresponding cache shard,
and verified frame eligibility. Actual checkpoint fields/transform/cache identity
rechecked. Sample pages from all classes inspected; class PDFs opened.
`conductance_provenance.json` records individual inclusion, mask support and
source/model/cache fingerprints; `conductance_display.json` records plotting.

## Quantitative facts for this browsing set

- 50/90 frames included:45 train,4 validation,1 test. These are U-net held-out
  frames with different CVAE split assignments, not90 CVAE held-out examples.
- Partial class:25/30 included (21 train,4 validation).
- Manual rejected class:10/30 included, all10 train.
- Manual full class:15/30 included (14 train,1 test).
- 32 exclusions due intrinsic coverage;8 due no accepted positive-weight pixels.
- Zero automatically retained conductance pixel-frame pairs outside finite-WIC
  support across all90, also zero in the included subset. High raw acceptance
  scores in gray exterior WIC regions therefore do not leak into this input set.
- Strict classification on finite WIC among30 manual-rejected examples is
  8 all-reject,20 partial,2 all-keep. This uses exact zero/all retained pixels
  and is distinct from the user's visual10-rejected estimate; neither implies
  the remaining observations are physically bad.

No conclusion established about the physical source of CVAE dayside noise.
Next: inspect retained H/P structure against masks and observed/generated
conductance; no threshold adoption, model retraining or cache rebuild queued.

## Follow-up: dayside appearance, coverage and exclusion wording

User reports barely any dayside coverage/signal in all three figure classes.
The90 examples span2000-07-09 to2000-07-29; actual training spans2000-05-16 to
2003-05-10. However, a direct comparison of the45 included training examples
against all123874 cached training frames does NOT show much lower broad-day
coverage:06≤MLT<18, MLAT≥60 fixed-grid pixel-frame fractions45.478%/45.745%.
Near noon (10≤MLT<14) coverage is22.827%/26.059%; restricting latitude60–75
reduces it to10.159%/15.021%. At latitude≥75 near-noon coverage61.481%/59.740%.
Thus the visual impression of sparse coverage near noon at lower latitudes
has support; do not equate it to blanket dayside absence or infer the browsing
set is much less covered across the entire broad MLT-day region. Denominators
are fixed region cells×frame count, not conditional on available finite WIC.
Saved exact sample/full maps and region calculations under
figures/validation/wic_browsing_coverage_check_20261014/; full counts are the
previous verified cache-linked coverage artifact, sample masks reread by exact
native timestamp from eligible training shards. No new source products read.

Code confirms decoder `decode(z,Kp)` predicts the whole grid, without the
individual input mask or SZA as decoder conditions. Reconstruction weight is
zero at masked target pixels: missing observations impose no zero-target
constraint. Sampling applies aggregate training support, not a per-image U-net
mask; every high-latitude dayside cell has some aggregate training support.
Generated dayside signal can therefore be a modeling issue even when a given
input has no retained noon data; this mechanism does not establish the cause
of the user-reported noise. No scientific model decision or retraining follows.

User asks meaning of “CVAE excluded: intrinsic coverage”: more than33% of the
entire46×46 grid fails original common H/P retrieval+uncertainty/coverage
validity, BEFORE applying the automatic mask. The entire frame is omitted;
it is not a verdict that the U-net rejected its pixels or that WIC looks bad.

## Confirmed policy change: at least5% intrinsic valid coverage

On2026-10-14 user explicitly clarifies their desired5% means MINIMUM valid
coverage, to admit more good data. New automatic-cache preparation default is
`--minimum-valid-fraction 0.05`, equivalent to maximum missing0.95. Legacy
`--maximum-missing-fraction 0.33` explicitly reproduces the original67%-valid
rule; the two flags are mutually exclusive and finite/range validated. New
manifest records both maximum missing and minimum valid fractions. Positive
accepted-quality requirement, pixel validity, U-net masks and source retrieval
are unchanged. Threshold still operates BEFORE automatic masking, not on the
post-mask retained area. Older direct WIC/SI12 workflows retain their own
compatibility defaults; change is scoped to automatic conductance cache prep.

Code and README updated, with fresh `_valid5` output example paths. Existing
full cache and trained checkpoint remain frozen at0.33 maximum missing; old
figure labels and coverage numbers describe that run. Applying policy to the
full corpus requires preparing a new cache and training a new run. No full
cache rebuild/retraining was launched by the agent.

On the same90 browsing examples,5%-valid admits82 versus50 old-rule frames;
remaining8 have no accepted positive-weight pixels. This is a sample result,
not the new full-corpus frame count. Real preparation comparison on orbits
85/86/89 yields366 new-rule versus262 old-rule eligible frames. Physical H/P,
uncertainty/quality arrays, automatic-valid masks, normalization and orbit splits
match exactly; every old-eligible frame remains eligible. Both caches load all
three splits correctly. Temporary caches are
`/tmp/icanalyzer_cache_valid5_20261014` and
`/tmp/icanalyzer_cache_missing33_20261014`.
Four existing loading tests pass including spawned workers outside sandbox;
initial sandbox worker failure was socket permission, not data/code behavior.
CLI default/legacy/equivalent flags, invalid/conflicting flags and syntax checks
pass. User was explicitly told dayside observations exist: full old-run broad
MLT-day coverage45.75%;15.02% referred only to noon10–14MLT, latitude60–75.
Noise source remains unresolved; coverage-rule relaxation is user-directed.

## Reproduction

`plot_current_probabilities.py` now accepts optional `--conductance-run` and
uses `example_conductance.py` for the distinct source/cache alignment calculation.
Use `--per-class 30`, `--include-selection` pointing to the expanded90-frame
selection, original `--wic-vmax 5086.3900585937545`, and a fresh output path.
Existing four-column mode and old figure folders preserved. Syntax checks,
exact90-frame selection parity, all saved-mask checks, native/cache comparisons
and10/10/10/30 PDF page counts pass. CVAE/U-net checkpoints, source products,
automatic masks, cache and main/audit annotations remain unchanged.
