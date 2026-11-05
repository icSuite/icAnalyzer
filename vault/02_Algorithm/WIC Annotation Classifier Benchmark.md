# WIC annotation classifier benchmark

Last reviewed: 2026-10-12
Status: First benchmark complete; automatic annotation not adopted
Repository: vae/wic-si12 at be31406, pre-existing worktree changes preserved

## Result and next decision

Spatial Random Forests learn useful signal, but this first benchmark does not
support reliable unattended acceptance or a demonstrated reduction of weeks
of review. Temporal feature context adds little. The native displayed-product
forest frequently rejects partial frames; total accuracy hides this weakness.
Next proposed step: inspect a small diverse set of held-out errors, reconcile
review-footprint/product-availability targets, then compare an image model or
pixel-mask proposals. No annotations, source products or conductance models
were modified during that benchmark. The subsequent first U-Net experiment
and manual workflow fallback are owned by [[WIC Mask Propagation Benchmark]].

## Frozen inputs and exclusions

The exact current annotation snapshot is archived in both output directories:

- figures/validation/wic_annotation_rf_20261012/
- figures/validation/wic_annotation_cs_rf_20261012/

First experiment: bin-first schema2 precipitation_IR_hardy_weighted products.
Allowed raw fields: wic,dwic,wic_corrected,dwic_corrected,wic_weight. Feature
quality is1-wic_weight; source audit verifies background-fit orientation.
Corrected WIC is computed before electron ratio inversion and not masked by
its success. Raw support is WIC-specific; exposure availability still requires
multi-instrument time matching. Legacy dwic is within-cell spread and corrected
std includes Poisson/proton correction; no SZA/DZA in this source.
20404 unique labelled source frames on83 orbits,58 orbit-start-day groups;
2222 full,8268 rejected,9914 partial.1760 review timestamps unmatched within2s;
none of the83 source files missing. Matching is unique and index-independent.

Second experiment: the current smoothed native precipitation_cs companion to
the GUI conductance products. Only wic_corrected,dwic_corrected,stored high-is-
auroral method_quality_weight,sza,dza enter features. Corrected WIC,std and
geometry inherit upstream method_valid spatial support: this predicts displayed
products, not intrinsic WIC-only quality.20875 labelled frames on79 orbits and
55 orbit-start-day groups:2142 full,9554 rejected,9179 partial. Four source
files unavailable(116/155/157/160);229 original review timestamps unmatched.
3263 frames have no finite displayed WIC. Populations differ between sources;
the two headline accuracies are not a matched-pair comparison.

Neither experiment uses H/P,image ratio,energy,flux,combined sensor weight,
manual masks,reviewed footprint,orbit IDs or availability flags as predictors.
Those last metadata fields support splits/audits only. Labels are current
frame_class targets, without relabelling or masking the source predictors.

## Model and evaluation

Each frame has192 spatial features: finite fraction,mean,std,p10/p50/p90 for
each allowed field, mean/fraction on4x4 spatial tiles, and WIC gradients.
The temporal design has964 features: centre plus changes to±1/±2 actual source
exposures and signed time gaps. Neighbours remain within the source orbit and
no window crosses a gap above300s. Built before target matching.

Compare majority baseline,seven quality/coverage features,spatial features and
temporal features. RF:200 trees,min_samples_leaf5,sqrt feature selection,
balanced classes,groups equally weighted. Five grouped development folds;
each outer training set reserves20% of its day groups for sigmoid probability
calibration. Imputation,missing indicators and forests fit only remaining
training groups. Calibration/test groups are disjoint. Development out-of-fold
macro-F1 selects spatial/temporal design; later-orbit test is not used for that
selection or confidence cutoffs. Whole orbits always stay together.

Chronological later checks: legacy2787 frames/17 orbits from2000-07-17;
native3865 frames/15 orbits from2000-07-18. Groups are orbit-start days, not
strict calendar-day segregation; the native development orbit spans part of
July18 but ends18.6h before the first held-out exposure. No exposure/window
overlap.500 bootstrap resamples use whole start-day groups. Reviewed periods
are early2000; transfer across later mission periods remains untested.

## Results

Selected using development predictions: temporal RF for legacy, spatial RF for
displayed CS. Probability calibration can change argmax decisions, so a
controlled raw-RF sensitivity was run on identical fit/calibration/test splits.
Reconstructed calibrated probabilities reproduce every original output.

| Selected RF | Later accuracy | Macro-F1 | Full precision | Partial recall |
|---|---:|---:|---:|---:|
| Legacy, calibrated |70.1%|0.653|52.0%|51.1%|
| Legacy, raw decisions |69.8%|0.653|44.4%|67.8%|
| Displayed CS, calibrated |69.0%|0.537|67.5%|6.7%|
| Displayed CS, raw decisions |71.4%|0.580|78.4%|28.2%|

Raw legacy accuracy95% grouped bootstrap interval:61.2–76.9%; native59.6–85.9%.
These broad intervals reflect few independent held-out orbits/groups. On2480
nonempty native test frames, raw accuracy drops to58.4% and macro-F1 to0.541.
Raw native reject precision74.3%,recall93.5%; it rejects779 of1085 partial
frames. Thus nominal71.4% is not adequate quality across all three classes.

Within grouped development, legacy calibrated quality-only macro-F1 was0.588,
spatial0.671,temporal0.676. Native quality-only0.532,spatial0.693,temporal0.665.
Spatial information matters; this temporal representation does not solve the
problem. No inference that all temporal/image architectures would fail.

## Potential review reduction

Confidence thresholds were selected on development OOF predictions, targeting
99% empirical precision with≥50 decisions across≥5 groups. Partial predictions
are excluded from automatic handling because they still require masks.

Legacy calibrated threshold proposed87 development rejections with no error,
but yielded zero decisions in the later check. Raw-score thresholds proposed
188 development full/rejected decisions with no error; on later orbits they
yielded one full-accept decision, which was wrong. Native achieved no eligible
threshold for either calibrated or raw scores. Therefore no useful automatic
decision set was established at the proposed target. Diagnostic test-set
precision/coverage curves must not be used retrospectively to claim a
validated operating point. Human time saved has not been measured.

## Label and provenance limitations

Native-empty reviewed orbits169/170/172/178/197 retain populated legacy WIC
and current rejected targets. The legacy model cannot directly observe their
native product failure. Availability remains audit metadata. Native-empty156
also exists but retains an older annotation record. This audit flag was
corrected after extraction without changing predictors,predictions or metrics.
Filtering native-empty later orbits is evaluation sensitivity only; no clean-
label retraining was performed.49 completed reviews retain restricted
footprints, so full acceptance does not uniformly mean whole46x46 review.

## Reproduction and verification

README owns feature/benchmark and --uncalibrated-check commands. Source code:
annotation_ml_features.py and benchmark_annotation_rf.py. Both runs archive
feature arrays,frames CSV,annotation snapshots,source file fingerprints,
calibration/fold group manifests,frame probabilities and summaries. The legacy
directory also contains benchmark_comparison.png and comparison.json.

Known-value tests verify quality orientation,feature whitelist and temporal
gap exclusion. Dataset audit verifies unique frame matches,finite-or-NaN
features,identical centre features and complete disjoint group coverage.
Independent read-only agent review found no prohibited-feature or temporal
leakage blocker. Confidence helper reproduces archived cutoffs; figures were
visually inspected. Existing scientific training weights are unchanged.
