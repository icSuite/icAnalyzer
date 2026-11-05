# Current Annotation Conductance Kp Test

Last reviewed: 2026-10-12
Status: verified current-selection comparison; historical masks remain unavailable

## Scope and reproduction

User explicitly requests conductance versus Kp using the current annotation JSON,
after reviewing the weak response of their saved conditional model. This uses
new selection evidence, without bypassing the historical annotation fingerprint.

Script: `scripts/IMAGE/vae_pipeline/measure_conductance_kp.py`.
Canonical results: `figures/validation/kp_current_annotation_comparison_20261012/`.
The directory contains frozen JSON/CSV, per-frame identities/Kp/classes/counts,
small statistic maps, summary, mean/median trend plot and quantile trend plot.
Earlier `kp_conditional_current_annotations*` outputs are intermediate audits;
the canonical output includes all Kp0–9 populations and a single cross-population
plot domain. README owns the command. Source/model/original ensembles/live JSON
were not changed. Checkpoint SHA and source sizes/mtimes are checked.

Selection reuses native ingestion and training eligibility: same source product
folders, completed JSON reviews override whole orbits with CSV fallback,
intrinsic maximum missing fraction0.33, no SZA cutoff. Native H/P are clipped
at zero at ingestion; no extra physical propagation is performed. Current
JSON SHA=`ad947fc3c474c98528b55b2ca92c73683816c3a446a00c89128d32827ba58a68`.

## Population and statistic definitions

All10046 retained hybrid frames contribute from140 orbits, range0.667–9.
6877 are JSON-selected and3169 CSV fallback. Comparison bins0–6 contain9834
hybrid/6665 JSON frames;212 JSON frames at Kp7.667–9 from orbits187/188 are
separately recorded. Those activities exceed original model training coverage.
The original training-orbit subset with current masks has7079 frames; it is
not the original6760-frame training population.

Bins are[k,k+1). Actual median Kp for bins1,3,5,6 is1.667/3.333/5.333/6.
Saved prior/Hardy evaluate exact integer k, so bin/model matching is approximate.
Bin0 observations all have Kp0.667 and cannot directly test generated Kp0.

Neutral physical mean and q10/q50/q90 are computed per observed pixel; excluded
pixels are NaN, never zero. These are maxima of statistical maps, not statistics
of individual image maxima. Every reported common-support cell requires≥20
observations and≥3 contributing orbits in each bin1–6. Separate common domains:
hybrid2069, JSON1262, original-training-orbits2060 cells. The plotted comparison
uses their intersection1262 at every Kp for all populations/models.

Orbit-balanced mean first averages frames within each orbit/cell, then equally
averages contributing orbit means. Quality sensitivity is inverse of weighted
mean log10(S+1), using the actual auroral-weight×coverage field; it is not an
arithmetic physical mean. Count/orbit maps and quality quantiles are archived.
These count thresholds are screening controls, not calibrated error bounds.

## Verified results

Pedersen maximum of physical mean map, on the identical1262-cell domain:

| Kp bin / model Kp | Current hybrid (S) | Saved prior (S) | Lompe Hardy (S) |
|---|---:|---:|---:|
|1|2.651|3.134|4.393|
|3|3.143|3.124|8.750|
|5|4.673|3.959|8.707|
|6|4.431|4.397|12.712|

The current retrieval mean already differs strongly from Hardy. JSON-only
Pedersen mean maxima at bins1/6 are2.687/5.381S; current original-training-orbit
subset2.919/4.513S. Orbit balancing of the hybrid gives2.242/5.115S on its own
2069-cell domain. No examined mean reaches Hardy's12.712S at Kp6. However,
JSON-only bin6 has just131 frames/four orbits/four activity intervals.

Model tail/other-channel mismatch remains: hybrid Pedersen q90 maxima at bins
1/3/5/6 are5.651/6.698/8.668/11.102S, versus prior5.091/5.343/6.321/6.967S.
Hall mean peaks at bins3/5 are12.334/19.870S versus prior8.132/10.120S.
JSON-only Hall means at bins5/6 reach27.902/25.907S. Matching one Pedersen
mean/peak does not establish correct distribution, spatial covariance or skill.

Hardy is live Lompe `hardy(MLAT,MLT,Kp)` evaluated on the saved model grid, with
no EUV or added background. Recomputed Kp6 peak12.712S differs slightly from the
user's12.88S because this comparison explicitly matches grid/support. Physical
components/retrieval calibration and sampling populations may still differ.
Retrievals have measurement uncertainty; no noise propagation/deconvolution
or comparison against independently known true conductance is performed.

## Interpretation and next diagnostic

Confirmed: the large current Pedersen mean gap versus Hardy is already present
in the selected retrievals; conditioning alone is not evidence it should vanish.
Also confirmed: upper Pedersen and Hall distributions differ from saved prior.
This does not identify the cause in unavailable historical training masks.

The loss fits quality-weighted log-space retrievals plus latent KL. No Hardy
amplitude target is supplied. Compression during representation/decoding and
activity-specific mismatch between aggregated encoded latents and N(0,I) remain
possible explanations for additional VAE discrepancies. Proposed next test:
compare current retrievals, decoder(mu), posterior draws and matched-activity
prior draws using exactly the same observed masks. Keep original train/test
orbit labels and call changed-input diagnostics exploratory, not independent
held-out scores. No latent-prior/architecture change or retraining adopted.

## Verification

Independent read-only review and central known-value checks verify neutral
quantiles, invalid-cell exclusion, orbit/log means and channel indexing.
All seven prior maps match original sampler archives. Saved populations sum to
10046, including all212 high-Kp frames; frozen JSON SHA and common supports
verify. Canonical script finishes with unchanged checkpoint/source evidence;
PNG figures inspected. Production model and live annotation files untouched.
