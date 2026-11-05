# Kp-Conditional Conductance Response

Last reviewed: 2026-10-12
Status: saved response and current-selection comparison verified; historical masks unavailable

## Scope and evidence

User trained a separate conditional model at
`~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_smoothed_b16_holes_hybrid`
and generated Kp0–6 ensembles. This is distinct from the older unconditional
`vae_runs/conductance_smoothed_b16_holes_hybrid`.
Live checkpoint:500 epochs, best validation epoch151, beta1.6, latent36,
6760/1044/1547 retained train/validation/test images,2076 supported cells.
Training Kp range0.667–6.667; Kp0 is extrapolation. Source inputs use the
smoothed image-ratio/Robinson retrieval path, not a Hardy-electron target.

Reproduction/report/figures:
`figures/validation/kp_conditional_response_20261012/`.
`reproduce.py` reads small saved maps/latent arrays, never covariance factors.
Each saved ensemble has50000 members; source-model SHA, identical latent arrays,
identical support and all member Kp labels verified across all seven archives.
No model, source, annotations or original ensemble was modified.

## Verified magnitude and spatial response

| Kp | Pedersen median-map max (S) | Pedersen mean-map max (S) |
|---|---:|---:|
|0|3.239|3.251|
|1|3.114|3.134|
|2|2.939|2.978|
|3|3.123|3.124|
|4|3.282|3.297|
|5|3.965|3.959|
|6|4.373|4.397|

The user reports Hardy Pedersen peaks2.95→12.88S over Kp0→6; these Hardy
values are not recomputed as a matched-grid benchmark. Using ensemble mean
rather than median does not resolve the weak peak response.

Spatial changes are appreciable despite the weak peak growth: endpoint
median-map spatial correlation H0.383/P0.551 and RMS difference H1.332/P0.769S
on2076 supported cells. Pedersen median-map cells above2S grow151→252;
these are grid counts, not area integrals. Supported-cell average ensemble
mean grows56.4% Hall/30.9% Pedersen. This demonstrates Kp response, not
physical validity of its patterns. Mean/quantile-map maximum is distinct
from the distribution of maxima of individual sample fields.

Independent read-only code/checkpoint review confirms Kp reaches both
encoder and decoder. A512-common-z decoder check independently reproduces
weak Pedersen peaks and spatial redistribution; no disconnected condition
input was found. Current loss fits quality-weighted standardized-log
retrievals plus KL to N(0,I), without Hardy, monotonicity or activity-bin
balance constraints. Empirical conditioning does not impose Hardy agreement.

## Historical-selection limitation and current test

Exact source/retrieval reload rejects current annotation provenance:
checkpoint annotation SHA=`b13f935719e11a1afbef36a5aa54e5c790f7a3c221e20a0bae21f7d63dc426e2`,
current SHA at review=`ad947fc3c474c98528b55b2ca92c73683816c3a446a00c89128d32827ba58a68`.
No matching copy was found in the run tree or the bounded local backup search.
User confirms no training-time annotation copy is available. Do not bypass the guard
or present current annotations as the exact training selection.

Exact historical masks cannot currently be recovered. User-authorized test
using current annotations is complete; [[Current Annotation Conductance Kp Test]]
owns the frozen-selection, count/orbit/quality-controlled comparison. Current
hybrid Pedersen mean peak is4.43S at bin6, prior4.40S and matched-grid
Hardy12.71S; q90 retrieval/prior peaks11.10/6.97S show additional tail mismatch. Then compare encoded reconstructions and
fixed-Kp prior generation to locate lost activity dependence. Wrong-decoder-Kp
checks at fixed encoded z and aggregate-posterior comparisons by activity are
proposed diagnostics, not yet run. Weak retrieval dependence, reconstruction
smoothing and activity in z remain hypotheses. No architecture change,
retraining, calibration or model adoption is supported by this checkpoint.
