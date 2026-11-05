# Kp-Conditional Conductance VAE

Last reviewed: 2026-10-12
Status: Optional implementation verified; full conditional training/scientific validation pending.

## Implemented checkpoint (2026-10-12)

User authorizes implementation while preserving existing functionality. Intended
science test is whether the ensemble mean at fixed Kp looks more like Hardy.
User will annotate more data, which will take time; no schedule is confirmed.
Their concern that limited data may worsen conditional teleconnections remains
a hypothesis, not a verified outcome or an assumption built into the model.

Implementation is opt-in through `--condition-on-kp --sample-kp VALUE` in the
native runner. Kp/intervals are loaded only for enabled runs, validated against
source UTC, and sliced through JSON/hybrid and intrinsic frame selection.
Normalization uses retained training labels. An8-unit nonlinear Kp embedding
is concatenated before encoder latent heads and decoder `fc_decode`; spline
geometry, transform and N(0,I) prior are retained. Unconditional model configs,
state shapes, module initialization/RNG and numeric outputs stay unchanged.

Training/evaluation forwards batch Kp. Artifacts store conditioning settings,
retained split labels, ranges/quantiles and GFZ source provenance. Reuse checks
mode/settings/source identity; held-out reload checks retained Kp labels. The
runner's sparse illustration uses a prior at its reference image's Kp, with
separate generation/reconstruction condition labels when needed.

`sample_conductance_at_kp.py` generates at multiple selected Kp with identical
z draws, saving physical mean/q10/q50/q90 maps, native anomaly factors, member
Kp and magnetic grid/support plus JSON/PNG/PDF. One Kp ensemble is held in memory
at a time; existing outputs are protected. Outside-training-range Kp is marked
as extrapolation. The model shares weights across all activity levels.

Shared visualizer reconstruction, posterior/prior and inpainting use per-image
Kp; summary/plots identify frame-matched conditional priors. Existing legacy
unconditional prior/event audits and arbitrary-frame covariance reconstruction
reject conditional artifacts to avoid silent fixed-Kp mismatches. Observation
browsing and old saved-run workflows remain supported. Loading example now
allows separate model and sampled-ensemble directories, checking model SHA.

**Verification:**307 tests pass. Independent pre-edit comparison confirms old
native and FUV model configuration, state tensors, constructor RNG and seeded
outputs are bit-identical. The initial compatibility tests found two issues
(a None keyword reaching an old decoder override and a minimal-artifact guard);
both were fixed and the final full suite passes. Real-source eager Kp reading
was tested explicitly, alongside timestamps/label drift and source-selection
alignment. Conditional synthetic training/reload/held-out/inpainting and
multi-Kp generation tests pass. Independent review confirmed core controls and
identified the legacy analysis mismatches now guarded.

A one-epoch, four-source-file CPU integration run succeeds in
`/tmp/icanalyzer_kp_conditional_real_smoke_20261012/`, with32-draw Kp2/3/5 sampling
in `/tmp/icanalyzer_kp_conditional_real_sampling_20261012/`. Kp5 is outside this
small smoke model's training range and is correctly marked. This is software
verification, not a scientifically trained result. Existing production model,
source NetCDFs and annotation JSONs were not changed. README owns commands for
a separate full hybrid conditional training run and subsequent50000-draw maps.

## Intended behavior and feasibility

Original feasibility assessment: the network change is small; integration
across ingestion, dataset, training, sampling and artifacts is moderate.
Scientific validation is the larger uncertainty. The implemented checkpoint
above supersedes the earlier proposal-only status.

Proposed target: paired conductance distribution p(H,P | Kp=k), with generation
from standard-normal latent draws and an explicitly requested Kp. This does
not require an input conductance image. It differs from the earlier event
tests that decoded q(z | one image) from the unconditional VAE. Kp conditioning
does not by itself calibrate image-conditioned uncertainty or fix teleconnections.

## Small first implementation

- Load source `Kp` alongside time/H/P in `icanalyzer/conductance.py`. Preserve
  it through source interval slicing, annotation filtering, concatenation and
  the dataset's retained-frame selection. Exact UTC audit has verified all
  hybrid input labels; integrate at ingestion rather than relying on a CSV join.
- Treat Kp as an ordered continuous scalar. Normalize with training-only
  statistics and save that normalization. Pass it as an explicit batch value,
  not a partially masked image channel. Validate finite values and label shape.
- In `SplineWICSI12VAE`, concatenate the condition with the flattened encoder
  features before mu/logvar heads and with z before `fc_decode`. Preserve the
  existing convolution/spline geometry, paired outputs and physical transform.
- Initially retain p(z)=N(0,I), independent of Kp. Encoder becomes q(z|H,P,Kp),
  decoder takes(z,Kp). A learned conditional latent prior is a later experiment,
  not required for a first conditional model. Same reconstruction/KL machinery
  can be used with conditions routed through training/evaluation calls.
- Extend sampling to a specified scalar Kp or per-member conditions. Save
  member Kp, condition normalization, requested generation settings and input
  Kp coverage with the run. Preserve old unconditional artifact loading and
  existing interfaces by making conditioning an explicit option.
- Train a new experimental model on the same hybrid selection and splits for
  an interpretable baseline comparison. Current weights/samples cannot become
  a trained conditional generator merely by assigning Kp labels afterward.

Data/model paths to change were inspected live: `icanalyzer/conductance.py`,
`icanalyzer/wic_si12/{data,model,workflow,ensemble}.py` and
`scripts/IMAGE/vae_pipeline/run_conductance_pipeline.py`. Artifact save/load,
diagnostics and tests must carry the conditioning settings consistently. The
shared model is also used by FUV runs; preserve their unconditional behavior.

## Validation and data limitations

Generate at several covered Kp values, initially1/3/5/6. Compare physical mean,
pointwise quantiles, spatial/cross-H/P covariance and tails with matched-Kp
held-out retrievals; inspect activity dependence and interpolation. Diagnose
aggregate encoder-prior agreement separately within Kp ranges, not only pooled
across Kp. Verify condition sensitivity using shared z draws; a working input
argument is insufficient evidence that the trained network uses it meaningfully.

Use orbit-held-out tests and consider a stronger time/event-block sensitivity
because nearby frames and Kp intervals are correlated. Hybrid training has6881
frames but179 distinct Kp three-hour intervals. Kp6–<7 contributes447 frames
from11 orbits/13 intervals; Kp<1 only39 frames/3 orbits/3 intervals. Known
range0.667–6.667; do not claim validated behavior atKp0 or7–9. No new population
balancing/default sampling policy is selected in this assessment.

Kp summarizes only part of the state; within-Kp variability can remain broad.
Mean/covariance changes need physical validation, not assumed monotonic changes
at every pixel. Match Hardy/VAE physical components, coordinates, support and
summary definition. The existing beta-weighted MSE objective still does not
establish calibrated true-conductance uncertainty.

## Reference

The fixed standard-normal latent prior with an explicit decoder condition
follows the conditional generative construction in Kingma et al.(2014), M2;
their example condition is categorical, while this proposal uses continuous
Kp. https://arxiv.org/html/1406.5298v2
General CVAE structured-output formulation: Sohn et al.(2015),
https://papers.neurips.cc/paper/5775-learning-structured-output-representation-using-deep-conditional-generative-models.pdf

The initial effort estimate was not a benchmark. Implementation is now complete;
full training compute and scientific acceptance remain separate work.
