# Native Conductance Prior Validation

Last reviewed: 2026-10-12
Status: Verified diagnostic mismatch; no model/prior change adopted.
Repository: `vae/wic-si12` at `be31406`, existing uncommitted work preserved.

## Question, source and experiment

User asked whether standard-normal latent draws represent high-probability
conductance regions and requested the proposed validation. The saved native
run `conductance_smoothed_b16_holes_annotated` uses beta 1.6, latent 36,
50,000 prior draws and annotated native 46x46 Hall/Pedersen retrievals.
Its source is IR_hardy: image-ratio electron energy with Hardy protons. It was
not retrained on the new Hardy-electron option. Model/ensemble/source/annotation
inputs were inspected read-only, with evidence checked unchanged after running.

New repeatable entry point:
`scripts/IMAGE/vae_pipeline/validate_conductance_prior.py`.
README owns the command. Dedicated outputs:
`/tmp/icanalyzer_prior_validation_20261012/validation.json`,
`orbit_metrics.csv`, latent parameter archives and four PNG/PDF pairs.
Four figure pairs and JSON/CSV are also archived in
`figures/validation/native_conductance_prior_20261012/`. Scratch arrays are
disposable; this note owns the durable result.

Reloaded exact saved selection: training 3754 frames/30 orbits; test 788 frames/
6 orbits. Excluded 435 unsupported cells. Generated and reference summaries
use the same masks. Priors are complete paired members from the original
50k ensemble, selected without latent-density filtering. Primary prior draw
counts match each population's frames; three independent 788-member selections
check held-out Monte Carlo variability.

## Latent result

The earlier radius check verified the configured Gaussian RNG, not empirical
representativeness. Compute exact mixture moments:

```text
mean_q = weighted_mean(mu)
cov_q = weighted_covariance_population(mu) + diag(weighted_mean(exp(logvar)))
```

The main population gives each orbit equal mass. Eight q samples/frame and 64
fixed random projections inspect distribution shapes, with an independently
sampled Gaussian-vs-Gaussian control. No iid pixel/frame p-values are claimed.

| Quantity | Training | Held-out |
| --- | ---: | ---: |
| Orbit-balanced aggregate covariance eigenvalue range | 0.422–2.643 | 0.182–7.161 |
| Frame-weighted covariance eigenvalue range | 0.446–2.500 | 0.179–6.445 |
| Orbit-balanced aggregate mean norm | 0.883 | 1.204 |
| Expected squared radius (prior 36) | 38.61 | 35.13 |
| Fraction in prior central 90% radial interval | 84.10% | 77.39% |
| Mean projected Wasserstein distance | 0.135 | 0.199 |
| Gaussian-vs-Gaussian projection control | 0.017 | 0.022 |

A standard-normal prior has covariance eigenvalues 1 and mean 0. Frame weighting
confirms the discrepancy is not caused solely by orbit balancing. Radius alone
misses substantial directional mismatch. The stronger test discrepancy may
include held-out population shift; thirty training orbits show mismatch too.
Conditional encoder KL (~40 nats/frame) is not aggregate-prior KL. The
moment-matched Gaussian gives an aggregate KL lower bound 2.34 (train)/7.91 (test)
nats; it is not an estimate of the full multimodal distribution divergence.

## Physical result

Primary distribution: equal orbit weights, physical cell-area normalization
within each frame, identical observation masks. No quality selection in this
primary table. S values below are pooled distributions under that weighting.

| Hall conductance | Training retrieval | Training prior | Held-out retrieval | Held-out prior |
| --- | ---: | ---: | ---: | ---: |
| Mean, S | 3.44 | 3.03 | 4.34 | 3.08 |
| Standard deviation, S | 8.92 | 5.57 | 10.69 | 5.56 |
| 95th percentile, S | 23.58 | 13.83 | 29.58 | 14.00 |
| Fraction above 20 S | 6.16% | 2.35% | 8.19% | 2.43% |

Three held-out prior selections give Hall means 3.05–3.08 S, STD 5.52–5.58 S,
p95 13.73–14.00 S; the tail gap exceeds this Monte Carlo spread. The
held-out neutral Hall mean difference is -1.26 S with a six-orbit bootstrap 95%
interval [-4.03,+0.71] S: no statistically established population-wide mean bias.
Pedersen held-out means 1.35 S (reference)/1.50 S (prior), STD 1.96/1.42 S and
p95 5.61/4.36 S. The strongest physical discrepancy is Hall variability/tails.

Positive-quality sensitivity amplifies the difference: test Hall means 15.04 S
reference versus 5.16 S prior. However auroral quality depends on signal, and an
independent prior member does not preserve observed signal-quality alignment.
That deficit and its negative bootstrap interval cannot alone establish an
unconditional prior bias. Even neutral masks can be activity-dependent.

## Dependence and limits

Exact pairwise-valid temporal correlations at fixed grid cell pairs, with
orbit weights, exceed the reference in generated fields. Held-out neighboring
Hall median correlation 0.803 (reference),0.952 (decoded q),0.934 (prior);
Pedersen 0.820,0.972,0.959. Same-cell H/P correlations 0.930,0.954,0.938.
Decoder(q) has similar smoothing, so this is lower small-scale variation
relative to retrievals, not proof that the latent prior alone is responsible.
Spline representation and denoising are plausible contributors, not diagnosed
causes. Lags are CS column offsets, not physical separation bins.

The references contain upstream retrieval assumptions and measurement errors;
no noise deconvolution or extra independent decoder noise is supplied. A noisy
retrieval can legitimately have more variation than a noise-free model field.
Six held-out orbits, annotation selection, unequal activity and observation
support limit population conclusions. Training and held-out populations were
analyzed separately; held-out data did not fit or change the model/prior.
The common-valid support mask is not a guarantee of positive training-loss
weight at every supported cell. No such stronger support claim is made.

## Verification and next action

Five focused tests verify exact mixture covariance, equal-orbit/quality-zero
weights, area/missing-mask handling, pairwise correlation and full paired-member
recovery for C/Fortran storage. Syntax and scoped whitespace checks pass.
Independent read-only review finds no consequential arithmetic bug; all 32
reported means match independently averaged per-orbit CSV values within
2.5e-14. Original model/ensemble/annotation/source evidence remains unchanged.
Four figure pairs were generated; main latent/distribution figures inspected.

Conclusion: withhold a claim that the standard-normal decoded prior represents
this empirical conductance population. Its draws sample their configured prior
correctly; the aggregate encoding and Hall tails differ materially from the
chosen retrieval reference. This is not a true-conductance calibration test.

Proposed next diagnostic: compare standard-normal sampling with a latent prior
fitted only to training encodings (for example the full empirical q mixture),
then evaluate unchanged held-out data. Keep reconstruction smoothing/noise
separate from latent-prior mismatch. That alternative is not adopted or fitted
here. A Lompe run can still illustrate sensitivity under the current assumed
prior, but its spread cannot yet be called representative event uncertainty.

## Event-conditioned covariance clarification (2026-10-12)

Confirmed by the user: an intended real use case is event-specific conductance
covariance obtained by encoding an available event image, then sampling and
decoding its encoder posterior. The image must match the trained inputs:
paired H/P for the native model, or the corresponding paired FUV inputs for
a FUV model. This is a distinct population from unconditional prior draws and
from the proposed mixture across training encodings.

For fixed input x, draw z from q(z|x), decode full paired physical fields,
and calculate their sample mean/covariance. The result is model-implied
conditional covariance; encoder variance is not yet calibrated camera/retrieval
error propagation and model-weight uncertainty is excluded. Conditional spread
needs validation before use as calibrated uncertainty in Lompe. No sampler,
model or inversion implementation was changed by this clarification.

Probability clarification: Gaussian latent density is not decoded-field
density. Probability mass of an output region equals the latent mass of its
full preimage. Common decoded fields under the generator can still be outliers
relative to observed data. High latent point density at zero is distinct from
typical probability mass; 36D standard-normal draws lie mostly near radius 6.
