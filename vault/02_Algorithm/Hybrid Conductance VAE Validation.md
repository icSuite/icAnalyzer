# Hybrid Conductance VAE Validation

Last reviewed: 2026-10-12
Status: Prior, event covariance, correlation convergence and representation audits rerun; physical spatial validation remains open.

## Model and scope

User directed switching to `conductance_smoothed_b16_holes_hybrid`. The previous
reply had verified loading/moments/grid only; this checkpoint records the
actual diagnostic rerun. Same saved native H/P spline VAE, beta1.6, latent36,
selected epoch207. No retraining or changes to source/model/ensemble/annotations.
Source remains smoothed image-ratio electrons/Hardy protons with Robinson
conductance, not the newly implemented Hardy-electron option.

Exact retained populations reload successfully:6881 training frames/100
contributing orbits,1117 test frames/19 contributing orbits. Saved split lists
115/24 files, respectively; some contribute no retained frames. Validation
summary has1203 frames. Learned support2077 cells on46x46;39 excluded. Joint
state4154 elements, supported Hall then supported Pedersen in row-major order.

Entry points: `validate_conductance_prior.py`,
`validate_event_conductance_covariance.py`,
`validate_event_correlation_convergence.py`, `check_event_representation.py`.
README owns exact rerun commands. Small outputs are archived under
`figures/validation/hybrid_conductance_validation_20261012/{prior,event,correlation}/`.
Large matrices and inference/streaming arrays remain in explicit scratch trees
`/tmp/icanalyzer_hybrid_{prior_validation,event_covariance,event_correlation}_20261012/`.

## Unconditional prior versus training and held-out retrievals

Prior-validation methods follow [[Native Conductance Prior Validation]], with
the hybrid model and its own exact retained selections. Aggregate q includes
Cov(mu) and mean conditional variance, with equal orbit weights.

| Split | Aggregate q covariance eigenrange | Hall STD retrieval / prior, S | Hall p95 retrieval / prior, S |
| --- | --- | --- | --- |
| Train | 0.418–2.325 | 6.448 / 3.968 | 11.032 / 8.336 |
| Test | 0.203–4.387 | 6.584 / 4.046 | 11.583 / 8.686 |

Physical summaries use neutral pixel-area weights and identical per-image
observed masks. Quality-weighted and frame-weighted sensitivities are saved.
Latent mismatch and compressed physical tails persist for this population.
Retrievals are noisy and selection is signal dependent; this does not calibrate
true-conductance errors or prove a population-wide bias. Different retained
populations/support prevent reading the earlier annotated-versus-hybrid numbers
as a controlled measure of improvement due to more training data.

## Within-event covariance and teleconnections

One chronological-middle image from **each of19 retained held-out orbits**,
chosen without reconstruction-skill filtering. Both prior reference and event
ensembles use4096 freshly decoded draws with shared base noise (seed12104).
Dense4154x4154 covariances and paired anomaly factors are exported in S²;
unsupported map cells remain NaN. The matrices are PSD by construction but
rank at most4095 at this sample count, hence singular even before zero-variance
states. No solver requiring an inverse is attempted.

Conditional traces are0.67–13.14% of prior trace (median3.95%). STD-map relative
L2 difference2048→4096 is1.01–2.82% (median1.28%). Independent4096 seed for
orbit98 changes trace0.11% and full C9.06% in Frobenius norm. Encoded-input
90% containment spans31–96% Hall and22–61% Pedersen; this is reconstruction
containment, not independent uncertainty coverage. No dH/dP noise propagation,
decoder residual uncertainty or model-weight uncertainty is added.

Removing144 observed patch cells for orbit98 reduces total trace to0.524x,
with removed-patch Hall/Pedersen variance0.171x/0.658x. This highlights that
current encoder spread is not guaranteed to behave as a calibrated observation-
error posterior; mean/physical-scale changes also affect variance.

Correlations in the19-event figure remain visibly patchy compared with the
broad unconditional auroral structure. Follow-up tests use the first two
predetermined events, orbits98/103, with4096/16384/50000 draws and independent
seeds12104/12105. Anchor(10,18) maximizes this run's prior Hall STD; sensitivity
anchors(28,34),(12,23) are fixed geometric cells. Evidence checks prevent
using event parameters from another model/run.

RMS below pools both source channels and all supported target H/P states at
the primary anchor; common finite entries only:

| Distribution | RMS4096 vs50000 | RMS independent50000 seeds | RMS from prior at50000 |
| --- | ---: | ---: | ---: |
| N(0,I) | 0.01387 | 0.00654 | — |
| q, orbit98 | 0.01806 | 0.00654 | 0.19039 |
| q, orbit103 | 0.01418 | 0.00487 | 0.20581 |

Pooling all three anchors gives prior-vs-q RMS0.16186/0.18945, versus seed
RMS0.00625/0.00605. Original4096 conditional rows reproduce to~1e-16 RMS.
Zero sampled variance at50000 occurs in20/77 supported states; correlations
remain NaN. The primary anchor itself has nonzero STD. Mean absolute distant
H→H correlation (at least12 grid pixels away) is0.1353 prior,0.0866 q98,
0.0708 q103; descriptive grid-distance summary only.

More samples preserve the patchy structures in the two checked events and
do not restore the prior pattern. The larger training set therefore does not
resolve the observed concern in these cases. This is not a proof that data
coverage is irrelevant, that every covariance entry converged, or that the
unconditional map is physically calibrated. No Jacobian/observation-likelihood
test isolates why conditioning changes correlations.

## Are these selected events unusual?

Training-only frame-weighted Cov(mu) whitens encoder means. Training nearest-
neighbor references exclude the entire query orbit. Rank diagnostics and
deterministic decoder(mu) reconstruction errors are saved for all19 cases.
Selected training-distance percentiles span3.31–87.28; none exceeds95th.
Orbits98/103 rank78.4/74.3 for distance and75.7/79.9 for held-out Hall error.
Other selected events include low-distance, well-reconstructed images whose
conditional correlation maps also differ from the broad prior pattern.

Thus an explanation based only on extreme event inputs is unsupported by
these diagnostics. Ranks are descriptive, not density estimates or calibrated
OOD probabilities. Encoder compression, missingness and correlated frames
limit interpretation; good mean reconstruction does not validate covariance.
Test input exclusion removes zero extra untrained cells across this population.

## Verification and next scientific step

Ten focused prior/event/streaming/representation tests pass. Saved model and
source evidence is unchanged before/after all substantive runs. Grid and
support were verified separately in [[Loading Saved Conductance Ensembles]].
Independent review verifies the live hybrid model hash in all three reports,
all19 chronological-midpoint selections, support and exact primary-anchor RMS
values recomputed from saved arrays. No numerical error is found; different
cohorts/support/primary anchors limit comparison with the annotated run.
Prior validation now saves its reconstruction cache directly for repeatable
representation ranks; the first hybrid cache was regenerated with the same
inputs/seed and its prior members aligned to stored selected indices.

The user's criterion stands: conditional teleconnections must be physically
reasonable before claiming that the model captures event spatial structure.
Next proposed diagnostic compares ordinary training-image q correlations with
ordinary held-out-image q correlations, then examines local decoder response
and inferred latent variance. No method/model adoption or Lompe solve.
