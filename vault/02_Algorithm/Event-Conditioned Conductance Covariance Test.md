# Event-Conditioned Conductance Covariance Test

Last reviewed: 2026-10-12
Status: Covariance generation verified; matched-count teleconnection difference persists; physical calibration open.
Repository: `vae/wic-si12` at `be31406`; pre-existing source/memory work preserved.

User-selected analysis target now `_hybrid` (2026-10-12). Results in this note
remain specific to the earlier `_annotated` model. Separate hybrid rerun is
now recorded in [[Hybrid Conductance VAE Validation]]. [[Loading Saved Conductance Ensembles]]
owns verified hybrid loading, training counts and grid conventions.

## Scope and source

User explicitly requested testing per-event covariance: encode an available
image, sample its approximate encoder posterior and decode many full fields.
The test uses the saved native `conductance_smoothed_b16_holes_annotated` VAE,
unchanged beta 1.6/latent 36 model, paired H/P in S. It is trained on the older
IR_hardy source (image-ratio electrons and Hardy protons), not the newly added
Hardy-electron route. No FUV-model propagation or Lompe solves were performed.

Entry point: `scripts/IMAGE/vae_pipeline/validate_event_conductance_covariance.py`.
README owns the command. Model/ensemble/annotation hashes and input-file stats
are checked unchanged. Output:
`/tmp/icanalyzer_event_covariance_20261012/`, including six full covariance
archives (~800 MB total), base latent noise and masked-input control. Small
figures/JSON/CSV are archived in
`figures/validation/native_event_covariance_20261012/`.

## Calculation and controls

Six cases: one chronological-middle image from each retained test orbit.
Selection depends only on orbit/time, not model skill. Full saved selection
has 788 frames in six orbits. For each fixed image, encode its normalized H/P,
quality channel and paired-valid mask to (mu,logvar). Draw 4096 independent
Gaussian vectors, transform to that image's q(z|x), and decode/inverse-transform
into physical fields. The same base Gaussian draws are shared across images
and the N(0,I) reference to reduce Monte Carlo differences between cases.
Both the original prior and conditional correlation rows use 4096 fresh draws.
The saved 50000-member population ensemble was not used in that figure.

The encoder and outputs exclude never-trained cells. This restriction excluded
zero extra observed input cells for all selected images. Covariance state has
3362 elements: Hall then Pedersen, each on the same 1681 supported cells.
Map fields remain 2x46x46 with NaN outside support. Dense C and mean are float64;
normalized anomaly factors are stored float32. Full cross-cell and H/P blocks
are present; covariance units S². C=AAᵀ uses sample denominator 4095. The
covariance is PSD, not guaranteed positive definite or invertible. Nonlinear
output covariance rank is not limited to the 36 latent dimensions.

Compare nested 256/512/1024/2048/4096 draws, one independent 4096-draw repeat
of orbit 99, and a 12x12 patch removal around its strongest joint log signal.
Covariance from these fixed-input ensembles is separate from the inter-event
covariance examined in [[Native Conductance Prior Validation]].

## Results

| Orbit | UTC image | Trace(C), S² | Median Hall STD, S | Median Pedersen STD, S |
| --- | --- | ---: | ---: | ---: |
| 99 | 2000-05-24 10:24:43 | 6365.8 | 0.198 | 0.136 |
| 106 | 2000-05-28 15:02:17 | 2716.0 | 0.356 | 0.239 |
| 109 | 2000-05-30 09:33:20 | 2647.6 | 0.567 | 0.296 |
| 115 | 2000-06-02 21:22:56 | 2094.3 | 0.296 | 0.216 |
| 136 | 2000-06-15 09:07:55 | 2090.7 | 0.356 | 0.232 |
| 145 | 2000-06-20 14:04:48 | 2932.3 | 0.366 | 0.242 |

The mean fields, variance maps and joint anchor correlations respond to the
encoded image. Conditional traces are 7.1–21.5% of the unconditional reference.
Maximum conditional Hall STD is 4.83–7.77 S; Pedersen 0.57–0.84 S. Medians include
all learned-support cells, not an auroral-brightness-selected footprint.

2048 versus 4096 draws changes STD maps by 1.03–1.42% in relative L2. The
independent seed changes trace by 0.73% but full covariance by 8.93% in relative
Frobenius norm: variance totals are steadier than individual covariance entries.
These are sampling diagnostics, not an absolute scientific acceptance threshold.

## Teleconnection follow-up: matched sample counts

User flagged the spatial structure in `event_correlation_maps.png`.
`validate_event_correlation_convergence.py` repeats N(0,I) and fixed q for
orbits 99/106 with nested 4096/16384/50000 draws and independent seeds
12104/12105. Streaming Welford merges retain physical variances and selected
covariance rows without materializing a 50000-member field ensemble. First
4096 draws exactly match the original noise prefix. Original conditional
anchor correlations reproduce to 1.46e-16/7.18e-17 RMS.

Original anchor is (row 28,column 10), selected by maximum prior Hall STD.
Sensitivity anchors (28,34) and (12,23) are geometrically fixed supported cells.
All correlation maps use the same [-1,1] limits. RMS below is over both source
channels and all supported target H/P cells at the **original anchor**;
undefined correlations are excluded, not filled with zero.

| Distribution | RMS 4096 vs 50000 | RMS independent 50000 seeds | RMS from prior at 50000 |
| --- | ---: | ---: | ---: |
| N(0,I) | 0.01306 | 0.00540 | — |
| q, orbit 99 | 0.01521 | 0.00568 | 0.22280 |
| q, orbit 106 | 0.01300 | 0.00504 | 0.25313 |

Pooling all three anchors gives prior-vs-q RMS 0.19308/0.18205, versus
independent-seed RMS 0.00591/0.00568 for q. The weaker, less coherent conditional
long-range structures persist with more samples and at the additional anchors.
Thus unequal sample counts cannot explain the original figure, and increasing
the count does not recover the prior correlation pattern in these two events.
This check does not establish convergence for all six events, all covariance
entries, or tail probabilities.

The original anchor itself is not degenerate: at 50000 draws q99/q106 Hall
STD is 4.44/0.753 S and Pedersen STD 0.328/0.260 S. Orbit 99 has 111 supported
state elements with exactly zero sampled variance (positivity clipping);
their correlations remain NaN. Orbit 106 has none at this count. Mean absolute
H→H correlation at least 12 grid pixels from the original anchor is 0.1515
for prior, 0.1076 for q99 and 0.0771 for q106; this is a descriptive pixel-distance
summary, not a physical teleconnection criterion.

**Confirmed:** the sampled distributions imply different correlation structures.
**Hypotheses:** conditioning a full image constrains population modes; relative
latent variances, local decoder sensitivities and nonlinear clipping can alter
correlations. Merely scaling every variance down cannot change Pearson
correlation. No Jacobian, unclipped-space or physical-error comparison was
performed, so these mechanisms are not individually established. Aggregate q
already differs from N(0,I); these six image posteriors cannot be used to
decompose the standard prior by the law of total covariance without a consistent
joint distribution. Neither correlation map is validated as true event-error
structure. Review that scientific role before adopting encoder covariance in Lompe.

Report and two PNG/PDF pairs:
`figures/validation/native_event_correlation_20261012/`.
Streaming mean/variance/row/correlation NPZ files stay in
`/tmp/icanalyzer_event_correlation_20261012/`. README owns the exact command.
Nine focused prior/event/streaming tests pass. Independent read-only review
recomputes saved-array metrics and checks state order, normalization and Welford
against direct covariance. Source/model/ensemble/annotations stay unchanged.

## Which model, which population, and are the events unusual?

The unconditional reference and all conditional ensembles use the **same saved
weights** in `conductance_smoothed_b16_holes_annotated/conductance_vae.pt`.
Architecture: native paired H/P `SplineWICSI12VAE`, 46x46 outputs, partial-
convolution encoder, 36 latent variables, 30x30 cubic-spline controls, beta 1.6,
standardized log10(S+1). Checkpoint selected at epoch 156 of 500; training-only
12x12 input holes at probability 0.5 retain original observed targets. Source
is the older smoothed IR_hardy/Robinson retrieval, as described above.

Selected data: 3754 retained training frames from 30 contributing orbits;
623 validation and 788 test frames. Train split names 33 files but three
contribute no retained frames. This is a restricted annotated population,
not the full IMAGE archive. No second VAE was trained on the six test images.
The reference draws N(0,I); each event draws q(z|that one image), using the same
encoder/decoder learned from the training population. Validation selects the
checkpoint; test images remain unseen in training.

`check_event_representation.py` uses cached exact-selection train/test encoder
means and held-out deterministic decoder(mu) reconstructions. Training-only
frame-weighted Cov(mu) has eigenrange 0.2648–2.4774; whitened squared distance
from the training center gives the following descriptive ranks. Reconstruction
ranks use all 788 retained test images and the same observed supported cells.
Lower reconstruction-error rank means better reconstruction.

| Orbit | Latent distance percentile among training images | Hall reconstruction error percentile among test images | Pedersen error percentile |
| --- | ---: | ---: | ---: |
| 99 | 98.4 | 96.3 | 95.1 |
| 106 | 14.2 | 0.8 | 5.5 |
| 109 | 40.4 | 59.9 | 46.8 |
| 115 | 12.9 | 5.6 | 12.2 |
| 136 | 27.2 | 34.8 | 30.5 |
| 145 | 39.0 | 14.1 | 51.5 |

Nearest training-mean distances after whitening support the same contrast:
q99 ranks 94.9th percentile, q106 12.5th against a training reference that
excludes the entire query orbit. Log-reconstruction ranks are less extreme:
q99 Hall/Pedersen 86.2/86.8 and q106 4.7/35.9. Event encoder means match cached
test means within 6.6e-7 (batch rounding); cached source/model evidence matches
the current inputs. Saved result:
`figures/validation/native_event_correlation_20261012/event_representation.json`.

Orbit 99 is relatively unusual and poorly reconstructed by these diagnostics.
The other events, particularly orbit 106 with its also-patchy correlations,
do not share that pattern. This argues against an outlier-only explanation.
These ranks are not density estimates, calibrated OOD probabilities or proof
of adequate morphological coverage: encoder compression, correlated frames,
reference masks and population selection limit interpretation. Good mean
reconstruction does not validate perturbation covariance.

Limited data coverage is a plausible contributor for both ensembles, but is
not established as the cause. Comparing q correlation maps for ordinary
training images versus similarly ordinary held-out images is the next proposed
way to separate seen/unseen generalization from a broader inference issue;
training-data expansion is not adopted as a demonstrated fix.

**User-confirmed criterion:** physically reasonable conditional teleconnections
are required before assuming that the event-conditioned distribution captures
conductance spatial structure. Numerical convergence alone does not meet this
criterion. No model adoption, retraining or Lompe solve follows from this audit.

Removing 144 observed patch cells raises total trace by 15.1%. In the removed
patch, Hall variance falls to 0.783x the full-input value, while Pedersen rises
to 1.312x. A pointwise uncertainty increase is not guaranteed; shifts in decoded
mean/physical scale can also change variance. This one control does not
calibrate uncertainty or prove its direction should always be monotonic.

## Reconstruction and interpretation

Fractions of the original input retrieval lying within the decoded 5–95%
interval range 22.2–53.9% (Hall) and 12.4–42.5% (Pedersen). Those images were
encoded first: these are input reconstruction-containment diagnostics, not
independent confidence coverage or proof of true-conductance error. They warn
against equating encoder spread with an envelope for input reconstruction error.
Retrieval noise, model bias and decoder smoothing are not separated here.

Encoder inputs include quality but not dH/dP. No retrieval-error propagation,
independent decoder residual noise or model-weight uncertainty is added.
The beta-weighted reconstruction objective does not establish a calibrated
physical posterior. The conditional covariance mechanism works; calibrated
uncertainty for real Lompe assimilation remains unverified. Withheld-pixel/data
prediction and a suitable retrieval-error likelihood are the next scientific
checks before that interpretation.

## Numerical verification and artifacts

Three focused tests verify a linear-Gaussian known covariance with cross-field
blocks, sample-covariance/variance agreement, support exclusion and a nonlinear
example whose covariance rank exceeds latent dimension. Eight tests pass when
combined with prior-validation tests. Syntax and scoped whitespace checks pass.

All in-memory matrices are symmetric to printed precision; diagonal/factor
errors are at most 2.2e-14 S², with positive random quadratic forms and agreement
with direct factor products (~3e-15 relative). Independent review of saved
orbit 99 confirms 3362² shape/state order and finite/NaN support policy. Stored
float32 factor diagonal relative L2 error is 1.43e-9; quadratic-probe discrepancy
is 1.03e-9. These export checks differ from in-memory float64 precision.

The first attempt hit a NumPy integer JSON serialization error after computing
all matrices; converting anchor indices to ordinary integers fixed export.
The complete rerun finished with unchanged source evidence. Figure labels
mark saturated scales. Large covariance archives are scratch/disposable; the
repeatable command and durable result remain in the repository.

Each event archive stores mean/std/quantiles, dense covariance, anomaly factor,
state indices, support, mu/logvar and reference/input masks. Base noise plus
encoder parameters and unchanged model reproduce the draws; factor columns
recover the finite paired ensemble without assuming Gaussian physical fields.
