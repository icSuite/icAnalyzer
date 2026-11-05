# Lompe Conductance Ensemble Experiment Plan

Last reviewed: 2026-10-12
Status: Proposed design; no inversion or new ensemble generated.
Evidence: icAnalyzer `vae/wic-si12` at `be31406`, with preserved tracked edits;
Lompe primary worktree `time_and_kalman` at `c3b3db3`, initially tracked-clean.

## User intent and scientific scope

Confirmed by the user: compare the ensemble mean/distribution with Lompe's
conductance model, then solve one event with the existing model and multiple
conductance realizations, visualizing solution variability. The demo event is
a candidate, not a final event selection. This adds an electrodynamic
sensitivity experiment to the conductance-prior programme; it does not replace
the previously stated probabilistic conductance-coestimation endpoint.

Proposed first source: existing native Hall/Pedersen VAE. An optional question
asks whether the user instead wants WIC/SI12 draws converted using the new
Hardy electron energy, or both. Source choice remains unconfirmed.

## Verified inputs and baseline

- `examples/lompe_demo.ipynb` selects 2012-04-05 02:00–02:05 UT, midpoint
  02:02:30, Kp 4, a 4200 km × 7000 km geographic CS domain at 100 km resolution.
  It also includes a 17:30–17:35 interval. Both are candidates.
- Its baseline is `lompe.utils.conductance.hardy_EUV`: Hardy 1987 fitted
  Hall/Pedersen conductance plus Moen–Brekke EUV, combined in quadrature;
  default F107=100, starlight=0, Apex mapping at 110 km.
  This fitted conductance is scientifically distinct from the new full-spectrum
  Hardy electron energy + WIC + Robinson route.
- All three 20120405 SuperMAG/Iridium/SuperDARN HDF files exist locally.
  Time-window support, errors and effective observations remain to be checked.
  The notebook sets SuperDARN importance weight0 in its first example: a
  convection validation experiment must explicitly use a nonzero weight or
  deliberately keep convection observations held out. Preserve and report
  the exact baseline configuration rather than silently changing it.
- `examples/demoscripts/demo.py` contains an inconsistent Kp 4 variable versus
  literal 5 in its conductance lambdas. Use the notebook as the candidate's
  explicit configuration; do not copy that mismatch.
- Primary Lompe uses `solve_steady_state` and explicit regularization. The
  historical `run_inversion` implementation is inside an inert string. Port
  the small notebook setup to the current solver; do not modify protected main
  or conflate this experiment with Käki/Kalman development.
- Existing run directory:
  `/home/bing/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_annotated`.
  Model, summary and ensemble archive are present. Header-only inspection
  confirms 50,000 latent draws (36 dimensions), physical-S mean (2,46,46),
  anomaly factor (4232,50000), and 46x46 supported mask.
  There are 1,681 supported cells and 435 unsupported cells. Supported mean
  ranges: Hall 0.0438–11.737 S; Pedersen 0.0848–3.575 S.
  These are unconditional prior moments, not event-conditioned estimates.
- The native data source is the IR_hardy retrieval route: Hardy protons with
  image-ratio electron energy. This existing artifact was not retrained on the
  newly implemented Hardy-electron option. No native conversion is reapplied.
  A new electron route is a separate source-model sensitivity.

## Distribution-preserving sampling

No multivariate Gaussian conductance fit is required. Two direct options:

1. Randomly select complete paired physical-S fields from the saved 50,000
   draws. Use the same member index for Hall, Pedersen and every spatial cell.
   For a pilot, choose distinct indices with a recorded seed. Bootstrap entire
   members with replacement only when estimating finite-ensemble sampling
   error, or when duplicate empirical draws are deliberately desired.
2. For fresh realizations, sample the model's existing latent prior
   z~N(0,I36), decode both channels and apply its saved physical inverse
   transform. The nonlinear decoder/transform induces a non-Gaussian field
   distribution. Preserve model/source hashes, seed and support policy.

The full saved factor retains individual draws, not just covariance. With
A=(X-mu)/sqrt(N-1), recover member i as mu+sqrt(N-1)*A[:,i], reshape 2x46x46.
Float32 factor storage adds rounding only. Sample selection can stream the
archive or re-decode its stored latent vectors without loading the 846 MB factor
into RAM. Truncated PCA eigenfactors/mean/covariance alone cannot reconstruct
this non-Gaussian empirical distribution.

The native log10(Sigma+1) inverse already clamps negative restored values to
zero; reproduce that saved policy and report zero mass. Do not add a new
Gaussian perturbation, pixelwise resampling, separate H/P draws or independent
retrieval-error noise. Model prior variability and measurement-error covariance
are distinct quantities.

General latent-variable sampling reference: Kingma and Welling,
https://arxiv.org/abs/1312.6114. The exact sampling contract is owned by live
`icanalyzer/wic_si12/ensemble.py` and the native runner.

## Latent representativeness check (2026-10-12)

User asks whether latent draws represent the higher-probability region.
Read-only inspection of the saved 50000x36 latent array is consistent with the
configured standard-normal prior: mean squared radius 36.0697 (expected 36),
coordinate variance 0.9942–1.0148, covariance eigenvalues 0.9494–1.0518, maximum
absolute coordinate mean 0.0151. The theoretical central 90% radial interval
is 4.8238–7.1413; 89.922% of stored draws lie inside it.

This confirms a bounded sampling diagnostic, not proof of decoder calibration
or joint empirical conductance representativeness. In high dimensions,
probability mass concentrates in a shell around radius sqrt(d), even though
point density is maximal at the origin. Selecting the smallest-radius/highest-
density latent vectors would alter the prior and suppress typical variability.
Reference: https://mc-stan.org/learn-stan/case-studies/curse-dims.html.

Before the pilot, compare saved prior draws with held-out encoded posterior
samples (including encoder variance, not just posterior means), then compare
prior-generated conductance with held-out observed fields on matched support:
physical distributions/tails, zero mass, morphology and spatial/Hall-Pedersen
dependence. VAE KL regularization encourages prior compatibility; it does not
establish exact equality of the aggregated encoded distribution and N(0,I),
or event-specific validity. Preserve any discovered mismatch as model evidence
rather than filtering latent vectors toward their mode without accounting for
that changed sampling distribution.

## Validation gate outcome (2026-10-12)

The user requested execution. [[Native Conductance Prior Validation]] now owns
the completed test: exact latent covariance mismatch persists with frame and
orbit weights; generated Hall tails/spread are smaller than noisy retrievals
on matched masks in both training and held-out populations. Six test orbits
limit generalization. Standard-normal draws are correct samples of the
configured prior, while empirical conductance representativeness is unverified.
Four figure pairs and JSON/CSV are archived in
`figures/validation/native_conductance_prior_20261012/`.

Before interpreting a Lompe ensemble as representative uncertainty, the next
proposed comparison is a training-fitted latent prior against current N(0,I),
with unchanged held-out evaluation. No alternative prior was fitted/adopted.
The solver experiment can still be an explicitly assumed-prior sensitivity
illustration. No Lompe inversions were run in this validation checkpoint.

## Event-conditioned path verified (2026-10-12)

The user explicitly requested the per-image test. [[Event-Conditioned Conductance Covariance Test]]
now owns six fixed-image q ensembles, full paired matrices, convergence and
input-removal diagnostics. The numerical mechanism works; physical confidence
calibration is unverified. Future IMAGE-conditioned Lompe tests can use these
model-implied ensembles, with input/retrieval error and observation independence
made explicit. This is separate from unconditional prior representativeness.
No electrodynamic solve, real-error calibration or source-model adoption has
been performed. Native matrices/factors are available in the test output.

## Stage 1: conductance comparison

Establish a common geographic/event mapping, with explicit Modified Apex 130 km
versus Lompe 110 km conventions, MLT periodicity, coordinate epoch and all
conductance-evaluation locations required by the solver. Do not rotate fields
by an arbitrary MLT offset to improve agreement.

Separate auroral components first. The learned native source uses electron
Robinson conductance, whereas Lompe's baseline also includes EUV/background.
For the total-conductance experiment, provisionally add exactly the same
EUV/background prescription to every sampled auroral field and baseline,
with combination order documented. Apply nonlinear composition to each member
before calculating moments; do not compose only the ensemble mean. Confirm
source components to avoid double counting proton or solar terms.

Apply one validated smooth/positive interpolation or spline-evaluation rule
to each member, including consistent Hall/Pedersen coordinates and support.
Conductance gradients drive the current/magnetic operators, so nearest-neighbor
mapping and grid artifacts can distort the inversion. Test constant and smooth
fields, periodic seam, physical bounds, and interpolation resolution.

Compare mean, median, standard deviation and 5/95% quantiles in S; map baseline
minus mean and its empirical pointwise percentile. Include Hall/Pedersen ratio,
auroral latitude profiles, joint H/P dependence and spatial gradients. An
unconditional VAE versus Kp 4 baseline is a morphology/prior-coverage diagnostic,
not an equal-conditioning accuracy test. Event-conditioned prior is a later
extension. Do not infer model skill from agreement between two empirical models.

Unsupported cells are numeric placeholders, not physical zeros. Either choose
an inversion domain and all required stencil/perimeter locations within learned
support, or prescribe a common deterministic exterior conductance with an
explicit smooth transition. If a common exterior is used, report which outputs
are constrained by it and repeat a bounded boundary sensitivity check.

## Stage 2: controlled event solves

1. Verify observation counts and geometry in the 02:00–02:05 window. Select
   the alternate interval only for a recorded coverage reason, not on ensemble
   result quality. Freeze event time, observations/errors/weights, grid,
   observation support, boundary rule and mathematical regularization.
2. Obtain a reproducible current-Hardy+EUV reference solution with the current
   steady-state API. Keep original demo reproduction distinct from any change
   in observation weighting/regularization made for the experiment.
3. Solve using the regridded total-conductance ensemble mean. This isolates
   changing mean structure from adding variability.
4. Run a pilot of 20 complete paired members and record runtime/failure rates.
   If affordable, extend nested member sets to 50,100 and200, checking the
   convergence of selected means, quantiles and integrated outputs. These
   counts are proposed milestones, not an accuracy guarantee.
5. Keep the same regularization operator and strength across members. If the
   solver internally rescales regularization with its data/design matrix,
   expose and fix the intended absolute scale for this sensitivity study or
   explicitly report the member-dependent effective prior. Do not retune each
   member to obtain attractive residuals. Rebuild conductance-dependent design
   matrices; verify no stale model/solution cache is reused.

## Outputs and interpretation

Create comparable maps for electric field/convection, horizontal currents and
FAC: baseline, mean-conductance solution, mean of member solutions, and empirical
5/95% limits or interval width. Map their differences. The solution at the mean
conductance need not equal the mean of solutions because the inversion depends
nonlinearly on conductance. Include integrated Joule heating and other chosen
scalars as empirical histograms; preserve samplewise products for heating.

Report observation residuals separately by instrument, held-out predictive
errors where possible, conditioning, runtime, and every failed/unphysical
member. Failures cannot simply be discarded from the reported spread. Pick
representative members by a recorded scalar/medoid rule; pointwise quantile
maps need not correspond to a realizable joint solution. If no independent
held-out observations are available, claim sensitivity rather than improvement.

Equal-weight fits propagate conductance-prior variability through a fixed-data
inverse procedure. Their spread is not a calibrated electrodynamic posterior.
For probabilistic inference later, infer p(z,m|d), where conductance is decoded
from z and m is the electrodynamic state; or draw/weight conductance using a
properly marginalized data likelihood and retain conditional state uncertainty.
Minimum residual alone is not that marginalized likelihood. Total posterior
variance includes both within-conductance state variance and between-conductance
conditional-mean variance under data-conditioned conductance weights.

## First implementation boundary

Proposed next deliverable: source/provenance/support audit, conductance adapter
and comparison figure, one reproducible baseline solve, then 20 paired draws.
Use isolated outputs and source models read-only; no full-day loop, retraining,
corpus rewrite, production-model change or Käki/Kalman modification is implied.
Ensemble source, event interval and exterior-support policy remain choices.
