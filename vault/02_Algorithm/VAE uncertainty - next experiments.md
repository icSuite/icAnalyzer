# VAE uncertainty - next experiments

Last reviewed: 2026-10-03
Status: Revised loss and artificial training holes implemented as separate
opt-ins; uncertainty encoder inputs and decoder outputs deferred

The live experiment queue, comparison controls, results, and settings decisions
are owned by [VAE Test Plan](<VAE Test Plan.md>). This note owns scientific
background and implementation interpretation, not a separate execution queue.

## Interpretation

The smoothed uncertainty experiment does use measurement variance in training;
the fields are not discarded. They are absent from encoder inputs and are not
propagated as an observation-noise distribution in prior samples. Weighting
training residuals does not, by itself, produce propagated measurement error.
The desired underlying-field covariance and observation-error covariance must
remain separately identified.

## Scientific considerations behind the test plan

1. The artificial-hole experiment retains originally observed cells as targets.
   Default training still uses only natural validity masks. Capacity changes
   should follow a bottleneck diagnostic on the corrected corpus.
2. Current capped, median-scaled inverse
   variance, divided by the batch's precision sum, is a heuristic objective
   rather than an absolute measurement likelihood. Separate channel balance,
   residual reduction, and KL scale explicitly. Consider a Gaussian likelihood
   with variance `measurement_variance + model_discrepancy_variance`, initially
   one controlled model-discrepancy parameter per channel. This acknowledges
   unresolved spline/model error and prevents tiny measurement variances from
   claiming near-exact representability. If discrepancy variance is fitted,
   retain the Gaussian log-variance term. Equal channel weighting is a chosen
   scientific objective, not a consequence of measurement uncertainty.
3. Test uncertainty as two additional encoder inputs, using a numerically
   stable representation and the same validity mask. It informs posterior
   inference of reliability, but does not alone define a correct likelihood
   or propagate observational uncertainty to output covariance. Keep decoder
   targets as the underlying WIC/SI12 fields.
4. Preserve measurement uncertainty downstream. Use the appropriate actual
   observation's error covariance in sparse updates; combine it with signal
   variability only for predictive noisy-observation statistics. Avoid adding
   the training instruments' noise to clean-field prior covariance. Product-2
   smoothing-induced correlations are currently unavailable, so independent
   pixel errors remain an approximation.

Before treating Gaussian log-space errors literally, check the delta-method
variance transformation near zero and clipped counts. Clipping approximately
25% of SI12 inputs makes a symmetric log-space Gaussian likelihood especially
questionable there. A future count-space or censored likelihood would need
the actual upstream error model, rather than assuming raw Poisson counts for
background-subtracted, coregistered measurements.

Reference: Gaussian likelihood form in the [PyTorch documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.GaussianNLLLoss.html).

## Implemented experiment

The user authorized the revised opt-in loss. `--gaussian-uncertainty` selects
`quality_weighted_gaussian_v1`; `--model-error-std WIC SI12` specifies fixed
model discrepancy in normalized log10 units, default 0.5/0.5. The original
`--measurement-uncertainty` experiment and no-uncertainty default are retained.
For measurement variance s2 and fixed model variance t2, each pixel contributes
`t2 * (error**2 / (s2+t2) + log1p(s2/t2))`. This is twice t2 times Gaussian NLL
with a fixed constant removed, quality-weighted and divided by quality sum,
then multiplied by the legacy state size. It is deliberately a reference-
scaled objective, not a literal untempered ELBO. No precision-sum denominator,
median scaling, or fitted variance is used. Zero measurement variance exactly
recovers legacy MSE; attenuation t2/(s2+t2) lies between zero and one.

On the existing smoothed held-out corpus, 0.5/0.5 yields WIC/SI12 effective
weight-mass shares 51.40%/48.60%, versus 87.38%/12.62% in the old uncertainty
experiment. This diagnoses weighting only; full training skill is not yet
established. All 67 maintained tests pass. A real three-orbit one-epoch smoke
completed training, artifact reload, sampling, propagation, sparse updating,
and all eight posterior/prior diagnostic figures. Reuse rejects changed model
error settings. The architecture and encoder inputs have not changed.

## Implemented artificial holes

The user-authorized hole run is complete. Its test-set screen improves patch
reconstruction but has SI12 fidelity/prior-spread tradeoffs; the live result
and remaining adoption checks are recorded in E02 of the VAE Test Plan.
`--training-holes --hole-probability 0.5 --hole-size 12`
selects one square on half the training images on average. Its center is
drawn from original valid support. Both signal channels and the quality input
are zeroed together and the encoder mask is updated; original targets, loss
weights, and loss mask are retained. Natural gaps never become targets.
Skip holes that remove all remaining observations. Validation and test inputs
are not augmented. Independent RNG seed is run seed + 3; defaults consume no
augmentation randomness. Artifact reuse verifies probability and hole size.

All 70 tests pass, including shared masking, reproducibility, unchanged loss
targets/support, and clean evaluation. A real three-orbit one-epoch smoke
completed training, sampling, propagation, sparse reconstruction, artifact
reload, and eight held-out diagnostic figures; changed hole size is rejected
on artifact reuse. This is operational verification, not evidence of improved
full-run reconstruction. Architecture, latent/spline dimensions, and loss
semantics are unchanged. Test holes separately from the uncertainty option.

## Additional observed evidence

`wic_si12_smoothed_b1` has only a small reconstruction gain over beta 1.6:
auroral RMSE 0.459/0.589 versus 0.469/0.591. Both selected epoch 167. KL is
40.03 rather than 29.36 nats/frame; inpainting remains mixed.

`wic_si12_smoothed_b05` is now complete with the same seed/splits/selection
and other settings. It selects epoch 86, has KL 57.55 nats/frame, and auroral
RMSE 0.451/0.567. Versus beta 1.6, reconstruction improves only about 4% while
prior standard deviations worsen from 243/5.41 to 189/4.12 counts (observed
332/7.24). Inpainting is mixed at 0.711/0.882 versus 0.718/0.816. KL changes
substantially, so beta is active, but this is not an overall model improvement.

Clarification/proposal: retain the legacy log10 encoder input. Test the existing
fixed-model-error loss separately after holes. A future observation likelihood
could instead evaluate residuals and supplied errors in original count units,
with explicit treatment of zero-clipped measurements and channel scales,
avoiding the local derivative approximation of measurement variance. This is
not implemented or authorized and does not assume a raw Poisson model for
background-subtracted, corrected Product-2 fields.

Background count uncertainty is not observed to be unnaturally low: zero-
count WIC cells have median supplied error 12.14 counts versus 25.30 in the
brightest decile. Their first-order normalized-log errors are 5.096 versus
0.0146, explaining the old precision contrast. Zero-count SI12 error is
0.729 counts versus 1.234 in the brightest decile. No inspected supported
cells in these groups have zero reported measurement error. Zero-count groups
include clipped measurements and are not a pure background classification.

Adding two uncertainty encoder channels would require only 288 extra learned
parameters in the existing 3x3, 16-output first convolution; conditioning and
normalization need evaluation, rather than capacity concerns. The user
explicitly deferred uncertainty encoder inputs and discussion of uncertainty
as decoder outputs until other options have been tested.
