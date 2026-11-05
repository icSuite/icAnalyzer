# WIC/SI12 interactive script: unsupported spline directions

Last reviewed: 2026-10-07
Scope: `scripts/IMAGE/step_1_vae_wic_s12_new.py` diagnosis and the authorized
hard-mask implementation in the user's `step_1_vae_wic_s12_new2.py` copy.

## Live workflow and effective constraints

This user-owned script loads smoothed precipitation_cs WIC/SI12 via
helpers.load_data_wic_new, despite internal H/P names and Hall/Pedersen plot
labels. It is not the direct-conductance script or icBuilder spline prototype.
CSV source-index intervals select orbit<503, followed by the33% common-missing
limit. No coverage>=0.8 filter or SZA threshold is applied in this helper.

Current zero-padding fills permanently missing WIC/SI12 with0 and weight0.5,
plus artificial uncertainties7/0.4 counts. Subsequent1-W leaves0.5 unchanged,
so these targets do enter the default reconstruction loss. The prior weight
ordering explanation does not apply to this patch. The target remains a soft
penalty, not a hard constraint on all N(0,I) latent draws.

USE_MEASUREMENT_UNCERTAINTY=False; uncertainty floors7/0.4 therefore do not
affect the active loss. Filling missing cells also changes the partial-conv
encoder mask and global log-field normalization. Treat these as artificial
observations, not ordinary preprocessing of measured values.

Current Spline(grid,40,3) means36x36 coefficients, not40x40; the old34 knots
would give30x30. Encoder6x6 and latent36 remain. This user change is not
authorization to silently restore dimensions. No spatial coefficient
roughness term or inactive-basis mask is present in the decoder/loss.
Sampling/covariance/PCA still include the full coefficient and grid domains.

## Reproduced original-input support

Read raw selected files one orbit at a time, matching helper source-index
selection and common finite WIC/SI12/error/quality masks before zero-padding.
There are8,007 retained frames,28 absent files,14,769,326 common valid samples,
and32 never-observed grid cells, including all four outermost corners.
Each individual camera has the same32 never-observed cells. Support minimum
among observed locations is just one frame. These counts differ from the
earlier conductance audit (6,257/83) because product and helper filters differ.

On the actual reader grid and current spline basis,6/1,296 columns of G
are identically zero over all original supported cells. At the old30x30
basis the count is3/900. Three outer corner coefficients are entirely
unsupported; the fourth influences at least one observed neighboring cell,
even though the outermost corner pixel itself is never observed.

Numerical check: set only these inactive coefficients to0 in three random
coefficient vectors and project through G. Predictions at ALL originally
supported cells are unchanged exactly (maximum difference0). This proves
that wholly unsupported directions can be removed without changing their
direct contribution to fitted observed fields. It does not prove the full
training trajectory or generated covariance will remain unchanged.

## Proposed natural treatment, not implemented

Determine the spatial domain from original retained training inputs, BEFORE
adding artificial values. Keep never-observed locations missing in the encoder
and reconstruction loss; temporary gaps inside the observed domain remain
eligible for inpainting. Remove/fix wholly inactive spline directions in the
decoder and exclude them from coefficient covariance/PCA. Preserve shape if
needed using a fixed coefficient mask rather than changing spline dimensions.

For coefficients that DO affect data but are weakly constrained near the edge,
an explicit spatial smoothness prior can control extrapolation. This is a
scientific modeling change requiring an opt-in, matched test, and measured
tradeoff in auroral detail; no claim that six masked coefficients alone solve
all corner artifacts. Even the fourth corner uses a data-active coefficient.

Covariance/assimilation should retain the original spatial support restriction;
display unobserved locations as missing rather than claiming learned zero
conductance or calibrated covariance there. If extrapolation outside that domain
is required, label it as prior-driven, not locally observed/model-validated.
No source changes, fitting, sample generation or default changes in this audit.

## Authorized point-1 implementation in new2

User copied the script to step_1_vae_wic_s12_new2.py and requested only the
inactive-coefficient mask, deferring targeted regularization until after their
run. The copy now captures actual first10,000-frame H/P/W finite support
immediately after loading, BEFORE any artificial corner filling; with uncertainty
enabled it also requires finite camera errors. After G construction, active
columns are those positive at any genuinely observed training pixel. The mask
is registered as a non-trainable model buffer and multiplies both decoded
coefficient channels before projection and return. This covers the training,
prior-sampling and coefficient-space assimilation decode paths; it is saved
in model state and moves with the model device. Log-space fixed coefficients
are0, not additional physical measurements.

Preserved the user's existing zero-padding, weight inversion, uncertainty
configuration, loss/beta, dimensions and training settings. No smoothness,
ridge or targeted regularization was introduced. Original new.py and canonical
package/runner are unchanged. This narrow change does NOT implement all of the
proposed domain/covariance restrictions above, nor fix the weakly supported
fourth corner coefficient.

Verification:6 focused tests pass, covering original support/training cutoff,
optional error validity, exact supported-field parity, both channels, zero
masked gradients, non-trainable/device-aware state-buffer restoration, a real
forward/backward/Adam step using the existing loss, and prior-sampler zero
coefficient variance. Isolated definitions avoid running the interactive
script or changing process working directory. A streamed real-corpus gate
reproduces8,007 frames/32 unsupported cells and fixes6 of1,296 coefficients
per channel; four CPU prior draws return exact zeros there. No full training,
50,000-sample run or output artifact was generated. Next: user's new2 run,
then assess remaining corners before deciding on targeted regularization.
