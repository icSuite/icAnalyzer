# Figure 6: randomly sampled WIC/SI12 reconstruction

Last reviewed: 2026-10-07
Repository evidence: `vae/wic-si12` at `be31406`, with ongoing uncommitted work.

## Scope

User explicitly clarified that figure6 is WIC/SI12, NOT conductance. The first
implementation incorrectly assumed conductance; use the corrected entry point
`scripts/IMAGE/vae_pipeline/plot_wic_si12_random_reconstruction.py`.
Existing figure4 and training code are not changed by this feature.

Use held-out corrected WIC/SI12, log10(counts+1) covariance, Gaussian
conditioning and the same2x4 layout as figure4. Only observation geometry
changes: scattered pixels instead of one missing half. Default pixel PCA
matches figure4; `--prior-space coefficients` reuses its existing legacy-style
coefficient covariance projected through G. No conductance conversion or
conductance ensemble is used. Covariance modes, mask and uncertainty
definitions are explicit; no decoder reconstruction or retraining.

## Run

From the repository root:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_random_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --frame-index 210 --p-remove 98 --prior-space coefficients --modes 20 --seed 0 \
  --output figures/presentation_random_wic_si12
```

`--p-remove` is required, accepts0--100 including fractional percentages, and
applies to currently available valid spatial pixels, not existing gaps.
Remove the nearest integer number without replacement; both fields use the
same mask. `--seed` defaults0. At0 all available pixels are observed; at100
none are observed and the figure shows a prior-only estimate and prior SD.
The prior-only image is the count-space median of the truncated log prior, not
the physical ensemble mean. Defaults otherwise match figure4:50 pixel modes,
0.1 dex assumed independent observation STD, test split and300 DPI. No noise
is added; retained values are exact reference subsamples. WIC/SI12 display
and uncertainty maxima use the existing figure4 option names. Coefficient
decoding options are `--decode-batch-size 64` and `--device cpu|cuda|auto`.

User-requested display update: keep real/sampled-image gaps, but show the full
reconstruction and posterior SD grid, including locations absent in the real
frame. Automatic SD colour limits now use the full grid; first3 colour scales
remain tied to valid reference values. RMSE still excludes original missing
pixels; JSON records display policy and full-grid reconstruction maxima.

Outputs: `figure_6.png`, `figure_6_reconstruction.npz` and JSON containing the
exact mask, frame identity, seed, requested/actual percentage, sample/rank
provenance, missing-pixel RMSE, maxima, clipping and uncertainty assumptions.
At the user's explicit request, reruns overwrite these three figure6 files
without the previous FileExistsError; unrelated files are untouched.
Uncertainty is formal shifted-lognormal SD in counts under the truncated log
covariance, not propagated dwic/dsi12 or calibrated model error. Source model/artifacts are read
only; no retraining, propagation change or adopted scientific setting.

## Verification

30 tests pass: six WIC/SI12 figure6 tests,15 random-mask/endpoint tests
from the initial variant, and nine existing coefficient-prior/OI parity tests.
Verify correct FUV archive fields instead of poisoned conductance fields,
count units/labels, pixel and coefficient paths, frame/mask provenance,
hidden-truth independence,0/100 endpoints, source preservation, requested
figure6 overwrite without touching other files, and unmasked predictions/SD
while reference gaps remain blank.
Real50k WIC/SI12 run, test-frame210/orbit267/time2000-08-31 22:07:14,
98% removal/20 modes/seed0:35 of1749 spatial pixels retained, actual97.999%
removed. Coefficient-space variant13.95 s and1.25 GiB peak CPU-process memory.
PNG visually inspected; WIC/SI12 identity/mask counts/coefficient arrays and
finite posterior maps verified. Missing WIC/SI12 RMSE261.66/2.730 counts;
maxima1437.85/15.40 versus reference3381.66/28.06 counts.
Peak underfitting persists; this single sparse example is not general-skill
validation or a controlled comparison with the old trained model/data.
