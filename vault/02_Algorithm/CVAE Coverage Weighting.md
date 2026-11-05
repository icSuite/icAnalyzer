# CVAE Coverage Weighting

Last reviewed: 2026-10-15

## Confirmed implementation

The user authorized smoothed coverage compensation in the CVAE reconstruction
loss. It is optional; existing defaults and the user-started unweighted
two-stage run are preserved. The user has now started full-weighted training;
completion and scientific performance remain unverified.
After inspecting the cap effect, the user agrees to a fully balanced comparison.
Both full and capped modes are now implemented and verified. The weighting
belongs directly in reconstruction loss, not in input images or the KL term.

For each grid cell and optional broad Kp bin, accumulate the sum of finite,
positive quality weights over **eligible training frames only**, intersected
with the cached common-valid mask. Validation/test data never fit the maps.
This is effective quality coverage, rather than an integer observation count.

Smooth each sum with a Gaussian, normalized by the similarly smoothed indicator
of positive training support. This preserves constant coverage at grid edges
and avoids treating unsupported cells as observed zero coverage. Default sigma
is1.5 native grid pixels. Smooth coverage **before** inversion.

Reference coverage is the maximum smoothed coverage on observed training cells
in that bin. The multiplier is reference/smoothed coverage. Full mode applies
this uncapped inverse; capped mode clips it to[1,3] by default. Unsupported
training cells receive multiplier1, and the original mask still excludes
missing targets. Full balance equalizes estimated smoothed coverage; raw
cumulative weights may retain small differences. Neither mode creates missing
information. Saved full-mode configuration records `maximum_boost: null`.

Multiply the original quality by this fixed map for reconstruction loss only.
The encoder keeps its original quality channel. Existing loss normalization
uses the boosted batch weight sum; a uniform multiplier cancels, and the KL
term remains unchanged. This does not balance the frequency of different Kp.

Default Kp edges[0,9] fit one shared spatial map. Optional edges[0,2,4,6,9]
fit four maps. Lower edges are inclusive; the last bin includes9. Empty training
bins fail explicitly. Runtime bin selection uses the same numerical precision
as fitting, including decimal edges with float32 cached Kp observations.

## Use and provenance

The Python entry `scripts/IMAGE/step_1_two_stage_cvae.py` is prepared with
`COVERAGE_BALANCE = "full"`, while weighting remains off unless enabled. Set
`COVERAGE_WEIGHTING = True` or run it with `--coverage-weighting` from
tConductance. It selects a separate
`conductance_two_stage_valid5_20261015_coverage_full` output automatically.
`--coverage-balance capped` selects the earlier `_coverage` trial. Explicit
`--output` overrides are respected. The direct generic trainer preserves its
capped default; use both `--coverage-weighting --coverage-balance full` there.
README owns full commands and sigma/cap/Kp settings. Full mode ignores the cap.

Every weighted run saves `coverage_weighting.npz` with raw/smoothed sums and
factors, plus a JSON recipe with exact frozen factors, training bin counts and
cache manifest hash. Factors are also embedded in the checkpoint loss config.
Training/validation/test epoch losses reuse them. Postprocessing applies them
to `training_effective` reconstruction metrics, retaining neutral and
quality-only metrics for comparison. Existing caches need no rebuild.

Implementation: `icanalyzer/wic_si12/coverage_weighting.py`, shared workflow,
and the automatic-cache trainer/fitter. The reference diagnostic is under
`figures/validation/wic_coverage_weighting_20261015/`.

## Verification and scientific limit

Fitted the default map on186,342 eligible frames from862 training shards in
2.25s. All119 cells at60<=MLAT<75,10<=MLT<14 receive multiplier3; the median
multiplier at22–02 MLT in the same latitude band is1.372. Across all supported
cells,82.97% reach the cap. These are pre-normalization multipliers, not claims
of equalized influence or improved model performance.

67 focused checks pass, including training-only fitting, quality/validity/
eligibility filtering, normalized smoothing, cap and Kp boundaries, masked
gradients, unchanged encoder input, unit-map loss parity, checkpoint reload,
trainer integration without optimization, postprocessing metrics and Python
launch settings. Saved NPZ/JSON factors and regenerated smoothing agree exactly.
Both modes pass trainer/save/load/evaluation checks without scientific model
optimization. Full-mode tests explicitly verify factors above3, equality of
factor times smoothed coverage, absence of a hidden cap and separate outputs.

Actual full-mode maps are saved under the diagnostic directory's
`full_balance/` subdirectory. They use186,342 training frames, sigma1.5 and no
cap. Factors range1–363.220, with median18.664 near noon and1.372 near midnight
in the same60–75deg band. Raw quality-weighted mean per-cell midnight/noon
ratio is11.454 before and1.079 after full compensation. This measures weighting,
not CVAE reconstruction performance; all supported cells exactly balance the
smoothed coverage to float32 precision, with raw residuals due to smoothing.

## Measured effect of the cap

Follow-up calculation uses summed quality coverage, not just integer counts.
For60<=MLAT<75, compare119 near-noon cells(10–14 MLT) with121 near-midnight
cells(22–02 MLT). Mean cumulative training weight per cell, midnight/noon:

| Maximum boost | Midnight/noon weight ratio |
|---|---:|
|1(no compensation)|11.454|
|3(current default)|5.214|
|6|2.622|
|12|1.589|
|24|1.164|

The3x cap raises noon relative to midnight by2.197x, because the map also
boosts some midnight cells. Noon share of aggregate full-grid training weight
rises2.272%→3.245%(1.428x). These are aggregate weight masses, not actual
residual losses/gradients or exact per-batch normalized influences. Existing
quality weights make the imbalance stronger than suggested by raw observation
counts. The3x value was a conservative starting ceiling, not a fitted optimum;
larger finite caps have only been calculated, not scientifically trained.
Full uncapped mode is now implemented and selected for the user's comparison.
`figures/validation/wic_coverage_weighting_20261015/cap_effect.json` owns totals.

Next scientific test: user-run weighted comparison against the unweighted
two-stage model, evaluated on observed held-out dayside/nightside pixels at
matched Kp. Weighting cannot correct an unobserved auroral population by itself.
