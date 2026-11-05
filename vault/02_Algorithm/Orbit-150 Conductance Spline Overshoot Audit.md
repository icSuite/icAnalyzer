# Orbit-150 direct conductance spline overshoot

Last reviewed: 2026-10-06
Scope: read-only diagnosis of the user's conference spline artifact; no
production fits, source changes, upstream corrections or default changes.

## Artifact and workflow

User identified `or_0150_spline_hall_0211.png` under the external
`conference/SMILE_MWG_2026/spline` directory. Its accompanying
`or_0150_spline.nc` contains an icBuilder direct space/time spline fit, NOT
the icAnalyzer VAE or its prior draws. Producer:
`../icBuilder/scripts/debugging/fit_conductance_cs_splines.py`, consuming
the smoothed Hardy/Robinson `conductance_cs` orbit 150 via icReader.

Saved configuration: 253 retained frames from 257 selected, four empty frames
removed; 40x40 cubic spatial controls, 108 quadratic temporal controls,
5-minute time knots, uncertainty weighting, quality-scaled coefficient ridge,
0.1 S artificial corner constraints, 120 s output cadence. Regularization
parameters H=0.01, P=0.001; effective ridge precisions 12.79498 and 0.0247809.
The quality complement affects coefficient regularization, not residual weights
as in the VAE. Both components use their own dH/dP in this prototype; the
historical legacy P-using-dH bug is not the call path here.

Plot 211 evaluates 2000-06-23T18:33:31, holding source frame 203 at
18:32:30. The one-minute display offset does not explain the artifact.

## Confirmed predictions and observations

Hall has a 126.4968 S dense-grid maximum near xi=-0.41642, eta=-0.08185.
Within +/-10 minutes and the nearest 3x3 observed grid neighborhood, the
maximum observation is 61.6239 S. At the nearest pixel, observations range
2.13--41.25 S with errors 9.38--34.41 S. Stored node predictions also overshoot;
this is not solely between-node contour interpolation.

Pedersen dense maximum is 28.5251 S near xi=-0.37080, eta=-0.09705, versus
a nearest 3x3/20-minute observed maximum of 10.2917 S. At the nearest pixel,
observations are 4.42--10.29 S while STD changes from 0.071--0.086 S to
8.13--31.18 S over the nearby sequence.

Raw dense predictions at plot 211 have minima -3.5493 S (H) and -14.0379 S
(P). Plot code clips negative predictions to zero, hiding their sign. On
the original node grid at the exact plot time, 17 Hall and 24 Pedersen pixels
with held observations above 1 S predict negative values. Example P pixel
(24,8): prediction -9.6753 S, observation 3.1986 S, STD 24.0381 S.
Across all stored observation-time grid cells, 21.16%/20.33% of H/P fits are
negative; on used real observations the fractions are 20.17%/18.88%.

## Confirmed uncertainty limitations

No zero dH or dP values occur in the orbit. Minimum used errors are
0.00154194 S (H) and 0.0134556 S (P). Values below 0.1 S carry 99.99765%
of total Hall inverse-variance precision and 90.5267% of Pedersen precision.
This shows very uneven leverage, not proof those formal measurement errors
were arithmetically computed incorrectly.

There are 6,502 valid cells (1.5227%) with E0=25 keV and dE0=0. At the
Pedersen-spike nearest pixel, the first three nearby observations have these
values, explaining their very small formal dP. Live icPhysics uses a
constant-ended ratio lookup and a finite-difference derivative: ratios beyond
the upper table limit 136.4865 return 25 keV with derivative zero. Checked at
ratios 138.4865 and 200. Thus formal energy uncertainty disappears on the
saturated branch. A saturated lookup is NOT evidence that the physical energy
is known exactly; unrepresented model/range uncertainty remains.

Product metadata explicitly excludes background-model uncertainty and
smoothing/coregistration-induced inter-pixel covariance. The fit treats the
stored errors as independent and has no extra model-discrepancy term. Its
coefficient ridge is not a spatial/temporal derivative smoothing penalty and
does not prevent negative predictions or local overshoot. Quality scaling can
greatly weaken ridge in auroral regions (saved penalty minimum 2.41e-9).
These are limitations of the uncertainty/fit assumptions, not evidence of
isolated enormous input conductance corrupting the map.

## Verified conditional local controls

Using saved knots, masks, targets, errors, coefficient penalty and absolute
ridge, select temporal coefficients active within +/-20 minutes of plot 211.
All 31 observed frames influencing these 17,600 coefficients are retained.
Hold all other coefficients fixed to the saved model and subtract their
contribution from the fit targets. The unmodified conditional solve reproduces
the full saved target-time fields to maximum absolute differences 3.7e-12 S
(H) and 9.9e-11 S (P). This verifies the coordinate/time evaluation and gives
a matched perturbation test without a full-orbit refit.

Change only real-observation STD to sqrt(STD^2 + (1 S)^2), preserving 0.1 S
corner constraints and absolute ridge:

- H original peak-location prediction 126.50 -> 91.82 S; global dense maximum
  126.50 -> 105.97 S; minimum -3.55 -> -0.0000618 S.
- P peak 28.53 -> 15.10 S; minimum -14.04 -> -1.16 S.

These results support an error-model contribution but do not eliminate all
overshoot, establish a calibrated 1 S floor or validate a production change.
Replacing the quality-scaled coefficient penalty by identity strongly suppresses
the Hall spot (126.50 -> 3.39 S), potentially losing aurora; not recommended
as an automatic fix. Multiplying ridge by ten actually increases the Hall
maximum to 181.91 S. Do not assume larger scalar ridge guarantees smoother
predictions under this basis/quality scheme.

An earlier unconditioned 19-frame local fit roughly reproduced P's peak but
not the Hall spot. It changes temporal-boundary context and ridge scaling;
the matched conditional test above owns the stronger evidence.

## Proposed next action, not approved

Test an opt-in model-discrepancy/error-floor setting jointly with spatial
roughness regularization on matched held-out measurements. Keep real propagated
measurement errors, added discrepancy and artificial boundary constraints
distinct. Track overshoot, negative predictions and auroral fidelity, not
only weighted residual RMS. Investigate the saturated-ratio uncertainty branch
with icBuilder/icPhysics owners before changing shared physics or products.
No positive default floor, precision cap, clipping rule or physics repair has
been adopted. No conference artifact was overwritten or refitted.

## Authorized follow-up: hard STD floor implemented

The user subsequently requested a lower uncertainty clip of1 S by default in
the icBuilder spline prototype, with a configurable value. Implemented
--minimum-std / minimum_std;0 disables. This is max(STD,floor), not the
sqrt(STD^2+1) diagnostic above. Apply only to valid real observations; retain
raw dH/dP, invalid masks, quality convention and separate corner STD. Uniform
weighting ignores it. NetCDF records the floor and affected counts. All67
focused prototype tests pass; seven real frames199:206 validate both components
with5,092/5,470 floored observations and294 unchanged0.1 S corner constraints.
Scratch /tmp/icbuilder-spline-floor-D0uPuq. No full-orbit/conference refit or
physics changes. The prior 'no adopted floor' statement describes the original
audit; the user-selected local default is now implemented but fit quality
under that hard floor is not yet evaluated on the intended40-control run.
