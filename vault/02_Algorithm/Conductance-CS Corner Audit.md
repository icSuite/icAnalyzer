# Direct conductance VAE: corner/support and weight audit

Last reviewed: 2026-10-06
Scope: user-owned `scripts/IMAGE/primary_log_spline_vae_new.py` and its
`helpers.load_data_new` input path; diagnosis only, no source/product changes.

## Confirmed input and selection

The script reads `conductance_cs/image_apex_130km_46x46_v1/`
`IR_hardy_smoothed_wic0p8_si120_si131p2/robinson` under the user's IMAGE_FUV
data root. These are schema-2 detector-derived Robinson conductance products,
not the canonical paired-WIC/SI12 fixed-energy propagation artifacts.

Replicating the helper's current CSV selection (orbit <503), positional good
intervals, coverage >=0.8 and missing fraction <=0.33 retains 6,257 frames and
10,612,733 common finite H/P/quality cells. The 357 CSV orbit rows are unique;
28 files are absent. Positional/source-index intervals agree on this corpus,
so that distinction is not the cause here. No retained cells have a false
conductance_valid flag or nonfinite coverage. For orbit 85, icReader H/P/
coverage/quality values exactly match the raw NetCDF fields.

## Confirmed problems

1. The new script does `W = 1-W` after loading method_quality_weight. The
   product explicitly describes the field as auroral quality, `1 - dgweight`,
   already inverted upstream. This extra inversion reverses both the loss and
   encoder's quality channel. On retained cells with q>=0.9, mean original
   quality is 0.97490 but mean used weight is 0.02510. On q<=0.01 cells, mean
   original quality is 0.00297 but used weight is 0.99703. This is a confirmed
   loader/consumer bug, not an ambiguity about upstream orientation.
   User clarification: this extra inversion was intentional during debugging.
   Do not silently remove it or treat it as the established corner-spike cause.
2. There are 83/2,116 grid locations with no retained training support,
   including all four outermost corner cells. Grid corner latitudes are about
   40 degrees. The script's loss masks gaps correctly, but its decoder,
   physical inverse transform, sample displays and covariance calculation
   retain all grid locations. Predictions there are unconstrained extrapolation,
   not locally learned conductance. Exponentiating log outputs can amplify
   extrapolation; unsupported corners cannot directly dominate the masked loss,
   but can contaminate full-grid sample/covariance statistics.

The user clarified that the artifacts appear in PCA of latent samples, not
in the random input maps inspected so far. Both PCA stages include unsupported
outputs: the first uses all spline coefficients and the second all physical
grid cells. On the actual spline basis, 17/900 coefficients have no influence
on any retained observed grid cell, including all four corner coefficients.
These coefficients have no direct reconstruction-loss constraint; convolutional
weight sharing does not provide a local observation there.

The old conductance corpus, using the same CSV intervals and 33% missing limit
but its original helper (no coverage threshold), retains 11,136 frames and
has no permanently unsupported cells on its 36x36 grid. The four corners have
8,648 / 7,790 / 6,036 / 6,460 observed frames; their latitudes are about
45 degrees, versus about 40 degrees for the new grid. The old script truncates
training to 10,000 frames, but each corner necessarily still has support in
that subset. Thus the unchanged full-domain PCA assumption is materially less
safe on the new dataset. This is a domain/support change, not evidence that
new raw conductance has enormous finite corner outliers.

## What is not established

Raw-data scan did not find enormous finite corner fill values. On retained
support, H/P maxima are 165.19/31.85 S; highest values are not at the four
outer corners. The Pedersen maximum is orbit 386/frame 41, cell (20,5);
the Hall maximum is orbit 134/frame 26, cell (20,33). These values alone are
not proof of invalid precipitation. No negative retained H/P values were found.

The actual corner-spike panel/model state has not been supplied. The user
identified PCA of latent samples as the affected analysis, but not which of
the two PCA stages. Unsupported outputs are a strongly supported mechanism;
the actual model's anomalous sample values/variance have not been reproduced.
The script's spline basis was checked on the actual reader grid: finite,
nonnegative, row sums within 1e-15 of one; no numerical basis blow-up at corners.

## Proposed correction, awaiting authorization

Use the declared auroral-quality field directly; do not arbitrarily clip
large conductance targets. Carry a training-derived spatial support mask into
display, sample/covariance statistics and sparse-assimilation scoring. This
must exclude unsupported locations rather than zero-filling them as physical
observations. Restore their full-grid positions as NaNs for display if needed.
Coefficient-space diagnostics need a basis-derived coefficient support mask,
not the differently shaped grid mask. Inspect the exact affected frame/panel
before inventing an outlier cutoff.
Existing model results with reversed weights require retraining after a fix.

Input-side alternative: derive never-observed cells
from common finite H/P/W across the actual retained training frames, before
changing those inputs. Set H=P=0 and W=1 at those locations after helper
filtering and any weight inversion, before log transformation/normalization.
This makes the existing training mask include artificial zero-boundary targets;
changing H/P alone while W remains NaN would still exclude them. It requires
retraining and imposes a zero boundary, not a physical measurement. It does
not guarantee all prior samples will obey that boundary exactly. Output masking
avoids retraining, but is not the only simple intervention. Neither approach
was implemented by the agent; do not regenerate the upstream data product.

## 2026-10-05 patch review: zero boundary still had zero loss weight

The user implemented input zero-filling and reports prior-sample corner spikes
persist. The script inspected on 2026-10-05 sets boundary W=1, fills H/P with zero, then runs
the deliberate `W = 1-W` debugging inversion. Thus final boundary W=0, so the
zero targets have no direct reconstruction-loss penalty. Executing the exact
current AST assignment block on a two-frame missing-corner fixture reproduced
H=P=0, W=0, a valid training mask, and effective loss weight zero.

The initially inspected version also selected P gaps using H after H was
filled; the latest live file uses P's own missing mask, so do not report that
as a remaining defect. The proposed minimal correction is to retain the
intentional inversion but place it before boundary W=1 assignment, storing a
single common-finite support mask before filling H/P/W. This requires retraining;
it creates a reconstruction penalty, not a hard constraint on every prior draw.
No source edits, retraining, or actual-sample variance checks were performed
by the agent during this patch review.

## 2026-10-06: boundary-weight experiment and reliability distinction

User reports setting unsupported W=0.5 removed the corner problem, whereas
setting it to 1 brought the problem back. The configurations/artifacts of those
runs were not supplied. Live code now assigns unsupported W=0 before the same
inversion, yielding final W=1. Executing the exact block with assignment values
0 / 0.5 / 1 confirms final/effective loss weights 1 / 0.5 / 0. Thus if the
reported experiment varied the pre-inversion assignment, the outcome is
consistent with removing boundary constraints, not evidence that a larger
final weight destabilizes the loss. If the user varied final weights instead,
that remains a separate unverified result. Record final weights when comparing.

User wants trustworthy aurora AND trustworthy background retained, with aurora
favored. Distinguish reconstruction emphasis from independent reliability.
The product's auroral-quality proxy q=1-dgweight is not a calibrated reliability
measure; a high background-fit residual can also indicate an artifact. Proposed,
not adopted: a positive background-weight floor, e.g. w=0.5+0.5*q on valid
measured cells, which gives background half the auroral weight. This preserves
background constraints but does not infer trustworthiness. Reliability needs
independent quality evidence; fabricated zero-boundary targets must stay labeled
as artificial constraints, not trustworthy observations. No loss/source changes
or retraining performed; uncertainty integration remains tabled.

No training, source edits, upstream changes or model defaults were made by
this audit. Paired-WIC/SI12 E05 regularized comparison remains a separate,
unapproved proposed next experiment; this user-owned script is not silently
promoted to the canonical workflow.
