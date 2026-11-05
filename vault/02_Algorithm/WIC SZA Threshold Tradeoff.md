# WIC conservative SZA threshold tradeoff

Last reviewed: 2026-10-14  
Status: Frozen-model diagnostic complete; no policy adoption.

## Question and boundaries

User asks whether acceptance can be more conservative, particularly below
SZA90°/105°, and whether using SZA outside a U-Net that already sees SZA is
circular. A separate threshold is a decision policy; it can express a higher
cost for retaining undesired observations under chosen conditions. It adds
neither information nor calibrated physical-good probabilities. No desired
false-keep/false-reject cost ratio has been confirmed by the user.

The production checkpoint remains`e5c009e6713b...`, threshold0.475, unchanged.
No training, corpus reannotation, CVAE cache rebuild or CVAE rerun. Current
manual labels describe conservative selection, not independently verified noise.

## Frozen validation protocol

Compare higher global thresholds and higher thresholds only below90°/105°.
Candidates0.55/0.65/0.75/0.85/0.9/0.95. Above each cut and when SZA is missing,
adaptive policies retain threshold0.475. Hard SZA90/105 exclusions are controls.
Raw SZA is reread from source NetCDF with native frame/time/grid checks rather
than inferred from normalized network inputs. Existing frozen inputs and
original validation/test splits match the production normalization exactly.

The later four-component validation orbitJ selects0.65 for each family before
test/audit evaluation. This differs from the original class/orbit-balanced pixel-disagreement
criterion that selected0.475. The objective is retained for comparison,
not an assertion of the user's preferred contamination-versus-coverage cost.
Audit uses effective mask classes; old splits retain original semantic labels.
The prior eight-orbit audit has informed hypotheses; results are exploratory.

## Audit results

The table uses available finite-WIC pixel-frame pairs atMLAT>=60 and06<=MLT<18.
Adaptive policies themselves depend on per-pixel SZA, which is not identical
to the MLT definition. They can also affect MLT-night observations.

| Policy | Manual-rejected day pixels retained | Manual-kept day pixels lost | AuditJ |
| --- | ---: | ---: | ---: |
| Current0.475 | 19.16% | 34.92% | 25.43% |
| Global0.65 | 14.50% | 44.37% | 26.63% |
| 0.65 belowSZA90°,else0.475 | 15.86% | 39.13% | 26.31% |
| 0.65 belowSZA105°,else0.475 | 14.72% | 42.23% | 26.36% |
| Hard excludeSZA<90° | 4.28% | 60.80% | 36.65% |
| Hard excludeSZA<105° | 0.66% | 80.36% | 41.29% |

Manual-rejected-retained divides by manual-rejected available pairs; useful-loss
divides by manual-kept available pairs. Audit SZA<90° cells specifically show
15.05→10.57% rejected retention and40.04→49.84% useful loss at threshold0.65.
Current original-testJ27.09% becomes26.96%(adaptive90°) or27.36%(adaptive105°).
No overall balanced improvement or actual physical-noise removal established.
The change is a transparent tradeoff, and hard exclusion is costly.

## Verification and artifacts

`scripts/automatic_masking/test_sza_thresholds.py` writes fresh diagnostics
to`figures/validation/wic_sza_thresholds_20261014_final/`: protocol, validation
curves, frozen selections, test/audit probabilities and SZA, comparisonJSON,
readable report and tradeoffPNG/PDF. Current inference exactly reproduces all
2053 original audit production masks. Every conservative mask is a subset of
current decisions; adaptive masks preserve all outside-band/missing-SZA decisions
exactly. No finite-WIC pixels have missing SZA in evaluated populations.

Independent validation selection and all regional/primary rates reproduce
exactly. Float32 probability boundary, SZA boundary and missing-value cases
pass. A draft threshold-map dtype promotion changed one unchanged-band pixel;
fixed by constructing the baseline Boolean mask then updating only affected
cells. Final invariants pass; earlier unsuffixed artifacts are superseded.
Model/main/audit annotation hashes verified unchanged. Plot inspected.

## Follow-up: actual fitted c(SZA)

[[WIC Fitted SZA Cutoff]] uses the original0.475 criterion to fit actual
monotone linear and PCHIP functions. Both improve validation slightly but
regress on held-out/audit agreement; no policy adopted. The earlier0.65
step comparisons below remain tradeoffs using the different J criterion.

## Prior proposed next decision

If the user accepts trading some manual-kept coverage for fewer manual-rejected
observations, the milder SZA<90° threshold0.65 is a candidate for a controlled
mask-policy comparison. It is not adopted. A later matched CVAE rerun would test
whether this tradeoff helps generated dayside fields; present CVAE outputs have
not changed. The user's accepted data-loss cost and desired policy remain open.
