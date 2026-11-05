# WIC held-out probability maps and fitted SZA cutoff

Last reviewed: 2026-10-14  
Status: Verified exploratory diagnostic; neither fitted cutoff adopted.

## Question and interpretation

User requests current U-net held-out WIC/mask/probability examples and an
optimized monotone decreasing c(SZA). Frozen production checkpoint remains
`e5c009e6713b...`, four physical inputs, epoch15, cutoff0.475. Sigmoid outputs
are model acceptance scores, not calibrated probabilities of physical quality.
The user's manual labels encode conservative line-based selection.

Both fitted curves slightly improve validation agreement but regress on the
previously held-out data and the eight-orbit manual audit. No production
threshold/model/masks, main/audit annotations, cache or CVAE outputs changed.
Recommendation: retain current cutoff pending a different justified approach;
this is not a user decision to reject all future SZA policies.

## Original objective recovered and used

Original0.475 was selected on33 cutoffs0.1–0.9 to minimize
`model2_unet.validation_correction`: finite-WIC pixel disagreement
per frame, averaged over frames within each semantic class/orbit, then equally
over orbits within class, then equally over full/bad/partial classes.

The prior [[WIC SZA Threshold Tradeoff]] selected0.65 using the later
four-component J, which is a different objective. It must not be described as
the criterion originally selecting0.475. Current fresh CPU inference on3438
validation frames reproduces the original grid winner0.475 (10.60517%).

Per-pixel weight is 1/(number of classes × orbits in class × valid frames in
that class/orbit × finite-WIC pixels in frame). Weighted raw errors reproduce
the original function exactly. Fit uses only validation scores/labels/SZA;
choices are saved before test/audit evaluation. These sets already informed
preceding work, so subsequent results are exploratory, not new untouched tests.

## Fits and results

Bounds are0.1–0.9. Optimized constant found exactly over raw probability
decision boundaries:0.4799267948. Linear c is clipped to bounds:

`c(SZA) = clip(0.2766377971 - 0.0087699170 × (SZA - 90°), 0.1, 0.9)`.

Monotone PCHIP spline at fixed SZA60/90/105/120° has values
0.5958310182/0.2900262103/0.2900165535/0.1231563172 and constant extrapolation
beyond end knots. Both searches use two seeded differential-evolution runs on
0.1°/0.0005-score histograms, candidate rescoring and bounded Powell refinement
on individual raw pixels. Reported objectives are exact; numerical optimum is
approximate. Linear refinement converged38 calls; spline hit100-call budget.
Best preceding candidate retained when refinement worsened its score.

| Policy | Validation disagreement | Old held-out disagreement | Audit disagreement |
| --- | ---: | ---: | ---: |
| Current0.475 | 10.605% | 29.132% | 25.092% |
| Optimized constant0.479927 | 10.599% | 29.123% | 25.107% |
| Fitted linear | 10.393% | 30.274% | 25.821% |
| Fitted monotone spline | 10.421% | 30.022% | 25.755% |

Old splits use semantic labels; audit uses effective mask classes to handle
partial-label blocks whose geometric mask is entirely kept or rejected.
The selected linear fit is stricter than0.475 only below~67.4° SZA and more
permissive above. c(60/75/90/105/120°)=0.540/0.408/0.277/0.145/0.100.
On audit MLT dayside (06≤MLT<18, MLAT≥60, finite WIC), manually rejected
retention rises19.160→20.119%, while manual-kept loss falls34.920→30.660%.
This objective therefore does not learn blanket conservative dayside masking.
SZA and the MLT dayside definition are distinct. Physical noise not measured.

## Probability examples

Nine old-test frames selected with seed20261014 before reading predictions,
three distinct orbits per manual class. Page columns: corrected WIC counts
(shared asinh display, width100 counts), manual mask, current p≥0.475 mask,
raw p on fixed0–1 scale. Noon at top; midnight bottom. Gray WIC/mask pixels
are unavailable WIC. Raw scores also shown outside finite-WIC support;
their apparent acceptance does not mean those observations are retained.
WIC contours mark SZA90°/105°; score contour marks0.475.

Partial examples195:87/210:43/206:240; rejected177:245/185:162/199:109;
accepted187:245/188:42/186:70 (orbit:native frame). Selected partial maps
show broad smooth regions, not detailed bright-structure outlines; this is
visual evidence from these examples, not a quantified model diagnosis.

### Expanded browsing set (user follow-up)

User requests more examples before interpreting patterns. Now90 held-out
examples,30 per semantic class, including the original nine. Additional frames
selected randomly before reading predictions, balancing counts over all
available class-specific held-out orbits:12 partial/20 rejected/14 accepted.
Prefer≥18 native-frame separation within orbit where possible. These are a
class/orbit-spread browsing sample, not a population-frequency estimate.

`figures/validation/wic_probability_examples_20261014_expanded/` contains
30 PNG pages, three10-page class PDFs and one30-page combined PDF. Original
nine-example folder preserved. Three examples/four columns per page; identical
WIC/asinh display limit5086.3900586 counts and probability0–1 scale. All90
native timestamps/SZA/domains and saved production masks verified. CPU
re-inference max absolute score difference1.1920929e-7 (batch-size float32
rounding), with every threshold decision unchanged; guard permits two float32
epsilons and requires exact decision equality. Sample pages from all classes
visually inspected; three class PDFs opened. No new scientific pattern claimed.

Reproduce via `plot_current_probabilities.py --per-class 30` with
`--include-selection figures/validation/wic_probability_examples_20261014/selection_before_predictions.json`,
`--wic-vmax 5086.3900585937545`, and a chosen fresh `--output` path.

User follow-up adds Hall/Pedersen to the same90 examples;
[[WIC Conductance Browsing Diagnostic]] owns CVAE input verification, new
six-column figures, inclusion checks and user interpretation.

## Evidence and reproducibility

- `scripts/automatic_masking/plot_current_probabilities.py` and
  `figures/validation/wic_probability_examples_20261014/`: selection/provenance,
  three PNG/PDF pages, native frame/time/domain/SZA guards, re-inference within
  1e-7 and exact saved production mask agreement. All pages inspected/opened.
- `scripts/automatic_masking/wic_sza_cutoff.py` and `fit_sza_cutoff.py`;
  `figures/validation/wic_fitted_sza_cutoff_20261014/`: protocol, validation
  score cache with IDs/raw SZA, frozen choices, optimization trace, comparison,
  report and inspected/opened curve/tradeoff PNG/PDF. Incomplete sibling folder
  is superseded (old SciPy result API; corrected via public callback).
- Twelve focused tests pass: objective parity including empty-domain frames,
  monotonicity/bounds, knot/extrapolation behavior, float32 inclusive0.475,
  missing-SZA fallback and exact constant fit versus exhaustive decisions.
- Independent review exactly reproduces all12 population/policy disagreements
  and all regional percentages. Audit targets reconstructed from native WIC
  and annotation geometry; raw audit SZA matches cache byte-for-byte.
- Checkpoint, annotations, split/frame-table/cache hashes, source stat/grid/time,
  normalization and original/audit orbit disjointness verified. Missing SZA
  uses0.475; no missing SZA exists in evaluated finite-WIC domains. Float32
  cutoffs preserve production threshold boundary semantics.
