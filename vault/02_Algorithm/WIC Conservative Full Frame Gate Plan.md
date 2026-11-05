# WIC conservative fully accepted frame gate

Last reviewed: 2026-10-15  
Status: User-approved two-stage dataset trial exported and valid5 cache ready;
CVAE training pending. Earlier classification benchmarks remain exploratory.

## User-approved two-stage application (2026-10-15)

Launch repair: primary user entry is now
`python scripts/IMAGE/step_1_two_stage_cvae.py` with visible #%% settings and
CLI overrides; old Bash file only forwards to it using active Python. Generic
trainer validates CUDA availability and a real allocation before cache/output
writes. User's failed initialization(no batches, only cache manifest) archived
without deleting data; intended output is clear. NVIDIA module was unloaded
despite installed kernel-matching595.99.02 driver; user load restored nvidia-smi
and outside-sandbox PyTorch CUDA on RTX3500 Ada. No proven update/version clash.
7 loader/device tests pass; config/overrides checked with mocked training.
No long CVAE run launched by agent. No scientific recipe/data change.

After visual review, user wants to try Model1+Model2 despite semantic Bad false
acceptances. Use latest corrected-target acceptance CNN(seed12027,epoch4) and
temporal veto(seed12029,epoch4), frozen Accept>=0.8046223521/Bad<0.0415420271.
All other frames retain Model2's existing U-net0.475 masks, with no whole-frame
forced rejection or retraining of Model2. No threshold retuning or annotation
overrides during inference. Empty WIC frames cannot trigger a full release.
Exporter skips veto evaluation below acceptance cutoff, preserving routing.

`apply_two_stage_masks.py` processed1,378 orbits/358,439 frames in172s;5,163
full releases(1.70% of303,731 nonempty WIC frames). Distinct masks under
`data/training_data/two_stage_masks_20261015/` retain times and scalar scores/
route flags. Output manifest's model_sha256 identifies policy.json containing
both Model1 checkpoint hashes/cutoffs and original Model2 manifest/hash. It
does not masquerade as a single U-net weight identity. Every old kept pixel is
preserved; missing native observations still excluded by intrinsic validity.

Separate complete `data/training_data/two_stage_cvae_valid5_20261015/` uses the
confirmed5%-minimum intrinsic H/P coverage, same source orbit splits as prior
CVAE. Eligible positive frames186342 train/37703 validation/38050 test. At a
fixed5% frame rule, current→combined positive-weight H/P observations:
training day43168955→43236975(+68020,0.158%), near noon4963104→4999374
(+36270,0.731%); all day61069610→61165388(+95778,0.157%), near noon7198989→
7248110(+49121,0.682%). DayMLT06–18/MLAT>=60; noon10–14/60–75. Each common
H/P pixel/frame counted once. Only3 extra eligible-positive frames from masks;
large frame-count growth versus old CVAE is mainly relaxed coverage, not gate.
Mask-only gains are small; new CVAE response not yet measured.

`plot_two_stage_coverage.py` and `figures/validation/wic_two_stage_application_20261015/`
own linear same-scale coverage maps(all/train), raw counts/report/verification
and ready CUDA launch script matching actual old500epoch,batch264,beta0.048,
latent36,30-controls,holes,seed1 recipe. New output is
`figures/cvae_runs/conductance_two_stage_valid5_20261015/`; training not started,
consistent with user-run CVAE workflow.15 focused tests pass; all serialized
mask routes/fallback/times/hashes checked, baseline masks match old-cache hashes,
smoke scores reproduce frozen predictions, all CVAE splits load. Independent
native-source comparisons on1407/1392/1409 reproduce values/validity/eligibility
and old/new count maps. Existing source products, masks, annotations, cache and
CVAE preserved. User approval authorizes this trial; no Bad relabels invented.

## Corrected acceptance-target retraining (2026-10-15)

User authorized retraining and regenerated plots. New runner
`train_model1_cnn.py` freshly fits acceptance(seed12027) and temporal
Bad veto(12029) with confirmed effective labels, promoting113 training/2
validation Partials. Labels also determine class/orbit/segment balance; segment
keys split by effective label. Same architectures/seeds/splits/normalizer/inputs
as preceding fit. Acceptance class influence50/25/25%; veto16.7/66.7/16.7% for
Accepted/Bad/Partial. Both minimum-validation-BCE checkpoints epoch4.

Validation-only tuning retains prior policy: highest acceptance cutoff reaching
80% corrected Accepted recall before veto, plus veto catching99% validation
Bad. Cutoffs0.804622/0.041542 frozen before evaluation. Combined validation
181 Accepted/101 Partial/0 Bad(40.9% Accepted recall). Held-out65/11/15 versus
42/12/20, audit0/11/9 versus41/121/20, for Accepted/Partial/Bad. Accepted totals
890/313. No promoted Partial released in test/audit. Acceptance head alone
passes202/64 good test/audit frames; veto reduces these to65/0. Added day WIC
good/rejected67/168 test and0/137 audit. Bad audit passes remain orbit838.
Correct labels have not rescued this gate; no adoption or physical-noise claim.
Old test/audit previously inspected, so result remains exploratory.

Fresh `figures/validation/wic_model1_accepted_partials_20261015/` owns report,
protocol, targets, cutoffs, new checkpoints/histories/scores and four12-example
PDFs/PNG pages. Correctly accepted book12 test-only, other books6+6. Model1
Accept/Bad scalars and Model2 pixel probability/mask labeled.14 tests include
finite/unobserved/empty-domain label handling. Effective weights, neighbor/split
orbits, exact normalized center/source identities checked. Independent checkpoint
reload matches8 held-out scores; archived-count and figure-score hash checks
pass. U-net0.475/model2/masks/source annotations/cache/CVAE unchanged. Next:
user inspect new outcomes, especially veto failures, before further changes.

## Confirmed Model 1 acceptance semantics (2026-10-15)

User confirms: manual Partial images whose masks accept all pixels count as
Accepted for Model1. Figure reference uses every observed finite WIC pixel
accepted, with saved annotation labels retained. Future Model1 training should
use that same effective acceptance target. The initial figure clarification
preserved old scores; the completed retraining above now implements the rule.
This supersedes the prior pending label-semantics decision.
Updated example books label these rows Accepted (saved label Partial), and
missed-Accepted counts become848 test/272 audit; correct releases remain42/41.
Pixel probability and thresholded U-net mask columns are Model2; row scalar
Full/Bad scores comprise Model1, final column combines its release with U-net
fallback. User says the first two reviewed accepted-Bad examples could be
Partial and the last is Bad; preserve as visual feedback without assigning
new dataset labels or partial masks. All source annotations remain intact.

## Temporal Bad-veto trial (2026-10-15)

User authorized the next idea: a dedicated Bad veto using short native temporal
context before the separately trained Full decision. `try_temporal_bad_veto.py`
trained a five-frame45-channel CNN on the same orbit-disjoint train/validation
split, weighting Bad at two-thirds of class influence. It used the frozen
seed12027 Full classifier at the prior0.7063834 diagnostic cutoff, sending all
other frames to the existing U-net. Native-neighbor source and center input
identities were checked; five-frame windows never cross orbit/gaps>300s.

Validation selected a veto cutoff for99% Bad recall *before* held-out/audit
evaluation. Passed Full/Partial/Bad was279/172/0 on validation,42/12/20 old
held-out,41/121/20 audit. Thus the veto cuts held-out Bad releases114→20 but
also Full215→42; all20 audit Bad releases are orbit838, while it catches the
previously troublesome orbit1016 sequence. Newly released broad-dayside WIC
good/rejected counts are148/441 held-out and8242/19433 audit. A stricter veto
with100% validation Bad recall still passes2/20 Bad held-out/audit, so none of
the prespecified rules meets the zero-Bad goal. The trials remain unadopted;
production U-net masks/cache/CVAE untouched. Manual labels are policy, not
independent proof of physical noise. Old held-out/audit sets have prior
inspection exposure. `figures/validation/wic_temporal_bad_veto_20261015/report.md`
and saved scores/protocol own full evidence.26 focused tests pass and an
independent archive recount confirms class counts.

Follow-up visual review: `figures/validation/wic_temporal_bad_veto_20261015/examples/`
has four 12-example PDFs (six test, six audit each): correct Full releases,
rejected Full, accepted Bad, and semantic Partial with every finite WIC pixel
manually accepted. Each row aligns native WIC/manual mask/U-net score+mask/trial
routing and prints Full/Bad scores. Rejected Full counts are804 oldtest and98
audit; among oldtest631 miss Full cutoff and173 trigger Bad veto, while all98
audit failures miss Full cutoff. Accepted Bad is20 each; all audit in orbit838.

Manual-label/pixel-target consistency check found158/8832 original semantic
Partial frames whose manual mask accepts the entire46×46 grid (113 train,1
validation,44 test); one additional validation Partial accepts every observed
WIC pixel but rejects an unobserved grid cell. Audit has174/1178 all-grid-
accepted Partials. Thus the frame classifier's semantic Partial-negative target
conflicts with an all-accepted pixel target in these cases. This may affect
learning/calibration, particularly under audit shift. The confirmed Model1
acceptance rule above now resolves the intended target. Crucially, none of12
held-out or121 audit Partial false releases is in the all-observed-manually-
accepted group, so this label ambiguity does not explain those specific errors.
Current U-net alone fully retains observed WIC in14 test and106 audit Partial
frames; this is a separate prediction count. Exact counts and plotted identities
are saved in `examples/summary.json`; source/timestamp/mask/probability
alignment assertions pass. Existing masks and annotations unchanged.

## User hypothesis and proposed architecture

User interprets the90-frame sheets as dayside coverage primarily supplied by
fully accepted images, with the U-net tending to reject observed dayside pixels
and showing high dayside scores mainly outside the observed footprint. This
is a hypothesis from visual review, not a verified full-corpus class attribution
or proof of the physical source of generated CVAE noise. Prior browsing/cache
facts remain in [[WIC Conductance Browsing Diagnostic]].

User proposes either stronger dayside acceptance through class/loss treatment,
or a two-model pipeline: conservative whole-frame Accept/Reject (partial is
negative), then U-net extraction on frames not fully accepted. The user
authorized both the reference measurement and a first separate-classifier trial.

## Concrete interpretation

1. Train a separate binary frame classifier using manual full versus bad+partial
   labels, interpreted as fully acceptable versus requiring pixel selection.
   Release all AVAILABLE valid pixels only when its full-accept score exceeds a
   threshold chosen on validation for high precision of fully accepted frames.
   Missing pixels remain missing; conductance validity/quality still applies.
2. All other frames pass through the current frozen U-net at0.475. This includes
   genuine rejected, partial and uncertain fully accepted images. Negative first
   stage does not force whole-frame rejection. Start by routing at inference,
   without retraining model2 on a newly selected population.

The irreversible classification error within this composition is a wrong
full-frame release: model2 cannot repair it. Tune full-accept precision, useful
dayside recovery and false-kept dayside pixels, not overall frame accuracy.
High numeric scores do not independently guarantee calibrated physical quality.
If retraining model2 only on non-full targets is later considered, evaluate
separately: removing full examples can reduce its positive dayside supervision.

## Evidence and distinctions

Live training code `sample_weights` equalizes class/orbit/annotation-segment
influence. Recalculation on frozen original9333 training frames gives full822,
bad3198, partial5313 but total weight exactly1/3 each. Thus raw class-count
imbalance is not the whole explanation; within-class/day pixel targets,
SZA-dependent sampling and generalization can still matter. This recomputation
checks the current training recipe, not a new model fit.

[[WIC Class and Temporal Mask Benchmark]] tested a SHARED three-class head
jointly with masks, including fixed argmax gating: full keeps all, bad rejects
all, partial gets its predicted mask. Single-frame head did not improve that
older baseline. The independent binary high-confidence acceptance gate with
U-net fallback for EVERY other frame is evaluated below. Temporal scopes and
older baseline differ from frozen production topology-weight0.1 model.

## Reference measurement complete (2026-10-14)

Implemented `scripts/automatic_masking/measure_full_frame_gate.py`, writing
fresh `figures/validation/wic_full_frame_gate_20261014/`. Frozen production
0.475 scores; original test4841 frames/25 orbits (846 semanticfull), audit2053
frames/8 orbits (139 semanticfull). Audit effectivefull313 was deliberately
not used. Only semanticfull releases all intrinsic-valid observations; every
other frame mask exactly unchanged. Native conductance timestamps/grids/source
fingerprints and existing cache validity/values/eligibility verified.

Primary comparison fixes5% intrinsic valid frame rule on BOTH sides. Current
and gated eligible frame counts identical:3892 test,1460 audit; gain is more
pixels in those frames, not more frames. H/P counts below are common-valid
pixel/frame observations (not doubled across the two channels).

| Population | Region | Extra H/P observations | Relative H/P gain |
| --- | --- | ---: | ---: |
| Original held-out | Dayside | 84590 | 10.75% |
| Original held-out | Near noon | 47793 | 78.87% |
| Eight-orbit audit | Dayside | 11301 | 3.45% |
| Eight-orbit audit | Near noon | 5556 | 14.93% |

Dayside is06≤MLT<18, MLAT≥60; near noon10≤MLT<14,60≤MLAT<75. These are
MLT regions, not SZA cuts. Additional quality-weight sums increase9.36%/84.12%
for test day/noon and10.62%/21.35% audit day/noon, so additions carry actual
positive training weight. Every added pixel in the reported regions has positive quality; none is manually rejected within finite WIC or outside finite
WIC in measured regions. Nightside recovery is small:7607 test(+0.54%),1162
audit(+0.24%). Perfect manual labels explain zero extra false keeps; a learned
frame gate can still release incorrect full frames.

Manual-good available WIC retained percentages improve day71.17→85.29% test,
65.08→68.34% audit; noon30.76→73.38% test,37.17→46.96% audit. Original
class/orbit-balanced disagreement falls29.132→23.831% test and24.865→19.805%
audit using SEMANTIC classes; the audit baseline differs from prior effective-
class25.092% comparisons. This is exploratory manual-policy potential, not
learned classifier skill, measured physical noise or predicted full-corpus gain.

Old67%-valid sensitivity holds that older cut fixed within each pair:
dayside relative gains14.96% test/5.52% audit; noon239.65%/29.95% (low old
baseline coverage). Frame counts2440/953 remain unchanged. Do not combine
coverage-cut changes with gate changes when attributing improvements.

Class attribution also checked: current retained broad-day WIC pixels from
semanticfull are32.3% test and15.4% audit; near-noon49.2%/25.4%. User's
“mainly fully accepted” hypothesis is therefore not established universally
by these labels/populations. A useful gate benefit exists despite that.

Independent read-only review reconstructed WIC targets and semantic labels,
source-based H/P masks and frame rules, exactly reproduced every regional
conductance count under both cuts and original disagreement, and checked all
model/annotation/frame/split/probability fingerprints. Non-full and subset
invariants pass. Figure inspected/opened; report, protocol, source evidence,
comparison JSON and per-cell added-coverage NPZs preserve results. Syntax passes.
No new tests mirroring the simple gate were added; independent raw-source
recalculation is the substantive verification.

## Model1 protocol and completed first trial (2026-10-14)

The user requested a concrete plan, then authorized implementation. The
bounded first-trial protocol was:

1. Standalone binary CNN trained separately from scratch. Reuse familiar
   convolution blocks/encoder widths8→16→32; retain spatial information with
   pooled2×2 feature tiles and one binary output. No shared pixel-mask objective.
   Inputs exactly production WIC, quality, SZA, DZA plus finite flags, with frozen
   training-only normalization. Frame score estimates manual full acceptability
   of available observations; it is not calibrated physical pixel quality.
2. Semantic manual full is positive; bad+partial are negative (needs masking).
   Frozen original split:train9333 (822 positive across22 orbits), validation3438
   (441 positive across10 orbits), test4841 (846 positive across14 orbits).
   Audit2053/139 positive remains outside fitting. Negative labels are routing
   labels and do not force whole-frame rejection by the final pipeline.
3. Binary BCE with equal positive/negative total weight; balance bad and partial
   within negatives and retain orbit/annotation-segment balancing. Existing
   `sample_weights` can be scaled full×1.5, bad/partial×0.75 to obtain50/25/25%
   influence. Early stopping on validation binary loss; repeat three fixed seeds
   to check whether high-confidence decisions and coverage recovery are stable.
   Single-frame first trial; new temporal context/input sweeps would be separate.
4. Examine validation full-accept precision versus recovered useful day/noon
   H/P coverage. Proposed initial target99% empirical full-accept precision,
   not a claim that sigmoid0.99 is calibrated or that true precision is99%.
   Require accepted decisions across multiple orbits, report frame/orbit support
   and orbit-group uncertainty; do not promote a threshold passing only a few
   correlated frames. Choose final rule on validation, before test/audit. If no
   useful conservative threshold exists, report that instead of relaxing the
   acceptance criterion silently.99% target is proposed, not user-confirmed
   contamination tolerance or production policy.
5. High-confidence full releases all intrinsic-valid observations; every other
   frame uses frozen current U-net0.475. Compare at fixed5% intrinsic valid
   frame cutoff, with original67% sensitivity if useful. No threshold retuning
   on old test/audit; these already inspected sets are exploratory. Record
   high-confidence false full releases, accepted-full recall and actual
   pipeline newly retained manual-good/manual-rejected day/noon pixels, including
   quality-weight contribution. Compare current pipeline, learned gate and
   manual-full reference. Frame accuracy alone is not the scientific criterion.
6. Review identical WIC/manual/U-net/score/H/P examples, add model1 score/routing
   labels and examples of every false full release. Keep CVAE fixed during the
   classifier trial. A subsequent full-mask/cache/CVAE comparison would be a
   separate concrete model-adoption/retraining decision, after routing quality
   and useful recovery are demonstrated.

Success means recovering a meaningful portion of the measured near-noon
reference benefit while adding few manually rejected observations. It does
not establish physical noise removal: manual rejected frames may be usable.
Classifier1 may fail to generalize because full frames are represented by
only22 training orbits, despite822 correlated images. Previous shared3-class
head findings motivate separate supervised training and explicit routing
precision, not an assumption that the binary classifier will succeed.

## Standalone classifier result (2026-10-14)

The three-seed separate binary CNN trial is complete in
`figures/validation/wic_full_frame_classifier_20261014/`. Code is
`model1_cnn.py`, `train_full_frame_gate.py`, and
`plot_frame_gate_examples.py` in `scripts/automatic_masking/`. Training used
the original train/validation orbit split, frozen production inputs and
normalization,50/25/25% full/bad/partial influence, and validation-loss early
stopping. Best epochs were1/1/3. Selection was frozen before opening the old
held-out and audit sets. The existing U-net remained at0.475.

No seed met the predeclared99% empirical manual-full precision with at least50
passed validation frames across3 orbits AND positive-weight near-noon H/P
recovery. The best precision with that frame/orbit support was84.31%,86.57%,
82.02% for seeds12026/12027/12028, each recovering zero targeted pixels. At
99% precision, at most11/16/0 validation frames passed, again with zero
targeted recovery. Validation reference benefit was1196 native H/P near-noon
pixels in36 semantic-full frames across4 orbits, versus47793 on the old held-out
set. Limited useful validation support and worsening validation BCE after early
epochs constrain interpretation; neither proves the cause of failure.

Diagnostic seed12027 had nine high-score semantic Partial frames, all from
orbit106. Their manual WIC accepted fractions were44–72%; the U-net already
kept every finite WIC pixel in those nine, so their wrong frame labels would
have added no observed WIC pixels. An exact validation-cutoff pixel check
addressed this nuance: among cutoffs passing50-frame/3-orbit support, every
cutoff recovering any positive-weight near-noon H/P also newly kept manually
rejected dayside WIC. At the first useful cutoff the three seeds recovered
only2/2/8 H/P pixels while adding19/57/149 manually rejected broad-dayside
WIC pixels (18/49/44 near noon). Manual differences are not a measured physical-
noise rate. See `validation_pixel_tradeoff.json`, `failure_diagnostics.json`,
the report, score archives and reviewed example PDFs.

The chosen threshold is `null`: every frame still uses the frozen U-net. On
4841 old held-out and2053 audit frames at fixed5% intrinsic-valid coverage,
there are zero additional H/P pixels, zero false whole-frame releases and exact
identity of final masks, frame eligibility and quality sums. Independent review
reproduced all three no-feasible decisions, native validation gain, score
samples and held-out identity.13 focused tests pass. This is an exploratory
old-test/audit inspection, not a fresh generalization claim. The manual-full
reference remains potential with perfect labels, not achieved classifier gain.

The user's clarified priority is to avoid sending manually Bad images to the
fully accepted pile; missing some Full images is acceptable. The previous
82–87% figure was precision, not Full recall, and the supported validation
errors behind that figure were Partial rather than Bad. The original binary
release target deliberately treated both Partial and Bad as negative because
Partial annotations reject some pixels. These two error types must be reported
separately.

Post hoc, the seed12027 cutoff0.7063834 is the highest validation score cutoff
that captures at least80% of Full validation frames. It would release
353 Full/187 Partial/0 Bad validation frames, but215 Full/23 Partial/114 Bad
old held-out frames and41 Full/212 Partial/34 Bad audit frames. Full recall
falls from80.0% to25.4%/29.5% on those two inspected sets. Zero Bad on
validation therefore fails the user's requirement when transferred to other
orbits. This alternate criterion was chosen after the initial experiment and
the old held-out/audit sets had already been inspected; it is exploratory, not
a calibrated deployable threshold. Exact counts and score-file hashes are in
`posthoc_80_percent_recall.json`. The no-release selected rule is unchanged.

The user then asked whether a higher cutoff or another model change helps.
An exhaustive score inspection shows seed12027 maximum Bad scores0.5813
validation,0.9063 old held-out and0.9105 audit. A cutoff just above the
largest **inspected** Bad score releases zero Bad in those sets but retains only
8/846 Full held-out and14/139 Full audit frames; it also releases32 Partial
audit frames. Relative to the frozen U-net, that cutoff adds zero broad-day
WIC pixels on held-out and only155 manual-good versus1246 manually rejected
broad-day WIC pixels on audit. It is a post hoc bound using evaluation labels,
not a future safety guarantee. At the validation80%-Full cutoff, the added
broad-day WIC counts are262 good/1217 rejected held-out and12215/29969 audit.
The `posthoc_cutoff_sweep.pdf` and JSON summary preserve the full tradeoff.

Requiring the minimum of all three separately trained scores at80% validation
Full recall still releases72 Bad old held-out and10 Bad audit frames. This
low-cost ensemble check does not repair the new-orbit failure. The highest-
score Bad example books under `examples/` show two patterns: narrow-footprint
old held-out frames from orbits191/201, often already nearly all kept by the
U-net; and a long broad-coverage Bad sequence in audit orbit1016 where the
U-net masks much of the observed image. These patterns motivate checking the
manual labels and training a dedicated Bad veto with hard Bad examples and
short temporal context. Visual appearance alone does not justify relabeling.

## Next proposed step

Keep the current U-net routing. The next controlled trial should check the
high-score Bad labels, then train a dedicated Bad veto alongside a separate
Full-release decision. Explicitly penalize Bad releases, include short native
frame context without forced smoothing across sudden changes, and send every
uncertain/Partial frame to the frozen U-net. Judge zero observed Bad releases
and useful dayside recovery across held-out orbits, with Partial releases and
newly retained rejected pixels reported separately. Current old test/audit
orbits are development evidence, not a fresh final test. Neither a higher
cutoff nor simple score consensus delivers useful safe coverage here. No
production mask/model/cache/CVAE change was made. Confirmed5%-minimum-valid
policy remains a preparation default; the existing CVAE still uses the old
33%-missing cache.
