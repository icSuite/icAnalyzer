# WIC mask propagation benchmark

Last reviewed: 2026-10-12
Status: Full-grid transfer corrected and rerun; earlier restricted runs historical
Repository: vae/wic-si12 at be31406 with existing worktree changes preserved

## Current full-grid correction and rerun (2026-10-12)

User rejects the added old-grid unreviewed area: transfer classes/lines across
the entire46x46 grid. [[Full Grid Annotation Transfer]] owns the verified live
repair. Restricted-domain results below are historical, not intended labels.

Same17612 frames/classes/segments/splits; supervision now covers every finite
WIC pixel. Adds2527409 supervised cells and715656 kept partial cells, retaining
earlier labels. All4841 held-out targets/domains are unchanged. Inputs,
normalization,weights and9333/3438/4841 split match the preceding all-class run.
Best16/24 epochs; validation-selected mask threshold0.65. No class head or time
context added. Output:figures/validation/wic_mask_unet_fullgrid_20261012/.

| Held-out class | Previous restricted run | Corrected full-grid run |
|---|---:|---:|
| Fully accepted |19.3%|29.2%|
| Fully rejected |36.5%|27.3%|
| Partial |23.9%|23.5%|

Values are orbit-average mask disagreement. Class-average error essentially
unchanged; correction restores intended supervision without establishing skill.
Partial gain0.42pp,95% day-cluster interval-1.85 to2.40pp (12 orbits/9 days).
Partial useful area discarded17.5%,rejected area retained31.1%,median correction
22.2%. Retained-fraction proxy59.0% accuracy/55.6% balanced accuracy,validation
thresholds0.01/0.93. Proxy remains ambiguous; not an explicit class head.

Partial/all-class30-sheets,2x2 results,weights,scores/probabilities and corrected
snapshot retained. Grey means nonfinite WIC only.22 tests and independent data/
geometry/split/selection/metric/plot checks pass. Reused test is exploratory;
no automatic mask adoption or human time-saving claim. The later class/time
tests are complete in [[WIC Class and Temporal Mask Benchmark]]; this corrected
headless run remains their baseline. [[WIC Annotation Classifier Plan]] owns
the proposed follow-up.

## Original clarification and first experiment

Each saved keyframe represents every following frame until the next keyframe
within that partial interval. Train on all those frames, not just the frames
where masks were drawn. Never carry a partial mask across the fully-good core
or outside the usable interval. User approved the small U-Net experiment and
30-example sheet. The agent incorrectly inferred approval for GUI navigation
changes; the user asked for an explanation and explicitly did not request a
reversion. Those navigation changes remain; do not describe them as agreed.

This tests mask proposals for already-identified partial frames. It does not
replace full/rejected/partial classification or measure human editing time.
Current shared geometry evaluation preserves original source-point line sides
across the full native grid. Nonfinite WIC pixels are excluded.

## Dataset and independence

Frozen JSON SHA256 matches the prior forest snapshot:
ad947fc3c474c98528b55b2ca92c73683816c3a446a00c89128d32827ba58a68.
8832 completed partial frames on58 orbits/450 mask segments:444 human keyframe
samples and8388 following/synthetic propagation samples. Missing source files
116/155/157/160 excluded;49 source timestamps unmatched within2s;347 partial
frames have no observed supervised pixels. No missing active keyframes.

Dataset builder uses stored propagated_mask and ignores every pixel outside
finite corrected WIC intersect target_annotated_runs. All rows/targets/domains
and human-keyframe provenance were independently checked. Synthetic native
conversion records on166 remain propagation boundaries; they are excluded
only from the human-keyframe evaluation subset, not discarded from propagation.

Inputs are five unmasked precipitation_cs fields:corrected WIC,corrected-WIC
std,high-is-auroral quality,SZA,DZA, plus each field's finite-value indicator.
No H/P,ratio,energy,flux,annotation-mask or reviewed-domain input channel.
Current CS input support still inherits electron retrieval validity.

5313 training frames/33 orbits,1936 validation/13 orbits,1583 test/12 later
orbits. Orbit-start-day groups and orbits are disjoint. Test begins2000-07-09;
last development frames precede it. Per-frame weights equalize orbits and
their mask segments while retaining every propagated frame in each epoch.

There is a substantial domain shift:training contains only131 full-grid
reviewed frames;validation has none. Every test frame has full native reviewed
footprint. Most training/validation supervision is the old restricted footprint.
The result is a later-orbit/full-grid transfer test, not established calibration
on representative full-grid validation data.

## Model and protocol

Small two-level U-Net:30585 parameters,8/16/32 convolution channels,GroupNorm,
SiLU,bilinear upsampling,10 inputs. Asinh WIC,log1p std,angle scaling and robust
normalization fit only unmasked training images. Unknown observations become
zero normalized values with separate finite flags. No arbitrary rotations.

AdamW lr0.001,weight decay0.0001,batch64; masked per-image BCE plus soft Dice.
CPU run stops after13 epochs,restoring validation-selected epoch5. Probability
threshold0.625 minimizes orbit-average validation pixel-correction area.
Test labels are never used to choose epoch,normalization or threshold.

Controls:always retain observed pixels; validation-tuned quality threshold
(selects0,so equivalent to always retain); fixed pixel-position retention prior
learned from training targets only with the same orbit/segment weights,threshold
0.55 selected on validation. The fixed prior sees no image values.

## Results: mean within orbit, then across12 held-out orbits

| Starting mask | Pixels requiring correction |
|---|---:|
| Keep every observed pixel |41.7%|
| Fixed training spatial mask |25.8%|
| U-Net |24.0%|

U-Net discards14.6% of the useful target area and retains26.8% of rejected area.
Mean per-frame correction area25.3%,median22.9%; orbit-average Dice0.768.
The39 held-out human keyframes have25.7% orbit-average correction area;
1544 propagated test frames have24.0%. No evidence of reliable automatic masks.

Gain versus fixed prior is only1.82 percentage points. Paired whole-orbit
bootstrap95% interval is-1.07 to4.85 percentage points; gain is not clearly
established. Within the old footprint, U-Net correction area22.1%; outside it,
34.1% (diagnostic only; primary result remains full reviewed test domain).
Worst cases retain substantial area despite an active all-rejected mask.
The model mainly resembles a common left/right cut and does not consistently
track the user's varying masks. Correction area is not minutes or mouse actions.

## First experiment outcome and GUI scope clarification

No model suggestions were added to the GUI and no live annotations changed.
Model development paused after this first experiment; the user subsequently
requested the all-class revision below. The annotation
GUI now skips completely image-empty orbits lazily on launch/navigation, ignoring
quality/coverage-only values. It does not mark skipped orbits rejected or edit
their review records. --show-empty-orbits enables inspection explicitly.
Next unreviewed jumps over complete/no-usable reviews; Mark orbit complete
advances to the next unfinished nonempty review. Ordinary previous/next still
permit revisiting completed reviews. Mask propagation itself is unchanged.

Eleven focused annotation tests pass, including empty orbit skipping without
automatic rejection,completed-review access and explicit empty-orbit inspection.
Real-source scratch launch skips169/170,opens171 and jumps to173 as next unfinished.
Main annotation hash is unchanged. Desktop interaction was not exercised.

## Revised all-class experiment (2026-10-12)

User requests removal of dwic_corrected, consideration of coverage, and full
and rejected frames as constant all-keep/all-reject targets. Confirmed quality
is an input only: no quality weighting of loss or metrics. Class influence is
equalized, then orbit/segment influence within each class; pixel BCE is equal
inside each observed reviewed domain, with the existing soft Dice term.

Coverage audit across all79 available completed/no-usable orbits (20924 native
frames): all26086020 finite wic_coverage cells equal1 and its support exactly
matches finite corrected WIC. It adds no information to the availability flag.
Generic precipitation coverage depends on finite energy/flux retrievals;
conductance coverage depends on conductance validity. Neither is permitted as
a WIC-only predictor. Coverage is archived for verification but excluded from
the model. Inputs: WIC,quality,SZA,DZA plus four finite indicators;30441 parameters.

Same frozen JSON and exact matched-frame population:17612 nonempty frames
(2109 full,6671 bad,8832 partial), plus3263 archived empty frames (33/2883/347).
Empty pixels provide no mask supervision. Four source files unavailable and49
times unmatched, as before. Every original partial input/target/domain matches
bit-for-bit. Constant full/rejected targets do not add keyframes to live JSON.

Original test start2000-07-09 and validation days retained:9333/3438/4841 frames
on35/13/25 disjoint orbits. Partial splits remain5313/1936/1583. CPU26 epochs,
validation-best epoch18, validation-selected mask threshold0.425. Training-only
normalization and all prediction/metric choices independently reproduced.

| Held-out class | Nonempty frames | Orbit-average mask disagreement |
|---|---:|---:|
| Fully accepted |846|19.3%|
| Fully rejected |2412|36.5%|
| Partial |1583|23.9%|

Partial median disagreement improves22.9% to19.7%, but orbit mean24.0% to23.9%
is essentially unchanged. Paired gain0.16pp;95% orbit-start-day cluster interval
-1.40 to1.68pp (12 orbits/9 days). This combines removal of std with all-class
training and reuses inspected test cases: exploratory comparison, not a fresh
final evaluation or isolated input ablation. Some draft masks match well; no
editing-time saving or reliable automatic application has been established.

Retained-fraction class proxy (validation thresholds0.01/0.91) has56.7% accuracy
and55.9% balanced accuracy. It is not a trained classification head.168/1583
partial test targets are all-keep/all-reject over observed pixels, so even
perfect masks cannot always identify the original frame class from area alone.

Historical label-domain issue: all822 full training and441 full validation
frames retain restricted legacy footprints; all846 full test frames are full
native reviewed grids. This run supervises only known reviewed pixels. User
subsequently rejects this restriction. Full-grid transfer is corrected above;
there is no pending outside-grid label question.

Superseded conversion detail, explained to user2026-10-12: masks were converted
to46x46 arrays by coordinate mapping, not stretched to fill the larger domain.
convert_partial_image_annotations.py computes target_annotated_runs from target
cell centres inside the original36x36 grid extent. In this snapshot inherited
records retain the same41x41 area (1681 of2116 cells,435 outside). The outside
cells are unreviewed and excluded from training loss, including full frames.
This is distinct from per-frame finite WIC support and from drawn rejection.

User queues two further experiments in [[WIC Annotation Classifier Plan]]:
explicit three-class frame output alongside the mask, then a five-frame image
window. This instruction adds items to try; no implementation,training or GUI
changes were made in that checkpoint. The later explicit full-grid correction
is complete and independent of those queued model ideas.

Outputs: figures/validation/wic_mask_unet_allclasses_20261012/; original artifacts
retained. New partial30-sheet, all-class30-sheet,2x2 results, class_evaluation,
split/validation predictions and checkpoint are archived. Sixteen focused tests
pass; independent row/target/grid/source and training/evaluation audits pass.
GUI/live annotation/original checkpoint hashes match pre-experiment guards.
No new GUI edits, model integration or annotation changes in this revision.

## Reproduction and artifacts

README owns prepare_model2_training_data.py and model2_unet.py commands.
All outputs are in figures/validation/wic_mask_unet_20261012/:
frozen JSON,dataset.npz,frames.csv,source fingerprints and exclusions;
model/mask_unet.pt,history,summary,test probabilities,per-frame scores,
mask_comparison_30.png,mask_results.png,spatial-template and transfer controls.
The30-example sheet includes six worst plus24 error-quantile examples,
covering11 orbits/19 segments,two actual human keyframe images. All are held out.

Known-value checks verify output shape,ignored-pixel loss and metric definitions.
Independent review validates propagation,training/validation/test separation,
normalization,input whitelist,weights and every saved test metric. Dataset JSON
hashes match; figures were visually inspected. Scientific conductance models,
source products and original annotation JSON are unchanged.
