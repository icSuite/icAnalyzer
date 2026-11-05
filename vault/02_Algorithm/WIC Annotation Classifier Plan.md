# WIC annotation classifier plan

Last reviewed: 2026-10-13
Status: Class/time and explicit topology-loss tests evaluated; no mask adopted
Repository evidence: vae/wic-si12 at be31406 with existing worktree edits

## Objective and user constraints

Reduce manual review of the remaining IMAGE corpus. Predict three frame classes:
fully accepted, fully rejected, partially accepted. Also propose spatial masks
for partial frames; classification alone cannot remove the line-drawing work.
User initially permits corrected WIC, WIC uncertainty, quality weight,SZA,DZA;
the subsequent mask revision removes dwic_corrected. Quality is an input only,
not a confidence/loss weight. WIC coverage is redundant with availability in
all79 reviewed source files; generic coverage contains forbidden retrieval
information. Accepted/rejected frames now have constant keep-all/reject-all
mask targets, with partial targets held forward as before. User explicitly
rejects the unintended old-grid restriction: all classes/line masks transfer
across the full native grid. [[Full Grid Annotation Transfer]] owns the repair;
prior restricted benchmarks are historical.
Exclude Hall/Pedersen, WIC/SI13 ratio, derived energy/flux, their validity flags,
and downstream retrieval-derived feature selection. User subsequently authorized
the experiments; [[WIC Annotation Classifier Benchmark]] and
[[WIC Mask Propagation Benchmark]] own their results and limitations. The user
is assessing usefulness as editable drafts; no automated masks adopted. Prior
GUI navigation changes were not explicitly authorized, remain after the user's
explanation-only request, and are separate from the model experiment.

## Confirmed starting evidence

- The frozen histogram population contains20875 distinct frames on79 available
  completed/no-usable reviewed orbits:2142 full,9554 rejected,9179 partial.
  Four completed native orbit files are unavailable and49 native timestamps
  are unmatched. Eight draft reviews are excluded. This is a snapshot, not a
  guarantee that the expanding live annotation population is unchanged.
- Current67 completed reviews contain513 mask keyframes:493 line_set and20
  raster-only. Propagated masks repeat these drawings rather than providing
  thousands of independent segmentation labels.
- The earlier snapshot imposed49 completed restricted footprints. This was an
  unintended conversion rule, since removed: full acceptance applies to the
  entire native46x46 grid, subject only to finite observations.
- Annotation classes come from one usable interval and at most one fully-good
  core. Partial masks hold forward only within their partial interval.
- precipitation_cs contains wic_corrected,dwic_corrected,method_quality_weight,
  sza,dza. Local samples confirm uncertainty in counts and angles in degrees.
  Serialized quality provenance is high-is-auroral,1-dgweight, with geometric
  propagation under smoothing. This is an auroral/background diagnostic, not
  an independently calibrated probability of valid scientific data.

## Resolve the input and label contract first

The previous quality-analysis CSV must not be used as ML predictors. It uses
finite paired H/P and accepted manual masks. Both violate the intended feature
boundary. Rebuild one unmasked WIC-based input row for each labelled source frame.
Annotation masks and boundaries are targets only, not predictors; orbit/date
IDs are split metadata, not shortcuts offered to the classifier.

Live icBuilder evidence: precipitationcs.py reduce_frame/reduce uses detector
method_valid for all mean fields except method_quality_weight and for propagated
uncertainties. This includes corrected WIC,SZA,DZA and WIC coverage. Consequently
their current CS missingness embeds retrieval validity. Sample orbit169 has no
finite WIC/SZA/DZA but has428883 finite quality cells. An intrinsic WIC-quality
model needs WIC-native support/geometry and pre-retrieval uncertainty, mapped to
the frozen grid independently of ratio/energy success. Recover source access
before proposing regeneration. No upstream files were changed for this plan.
An exploratory existing-product benchmark must explicitly state its narrower
meaning and must not claim pure WIC quality.

Audit a small representative set of full/partial/rejected transitions and
legacy/native reviews. Identify labels whose rejection depends on other
instruments or retrieval physics that WIC cannot observe. Verify fully-good
semantics. The user's intended classes and drawn masks cover the entire native
grid; do not reintroduce unknown pixels at the original grid boundary.
Empty WIC products are an availability case, not proof the original detector
image is bad. Report those separately from classification on nonempty WIC.

## Bounded experiments in order

### Confirmed experiments

User returned to these queued improvements after the full-grid correction.
Both are now implemented and tested; no model adoption or GUI change follows.
[[WIC Class and Temporal Mask Benchmark]] owns results and limitations.

- [x] Remove unintended old-grid restriction and repair transferred masks.
- [x] Add an explicit three-class frame output alongside the pixel mask.
  Share the image encoder; jointly train accepted/rejected/partial labels and
  mask targets. Compare with the current retained-mask-fraction class proxy.
- [x] Add short image-sequence context: two preceding frames,current frame,
  two following frames, using WIC/quality/SZA/DZA and finite indicators.
  Respect orbit boundaries/time gaps; predict the current frame with the
  existing held-forward keyframe target. First compare stacked frames with
  the single-frame version; ConvLSTM remains a later option.

Confirmed outcome: the single-frame head worsened this mask baseline; five
frames improved rejected recognition, with partial-mask error unchanged and
more useful partial pixels discarded. Class-gating worsened partial masks.

- [x] Five-frame mask-only ablation, authorized 2026-10-13. Match the joint
  model's initial mask weights and shuffled frames; remove class output/loss.
  Partial error25.9% versus23.8% joint and23.5% baseline, rejected error27.7%
  versus15.2% joint. Useful partial area discarded27.8% versus31.8% joint and
  17.5% baseline. No partial benefit or new model adoption established.

- [x] Explicit topology-loss test, authorized 2026-10-13: at most one accepted
  region and no enclosed rejected holes. Keep empty masks allowed; apply the
  prior to full-grid probabilities, never the clipped finite-WIC domain.
  Gradient/target/CPU checks pass. Validation chose weight0.1 over1.0. On
  held-out frames observable shape violations fell598→143, but fully rejected
  pixel error rose27.3%→38.0%. [[WIC Topology Loss Benchmark]] owns details.

Next compare explicit shape projection and its pixel cost if an automatic mask
is needed; decide how fully rejected frames are handled. No VAE run or live
annotation adoption follows automatically.

Keep inputs,training/validation/test groups and evaluation fixed within each
comparison. User resolves the domain: transfer labels/line sides across the
whole46x46 grid. The old41x41 clipping is removed from live annotations and
future conversion; nonfinite observations alone are outside mask supervision.

### Broader proposal and completed earlier stages

1. Freeze the annotation JSON, WIC feature/source evidence and frame matching.
   Use completed/no-usable reviews, one frame target each. Masked and unmasked
   partial versions are the same class, not two examples/classes.
2. Fit a class-balanced Random Forest baseline, compared with a simple majority
   and coverage/quality baseline. Use unmasked WIC finite fraction, intensity
   quantiles and spread, uncertainty quantiles, quality quantiles/fractions,
   SZA/DZA ranges and angular-band statistics. Include coarse spatial tiles,
   local gradients and missing-support geometry to retain spatial information.
   Avoid absolute-scale removal through per-image normalization. Fit imputation
   and any learned preprocessing on training folds only.
3. Add short neighbouring-frame feature windows, changes and time gaps within
   an orbit. Offline annotation permits past and future observations. Never
   cross orbit boundaries or substantial data gaps. Compare single-frame versus
   contextual RF and soft smoothing of predicted class probabilities. Do not
   force every sequence into the GUI's one-core temporal grammar.
4. If held-out errors demonstrate missing spatial recognition, compare a small
   CNN on WIC,uncertainty,quality,SZA,DZA and explicit observed-pixel masks.
   Keep normalization on training data, preserve physical amplitude and fixed
   geography. Arbitrary rotations/flips may be inappropriate for this grid.
   CNN+temporal convolutions/LSTM,3D CNN and ConvLSTM are available later options;
   the number of independent labelled orbits argues against starting there.
5. Build mask proposals to address the largest remaining workload. Start with
   pixel/local-patch Random Forest using WIC and spatial neighbourhood features;
   compare a small U-Net if coherence/boundaries require it. User clarifies:
   train on every frame represented by each active keyframe, using the saved
   propagation rules. Keep whole orbits/segments together and report drawn
   and following frames separately; repeated labels are not independent tests.
   Ignore nonfinite observations. The user's subsequent all-class experiment
   explicitly assigns accepted/rejected frames constant all-keep/all-reject
   targets inside the reviewed observed domain. Evaluate those class targets
   and partial masks separately, including held-out drawn keyframes.
6. Present suggested full/rejected runs, partial masks and uncertainty in a
   review layer. Review transitions, uncertain masks, unfamiliar images and a
   random sample of confident proposals. Confirm whole runs/keyframes instead
   of labelling every image. Add diverse reviewed examples and retrain; retain
   an untouched test set and independent audit of each new deployment population.

## Evaluation and decision gates

Keep whole orbits together for training,tuning/calibration and testing; use
larger day/event groups if adjacent orbits share conditions. Use grouped folds
and a separate chronological holdout. Random frame splits and RF out-of-bag
scores would overstate transfer. All model/feature selection remains inside
the training/validation groups; bootstrap complete orbits for uncertainty.

Report confusion matrix,per-class precision/recall,macro-F1 and balanced
accuracy, plus error versus fraction of decisions that avoid review. Report
false full-accept decisions and lost usable frames under automatic rejection
separately. Evaluate boundary timing,mask overlap,false retained pixels and
lost useful area. Break results down by orbit,footprint,coverage,geometry and
nonempty versus unavailable data. RF probabilities require calibration on
held-out groups before using them as confidence. A proposed starting target is
99% full-accept precision for decisions made automatically, with grouped
uncertainty reported; this is not achieved performance or a user-approved rule.

The decisive outcome is verified manual effort saved at an acceptable error
rate, including partial-mask correction time. Measure review time on a small
representative batch of previously unreviewed orbits, not only accuracy on
historically selected data. No promise of full automation or a fixed saving.

First concrete deliverable: audited WIC-only feature/label dataset, grouped
RF versus temporal-context benchmark, example errors and a review-workload
curve. Choose CNN/mask-model expansion from that evidence. Do not turn model
proposals into completed human annotations automatically.

## Primary method references

- [Random Forest classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Grouped stratified splits](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html)
- [Probability calibration](https://scikit-learn.org/stable/modules/calibration.html)
- [ConvLSTM](https://arxiv.org/abs/1506.04214)
- [U-Net](https://arxiv.org/abs/1505.04597)
- [Active learning survey](https://burrsettles.com/pub/settles.activelearning.pdf)
