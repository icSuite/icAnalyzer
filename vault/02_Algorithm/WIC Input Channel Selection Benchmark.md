# WIC input-channel selection benchmark

Last reviewed: 2026-10-14  
Status: Completed; retain the current four inputs, no model adoption

## Outcome

Eight channel sets and three-seed confirmation of the two best sets plus CORE
produced 14 completed training runs (16–27 epochs each, no 40-epoch cap hits).
The numerical winner is the broad 33-map input without SZA, but it does not
meet the predeclared improvement rule. Three-seed validation J is 13.87% versus
14.50% for CORE: a 0.63 pp gain, one seed worsens, and the paired 95% interval
[-2.14,+0.48]pp includes zero. Removing derived products averages 14.38%.

After freezing that selection, the exploratory previously inspected test has
J 26.89% versus 27.06%, difference -0.17pp with interval [-1.99,+1.52]pp.
The broad model trades better rejection for more useful-pixel loss:

| Test endpoint, three-seed orbit means | CORE | Broad without SZA |
| --- | ---: | ---: |
| Fully accepted pixels lost |22.65% |25.50% |
| Fully rejected pixels retained |42.87% |36.81% |
| Useful partial pixels lost |19.45% |24.20% |
| Rejected partial pixels retained |23.26% |21.04% |

Retain `wic_corrected`, `method_quality_weight`, `sza`, `dza` plus finite flags.
The broad-model DZA/SZA ablations do not establish whether either should be
removed from CORE. No meaningful overall input-choice gain is demonstrated
within the tested architecture/field matrix. Test acceptance fraction falls
46.08→43.99%; observable shape violations fall 220→192.7 frames, while useful
partial-pixel loss worsens. This experiment does not block the accepted
automatic-mask VAE/CVAE workflow and does not measure annotation-time savings.

Artifacts: `report.md`, `comparison.json`, `ranking.csv`, `comparison.png`,
the same fixed-nine-frame `same_frame_examples.png`, `learning_curves.png`,
failure-case list, thresholds/history/predictions/checkpoints per run, inventory,
protocol/splits/fingerprints, inference dry run and reproduction code/commands
under `figures/validation/wic_input_selection_20261014/`.

## Frozen protocol and population

[[WIC Input Channel Selection Plan]] owns the predeclared search and decision
rule. Artifacts live in `figures/validation/wic_input_selection_20261014/`.
The corrected manual JSON hash remains
`c272e0b57c36e020ebeede12aa0cfbfb28fd06eaaf6adda10b9cf70db5331895`;
the preserved production-checkpoint hash is in `protocol.json`.
The 17,612 reviewed nonempty frames have exactly the reference
9,333/3,438/4,841 train/validation/test frames. Frame identities, classes,
segments, domains, source times and baseline normalization match the saved
manual dataset/reference split. Partial keyframes remain held forward.
Automatic masks are never training labels.

## Source audit

All 1,378 precipitation-CS files / 358,439 frames were inventoried one orbit
at a time: 32 time-dependent image maps and four static grid maps. Three fields
are excluded because their finite training values are constant: WIC coverage=1,
Hardy dEp=0, and varE0Fe=0. Non-image time/index/scalar/edge fields are excluded
by the plan. No Hall/Pedersen map is present; no sibling products are imported.

The broad candidate uses 34 value maps, with static MLT expanded to sine/cosine,
plus 34 finite flags. The additional-WIC-signal group is empty, leaving eight
screen sets: CORE, ALL, ALL minus each WIC-support/geometry/other-FUV/derived
group, and ALL minus DZA or SZA. The source manifest, availability/finite
fractions, units, missing conventions and upstream evidence are saved.
WIC/SI/geometry means already inherit upstream retrieval-valid support;
method quality is defined independently. These are upstream supports, not
human annotations. Evidence: icBuilder `precipitationcs.py:201/224/233` and
`precipitationdetector.py:264/644/666/685/714`, recorded in the inventory.

## Controls and execution

Two separate chronological scripts under `scripts/automatic_masking/`
prepare memory-mapped inputs and run screen/confirmation/report. Existing
user training/annotation scripts are unchanged. SmallUNet, BCE/Dice, GUDHI
persistence pairs and topology weight0.1 are reused; quality is input only.
Batching the detached probability transfer and gradient gathering preserves
the per-image squared-persistence formula. Forward/gradient checks on CPU
and CUDA pass. Input-selection/topology/U-Net suites pass 53/53.

An independently seeded Torch shuffle generator prevents channel count from
changing data order through initialization RNG consumption. The score is
frame→orbit→equal four-endpoint mean, with component-wise zero-denominator
exclusion. Thresholds use validation only; bootstrap obtains 2,000 valid
whole-day paired draws. Three seeds are 12026/12027/12028. CORE and a
mathematically guaranteed top-two candidate repeat alongside screening;
repeat scores never enter screen ranking. CUDA works outside the sandbox despite NVML
driver/library mismatch. Processes use disjoint run outputs. A sequential
dispatch was stopped after completed CORE and the first ALL epoch to use
parallel dispatch; that partial ALL start is retained under
`aborted_dispatch_start/`, and ALL is rerun from its declared seed.

## Final verification and boundary

Independent actual-data decoding verifies all 17,612 labels/targets, including
8,832 held-forward partial masks. Independent confusion counts reproduce all
confirmed validation and final test metrics. Both 2,000-draw intervals reproduce
exactly. Original annotation/checkpoint/frame/source fingerprints remain intact;
all 1,378 source size/mtime entries match. A source inference check reproduces
cached preprocessing exactly and predicts three genuinely unreviewed orbits.
Figures were inspected and agree with the frozen example IDs/error semantics.
53 focused tests pass; CPU/CUDA loss and gradient equivalence pass.

Aggregate training process time is 2.13 h with overlapping independent runs.
No failed candidates; the retained single-epoch aborted dispatch start is
separate from the completed matrix. No production checkpoint, manual labels,
automatic masks, threshold or VAE/CVAE input is replaced. No portfolio priority,
deadline or scientific adoption changes. Next use the current inputs/model
unless a separate model/target experiment is explicitly chosen; no further
field sweep is queued.
