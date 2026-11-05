# WIC mask U-Net input-channel selection: execution plan

Last reviewed: 2026-10-14  
Status: Executed; [[WIC Input Channel Selection Benchmark]] owns the completed audit, 14 runs and report. Retain CORE inputs; no production model adoption.  
Owner of the eventual experiment: the executing agent; owner of any production-model decision: the user.

## Objective and boundaries

Find a stronger set of IMAGE input maps for predicting the manual 46×46 pixel-acceptance mask. Optimize held-out mask quality, not the number of channels or a narrative about which physical quantity *should* matter. The deliverable is a reproducible comparison and a recommendation; do not replace `data/training_data/U_net_weight_0p1.pt`, regenerate `data/training_data/automatic_masks_*`, change annotations or the GUI, or train the VAE/CVAE.

This is **input selection for the current single-frame, ordinary-convolution U-Net with topology weight 0.1**. Keep architecture, loss, training population, split, and optimization protocol fixed across candidates. Temporal context, a class head, partial convolutions, postprocessing, and different labels are separate experiments. In particular, do not use the previously tested partial-convolution checkpoint as a comparator in the channel ranking: it changes architecture and makes attribution ambiguous. This study may run alongside the separate automatic-mask VAE-ingestion work and must not block it.

The earlier WIC-only field restriction was superseded by the user's explicit statement that any fields are acceptable if they improve the classifier. This plan therefore permits other source products **only when they pass the eligibility check below**. A report must identify what was tested and what was excluded. No variable may be admitted merely because it predicts the annotation if it leaks the annotation or will be unavailable when the model is used.

## Frozen reference

- Repository: `icAnalyzer`; read `AGENTS.md`, `vault/01_Project/Current State.md`, and `vault/05_Handoff/Handoff - Latest.md` before starting. Preserve the existing dirty worktree. Work on the live branch but write only new experiment code, new run artifacts, and the relevant vault checkpoint.
- Labels: the current corrected full-grid manual annotations in `data/partial_image_annotations_46x46.json`, with frame classes and partial keyframes propagated until the next keyframe. Use fully accepted, fully rejected, and partially accepted reviewed frames. Never train on automatic predictions or treat an unreviewed frame as a negative example.
- Training reference: `scripts/automatic_masking/train_model2_unet.py`, its four fields (`wic_corrected`, `method_quality_weight`, `sza`, `dza`), and `data/training_data/U_net_weight_0p1.pt`. The current manual snapshot is `data/training_data/manual_dataset_c272e0b57c36`; the split reference is `figures/validation/wic_mask_unet_fullgrid_20261012/model/summary.json` plus that run's `frames.csv`. Recompute membership exactly as the training script does, then compare the ordered frame identities and split IDs against the saved reference. Abort if identities or split membership differ. Do not make a fresh random frame split.
- Source: the same 46×46 IMAGE product directory currently used by the training/annotation scripts, `~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2`. Resolve it from the live scripts at execution time, record the resolved path, and fingerprint the actual files used. The data are outside the repository; do not copy the full source corpus into Git.
- Treat the current test orbits as **previously inspected**, so this is an exploratory comparison. Do all selection on validation only and inspect test predictions only after a final candidate and threshold are frozen. A strong claim of generalization would require newly reviewed, genuinely uninspected orbits later.

Before running, write a machine-readable protocol manifest with the annotation and reference-checkpoint SHA256 hashes; an ordered source manifest of relative path, byte size, and modification time in nanoseconds, plus its SHA256; exact frame IDs and split IDs; model/loss/optimizer settings; metric definitions; threshold grid; feature groups; candidate list; seeds; and intended outputs. Verify the source manifest again at completion. If those inputs change during the experiment, stop and start a new versioned run; never silently combine snapshots.

## Step 1 — inventory and eligibility

At **execution time**, inventory 46×46 numeric or Boolean image maps in the source files and their provenance. This plan intentionally does not pre-enumerate the IMAGE schema. For each field, record units, shape/dimensions, missing-value convention, fraction finite in annotated train/validation/test frames, fraction finite across the source corpus, whether the value exists at inference, and its upstream derivation. A lightweight header/schema pass first is preferable; inspect values only as needed to quantify coverage and detect constants or invalid ranges.

Admit a map only if it is available before a human annotation, can be reproduced by the later full-corpus annotator, aligns exactly to the same frame/time/grid, has a defined missing-value treatment, and has nontrivial information in the reviewed training set. Reject and document: manual labels/masks/review metadata, automatic masks or model outputs, anything derived from an annotation, frame/orbit identifiers or timestamps used as shortcuts, future-frame data, and products unavailable for the source corpus to be annotated. Inspect the provenance of derived Hall/Pedersen/ratio/energy/flux maps specifically; they are *eligible in principle* under the user's current goal but must pass the same availability and leakage checks. If provenance cannot establish eligibility, exclude the field and say why. Do not infer safety from a field name.

Assign every eligible optional map to exactly one group, recording the mapping in `field_inventory.csv`:

1. Additional WIC signal maps (WIC variants beyond the mandatory `wic_corrected`).
2. WIC reliability/support maps (`method_quality_weight`, coverage, WIC standard deviation or uncertainty, and analogous upstream support maps).
3. Geometry and viewing maps (`sza`, `dza`, and other spatial angle/location maps).
4. Other observed FUV channels or their uncertainty/support maps.
5. Derived physical-product maps that passed the provenance check.

If a source field fits none of these groups, list it as excluded with a reason; do not create ad hoc groups after seeing validation performance. `wic_corrected` is the sole mandatory anchor. Empty groups produce no ablation run. A field absent in only some orbit files may be retained only if a deterministic inference-time missingness rule exists and its missingness pattern is represented in training; otherwise exclude it. Use the same rule in training and in any proposed deployment adapter.

## Step 2 — reproduce the controlled baseline

Implement a **separate experiment runner** under `scripts/automatic_masking/`; do not generalize or rewrite the two existing user-facing scripts merely to run this search. Reuse the existing dataset-target construction, split, model, and loss code wherever possible. A reader should be able to follow the stages chronologically (`#%%` sections are welcome). Add focused checks for channel alignment/order, split identity, train-only normalization, invalid values, and metric aggregation. Do not build a generic hyperparameter-search framework.

Load/rebuild only reviewed training frames. If all eligible maps cannot fit comfortably in memory, use a disk-backed cache keyed by the frozen manifest, with bounded batches; do not load the 358,439-frame source corpus into host RAM. For each field, use the current `prepare_inputs` convention (its physical transform where defined, train-only robust centre/scale, clipped normalized values, zero fill, plus a finite-value flag). Extend it explicitly for new fields and record units/transforms; do not apply an arbitrary WIC transform to a different physical variable. Fit every centre/scale on train frames only, store them with the checkpoint, and verify that validation/test/inference preprocessing is identical. The quality weight remains an **input only** and does not reweight pixel loss.

First train a fresh four-field `CORE` run in the new runner. Reproduce the existing data counts, model parameterization apart from the first-layer channel count, split, topology loss, sample weighting, and validation behavior. Re-score the saved `U_net_weight_0p1.pt` checkpoint with this study's metrics to give context, but rank only fresh runs from the common runner. Data/target/split IDs must match exactly, and transformed baseline inputs on a fixed sample must agree numerically; investigate any discrepancy before comparing additional channels. A changed validation score alone may reflect training randomness and is reported, not silently used to redefine the reference.

Use one frame at a time; same ordinary U-Net, full/rejected/propagated-partial target masks, finite-WIC supervised domain, equal class/orbit/segment training influence, BCE + Dice + topology penalty 0.1, AdamW and the baseline learning rate/weight decay, batch 64, maximum 40 epochs, patience 8, same validation-loss checkpoint rule. Fix data order, initialization, and RNG seeds. Record software versions and device. If memory requires a smaller batch, make that change for **every** candidate and restart the comparison; if a field causes NaN/OOM, report the failed candidate rather than uniquely changing its training setup.

## Step 3 — predeclared score and comparisons

The primary decision score is the equal mean of four error rates on finite-WIC pixels:

| Component | Frames | Error numerator / denominator |
| --- | --- | --- |
| `full_lost` | fully accepted | rejected predicted pixels / eligible pixels |
| `rejected_kept` | fully rejected | accepted predicted pixels / eligible pixels |
| `partial_lost` | partial | manually accepted pixels predicted rejected / manually accepted eligible pixels |
| `partial_kept` | partial | manually rejected pixels predicted accepted / manually rejected eligible pixels |

For each component, compute the rate per eligible frame, average within each orbit, then average equally over orbits with that component. Omit zero-denominator frames **for that component only**, count and report them, and require every component to have evaluable validation and test orbits. The primary score `J` is the unweighted mean of these four orbit-averaged rates; lower is better. This makes the false-keep/false-reject tradeoff visible instead of allowing gains on fully rejected images to hide lost useful partial pixels. Equal component weights are an operational choice for this study, not a claim that the four scientific costs are identical. Always show the four rates separately. Also report accepted-pixel fraction and count, class-wise mask disagreement, and raw/observable shape violations; none replaces `J` after outcomes are seen.

For **each** run, select the acceptance-probability threshold on validation from `0.100, 0.125, ..., 0.900` by minimum `J`; ties go to lower `partial_lost`, then the lower threshold. Save the entire threshold curve. Apply that one threshold to test. Do not tune thresholds on test or compare one model at its optimal threshold with another at the current checkpoint's 0.475. Select the epoch by the common validation-loss rule above, then select the threshold; these are distinct choices.

Run the following fixed candidate matrix. Remove duplicate channel sets before training and list skipped duplicates:

| Stage | Channel set | Purpose |
| --- | --- | --- |
| Screen | `CORE` = current four fields | Reproducible anchor |
| Screen | `ALL` = anchor + every eligible optional field | Broad performance candidate |
| Screen | `ALL - group` for each of the five nonempty optional groups | Group ablation with retraining |
| Screen | `ALL - dza`, `ALL - sza`, if those maps are eligible and the sets are distinct | Directly test the named geometry fields |
| Confirm | Best two distinct screen sets by validation `J`, plus `CORE` | Check seed stability |

Screen with seed **12026**. Confirmation repeats each selected set with seeds **12027 and 12028** on the same frozen split (the seed-12026 screen run counts as its first seed). Ties in screen ranking: lower `partial_lost`, then fewer channels, then lexicographic field-list order. No candidate can be added after viewing test results. This is at most nine screening runs and six repeat runs; report any skipped or failed run. If the budget is infeasible on the available machine, stop after the complete screen and report the incomplete confirmation rather than substitute an opportunistic subset. No exhaustive `2^N` search.

Choose the final field set by the lowest **three-seed mean validation `J`** among confirmed sets. Report per-seed values and paired differences against `CORE`; a nominal gain below one percentage point, unstable sign across seeds, or a paired orbit-bootstrap 95% interval spanning zero is **not convincing evidence** of improvement. In that case recommend retaining `CORE` for now, while still reporting the best numerical candidate. Bootstrap whole orbit/day groups (not pixels or frames), using the same resampled groups for paired candidates; in each draw calculate the score difference for matching seeds and average the three differences. Use at least 2,000 draws with a fixed bootstrap seed. These intervals are descriptive because the same validation data selected the candidates; do not present them as independent confirmation. If a confirmation run fails, do not rank it from fewer seeds; report the failure and the best fully confirmed set. Do not infer individual-field causal importance from the group results or from correlated inputs. The bounded matrix can miss interactions among groups; say so in the report. Optional fixed-model permutation/dropout importance may be included as a diagnostic but never decides the winner.

## Step 4 — final evaluation and report

After the winner, `CORE`, threshold rule, and report template are fixed from validation, evaluate their saved checkpoints on the previously inspected test orbits. Show the same four components, `J`, accepted-pixel counts/fractions, class-wise errors, shape violations, and paired orbit/day-bootstrap intervals. Use **the same predetermined example frames** for every model: select IDs before examining candidate predictions, spanning fully accepted, fully rejected, partial, and WIC missingness levels. Include side-by-side WIC, manual mask, probability, thresholded mask, and error map. Do not cherry-pick attractive examples. If the chosen set uses derived or extra-camera fields, verify those maps exist and align in a small full-corpus inference dry run; do not annotate the full corpus in this task.

Write all outputs into a new dated/versioned directory under `figures/validation/wic_input_selection_YYYYMMDD/` and do not overwrite an existing run. Required files:

- `protocol.json`, `field_inventory.csv`, `split_ids.json`, source/annotation/checkpoint fingerprints, and a command transcript or reproducible command list.
- One subdirectory per run with ordered input fields, preprocessing statistics, seed, training history, best epoch, threshold curve, checkpoint, validation metrics, and failure record if applicable. Store checkpoints in the versioned experiment directory; do not overwrite the standalone model.
- `ranking.csv` with every screen and confirmation run, four component rates, `J`, threshold, best epoch, seed, and runtime.
- `comparison.png` for validation/test tradeoffs, `same_frame_examples.png` for fixed examples, and `report.md`.

`report.md` must lead with the recommended **ordered channel list** and whether it convincingly improves over `CORE`. Include eligible/excluded fields and reasons; sample/split counts and hashes; exact run matrix and deviations; validation and exploratory test tables with all four errors and intervals; accepted-pixel-volume and shape tradeoffs; compute cost; failure cases; and the practical next action. Say explicitly if no field set improves enough: then input selection is unlikely to be the immediate bottleneck, and a later architecture/target experiment is a separate decision. State that manual-mask prediction quality is a proxy for, not a demonstrated improvement in, VAE science or annotation time.

## Completion and handoff

Done means the frozen protocol, inventory, controlled runs, verification checks, figures, and `report.md` exist and can be reproduced; the user receives the recommended field set and evidence, including an honest no-improvement result if applicable. Update the project vault's Current State and latest Handoff with the result and `Portfolio impact`. Do not modify the central second brain unless portfolio status, priority, deadline, or research significance actually changes. Leave any model adoption, full-corpus reannotation, or VAE integration for a separate user decision.
