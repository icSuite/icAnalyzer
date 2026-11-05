# Automatic-mask, memory-bounded VAE/CVAE training

Last reviewed: 2026-10-14
Status: Native CVAE steps1–4 implemented; user training complete; scientific evaluation pending

## Objective and isolation

Implementation: scripts/CVAE/prepare_automatic_cvae_cache.py and
train_automatic_cvae.py. Current implementation targets native Hall/Pedersen
Kp-CVAE only; the proposed WIC/SI12 cache counterpart is not implemented.
Separate postprocess_automatic_cvae.py now implements step4; Current State
owns completed-run evidence and focused numerical verification.
README owns actual commands, defaults and memory/disk-budget qualifications.

Train on the expanded automatic-mask corpus without loading it all into RAM.
The user accepts more mask errors in exchange for more training data.
Use separate preparation/training/postprocessing entry points and fresh output
directories. Existing runners, loaders, CLI defaults, annotations, trained
artifacts and plotting commands remain unchanged.

Reuse the existing scientific model, loss and training functions. Do not fork
or simplify the partial-convolution encoder, spline decoder, latent dimensions,
quality input, log transform, scheduler, beta, artificial holes or optional
uncertainty modes. Preserve existing Kp conditioning for the native conductance
CVAE; this plan does not add a new conditional sensor model.

## 1. Read automatic masks separately

Input: data/training_data/automatic_masks_e5c009e6713b/manifest.json and
per-orbit NPZs with time=datetime64[us] and mask=bool(N,46,46), True=keep.
The verified collection has1378 orbits/358439 frames.

- Match source orbit, timestamps and actual grid; never assume cross-product
  frame numbers match. Reject ambiguous alignment rather than borrow a mask.
- Combine the mask with the selected product's intrinsic validity. For paired
  fields use the current common-valid convention. Do not manufacture missing
  observations, invert quality weights or treat the mask as measurement noise.
- Exclude frames with no usable, positively weighted paired pixels; retain the
  existing intrinsic missing-data criterion and report every exclusion reason.
- Do not merge manual reviews, change threshold0.475, or clean mask shapes.
- Support precipitation_cs WIC/SI12 and native conductance_cs Hall/Pedersen.
  Require valid conditioning labels for CVAE; do not fabricate missing Kp.

Done: preparation applies timestamp-aligned masks and reports frame eligibility
without touching the old annotation path. Alignment checks are part of normal
ingestion, not a separate test phase.

## 2. Prepare a disk-backed cache

Read/process one orbit or bounded chunk at a time. Save float32 scientific
arrays and Boolean masks in memory-mappable arrays, with a small frame index
containing source identity, time, orbit, optional Kp and split membership.
Estimate disk requirements before writing; compressed-mask size is not an
estimate of the much larger image cache. Never overwrite an existing cache.

Freeze orbit-held-out splits and record seed/source/mask fingerprints. Expanded
orbit sets change the population; the same seed does not reproduce old held-out
frames automatically. Fit log normalization and optional uncertainty calibration
in bounded passes using only the appropriate training population. Preserve the
current order of masking, normalization and intrinsic frame filtering; document
and test any intentional change separately.

Done: preparation RAM is bounded by a chunk/orbit, no corpus concatenation;
cache metadata identifies the exact frozen training inputs and configuration.

## 3. Train from batches, not whole-split arrays

Add a disk-backed Dataset returning the same batch fields as the existing
ArrayDataset. Transform values and uncertainties per accessed frame/batch,
using the frozen training statistics. Shuffle the compact frame index rather
than the image arrays; include all eligible training frames each epoch.

Bound the open-shard/data cache, start with num_workers=0 and minimal prefetch,
and expose a cache-memory budget. This is a data-cache cap, not a promise about
total process RAM; measure CPU peak memory and GPU memory separately.
Validation and test evaluation must also iterate in batches.

Save checkpoints to a new run directory, with the frozen cache/selection and
mask provenance. Give new runs a distinct entry point/checkpoint name so old
plotting code cannot silently reload unmasked source data instead. Reuse the
existing model-state representation where possible; old artifacts remain valid.

Done: the new path runs the actual model and saves its checkpoint, training
history and validation/test results without changing current scientific settings.
Steps1–3 are sufficient for training; ensemble production is not required.

## 4. Separate and bound postprocessing

Training completion must not automatically allocate a huge latent ensemble.
Generate samples in a separate stage with batched decoding and disk-backed
storage; process one requested Kp ensemble at a time. Accumulate means/variances
and use existing covariance/eigenmode science on bounded or disk-backed arrays.
Avoid duplicating the full image ensemble and anomaly factor in RAM.

Use separate cache-aware diagnostic entry points for held-out examples and
inpainting, with bounded inference and explicit example selection. The same
memory restrictions apply to posterior diagnostics and covariance analysis;
do not redirect existing old-run commands to this backend implicitly.

Done: small-ensemble means/covariances match existing calculations and large
sample counts do not cause RAM proportional to the full retained ensemble.

## Execution strategy — rapid implementation, run, compare

User prefers skipping the former standalone verification/staged-rollout step5.
Do not introduce a mandatory parity/CPU-GPU/memory-benchmark campaign before
running the model. Keep necessary shape/time/grid validation and bounded
allocations in ordinary implementation; preserve isolation from old workflows.

Initial scope was steps1–3 for CVAE, with actual training launched by the user.
On2026-10-14 the completed run is verified and the user authorizes step4,
now implemented: bounded cache-aware held-out diagnostics and fixed-Kp prior
samples/physical/log covariance/PCA/teleconnections. Training records numerical
loss history and held-out loss metrics; no automatic retraining is performed.
A scientific
comparison still needs genuinely held-out common evaluation frames when models
use different training populations; no improvement is established in advance.

Implementation order: mask adapter + cache, then trainer, then postprocessing.
Three workflow stages now have separate scripts, sharing existing scientific
model/inference/plotting calculations. Two source files implemented steps1–3,
followed by one separate postprocessor for step4. User training is complete;
the agent only verifies cache/inference/covariance with focused checks and a
small saved-model smoke run. No agent retraining or large-ensemble benchmark;
scientific quality and covariance calibration remain open.
