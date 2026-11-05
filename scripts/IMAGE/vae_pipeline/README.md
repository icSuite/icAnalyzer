# New Product 2/3 Model-2 training data

Generate fresh Model-2 masks from the current fuvpy-background-subtraction
Product 2 corpus, then combine those masks with Product 3 Hall/Pedersen fields
in a new disk-backed CVAE cache:

```bash
python scripts/automatic_masking/apply_model2_unet.py
PYTHONPATH=. python scripts/CVAE/prepare_cvae_hp_cache.py \
  --minimum-valid-fraction 0.05 \
  --output data/training_data/automatic_cvae_fuvpy_bs2d_valid5
```

The defaults identify `IMAGE_FUV_new`,
`IR_hardy_smoothed_wic1_si121_si131`, and the frozen Model-2 checkpoint hash.
Both commands require fresh output directories and preserve the old masks and
cache. The first command applies the existing Model 2; it does not retrain it.

# WIC-only Kp-CVAE

This diagnostic path models corrected WIC counts directly from Product 2. It
uses the same Model-2 masks, Kp conditioning, partial-convolution encoder,
spline decoder, artificial training holes and legacy quality-weighted loss as
the conductance CVAE, but has one learned physical output channel. It does not
use SI12 or enable measurement-uncertainty/coverage experiments during
training.

Prepare a distinct WIC cache:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py \
  --masks data/training_data/automatic_masks_fuvpy_bs2d_sigma2_pointing_v1_e5c009e6713b \
  --minimum-valid-fraction 0.05 \
  --output data/training_data/automatic_wic_cvae_fuvpy_bs2d_valid5
```

The source default points to the current Product-2 generation. The explicit
mask manifest records that same generation; source path, size, modification
time, timestamp and grid checks prevent masks from being silently paired with
another product.

Train the WIC CVAE to a fresh directory:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  data/training_data/automatic_wic_cvae_fuvpy_bs2d_valid5 \
  --output figures/cvae_runs/wic_model2_fuvpy_bs2d_valid5_b264_beta0048 \
  --epochs 500 --batch-size 264 --beta 0.048 \
  --latent-dim 36 --spline-control-points 30 --spline-degree 3 \
  --learning-rate 0.0001 --seed 1 --training-holes --device cuda
```

These settings match the recent conductance-CVAE comparison, but beta is not
calibrated for the one-field state. Treat the first run as a diagnostic of the
generated WIC distribution and background, not a final model.

Generate matched latent draws at Kp0--6 after training and propagate every
member through the flux-weighted collapsed Zhang--Paxton electron energy and
Fang (2025) electron conductance:

```bash
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py \
  figures/cvae_runs/wic_model2_fuvpy_bs2d_valid5_b264_beta0048 \
  --kp 0 1 2 3 4 5 6 \
  --output figures/cvae_runs/wic_model2_fuvpy_bs2d_valid5_b264_beta0048/wic_kp_0_6_fang_igrf \
  --draws 50000 --device cuda --no-plots \
  --fang-epoch 2001-01-01T00:00:00 \
  --fang-low-energy-policy clip
```

Each Kp folder contains `ensemble_products.npz` with WIC and joint Hall/Pedersen
means, anomaly factors, support and 10/50/90th percentiles. It also records the
collapsed mean energy, structural latitude spread, energy flux per WIC count,
Fang domain masks, reconstructed geographic grid, Apex conversion error and
spatial150km IGRF field. The conductance field order is Hall, Pedersen.
Identical latent draws are used at every Kp.

The PNG/PDF summaries use the same `Polarplot.contourf` presentation as
`Hardy_CVAE_comparison.py`: the saved46x46 MLAT/MLT statistics are interpolated
only for smooth polar rendering. The model spline is not reevaluated on a
denser scientific grid, and the plots do not claim additional resolution. If
sampling was run with `--no-plots`, generate the same figures afterward without
model decoding or Fang propagation:

```bash
PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py \
  figures/cvae_runs/wic_model2_fuvpy_bs2d_valid5_b264_beta0048/wic_kp_0_6_fang_igrf
```

The shared default limits are24S Hall and12S Pedersen, chosen to contain the
saved Kp0--6 90th-percentile maxima of23.13S and10.73S while preserving direct
comparison across Kp. Override them with `--hall-vmax` and `--pedersen-vmax`.

This follows current icBuilder Product3 Fang conventions: positive energies
below1keV are clipped to1keV and flagged; energies above20keV remain
unavailable. `--fang-low-energy-policy reject` is retained as an optional
strict-domain diagnostic. The CVAE grid stores MLAT/MLT but no frame time or
geographic footprint, so the sampler converts MLT to Modified-Apex longitude,
uses `apex2geo` at the130km IMAGE reference height, and evaluates ApexPy IGRF at
150km at that same geographic latitude/longitude. The reference epoch is
recorded in every output and can be changed with `--fang-epoch`. Its default is
the round in-period epoch2001-01-01T00:00:00UTC; the full WIC cache spans
2000-05-16 to2003-05-12. It is a reproducible climatological approximation;
unlike Product3 it cannot use an event-specific frame time.

Outputs are electron-impact conductance only: there is no SI12 proton or
solar-EUV contribution, WIC measurement-noise propagation, or sampling of the
collapsed Zhang--Paxton structural spread.

Two full anomaly factors are stored. Seven50,000-member Kp ensembles require
roughly9GB before NPZ compression, in addition to temporary memory while one
Kp is propagated. Existing two-field artifacts remain loadable; their default
field count and semantics are unchanged.

## WIC Model-1 + Model-2 sensitivity run

This is a separate mask-policy comparison, not a replacement for the completed
Model-2-only WIC run. The current policy has exactly two networks: Model1
releases every intrinsically observed pixel when its frozen whole-frame
acceptance score passes; all other frames retain the new-product Model2 pixel
mask exactly. There is no temporal Bad-veto network.

The earlier new-product combined masks are complete at
`data/training_data/two_stage_masks_fuvpy_bs2d_m1m2_20261104`. They were made
with the now-retired temporal Bad veto and are historical artifacts. They do
not represent the current acceptance-only policy. Generate a distinct current
mask lineage with:

```bash
PYTHONPATH=. python scripts/automatic_masking/apply_two_stage_masks.py \
  --baseline data/training_data/automatic_masks_fuvpy_bs2d_sigma2_pointing_v1_e5c009e6713b \
  --trial figures/validation/wic_model1_accepted_partials_20261015 \
  --output data/training_data/two_stage_acceptance_masks_fuvpy_bs2d_20261105
```

The historical manifest records5,343 releases among400,554 frames across416
orbits and350,786 newly retained finite WIC pixels. Those counts include the
retired veto and must not be attributed to the current policy. Model1 was
trained on the earlier Product-2 generation; any acceptance-only export remains
a mask-transfer sensitivity test, not a validated superior mask.

Prepare a fresh WIC cache using the combined masks. The command uses the
repository SSD; ensure at least approximately16GiB plus working headroom is
free before retaining both the approximately7.3GiB cache and7.4GiB Kp ensemble:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py \
  --masks data/training_data/two_stage_acceptance_masks_fuvpy_bs2d_20261105 \
  --minimum-valid-fraction 0.05 \
  --output data/training_data/two_stage_acceptance_wic_cvae_fuvpy_bs2d_valid5_20261105
```

Then train with settings matched to the Model-2-only WIC run:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  data/training_data/two_stage_acceptance_wic_cvae_fuvpy_bs2d_valid5_20261105 \
  --output figures/cvae_runs/wic_model1_acceptance_model2_fuvpy_bs2d_valid5_b264_beta0048 \
  --epochs 500 --batch-size 264 --beta 0.048 \
  --latent-dim 36 --spline-control-points 30 --spline-degree 3 \
  --learning-rate 0.0001 --seed 1 --training-holes --device cuda
```

Keep the same seed, epoch and Fang policy during fixed-Kp sampling. Its full
anomaly factors are comparable in size to the existing7.4GiB Model-2 output.

```bash
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py \
  figures/cvae_runs/wic_model1_acceptance_model2_fuvpy_bs2d_valid5_b264_beta0048 \
  --kp 0 1 2 3 4 5 6 \
  --output figures/cvae_runs/wic_model1_acceptance_model2_fuvpy_bs2d_valid5_b264_beta0048/wic_kp_0_6_fang_igrf \
  --draws 50000 --batch-size 256 --device cuda --no-plots

PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py \
  figures/cvae_runs/wic_model1_acceptance_model2_fuvpy_bs2d_valid5_b264_beta0048/wic_kp_0_6_fang_igrf
```

# Automatic-mask, disk-backed conductance CVAE (separate workflow)

Steps1–3 are implemented in two new scripts. Existing runners, annotations,
checkpoints and old-run plotting commands are unchanged. This workflow always
trains the existing Kp-conditional Hall/Pedersen spline model: same architecture,
quality encoder channel, legacy loss, scheduler and optional holes/uncertainty.
It does not modify or clean the automatic masks, apply CSV/manual selection, or
repeat upstream conductance propagation.

The former conductance two-stage masks and cache used Model1 acceptance plus the
retired temporal Bad veto. They remain historical artifacts and must not be
presented as outputs of the current acceptance-only exporter. A new comparison
requires fresh mask and cache directories. Current manifests link only the
Model1 acceptance checkpoint, its frozen threshold and Model2 provenance.

For that prepared two-stage cache, launch directly from Python in tConductance:

```bash
python scripts/IMAGE/step_1_two_stage_cvae.py
```

Edit the visible settings in the script or pass overrides such as `--epochs 20`.
It matches the actual previous run's batch264/beta0.048/500epochs with holes.
CUDA is checked before loading caches or creating output; CPU requires the
explicit `--device cpu` override for this entry point. The generic trainer's
existing `--device auto` behavior remains available. The old Bash command is
only a compatibility shim.

Optional coverage weighting uses the **eligible training frames only**. Sum the
original valid positive quality weights per grid cell, smooth that coverage
with a Gaussian (`sigma=1.5` grid pixels), then invert it relative to the highest
smoothed observed coverage. `--coverage-balance full` uses this full uncapped
inverse; `--coverage-balance capped` limits the multiplier to3 by default.
The generic trainer defaults to capped mode. Gaussian smoothing is normalized
at unsupported cells and grid edges; cells with no training coverage receive
no boost. Original masks remain in force, and encoder inputs retain their
original quality values. The loss multiplies quality by the map and retains
its existing batch normalization and KL term. Smoothing is performed before
inversion, so full balance equalizes estimated smoothed coverage rather than
guaranteeing exactly equal unsmoothed cumulative weights.

The Python entry is prepared for the chosen full-balance comparison:
`COVERAGE_BALANCE = "full"`. Set `COVERAGE_WEIGHTING = True`, or enable it on
the command line. Both select a separate `_coverage_full` output automatically:

```bash
python scripts/IMAGE/step_1_two_stage_cvae.py --coverage-weighting
```

`--coverage-balance capped` selects the earlier partial compensation and the
separate `_coverage` output. An explicit `--output` always takes precedence;
existing run directories are never overwritten. The direct generic trainer
requires both `--coverage-weighting --coverage-balance full` for full balance.

The default is one shared spatial map for all Kp. Optional
`--coverage-kp-edges 0 2 4 6 9` fits four spatial maps from their training
populations; intervals include their lower edge, and the final bin includes9.
Empty training bins are rejected: use broader bins. `--coverage-sigma` controls
smoothing. `--coverage-max-boost` applies only to capped mode and is ignored in
full mode. Weighting remains opt-in, and a running process does not acquire
new settings.

Each new weighted run saves `coverage_weighting.npz` (raw/smoothed quality sums,
factors and Kp edges) and `coverage_weighting.json` (frozen factors, settings,
training counts and cache hash). The model checkpoint also embeds the factors
in its loss configuration. Validation/test losses and the postprocessing
`training_effective` metric reuse these training-fitted maps; neutral and
quality-only reconstruction metrics remain available for comparisons. No
existing cache needs rebuilding. This implements a controlled experiment;
improved dayside reconstructions or samples have not yet been demonstrated.

First prepare the cache (no training):

```bash
PYTHONPATH=. python scripts/CVAE/prepare_cvae_hp_cache.py \
  --masks data/training_data/automatic_masks_e5c009e6713b \
  --minimum-valid-fraction 0.05 \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_cache/automatic_cvae_e5c009e6713_valid5
```

The default source is the current smoothed Hardy/Robinson conductance_cs
directory; `--source` selects another directory. Preparation matches the
precipitation masks to conductance by exact timestamp and checks actual grids
and recorded mask-source evidence. Unmatched or entirely unusable masks are
excluded. New preparation defaults to **at least5% intrinsic common-valid H/P
coverage** across the whole46x46 grid, before automatic masking. This permits
up to95% missing data and retains more frames. `--minimum-valid-fraction 0.05`
makes the choice explicit; the equivalent `--maximum-missing-fraction 0.95`
is also accepted. These flags are mutually exclusive. Use
`--maximum-missing-fraction 0.33` to reproduce the original67%-valid cutoff.
The completed automatic CVAE and its original cache still use that old rule;
new eligibility requires a fresh cache and training run.
No new masked-area cutoff is imposed. `--orbits 85 86 89` optionally restricts
preparation. Split defaults are validation/test0.15 each, seed1, whole orbits
held out. These splits are frozen into the cache; expanded orbit lists do not
reproduce old held-out identities merely by keeping the same seed.

One orbit at a time is written as `physical.npy` (H,P,dH,dP,quality),
`valid.npy` and compact frame metadata. Normalization uses positive masked
training frames before the intrinsic cutoff, preserving the current annotation
workflow's ordering, with float64 log-moment accumulation. Kp is retained from
the native source with its3-hour interval provenance; no labels are fabricated.
Final `manifest.json` marks completed preparation; incomplete caches cannot
be trained. Existing directories are never overwritten. The full-corpus upper
disk estimate is about14.8GiB for uncompressed arrays, not the~17MiB compressed
mask collection. Actual size decreases when unusable frames are excluded.

Then train from that cache:

```bash
PYTHONPATH=. python scripts/CVAE/train_automatic_cvae.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_cache/automatic_cvae_e5c009e6713_valid5 \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_automatic_cvae_b16_holes_valid5 \
  --epochs 500 --batch-size 8 --beta 1.6 --training-holes --device auto
```

Kp conditioning is always enabled; `--condition-on-kp` and `--sample-kp` are
not needed in this training-only command. The illustrated beta/holes/epochs
are explicit experiment settings, not changed defaults. Defaults otherwise
match the current native runner:1000 epochs, batch8, latent36,30x30 cubic spline,
Adam1e-4, beta1, no holes/uncertainty, plateau factor0.5/patience10. Training
seed defaults to the cache split seed; `--seed` overrides training RNG only.
`--measurement-uncertainty` and `--gaussian-uncertainty` retain the current
native loss modes and associated precision/model-error settings. Precision
calibration uses the same bounded global training-frame sample/formula.

The Dataset maps scientific arrays from disk and copies only accessed frames.
Training defaults to `--num-workers 2 --prefetch-factor 2`: two background
workers, at most two prefetched batches each. `--cache-mib 512` is the total
training mapping budget, divided across those workers (256MiB each); validation
and test use serial loading with their own512MiB budgets. `--max-open-orbits 64`
is the per-worker file-map limit, still constrained by its byte budget. Verified
array headers are remembered when maps must be reopened. A single orbit larger
than a worker's share requires increasing `--cache-mib` or reducing workers.
Total RAM also includes the model, split/worker metadata, small prefetched
batches and reclaimable OS file cache; this is not an absolute RAM limit.
`--num-workers 0` restores serial loading. CUDA batches use pinned host memory
and non-blocking transfers. Uncertainty arrays/calculations are skipped unless
an uncertainty loss is requested; cache files are unchanged.

Workers are spawned afresh each epoch, avoiding fork-after-CUDA and retaining
the existing main-process RNG sequence, shuffled batch order and hole draws.
Epoch losses accumulate in float64 on the device and transfer once per epoch,
instead of transferring three scalars per batch. Architecture, loss, batch size,
optimizer and scheduler are unchanged. Train/validation/test all use batches;
epoch/batch and test-evaluation progress remain. Existing running processes
must restart to use these changes. There is no mid-training resume checkpoint;
use a fresh run output rather than overwriting or deleting an existing run.

Outputs: `automatic_cvae.pt`, `cache_manifest.json`, `history.json` and
`summary.json`. The same model-state format is used with a distinct filename
so old plots cannot silently reload unmasked source data. The cache must be
kept for future cache-aware diagnostics. No ensembles or covariance arrays
are generated during training; those belong to step4 and are not implemented
by training itself. The separate cache-aware postprocessing command below
now implements these outputs.

## Automatic CVAE postprocessing (no retraining)

Use the completed run and its original cache. The cache manifest, normalization
and training support must match the checkpoint; `--cache PATH` accepts a relocated
copy with unchanged contents. Native data/masks are loaded from that cache,
not reconstructed from today's source files or old manual annotations.

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/postprocess_automatic_cvae.py \
  ~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_automatic_cvae_b16_holes \
  --output figures/cvae_runs/conductance_automatic_kp3 \
  --kp 3 --samples 5000 --batch-size 256 --device auto
```

This generates loss/learning-rate history, normalized-log reconstruction RMSE,
full-input versus damaged-input Hall/Pedersen examples, five fixed-Kp prior
samples, prior mean/standard deviation, four joint PCA modes and Hall/Pedersen
teleconnections. `--kp` is required for sampling: a fixed-Kp prior must not be
silently mixed across geomagnetic conditions. Held-out reconstructions always
use each frame's actual cached Kp. Kp outside the training range is flagged.
Prior samples are unrelated to the individual held-out observations.

Defaults select128 evenly spaced frames from the retained cached test split,
plot3 examples and remove a12x12 square around the strongest joint log signal.
`--split validation` changes the held-out split; `--heldout-count N` changes the
bounded subset; `--frames 210 660 1020` selects explicit split IDs instead.
Titles and heldout_reconstructions.npz preserve actual cache split frame IDs,
orbit, native source-frame ID, time and Kp. These IDs need not match old runs.
`--hole-policy random --seed 1` uses seeded valid-pixel patch centers instead.
`--hole-size N`, `--examples N`, `--modes N` and `--grid-id ID` expose the remaining
choices; modes sets the number saved (plots show at most the first4). Default
anchor is maximum Hall variance on the46x46 flattened grid.

For only loss and held-out diagnostics, add `--diagnostics-only` and omit Kp;
no ensembles/covariance are then allocated or saved. All outputs go into a
fresh directory; no overwrite or source-run rename is needed.

Numerical outputs include heldout_reconstructions.npz, conductance_samples.npy,
latent_samples.npy, physical conductance mean/joint covariance, leading PCA,
log10(conductance/S+1) mean/joint covariance, training_support.npy, grid.npz
and summary.json. State order is flattened Hall grid then flattened Pedersen
grid, C order. Covariance units areS² or dex²; log covariance is not standardized
by the model's channel spreads. Each covariance is~137MiB on this grid.
Samples are batched float32 disk-backed arrays (~81MiB at5000 or~807MiB at50000
paired draws), with mapped pages released when supported. Stable float64
covariance is accumulated in bounded row blocks, one physical/log matrix at a
time; no sample-by-sample Gram matrix or full ensemble/anomaly copy in RAM.

Reconstruction means decode the posterior latent mean; they are not the
physical mean of many posterior draws. Reference images are native IMAGE
retrievals, not independent truth. The prior standard deviation/covariance is
model-implied variability, not calibrated measurement error. Unsupported
training cells are zero storage placeholders and masked out of plots/scoring.
Large-ensemble/PCA outputs do not establish uncertainty calibration or improved
scientific skill by themselves. The old-run plotting commands remain unchanged.

# Direct native conductance_cs VAE

`run_conductance_pipeline.py` trains on native Hall/Pedersen retrievals, not
on WIC/SI12 and not on a second conductance conversion. From the repository root:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_conductance_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/conductance_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2/robinson \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --epochs 1000 --batch-size 8 --latent-dim 36 \
  --beta 1.6 --training-holes --samples 50000 --device auto
```

The beta/hole flags above match the existing experiment; they are not new
defaults. Defaults remain beta1, no artificial holes, no uncertainty weighting,
latent36,30x30 cubic controls, partial-convolution encoder including quality,
6x6 encoder features and legacy18x18 decoder feature stage, Adam1e-4/batch8,
1,000 epochs, seed1 and the same ReduceLROnPlateau factor0.5/patience10 on
training total loss. Best validation loss selects the saved model. This uses
the SAME `SplineWICSI12VAE` class and exact legacy reconstruction/KL reduction;
its historical class name is retained for model compatibility. Channel meaning
is explicitly Hall/Pedersen in S in the distinct native artifact.

Both VAE runners now show automatic `tqdm` progress: training device and batch
count, an epoch bar with elapsed time/ETA and train/validation loss plus learning
rate, and a batch bar for the current training/validation epoch. No extra flag
is needed. ETA covers training and epoch validation, not loading or downstream
sampling/plotting. An already-running process cannot pick up this display
change; it appears on the next invocation. Training/history semantics are unchanged.

Input requirements: reviewed46x46 grid, Robinson `conductance_cs`, H/P/dH/dP
in S, declared high-is-auroral quality, native value/uncertainty-valid flags,
finite nonnegative uncertainties and positive value/uncertainty coverage.
Use quality directly, multiplied by conductance coverage; NEVER invert again.
Both fields share valid retrieval support and this common loss/encoder weight.
This follows the current canonical positive-coverage policy, not the interactive
legacy helper's separate coverage>=0.8 cutoff. Negative physical values are
clipped to zero with a warning/counts in provenance; input files are not edited.
Use training-only standardized log10(conductance_S+1), with dH/dP propagated
through that transform. There is no imposed2-keV energy: energy/flux retrieval
and nonlinear conversion/binned conductance already happened upstream.
icReader/icBuilder/icPhysics are unchanged.

By default manual intervals apply through WIC **source indices**, then the
same intrinsic33% missing-data frame cutoff and orbit-separated seed1 splits.
`--no-csv-selection` bypasses annotation. Default corpus audit:157 annotated
orbits;4,033 train,487 validation,1,101 test retained frames;39 never-observed
training cells. These frame IDs are native-run-specific, NOT the FUV run IDs.
Current copied conductance_cs has NO SZA field: `--minimum-sza` explicitly
fails instead of applying an invented or unaligned mask.

Keep uncertainty disabled for the baseline. `--measurement-uncertainty` opts
into the SAME training-median relative-precision loss, now from dH/dP, with
`--uncertainty-variance-floor 1e-3`, `--maximum-relative-precision 100` and
`--precision-scale-samples 2000000` defaults. Alternatively
`--gaussian-uncertainty --model-error-std 0.5 0.5` selects the existing Gaussian
measurement-plus-discrepancy loss (STD in standardized log units). Options
cannot be mixed; related settings require their corresponding opt-in. Count
likelihood is not exposed for conductance in S. No uncertainty encoder channels,
uncertainty output or noise covariance is added; this does not establish a
noise-deconvolved prior or calibrated uncertainty.

Fresh output directories are protected. Saved files are `conductance_vae.pt`
(type `conductance_vae_run_v1`, model/transform/settings/history/provenance/
splits), `summary.json`, and `ensemble_products.npz` (stored latent draws,
conductance mean/anomaly factor, training support, sparse illustration).
There are deliberately NO fictitious FUV arrays. `figures/` contains loss/
learning-rate history, five conductance draws and latent diagnostics. No full
sample-by-sample covariance is formed. `--sample-batch-size` controls decoding
(default256). Fifty-thousand-member storage still scales with the grid/sample
count; only one conductance family is stored.

### Optional Kp-conditional native conductance VAE

`--condition-on-kp` enables a new conditional model; existing defaults and saved
unconditional native/FUV models retain their previous behavior. The encoder
receives H/P, quality and an explicit global Kp; the decoder receives z and Kp.
Kp normalization is fitted on retained training images only and stored with
weights, source provenance, input Kp labels/coverage and generation settings.
The standard-normal latent prior and current physical/spline transform remain.

Example training command in `tConductance` (use a new output directory):

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_conductance_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/conductance_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2/robinson \
  --hybrid-annotations data/partial_image_annotations_46x46.json \
  --condition-on-kp --sample-kp 3 \
  --epochs 500 --beta 1.6 --training-holes --samples 1000 \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid_kp
```

`--sample-kp` is required explicitly when conditioning is enabled. Missing,
nonfinite, misaligned or outside0–9 labels are rejected. Source Kp intervals
must contain their image times. Checkpoint reload verifies retained labels;
old unconditional artifact metadata/configuration is unchanged. Reusing a
conditional artifact requires `--condition-on-kp`, the original input settings,
and an explicit new sampling Kp; this does not retrain weights.
If annotation continues while a run is being trained or reviewed, pass a frozen
JSON copy: saved-run fingerprint checks reject later edits to that input file.

Generate comparable mean and0.1/0.5/0.9 quantile maps after training:

```bash
PYTHONPATH=. python scripts/CVAE/sample_conductance_at_kp.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid_kp \
  --kp 1 3 5 6 --draws 50000 --threads 4 \
  --pedersen-vmax 15 \
  --output figures/conductance_hybrid_kp_ensembles
```

The same sampler now accepts either conductance_vae.pt or automatic_cvae.pt
without renaming or modifying the checkpoint. For the automatic run, use the
separate ensemble-export step (not seven full postprocessing runs):

```bash
PYTHONPATH=. python scripts/CVAE/sample_conductance_at_kp.py \
  ~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_automatic_cvae_b16_holes \
  --kp 0 1 2 3 4 5 6 --draws 50000 --device auto --no-plots \
  --output ~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_automatic_kp_0_6
```

Each Kp folder contains the SAME ensemble_products.npz layout as before,
plus sampling.json; the root contains sampling_summary.json. No held-out
diagnostics, dense covariance or PCA are repeated. --no-plots skips the optional
mean/quantile PNG/PDF figures; omitting it preserves the original behavior.
The original export still materializes one ensemble and anomaly factor in RAM;
this compatibility update does not make that legacy NPZ writer disk-backed.
The separate postprocessor remains available for bounded-memory large ensembles
and covariance diagnostics. Existing output directories remain protected.

The sampler uses identical z draws at each Kp and stores one ensemble at a time.
Each subfolder contains native mean/anomaly-factor/support/latent arrays,
per-member `conductance_sample_kp`, q10/q50/q90 maps, MLAT/MLT, PNG/PDF figures
and JSON condition/model provenance. Nonempty outputs are protected. Kp0–9
requests are allowed; generation outside the recorded training range warns and
is marked as extrapolation. No mean trend or agreement with Hardy is imposed.
The shared encoder/decoder uses all training images; no separate VAE per Kp bin.

To load a sampled subfolder with `load_conductance_ensemble.py`, set
`run_directory` to the trained model and `ensemble_directory` to that subfolder.
`sample_kp` then exposes the per-member labels. Model provenance is checked.

Shared held-out/inpainting visualizer diagnostics use each input image's Kp
for its encoder and matched prior draws. The sparse illustration in the training
runner likewise uses its reference image's Kp, even if `--sample-kp` differs.
Legacy unconditional prior/event audits and covariance-reconstruction scripts
reject conditional artifacts rather than silently compare mismatched activity.
Observation-only browsing and saved-ensemble plots remain available.

Verification:307 tests pass; old native/FUV checkpoint decoding and model RNG/
configuration are bit-identical to the pre-change code. A one-epoch real-source
training/reload/sampling smoke passes. This checks software integration, not
conditional scientific skill. The older `vae_runs` hybrid remains unconditional;
the separate user-trained `cvae_runs` hybrid is reviewed in the current
Kp-response and annotated-population notes. Physical calibration remains open.

### Measure current annotated conductance versus Kp

This comparison freezes the current JSON and CSV and loads a new hybrid
selection, independently of the historical checkpoint annotation fingerprint.
It computes physical means and0.1/0.5/0.9 quantile maps, count/orbit maps,
orbit-balanced means and a quality-weighted log-mean sensitivity. Saved
conditional ensembles and Lompe's auroral Hardy function use the same grid
and covered comparison cells. No training or original-output changes occur.

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/measure_conductance_kp.py \
  ~/Dropbox/work/data/IMAGE_FUV/cvae_runs/conductance_smoothed_b16_holes_hybrid \
  --annotations data/partial_image_annotations_46x46.json \
  --output figures/current_conductance_kp
```

Choose a new output directory. Defaults require20 observations and3 orbits
per cell in every bin1–6. Bins[k,k+1) contain fractional Kp; the saved models
use exact integer k, so the comparison is approximate. All retained Kp0–9
images are recorded, including activities beyond original training coverage.
Results/qualifications: `vault/02_Algorithm/Current Annotation Conductance Kp Test.md`.
Canonical2026-10-12 output: `figures/validation/kp_current_annotation_comparison_20261012/`.

### Load paired conductance members, moments and grid

User-selected analysis run (2026-10-12):
`conductance_smoothed_b16_holes_hybrid`. This includes6881/1203/1117 retained
train/validation/test frames, selected epoch207, and2077 supported grid cells.
Earlier diagnostics used `_annotated`; a separate full hybrid rerun is now
recorded in `vault/02_Algorithm/Hybrid Conductance VAE Validation.md`.

From the repository root in the `tConductance` environment:

```bash
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 PYTHONPATH=. \
python -i scripts/IMAGE/vae_pipeline/load_conductance_ensemble.py
```

The interactive example leaves `Hall`, `Pedersen`, `Hall_mean`,
`Pedersen_mean`, `covariance`, `covariance_HH`, `covariance_HP`, `covariance_PP`,
`support`, `state_flat_indices`, `grid`, `mlat` and `mlt` available. Each channel
has shape(50000,46,46), in S, with paired member ordering. The joint4154x4154
covariance has units S² and uses supported Hall cells followed by supported
Pedersen cells. The saved factor is `(sample-mean)/sqrt(N-1)`, so recovery uses
the matching saved mean and scales bysqrt(N-1); float32 storage causes tiny
roundoff differences. Unsupported map values are NaN and excluded from C.

Full float64 samples occupy about1.7GB; covariance adds about138MB. The example
computes sample moments in blocks and verifies the CSgrid loaded from a saved
source file against artifact MLAT/MLT coordinates and grid ID. These are the
native magnetic coordinates; geographic placement for a Lompe event is a
separate coordinate conversion. Run the `#%%` sections in an editor if preferred.
The loading/moment workflow is verified on all50000 saved hybrid members.

Never-observed locations are excluded using common validity of RETAINED
training frames only, not test/validation support. Sample masking occurs BEFORE
inverse exponentiation. Zero values outside this domain are numeric storage
placeholders identified by `conductance_supported`; they are NOT synthetic
training background. Samples/plots/reconstruction must respect this mask.
This does not fix weakly observed corners or constrain the raw decoder there.

To resample without retraining, rerun the same command/settings with
`--reuse-artifact`; change `--samples` if desired. Data files/metadata/file
timestamps, selection, splits and training settings must match. The saved model
is preserved; derived ensemble/summary/preview figures are replaced explicitly.
The sparse number printed at completion is a single-frame physical-space
illustration on unobserved supported state cells, using exact reference
subsamples and assumed0.5 S errors, NOT a noisy/calibrated held-out benchmark.
Physical Gaussian updates may be negative; counts are reported, not hidden.

The existing conductance figure4/5 and held-out browser scripts automatically
recognize native runs:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --frame-index 210 --modes 50 --output figures/native_conductance

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_field_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --frame-index 210 --modes 50 --output figures/native_conductance

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_heldout_samples.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --output figures/native_conductance_browser
```

Browser shows original native common-valid inputs; reconstructions additionally
exclude never-observed training cells. IDs remain identical within a run.
`--fields fuv` is invalid for native runs. The older WIC/SI12 figure1/2/3
scripts remain FUV-specific. The general `visualize_wic_si12_run.py` script
now detects native `conductance_vae.pt` artifacts and uses Hall/Pedersen
labels and siemens throughout. For actual encoder/decoder held-out tests:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_annotated \
  --posterior-prior-only \
  --output figures/conductance_annotated_vae_diagnostics
```

This generates `posterior_prior_examples.png` (full-input reconstructions),
`heldout_inpainting_examples.png` (truth, removed patch, full-input and
damaged-input reconstructions, error), plus prior/moment/spectrum/latent
diagnostics and JSON metrics. Saved split selection and manual masks are
reloaded; never-observed training cells are excluded before exponentiation.
It neither retrains the model nor loads the large saved prior anomaly factors.
Use `--inpainting-patch-size`, `--inpainting-examples`, `--posterior-examples`
and `--posterior-device` to configure it. Omit `--posterior-prior-only` for
the usual saved prior ensemble/covariance plots, or use `--posterior-prior`
to generate both groups. The run directory must contain exactly one of the
two artifact types; use a new output directory to preserve earlier figures.

Native presentation figure4/5 use log-pixel covariance OI only,
not coefficient or decoder fitting. Figures4/5/6 now use native per-pixel
dH/dP by default: first-order propagation to dex is
`sigma_log = sigma_S / (ln(10) * (conductance_S + 1))`. Only retained
measurements and their errors enter conditioning; no method-quality weight is
multiplied into the likelihood. Measurement errors are assumed diagonal;
spatial and H/P retrieval-error correlations are unavailable. This uses the
observed uncertainties, not a calibrated model-error term.

Omit `--observation-std` to use dH/dP on a native run, or require it explicitly
with `--observation-errors propagated`. An explicit `--observation-std 0.1`
retains the old fixed-error calculation; `--observation-errors fixed` also
selects it (default0.1 dex). FUV-derived proxy runs retain fixed errors by
default and reject native-dH/dP mode. No uncertainty floor is imposed by
default. `--minimum-observation-std 1` explicitly floors dH/dP at1 S before
log propagation; this is optional and recorded, not an adopted calibration.
Retained zero dH/dP fail with a request for an explicit positive floor.

For example, reprocess native frame660 with its actual uncertainties:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --frame-index 660 --remove-half top --modes 100 \
  --observation-errors propagated --output figures/native_conductance_dhdp
```

Companion NPZs retain raw input dH/dP and the actual per-observed-cell dex
errors (NaN elsewhere); JSON records the error source, transformation, floor
and clipped-cell counts. Posterior SD is conditional uncertainty in S under
the truncated Gaussian log prior, not the input dH/dP itself. First-order log
propagation may be poor at large relative uncertainty, and omitted modes/model
discrepancy remain absent. VAE training and source artifacts are unchanged.

Native and FUV-derived conductance runs also support figure3:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_teleconnections.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes \
  --grid-id 750 --output figures/native_conductance_teleconnections
```

The zero-based row-major spatial ID runs0--2115 for46x46 (`row*46+column`),
with no channel offset. Hall is the top panel and Pedersen the bottom, axes
off with an anchor marker. Figure3a is Pearson correlation with Hall at that
location;3b uses Pedersen;3c is unsigned jointly explained variance R^2 using
both. Single-anchor scales are[-1,1], joint scale[0,1]; redundant anchors use
a pseudoinverse without double counting. This uses the **physical-S ensemble
covariance**, as in original figure3, not the log covariance of reconstruction
figures4--6. All stored draws are included without PCA truncation or added
measurement-error covariance. Unsupported native training cells and
zero-variance targets are blank; unsupported/zero-variance anchors fail.
Optional --batch-size (state rows,default64), --no-colorbar and --dpi are
unchanged from the WIC/SI12 version. Companion NPZ/JSON records the maps,
field labels, anchor coordinates/correlation, sample count and assumptions;
existing figure3 outputs are protected unless --overwrite is given, which
replaces only the three PNGs and companion NPZ/JSON, not unrelated files or
source runs. Native H/P are not forced proportional;
the fixed-2-keV WIC-derived proxy is, so its two anchors are redundant.

## Train with fully good and masked partial frames

Both `run_wic_si12_pipeline.py` and `run_conductance_pipeline.py` accept:

```bash
--annotations data/partial_image_annotations_46x46.json
```

Append this to your training command and use a **new output directory**.
It replaces CSV selection; do not combine it with `--no-csv-selection`.
Without this option, the existing CSV default is unchanged.

Only completed orbit reviews are used. Native frames match the time axis of
each orbit's annotation source within one second; unmatched and bad frames
are excluded. Untouched converted reviews reference the original conductance
files; newly reviewed orbits reference native CS files. Fully good frames
retain all valid pixels on the whole46x46 grid. Partial
frames use the latest earlier keyframe in the same partial interval; masks
never carry across the fully good interval. Both channels share the manual
mask, intersected with upstream validity and any SZA filter. Excluded pixels
are absent from encoder input, normalization and reconstruction loss, not
inserted as zero-valued observations. Upstream uncertainties/quality remain
unchanged on retained pixels. The intrinsic 33% missing-data cutoff is applied
before manual masking, so an intentional partial mask does not count against it.

Each orbit's source file and the conversion target-grid file must remain
readable for timestamp/grid checks. Untouched converted reviews still need
their original36x36 source, but new native reviews do not. Artifact provenance stores the JSON and source-time
hashes plus per-split full/partial counts (before the intrinsic cutoff).
Held-out figures/browser reload the same masks automatically and reject changed
annotations. Existing trained models cannot acquire these masks without retraining.
Model architecture, loss, uncertainty defaults and artificial-hole options are
unchanged. No source products or original annotations are edited.

For a WIC/SI12 run with native/converted JSON masks and CSV fallback:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes_native_annotations \
  --hybrid-annotations data/partial_image_annotations_46x46.json \
  --epochs 500 --samples 50000 --beta 1.6 --training-holes --seed 1 --device auto
```

Replace `--hybrid-annotations` with `--annotations` for JSON-only selection.
The same options work for `run_conductance_pipeline.py`. Neither option
changes the model, loss or the opt-in uncertainty settings.

## Combine JSON masks with the older CSV selection

Use this **instead of** `--annotations` in either VAE runner:

```bash
--hybrid-annotations data/partial_image_annotations_46x46.json
```

Completed JSON reviews override the whole orbit: good/partial masks are used,
and bad or unmatched frames remain excluded even if CSV would accept them.
An explicit JSON `no_usable` review excludes the entire orbit. For other orbits,
the existing CSV WIC-source-index interval and upstream masks/weights are kept
unchanged. Unfinished JSON reviews do not override CSV. Use `--segments` to
choose a different fallback CSV; the default remains `data/good_orbit_segments.csv`.
JSON-only, hybrid and `--all-frames` options are mutually exclusive.

Use a fresh output directory and retrain. Both JSON and CSV fingerprints and
original timestamps are saved; held-out plots/browser apply the same priority
automatically. Changed annotations are rejected during reload/resampling.
The intrinsic missing-data cutoff and model/loss settings remain unchanged.

On the native smoothed corpus, seed1/default split fractions and33% cutoff,
hybrid uses6881/1203/1117 train/validation/test frames versus CSV4033/487/1101.
The expanded orbit list changes the held-out splits; use matched evaluation
frames for a controlled scientific comparison.

# Presentation: five saved WIC/SI12 draws

`plot_wic_si12_samples.py` makes a clean 2-by-5 figure: corrected WIC on top,
SI12 below, no titles, axes or colourbars. Indices are zero-based and appear
left to right in the supplied order. Each channel uses a shared linear counts
scale across its five panels. Optional `--wic-max` and `--si12-max` set those
scale maxima explicitly; otherwise each row uses its selected-sample maximum.

From the repository root:

```bash
python scripts/IMAGE/vae_pipeline/plot_wic_si12_samples.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --indices 0 100 1000 10000 49999 \
  --output figures/presentation/figure_1.png
```

Only `ensemble_products.npz` is needed. The archive is streamed in chunks,
retaining just the five selected members rather than the whole factor or
ensemble. No training, new latent draws, propagation, original observations
or model loading is performed. PNG, PDF and SVG outputs follow the output
filename extension; default PNG resolution is 300 dpi (`--dpi` overrides it).

# Presentation: mean and joint covariance PCA

`plot_wic_si12_covariance.py` makes figure 2 from all saved FUV ensemble
members: WIC top, SI12 bottom; mean in column 1 and the first four joint PCA
patterns in columns 2--5. Axes and colourbars are absent. Panel text sits
above the WIC row and below the SI12 row: `WIC ensemble mean` /
`SI12 ensemble mean`, then `PC x : y%` using each joint component's explained
variance (the same percentage on both sensor panels, to two decimals).

```bash
python scripts/IMAGE/vae_pipeline/plot_wic_si12_covariance.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --output figures/presentation/figure_2.png
```

The exact physical-count covariance `C = A @ A.T` is accumulated in sample
batches from a temporary disk-backed copy of the saved factor, never a
sample-by-sample covariance. Four leading eigenpairs are calculated exactly.
Each signed pattern is `sqrt(eigenvalue) * eigenvector`: a one-standard-
deviation departure in counts, not mean-plus-pattern and not a positive-only
sample image. Each eigenvector is joint; WIC and SI12 share its sign and rank.
No per-channel standardization is applied, so larger WIC numerical variance
can dominate raw-count PCA. No measurement-error covariance is added.

Means use `viridis`, with 0 to the channel mean maximum. PCA panels use
`RdBu_r`, with zero white and symmetric limits shared across each channel's
four patterns. Use `--wic-mean-max`, `--si12-mean-max`, `--wic-mode-max` and
`--si12-mode-max` to override the maxima; mode maxima define +/- limits.
`--batch-size` (default 1024) controls working memory; `--dpi` defaults to 300.
The script also saves a compact `figure_2_pca.npz` with mean, patterns,
eigenvalues and explained fractions, and prints PC1--PC4 percentages.
Temporary factors are removed automatically; the saved run is unchanged.

# Presentation: teleconnections at a spatial grid ID

```bash
python scripts/IMAGE/vae_pipeline/plot_wic_si12_teleconnections.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --grid-id 750 \
  --output figures/presentation
```

The grid ID is zero-based, row-major and spatial only: `id = row * 46 + column`,
range 0--2115. ID 750 means row 16, column 14; no SI12 channel offset is needed.
The script writes:

- `figure_3a.png`: correlations with WIC at this location.
- `figure_3b.png`: correlations with SI12 at this location.
- `figure_3c.png`: fraction of target variance linearly explained jointly by
  WIC and SI12 at this location (`R^2`, not signed correlation).

Each figure has WIC above SI12, axes off, an anchor marker and a shared
colourbar. 3a/3b use `RdBu_r` from -1 to 1; 3c uses `viridis` from 0 to 1.
`--no-colorbar` removes the colourbar; `--dpi` defaults to 300.
Small companion `figure_3_teleconnections.npz` retains maps and anchor metadata.

All saved draws are used, not just the first four PCs. Batched factor products
avoid a dense covariance. For 3c, `R^2 = r.T @ pinv(R_anchor) @ r`, where `r`
contains the two target-anchor correlations and `R_anchor` is the 2x2 anchor
correlation. This handles channel count scales and redundant anchors without
double counting (`pinv` cutoff 1e-12). It requires no observed values/noise model:
it is an ensemble linear-predictability summary, not a performed assimilation
or independently established physical teleconnection. Undefined correlations
at zero-variance target cells are NaN; zero-variance anchors are rejected.
Temporary factors are automatically removed; input runs/models are unchanged.

# Presentation: held-out half-image covariance reconstruction

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --frame-index 0 --remove-half right \
  --output figures/presentation/figure_4.png
```

WIC above SI12, four columns: held-out truth, input with half removed,
covariance reconstruction, inferred posterior standard deviation in counts.
Axes are off; panel text is above/below the rows as in figure 2. Truth, input
and reconstruction share a linear counts scale within each channel; the
uncertainty column has a separate linear counts scale. Intensity maxima
default to the held-out channel maximum; brighter reconstruction values
saturate the colour scale, but numerical values remain unchanged. Override
with `--wic-max` / `--si12-max`, and `--wic-uncertainty-max` /
`--si12-uncertainty-max` for uncertainty panels. Originally missing
cells remain blank in every panel. The reconstruction displayed in column3
is the full posterior field, not a truth/reconstruction splice. Stdout/JSON
report hidden-half RMSE and truth/reconstruction maxima, including outliers.

Select a zero-based frame in the saved, filtered held-out split with
`--frame-index`; `--split test` is default (`validation` is also available,
never training). `--remove-half left|right|top|bottom` removes the same
geometric half from both sensors; `right` is default. Original source images
and the saved selection CSV must be available. Orbit/time and assumptions
are written to `figure_4_reconstruction.json`; truth, masks, posterior mean/
STD in log space and physical outputs are saved in the companion NPZ.

Recover every saved prior draw, transform each to `log10(max(counts,0)+1)`,
recompute the joint log-space mean/covariance and use PCA modes 1--50
(`--modes 50` is default). This is not log10 of the physical ensemble mean,
not figure 2's raw-count PCA, and not encoder/decoder inpainting. It uses
only retained observations in a coefficient-space optimal-interpolation
solve; removed truth never enters that calculation. Temporary disk-backed
factor conversion and sample-batched exact PCA avoid large ensembles or
observation-space inversions in RAM. Figure2 still defaults to four PCs.

Default `--observation-std 0.1` is an assumed independent log-space error
STD in dex, matching legacy OI; it does NOT use dwic/dsi12. No random
observation noise is added. The physical reconstruction is the back-
transformed conditional median, clipped to0; clipping is reported. Physical
posterior STD uses the exact shifted-lognormal variance formula, not
`10**log_std`. This is formal uncertainty under the truncated Gaussian PCA
prior/error assumptions, not demonstrated calibrated uncertainty, and does
not include neglected PCA modes or model discrepancy. `--dpi` defaults to 300.

# Presentation: reconstruct an entirely missing sensor

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_sensor_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --frame-index 0 \
  --output figures/presentation
```

This produces `figure_5a.png` with all SI12 observations removed (WIC-only
input), and `figure_5b.png` with all WIC observations removed (SI12-only input).
The same held-out pair and exact log-space PCA prior are used in both cases;
the prior is calculated once. Each figure has the same 2x4 layout as figure 4:
truth, available input, reconstruction of both sensors, formal posterior STD
in counts. The removed sensor's input panel is entirely blank; it is not
replaced with zeros as observations. The retained sensor keeps every originally
valid pixel. Original gaps stay blank in displayed outputs.

Use the same `--frame-index` and `--split` as figure 4 for matched examples.
PCA 1--50, assumed 0.1 dex independent observation errors, clipping, physical
median/STD, colour scales and limitations are unchanged. `--modes`,
`--observation-std`, the four intensity/uncertainty scale options and `--dpi`
are available as in figure 4. Each case saves companion reconstruction NPZ
and JSON with observed/missing sensor identities, counts, frame provenance,
RMSE, numerical maxima, clipping and uncertainty assumptions. No VAE decoder
inference, retraining or missing-channel truth is used in conditioning.

# Presentation: coefficient-space variants of figures 4 and 5

Both scripts also support `--prior-space coefficients`. Use a separate output
directory to preserve the existing pixel-PCA figures:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --prior-space coefficients --modes 50 --frame-index 0 --remove-half right \
  --output figures/presentation_coefficients/figure_4.png

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_sensor_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --prior-space coefficients --modes 50 --frame-index 0 \
  --output figures/presentation_coefficients
```

This re-decodes the **stored latent draws** from `ensemble_products.npz` using
the saved spline model, without drawing new samples or retraining. Restore
coefficients from training-standardized units to `log10(counts+1)` units
before computing the joint WIC/SI12 coefficient covariance and its leading
PCA modes. Project the coefficient mean and retained eigenfactors through
the saved spline bases. The existing low-rank OI then implements the legacy
`G Q G.T` reconstruction, where `Q` is truncated in coefficient space, not
pixel space. No physical-count clipping occurs before coefficient PCA.
The decoder only constructs the prior; observations are assimilated using
that covariance, not through encoder/decoder image inpainting.

Defaults stay at 50 modes and 0.1 dex errors for a matched comparison with
the pixel variant. Set `--modes 100` for the legacy mode count. Both variants
retain the same held-out frame selection, masks, plots and posterior physical
median/STD calculations; they do not recreate the legacy in-training example
or its scattered observation mask. Pixel PCA remains the default.
`--decode-batch-size 64` and `--device cpu|cuda|auto` control only coefficient
decoding (default CPU); exact coefficient covariance/PCA remains CPU-side.
One temporary disk-backed coefficient ensemble limits RAM use. The saved
source model and run artifacts are not changed.

Companion NPZ files additionally save the coefficient mean, retained
one-sigma coefficient patterns and spline bases. JSON identifies the prior
space and decoding device. Its explained-variance fraction refers to the
**coefficient covariance**, not pixel-map variance. Recovered physical-draw
clipping fields are null in coefficient mode because those physical draws
are not used; negative posterior physical medians are still clipped/reported.
Formal posterior uncertainty remains uncalibrated. A coefficient-space prior
does not by itself establish better reconstruction or assimilation skill.

# Presentation: decoder-based variants of figures 4 and 5

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --reconstruction-method decoder --frame-index 210 \
  --output figures/presentation_decoder/figure_4.png

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_wic_si12_sensor_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --reconstruction-method decoder --frame-index 210 \
  --output figures/presentation_decoder
```

Fit a latent vector directly through the saved, frozen decoder using ONLY
the retained observations. No encoder, PCA, stored ensemble, hidden truth,
quality-channel input or retraining is used. Fit both remaining halves for
figure4, WIC only for5a, and SI12 only for5b; decode both full sensor maps.
Masking, held-out frame selection, scales and axes-off2x4 layout stay unchanged.
The reconstruction column says `Decoder fit`; the uncertainty column says
`Approx. SD (counts)` to distinguish it from covariance OI.

Minimize `0.5*sum(((decoder_log10(z)-observed_log10)/sigma)**2)` plus
`0.5*lambda*sum(z**2)`. `decoder_log10` restores training standardization to
log10(counts+1), without clipping before fitting. Defaults: sigma0.1 dex,
lambda1,8 starts (zero plus7 independent standard-normal starts), seed1 and
2,000 L-BFGS-B iterations per start. Choose the smallest final objective.
`--observation-std`, `--latent-prior-weight`, `--latent-restarts`,
`--latent-max-iterations`, `--seed` and `--device cpu|cuda|auto` expose these
settings. CPU is default. The latent prior weight is NOT training beta;
both sigma and lambda must be positive. Decoder fitting uses float64 for
optimization/derivatives and leaves the saved model untouched. Covariance
remains the default method; `--modes`, `--prior-space`, and covariance decode
batch size do not apply to the decoder method (JSON records these as null).
Original source images and saved selection still need to be available.

The plotted reconstruction is the decoder at the fitted latent MAP, clipped
to nonnegative counts. It is not a full nonlinear posterior mean or a splice
with observed truth. Approximate latent covariance is
`inv(lambda*I + J_obs.T@J_obs/sigma**2)`, using the decoder's log-field Jacobian
at that solution. Project it to marginal log variances and use the same
shifted-lognormal counts-STD calculation as the OI plots. This local
Gauss-Newton/Laplace approximation ignores other latent solutions,
linearization failure, model discrepancy and calibration; it is NOT an
empirically calibrated prediction error or measurement dwic/dsi12.

NPZ saves truth/masks, fitted fields, local log STD, latent MAP and latent
covariance, with no PCA arrays. JSON records the method, objective, all final
restart objectives/iterations/convergence, selected restart, latent norm,
observed log RMSE, seed, settings and uncertainty assumptions. Nonconverged
best fits are explicitly reported, not silently labelled successful. A frozen
decoder is a restricted model; even full-sensor input need not fit perfectly.

# Presentation: conductance versions of figures 4 and 5

Run from the repository root; both commands take an output **directory**:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --frame-index 210 --modes 50 \
  --output figures/presentation_conductance

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_conductance_field_reconstruction.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --frame-index 210 --modes 50 \
  --output figures/presentation_conductance
```

Hall is the first row and Pedersen the second. `figure_4.png` removes the same
half of both fields (`--remove-half right` by default). `figure_5a.png` removes
all Pedersen; `figure_5b.png` removes all Hall. Columns are held-out reference,
remaining input, covariance reconstruction and formal posterior SD in siemens.
Axes are off; labels are above Hall and below Pedersen. Existing gaps stay blank.
Each figure has companion `_reconstruction.npz` and JSON with masks, frame
identity, predictions, errors, units and assumptions. No retraining is needed.

The frame index is the SAME filtered held-out WIC/SI12 index as the sensor
scripts; `--split test|validation` also matches them. Convert that frame's
already-proton-corrected WIC using the pipeline's fixed-2-keV Robinson adapter.
Use the matching saved `conductance_mean` and `conductance_anomaly_factor` to
recover all ensemble members, transform each with `log10(conductance + 1 S)`,
and calculate the joint Hall/Pedersen pixel covariance/PCA. Defaults are50
modes and assumed independent0.1 dex observation STD (`--observation-std`).
Use only remaining observations in the same Gaussian optimal interpolation
as the sensor figures; hidden values never enter conditioning. Display the
nonnegative physical conditional median and shifted-lognormal conditional SD.
There is no conductance decoder or coefficient-space option here.

**These references are WIC-derived conductance, not native `conductance_cs`
or independent conductance truth.** Conversion after WIC gridding is not
equivalent to upstream conversion before conductance gridding. The saved
baseline has no SI12-derived proton contribution; fixed electron energy makes
Hall and Pedersen proportional, so figure5 is largely a redundant-field test,
not evidence of independent cross-field predictive skill. Posterior SD assumes
the retained PCA prior and exposed log-space error; it is NOT propagated dH/dP
or empirically calibrated model error. Omitted modes/discrepancy are absent.

The same scripts also auto-detect native conductance runs; the preceding
fixed-energy/reference/error caveats apply only to the FUV-derived example.
Native references load H/P directly and default to measured dH/dP as described
in the direct-native pipeline section above.

First three columns share a linear0-to-reference-maximum scale per field;
SD has a separate scale. Optional limits, all in siemens: `--hall-max`,
`--pedersen-max`, `--hall-uncertainty-max`, `--pedersen-uncertainty-max`.
`--dpi` defaults300. JSON reports actual maxima even if colours saturate.
Real50,000-member frame210 verification took about26 s per command and1.5 GiB
peak RAM; one temporary disk-backed factor is used, not sample-by-sample PCA.
Peaks remain too weak on this example; this is not a successful-recovery claim.

# Browse all held-out reference pairs

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/plot_heldout_samples.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --output figures/heldout_conductance
```

Open `figures/heldout_conductance/index.html` in a browser. The clickable gallery
shows every held-out **test** pair; individual PNGs can also be browsed in an
image viewer. Each image gives the full-split frame ID, orbit and timestamp.
Use that frame ID with `--frame-index` in the reconstruction scripts. `frames.csv`
records those IDs, filenames, valid-pixel counts and plotted colour maxima.
Choose an empty output directory; existing files are never overwritten.

Hall/Pedersen is default and uses the SAME fixed-2-keV corrected-grid-WIC
conversion/common WIC/SI12 mask as the conductance figures above, not native
conductance_cs. Add `--fields fuv` to browse corrected WIC/SI12 instead, or
`--split validation` for the validation split. Source products and the saved
artifact/selection must remain available; the ensemble archive is not needed.
No inference, reconstruction, latent sampling or PCA is performed.

By default each field/frame has its own linear zero-to-maximum colour scale,
with colourbars (S or counts). For consistent scales across all pairs, use
`--maxima 10 15` for Hall/Pedersen, or supply the desired WIC/SI12 count limits
with `--fields fuv`. `--dpi` defaults to100 for browsing. Optionally export a
subrange using `--start 200 --stop 250`; stop is exclusive and frame IDs are
NOT renumbered. The full selected split is loaded only once and each figure
is closed immediately. Gallery thumbnails load lazily. Real frames209--211
were verified against the reconstruction reference:4.35 s,755 MiB peak.

# Partial-image annotation

`annotate_partial_images.py` is a resumable graphical annotation tool for the
native46x46 IMAGE conductance_cs or precipitation_cs data. It avoids
per-frame pixel labeling by recording:

- one usable interval per orbit;
- an optional fully good interval inside it;
- a few masks drawn on representative partial frames.

Masks can be drawn either by lassoing the **valid area to retain** or by
dragging a straight divider across the image and choosing which half is valid.
A mask becomes active at its keyframe and is held unchanged for all later
frames until another keyframe replaces it. There is no temporal interpolation.

The tool only reads `data/good_orbit_segments.csv`. It does not alter that
file. New annotations and the current cursor are saved in
`data/partial_image_annotations_46x46.json`. The original36x36 JSON is untouched.
Existing converted reviews are preserved under `legacy_annotation` when an
orbit is opened: its boundaries/masks are aligned to native timestamps, not
copied by frame number. Sparse masks survive even if their original keyframe
was dropped by the CS product. Converted and new native reviews both apply
their classes and drawn masks across the entire46x46 grid. The old-grid border
does not restrict them.

## Run

Use the environment that provides `icreader`:

```bash
mamba activate tConductance
PYTHONPATH=. python scripts/manual_masking/annotate_partial_images.py
```

The default data directory is native smoothed Hardy/Robinson conductance_cs.
All files there are discovered; old overview PNGs do not limit the orbit list.
An optional `--overview-dir` restricts discovery only when explicitly supplied.
To annotate corrected WIC/SI12 instead:

```bash
PYTHONPATH=. python scripts/manual_masking/annotate_partial_images.py \
  --data-dir ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2
```

Conductance channels are Hall, Pedersen, auroral quality, coverage and corrected
WIC from the matching local precipitation_cs file. The channel button cycles
through all five; the mask editor shows all five together. WIC retains its own
finite pixels even where conductance retrieval is invalid. A missing companion
file produces a warning and leaves the original four channels available.
precipitation panels are corrected WIC, SI12, auroral quality and WIC coverage.
Masks can be used by either pipeline on the same grid, matching timestamps.

Every overview and mask-editor panel includes stored SZA contours:90° is white
and solid;105° is magenta and dashed. Enlarged panels label the lines. SZA comes
from precipitation_cs (the matching companion when viewing conductance).
Contours appear only where the level crosses finite stored SZA pixels; missing
regions are not extrapolated. These overlays do not change annotation masks.

To start or resume at a particular orbit:

```bash
PYTHONPATH=. python scripts/manual_masking/annotate_partial_images.py \
  --start-orbit 150 \
  --start-frame 72
```

To try the interface without writing to the main annotation file:

```bash
PYTHONPATH=. python scripts/manual_masking/annotate_partial_images.py \
  --start-orbit 150 \
  --start-frame 72 \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

Closing the window, pressing `Q` or Escape, or clicking **Stop and save**
records the current orbit, page, and frame. The next ordinary launch resumes
there automatically. Boundary changes, mask changes, navigation, and
selection are also saved immediately using atomic file replacement.

The `/tmp` path above is suitable only for trying the interface; an operating
system cleanup or reboot may remove it. The default
`data/partial_image_annotations_46x46.json` is the durable project location. To
resume a temporary trial at its saved cursor, omit the start overrides:

```bash
PYTHONPATH=. python scripts/manual_masking/annotate_partial_images.py \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

## Automatic masking

The adopted automatic-mask workflow is documented in
`scripts/automatic_masking/README.md`. Model2 is the pixel U-Net; Model1 is the
single-frame whole-frame acceptance CNN. The rejected five-frame Bad-veto and
other historical variants are retained only in Git history.

```bash
PYTHONPATH=. python scripts/automatic_masking/prepare_model2_training_data.py --help
PYTHONPATH=. python scripts/automatic_masking/train_model2_unet.py
PYTHONPATH=. python scripts/automatic_masking/apply_model2_unet.py --help

PYTHONPATH=. python scripts/automatic_masking/prepare_model1_training_data.py --output <prepared-model1>
PYTHONPATH=. python scripts/automatic_masking/train_model1_cnn.py --help
PYTHONPATH=. python scripts/automatic_masking/apply_two_stage_masks.py --help
```

The GUI now skips wholly empty image orbits on launch and previous/next
navigation, even if quality/coverage contain values. Their annotations remain
unchanged. Use --show-empty-orbits to inspect them explicitly. Next unreviewed
jumps over completed/no-usable reviews; marking complete moves to the next
unfinished nonempty review. Previous/next still revisit completed reviews.

## Orbit workflow

The four boundaries describe this ordering:

```text
bad | partial | fully good | partial | bad
      ^          ^      ^             ^
   usable     full    full          usable
    start     start   stop           stop
```

Either partial interval may be absent, and the fully good core may be cleared.
The normal workflow is:

1. Set only the outer boundary that needs extending. For partial images after
   an existing fully good sequence, select the last partially useful image and
   click **Set LAST usable**. Do not change **FIRST usable**.
2. Double-click an orange partial frame, or select it and click **Edit selected
   mask**.
3. At the first orange frame, either lasso the valid region or draw a dividing
   line and flip its assignment if necessary. Save the initial keyframe. That
   mask remains active until it is replaced.
4. Add another keyframe only where the shape of the bad region changes enough
   to need it.
5. Click **Mark orbit complete**. Each partial interval must have its initial
   mask keyframe on its first frame.

CSV seeds are mapped through native `source_indices['wic']`; detector frame
IDs are never treated as CS row indices. Use the displayed native frame numbers
when extending an interval, and put the initial partial mask on the first
orange frame. Its mask remains active until a later keyframe replaces it.

If an orbit contains no useful data, **No usable data** requires two clicks
because it clears that orbit's draft boundaries and keyframes. **Undo last
edit** restores the preceding annotation state for the current orbit.

## Mask editor

Use **Lasso valid region** for irregular boundaries. Everything inside the
lasso is valid.

For the common half-image case:

1. use the default **Draw dividing line** mode;
2. drag the boundary across any image panel, or click its two endpoints;
3. inspect the translucent red excluded half in all four panels;
4. click **Flip last line** when the wrong side of the newest line is red;
5. draw another line when another edge needs excluding;
6. use **Undo last line** to remove the newest cut, if necessary;
7. click **Save keyframe**.

Each new line is an additional cut: the final valid region is the intersection
of the valid sides of all lines. Older cuts are white and the newest cut is
cyan. **Clear line cuts** returns the draft to an all-valid image.

All line endpoints, side assignments, and the resulting46x46 mask are stored.
Reopening that exact keyframe restores the complete line set and mask.
Intersecting half-planes produce convex valid regions; retain lasso mode for a
strongly concave or disconnected valid region.

Editing the JSON changes its fingerprint. Older saved-run diagnostic reloads
intentionally reject changed annotations; preserve the JSON used by an older
run if you still need to reproduce its exact selection. Use a fresh output
directory when training with updated annotations.

## Display

- green border: fully good;
- orange border: partial;
- red border: outside the usable interval;
- gray border: boundaries are not yet sufficient to classify the frame;
- blue border: selected frame;
- star in the frame title: exact mask keyframe;
- translucent red pixels: excluded by the propagated validity mask.

The channel button cycles between WIC, Hall conductance, Pedersen conductance,
and the existing weight array. Arrow keys navigate pages and orbits; `E` opens
the mask editor for the selected frame.

The GUI records annotations only. The converted schema2 file can now be used
explicitly by either VAE runner with `--annotations`; the VAE loader intersects
the saved manual masks with product validity and existing measurement weights.

# Corrected-WIC/SI12 VAE baseline

`run_wic_si12_pipeline.py` is the canonical fixed-grid baseline. It reads
schema-1 `precipitation_cs` files through `icreader.load()` and requires all
six coordinated camera fields: corrected WIC, corrected SI13, SI12, and their
uncertainties. The input `wic_corrected` field is already proton corrected;
the workflow never repeats that correction. The accepted production contract
uses the upstream Hardy proton-energy model; the separate downstream electron
propagation uses fixed 2 keV energy.

By default, the runner reads `data/good_orbit_segments.csv` and keeps only the
manually accepted inclusive `good_start:good_stop` interval for each orbit.
Those bounds are mapped through each Product-2 file's `wic_source_index` before
the camera arrays are materialized. This preserves the archive workflow while
remaining valid when regenerated products omit source frames.

The pipeline splits whole orbit files, clips finite negative WIC/SI12 input
counts to zero, and fits a standardized `log10(value + 1)` transform using the
training split only. It trains a partial-convolution encoder with a
tensor-product cubic-spline decoder, samples paired corrected-WIC/SI12 fields,
propagates corrected WIC to Hall and Pedersen at a fixed 2 keV electron energy,
and performs an ensemble-space sparse update. The inverse transform clips
decoded values below the physical zero boundary to zero.

The encoder receives corrected WIC, SI12, and one continuous corrected quality
channel, plus the separate common binary validity mask required by partial
convolution. The accepted Product-2 field is already auroral-interest weight;
it is used directly with channel coverage.

The restored legacy-lineage defaults are 36 latent dimensions, batch size 8,
Adam learning rate `1e-4`, `beta=1`, seed 1, and 1,000 epochs. A
`ReduceLROnPlateau` scheduler monitors training total loss, matching the old
script, with factor 0.5 and patience 10. Validation loss independently selects
the saved checkpoint. The
decoder uses 30-by-30 cubic-spline control points and projects them onto the
46-by-46 product grid. These values are command-line options and are saved in
the run artifact.

The training objective now reproduces the legacy arithmetic. One common weight
map is formed from the smaller of the two channel-specific corrected weights.
Weighted squared error is summed across both fields and the complete batch,
divided by the common weight sum, and multiplied by the full two-field state
size. KL is summed across both batch and latent dimensions. Consequently the
legacy `beta=1` has its original numerical meaning. `dwic` and `dsi12` remain
available in the loaded product but are not part of this baseline loss.

### Opt-in artificial training holes

Add these options to the usual training command, with a new output directory:

```bash
--training-holes --hole-probability 0.5 --hole-size 12
```

Half the training images receive one 12x12-cell square hole on average. Its
center is selected from originally valid cells; WIC, SI12, and the quality
input are hidden together and the encoder mask is updated. Original targets,
loss weights, and loss support are retained, so the hidden observed cells
still contribute to reconstruction loss. Naturally missing cells do not
become targets. Holes that would remove all remaining observations are skipped.

This is disabled by default and works with all uncertainty loss modes. Validation,
test, and plotting inputs are never augmented. The independent hole generator
uses `--seed + 3`; it does not consume the model's random-number stream. Hole
settings are saved in the artifact and checked by `--reuse-artifact`. Keep the
same beta, seed, selection, and loss when comparing against a run without holes.

### Opt-in measurement uncertainty experiments

`--measurement-uncertainty` retains the original median-scaled, capped
inverse-variance experiment. `--gaussian-uncertainty` enables the revised
fixed-model-error experiment instead. `--count-uncertainty` is the separate
count-space experiment described below. The three options are mutually exclusive;
neither is enabled by default. Architecture, encoder inputs, and decoder
outputs are unchanged.

For the revised experiment, set `--model-error-std WIC SI12`, default `0.5 0.5`,
in standardized `log10(counts + 1)` units. These are experimental fixed model
errors, not additional measured detector errors. Each channel uses
`v = measurement_variance + model_error_std**2`. Larger model errors weaken
the contrast between pixel precisions and approach ordinary MSE gradients.

The pixel loss is
`tau**2 * (residual**2 / v + log(v / tau**2))`, with `tau=model_error_std`.
It is a Gaussian negative log likelihood scaled by the fixed reference
`2*tau**2`, with a constant removed. Common quality weights, their sum, and
the legacy state-size multiplier are retained. There is no median-precision
normalization or denominator based on precision sum. Residual attenuation
`tau**2 / v` is bounded by one, and zero measurement variance reproduces the
literal legacy loss for both channels. This deliberately retains the legacy
reference scale for beta comparisons; it is not an untempered probabilistic
ELBO, and it does not guarantee equal total gradient contributions by channel.
The log-variance term is constant with respect to model parameters because
both supplied measurement variance and model-error variance are fixed.

To compare against `wic_si12_smoothed_b16`, keep the same input directory,
seed, beta, epochs, and selection, use a new output directory, and add:

```bash
--gaussian-uncertainty --model-error-std 0.5 0.5 --beta 1.6 --seed 1
```

Those existing runs used no SZA threshold; omit `--minimum-sza` for this
comparison. Save/reload and the standard visualizer support the new loss.
The old `--uncertainty-variance-floor` and `--maximum-relative-precision`
options apply only to the original experiment. The first-order log-space
uncertainty transformation remains approximate, especially near zero-clipped
measurements; this experiment does not resolve that observation-model issue.

### Opt-in count-space uncertainty (E06)

Keep the usual smoothed input, holes, beta, seed and architecture settings,
choose a new output directory, and replace any earlier uncertainty flag with:

```bash
--count-uncertainty --count-model-error-fraction 0.5 0.5
```

The fractions default to `0.5 0.5`; they are experimental model-discrepancy
settings, not validated detector errors. Encoder inputs remain standardized
`log10(counts + 1)` with the quality channel. No encoder, latent (36), spline
(30x30), scheduler, masking, or KL change is made. This mode is OFF by default.

For each channel, fit `s = sqrt(sum(q*y**2)/sum(q))` using ONLY retained training
frames and positive common quality support. These RMS scales and their fitting
policy are saved. Work with `y/s`, decoder mean `m/s`, and ORIGINAL count
uncertainty `sigma/s` together; no log-space delta-method variance is used.
With model-error fraction `f`, the physical model-error standard deviation is
`f*s` counts and total normalized variance is `v = (sigma/s)**2 + f**2`.
This removes the count-unit disparity between channels; it does not guarantee
equal gradients or equal scientific skill.

Positive targets use `f**2 * (((m-y)/s)**2/v + log(v/f**2))`.
Zero targets use `-2*f**2*log(Phi(-m/(s*sqrt(v))))`: an event indicating the
unclipped measurement was at or below zero, not an exactly known zero source.
The stable Gaussian log-CDF uses [PyTorch `log_ndtr`](https://docs.pytorch.org/docs/stable/special.html#torch.special.log_ndtr).
The common quality-sum denominator and full two-field state-size multiplier
remain, as does summed batch/latent KL. Positive-cell density constants are
removed using a fixed reference. This is a reference-scaled likelihood, not
an untempered ELBO. Its count-space residual scale differs from the legacy
log-space objective: do not compare total/reconstruction losses across modes
or assume beta=1.6 gives identical KL pressure. Keep beta fixed for the first
controlled screen, then inspect reconstruction/gradient and KL diagnostics.

The internal pre-clipping mean is the **unclipped** inverse `10**log_value - 1`.
It can approach -1 count, allowing a gradient below the physical-zero boundary.
Exported physical fields still use the existing zero-clipped inverse. This is
a corrected-count observation model, not a nonnegative raw photon-rate model.

Important approximation: native corrected WIC is clipped upstream BEFORE CS
area averaging; SI12 is clipped by this loader after binning. All observed zeros
are treated as censored, because the exact pre-clipping detector values/events
are not in this ingestion contract. A censored Gaussian on the CS product is
NOT the exact likelihood of an average of clipped native pixels, and positive
CS WIC can retain clipping bias. Background subtraction, proton correction,
spatial smoothing and shared detector errors also preclude a raw Poisson or
fully independent-error claim. The supplied uncertainties are used as marginal
standard deviations; cross-pixel/channel measurement covariance is not fitted.

Saved-artifact reuse checks this loss mode, discrepancy fractions, transform
and refitted training count scales. The normal visualizer and frozen sparse
conductance benchmark support these runs. Its `training_effective` log-RMSE
panel is labelled **count attenuation (diagnostic only)**, not the count NLL.
Use neutral/quality held-out error, zeros/intensity strata, prior dispersion,
and E08 point skill/interval coverage to decide whether to adopt this mode.
Changing this loss does not add independent detector noise to prior samples.

Explicit uncertainty settings now require their matching mode: count fractions
require `--count-uncertainty`, `--model-error-std` requires `--gaussian-uncertainty`,
and precision floor/cap/scale settings require `--measurement-uncertainty`.
Unmatched settings fail instead of being silently ignored.

### Reconstruction bottleneck: encoder, optimized latent, free splines (E04)

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/diagnose_reconstruction_bottleneck.py \
    ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
    --output figures/vae_runs/e04_bottleneck_holes
```

Defaults: 32 orbit-balanced training and 32 validation frames, seed 1,
1,000 Adam steps per latent initialization, three starts (encoder mean plus
two standard-normal starts), learning rate 0.03, batch 8, `--device auto`.
Use a fresh output directory. A small check is `--frame-count 2 --steps 100
--starts 2`; a validation-only check uses `--splits validation`.

All methods use identical targets, common masks, auroral-quality weights and
the saved training transform. The saved model must use the legacy loss and
partial-convolution/spline architecture. Encoder reconstruction is `decode(mu)`
with full input. Latent optimization changes ONLY the latent vectors, without
KL, prior constraint, uncertainty weighting or model updates, retaining the
best joint WIC/SI12 fit per frame across every step/start, including the initial
encoder fit. Free coefficients are solved by weighted double-precision QR
least squares using the model's exact saved basis and coefficient-axis order;
`--spline-rcond 1e-10` controls numerical rank, not a ridge penalty.

Outputs: `reconstruction_rmse.png`, `reconstruction_examples_train.png`,
`reconstruction_examples_validation.png`, convergence plots for both splits,
`metrics.json`, `selection.json`, and compact `comparisons.npz` arrays.
Examples show positive-quality supported cells with common per-row colors
in log10(counts+1). RMSE is in standardized-log coordinates, without display
clipping. Metrics also record all-valid errors, numerical spline ranks, latent
norms, winning starts and last-window convergence. Reuse `--selection
path/to/selection.json` for identical frames in another matched model; frame
identity, transform and selected target/mask/quality hashes are checked.

The optimized-latent and free-spline columns use the TARGETS: these are fitting
oracles, not ordinary held-out reconstruction or covariance assimilation.
Finite multi-start results do not prove the global decoder fitting floor;
inspect convergence and try more steps/starts if necessary. Zero-quality/gap
extrapolation is not constrained by free fitting. Model parameters, buffers
and artifact hashes are verified unchanged; no prior ensembles are loaded.
Deterministic subset retraining is a separate follow-up, not part of this script.

### Saved-model loss/gradient audit (no retraining)

Compare matched saved models on the same validation frames:

```bash
PYTHONPATH=. OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 python scripts/IMAGE/vae_pipeline/diagnose_loss_balance.py \
    /path/to/wic_si12_smoothed_b16_holes \
    /path/to/wic_si12_smoothed_b16_holes_mes \
    /path/to/wic_si12_smoothed_b16_holes_cmef \
    --output figures/vae_runs/loss_audit
```

Choose a fresh output directory. This CPU diagnostic defaults to all validation
frames, batch size 8 and posterior-noise seed 1001. Use `--max-frames 8` for a
smoke check; it selects a seeded subset rather than the first frames. Saved
splits, transforms, architecture, beta and data provenance must match.

`native_gradient_budget.png` splits absolute output-gradient contributions
between WIC/SI12 and zero/faint/middle/bright observations. Its coordinates are
the decoder's standardized logs, not raw counts. Faint means positive WIC <=50
or SI12 <=1 count; bright uses each channel's quality-weighted positive 90th
percentile. `measurement_gradient_change.png` measures the relative change in
the shared encoder gradient when measured variance is set to zero (including
direction changes). `allowance_and_KL_sensitivity.png` separates smaller count
error fractions from their coupled loss-temperature effect. Full numeric
results, per-batch norms, settings and hashes are in `metrics.json`.

The audit uses eval-mode full inputs and one identical seeded posterior draw
per batch/model; it is not the exact train-mode/augmented optimization. Native
and counterfactual losses share each model's frozen predictions and encoder
Jacobians. Compare counterfactual gradients within a model, not absolute norms
between independently trained models. Fixed-reference cases are diagnostic
only, not a new training option. Smaller fractions are not demonstrated skill
improvements. No optimizer, prior ensemble, training change or detector noise
injection is involved; saved model hashes are checked unchanged.

The fuvpy Product-1 Tukey weight approaches zero where the background method
cannot fit the data, which commonly identifies aurora. The accepted Product-2
corpus explicitly stores the inverted auroral-quality field, so zero means
background-like and one means auroral-interest. Product-2 ingestion uses
`method_quality_weight` directly and multiplies it by WIC or SI12 channel
coverage. The loader now requires metadata that explicitly describes this
`1 - dgweight` orientation; it rejects the older ambiguous corpus rather than
guessing from the field name. Only the legacy raw `w` field requires `1 - w`.

Frames follow the legacy sparsity rule: if more than 33% of common WIC/SI12
pixels are intrinsically invalid before optional SZA masking, the frame is
excluded. The threshold can be changed with `--maximum-missing-fraction`, but
`0.33` is the compatibility default. Compare physical-space held-out residuals,
sample calibration, covariance, and reconstruction skill in addition to the
optimization loss.

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --output outputs/wic_si12_baseline \
  --samples 1000 \
  --minimum-sza 105 \
  --observation-mode joint
```

This uses the 1,000-epoch restored default. Add `--epochs 20` only for a short
diagnostic run, not as the canonical training setting.

If training completed and `wic_si12_vae.pt` was saved but a later sampling or
propagation stage failed, rerun the same command with `--reuse-artifact`. The
runner verifies the saved orbit split, seed, latent dimension, SZA/manual
selection, data version, and legacy directory before skipping training. It
then regenerates `ensemble_products.npz` and `summary.json` from the saved
model. This is a recovery option, not a way to reuse a model with different
scientific selection settings.

New log-transform artifacts clip decoded WIC and SI12 below physical zero to
zero during inverse transformation. The older signed-asinh convolutional
artifacts remain loadable; for those artifacts only, the prior roundoff
recovery still records approximately `1e-6`-count WIC boundary corrections and
rejects materially negative corrected WIC.

Input clipping is not silent. Every run emits a warning and stores finite,
technically valid, and clipped counts under `input_negative_clipping` in both
the artifact provenance and `summary.json`. The large negative-SI12 audit in
the project notes applies to the older unsmoothed corpus; regenerated products
must be assessed from their own saved diagnostics.

`--minimum-sza 105` applies an inclusive, per-pixel night-side filter after
the existing CSV/source-index frame selection: cells with `SZA < 105°` or
nonfinite SZA become invalid and receive zero training weight. The common
33% frame limit is evaluated on intrinsic camera validity before this
scientific mask, so the SZA filter does not itself reject nightside frames.
Omit the option to reproduce the original no-SZA-threshold baseline. The
threshold is recorded in both model configuration and data provenance.

To use SZA as the only scientific selection criterion and bypass
`good_orbit_segments.csv`, add `--no-csv-selection` (the older equivalent name
is `--all-frames`):

```bash
python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --output outputs/wic_si12_sza105 \
  --minimum-sza 105 \
  --no-csv-selection
```

This still enforces finite values, nonnegative uncertainties, method validity,
positive coverage, and positive training weight. Frames with no qualifying
SZA pixels are removed before the six camera fields are materialized. Orbit
files with no qualifying frame are skipped and listed by split under
`sza_excluded_files` in the saved data provenance.

The current copied corpus contains 157 of the 173 manually accepted orbits.
With the saved split, `SZA >= 105`, and the 33% intrinsic-missing limit, it
provides 3,437 training, 887 validation, and 372 test frames.

Pass a different annotation CSV with `--segments`. `--all-frames` bypasses
manual selection and is intended only for bounded diagnostics; using it on the
complete corpus will require substantially more memory.

Use `--observation-mode hall` or `--observation-mode pedersen` for the
single-channel sparse-observation experiments. The output directory contains:

- `wic_si12_vae.pt`: model state, transform, histories, split identifiers,
  grid coordinates, upstream product metadata, and training configuration;
- `ensemble_products.npz`: FUV and conductance ensemble means and low-rank
  anomaly factors plus the sparse posterior mean and uncertainty; and
- `summary.json`: sample, physical-propagation, and validation provenance.

The low-rank anomaly factors are the default covariance representation. A
dense 4232-by-4232 covariance need not be materialized for reconstruction.
The propagated covariance represents latent field variability at fixed energy;
detector uncertainty remains available for weighting and observation-error
models but is not silently injected into the latent ensemble.

## Comparing the legacy and rerun inputs

The legacy `precipitation_IR_hardy_weighted` products can be compared to the
current Product-2 corpus without changing either upstream repository. Product
2 remains authoritative for orbit selection and SZA. The comparison keeps
only same-named orbit files and exact common product timestamps, applies the
same Product-2 `SZA >= threshold` mask to both versions, and uses the legacy
auroral-interest weight `1 - w` as the legacy VAE quality weight. The legacy
`w` is background-fit oriented; the accepted Product-2 method weight is already
in the opposite, auroral-quality orientation.

VAE artifacts created before the `1 - w` correction used the opposite loss
orientation and are not valid auroral-model results. Retrain them. Existing
direct comparison maps of the physical camera fields are unaffected, but
quality-weight statistics must be regenerated.

First generate a streaming field-level report. This reads one orbit pair at a
time and writes `paired_input_comparison.json` plus mean legacy, rerun, and
difference maps:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/compare_legacy_current_inputs.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --legacy-directory ~/Dropbox/work/data/IMAGE_FUV/precipitation_IR_hardy_weighted \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/input_comparison_sza105 \
  --minimum-sza 105 \
  --no-csv-selection
```

Then train matched current and legacy VAE runs. Both commands use the current
directory as the positional input because it supplies the common orbit list,
timestamps, and SZA. Change only `--data-version` and `--output`; keep all
model, split, and seed settings identical:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --legacy-comparison ~/Dropbox/work/data/IMAGE_FUV/precipitation_IR_hardy_weighted \
  --data-version current \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_paired_current_sza105 \
  --minimum-sza 105 --no-csv-selection \
  --samples 1000 --device auto

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic0p8_si120_si131p2 \
  --legacy-comparison ~/Dropbox/work/data/IMAGE_FUV/precipitation_IR_hardy_weighted \
  --data-version legacy \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_paired_legacy_sza105 \
  --minimum-sza 105 --no-csv-selection \
  --samples 1000 --device auto
```

The paired experiment measures the net change between the two product
pipelines. It must not be described as a pure fuvpy effect: source matching,
detector-to-CS reduction, coregistration, uncertainty propagation, and
validity handling also changed between the legacy and current products.

## Visualizing a completed run

Generate the legacy-style model and covariance diagnostics without retraining:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py \
  outputs/wic_si12_baseline
```

By default, figures and `diagnostics.json` are written under the run's
`figures/` directory. Use `--output` to write elsewhere, `--examples` to
change the sample-mosaic size, or `--anchor ROW,COLUMN` to select the common
correlation probe. The default probe is the maximum-variance corrected-WIC
pixel for FUV and Hall pixel for conductance.

Large saved ensembles are processed one product family at a time. Only the
requested mosaic members are reconstructed, variances are calculated without
a squared-factor temporary, and leading modes use a deterministic randomized
SVD when both covariance orientations are large. `--spectrum-modes` controls
how many leading modes are retained; the cumulative spectrum is therefore the
captured fraction of total variance and need not end at one. A smaller value,
for example `--spectrum-modes 8`, gives a faster overview. The real 50,000-
sample archive completed with this path at about 2.2 GiB peak resident memory.

New run archives retain float32 latent samples and anomaly factors. Sampling
is decoded and inverse-transformed batch by batch into preallocated arrays,
without concatenating a second complete ensemble. Storage and propagation
still grow linearly with sample count, so a 500,000-sample run remains a large
disk and memory job even though plotting no longer constructs a covariance
matrix whose size is quadratic in the number of samples.

The figures cover training history (including the recorded learning-rate
schedule), decoded FUV and propagated-conductance
samples, ensemble mean and standard deviation, joint-channel correlation
maps, leading covariance modes, covariance spectra, and the sparse-update
prior and posterior. Runs produced by the current runner also store the
sampled latent vectors and sparse-validation truth and observations, enabling
latent-draw and truth/error reconstruction figures. Older compatible runs
remain visualizable but cannot recover values that were not saved originally.

To diagnose whether variability is lost by the reconstruction or by sampling
the unit-Gaussian prior, reload the saved held-out split and compare its
observations with `decoder(mu)`, one seeded posterior draw per frame, and an
equal number of prior draws:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py \
  outputs/wic_si12_baseline \
  --posterior-prior-only \
  --posterior-device auto
```

This lightweight path deliberately does not load the potentially large saved
prior anomaly factors. It writes physical moment and distribution comparisons,
mask-matched normalized-log covariance spectra, latent-dimension activity, and
example fields, plus `posterior_prior_diagnostics.json`. Observation-conditioned
examples contain only truth, `decoder(mu)`, and posterior draws; unrelated
unconditional prior draws are displayed in a separate gallery.

The same command also performs a controlled held-out inpainting test. For every
test frame it removes a fixed square centred on the brightest valid joint
log-WIC/log-SI12 signal, clears both transformed channels and the quality input
there, marks the patch invalid for partial convolution, and decodes the
posterior mean. `heldout_inpainting_examples.png` compares truth, damaged input,
the ordinary full-input reconstruction, the damaged-input reconstruction, and
its error. The default 12-by-12 patch and three displayed examples can be
changed with `--inpainting-patch-size` and `--inpainting-examples`.

For the E02 random-patch comparison, evaluate both saved models without training:

```bash
for run_name in wic_si12_smoothed_b16 wic_si12_smoothed_b16_holes; do
  PYTHONPATH=. python scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py \
    ~/Dropbox/work/data/IMAGE_FUV/vae_runs/"$run_name" \
    --posterior-prior-only --posterior-split validation --spectrum-modes 8 \
    --inpainting-policy random --inpainting-seed 0 --inpainting-patch-size 12 \
    --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/"$run_name"/figures_validation_random
done
```

Random centres are sampled uniformly from originally valid cells and the
square is clamped at image edges. Sampling does not depend on brightness or
training seed; one patch is applied per frame. Use the same split, size and
patch seed for both models. The JSON saves frame identities, bounds, removed
mask SHA256 and per-frame error sums/counts for paired checks. Default policy
remains `auroral`. A patch that removes all observations raises an error; use
a smaller size rather than silently changing the sample. Random examples are
still displayed by removed-signal strength; aggregate metrics use every frame.

All four distribution ensembles use the identical held-out observation mask.
Because the observations remain incomplete, the covariance plot is explicitly
a positive-semidefinite,
coverage-adjusted diagnostic: missing centered residuals are zero and each
state element is scaled by its available sample count. It is not presented as
an exact pairwise-complete covariance estimate. Use
`--posterior-minimum-coverage` to change the default 10% support threshold.

## Reusable sparse conductance reconstruction test

This tests the scientific covariance endpoint, without encoding held-out images
or fitting latent vectors. It follows legacy covariance/eigenmode optimal
interpolation, now applied to sampled **conductance** fields on the current grid.
It does not reproduce the old coefficient-space truncation convention.

Compare the existing hole and no-hole models without retraining:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/test_sparse_conductance.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16 \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --output figures/vae_runs/sparse_conductance_comparison
```

The first run supplies all validation frames and their original valid masks;
permanently unsupported training locations are excluded. A saved `benchmark.npz`
freezes reference conductance, frame identities, observations and injected noise
for all models. Reference conductance comes from corrected WIC at fixed 2 keV,
not independent conductance truth. Never-observed locations are not scored.

Defaults: 100 retained covariance modes, `log10(conductance + 1 S)`, observation
noise 0.1 dex, and 10/30/100 random spatial sites with both Hall and Pedersen
observed at each site. Observations are noisy, not exact reference subsamples.
The same sites are nested across counts; seed 0 fixes sites and noise. Use
`--geometry track` for a narrow meridional grid track, `--observation-mode hall`
or `pedersen` for single-field observations, and `--modes` for rank sensitivity.
`--state-space physical --observation-std 0.5` instead tests physical covariance
with 0.5 S noise. State-space/noise changes require a new benchmark fixture.
`--max-frames 8` is a seeded smoke subset; omit it for the full split.

Reuse the exact same benchmark with a future model:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/test_sparse_conductance.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/NEW_MODEL \
  --benchmark figures/vae_runs/sparse_conductance_comparison/benchmark.npz \
  --output figures/vae_runs/sparse_conductance_NEW_MODEL
```

When reusing a fixture, its saved creation settings take precedence over CLI
creation options. The new model must share the saved orbit splits, scientific
selection, grid and propagation. Changing the training seed currently changes
orbit splits, so it is rejected unless the same splits were explicitly retained.
Existing result directories are protected; choose a new output directory.

Outputs: `sparse_skill.png`, three reconstruction examples per model, and
`metrics.json` with frame-level error sums, bias, negative predictions, marginal
95% interval coverage, mode truncation and provenance. Example columns show
reference, noisy observations, prior, reconstruction and error; cyan markers
show observation sites. Scores exclude observed **spatial sites from both
fields**, also in single-field tests. The prior is the no-assimilation control;
diagonal covariance gives the same unobserved-cell estimates and variances.
Log-space inverse transforms give medians, not physical posterior means.
Negative reconstructed values are reported, not silently clipped. Future models
can be compared without regenerating held-out data or changing observations.

Only the conductance factor is loaded. Log-space members are transformed in
batches into a temporary disk-backed array; large modes use randomized SVD,
not a sample-by-sample covariance matrix. Models are processed sequentially.
This still requires space for the saved factor in RAM and a similarly sized
temporary array on disk. Keep observation errors R separate from prior covariance.

## Matched deterministic subset overfit (E04)

No new beta-zero pipeline run is needed. Use the existing legacy-loss model
and the exact training selection from the three-way E04 diagnostic:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/diagnose_latent_capacity.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --selection figures/vae_runs/e04_bottleneck_holes/selection.json \
  --output figures/vae_runs/e04_matched_overfit_holes
```

The script trains a SEPARATE model from scratch, using the saved transform,
exact masks/quality, spline basis and architecture. It defaults to the saved
latent size (36 here), not a capacity sweep. Adam defaults are 500 epochs,
batch 8, LR 1e-4 and seed 1, with automatic GPU selection. Training is
deterministic `decode(mu)`, beta=0, no latent sampling, no input holes and no
scheduler. These are diagnostic settings, not production-default changes.
Explicit dimension sweeps remain optional; do not run one for this control.

`--selection` overrides frame-count/seed for frame selection and requires the
paired E04 `metrics.json` and `comparisons.npz` beside it. It checks exact
frame/data identities and source-model hash before fitting, reusing the
optimized-latent and free-spline baselines. Without it, a seeded orbit-balanced
subset and new free-spline baseline are used, not the old linspace selection.

The best checkpoint minimizes eval-mode loss on these SAME TRAINING frames;
the last epoch is not blindly selected. `overfit_convergence.png` records both
train-mode batch averages and full-subset eval loss, which differ in BatchNorm
mode and reduction. `overfit_comparison.png` and
`overfit_reconstruction_examples.png` compare the source encoder, both E04
oracles and the overfit reconstruction on identical support. Full metrics and
epoch history are in `latent_capacity_sweep.json`; compact predictions are in
`overfit_comparisons.npz`. The separate best model is saved as
`deterministic_overfit_latent36.pt`, explicitly marked as diagnostic, not as a
new `wic_si12_vae.pt`. Source parameters/buffers and artifact hashes are
verified unchanged. Existing output directories must be empty.

This is training-set fitting ability, not held-out generalization or prior
covariance skill. A failed finite training run does not prove an architectural
limit. Inspect convergence before deciding between optimization and capacity.

### Opt-in decoder spatial-stage test (E05)

Compare 30x30 decoder features with the completed latent-36, 18x18 control:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/diagnose_latent_capacity.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_smoothed_b16_holes \
  --epochs 2000 \
  --decoder-feature-size 30 \
  --selection figures/vae_runs/e04_bottleneck_holes/selection.json \
  --overfit-control figures/vae_runs/e04_matched_overfit_holes_2000 \
  --output figures/vae_runs/e05_matched_overfit_decoder30_2000
```

Use a fresh output folder. This keeps latent 36, encoder 6x6 features, 30x30
cubic splines and parameter count unchanged. Only the second upsampling stage
changes: features are enlarged from 9x9 to 30x30 BEFORE the second convolution,
instead of processing at 18x18 then enlarging to 30x30 by adaptive averaging.
The latter averaging is identity at the experimental resolution. The default
remains the legacy 18x18 stage; the production runner has no new CLI setting.

`--overfit-control` adds the completed deterministic 36-latent control to all
comparison figures and its loss as a convergence reference. It checks exact
data/frame/source identities, training budget, seed, optimizer settings and
control architecture before training. A differently shaped latent-72 control
or a different epoch budget is rejected. The JSON records control report hash,
device and metrics; CPU/GPU results may differ slightly. Neither control nor
source is retrained or overwritten.

The candidate is labelled with its decoder resolution; its saved checkpoint
is `deterministic_overfit_latent36_decoder30.pt`. Configuration records the
opt-in stage so reload uses the correct spatial computation, despite identical
weight shapes. Old default-model configurations still load unchanged. As with
E04, this is training-set fitting ability, not a production setting or a target
to reproduce counting noise. Generalization and E08 covariance tests are still
required before adoption.

## Native conductance prior representativeness

`validate_conductance_prior.py` checks the saved standard-normal latent prior
against the aggregate encoded distribution and physical retrieval population.
It reloads the exact saved training/test selection and annotations, reads the
original paired ensemble members without loading the entire factor, and keeps
models/data unchanged. From the repository root:

```bash
MPLCONFIGDIR=/tmp/icanalyzer-prior-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python scripts/IMAGE/vae_pipeline/validate_conductance_prior.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_annotated \
  --output /tmp/icanalyzer_prior_validation_20261012 \
  --seed 1012 --threads 4 --prior-repeats 3
```

CPU execution uses the saved model; no training or artifact regeneration.
Outputs: `validation.json`, per-orbit CSV, train/test encoder parameters,
and four PNG/PDF pairs for latent shape, physical distributions, mean/std maps
and spatial dependence. The directory is a dedicated diagnostic output;
rerunning replaces its derived files. Original model/ensemble/annotation hashes
and input-file stats are recorded and checked unchanged after execution.

Primary metrics give each orbit equal weight, normalize physical pixel area
within each frame and use identical masks across reference and generated fields.
Frame-weighted latent moments, positive-quality physical summaries, complete-
orbit bootstrap intervals and repeated prior draws are explicit sensitivities.
Exact aggregate latent covariance includes encoder conditional variance.
The physical reference is an upstream retrieval, not true noise-free conductance.
Signal-dependent quality and missingness can bias conditional comparisons; the
quality deficit alone is not an unconditional-prior calibration test.

The 2026-10-12 run finds latent directional mismatch and compressed Hall tails
relative to training/held-out retrievals. Detailed results/limitations are owned
by `vault/02_Algorithm/Native Conductance Prior Validation.md`. Keep these
results separate from latent-radius coverage, event-conditioned inference,
Hardy-electron validation and scientific model adoption.

## Per-image conductance covariance

`validate_event_conductance_covariance.py` holds each event image's encoder
mu/logvar fixed, draws paired physical Hall/Pedersen fields and forms the full
joint covariance in S². It uses the unchanged native model and saved test
selection. Default cases are chronological-middle images from all six retained
held-out orbits; no reconstruction outcome selects the images.

```bash
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/icanalyzer-event-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python scripts/IMAGE/vae_pipeline/validate_event_conductance_covariance.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_annotated \
  --output /tmp/icanalyzer_event_covariance_20261012 \
  --draws 4096 --seed 12104 --threads 4
```

Encoder input retains saved normalization, paired masks and the quality
channel; never-trained cells are excluded from both input and output. The test
samples q(z|one image), not the aggregate posterior or N(0,I) population prior.
It also evaluates an unconditional reference, one independent seed and one
12x12 auroral-patch removal control. CPU sampling never adds retrieval-error
or decoder-residual noise and never updates the model.

Each `event_XXXX.npz` contains the 3362x3362 dense `covariance_S2`, normalized
`anomaly_factor_S`, 46x46 paired mean/STD/5th/95th percentile maps, state indices,
encoder parameters and input/reference maps. State order is Hall supported
cells then Pedersen supported cells. `state_flat_indices` map that state into
C-order (2,46,46). Unsupported map entries are NaN, not known zeros. Covariance
is float64; stored factors are float32. Recover original supported samples as
`mean_flat[state_indices] + sqrt(draws-1)*factor[:,member]`; random linear
combinations of factor columns do not preserve the empirical non-Gaussian law.
Large matrices remain in the specified output directory (~800 MB for six cases).

Summary JSON/CSV and three PNG/PDF pairs report event dependence, convergence
and comparisons. Full paired covariance is PSD by construction, but need not
be invertible. Finite sampling still affects off-diagonal estimates; use a
factor or justified regularization when an inversion eventually requires it.
The original input's containment in decoded intervals is a reconstruction
check, not calibrated physical confidence coverage. Detailed results are owned
by `vault/02_Algorithm/Event-Conditioned Conductance Covariance Test.md`.

### Teleconnection sampling check

The original correlation figure uses **4096 fresh draws for both rows**, not
the saved 50k ensemble for the prior. The following repeat checks matched
4096/16384/50000 counts, seeds 12104/12105 and three fixed anchors for the
unconditional reference and the first two selected events (orbits 99/106).
It streams only variances and six covariance rows, avoiding full 50k field
ensembles or new dense covariance matrices. Run the event command above first
if its scratch archives are unavailable.

```bash
OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/icanalyzer-event-mpl PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. \
python scripts/IMAGE/vae_pipeline/validate_event_correlation_convergence.py \
  --run-directory ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_annotated \
  --events /tmp/icanalyzer_event_covariance_20261012 \
  --output /tmp/icanalyzer_event_correlation_20261012
```

NPZ files store mean, variance, selected covariance rows and correlations at
each count. Undefined zero-variance correlations remain NaN. JSON distinguishes
the original anchor from the three-anchor pooled metrics. Small report and
two figure pairs are archived in
`figures/validation/native_event_correlation_20261012/`. More samples preserve
the weaker conditional teleconnections in the two tested events; numerical
convergence does not establish their physical suitability for Lompe.

`check_event_representation.py` audits whether the selected event encodings
and reconstructions are unusual, using the cached prior/event diagnostics:

```bash
OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
python scripts/IMAGE/vae_pipeline/check_event_representation.py \
  --prior /tmp/icanalyzer_prior_validation_20261012 \
  --events /tmp/icanalyzer_event_covariance_20261012 \
  --output figures/validation/native_event_correlation_20261012/event_representation.json
```

Training-only moments whiten encoder means. Nearest-neighbor training
references exclude the query's whole orbit, avoiding trivial adjacent-image
matches. Empirical distance/reconstruction ranks are descriptive, not latent
density or OOD probabilities. Reconstruction is deterministic decoder(mu),
not the average physical field over posterior draws. Both prior and event
ensembles use the same trained VAE; no separate conditional model is fitted.

### Repeat all diagnostics for the hybrid run

The2026-10-12 hybrid rerun evaluates the exact retained training/test populations,
one midpoint image from each of19 held-out orbits, and50000-draw correlation
checks for the first two predetermined events. Run from the repository root in
`tConductance`; each command completes before its dependent command begins.

```bash
export PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1
export OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLCONFIGDIR=/tmp/icanalyzer-event-mpl

python scripts/IMAGE/vae_pipeline/validate_conductance_prior.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid \
  --output /tmp/icanalyzer_hybrid_prior_validation_20261012 \
  --seed 1012 --threads 4 --prior-repeats 3

python scripts/IMAGE/vae_pipeline/validate_event_conductance_covariance.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid \
  --output /tmp/icanalyzer_hybrid_event_covariance_20261012 \
  --draws 4096 --seed 12104 --threads 4

python scripts/IMAGE/vae_pipeline/validate_event_correlation_convergence.py \
  --run-directory ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid \
  --events /tmp/icanalyzer_hybrid_event_covariance_20261012 \
  --output /tmp/icanalyzer_hybrid_event_correlation_20261012

python scripts/IMAGE/vae_pipeline/check_event_representation.py \
  --prior /tmp/icanalyzer_hybrid_prior_validation_20261012 \
  --events /tmp/icanalyzer_hybrid_event_covariance_20261012 \
  --output /tmp/icanalyzer_hybrid_event_covariance_20261012/event_representation.json
```

Prior validation saves `inference.npz` for the representation audit. Correlation
convergence reads the primary anchor and first two event orbits from the event
report; `--orbits ORBIT1 ORBIT2` selects two other saved cases. Model/source
evidence must match the event report. Small figures/reports are archived under
`figures/validation/hybrid_conductance_validation_20261012/`; full dense event
covariance/factors require several GB in scratch. Patchy conditional structure
persists in this run, including the two larger-sample repeats; physical spatial
validation remains open. Earlier annotated diagnostics retain their own scope.

### Recover Kp coverage of the retained hybrid inputs

Kp labels/range are not saved in the VAE checkpoint or ensemble, but original
conductance source files store GFZ `Kp` and `Kp_interval_start`. Reload exact
saved train/validation/test selections and join by original UTC image times:

```bash
PYTHONPATH=. PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=4 \
MPLCONFIGDIR=/tmp/icanalyzer-event-mpl \
python scripts/IMAGE/vae_pipeline/audit_run_kp.py \
  ~/Dropbox/work/data/IMAGE_FUV/vae_runs/conductance_smoothed_b16_holes_hybrid \
  --output /tmp/icanalyzer_hybrid_kp_coverage_20261012
```

`kp_coverage.json` stores split-specific ranges, frame and three-hour-interval
quantiles, activity-bin counts and source provenance. `retained_frame_kp.csv`
contains every exact retained-image match, including original source row.
Training Kp spans0.667–6.667, median3.333, with19.78% of frames atKp≥5. All9201
retained images across splits have finite Kp. Archived report/CSV live under
`figures/validation/hybrid_conductance_validation_20261012/kp/`; detailed
interpretation belongs to `vault/02_Algorithm/Hybrid Training Kp Coverage.md`.
Generated members have no Kp label: conductance quantiles are not Kp quantiles.
