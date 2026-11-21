# 184x184 WIC/SI12 CVAE Transition Plan

Last reviewed: 2026-11-21
Status: Hierarchical architecture implemented; scientific pilots not started

## Purpose

Prepare the paired corrected-WIC/SI12 conditional VAE for the new Product-2
Cubed-Sphere grid `image_apex_130km_184x184_50km_v1`, then make a controlled
decision about a higher-capacity encoder, latent representation and decoder.
The transition must preserve the existing 46x46 workflow long enough to make
matched comparisons and read historical artifacts.

The publication objective remains a conditional joint WIC/SI12 distribution
whose samples retain useful spatial and cross-sensor covariance for later
conductance-prior construction. Pixel reconstruction alone is not the model
selection target.

## Verified starting evidence

- `or_0085.nc` is readable through `icreader` as precipitation CS with shape
  `122 x 184 x 184` and exposes every field required by the active paired
  cache: corrected WIC and uncertainty, SI12 and uncertainty, method validity
  and quality, sensor coverages, Kp, SZA and DZA.
- The new grid has the same projection, radius and outer edges as the annotated
  46x46 grid. Pixel-centre coordinates map exactly as
  `new = 4 * old + 1.5` in both axes.
- The converted annotation contains 121 orbit records, 437 keyframe masks and
  660 line segments. All temporal metadata and review states are preserved.
- All 382 line-set keyframes reproduce their stored 46x46 masks exactly when
  rasterized on the old grid. The lines therefore provide valid higher-
  resolution boundary intent; rasterizing them on the 184x184 grid is not a
  new manual decision. Geometry-free masks retain the conservative 4x4 cell
  expansion.
- At the previous 268,251-frame population, the current paired-cache layout
  would grow from about 13.7 GiB at 46x46 to about 219.9 GiB at 184x184.
- Instantiating the present flat CVAE at 184x184 changes its encoded tensor
  from `64 x 6 x 6` to `64 x 23 x 23` and raises parameters from about 0.30 M
  to 3.74 M, while the decoder still emits only 30x30 spline coefficients.

## Required transition rule

Grid shape and grid identity must be data contracts, not command-line guesses
or module constants. Product readers validate one consistent grid; annotation,
mask and cache artifacts record it; downstream code reads it from the artifact.
Existing 46x46 manifests without an explicit image shape remain readable with
their historical shape inferred as 46x46.

## Implementation checkpoint: 2026-11-19

The first prerequisite slice is implemented in commits `518981d` and
`369789d`. The canonical
184x184 annotation now rasterizes all 382 line-set keyframes on the target
grid, retains source-cell expansion for 55 geometry-free keyframes and records
82,675 changed boundary cells relative to block replication. A real GUI/editor
smoke opened orbit85 at 184x184 without touching the tracked annotation source.

Model2 and Model1 artifact paths now carry image shape and grid identity, and
legacy checkpoints fall back to the historical 46x46 contract. A random-weight
Model2 application smoke exported 122 aligned 184x184 masks for orbit85. This
checks mechanics only; both masking models still require real retraining and
validation after the complete Product-2 generation is available.

Conditioning preparation now infers and validates the Product-2 grid. A real
orbit85 timestamp read returned all 122 frames with the expected new grid ID.
The paired cache and generic memory-mapped dataset now use manifest-declared
dimensions. A manual-mask orbit85 cache retained 119 eligible paired frames,
occupied 107,568,159 bytes, loaded through the generic dataset, and completed a
finite current-model forward/backward smoke at 184x184. No scientific model
training was run.

On 2026-11-20 the user reported that the complete 184x184 Product-2 CS
generation is available on the server. This has not yet been independently
inspected, but it removes the source-generation wait and makes native Model2
and Model1 retraining the next executable stage.

The native Model2 run keeps BCE and Dice at184x184 but evaluates persistent-
homology regularization on an average-pooled46x46 probability grid. This
preserves the approximate physical topology scale of the adopted46x46 mask
model and avoids a16-fold cell-count penalty. Per-epoch atomic recovery and
exact resume support protect the multi-hour server run.

The user subsequently reported that native Model2 and Model1 training, full
Model2 application and the full two-stage mask export all completed. The
artifacts have not been transferred for independent local inspection.

## Architecture implementation checkpoint: 2026-11-21

The first hierarchical model is implemented as
`HierarchicalSplineConditionalVAE` and selected with
`train_wic_cvae.py --architecture hierarchical`. It follows the five encoder
scales, GroupNorm/SiLU residual blocks, partial-convolution support propagation,
conditioning FiLM and coordinate injection specified below. On the184x184
grid its latent is36 global values plus `8 x 6 x 6` spatial values by default,
for324 values total. The decoder uses the planned residual scales and separate
WIC/SI12 spline-coefficient heads; `--spline-control-points 30` and `92` permit
the controlled decoder-capacity comparison.

Artifacts record the global, spatial and total latent dimensions and retain
the grid-coordinate buffer. Training reports total, global and spatial KL
terms separately. `--kl-warmup-epochs N` supplies a deterministic linear
schedule from beta zero at epoch1 to the configured beta at epochN; the legacy
default remains no warm-up. Existing artifacts and the legacy architecture
remain readable.

A native184x184 Kp-conditioned synthetic forward/backward smoke completed
with428,290 model parameters, output shape `1 x 2 x 184 x 184`, spatial latent
shape `8 x 6 x 6` and finite loss. This verifies mechanics, not scientific
quality. No training or sampling run was started.

## Prerequisites for scientific architecture experiments

### 1. Native-resolution annotations and masking

- [x] Convert the annotation document and embedded grid to 184x184 without
  changing frame classifications or review status.
- [x] Rasterize scaled line-set geometry at 184x184 for native-resolution
  partial-mask targets; retain expanded raster masks when no editable geometry
  exists. Record the policy and counts in conversion provenance.
- [x] Make annotation loading and mask propagation use the document's declared
  `mask_shape` rather than `(46, 46)`.
- [x] Make the annotation browser accept the document/product grid shape so
  converted masks can be inspected without editing the 46x46 source.
- [x] Make Model2 preparation, training and application shape-aware. Save
  `image_shape`, `grid_id` and source-grid evidence in prepared data,
  checkpoints and mask manifests.
- [x] Make Model1 preparation/application shape-aware.
- [x] Retrain both masking models on the new Product-2 generation; do not reuse
  46x46 checkpoints as production 184x184 models.
- [x] Verify a one-orbit Model2 application on `or_0085.nc` before the full
  Product-2 collection is complete.
- [x] Complete the full two-stage Model1/Model2 mask export with newly trained
  184x184 checkpoints (reported complete by the user; not inspected locally).

### 2. Conditioning preparation

- [x] Remove the fixed 46x46 grid ID from conditioning preparation. Infer and
  validate one Product-2 grid ID and image shape across all input orbits.
- [x] Store `grid_id` and `image_shape` in the conditioning manifest.
- [ ] Regenerate IMAGE-aligned conditioning against the completed 184x184
  Product-2 directory so frame identities and source provenance are current.
  The SuperMAG calculations and conditioning feature definitions do not change.

### 3. Paired CVAE cache

- [x] Make paired Product-2 loading validate the grid declared by the source,
  rather than the legacy 46x46 constant.
- [x] Record `image_shape` in the cache manifest and use it for support arrays,
  disk estimates, shard validation and memory mapping.
- [x] Preserve backward reading of version-1 46x46 paired caches.
- [x] Decide the high-resolution storage policy before a full build. Keep the
  existing uncompressed orbit-sharded layout on `/disk/CVAE_cache`; do not
  copy it to `/dev/shm` and do not add compression. The user verified write
  access to `/disk`, which has ample free ZFS capacity.
- [x] Complete a one-orbit manual-mask cache smoke using `or_0085.nc`.
- [x] Prepare the full automatic-mask cache on `/disk/CVAE_cache` (reported
  complete by the user; manifest and counts not inspected locally).

### 4. Training and sampling semantics

- [x] Construct the model from the cache-declared image shape.
- [x] Make reconstruction/KL scaling comparable across grid sizes. New paired
  runs use a fixed2,116-pixel (46x46) reconstruction reference; legacy
  version-1 artifacts retain native-pixel behavior. Batch-size-dependent KL
  scaling remains separate.
- [x] Scale artificial holes by physical size. An omitted `--hole-size`
  resolves from12 pixels at46x46 to48 at184x184; an explicit value remains a
  literal native-pixel override.
- [ ] Establish a memory-safe batch size and use gradient accumulation if a
  comparable optimizer batch is required.
- [x] Add bounded real-batch GPU probing and atomic per-epoch CVAE
  checkpoint/resume before starting long native-grid training.
- [ ] Make ensemble summaries out-of-core. A 50,000-member paired 184x184
  float32 ensemble is about 12.6 GiB before median work.
- [ ] Remove fixed 46x46 assertions from active comparison plots and validate
  coordinates against the artifact/product grid.

### 5. Gate for scientific architecture pilots

Implementation can proceed while the full cache is built, but do not begin the
controlled scientific pilots until all of the following pass:

1. converted annotations load and reproduce the intended 184x184 targets;
2. Model1/Model2 preparation and inference complete on the sample orbit;
3. conditioning preparation records the new grid contract;
4. a paired one-orbit cache is readable by the generic dataset loader;
5. a current-model forward/backward smoke completes at 184x184;
6. cache, GPU-memory and 50,000-draw sampling plans have explicit bounds.

## Proposed encoder

Use five masked residual downsampling stages so the physical receptive field
scales with the four-times-finer grid while the bottleneck remains 6x6:

```text
184x184 -> 92x92 -> 46x46 -> 23x23 -> 12x12 -> 6x6
channels:    16       24       32       48       64
```

- Retain partial convolutions and propagated support masks.
- Use GroupNorm and SiLU in place of batch-size-sensitive BatchNorm.
- Inject the conditioning embedding at every scale through learned feature
  scale and offset parameters.
- Supply fixed normalized CS coordinates, initially `xi`, `eta` and magnetic
  latitude, so the convolutional network can represent absolute position.
- Do not use encoder-to-decoder U-Net skips: sampling from conditioning alone
  has no observation encoder, and training-only skips would bypass the latent
  representation.

Partial convolution is retained because it explicitly renormalizes filters on
valid input support: Liu et al. (2018),
<https://openaccess.thecvf.com/content_ECCV_2018/html/Guilin_Liu_Image_Inpainting_for_ECCV_2018_paper.html>.

## Proposed latent representation

Use two stochastic components:

```text
global latent:  36--48 values
spatial latent: 8 x 6 x 6 values initially
```

The global group controls whole-oval morphology and intensity. The spatial
group supplies regional variability that a single 36-value vector is unlikely
to represent on a 184x184 grid. Use separate KL diagnostics, deterministic
warm-up and an explicit collapse check for each group. A later 12x12 group is
an experiment, not part of the first implementation.

Relevant precedents are the Ladder VAE
<https://arxiv.org/abs/1602.02282> and NVAE
<https://arxiv.org/abs/2007.03898>; this project needs a small auditable model,
not a reproduction of either large image architecture.

## Proposed decoder

Decode from the global condition and spatial latent through explicit residual
upsampling stages:

```text
[global latent, spatial latent, condition]
                 -> 64x6x6
                 -> 64x12x12
                 -> 48x23x23
                 -> 32x46x46
                 -> 24x92x92
                 -> separate WIC and SI12 spline-coefficient heads
                 -> cubic-spline evaluation on 184x184
```

- Retain a shared trunk for joint morphology and separate sensor heads at the
  coefficient output.
- Inject conditioning and fixed grid coordinates at every resolution.
- Replace the current adaptive-pooling coefficient path; it cannot create
  information absent from its smaller feature map.
- Test 30x30 coefficients as the legacy spatial-capacity control and 92x92 as
  the primary high-resolution candidate. The 92x92 spline grid targets roughly
  100 km coefficient spacing while retaining cubic smoothness.

## Controlled architecture comparison

Run pilots on one frozen cache and split:

1. global latent, 30x30 coefficient grid: legacy-capacity control;
2. global latent, 92x92 coefficient grid: decoder-capacity test;
3. global plus `8 x 6 x 6` spatial latent, 92x92 coefficients: hierarchical
   stochastic candidate.

Model selection must include held-out reconstruction, spatial spectra or
structure functions, WIC/SI12 cross-covariance, ensemble spread versus held-
out residuals, morphology diagnostics and per-latent-group KL utilization.
Orbit261 is an event diagnostic, not the sole selection criterion.

## Deferred alternatives

A point transformer is not the first choice because the CS target is a fixed
regular grid and full self-attention over 33,856 cells is unnecessarily costly.
A convolutional neural process becomes relevant if the model later conditions
on arbitrary spatial IMAGE observations or image histories rather than only
global geophysical variables. ConvCNP reference:
<https://arxiv.org/abs/1910.13556>.

## Completion definition

The transition is complete when a documented 184x184 masking generation,
conditioning table and paired cache can train matched Kp and hybrid models;
50,000-member summaries run within explicit memory bounds; the architecture
pilots use one frozen population; and the selected model improves covariance
and morphology diagnostics without sacrificing calibration or provenance.
