# Partial-image annotation

`annotate_partial_images.py` is a resumable graphical annotation tool for the
IMAGE conductance data. It avoids per-frame pixel labeling by recording:

- one usable interval per orbit;
- an optional fully good interval inside it;
- a few masks drawn on representative partial frames.

Masks can be drawn either by lassoing the **valid area to retain** or by
dragging a straight divider across the image and choosing which half is valid.
A mask becomes active at its keyframe and is held unchanged for all later
frames until another keyframe replaces it. There is no temporal interpolation.

The tool only reads `data/good_orbit_segments.csv`. It does not alter that
file. New annotations and the current cursor are saved separately in
`data/partial_image_annotations.json`.

## Run

Use the environment that provides `icreader`:

```bash
mamba activate tConductance
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py
```

The default external data locations match the current project layout. They can
be overridden with `--data-dir` and `--overview-dir`.

To start or resume at a particular orbit:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --start-orbit 169 \
  --start-frame 72
```

To try the interface without writing to the main annotation file:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --start-orbit 169 \
  --start-frame 72 \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

Closing the window, pressing `Q` or Escape, or clicking **Stop and save**
records the current orbit, page, and frame. The next ordinary launch resumes
there automatically. Boundary changes, mask changes, navigation, and
selection are also saved immediately using atomic file replacement.

The `/tmp` path above is suitable only for trying the interface; an operating
system cleanup or reboot may remove it. The default
`data/partial_image_annotations.json` is the durable project location. To
resume a temporary trial at its saved cursor, omit the start overrides:

```bash
python scripts/IMAGE/vae_pipeline/annotate_partial_images.py \
  --annotations /tmp/icAnalyzer_annotation_trial.json
```

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

For orbit 169, the existing `0:51` good sequence seeds all four boundaries.
Here, **FIRST usable** and **FIRST fully good** are correctly fixed at frame
zero. To retain later partially useful images, select the last useful frame
and click **Set LAST usable**. Frames 52 through that selected frame will turn
orange; add sparse masks within that new orange interval.
The first keyframe must be at frame 52. Its mask remains valid through the end
of the interval unless a later keyframe explicitly replaces it.

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

All line endpoints, side assignments, and the resulting 36x36 mask are stored.
Reopening that exact keyframe restores the complete line set and mask.
Intersecting half-planes produce convex valid regions; retain lasso mode for a
strongly concave or disconnected valid region.

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

This stage records annotations only. Rasterizing the masks together with
finite-value checks and measurement weights belongs in the processed-data
builder, so the policy remains explicit and reproducible.

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
convolution. Product-2's background-fit weight is inverted first; the encoder
and loss therefore receive auroral-interest weight rather than the upstream
background-fit orientation.

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

The upstream `method_quality_weight` is a background-fit weight, not a measure
of confidence that a pixel is scientifically useful. A value of zero means
the background method could not fit the data and commonly identifies aurora;
a value of one means the data are explained by the background fit. icAnalyzer
therefore uses `1 - method_quality_weight` as the auroral-interest weight,
then multiplies it by the WIC or SI12 channel coverage. This convention and
its source field are saved in the run provenance.

Frames follow the legacy sparsity rule: if more than 33% of common WIC/SI12
pixels are intrinsically invalid before optional SZA masking, the frame is
excluded. The threshold can be changed with `--maximum-missing-fraction`, but
`0.33` is the compatibility default. Compare physical-space held-out residuals,
sample calibration, covariance, and reconstruction skill in addition to the
optimization loss.

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
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
technically valid, and clipped counts under `input_negative_clipping` in both the artifact
provenance and `summary.json`. A complete audit of the 157 manually selected
Product-2 files found no negative WIC cells but 3,482,898 negative SI12 cells
among 14,030,013 finite selected cells (24.82%). With `SZA >= 105`, 390,848 of
2,718,083 technically valid SI12 cells were negative (14.38%). This is an upstream
data-processing issue; clipping makes the requested VAE experiment runnable
but does not resolve its cause.

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
  /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
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
Their intervals map to 10,680 Product-2 frames, of which 9,728 have valid
weighted support. The core selected input arrays are approximately 0.7 GiB,
before transformed training arrays and model state, rather than the full 31
GiB NetCDF corpus.

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
`w` has the same background-fit orientation as the current Product-2 method
weight.

VAE artifacts created before the `1 - w` correction used the opposite loss
orientation and are not valid auroral-model results. Retrain them. Existing
direct comparison maps of the physical camera fields are unaffected, but
quality-weight statistics must be regenerated.

First generate a streaming field-level report. This reads one orbit pair at a
time and writes `paired_input_comparison.json` plus mean legacy, rerun, and
difference maps:

```bash
PYTHONPATH=. python scripts/IMAGE/vae_pipeline/compare_legacy_current_inputs.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
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
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
  --legacy-comparison ~/Dropbox/work/data/IMAGE_FUV/precipitation_IR_hardy_weighted \
  --data-version current \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_paired_current_sza105 \
  --minimum-sza 105 --no-csv-selection \
  --samples 1000 --device auto

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
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
