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

The pipeline splits whole orbit files, fits the channel transform using the
training split only, trains a mask-aware two-channel VAE, samples paired
corrected-WIC/SI12 fields, propagates corrected WIC to Hall and Pedersen at a
fixed 2 keV electron energy, and performs an ensemble-space sparse update.
The decoder constrains corrected WIC to be nonnegative without clipping
generated fields.

Measurement uncertainty enters the reconstruction loss through `dwic` and
`dsi12`; it is not decoded as an additional science field. The runner fits one
median transformed-space precision scale per channel from the training split.
Each pixel is weighted by quality times precision relative to that fixed
channel scale. Loss reduction divides by valid support, not by the sum of
weights, so an image whose uncertainties are uniformly larger contributes
less than a precise image. The transformed variance floor defaults to `1e-3`
and relative precision is capped at `100`; both values and the fitted scales
are stored under `uncertainty_weighting` in the run configuration.

The upstream `method_quality_weight` is a background-fit weight, not a measure
of confidence that a pixel is scientifically useful. A value of zero means
the background method could not fit the data and commonly identifies aurora;
a value of one means the data are explained by the background fit. icAnalyzer
therefore uses `1 - method_quality_weight` as the auroral-interest weight,
then multiplies it by the WIC or SI12 channel coverage. This convention and
its source field are saved in the run provenance.

This loss is not numerically comparable to runs made before the absolute
uncertainty revision. In particular, `beta` must be treated as a new tuning
parameter rather than copied from an older run solely because its value was
previously adequate. Override the safeguards with `--variance-floor`,
`--maximum-relative-precision`, and `--precision-scale-samples` only as an
explicit sensitivity experiment.

Matched legacy and current runs fit their precision scales independently, so
their raw weighted reconstruction losses are also not direct measures of which
physical fields are closer to truth. Compare physical-space held-out residuals,
sample calibration, covariance, and reconstruction skill; use each weighted
loss for optimization and within-run checkpoint selection.

```bash
python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
  --output outputs/wic_si12_baseline \
  --epochs 20 \
  --samples 1000 \
  --minimum-sza 105 \
  --observation-mode joint
```

`--minimum-sza 105` applies an inclusive, per-pixel night-side filter after
the existing CSV/source-index frame selection: cells with `SZA < 105°` or
nonfinite SZA become invalid and receive zero training weight. A frame is
dropped only when no valid positively weighted pixels remain after all
filters. Omit the option to reproduce the original no-SZA-threshold baseline.
The threshold is recorded in both model configuration and data provenance.

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
  --epochs 20 --batch-size 32 --latent-dim 32 --samples 1000 --device auto

PYTHONPATH=. python scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py \
  ~/Dropbox/work/data/IMAGE_FUV/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy \
  --legacy-comparison ~/Dropbox/work/data/IMAGE_FUV/precipitation_IR_hardy_weighted \
  --data-version legacy \
  --output ~/Dropbox/work/data/IMAGE_FUV/vae_runs/wic_si12_paired_legacy_sza105 \
  --minimum-sza 105 --no-csv-selection \
  --epochs 20 --batch-size 32 --latent-dim 32 --samples 1000 --device auto
```

The paired experiment measures the net change between the two product
pipelines. It must not be described as a pure fuvpy effect: source matching,
detector-to-CS reduction, coregistration, uncertainty propagation, and
validity handling also changed between the legacy and current products.

## Visualizing a completed run

Generate the legacy-style model and covariance diagnostics without retraining:

```bash
python scripts/IMAGE/vae_pipeline/visualize_wic_si12_run.py \
  outputs/wic_si12_baseline
```

By default, figures and `diagnostics.json` are written under the run's
`figures/` directory. Use `--output` to write elsewhere, `--examples` to
change the sample-mosaic size, or `--anchor ROW,COLUMN` to select the common
correlation probe. The default probe is the maximum-variance corrected-WIC
pixel for FUV and Hall pixel for conductance.

The figures cover training history, decoded FUV and propagated-conductance
samples, ensemble mean and standard deviation, joint-channel correlation
maps, leading covariance modes, covariance spectra, and the sparse-update
prior and posterior. Runs produced by the current runner also store the
sampled latent vectors and sparse-validation truth and observations, enabling
latent-draw and truth/error reconstruction figures. Older compatible runs
remain visualizable but cannot recover values that were not saved originally.
