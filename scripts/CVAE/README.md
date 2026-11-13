# Conditional VAE workflows

This directory contains the supported Hall/Pedersen and WIC CVAE workflows. Both
use the generic spline model, cache loaders, losses, training loop, artifact
format and ensemble machinery under `icanalyzer/cvae/`.

`prepare_conditioning_data.py` is the independent first step toward replacing
scalar Kp with a selected space-weather conditioning vector. It reads yearly
SuperMAG text exports, converts `999999` fill values to missing data, aligns
the selected one-minute parameters to Product-2 frame times and writes a
compact external table plus quantitative coverage/provenance. It does not
modify a CVAE cache or model.

`analyze_conditioning_importance.py` joins that table exactly to an existing
WIC cache by orbit, source-frame index and timestamp. It evaluates named
parameters against interpretable masked-WIC brightness and morphology targets
with held-out years, drop-column refits and permutation. The full discovery
model tests exactly SML, SMR, IMF By, IMF Bz, Vx, clock angle, dynamic
pressure, Newell coupling, Kp, day of year and centered-dipole tilt. Clock
angle and day of year use paired sine/cosine columns internally and are removed
or permuted jointly as one conceptual feature. Kp-only is an explicit reference
model. This selects candidate conditioning variables; it does not train or
alter a CVAE.

Existing spline artifacts created before the package/class rename remain
loadable. New models must explicitly name their modeled fields, so WIC and
Hall/Pedersen artifacts cannot silently inherit the old WIC/SI12 labels.

## Hall/Pedersen CVAE

- `prepare_cvae_hp_cache.py`: combine native Hall/Pedersen Product 3
  fields with an aligned automatic-mask corpus and write the disk-backed cache.
- `train_automatic_cvae.py`: train the paired Hall/Pedersen Kp-conditional CVAE.
- `sample_conductance_at_kp.py`: draw paired Hall/Pedersen fields at selected Kp.

## WIC CVAE

- `prepare_wic_cvae_cache.py`: combine corrected WIC Product 2 with either
  manual annotations or an aligned automatic-mask corpus and write the
  disk-backed cache. `--wic-field preproton` instead selects the stored
  background-subtracted, smoothed WIC before SI12 proton correction.
- `prepare_wic_si12_cvae_cache.py`: apply the same manual or automatic WIC
  selection to corrected WIC and SI12 while retaining separate sensor-validity
  and sensor-quality maps. It caches `dwic_corrected` and `dsi12` for future
  uncertainty experiments.
- `train_wic_cvae.py`: train either the one-channel WIC CVAE or the paired
  corrected-WIC/SI12 CVAE with historical scalar Kp conditioning, the full
  11-feature condition vector, the reduced `hybrid4` profile (SML, IMF By,
  IMF Bz and dipole tilt), or `hybrid6_sml_history`, which adds causal SML
  mean and slope terms.
- `sample_wic_at_kp.py`: draw WIC at selected Kp, exact named conditions or
  empirical IMF regimes and propagate each member through collapsed
  Zhang--Paxton and Fang electron conductance.
- `sample_wic_si12_event.py`: use a real one-minute SML/IMF/tilt sequence to
  sample paired corrected-WIC/SI12 medians from either reduced profile, with
  per-minute checkpoints.
- `plot_wic_si12_event_comparison.py`: match IMAGE orbit frames to the nearest
  event minute and export 2x3 IMAGE/median/medoid WIC/SI12 figures.
- `plot_wic_at_kp.py`: regenerate polar WIC and conductance summaries from
  saved Kp or named-condition ensembles without resampling.

The current command interfaces are available through:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_cvae_hp_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_automatic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/sample_conductance_at_kp.py --help

PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py --help
PYTHONPATH=. python scripts/CVAE/prepare_wic_si12_cvae_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py --help
PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py --help
```

Prepare the combined conditioning table outside the repository:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_conditioning_data.py \
  --supermag-dir /media/bing/LaCie/supermag_data \
  --source ~/Dropbox/work/data/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131 \
  --output ~/Dropbox/work/data/IMAGE_CVAE/conditioning/supermag_sml_history_v2
```

The output includes all Product-2 frames overlapping the supplied SuperMAG
time span. Nearest-minute alignment is limited to 30 seconds by default;
long gaps and source fill values remain missing rather than being interpolated.
Before IMAGE alignment, the continuous one-minute series is also used to
derive a causal 30-minute SML mean over `t-29...t` and a causal five-minute
least-squares SML slope over `t-4...t`, in nT/min. Every sample in a history
window must be finite and exactly one minute apart. The version-2 manifest
records these definitions. Existing version-1 `supermag_all` tables remain
readable for `all11` and `hybrid4`, but do not contain the two history columns.

Run the scientific parameter benchmark against the corresponding frozen
Model1+Model2 automatic WIC cache:

```bash
PYTHONPATH=. python scripts/CVAE/analyze_conditioning_importance.py \
  ~/Dropbox/work/data/IMAGE_CVAE/training_data/automatic_wic_cvae_model1_model2_fuvpy_bs2d_valid5_20261106 \
  ~/Dropbox/work/data/IMAGE_CVAE/conditioning/supermag_all \
  --output ~/Dropbox/work/data/IMAGE_CVAE/conditioning/importance_model1_model2_broad_20261107 \
  --split-mode year \
  --trees 500 \
  --maximum-training-frames 1000000 \
  --threads 4
```

The benchmark uses only the jointly finite predictor population and
standardizes all WIC targets within each training fold. Its total-brightness
target is explicitly the observed masked total and therefore remains sensitive
to mask coverage. Uniform-average R2 is the primary cross-target summary, and
the output also retains variance-weighted and per-target scores. Conditional
importance can be suppressed by correlated inputs, so the results do not by
themselves establish causality. `--split-mode cache` exists only for a quick
mechanical smoke on a small manual cache; its scores are not the scientific
comparison.

Train the 11-feature WIC CVAE by joining the existing WIC cache and conditioning
table at startup. No new image cache is created. Frames lacking any requested
condition are excluded, and the 13 numerical columns are standardized using
the retained training split only:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  ~/IMAGE_CVAE/training_data/automatic_wic_cvae_model1_model2_fuvpy_bs2d_valid5_20261106 \
  --conditioning ~/IMAGE_CVAE/conditioning/supermag_all \
  --conditioning-embedding-dim 32 \
  --output ~/IMAGE_CVAE/cvae_runs/wic_conditions11_model1_model2_20261107 \
  --epochs 500 --batch-size 264 --beta 0.048 \
  --latent-dim 36 --spline-control-points 30 --spline-degree 3 \
  --learning-rate 0.0001 --seed 1 --training-holes \
  --num-workers 2 --device cuda
```

The reduced four-variable experiment reuses the same cache and conditioning
table. Only rows missing SML, IMF By, IMF Bz or dipole tilt are excluded; a
missing unused field does not exclude a frame. Use a new run directory because
model runs are never overwritten:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  ~/IMAGE_CVAE/training_data/automatic_wic_cvae_model1_model2_fuvpy_bs2d_valid5_20261106 \
  --conditioning ~/IMAGE_CVAE/conditioning/supermag_all \
  --conditioning-profile hybrid4 \
  --conditioning-embedding-dim 32 \
  --output ~/IMAGE_CVAE/cvae_runs/wic_hybrid4_model1_model2_b264_beta0048_20261108 \
  --epochs 500 --batch-size 264 --beta 0.048 \
  --latent-dim 36 --spline-control-points 30 --spline-degree 3 \
  --learning-rate 0.0001 --seed 1 --training-holes \
  --num-workers 4 --prefetch-factor 4 --device cuda
```

For a paired corrected-WIC/SI12 run, prepare a distinct two-channel cache. The
scientific selection remains entirely WIC-derived, but native WIC and SI12
validity and coverage remain separate. A frame must satisfy the requested
minimum valid fraction in both channels:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_wic_si12_cvae_cache.py \
  --source /path/to/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131 \
  --masks /path/to/model1_model2_masks \
  --minimum-valid-fraction 0.05 \
  --output /path/to/training_data/automatic_wic_si12_model1_model2_valid5
```

Use that cache with the same trainer and hybrid4 options shown above. The
paired cache supplies two validity masks and two quality maps. The encoder uses
their union while receiving both quality maps, and the reconstruction loss is
the equal mean of independently normalized WIC and SI12 errors. Formal sensor
uncertainties are stored but deliberately do not enter this baseline loss.
Training artifacts and held-out evaluation report the two channel
reconstruction contributions separately. The same sampler accepts the paired
artifact: every latent draw produces one aligned corrected-WIC/SI12 member.
It saves separate means, quantiles, anomaly factors, medoids, exceedance maps
and matching 6x6 mosaics for both sensors. The anomaly-factor columns and
mosaic indices are shared, preserving cross-sensor covariance and member
identity. Corrected WIC still enters collapsed Zhang--Paxton/Fang electron
conductance. SI12 remains a proton-emission proxy; no proton-conductance
conversion is applied or implied.

For paired products the plot command additionally writes `si12_at_kp`,
`si12_sample_mosaic`, `si12_medoid` and `si12_exceedance_probability` PNG/PDF
files. Defaults are a 50-count SI12 color limit and 10-count exceedance
threshold; override them with `--si12-vmax` and
`--si12-exceedance-threshold`. Sampling and plotting use the same commands
shown below; point `CVAE_RUN` at the paired run directory.

For the controlled SML-history test, keep the paired image cache, architecture,
split, seed and optimization settings fixed and change only the conditioning
table/profile and output directory. No image cache regeneration is needed:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  /path/to/training_data/automatic_wic_si12_model1_model2_valid5 \
  --conditioning ~/Dropbox/work/data/IMAGE_CVAE/conditioning/supermag_sml_history_v2 \
  --conditioning-profile hybrid6_sml_history \
  --conditioning-embedding-dim 32 \
  --output ~/Dropbox/work/data/IMAGE_CVAE/cvae_runs/wic_si12_hybrid6_sml_history \
  --epochs 500 --batch-size 264 --beta 0.048 \
  --latent-dim 36 --spline-control-points 30 --spline-degree 3 \
  --learning-rate 0.0001 --seed 1 --training-holes \
  --num-workers 4 --prefetch-factor 4 --device cuda
```

The six inputs are ordered as current SML, trailing 30-minute SML mean,
trailing five-minute SML slope, current IMF By, current IMF Bz and dipole tilt.
Frames with an incomplete requested history window are excluded just like any
other nonfinite condition. Compare the retained split counts with the matched
`hybrid4` run before interpreting model differences.

The event sampler accepts paired `hybrid4` and `hybrid6_sml_history` artifacts
and requires the original one-minute SuperMAG export. For the history profile,
the trailing terms are derived from the full file before the event interval is
selected, so the first event minute uses its actual pre-event context. It does
not interpolate the IMAGE-aligned conditioning table, because IMAGE frames
are only about two minutes apart. Event bounds are inclusive; partial boundary
minutes are omitted. The same latent seed is reused at every minute to reduce
Monte Carlo flicker, but the result remains a sequence of conditional
marginals rather than a learned temporal trajectory. For orbit 261 frames
42--185, the exact bounds select 293 minutes from 08:50 through 13:42 UTC:

```bash
CVAE_RUN=~/Dropbox/work/data/IMAGE_CVAE/cvae_runs/wic_si12_hybrid4_model1_model2_valid5_20261110
EVENT_OUT=~/Dropbox/work/temp_storage/CVAE_orbit_261/event_hybrid4_50k

PYTHONPATH=. python scripts/CVAE/sample_wic_si12_event.py \
  "$CVAE_RUN" /path/to/supermag/year_2000.txt \
  --start 2000-08-28T08:49:53 --stop 2000-08-28T13:42:02 \
  --output "$EVENT_OUT" \
  --draws 50000 --batch-size 500 --device cuda \
  --maximum-imf-gap-minutes 1
```

The orbit-261 interval has one missing IMF minute at11:42 UTC, bracketed by
finite values at11:41 and11:43. The explicit option above linearly interpolates
that single By/Bz row and records `imf_interpolated` in the condition file.
Without the option, any missing IMF value remains a hard error. Gaps longer
than the requested maximum and unbounded endpoint gaps are never filled.

The output stores `event_medians.npy` with dimensions
`minute, sensor, row, column`, `event_conditions.npz`, exact source/model
provenance and one completion flag per minute. If an interrupted run has the
same configuration, repeat the command with `--resume`; completed minutes are
not decoded again.

Medians do not retain a coherent ensemble member. To add separate WIC and SI12
morphology medoids to an already completed event, repeat the exact sampling
command with `--resume --add-medoids`. This regenerates the deterministic
ensembles but skips median calculation, then saves `event_medoids.npy`, the two
member indices and independent medoid completion flags. The WIC and SI12
medoids are selected separately and therefore need not be the same paired
member:

```bash
PYTHONPATH=. python scripts/CVAE/sample_wic_si12_event.py \
  "$CVAE_RUN" "$SUPERMAG" \
  --start 2000-08-28T08:49:53 --stop 2000-08-28T13:42:02 \
  --output "$EVENT_OUT" \
  --draws 50000 --batch-size 500 --device cuda \
  --maximum-imf-gap-minutes 1 --resume --add-medoids
```

After the event run completes, compare every selected IMAGE frame with its
nearest CVAE minute. The2x3 layout uses corrected WIC on the first row and SI12
on the second; columns are IMAGE observation, CVAE pixelwise median and CVAE
morphology medoid. IMAGE negative values are clipped to zero to match CVAE
cache preparation. The default fixed scales are1500 WIC counts and50 SI12
counts, and `--frame-stop` is inclusive:

```bash
EVENT_OUT=~/Dropbox/work/temp_storage/CVAE_orbit_261/event_hybrid4_50k
IMAGE_ORBIT=~/Dropbox/work/data/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131/or_0261.nc
COMPARISON_OUT="$EVENT_OUT/image_comparison"

PYTHONPATH=. python scripts/CVAE/plot_wic_si12_event_comparison.py \
  "$EVENT_OUT" "$IMAGE_ORBIT" \
  --frame-start 42 --frame-stop 185 \
  --output "$COMPARISON_OUT"
```

This writes144 PNGs, `frames.csv` with the IMAGE-to-CVAE time mapping and
conditions, and a lazy-loading `index.html` gallery. The CVAE panels retain the
model's learned support; the IMAGE panels retain each observation's finite
footprint. A source frame with no finite observation remains blank rather than
being fabricated.

Exact-condition sampling remains available for diagnostics through a JSON list
containing all 11 conceptual conditions. It should not be used to represent a
broad IMF regime because every decoded member receives that same vector. Its
legacy `clock_angle` field uses the SuperMAG column's trained convention, not
the standard 0-degree-north/180-degree-south convention used below.

```json
[
  {
    "label": "example",
    "sml": -200,
    "smr": -20,
    "imf_by": 2,
    "imf_bz": -3,
    "vx": -450,
    "clock_angle": 150,
    "dynamic_pressure": 2,
    "newell": 5000,
    "kp": 3,
    "day_of_year": 100,
    "dipole_tilt": 10
  }
]
```

The controlled hybrid4 comparison does not resample empirical condition rows.
It creates 12 designed groups from four IMF orientations and three SML bands:
low/quiet-like `[-100, 0)` nT, moderate/normal `[-300, -100)` nT and
strong/disturbed `[-1000, -300)` nT. SML and the selected IMF component are
sampled independently and uniformly. The selected IMF component spans 2--10
nT, standard clock angle is uniform within 30 degrees of the requested
direction, dipole tilt is fixed at zero and Kp is fixed at 3 only for the
collapsed Zhang--Paxton/Fang conversion. Identical latent draws and uniform
ranks are used across all groups to reduce Monte Carlo noise in comparisons.

```bash
CVAE_RUN=~/IMAGE_CVAE/cvae_runs/wic_hybrid4_model1_model2_b264_beta0048_20261108
SAMPLES="$CVAE_RUN/controlled_imf_sml_samples"

PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py \
  "$CVAE_RUN" \
  --controlled-regimes \
  --output "$SAMPLES" \
  --draws 50000 --batch-size 500 --device cuda --no-plots

PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py "$SAMPLES"
```

Each group saves the first 36 paired WIC realizations and produces
`wic_sample_mosaic.png`/`.pdf` as a 6x6 polar-contour mosaic. The plot-only
command regenerates these mosaics from the saved members without decoding or
conductance propagation. For older completed sample directories, it exactly
reconstructs the first 36 members from the saved mean and full anomaly factor,
so existing 50,000-member ensembles do not need to be sampled again.

Two additional distribution summaries preserve the distinction between a
coherent realization and a marginal pixel statistic. `wic_medoid.png`/`.pdf`
shows the actual morphology-space medoid after robust scaling of total
brightness, latitude centroid and width, active area, and MLT sector fractions.
It minimizes total pairwise squared distance in that morphology space.
`wic_exceedance_probability.png`/`.pdf` shows the fraction
of members exceeding 100 corrected-WIC counts at each pixel. Change that
threshold with `--wic-exceedance-threshold`. The plot-only command derives both
summaries from the saved mean and full anomaly factor for older completed
ensembles, so this also requires no resampling.

For the SI12-misalignment diagnostic, first regenerate precipitation CS with
the optional `wic_smoothed`/`dwic_smoothed` fields and export a fresh automatic
mask corpus tied to those regenerated files. Then prepare a distinct cache:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py \
  --source /path/to/regenerated/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131 \
  --masks /path/to/regenerated/model1_model2_masks \
  --wic-field preproton \
  --minimum-valid-fraction 0.05 \
  --output /path/to/training_data/automatic_wic_preproton_model1_model2
```

Training uses the ordinary `train_wic_cvae.py` command. The artifact records
`wic_preproton` as its modeled field. Sampling produces WIC ensemble products,
mosaics, medoids and exceedance probabilities, but deliberately omits
Zhang--Paxton/Fang conductance: pre-proton WIC still contains proton emissions
and cannot be interpreted as electron-only counts.

The older `--regimes CACHE CONDITIONING` mode remains available as an
empirical-distribution diagnostic. It is not the controlled climatological
comparison because its nuisance conditions and Kp inherit their observed joint
distribution.

For a manual-mask WIC cache, use a fresh output directory:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py \
  --annotations data/partial_image_annotations_46x46.json \
  --minimum-valid-fraction 0.05 \
  --output data/training_data/manual_wic_cvae_fuvpy_bs2d_valid5
```

Use `--masks <directory>` instead of `--annotations` for a frozen automatic
mask corpus. The two selection sources are mutually exclusive and recorded in
the cache manifest.

The former `scripts/IMAGE/vae_pipeline/` directory retains the older
WIC/SI12 VAE, reconstruction studies, diagnostics and presentation scripts.
Those are not required to prepare, train or sample either supported automatic
CVAE above.

Both trainers accept `--hole-device auto|cpu|cuda`. The default `auto` retains
the historical CPU implementation for CPU training and uses vectorized CUDA
hole generation for CUDA training. Run artifacts record the resolved backend.

Both trainers also accept `--latent-samples K` (default `1`). For `K>1`, the
encoder runs once per image and the decoder evaluates all K reparameterized
latent draws as one flattened GPU batch. The reconstruction terms are averaged
across draws and the analytic KL term is evaluated once, so batch size, beta
and optimizer-update semantics do not change. This affects training and
held-out loss evaluation only; the fixed-Kp sampling commands are unchanged.
