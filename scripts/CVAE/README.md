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
  disk-backed cache.
- `train_wic_cvae.py`: train the one-channel WIC CVAE with historical scalar
  Kp conditioning or the selected 11-feature condition vector.
- `select_conditioning_cases.py`: select representative complete condition
  vectors from actual cache-aligned training frames for ensemble sampling.
- `sample_wic_at_kp.py`: draw WIC at selected Kp or named 11-feature conditions
  and propagate each member through collapsed Zhang--Paxton and Fang electron
  conductance.
- `plot_wic_at_kp.py`: regenerate polar WIC and conductance summaries from
  saved Kp or named-condition ensembles without resampling.

The current command interfaces are available through:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_cvae_hp_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_automatic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/sample_conductance_at_kp.py --help

PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/select_conditioning_cases.py --help
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py --help
PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py --help
```

Prepare the combined conditioning table outside the repository:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_conditioning_data.py \
  --supermag-dir /media/bing/LaCie/supermag_data \
  --source ~/Dropbox/work/data/IMAGE_FUV_new/precipitation_cs/image_apex_130km_46x46_v1/IR_hardy_smoothed_wic1_si121_si131 \
  --output ~/Dropbox/work/data/IMAGE_CVAE/conditioning/supermag_all
```

The output includes all Product-2 frames overlapping the supplied SuperMAG
time span. Nearest-minute alignment is limited to 30 seconds by default;
long gaps and source fill values remain missing rather than being interpolated.

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

Sampling accepts a JSON list whose records contain exactly the 11 conceptual
conditions. DOY and clock angle are converted to cyclic pairs internally:

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

Rather than inventing combinations of correlated parameters, generate the
JSON from complete, actually observed training rows. The default cases are
typical, quiet, active-substorm, storm, By negative/positive and Bz
south/north. Within each explicitly thresholded regime, the chosen row is
nearest the regime's multivariate median:

```bash
PYTHONPATH=. python scripts/CVAE/select_conditioning_cases.py \
  ~/IMAGE_CVAE/training_data/automatic_wic_cvae_model1_model2_fuvpy_bs2d_valid5_20261106 \
  ~/IMAGE_CVAE/conditioning/supermag_all \
  --output ~/IMAGE_CVAE/conditioning/sample_conditions.json
```

```bash
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py \
  ~/IMAGE_CVAE/cvae_runs/wic_conditions11_model1_model2_20261107 \
  --conditions ~/IMAGE_CVAE/conditioning/sample_conditions.json \
  --output ~/IMAGE_CVAE/cvae_runs/wic_conditions11_model1_model2_20261107/condition_samples \
  --draws 50000 --batch-size 500 --device cuda
```

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
