# Conditional VAE workflows

This directory contains the two supported Kp-conditional CVAE workflows. Both
use the generic spline model, cache loaders, losses, training loop, artifact
format and ensemble machinery under `icanalyzer/cvae/`.

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
- `train_wic_cvae.py`: train the one-channel WIC Kp-conditional CVAE.
- `sample_wic_at_kp.py`: draw WIC at selected Kp and propagate each member
  through collapsed Zhang--Paxton and Fang electron conductance.
- `plot_wic_at_kp.py`: regenerate polar WIC and conductance summaries from
  saved fixed-Kp ensembles without resampling.

The current command interfaces are available through:

```bash
PYTHONPATH=. python scripts/CVAE/prepare_cvae_hp_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_automatic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/sample_conductance_at_kp.py --help

PYTHONPATH=. python scripts/CVAE/prepare_wic_cvae_cache.py --help
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py --help
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py --help
PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py --help
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
