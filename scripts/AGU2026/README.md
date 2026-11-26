# AGU 2026 poster workflow

Run the scripts in numeric order. Each command requires a new, empty output
directory and writes PNG, PDF, machine-readable NPZ/CSV, and a JSON provenance
report. The calculations retain WIC/SI12 pairing and Hall/Pedersen pairing.

## 1. Held-out model validation

This evaluates the decoder at the posterior mean for every held-out frame. Ten
distinct reconstruction panels are selected at the 5th, 15th, ..., 95th
percentiles of joint NRMSE, spanning easy through difficult cases without
visual cherry-picking.

```bash
RUN=~/IMAGE_CVAE_50km/cvae_runs/wic_si12_kp_hierarchical_jointprior_k6_rank8_beta001_20261124
CACHE=/disk/CVAE_cache/wic_si12_184x184_model1_model2_valid5_lean_20261121
POSTER=~/IMAGE_CVAE_50km/agu2026

PYTHONPATH=. python scripts/AGU2026/01_model_validation.py \
  "$RUN" "$CACHE" \
  --output "$POSTER/01_model_validation" \
  --split test \
  --batch-size 64 \
  --cache-mib 4096 \
  --device cuda
```

Omit `--maximum-frames` for the full held-out split. Add, for example,
`--maximum-frames 1000` only for a preliminary run.

## 2. Conductance covariance

Choose one fixed-condition sampling directory, normally the Kp value used in
the poster. The default saves 100 joint covariance modes, displays the first
three, and chooses the teleconnection anchor at maximum joint variance.

```bash
ENSEMBLE=~/IMAGE_CVAE_50km/prior_gallery/jointprior_k6_rank8/kp_01_2

PYTHONPATH=. python scripts/AGU2026/02_conductance_covariance.py \
  "$ENSEMBLE" \
  --output "$POSTER/02_conductance_covariance_kp2" \
  --modes 100 \
  --plot-modes 3 \
  --workspace /disk/CVAE_cache/agu2026_workspace
```

The workspace temporarily holds one uncompressed anomaly factor. It is removed
when the script exits normally.

## 3. Sparse reconstruction

This conditions the joint Hall/Pedersen modes from step 2 on noisy values at
the same sparse spatial sites in both fields.

```bash
COVARIANCE="$POSTER/02_conductance_covariance_kp2/02_conductance_covariance.npz"

PYTHONPATH=. python scripts/AGU2026/03_sparse_reconstruction.py \
  "$COVARIANCE" "$ENSEMBLE" \
  --output "$POSTER/03_sparse_reconstruction_kp2" \
  --observation-counts 10 30 100 \
  --observation-std 0.5 \
  --truth-members 16 \
  --modes 100 \
  --workspace /disk/CVAE_cache/agu2026_workspace
```

Using the same ensemble demonstrates internal distribution consistency, not
independent validation. For a stronger check, sample the same Kp again with a
different seed and pass that second sampling directory as `ENSEMBLE` in step 3.
