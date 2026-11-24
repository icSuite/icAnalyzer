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
  and sensor-quality maps. Formal uncertainty remains in Product2 and is not
  duplicated in the training cache.
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
the equal mean of independently normalized WIC and SI12 errors. The version-2
paired cache omits formal sensor-uncertainty arrays entirely; Product2
remains their authoritative source. Historical version-1 paired caches that
contain them remain readable.
New paired runs multiply those errors by the fixed46x46 reference area of2,116
pixels rather than by the native grid area. This keeps the reconstruction/KL
scale grid-comparable; historical version-1 artifacts retain their native-
pixel reduction. Batch-size dependence of the summed KL term is separate and
still requires a matched effective batch or beta.
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

When low-beta posterior reconstruction is good but fixed-normal prior samples
are implausible, diagnose the learned latent support before retraining. The
empirical sampler selects a bounded mission-spanning training subset near each
requested Kp, encodes it once, and samples complete joint global+spatial rows.
It writes separate mosaics for posterior means and posterior draws; latent
dimensions and groups are never shuffled independently:

```bash
PYTHONPATH=. python scripts/CVAE/sample_empirical_latent_at_kp.py \
  /path/to/cvae_run /path/to/matching/cache \
  --kp 0 2 4 6 \
  --output /path/to/empty/empirical_latent_gallery \
  --bank-frames-per-kp 4096 --kp-half-width 0.34 \
  --draws 500 --posterior-mode both \
  --encode-batch-size 264 --decode-batch-size 64 \
  --cache-mib 4096 --num-workers 4 --prefetch-factor 4 \
  --seed 12104 --device cuda
```

The cache manifest hash must match the run artifact. `posterior means` test
whether encoded latent centers occupy a useful manifold; `posterior samples`
add each selected frame's learned diagonal variance. Plausible empirical
galleries alongside bad unit-normal galleries identify prior mismatch. They do
not themselves define a deployable generative prior.

For an uncapped bank containing every training frame in the requested native
Kp categories, use a narrow tolerance around each stored category and skip the
unneeded gallery decoding:

```bash
PYTHONPATH=. python scripts/CVAE/sample_empirical_latent_at_kp.py \
  /path/to/cvae_run /path/to/matching/cache \
  --kp 0 2 4 6 \
  --output /path/to/empty/complete_empirical_latent_bank \
  --kp-half-width 0.01 --all-matching-frames --bank-only \
  --encode-batch-size 264 --cache-mib 4096 \
  --num-workers 4 --prefetch-factor 4 --device cuda
```

Kp remains a discrete index. GFZ and Product2 encode its traditional thirds
as decimals (`1.667` is the discrete `2-` category); these are not arbitrary
continuous Kp values. The narrow numerical tolerance only accommodates that
storage representation.

Test multimodality from the saved bank without reopening the image cache. Use
individual Kp categories so neighboring activity levels cannot manufacture
apparent modes. The analyzer fits PCA on orbit-grouped training rows, compares one
through six diagonal Gaussian components using training BIC and orbit-held-out
log density, measures assignment stability, and decodes both each mathematical
component center and one real high-responsibility representative:

```bash
PYTHONPATH=. python scripts/CVAE/analyze_empirical_latent_modes.py \
  /path/to/empirical_latent_bank.npz /path/to/cvae_run \
  --kp 0 2 4 6 \
  --output /path/to/empty/latent_mode_analysis \
  --kp-tolerance 0.01 --minimum-frames 500 \
  --pca-variance 0.95 --pca-max-components 40 \
  --max-components 6 --covariance-type diag \
  --minimum-component-fraction 0.03 \
  --initializations 5 --stability-repeats 10 \
  --seed 12104 --device cuda
```

The selected mixture is a statistical candidate only. A physical-mode claim
also requires stable assignments, distinct plausible decoded morphology,
adequate component mass, multiple source orbits and checks against data
quality, coverage and omitted physical conditions.

Fit a deployable-form post-hoc conditional prior before changing CVAE
training. This compares a learned conditional Gaussian (`K=1`) with a
six-component conditional mixture on orbit-held-out frames. The density is fit
to complete global+spatial posterior samples, including encoder variance, and
the same mixture component controls the entire joint latent row:

```bash
PYTHONPATH=. python scripts/CVAE/fit_posthoc_conditional_prior.py \
  /path/to/empirical_latent_bank.npz /path/to/cvae_run \
  --output /path/to/empty/posthoc_prior \
  --components 1 6 --kp 0 2 4 6 \
  --epochs 200 --patience 20 --batch-size 1024 \
  --hidden-dim 64 --learning-rate 0.001 \
  --validation-fraction 0.2 --validation-samples 4 \
  --draws 500 --decode-batch-size 64 --seed 12104 --device cuda
```

Each candidate writes its training history, Kp-dependent component weights,
saved prior and decoded WIC/SI12 sample galleries. The root
`conditional_latent_prior.pt` is the candidate with the highest
orbit-held-out posterior-sample log density. Components are density basis
functions, not physical auroral modes. This fit does not modify or retrain the
CVAE decoder.

To test residual dependence that a diagonal component cannot represent, fit
matched diagonal and low-rank six-component candidates. Rank8 adds a
Kp-conditioned factor matrix to each component, so its covariance is
`diagonal + factor @ factor.T`. Likelihood uses the Woodbury identity and does
not construct a324x324 covariance matrix:

```bash
PYTHONPATH=. python scripts/CVAE/fit_posthoc_conditional_prior.py \
  /path/to/empirical_latent_bank.npz /path/to/cvae_run \
  --output /path/to/empty/posthoc_lowrank_prior \
  --components 6 --covariance-ranks 0 8 --kp 0 2 4 6 \
  --epochs 200 --patience 20 --batch-size 1024 \
  --hidden-dim 64 --learning-rate 0.001 \
  --validation-fraction 0.2 --validation-samples 4 \
  --draws 500 --decode-batch-size 64 --seed 12104 --device cuda
```

The rank-zero candidate preserves the original diagonal implementation. The
same orbit split and posterior standardization are used for both candidates;
the saved summary records the covariance rank and selects by held-out density.
For CUDA low-rank fits the script disables only PyTorch's optional
Python-native/Triton `bmm` override and uses the standard ATen CUDA operation.
This avoids runtime compilation against incompatible system CUDA headers and
does not change the likelihood calculation.
Run the same evaluator on this new output to determine whether low-rank
covariance improves both per-Kp density and decoded tail/cross-channel
calibration.

The same six-component rank-eight form can instead be learned jointly with
the hierarchical CVAE. The encoder remains diagonal Gaussian, while the KL is
estimated from the same reparameterized latent draw already decoded for the
reconstruction term: `log q(z|image,Kp) - log p(z|Kp)`. The mixture density is
evaluated exactly, so this adds no encoder or decoder pass. The complete prior
network is stored inside `wic_cvae.pt`, and the ordinary fixed-Kp samplers use
it automatically after reload:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  /path/to/paired/cache \
  --output /path/to/new/run \
  --architecture hierarchical --batch-order orbit \
  --latent-prior conditional-mixture \
  --prior-components 6 --prior-covariance-rank 8 --prior-hidden-dim 64 \
  --epochs 100 --batch-size 264 \
  --latent-dim 36 --spatial-latent-channels 8 \
  --spline-control-points 92 --spline-degree 3 \
  --beta 0.001 --kl-warmup-epochs 30 \
  --learning-rate 0.0001 --seed 1 --training-holes \
  --hole-device cuda --num-workers 4 --prefetch-factor 4 \
  --cache-mib 4096 --device cuda
```

This option currently supports scalar Kp conditioning only. Because the prior
couples all global and spatial latent dimensions, `kl_global` and `kl_spatial`
are not separately defined; histories report only the joint KL. Omitting
`--latent-prior` preserves the fixed standard-normal objective and artifacts.

Evaluate the saved candidates separately within each Kp group before using the
selected prior for production sampling. This reuses the exact saved
orbit-held-out rows, reports paired candidate-minus-baseline density gains,
and compares decoded prior
samples with decoded held-out posterior samples. The latter reference isolates
prior calibration from decoder reconstruction error:

```bash
PYTHONPATH=. python scripts/CVAE/evaluate_posthoc_conditional_prior.py \
  /path/to/empirical_latent_bank.npz /path/to/cvae_run \
  /path/to/posthoc_prior \
  --output /path/to/empty/posthoc_prior_evaluation \
  --kp 0 2 4 6 --kp-tolerance 0.34 \
  --posterior-samples 4 --draws 2000 \
  --density-batch-size 1024 --decode-batch-size 64 \
  --seed 12104 --device cuda
```

The output contains per-Kp density estimates, the paired comparison-minus-
baseline gain, intensity/morphology quantiles, Kolmogorov--Smirnov distances,
compact figures and the individual decoded metric samples. The default active
area thresholds are100 corrected-WIC units and5 SI12 units; they can be changed
with `--wic-active-threshold` and `--si12-active-threshold`.

For the native184x184 architecture experiments, select the hierarchical model
explicitly. `--latent-dim` remains the global latent size; the spatial group is
configured separately. The primary candidate is36 global values, eight6x6
spatial channels and92x92 spline coefficients:

```bash
PYTHONPATH=. python scripts/CVAE/train_wic_cvae.py \
  /disk/CVAE_cache \
  --architecture hierarchical \
  --latent-dim 36 --spatial-latent-channels 8 \
  --spline-control-points 92 --spline-degree 3 \
  --kl-warmup-epochs 50 \
  --output /path/to/new/run \
  --epochs 500 --batch-size BATCH_SIZE --beta 0.048 \
  --learning-rate 0.0001 --seed 1 --training-holes \
  --num-workers 4 --prefetch-factor 4 --device cuda
```

Choose `BATCH_SIZE` only after a GPU-memory pilot. Artifacts record global and
spatial latent dimensions, and histories include `kl_global`, `kl_spatial` and
the beta used each epoch. Omit `--kl-warmup-epochs` to preserve constant-beta
legacy behavior. The legacy architecture remains the default for backward-
compatible commands.

This command is the scalar-Kp baseline because no named conditioning table is
supplied. Add the matched `--conditioning` and `--conditioning-profile` options
only for the later hybrid comparison.

Before the full native-grid run, execute the same scientific configuration
with `--batch-probe-batches 2` and a disposable output directory. The probe
performs two real optimizer steps, writes `batch_probe.json`, reports peak CUDA
allocated/reserved memory and does not create a trained artifact. Start with
the historical batch264 so the summed-KL balance remains comparable.

For the orbit-sharded184x184 cache, also use `--batch-order orbit`. It shuffles
orbit order every epoch, retains sequential source-frame reads inside each
orbit and packs nearby orbit frames into optimizer batches. Order within a
batch does not change its gradient. This avoids both hundreds of random orbit
mappings and random reads within each shard. The historical `frame` order
remains the default for old commands; workers and prefetching do not correct
that access pattern.

Full training atomically writes `training_state.pt` after every completed
epoch. After an interruption, repeat the identical command with the same
output directory and add `--resume`. The cache hash, model configuration and
training configuration must match; continuation restores the current model,
optimizer, scheduler, best-validation model, history, random streams and
training-hole generator. A failure during the first epoch has no completed
epoch from which to resume.

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

Fixed-Kp sampling is disk-backed and grid independent. Decoded sensor members
and Fang conductance members are written state-major under a temporary
`.sampling_work` directory, exact per-cell quantiles are evaluated in bounded
blocks, and those same files are converted in place to the anomaly factors in
`ensemble_products.npz`. The temporary files are removed only after the final
archive is complete. At184x184,50,000 paired draws with conductance need about
25.2GiB for the final uncompressed factors and roughly twice that free space
during one condition. `--summary-memory-mib` bounds quantile workspace and
defaults to256MiB; it does not change the result.

```bash
PYTHONPATH=. python scripts/CVAE/sample_wic_at_kp.py \
  "$CVAE_RUN" --kp 0 1 2 3 4 5 6 --output "$SAMPLES" \
  --draws 50000 --batch-size 500 --device cuda --no-plots

PYTHONPATH=. python scripts/CVAE/plot_wic_at_kp.py "$SAMPLES"
```

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
For `train_wic_cvae.py`, omitting `--hole-size` preserves the historical
physical scale:12 pixels at46x46,24 at92x92 and48 at184x184. Supplying
`--hole-size` explicitly continues to mean literal native-grid pixels.

Both trainers also accept `--latent-samples K` (default `1`). For `K>1`, the
encoder runs once per image and the decoder evaluates all K reparameterized
latent draws as one flattened GPU batch. The reconstruction terms are averaged
across draws and the analytic KL term is evaluated once, so batch size, beta
and optimizer-update semantics do not change. This affects training and
held-out loss evaluation only; the fixed-Kp sampling commands are unchanged.
