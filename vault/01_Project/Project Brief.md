# Project Brief

Last reviewed: 2026-11-08

## Purpose

`icAnalyzer` documents statistical analysis of height-integrated ionospheric
Hall and Pedersen conductance estimates derived from IMAGE satellite data.

**Confirmed by the user:** The intended publication project is an empirical
conductance model using a variational autoencoder. Spatial covariance is
particularly important for the user's broader work on data assimilation of
ionospheric electrodynamics.

The desired end-to-end path is now more specific: train the generative model
on paired WIC/SI12 fields, sample and analyze their joint spatial covariance,
propagate the corrected-WIC component of each paired sample through the current
physical conversion to Hall and Pedersen conductance, estimate the resulting
joint conductance covariance, and use that covariance for sparse conductance
reconstruction.

Reaffirmed by the user on 2026-10-03: covariance-based data assimilation is the
primary endpoint. Legacy sparse reconstruction used covariance eigenmodes,
not the decoder. Decoder sampling constructs the prior; encoder inpainting
is a secondary diagnostic and cannot alone determine model adoption.

Latest user clarification on 2026-10-07: the more specific use case is a
conductance prior in a probabilistic inversion. Treat conductance as an unknown
constrained by the learned joint spatial distribution; available IMAGE
retrievals can supply additional observations, not the prior-error definition.
The earlier single-timestep ensemble of Lompe runs remains a possible
sensitivity illustration, not the primary scientific framing. Learned
inter-event covariance must not be equated with IMAGE retrieval-error
covariance. Gaussian log-conductance and latent-decoder priors are possible
representations; neither a representation nor an implementation is adopted.

The longer-term model target is conditional: geomagnetic-activity descriptors
should condition the generated conductance distribution, allowing an end user
to request conditions and estimate a representative field and conditional
spatial covariance. The unconditional spline VAE remains the necessary
baseline before this extension.

The repository README links the published conductance estimates and associated
uncertainties at `https://doi.org/10.5281/zenodo.15579301`.

## Repository shape

- `scripts/manual_masking/`: one interactive annotation GUI plus annotation
  persistence, geometry, native loading and downstream-data adapters.
- `scripts/automatic_masking/`: the adopted Model1 CNN and Model2 U-Net data
  preparation, training and application workflows (nine implementation modules).
- `scripts/CVAE/`: the supported Hall/Pedersen and WIC conditional CVAE
  cache preparation, training, sampling and WIC-result plotting, plus
  SuperMAG-to-IMAGE conditioning preparation and time-blocked predictive-
  importance analysis. The WIC model supports scalar Kp, the full 11-feature
  candidate vector and the selected four-variable SML/IMF By/IMF Bz/dipole-
  tilt profile, plus empirical IMF-regime sampling over actual training rows.
  WIC preparation accepts manual annotations or automatic masks;
  Hall/Pedersen preparation is `prepare_cvae_hp_cache.py`.
- `icanalyzer/cvae/`: shared field-agnostic cache, model, loss, training,
  artifact and ensemble implementation for those supported CVAEs.
- `icanalyzer/wic_si12/`: isolated legacy implementation retained for older
  WIC/SI12 analysis scripts pending classification.
- `scripts/IMAGE/vae_pipeline/`: older WIC/SI12 VAE workflows, diagnostics,
  reconstruction studies and presentation scripts.
- Remaining scripts are historical or not yet classified during cleanup.
- `data/`: tracked and untracked analysis inputs and derived arrays.
- `figures/`: tracked and large untracked analysis-output trees.
- `README.md`: public description and dataset link.
- `pyproject.toml`: minimal package metadata.

## Candidate technical entry points

The canonical fixed-grid baseline is now:

- `scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`
- `icanalyzer/wic_si12/`

An alternative direct-native path is now implemented:
`scripts/IMAGE/vae_pipeline/run_conductance_pipeline.py` with
`icanalyzer/conductance.py`. It loads Hall/Pedersen and dH/dP from
`conductance_cs`, reuses the SAME spline model/training/loss and saves a distinct
native artifact, without FUV-to-conductance propagation. Uncertainty weighting
and holes remain opt-in; native model scientific quality/calibration remains
under review. A production native run now records500 epochs/50k draws.
Conductance teleconnection/reconstruction figures and the held-out browser
accept both paths; physical-S teleconnections and log-space reconstructions
are distinct statistics.

The shared annotation entry point is
`scripts/manual_masking/annotate_partial_images.py`: native46x46
conductance_cs or precipitation_cs, saved to the converted schema2 JSON.
It preserves timestamp-aligned legacy frame classes and drawn line sides across
the entire native grid. The earlier unintended old-grid clipping is corrected;
[[Full Grid Annotation Transfer]] owns the repair and verification.
Both runners accept JSON-only or whole-orbit JSON/CSV hybrid selection through
the shared annotation loader; annotations do not change the VAE architecture.

The automatic-mask native Kp-CVAE has a separate, disk-backed path:
`scripts/CVAE/prepare_cvae_hp_cache.py` followed by
`scripts/CVAE/train_automatic_cvae.py`. It uses the shared field-agnostic
`icanalyzer/cvae/` implementation and leaves the old pipelines untouched.
Preparation/cache batches have been checked;
actual training is reserved for the user. The user-trained full run is now
complete. Separate postprocess_automatic_cvae.py makes cache-aware held-out
diagnostics and disk-backed fixed-Kp ensembles/covariance without retraining.
Current State owns readiness, numerical verification and scientific limitations.

The following older filenames remain historical candidates or experiments,
not the canonical WIC/SI12 workflow:

- `scripts/Cleaned_VAE_final.py`
- `scripts/cond_est_VAE.py`
- `scripts/conductance_estimator.py`
- `scripts/IMAGE/gen_cov_iterative_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_PCA.py`
- `scripts/IMAGE/linear_reconstruction.py`

Inspect live call paths, inputs, saved models, and output provenance before
treating any of these as publication code.

## Cross-project dependencies

- `fuvpy` supplies native FUV preprocessing to `icBuilder`; it is an indirect
  dependency and should not be imported by `icAnalyzer`.
- `icBuilder` owns preprocessing, geometry, fixed-grid products, and their
  provenance.
- `icReader` is the public product-loading and schema-validation boundary for
  `icAnalyzer`.
- `icPhysics` owns reusable array-to-array precipitation and conductance
  calculations used to propagate ensembles in memory.
- A future `nandir` integration may be relevant where inverse problems are
  solved, but no integration decision has been confirmed.

## Publication context

- AGU 2026 abstract deadline: 2026-08-05.
- Associated journal paper target: December 2026.

The target journal, co-authors, review path, final figures, and validation
standard are not yet recorded.
