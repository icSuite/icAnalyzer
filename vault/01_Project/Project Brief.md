# Project Brief

Last reviewed: 2026-09-25

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

The longer-term model target is conditional: geomagnetic-activity descriptors
should condition the generated conductance distribution, allowing an end user
to request conditions and estimate a representative field and conditional
spatial covariance. The unconditional spline VAE remains the necessary
baseline before this extension.

The repository README links the published conductance estimates and associated
uncertainties at `https://doi.org/10.5281/zenodo.15579301`.

## Repository shape

- `scripts/`: PCA, neural-network, VAE, conductance-estimation, substorm, data
  preparation, and visualization experiments.
- `scripts/IMAGE/`: IMAGE-specific reconstruction and covariance candidates.
- `data/`: tracked and untracked analysis inputs and derived arrays.
- `figures/`: tracked and large untracked analysis-output trees.
- `README.md`: public description and dataset link.
- `pyproject.toml`: minimal package metadata.

## Candidate technical entry points

The canonical fixed-grid baseline is now:

- `scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`
- `icanalyzer/wic_si12/`

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
