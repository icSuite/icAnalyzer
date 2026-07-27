# Project Brief

Last reviewed: 2026-07-26

## Purpose

`icAnalyzer` documents statistical analysis of height-integrated ionospheric
Hall and Pedersen conductance estimates derived from IMAGE satellite data.

**Confirmed by the user:** The intended publication project is an empirical
conductance model using a variational autoencoder. Spatial covariance is
particularly important for the user's broader work on data assimilation of
ionospheric electrodynamics.

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

The following filenames are candidates, not confirmed canonical workflows:

- `scripts/Cleaned_VAE_final.py`
- `scripts/cond_est_VAE.py`
- `scripts/conductance_estimator.py`
- `scripts/IMAGE/gen_cov_iterative_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_PCA.py`
- `scripts/IMAGE/linear_reconstruction.py`

Inspect live call paths, inputs, saved models, and output provenance before
treating any of these as publication code.

## Cross-project dependencies

- `icBuilder` produces the conductance products.
- `icReader` provides reader code/classes used by analysis scripts.
- A future `nandir` integration may be relevant where inverse problems are
  solved, but no integration decision has been confirmed.

## Publication context

- AGU 2026 abstract deadline: 2026-08-05.
- Associated journal paper target: December 2026.

The target journal, co-authors, review path, final figures, and validation
standard are not yet recorded.
