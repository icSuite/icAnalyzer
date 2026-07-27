# Handoff - Latest

Last updated: 2026-07-26
Repository snapshot: `main` at `5028449`

## Project state

`icAnalyzer` analyzes IMAGE-derived height-integrated Hall and Pedersen
conductance. The intended publication project is a variational-autoencoder
empirical model with particular emphasis on spatial covariance for later
ionospheric-electrodynamics data assimilation.

The project is the highest known near-term publication priority. The AGU 2026
abstract is due 2026-08-05; the associated paper is targeted for December 2026.

## Live repository snapshot

Before the documentation-only vault creation:

- `main` was aligned with `origin/main` at `5028449`;
- 26 tracked files were modified;
- 13,793 files were untracked, dominated by generated figure trees;
- candidate VAE, covariance, conductance-estimation, PCA, and neural-network
  scripts were present;
- no maintained automated test suite or CI workflow was found;
- all 39 Python files under `scripts/` passed a read-only AST parse.

Preserve the dirty worktree. Inspect live Git again before technical work.

## Next action

Perform a non-destructive publication evidence audit before additional model
experimentation:

1. establish the intended VAE and covariance workflow;
2. trace inputs, saved models, and outputs;
3. identify existing quantitative validation and candidate figures;
4. define only the minimum missing validation;
5. draft an evidence-backed abstract outline.

## Known risks

- Canonical scripts and final model artifacts are not identified.
- Output provenance is unclear across the large dirty worktree.
- Spatial-covariance calculation and validation remain unverified.
- Open-ended experimentation risks displacing abstract preparation.
- Co-authors, target journal, and review requirements are unknown.

## Relevant entry points

- `README.md`
- `scripts/Cleaned_VAE_final.py`
- `scripts/cond_est_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_PCA.py`
- `scripts/IMAGE/linear_reconstruction.py`
- `vault/01_Project/Current State.md`

## Portfolio impact

- Central update needed: No
- Changes: None
- Sync summary: `icAnalyzer` remains the active highest-priority project with
  the 2026-08-05 AGU abstract deadline and December 2026 paper target. The next
  portfolio decision remains selecting credible validation evidence and
  figures for the conductance model and spatial covariance.
