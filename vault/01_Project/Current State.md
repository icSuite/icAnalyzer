# Current State

Last reviewed: 2026-07-26
Repository snapshot: `main` at `5028449`, aligned with `origin/main`
Worktree before vault creation: 26 tracked modifications and 13,793 untracked
files

## Current position

**Confirmed by the user:** The intended project is an empirical model of
height-integrated Hall and Pedersen conductance using IMAGE-derived data and a
variational autoencoder. The spatial covariance representation is especially
important for later ionospheric-electrodynamics data assimilation.

The user considers this the quickest project to turn into a publication. The
AGU 2026 abstract is due 2026-08-05, and the associated paper is targeted for
December 2026.

## Verified repository evidence

- `main` tracks `origin/main`; the latest commit is `5028449` from 2025-06-16.
- Before documentation changes, 26 tracked data, figure, and script files were
  modified.
- The repository contained 13,793 untracked files: 13,758 under `figures/`, 29
  under `scripts/`, and 6 under `data/`.
- Candidate VAE and covariance scripts are present, including files under
  `scripts/IMAGE/`.
- Many scripts reference `icReader` classes or interfaces.
- No maintained automated test suite or CI workflow was found.
- On 2026-07-26, a read-only AST parse succeeded for all 39 Python files under
  `scripts/`.

## Unknown

- which script and model artifact represent the intended final VAE workflow;
- how spatial covariance is calculated, stored, and validated;
- which results quantitatively validate the conductance estimates;
- which generated figures have known inputs and configuration;
- the intended abstract claim, final figures, co-authors, and target journal;
- which uncommitted outputs should eventually be retained or versioned.

## Risks

- The large dirty worktree makes provenance and reproducibility uncertain.
- Open-ended model experimentation could consume the short abstract window.
- Existing figures and model artifacts are not self-validating.
- Analysis scripts may write large output trees or depend on local data and
  undeclared environments.

## Next action

When technical discussion resumes, perform a non-destructive publication
evidence audit:

1. identify the intended VAE and covariance workflow;
2. trace inputs, model artifacts, and generated outputs;
3. inventory existing validation evidence and candidate figures;
4. identify the minimum missing validation for the abstract;
5. formulate an evidence-backed title and abstract outline.
