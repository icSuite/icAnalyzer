# Repository guidance

## Purpose

`icAnalyzer` contains statistical analysis and empirical modeling of
height-integrated ionospheric Hall and Pedersen conductance derived from IMAGE
satellite products. The active publication concept uses a variational
autoencoder and places particular importance on spatial covariance for later
ionospheric-electrodynamics data assimilation.

## Project memory

Before substantive work, read:

1. `vault/01_Project/Current State.md`
2. `vault/05_Handoff/Handoff - Latest.md`
3. `vault/01_Project/Project Brief.md`

Then read relevant decisions or dated history only as needed.

Use this source-of-truth order:

1. live code, Git state, inspected data, tests, and regenerated results;
2. `Current State.md` and `Handoff - Latest.md`;
3. durable decisions and project notes;
4. dated session history.

If memory conflicts with live evidence, follow the repository and repair the
live notes when the task changes project understanding.

## Working rules

- Preserve all unrelated and pre-existing worktree changes.
- Do not clean, discard, stage, or broadly rewrite the large dirty worktree
  during orientation.
- Treat candidate VAE, PCA, neural-network, conductance-estimation, substorm,
  and covariance scripts as experiments until their role and provenance are
  verified.
- Do not infer the canonical model or final figures from filenames alone.
- Do not regenerate data, figures, overview images, or model artifacts unless
  their inputs, configuration, and intended output scope are explicit.
- Keep scientific claims, hypotheses, and unverified interpretations clearly
  separated.
- Preserve compatibility with the `icReader` interfaces and `icBuilder`
  product schema unless a task explicitly coordinates a cross-project change.
- Treat unpublished research context and local data products as private.
- Use repository-relative paths in documentation.

## Verification

No automated test suite or CI workflow was found at the 2026-07-26 review.
Files named `Test_*` or `*_test.py` under `scripts/` are research scripts, not
evidence of a maintained automated suite.

A read-only syntax check that does not import the project or write bytecode is:

```bash
PYTHONDONTWRITEBYTECODE=1 python -c "import ast, pathlib; files=sorted(pathlib.Path('scripts').rglob('*.py')); [ast.parse(p.read_text(), filename=str(p)) for p in files]; print(f'parsed {len(files)} files')"
```

Do not run analysis scripts merely for orientation. Many scripts depend on
local data and may write large output trees, figures, or model artifacts.

## Automatic memory checkpoints

Project-memory maintenance is a default responsibility. Do not wait for the
user to request a vault update or announce that a session is ending.

Checkpoint after a verified fix or result, a durable scientific or
implementation decision, a changed blocker or next action, and any milestone
that would otherwise leave important understanding only in the conversation.

At a meaningful checkpoint:

1. create `vault/04_Sessions/YYYY-MM-DD.md` only when historical detail is
   worth preserving;
2. rewrite `Current State.md` when verified project state changed;
3. append only durable choices to the decision log;
4. replace obsolete content in `Handoff - Latest.md`;
5. add an algorithm note only when scientific interpretation is sufficiently
   verified to be worth preserving;
6. refresh the handoff's `Portfolio impact` section, using `Central update
   needed: No` when no portfolio-level information changed.

Do not write raw logs, transient speculation, or unchanged state into the
vault. An explicit read-only or no-file-changes request disables automatic
memory writes for that task. Do not edit the central second brain directly;
communicate portfolio changes through the latest handoff.
