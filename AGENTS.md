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

## Scientific coding style

- Write for a small research group. The expected reader is a student or
  scientist who should be able to follow the calculation from top to bottom.
- Write for a scientist reading the code interactively in an editor. Make
  workflows visually scannable with `#%%` sections where appropriate, blank
  lines between conceptual stages, and short comments that identify each block.
- Keep the main calculation linear and visible from top to bottom. Prefer
  familiar intermediate variables and explicit operations over nested
  expressions, generic plumbing, or compressed control flow.
- Extract a helper when it represents a distinct calculation or removes
  meaningful repetition. Do not hide a few obvious sequential steps behind an
  abstraction merely to shorten the main function.
- Comments may serve as navigational headings even when the underlying Python
  is straightforward. Treat formatter conventions and line-length targets as
  secondary to human readability, while avoiding ambiguous or unwieldy code.
- Use nearby code as the stylistic baseline, not as a ceiling. Do not copy weak
  patterns blindly: identify scientific, numerical, or code choices that could
  be improved, explain the tradeoff, and propose a clearer or safer alternative.
- Adopt improvements when they materially improve correctness,
  reproducibility, clarity, or demonstrated performance. Do not add complexity
  merely because it is conventional in large production systems.
- Let complexity follow the science, numerical method, or actual reuse
  requirements. Prefer direct functions, NumPy arrays, ordinary loops and
  dictionaries, and keep the main calculation visible in execution order.
- Unless current requirements justify them, avoid dataclasses, manager or
  factory classes, generic schemas, version and compatibility frameworks,
  checkpoint/resume machinery, and speculative extension points.
- Retain scientific rigor: make units, coordinates, assumptions, provenance,
  and uncertainty explicit, and add focused tests or reference comparisons for
  consequential calculations.
- Scale packaging, validation, documentation, and abstractions to the code's
  real reuse. A reusable package may justify more structure, but that structure
  should solve a current, explained need.
- If a nominally small feature grows beyond roughly 200 lines or more than two
  new source files, pause and explain why before continuing.

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
