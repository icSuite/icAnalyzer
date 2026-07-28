# Decision Log

Last reviewed: 2026-07-27

## 2026-07-26 - Use the standard project vault

**Decision:** Store project memory in the repository-root `vault/`, with root
`AGENTS.md` as the machine entry point.

**Rationale:** The user wants one predictable project-memory path across every
repository and automatic continuity between project and central sessions.

The vault uses automatic milestone checkpoints and a structured
`Portfolio impact` handoff. This documentation change does not imply
scientific, model, or publication progress.

## 2026-07-26 - Preserve the publication concept

**Confirmed context:** The project aims to create a VAE-based empirical model
of IMAGE-derived height-integrated Hall and Pedersen conductance. Spatial
covariance is especially important for the user's broader
ionospheric-electrodynamics data-assimilation work.

This records the user's stated research direction. It does not select a
specific script, covariance formulation, validation result, or final abstract
claim.

## 2026-07-27 - Preserve the working VAE lineage

**Working direction confirmed by the user:** Treat
`scripts/IMAGE/doing_it_manually_log_spline.py` as the strongest architectural
starting point, while retaining the following lineage for later
implementation:

1. keep the log-space spline VAE and joint coefficient covariance;
2. keep the control-space optimal-interpolation formulation;
3. discard or repair the current latent-assimilation experiment;
4. restore the Hall-only, Pedersen-only, and joint-observation experiments
   from `scripts/Cleaned_VAE_final.py`;
5. restore adaptive rank selection and posterior uncertainty;
6. add orbit-separated quantitative validation and an independent baseline.

This is a working lineage, not yet a declaration that either script is
canonical or that its scientific results are validated. Audit the remaining
`scripts/IMAGE/` experiments before changing the implementation.

## 2026-07-27 - Establish the primary log-spline script

**Decision:** Use `scripts/IMAGE/primary_log_spline_vae.py`, copied from
`doing_it_manually_log_spline.py`, as the working implementation. Preserve the
source experiment unchanged.

The primary script:

- normalizes Hall-Pedersen cross-covariance using variances from the matching
  Hall and Pedersen marginal blocks;
- performs latent assimilation in normalized log10 space, including
  first-order transformation of physical observation errors;
- converts assimilated spline fields back to non-negative physical
  conductance.

**Scope decision confirmed by the user:** Do not pursue the iterative-PCA
baseline. Defer the independent-versus-joint observation experiments and the
larger validation/uncertainty/reproducibility work for a later task.

## 2026-07-27 - Sequence development around data quality and a conditional VAE

**Direction confirmed by the user:** The next development phase has four
linked objectives:

1. overhaul the training data through manual quality labeling, while retaining
   partially useful images by marking and ignoring only their bad regions;
2. study and revise the current VAE formulation until its design choices are
   theoretically defensible;
3. develop a conditional VAE driven by geomagnetic-activity descriptors, so
   users can request conditions and generate a conditional conductance field
   and covariance;
4. split the monolithic workflow into data preparation, model training,
   initial analysis, and advanced comparison stages.

**Dependency decision:** Establish the annotation/data contract and modular
pipeline first, then build a defensible unconditional baseline, then extend it
to a conditional VAE. Do not change data selection and model class
simultaneously when a controlled comparison is possible.

The exact condition variables, annotation interface, processed-data format,
and model architecture remain to be selected from inspected data and
scientific requirements.

## 2026-07-27 - Use resumable interval and keyframe-mask annotation

**Implementation decision:** Represent each reviewed orbit with a single
usable envelope and an optional fully good core. Frames between those
boundaries are full, partial, or bad without requiring a separate label for
every image.

For partial regions, store sparse 36x36 valid-mask keyframes and their lasso
vertices. Reuse the nearest keyframe outside the keyframe span and interpolate
between keyframes through signed-distance fields. Require a keyframe in every
partial interval before an orbit can be marked complete.

Keep `data/good_orbit_segments.csv` read-only and seed unambiguous historical
good intervals from it. Store new annotations, explicit review states, and the
resume cursor in a separate schema-versioned JSON file, saved atomically after
each interaction. Rasterization into a processed training dataset remains a
separate pipeline stage.

## 2026-07-27 - Hold masks forward instead of interpolating

**Decision confirmed by the user:** A mask keyframe becomes active on its
frame and remains unchanged on subsequent frames until another keyframe
explicitly replaces it. Do not interpolate mask shape between keyframes.

Every partial interval must therefore have its initial mask on its first
frame. This supersedes the temporal-interpolation part of the earlier
annotation decision while retaining its orbit boundaries, sparse keyframes,
separate JSON persistence, and resumable cursor.

## 2026-07-27 - Support lasso and additive half-plane mask geometry

**Implementation direction requested by the user:** Retain lasso masks for
irregular regions, but support the common case in which a straight diagonal
separates a bad half-image from a valid half-image.

The line tool stores two endpoints and the selected valid side for every cut.
Multiple lines combine by intersection, so each new valid half-plane further
trims the retained region. The user may flip or undo the newest line because
line direction should not require mental side calculation. The resulting
36x36 mask is stored alongside the geometry.

Both geometry types use the same forward-held keyframe semantics.
Dividing-line mode is the default for new keyframes; an existing lasso reopens
in lasso mode. Lasso remains the fallback for concave or disconnected valid
regions that an intersection of half-planes cannot represent.
