# Handoff - Latest

Last updated: 2026-07-27
Repository snapshot: `main` at `f0a139a`, aligned with `origin/main`

## Project state

`icAnalyzer` analyzes IMAGE-derived height-integrated Hall and Pedersen
conductance. The intended publication project is a variational-autoencoder
empirical model with particular emphasis on spatial covariance for later
ionospheric-electrodynamics data assimilation.

The project is the highest known near-term publication priority. The AGU 2026
abstract is due 2026-08-05; the associated paper is targeted for December 2026.

## Live repository snapshot

- `main` and `origin/main` are at `f0a139a` (`Checkpoint after prolonged pause
  of project`, 2026-07-27).
- The prior dirty state was incorporated into that checkpoint; the worktree
  was clean before the current vault-only memory update.
- All 43 Python files under `scripts/` pass a read-only AST parse at the
  annotation-tool checkpoint.
- No maintained automated test suite or CI workflow was found.

## Next action

Have the user retry
`scripts/IMAGE/vae_pipeline/annotate_partial_images.py` on orbit 169 using an
alternate temporary annotation JSON. It provides a 6x6 live viewer, enlarged
lasso editor, sparse forward-held masks, atomic autosave, automatic
resume, and explicit orbit/frame restart options. The first trial exposed
ambiguous boundary wording; the controls now distinguish FIRST and LAST. With
the existing fully good `0:51` seed, select the final partially useful frame
and change only LAST usable, creating partial frames `52:last`. Confirm image
orientation, lasso semantics, interaction speed, and the revised interval
controls before starting large-scale annotation.

The second trial exposed and prompted fixes for a mask-editor display bug and
the propagation policy. Qt5Agg/TkAgg editors are now explicitly shown and
raised, with errors displayed in the main panel. Masks are no longer
interpolated: an initial mask is required at the first partial frame and stays
active until a later keyframe replaces it.

The editor additionally supports additive straight-line cuts. Each new valid
half-plane intersects the prior valid mask; the newest line can be flipped or
undone, and all cuts can be cleared. Complete line sets and the raster mask
survive JSON reload. The existing lasso remains available for concave,
disconnected, or otherwise irregular regions. Dividing-line mode is the
default for new keyframes, while existing lasso keyframes reopen in lasso mode.
Divider input accepts either dragging or two endpoint clicks on any of the four
aligned image panels; this fixed the user's failed frame-58 attempt in a
real-callback regression.

Read-only inspection confirms that the user's trial JSON is saving orbit 169
and its four keyframes correctly. It is still under `/tmp`, so move it to the
default durable project annotation path before relying on it across reboot or
temporary-file cleanup.

After that trial, define how manual validity combines with finite-data masks
and continuous `w`, then select a versioned processed-data format that
preserves orbit/time, provenance, condition variables, split assignment, and
normalization metadata.

Next, scaffold four explicit stages:

1. import, annotate/clean, visualize, and save processed data;
2. load processed data, train the VAE, and save a complete checkpoint;
3. load the checkpoint and perform initial reconstruction/covariance analysis;
4. perform advanced comparison with other conductance models.

After the foundation, make the unconditional VAE defensible before extending
it to a geomagnetic-condition conditional VAE. The user explicitly chose to
skip iterative PCA.

## IMAGE audit findings

- `doing_it_manually_log_spline.py` is the newest and strongest architecture.
  Its main coefficient-space OI is coherent, but its later latent-assimilation
  experiment has a physical/normalized-log unit mismatch.
- `primary_log_spline_vae.py` is the working copy. It corrects cross-channel
  covariance normalization and replaces the invalid latent-assimilation path
  with a normalized-log10 formulation that returns physical conductance.
- The user reports that its average training loss falls from about 397 to 125
  by epoch 352 and then remains there through epoch 1000, closely matching the
  source experiment. This has not been independently rerun. The unchanged
  training path explains the similarity, and repeated learning-rate reduction
  is the leading explanation for the late plateau.
- `doing_it_manually_spline.py` is the preceding physical-space spline
  experiment. It establishes the spline decoder but does not justify replacing
  the log-space version.
- `doing_it_manually.py` is the strongest direct-pixel predecessor and
  contains useful comparison and error-evaluation ideas omitted later.
- `gen_cov_iterative_PCA.py` is a potentially useful baseline prototype, not a
  validated baseline implementation.
- `gen_cov_iterative_VAE.py` is a direct-pixel covariance prototype with a
  correctly formed full-state correlation calculation but no assimilation.
- `linear_reconstruction.py` is an intermediate direct-pixel experiment and
  is superseded by the manual variants.
- The three earliest scripts currently fail at the `helpers.load_data`
  interface because they expect five return values rather than six.
- Cross-block teleconnection normalization is mathematically invalid in all
  three historical manual variants. It is corrected only in the primary copy.
- The user changed the IMAGE loader so a missing good start becomes frame
  zero. They report that the loaded set increased from 3,664 to 11,136 images
  and that raising the primary training subset from 3,000 to 10,000 caused a
  substantial slowdown. These results have not been independently regenerated.
- The new annotation tool was verified headlessly against real orbit 169 with
  a temporary JSON. The first desktop trial found confusing boundary
  terminology, and the second found that the editor window was not shown under
  Agg-suffixed GUI backends. Both fixes and the new forward-held mask semantics
  pass headless regressions. The new dividing-line/flip geometry also passes a
  JSON round trip but still needs a desktop trial.

## Known risks

- The working architectural candidate is identified, but no canonical
  executable workflow or validated model artifact exists.
- Spatial covariance is implemented but not quantitatively validated.
- No orbit-separated split, numerical assimilation benchmark, uncertainty
  calibration, or saved run provenance exists.
- Hard-coded paths, process-wide `chdir`, global decoder variables, silent
  loader exceptions, and monolithic scripts impede reproducibility.
- Open-ended experimentation risks displacing abstract preparation.
- Co-authors, target journal, and review requirements are unknown.
- Manual annotation could become the dominant cost unless partial-region
  labeling is efficient and the label schema is fixed before large-scale work.
- Historical zero/blank segment semantics remain imperfect, although the live
  IMAGE helper and the new annotation seeding now both interpret a blank good
  start as frame zero when the stop is a finite positive value.
- Conditional covariance will be poorly constrained in sparsely represented
  activity regimes unless condition-space coverage is measured.

## Relevant entry points

- `README.md`
- `scripts/Cleaned_VAE_final.py`
- `scripts/cond_est_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_VAE.py`
- `scripts/IMAGE/gen_cov_iterative_PCA.py`
- `scripts/IMAGE/linear_reconstruction.py`
- `scripts/IMAGE/doing_it_manually.py`
- `scripts/IMAGE/doing_it_manually_log_spline.py`
- `scripts/IMAGE/primary_log_spline_vae.py`
- `scripts/IMAGE/vae_pipeline/annotate_partial_images.py`
- `scripts/IMAGE/vae_pipeline/README.md`
- `vault/01_Project/Current State.md`

## Portfolio impact

- Central update needed: No
- Changes: None. This checkpoint implements the already-recorded annotation
  prerequisite without changing project priority, deadline, or publication
  direction.
