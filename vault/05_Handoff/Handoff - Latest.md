# Handoff - Latest

Last updated: 2026-09-25
Repository snapshot: `main` at `7c71591`, aligned with `origin/main`

## Project state

`icAnalyzer` analyzes IMAGE-derived height-integrated Hall and Pedersen
conductance. The intended publication project is a variational-autoencoder
empirical model with particular emphasis on spatial covariance for later
ionospheric-electrodynamics data assimilation.

The user has described this as the quickest project to turn into a publication.
The AGU 2026 abstract deadline was 2026-08-05, but the repository does not
record whether an abstract was submitted or accepted. The associated paper is
still targeted for December 2026 in the latest confirmed project context.

On 2026-09-25 the user set the desired technical direction: update icAnalyzer
for the regenerated upstream dataset and build a WIC/SI12 VAE through latent
sampling, WIC/SI12 covariance, propagation of sampled corrected WIC to
Hall/Pedersen, joint conductance covariance, and sparse conductance
reconstruction.

The first propagation baseline now assumes constant 2 keV proton and electron
mean energies. Neither energy is sampled in the first ensemble.

The user has coordinated the upstream input contract: Product 2 will carry
corrected WIC, corrected SI13, SI12, and their uncertainties into
`precipitation_cs`, which icAnalyzer will ingest. Corrected WIC must not be
proton corrected again in icAnalyzer; no separate `fuv_cs` is needed.

## Live repository snapshot

- `main` and the locally recorded `origin/main` are both at `7c71591`, the
  2026-07-28 checkpoint containing the primary log-spline working copy, loader
  correction, annotation implementation, initial annotations, and memory.
- The tracked worktree has nine modified files: `AGENTS.md`,
  `data/partial_image_annotations.json`, `scripts/IMAGE/helpers.py`, its
  tracked bytecode cache, `scripts/make_overview_images.py`, `Current State`,
  `Project Brief`, the decision log, and this handoff. Relevant untracked
  paths are `figures/ncso_new/`, `literature/`,
  `scripts/IMAGE/step_1_vae_wic_s12.py`, and `vault/02_Algorithm/`.
- The annotation document contains 53 represented orbit records: 49 complete
  and four in progress. Its cursor is orbit 167, and it contains 469 keyframes
  across 50 orbits. Compared with `HEAD`, 46 orbit records changed and 45 moved
  from in progress to complete.
- Schema, boundary, completion, mask-decoding, and recorded-valid-fraction
  checks succeeded for all represented annotation records and all 469 masks.
- All 44 Python files under `scripts/` passed a read-only AST parse on
  2026-09-25. The active analysis and overview scripts were not executed.
- No maintained automated test suite or CI workflow was found.

## Next action

1. Confirm the proposed Git transition: preserve the current mixed worktree on
   a dated archive branch, exclude generated/cache artifacts, then create a
   dedicated WIC/SI12 VAE branch from that checkpoint. No branch or commit has
   yet been made.
2. Verify the coordinated Product-2 update after it lands: corrected WIC,
   corrected SI13, SI12, and their uncertainties must survive into
   `precipitation_cs` and load through icReader with provenance intact. Add an
   explicit guard against applying proton correction twice. Audit the current
   icPhysics/icBuilder fixed-electron-energy path before proposing any API
   change, and require frozen numerical parity if a change is later approved.
3. Prove one representative orbit end to end: load a 46x46 product through
   `icreader.load()`, train/reconstruct and sample corrected WIC/SI12,
   propagate sampled corrected WIC, form both covariance families, and
   reconstruct sparse conductance with explicit observation error.
4. Resume annotation from orbit 167 as bounded background work. Convert the
   36x36 annotation geometry non-destructively before applying it to a new
   grid; the 49 completed represented records are not a whole-dataset
   completion metric.

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
- Production annotation is underway in
  `data/partial_image_annotations.json`. Read-only inspection on 2026-09-25
  found 53 represented orbit records, 49 complete and four in progress, with
  the cursor at orbit 167 and 469 stored keyframes.
- The dirty `load_data_wic` helper exposes WIC/S12 means and standard
  deviations for `step_1_vae_wic_s12.py`; that script currently discards the
  standard deviations and carries the means through Hall/Pedersen-labelled
  variables. This is an unexecuted experiment, not a verified result.
- `make_overview_images.py` now points at `conductance_new` and new overview
  output locations. The path change was not executed or provenance-checked.

## Known risks

- The working architectural candidate is identified, but no canonical
  executable workflow or validated model artifact exists.
- Spatial covariance is implemented but not quantitatively validated.
- No orbit-separated split, numerical assimilation benchmark, uncertainty
  calibration, or saved run provenance exists.
- Hard-coded paths, process-wide `chdir`, global decoder variables, silent
  loader exceptions, and monolithic scripts impede reproducibility.
- The AGU abstract deadline has passed, and submission/acceptance status is not
  recorded in the repository.
- Co-authors, target journal, and review requirements are unknown.
- Information-preserving manual annotation is expected to take weeks and will
  continue as a bounded daily background task.
- Historical zero/blank segment semantics remain imperfect, although the live
  IMAGE helper and the new annotation seeding now both interpret a blank good
  start as frame zero when the stop is a finite positive value.
- Conditional covariance will be poorly constrained in sparsely represented
  activity regimes unless condition-space coverage is measured.
- The only public Zhang-Paxton Python implementation found so far is partial
  and unlicensed. It cannot supply the needed `E0` field or be used unchanged
  in a publication workflow.
- The WIC/S12 loader silently catches all per-orbit exceptions, and its
  consumer does not yet define how measurement standard deviations enter the
  model or validation.
- The current modular CS products use a fixed 46x46 grid, while the legacy VAE
  and annotation paths assume 36x36.
- The current live `precipitation_cs` reader has not yet been verified to
  expose the newly agreed corrected-WIC/corrected-SI13/SI12 fields. Treat the
  interface as pending until the upstream implementation and schema tests
  land.
- Applying `proton_correct_images()` to `precipitation_cs.wic_corrected` would
  double-correct WIC and invalidate the downstream ensemble.
- The existing public icPhysics functions may or may not compose into the
  required fixed-energy ensemble path. Changing that shared package without a
  call-site audit and frozen icBuilder parity evidence risks altering upstream
  products; no such change is currently approved.
- The scientific covariance target must distinguish geophysical variability
  from detector noise. The first baseline resolves the energy condition by
  fixing both proton and electron mean energies at 2 keV; later energy-model
  sensitivity remains deferred.

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
- `scripts/IMAGE/step_1_vae_wic_s12.py`
- `scripts/IMAGE/helpers.py`
- `scripts/make_overview_images.py`
- `scripts/IMAGE/vae_pipeline/annotate_partial_images.py`
- `scripts/IMAGE/vae_pipeline/README.md`
- `vault/02_Algorithm/WIC-SI12 VAE Pipeline Plan.md`
- `vault/01_Project/Current State.md`

## Portfolio impact

- Central update needed: Yes
- Changes: The desired implementation path is now WIC/SI12-first, with paired
  sample propagation to Hall/Pedersen and joint sparse reconstruction. This is
  now expected to ingest corrected WIC, corrected SI13, SI12, and uncertainty
  fields from the updated `precipitation_cs`; no separate `fuv_cs` is needed.
  The first baseline fixes upstream proton energy and downstream electron
  energy at 2 keV, and no icPhysics change is approved without an icBuilder
  compatibility audit. The user has proposed preserving the current state and
  moving the work to a dedicated VAE branch; execution awaits confirmation.
- Sync summary: Project priority and December 2026 target are unchanged. The
  immediate cross-project work is to stabilize the fixed-grid WIC/SI12 reader
  and propagation interfaces before training, while annotation continues in
  the background. The AGU abstract outcome remains unknown.
