# Handoff - Latest

Last updated: 2026-09-28
Repository snapshot: `vae/wic-si12` at `9dbab1e`; `main` remains at `7c71591`

## Outcome

The first coherent WIC/SI12-to-conductance VAE baseline is implemented and
committed. The old mixed worktree was preserved on
`archive/pre-wic-si12-vae-2026-09-25`; development now lives on
`vae/wic-si12`.

The canonical command is documented in
`scripts/IMAGE/vae_pipeline/README.md` and implemented by
`scripts/IMAGE/vae_pipeline/run_wic_si12_pipeline.py`.

The workflow:

```text
icreader.load(precipitation_cs)
  -> corrected-WIC/SI12 tensors, masks, uncertainty, grid, provenance
  -> orbit-held-out mask-aware VAE training and artifact reload
  -> paired prior latent samples and WIC/SI12 anomaly factor
  -> corrected WIC only, fixed 2 keV electron propagation
  -> joint Hall/Pedersen anomaly factor
  -> Hall-only, Pedersen-only, or joint sparse Gaussian update
```

Corrected WIC is already proton corrected upstream. icAnalyzer does not call a
proton-correction routine and sampled SI12 is not subtracted a second time.
Both proton and electron mean energies are fixed at 2 keV in this baseline.

## Evidence

- icAnalyzer commit: `5f6de08 Implement WIC SI12 VAE baseline`.
- Focused tests: 18 passed.
- Syntax: 56 Python files parsed.
- Live-contract smoke: three generated Product-2 orbits passed through current
  icBuilder and icReader code, one training epoch, held-out evaluation,
  artifact reload, eight latent samples, physical propagation, covariance,
  and sparse reconstruction.
- Repeated CPU smoke metrics were identical at the same seed.
- Upstream review found `wic_corrected`, `dwic_corrected`,
  `si13_corrected`, `dsi13_corrected`, `si12`, and `dsi12`, each in counts on
  `(time, 46, 46)`. The earlier synthetic contract fixture used SI12 and
  constant-2-keV proton provenance; the copied production files do not.
- Compatibility review found `hall()` and `ped()` already broadcast arbitrary
  leading dimensions and reproduced the frozen 2 keV reference fixture.

The smoke demonstrates execution and interface coherence only. It is not a
trained scientific result.

## Important boundaries

- The contract changes are committed as icBuilder `38b5afb` and icReader
  `3a3b73f`. The copied files report producer version `38b5afb+worktree`.
- No icPhysics code was changed. The local adapter freezes the current
  `fWm(2.0) == 477.5` response, then uses public `hall()` and `ped()`. Because
  `fWm` is private, the requested public grid-independent batched propagation
  boundary remains a separate approval and compatibility task.
- Propagation occurs after CS gridding. Existing `conductance_cs` instead
  averages detector-space nonlinear conductance, so equality between the two
  products is neither expected nor a valid regression criterion.
- Measurement uncertainty is used for loss weighting and observation error;
  latent covariance remains a separately named geophysical component.
- Generated corrected WIC is nonnegative by decoder construction. The workflow
  still rejects any negative physical sample rather than clipping silently.

## Production-corpus audit and next decision

The copied directory now contains 1,378 `precipitation_cs` files, 358,439
frames, and about 31 GiB. All files passed a header audit for product type,
schema, grid, dimensions, count units, and the six coordinated camera fields.

Training has deliberately not started. All 1,378 files declare
`proton_energy_model="hardy"` and lack `proton_energy_constant`; the confirmed
baseline requires upstream constant 2 keV proton correction. The strict loader
therefore raises the expected contract error. The user must choose between:

1. regenerating/copying a constant-2-keV Product-2 corpus; or
2. explicitly revising the baseline to accept Hardy-corrected WIC while
   retaining fixed 2 keV electron propagation downstream.

A stratified 32-orbit value audit otherwise supports the implementation:
7,011 of 7,960 frames had jointly valid cells, corrected WIC was nonnegative,
uncertainties were nonnegative, and SI12 contained valid negative
background-subtracted counts requiring the signed transform.

After resolving proton provenance, run a representative real-orbit baseline
and inspect:

1. split counts, finite support, and WIC/SI12 distributions;
2. tiny-subset overfit and held-out reconstruction;
3. prior-sample calibration and covariance convergence across seeds/counts;
4. Hall-only, Pedersen-only, and joint realistic sparse geometries; and
5. prior-mean and independent-channel baselines.

Annotation conversion is also open. The annotation JSON contains 53 orbit
records and 469 valid keyframes, but the old 36x36 source grid files are absent
at their recorded path. Recover those coordinates before writing the agreed
non-destructive Cubed-Sphere converter; do not rescale pixel indices.

## Portfolio impact

- Central update needed: Yes
- Changes: The production corpus is now present and structurally valid, but it
  uses Hardy rather than the confirmed constant-2-keV proton model. Scientific
  training is blocked on that explicit baseline decision. A
  public icPhysics propagation boundary and coordinate-based annotation
  conversion remain explicit follow-up decisions.
- Priority/deadline: unchanged. The December 2026 paper target remains the
  latest recorded target; the AGU abstract outcome is still unknown.
