# CVAE Compute Scaling Experiments

Last reviewed: 2026-11-06
Status: Experiment 1 implemented as an opt-in; scientific evaluation pending

## Motivation

The current WIC CVAE has about 317,000 parameters and uses roughly 22% peak
RTX6000 Ada compute at batch264 after moving the cache to RAM and moving
artificial-hole generation to CUDA. Spare GPU capacity is not itself evidence
that the model needs more capacity. Additional computation should address a
measured limitation in reconstruction, latent use, covariance or epistemic
uncertainty.

Keep the current batch264 run as the baseline. Compare candidates on the same
orbit split and optimizer-step budget using held-out reconstruction, artificial-
hole reconstruction, active latent dimensions, spatial power/covariance and
fixed-Kp ensemble behavior.

## Prioritized experiments

1. **Multiple latent samples per training image.** Encode once, draw and decode
   4--8 latent realizations in parallel, and average their reconstruction terms
   while retaining the analytic KL term. This preserves batch size and optimizer
   updates, reduces Monte Carlo gradient noise and converts spare GPU capacity
   directly into a better-estimated VAE objective. Test ordinary multi-sample
   ELBO estimation before an importance-weighted objective.

   Implemented on 2026-11-06 as `--latent-samples K` in both supported
   trainers. The default remains `1`. For `K>1`, one encoder result feeds a
   flattened `batch*K` decoder call; reconstruction is averaged over K and the
   analytic KL is counted once. Held-out evaluation uses the same K and the run
   artifact records it. Focused CPU checks pass; CUDA performance and the
   scientific comparison with the one-sample baseline remain pending.

2. **Multiple model seeds and sample-level ensembling.** Train several complete
   CVAEs and pool equal numbers of generated samples from each artifact. This
   represents parameter uncertainty that latent draws from one fitted model do
   not capture. Do not average independently trained weights. Concurrent runs
   may improve aggregate GPU use, but scientific comparison remains the reason
   for the ensemble.

3. **Diagnose latent saturation before increasing capacity.** Measure per-
   dimension KL and variation of posterior means on held-out data. Increase
   latent dimension from36 to72 only if most dimensions are active and
   reconstruction remains capacity-limited. Current approximate parameter
   counts are317k at latent36,567k at72 and733k at96.

4. **Test a spatially richer decoder.** Compare the legacy18x18 internal decoder
   feature grid with30x30 while retaining30 cubic-spline control points. Proceed
   only if baseline residuals or spatial spectra show missing fine auroral
   structure. Increase spline control points only with evidence that spline
   smoothness, rather than observation noise, is limiting the result.

5. **Consider richer latent distributions only after the above tests.** A
   Kp-conditioned mixture prior, normalizing-flow posterior or hierarchical
   global/regional latents could represent multimodality, but each changes the
   scientific model substantially and requires stronger validation.

## Supporting engineering options

- Mixed precision and `torch.compile` may reduce runtime but are engineering
  optimizations, not reasons to change model capacity.
- A GPU-resident contiguous training set and asynchronous CUDA prefetch could
  reduce remaining per-batch overhead while preserving batch264 semantics.
- Increasing batch size is the simplest utilization improvement, but changes
  optimizer updates per epoch and requires the existing beta rescaling.
- Synchronous distributed data parallel is the standard multi-GPU solution.
  Independent checkpoint averaging or loosely synchronized local SGD is not a
  drop-in replacement for one-model training.

## Decision gates

- If many latent dimensions are inactive, do not enlarge the latent space.
- If train reconstruction is good but validation is worse, do not add model
  capacity.
- If covariance or fixed-Kp samples vary materially across seeds, prioritize
  the seed ensemble.
- If one-sample gradients are noisy while capacity diagnostics are healthy,
  prioritize multi-sample latent training.
- If residual maps systematically lack small-scale structure, test the30x30
  decoder grid before widening the full network.
