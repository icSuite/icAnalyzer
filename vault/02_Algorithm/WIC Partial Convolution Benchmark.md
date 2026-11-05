# WIC partial-convolution mask benchmark

Last reviewed: 2026-10-13
Status: Controlled single-frame ablation evaluated; no model adoption
Repository: `vae/wic-si12` at `be31406` with existing worktree changes preserved

## Question and design

Does replacing ordinary U-Net encoder convolutions with partial convolutions
help annotation masks specifically where corrected WIC has missing pixels?
The current user-trained topology-weight-0.1 checkpoint is the reference;
`data/training_data/U_net_weight_0p1.pt` has SHA256
`e5c009e6713b9fc1517d019eb9c549283036d029c04fe3377e38aa020b58fcf7`.
The candidate and all results live under
`figures/validation/wic_partial_conv_20261013/`; the reference checkpoint,
manual annotations and automatic-mask scripts were not modified.

The frozen dataset has 17,612 reviewed nonempty frames and the same
9,333/3,438/4,841 train/validation/test split. The held-out frames average
27.9% nonfinite WIC pixels. Every frame has some gaps, so missingness is common
enough for the ablation to matter. Near-gap means a finite-WIC pixel within
two Euclidean grid cells of a nonfinite WIC pixel.

Both models have 30,441 learned parameters, one frame, the same four input
maps and finite flags, seed 12026, BCE + Dice + topology weight 0.1, class/
orbit/segment sample weights, AdamW, batch 64, at most 40 epochs with patience
8, and validation-selected checkpoint and mask threshold. The candidate
changes only the six encoder convolutions: it multiplies inputs by the
propagated intrinsic finite-WIC mask, renormalizes for available neighbors,
and leaves the decoder ordinary. Its border normalization preserves exact
ordinary-convolution behavior when all WIC is valid. This is a VAE-inspired
partial-convolution ablation, not a byte-for-byte copy of the VAE's layer.
The target acceptance mask is never an input. SZA and DZA are also absent at
all pixels with missing WIC in this dataset; method quality remains finite at
16.3% of those pixels and is excluded there by the candidate.

The candidate ran 22 epochs and saved epoch 14. Both mask thresholds are
0.475. Focused numerical tests verify identical seeded initial parameters,
near-exact all-valid forward equivalence, exclusion of missing-site values,
and finite gradients. The 22 focused mask/topology tests pass. The baseline
was reloaded and reproduced its saved test metrics within 0.01 percentage
points; tiny differences stem from probabilities at the saved threshold.

## Outcome

The candidate's validation class-balanced orbit-average pixel disagreement
is 11.40%, versus 10.61% for the current standard-convolution checkpoint.
On the previously inspected held-out test set:

| Orbit-average test error | Standard | Partial conv | Change |
| --- | ---: | ---: | ---: |
| Fully accepted | 15.9% | 22.7% | +6.8 pp |
| Fully rejected | 49.6% | 31.7% | -17.9 pp |
| Partial | 21.9% | 27.4% | +5.5 pp |
| Partial useful pixels discarded | 14.5% | 33.6% | +19.0 pp |
| Partial rejected pixels retained | 28.3% | 21.9% | -6.4 pp |

Observable shape violations rise from 161 to 199 of 4,841 frames. Paired
orbit-bootstrap 95% intervals for the test-error changes above are
full [+0.6,+13.5] pp, rejected [-25.6,-11.2] pp, partial [+0.9,+10.6] pp.
The candidate is more willing to reject pixels, improving rejected frames
but discarding many useful pixels in partial frames.

The gain is not concentrated next to data gaps. Within two pixels of a gap,
error changes are full +9.4 pp, rejected -16.0 pp, partial +2.3 pp; farther
inside observed areas they are +6.1, -17.9 and +5.8 pp, respectively.
Fixed examples selected by class and missingness percentile show the same
tradeoff. The original concern about invalid values entering ordinary
convolutions is real, but this tested replacement does not improve the
partial-image masks needed for annotation and does not show a gap-specific
rescue. Do not substitute this checkpoint into `apply_model2_unet.py`.

This is one seed on an already inspected holdout and is exploratory. A
different per-channel masking design or class-aware decision rule remains
untested. Neither is adopted by this result. `comparison.json`,
`gap_comparison.png`, `same_frame_examples.png`, candidate `summary.json`,
predictions and history preserve the evidence. The README owns reproduction.
