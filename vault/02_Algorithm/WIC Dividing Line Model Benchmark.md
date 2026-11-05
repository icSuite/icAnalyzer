# WIC dividing-line model benchmark

Last reviewed: 2026-10-14  
Status: Authorized three-seed trial complete; worse agreement; not adopted.  
Repository: `vae/wic-si12` at `be31406`, existing dirty worktree preserved.

## Question and protocol

User authorizes trying a model that predicts dividing lines after explaining
their conservative line-based and temporally reused annotation policy.
[[Manual Annotation Policy and Blind Audit]] owns that policy and the separate
eight-orbit audit. This trial is single-frame only, with no temporal changes.

Reuse frozen corrected manual targets,9333/3438/4841 orbit/day splits, current
four inputs(WIC,quality,SZA,DZA) plus finite flags and exact training-only
normalization. Eight audit orbits are absent from the entire original dataset.
Same first/second/bottom U-Net encoder blocks; flattened spatial features predict
bad/full/partial class, latent1/2/3 line-count probabilities, angles and offsets.
Candidate65244 parameters versus pixel U-Net30441. No direct line/count labels.

Training uses sigmoid half-planes(temperature0.04 in normalized native coordinates),
mixtures of first-k intersections, class mixture, BCE+Dice and effective-class
cross entropy0.2. No topology penalty. Labels for class CE come from full-grid
mask contents, so constant keyframes inside semantic partial intervals are
correctly treated as effective full/rejected. Hard inference uses class/count
argmax and exact oriented half-plane intersection, with boundary tolerance1e-6.

Seeds12026/12027/12028, AdamW lr0.001/weight decay0.0001, batch64, max40 epochs,
patience8, original class/orbit/segment weights and independent shuffle generator.
All selected epoch4 and stopped after12. Validation selects only a global
expansion margin by the four-endpoint orbit-averagedJ score; selected margins
-0.2/-0.2/-0.1. Two optima lie at the search boundary; no post-test range expansion.
Epoch/margin and representative seed12026 frozen before test/audit evaluation.

## Results

Compare matched-seed CORE pixel U-Nets from the completed input-channel study.
J averages four error rates(full useful loss,rejected retained,partial useful
loss,partial rejected retained), averaging frames within orbit then orbits.
These are not aggregate percentages of all available pixels.

| Three-seed mean | Pixel U-Net | Line model | Change |
| --- | ---: | ---: | ---: |
| Existing exploratory testJ | 27.06% | 33.93% | +6.87pp |
| Eight-orbit auditJ | 26.80% | 35.77% | +8.97pp |
| Audit partial useful pixels discarded | 33.33% | 36.11% | +2.78pp |
| Audit partial rejected pixels retained | 22.26% | 54.87% | +32.61pp |

Paired whole-day2000-draw95% intervals forJ regression are+4.15→10.16pp(test)
and+1.24→18.44pp(audit), both excluding zero. Audit partial rejected-retained
regression interval+13.35→51.89pp. Candidate is worse across all three seeds.
Current production single-seed U-Net auditJ25.43% is separate context, not the
matched-seed three-run baseline. Representative candidate finite-WIC dayside
acceptance23.71% versus manual39.58%/production37.33%; nightside50.84% versus
manual54.41%/production58.24%. Smaller retained area is not proof of cleaner data.

## Observed failure and limits

Independent parameter inspection finds1117/203/2 nonconstant raw audit masks
by seed, despite1118/623/801 partial-state head decisions. Two seeds frequently
collapse partial geometry into all-zero/all-one intersections. Count2 never wins
argmax in any seed. Strongest seed also worsens partial agreement, so collapse
does not explain the whole regression. This demonstrates failure of this initial
training/decoding setup, not impossibility of using lines.

Historic native endpoint geometry exactly reproduces all505 stored line-set
keyframes, without projection mismatch. However92 have4–7 lines. Their best
three-original-line subsets differ on2.525% of full-grid cells; this is not a
lower bound over freely fitted lines. The three-line cap may limit those targets.
Capacity, loss, state decoding and representation differ together; no pure
architecture attribution. Literal predicted count need not equal human line count.

Existing test has been inspected repeatedly; the audit informed this hypothesis,
although excluded from training/selection. Eight grouped orbits are a pilot.
Manual disagreement measures conservative policy imitation, not physical noise.
No production model/mask replacement, corpus reannotation or VAE/CVAE training.

## Verification and artifacts

Model/loss:`scripts/automatic_masking/wic_line_model.py`; runner:
`try_line_model.py`; tests:`tests/test_wic_line_model.py`. Nineteen focused tests
pass, including independent native geometry, boundary/count/state handling,
numerical gradients, missing-pixel loss exclusion and real encoder gradients.
Independent input/split/normalization parity checks and audit mask/source hashes
pass. All candidate mask rescoring and audit rerasterizations reproduce exactly;
every hard mask has at most one accepted interval per row/column. Test bootstrap
intervals reproduce exactly. CPU versus GPU CORE audit roundoff changes at most
0.00102pp in a component and0.000881pp in an interval; conclusions unchanged.

Artifacts:`figures/validation/wic_line_model_20261014/`: readable`report.md`,
protocol/selection, per-seed checkpoints/history/margin curves/predictions,
comparisonJSON/PNG/PDF, fixed audit example sheet. Standalone checkpoints include
normalization/shape/class order/protocol hash; prediction parameters carry audit
prefix and orbit/frame/time IDs. Metadata enrichment preserves all numeric model
weights and predictions exactly. Source/model/main/audit annotation hashes are
verified unchanged after execution. GPU training was explicitly authorized and
ran outside the sandbox; the NVML warning did not prevent CUDA execution.

## Next actionable state

Retain the current production masks. Proposed next line-specific experiment:
direct line/count supervision with permutation-aware matching and improved
soft-to-hard alignment, addressing the observed degeneration. A larger line cap
and temporal context remain separate options. No additional training is queued
or adopted; user decides whether to continue this direction.
