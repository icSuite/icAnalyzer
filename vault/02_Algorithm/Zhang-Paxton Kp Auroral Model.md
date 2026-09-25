# Zhang-Paxton Kp Auroral Model

Last reviewed: 2026-07-28

## Source

Y. Zhang and L. J. Paxton (2008), “An empirical Kp-dependent global
auroral model based on TIMED/GUVI FUV data,” *Journal of Atmospheric and
Solar-Terrestrial Physics*, 70, 1231–1242.
doi:10.1016/j.jastp.2008.03.008.

Local copy: `literature/zhang_paxton_2008.pdf`

## What the model provides

**Confirmed from the paper:** The model accepts Kp together with magnetic
local time and magnetic latitude and returns spatially varying maps of:

- precipitating-electron mean energy, `E0`, in keV;
- precipitating-electron energy flux, `Q`, in erg/(cm² s), numerically
  equivalent to mW/m².

The model was fitted to about 43,200 TIMED/GUVI maps from 2002–2005. It uses:

- six Kp-bin centers: `[0.75, 2.25, 3.75, 5.25, 7.0, 9.0]`;
- an Epstein function for magnetic-colatitude dependence;
- a sixth-order Fourier representation for MLT dependence;
- linear interpolation between neighboring Kp bins for `E0`;
- hemispheric-power-based nonlinear interpolation for `Q`.

The paper tabulates the coefficients needed to reconstruct both maps for
Kp 0–9.

## Interpretation of Figure 8

Figure 8(b) is not the local mean-energy model. It is a diagnostic derived
from the full spatial model: the global average `E0` over both hemispheres,
using only pixels where modeled `Q > 0.05 erg/(cm² s)`.

Using that curve to assign one constant `E0` to an entire IMAGE frame is
possible as a deliberate approximation, but it discards the model's strong
MLT and latitude structure. Figure 7 illustrates the loss: at fixed Kp the
nightside oval can have substantially higher `E0` than the Figure 8 global
average. For example, the paper describes nightside values around 4 keV at
Kp 2 and 7–8 keV at Kp 5, while the Figure 8 global averages are only about
1.9 and 2.5 keV.

Treat the Figure 8 scalar as a simple baseline or sensitivity case. The raw
full spatial model is computationally cheap, but colleague-reported experience
below indicates that its auroral-oval latitude should not be trusted directly
for individual IMAGE frames.

## Fasil's IMAGE-relative implementation

**Confirmed through the user's colleague:** Fasil has implemented the
Zhang-Paxton model and found that it can produce high electron energy in the
auroral oval while placing the oval at the wrong magnetic latitude for a
particular observation.

Fasil reportedly developed a latitude-normalization method that uses primarily
the Zhang-Paxton MLT dependence while taking magnetic-latitude information
from the images. The exact transformation, code, tests, and validation are not
yet available in this repository.

**Rejected interpretation:** A method that requires manual oval boundaries,
manual registration, or fragile per-frame fitting is not compatible with the
automatic IMAGE dataset pipeline and is not an option at dataset scale.

**Possible automatic interpretation, not confirmed:** Fasil may collapse or
normalize the Zhang-Paxton latitude dependence once, yielding an MLT-dependent
energy prescription that can be applied wherever IMAGE itself contains auroral
signal. This would avoid explicit per-frame oval registration. Do not treat
this as Fasil's method until he confirms it.

**Next evidence needed from Fasil:**

1. whether the method is fully automatic with no per-frame manual input;
2. the implementation or precise equations;
3. whether it detects an oval in each image or removes/collapses latitude
   dependence without registration;
4. if it detects an oval, the image quantity, boundary definition, failure
   handling, and quality control;
5. whether the transformation is fitted independently in each MLT sector;
6. whether it is applied to `E0`, `Q`, or both;
7. treatment outside the oval and when image coverage is incomplete;
8. validation cases, metrics, runtime, and known failure modes;
9. Kp source, timestamp alignment, and any uncertainty treatment.

Do not duplicate or independently finalize a Zhang-Paxton implementation
before coordinating with Fasil.

## Audit of `habtie-phys/auroraloval`

Repository: <https://github.com/habtie-phys/auroraloval>

**Assessment on 2026-07-28:** This appears to be a scientifically plausible
third-party transcription of the Zhang-Paxton **energy-flux and oval-boundary
calculation**, but it is not a complete or publication-ready implementation of
the Zhang-Paxton model.

Positive evidence:

- the author is an ionospheric electron-precipitation researcher;
- the stored energy-flux coefficients spot-checked against Table 1 of the
  paper, including several constants and Fourier terms;
- the Epstein function uses `x-B` in both exponent arguments, consistent with
  Figure 5 and correcting the apparent `x-A` appendix typo;
- the hemispheric-power equations and nonlinear `Q` interpolation follow the
  paper;
- the supplied Kp=5 morphology is visually consistent with the
  Zhang-Paxton panel in Figure 6;
- a 2026 *Scientific Data* article explicitly cites and uses this repository
  as a third-party Zhang-Paxton implementation for energy-flux threshold
  boundaries:
  <https://www.nature.com/articles/s41597-026-06884-2>.

Material limitations found by direct inspection and execution:

- it includes only Table 1 energy-flux coefficients and returns `Q` plus
  threshold-derived boundaries; it does **not** implement Table 2 electron
  mean energy `E0`, which is the quantity currently needed by `icBuilder`;
- there is no license, release, package metadata, dependency declaration, CI,
  or numerical validation suite;
- `test_ZP_oval.py` is a plotting example with no assertions;
- the nominal package entry point, `auroraloval.py`, has a syntax error;
- the coefficient path is relative to the process working directory, so
  importing and calling it from another directory raises `FileNotFoundError`;
- Kp exactly 9, although within the paper's stated 0–9 range, raises
  `IndexError`;
- boundary cases that need a missing-value marker use `np.NaN`, removed in
  NumPy 2;
- the paper specifies `x = 90 - |Mlat|`, while the code uses
  `x = 90 - mlat`; southern magnetic latitudes therefore produce incorrect
  near-zero results instead of hemispheric symmetry;
- Kp=0 produced some negative interpolated energy fluxes in the audit and the
  implementation neither documents nor handles them;
- the full 20-commit history occurred on 2023-04-03 and 2023-04-04, with no
  subsequent maintenance.

**Use boundary:** Do not vendor or depend on this repository unchanged. It is
useful as a reference implementation for `Q` and as an independent comparison
when reconstructing the published equations. Before any reuse, obtain an
explicit license from the author. For `icBuilder`, either obtain Fasil's
extended implementation or construct a tested implementation from both
published coefficient tables, with explicit Kp endpoints, `|Mlat|`,
module-relative resources, non-negative output policy, and regression tests
against published figures or trusted model output.

Ask Fasil whether this repository is the starting point for his code and,
specifically, where his Table 2 `E0` coefficients and latitude-normalization
logic reside.

## Why GUVI does not replace IMAGE

**Decision confirmed by the user:** Simultaneous global auroral snapshots are
important to this project because the VAE is intended to represent global
spatial covariance. Retain IMAGE as the primary observational geometry.

GUVI Level 2B and Level 3 products are valuable because they provide retrieved
electron mean energy and energy flux, but an individual orbital swath covers
only part of the auroral oval. Successive swaths are not a simultaneous global
state and are therefore not a drop-in replacement for IMAGE.

Evaluate GUVI as an external calibration, validation, or benchmark source. The
official GUVI auroral product spans 2002-01-20 through 2007-11-04, while the
current IMAGE product is recorded as 2000-2001, so temporal overlap must be
verified before proposing frame-coincident validation.

Sources:

- NASA CCMC auroral-oval study:
  <https://ccmc.gsfc.nasa.gov/ccmc-studies/auroral-oval/>
- NASA/SPASE TIMED/GUVI aurora product:
  <https://spase-metadata.org/NASA/DisplayData/TIMED/GUVI/Aurora_PNG.html>

## Alternative precipitation models

The model's age alone is not a reason to reject it, but newer alternatives
should be included in the model-selection audit:

- **OVATION Prime (2010/2013)** is the strongest immediately practical
  alternative. It is driven by solar-wind and IMF history rather than Kp,
  produces global average-energy and energy-flux patterns for multiple
  precipitation types, has public IDL and Python implementations, and can be
  run through NASA CCMC. Its climatological oval location still requires
  validation against IMAGE.
- **ASHLEY-A (2021; updated applications continue)** is the strongest modern
  spectral alternative identified so far. It is parameterized by IMF strength
  and clock angle plus solar-wind velocity and density and returns
  differential energy flux in the 19 DMSP energy channels. Mean energy and
  total energy flux can therefore be derived without imposing a Maxwellian
  spectrum. Public implementation and batch-operability for the IMAGE interval
  remain to be established.
- **FTA (2021)** is an AE-driven, Polar-UVI image-based model designed to track
  auroral morphology and boundaries. It predicts UV emissions rather than a
  particle spectrum; mean energy would require a band-ratio inversion and a
  spectral assumption, so it is complementary rather than a direct
  replacement.
- **PrecipNet (2021)** is a solar-wind/geomagnetic-history neural network that
  improves total electron energy-flux prediction relative to OVATION Prime,
  but its published target is total energy flux rather than mean energy.
- **Feng et al. (2025)** provide a Kp-driven XGBoost model trained on
  DMSP/SSUSI imagery and report improved oval-boundary prediction relative to
  Zhang-Paxton at higher Kp. It predicts UV radiance and boundary morphology,
  not the required `E0` and `Q` pair.

For the automatic IMAGE pipeline, the first implementation comparison should
be Zhang-Paxton versus OVATION Prime. Add ASHLEY-A if its code and historical
inputs can be obtained without creating a separate implementation project.
Treat FTA, PrecipNet, and the 2025 XGBoost model as supporting comparisons or
future methods rather than drop-in mean-energy sources.

### Why Zhang-Paxton may have been preferred to OVATION Prime

The colleagues' actual rationale is not yet confirmed. The leading
project-specific explanations are:

1. Zhang-Paxton is an FUV-image-derived retrieval of `E0` and `Q`, making it
   conceptually closer to the IMAGE FUV conductance pipeline than a model
   derived from along-track DMSP particle measurements.
2. It needs only Kp, which is easy to obtain continuously for every historical
   IMAGE frame. OVATION Prime needs propagated solar-wind and IMF history and
   therefore adds input gaps, timing uncertainty, and preprocessing.
3. The pipeline may need only an external `E0` prescription while retaining
   IMAGE-observed morphology and brightness. OVATION Prime is designed to
   predict the precipitation pattern itself, so using it wholesale would
   replace or double-count information already present in IMAGE.
4. OVATION's separate diffuse, monoenergetic, broadband, and ion components do
   not necessarily reduce cleanly to the single spectral-energy parameter used
   by the existing IMAGE response and Robinson-conductance calculation.
5. Fasil already has a Zhang-Paxton implementation and an IMAGE-relative
   latitude treatment, which may dominate a small theoretical advantage from
   another model.
6. OVATION Prime is not substantially newer: its main versions date from 2010
   and 2013, versus 2008 for Zhang-Paxton, and both are climatologies built
   from older satellite observations.

Ask Fasil directly whether the choice was made because the pipeline needs only
`E0` while preserving IMAGE morphology, because of the Kp-only input, because
of spectral-definition compatibility, or simply because a validated local
implementation already existed. Do not record any of these as his reason
until confirmed.

Sources:

- NASA CCMC OVATION Prime model and variables:
  <https://ccmc.gsfc.nasa.gov/models/Ovation-Prime~1.0/>
  and <https://ccmc.gsfc.nasa.gov/VIS-DOCS/physical-variables/>
- Zhu et al. (2021), ASHLEY:
  <https://doi.org/10.1029/2020SW002671>
- Wu et al. (2021), FTA:
  <https://doi.org/10.1029/2020SW002629>
- McGranaghan et al. (2021), PrecipNet:
  <https://doi.org/10.1029/2020SW002684>
- Feng et al. (2025), Kp-driven XGBoost oval model:
  <https://doi.org/10.1029/2024JH000543>

## Compatibility with the current conductance pipeline

Before substituting this model, verify two definitions and one coupling:

1. Zhang-Paxton `E0` is the mean energy of an assumed Gaussian precipitating
   electron distribution. `icBuilder` uses both “mean” and “characteristic”
   energy terminology; confirm that its response functions and the Robinson
   conductance formulas expect the same spectral parameter.
2. In `icBuilder`, electron energy flux is not independent of `E0`.
   `imagesat_e0_eflux_estimates.py` uses `E0` to calculate the expected WIC
   response per unit electron energy flux. Replacing `E0` therefore requires
   recomputing `Fe`, not merely inserting a new energy into the final Robinson
   conductance formulas.
3. The paper supplies no per-pixel uncertainty model for the climatological
   `E0`. A defensible use needs model uncertainty or a sensitivity envelope.

A useful comparison would retain the same cleaned IMAGE measurements and
evaluate:

- the current pixelwise WIC/SI13-ratio `E0`;
- the raw Zhang-Paxton spatial `E0`;
- Fasil's method, only if it is automatic and robust at dataset scale;
- the Figure 8 frame-constant `E0` as a deliberately simplified baseline.

Final Hall and Pedersen conductance must be recomputed consistently in every
case.

## Conditional-model confound

The longer-term VAE is intended to be conditioned on geomagnetic activity.
If Kp is first used to construct `E0` and therefore the conductance training
targets, and Kp is also supplied to the conditional VAE, then some apparent Kp
dependence is imposed by label construction rather than learned independently
from IMAGE.

This does not automatically invalidate the approach, but it changes the
scientific interpretation. The imposed Zhang-Paxton prior must be explicit,
and conditional-model evaluation should distinguish:

- structure inherited deterministically from the Kp-based `E0` prior;
- IMAGE-specific variation retained through measured FUV intensity;
- additional conditional structure learned by the VAE.

Independent particle, radar, or riometer validation would be especially
valuable for separating these components.

## Limitations relevant to IMAGE

- Kp is a three-hour activity index, so many IMAGE frames will receive the
  same Kp and the model cannot resolve substorm evolution within the bin.
- The formulation combines northern and southern data and uses `|Mlat|`; it
  does not model hemispheric or seasonal differences.
- It is Kp-only. The authors identify Dst, IMF Bz, and other drivers as future
  multi-parameter extensions.
- The Kp 8–10 bin has relatively few samples and visibly noisier maps.
- The inversion assumes Gaussian precipitating electrons and does not fully
  separate proton/ion contributions.
- GUVI has limited sensitivity to weak dayside precipitation.
- The paper presents one detailed GUVI–DMSP case and a morphology comparison
  with IMAGE SI-13, not comprehensive uncertainty calibration.

## Implementation caution

Appendix A prints the Epstein function with `x-A` in one denominator term,
whereas Figure 5 and the usual parameter interpretation use `x-B`. This
appears internally inconsistent. Do not transcribe the appendix blindly:
verify the formula against Figure 5, published model output, or the original
software before relying on a reconstruction.
