# Research Plan: Choosing Coherent-Chunk Coordinates for Semicoherent Searches

## Audit Metadata

- Source of truth: `idea.md` and the attempt-2 human feedback.
- Status: research planning artifact, not a final scientific report.
- Attempt: 2.
- Required downstream output: a report on sensitivity-versus-compute tradeoffs for semicoherent searches using higher-dimensional coherent-chunk spectra.
- Citation policy: no external result is treated as established until verified by a literature agent. Search targets below are not citations.
- Execution policy for future scripts: run with `conda run -n PBH python ...`; scratch code should live under `/tmp/`.

## Central Research Question

How should a semicoherent search choose the number and form of coherent-chunk coordinates

```math
\zeta
```

so that the final detection statistic gives the best sensitivity at fixed computational cost?

Equivalently: given a modeled signal family with physical parameters

```math
\theta
```

and phase

```math
\phi(t;\theta),
```

what lower-dimensional chunk-local coordinate system

```math
\zeta_k(\theta)
```

should be used inside coherent chunks of duration

```math
T_{\rm coh}
```

before semicoherently combining power along template tracks?

The goal is not to beat fully coherent matched filtering as the statistical optimum. The goal is to find better sensitivity-compute tradeoffs than ordinary frequency-time semicoherent summing, and to decide when adding extra coherent-chunk dimensions is worth the cost.

## Established Premises

### Established From `idea.md`

1. Conventional semicoherent analyses often use time-frequency spectra and sum power along model-predicted tracks.

2. The proposed generalization is to construct spectra in a richer coordinate space

```math
\zeta,
```

where `zeta` includes frequency and time information and may also include combinations of physical phase parameters.

3. The nearby project already contains an example resembling

```math
\zeta = \{f, t, \beta\},
```

where `beta` is a 0PN-like parameter.

4. Richer coherent-chunk models may allow longer useful coherent times, but they also increase spectrum-generation and track-combination cost.

5. The final report should include:
   - literature review,
   - mathematical formalism,
   - sensitivity analysis,
   - computational cost estimation.

6. The final report must specifically treat:
   - the generic signal-processing question of choosing `dim(zeta)`,
   - equal-mass non-spinning 3.5PN gravitational-wave waveforms,
   - equal-mass spinning 3.5PN waveforms with spin up to `chi = 0.2`.

### Established From Project Context

1. The active codebase is a gravitational-wave search prototype based on phase demodulation, nonuniform resampling, NUFFT spectra, and 5-vector reconstruction.

2. Internal frequencies in the active resampler code are usually angular frequencies in radians per second.

3. Any later implementation work should prefer `Resampling/Nov2025/` unless a task explicitly targets another area.

## Conjectures To Test

1. **Finite useful dimension**

For fixed observing time, parameter-space volume, false-alarm probability, and compute budget, there is a useful finite dimension

```math
d_\zeta^\star = \dim(\zeta)
```

that minimizes detectable amplitude.

2. **0PN near-optimality for one parameter**

For equal-mass non-spinning 3.5PN waveforms, the best one-parameter chunk-local basis may be close to the physically motivated 0PN chirp direction. This is plausible but unproven.

3. **SVD/PCA usefulness**

Reduced-basis, SVD, or PCA coordinates may improve over raw PN coordinates once two or more chunk-local dimensions are allowed.

4. **Spin increases effective dimension**

Allowing spin up to `chi = 0.2` may introduce additional phase directions and increase the useful number of `zeta` coordinates.

5. **Compute-limited improvement**

There may be a practical regime where `{f, beta}` or another low-dimensional chirp-aware coordinate set improves sensitivity at equal compute relative to frequency-time StackSlide-like accumulation.

## Open Questions

1. Is the proposed higher-dimensional spectrum construction already equivalent to a known fast chirp transform, polynomial-phase transform, Hough/Radon transform, or semicoherent template-bank method?

2. Can an optimal low-dimensional coordinate system be derived analytically for PN waveform families, or only numerically through metric eigenvectors or SVD/PCA?

3. Is the 0PN coordinate truly the dominant one-dimensional direction for the equal-mass non-spinning 3.5PN family over the parameter ranges relevant to this project?

4. Does modest spin mostly project onto existing non-spinning directions, or does it require new basis directions?

5. Does adding dimensions improve detectable amplitude after memory cost, interpolation cost, trials factor, and threshold penalties are included?

## Subquestions

### Generic Signal-Processing Questions

1. Given a phase family

```math
\phi(t;\theta),
```

what is the best local approximation

```math
\Phi_{\rm loc}(u;\zeta_k),
\qquad u = t - t_k,
```

inside a coherent chunk?

2. Should `zeta` be made from:
   - physical parameters,
   - Taylor coefficients such as `{f, fdot, fddot}`,
   - PN or chirp-time parameters,
   - metric eigenvectors,
   - SVD/PCA basis coefficients,
   - another analytic basis?

3. What criterion should decide whether an additional coordinate is retained?

4. How does the optimal dimension depend on:
   - `T_coh`,
   - mismatch tolerance,
   - parameter-space volume,
   - signal family,
   - noise model,
   - false-alarm threshold,
   - compute and memory budget?

### Sensitivity-Compute Questions

1. How much coherent mismatch is removed by adding a coordinate?

2. How many extra grid points does that coordinate require?

3. How does the extra coordinate affect:
   - spectrum construction,
   - track summation,
   - interpolation,
   - memory,
   - effective number of trials,
   - detection threshold?

4. Is the right objective detectable amplitude, expected recovered power, expected SNR, or likelihood ratio at fixed false alarm?

Working assumption: detectable amplitude at fixed false-alarm probability and fixed compute is the main figure of merit.

### Equal-Mass Non-Spinning 3.5PN Questions

1. What is the intrinsic dimension of the equal-mass non-spinning 3.5PN phase family over relevant chunk durations?

2. How different is the best one-dimensional SVD/PCA basis from the 0PN phase direction?

3. How much mismatch remains after using:
   - frequency only,
   - frequency plus 0PN,
   - frequency plus best one SVD/PCA coordinate,
   - frequency plus two or more SVD/PCA coordinates,
   - higher-PN coordinate sets?

4. For which `T_coh` values does a one-parameter chirp coordinate stop being adequate?

5. Does the answer depend strongly on mass range, frequency range, or chunk placement?

### Equal-Mass Spinning 3.5PN Questions

1. What spin convention is used: aligned only, anti-aligned only, or both signs?

2. How much additional phase variation appears for `|chi| <= 0.2` or `0 <= chi <= 0.2`, depending on the intended physical range?

3. Does the spin variation align with the non-spinning basis or introduce new directions?

4. Is one spin-related coordinate enough to recover the non-spinning mismatch performance?

5. Does a spin-extended physically motivated PN basis perform comparably to a numerical SVD/PCA basis?

## Required Derivations

### 1. General `zeta`-Space Semicoherent Statistic

Start from

```math
x(t) = n(t) + h(t;\theta).
```

Split the data into chunks `I_k` centered at `t_k`, and define local time

```math
u = t - t_k.
```

Define a coherent chunk output

```math
X_k(\zeta)
=
\int_{I_k}
dt\,
W_k(t)\,
x(t)
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

Here `W_k(t)` is a window and `Phi_loc` is the chunk-local phase model.

Define a normalized chunk statistic, for example

```math
P_k(\zeta)
=
\frac{|X_k(\zeta)|^2}{\sigma_k^2(\zeta)}.
```

A physical template predicts a track

```math
\theta \mapsto \zeta_k(\theta).
```

The semicoherent statistic is

```math
S(\theta)
=
\sum_k
w_k(\theta)
P_k[\zeta_k(\theta)].
```

Required result: show that ordinary frequency-time StackSlide-like accumulation is recovered when the only searched chunk-local coordinate is frequency,

```math
\zeta = \{f\}.
```

Status: derivation required.

### 2. Residual-Phase Mismatch

Write the exact phase in chunk `k` as

```math
\phi(t_k+u;\theta)
=
\phi_k
+
\Phi_{\rm loc}(u;\zeta_k)
+
R_k(u;\theta,\zeta_k).
```

After removing the irrelevant constant phase, the small-residual coherent mismatch should be derived as

```math
\mu_k
\approx
\langle R_k^2\rangle
-
\langle R_k\rangle^2,
```

where the inner product must specify the window and any noise weighting.

This derivation connects basis choice directly to sensitivity loss.

Status: derivation required.

### 3. Scaling With Coherent Time

If the local model includes phase terms through order `p`, and the first omitted term behaves like

```math
R_k(u) \sim \phi^{(p+1)}(t_k) u^{p+1},
```

then the mismatch should scale as

```math
\mu_k
\sim
[\phi^{(p+1)}(t_k)]^2
T_{\rm coh}^{2p+2},
```

up to a window-dependent constant.

Required result: derive exact constants for simple rectangular or Hann windows if feasible.

Status: scaling conjecture until derived in the chosen convention.

### 4. Dimension Selection From Metric Eigenvalues Or SVD/PCA

Define a residual phase family after removing nuisance directions such as constant phase and possibly frequency:

```math
r_i(u)
=
\phi(u;\theta_i)
-
\phi_{\rm fitted}(u;\zeta_{\rm baseline}).
```

Define an inner product

```math
\langle a,b\rangle
=
\int_{-T_{\rm coh}/2}^{T_{\rm coh}/2}
du\,
q(u)a(u)b(u),
```

where `q(u)` contains windowing and optional noise weighting.

Build a residual matrix and compute a basis

```math
r_i(u)
\approx
\sum_{\alpha=1}^{d}
c_{i\alpha} e_\alpha(u).
```

Candidate mismatch rule:

```math
\mu_{\rm discarded}(d)
\le
\mu_{\rm max}.
```

This gives a candidate dimension, not the final optimum, because compute and threshold costs are not included.

Status: derivation and implementation required.

### 5. Cost Model

Derive and calibrate

```math
C_{\rm total}
=
C_{\rm spectra}
+
C_{\rm tracks}
+
C_{\rm interp}
+
C_{\rm memory}
+
C_{\rm threshold}.
```

With

```math
N_{\rm seg}=T_{\rm obs}/T_{\rm coh},
```

and `N_zeta` grid points, model spectrum construction as

```math
C_{\rm spectra}
\sim
N_{\rm seg} N_\zeta c_{\rm coh}.
```

Model track summation as

```math
C_{\rm tracks}
\sim
N_\theta N_{\rm seg} c_{\rm lookup}.
```

Model stored spectra as

```math
M_{\rm spectra}
\sim
N_{\rm seg} N_\zeta B,
```

where `B` is bytes per stored statistic.

If multilinear interpolation is used in `d_zeta` dimensions, test whether lookup cost scales like

```math
c_{\rm lookup} \propto 2^{d_\zeta}.
```

Status: derivation and empirical calibration required.

### 6. Sensitivity At Fixed False Alarm

For ideal independent normalized powers,

```math
P_k \sim \mathrm{Exp}(1)
```

under noise-only data, so an unweighted sum over `N_seg` chunks has a Gamma null distribution.

This is only an idealized starting point. The final model must include:
- coherent mismatch,
- semicoherent weights,
- non-independent bins,
- interpolation error,
- colored noise,
- effective trials factor,
- non-Gaussianity if real data is used.

The optimization target should be

```math
(d_\zeta^\star,T_{\rm coh}^\star,\mathcal{B}^\star)
=
\arg\min h_{\min}
```

subject to

```math
C_{\rm total} \le C_{\rm budget},
\qquad
M_{\rm total} \le M_{\rm budget},
\qquad
P_{\rm FA} \le P_{\rm FA}^{\rm target}.
```

Here `mathcal{B}` denotes the chosen coordinate basis.

Status: ideal derivation required first; realistic calibration required later.

### 7. Analytical Low-Dimensional Bases

The report must address whether optimal lower-dimensional expressions can be derived analytically.

For polynomial phase models,

```math
\phi(u)
=
\sum_j a_j u^j,
```

derive whether orthogonal polynomial bases diagonalize the mismatch under simple windows.

For PN phase models, derive whether the 0PN phase direction is expected to dominate the equal-mass non-spinning 3.5PN residual family. This should be treated as an open question until the derivation and SVD/PCA comparison are complete.

Status: open derivation.

## Required Literature Searches

No item below is a citation yet. Each is a search target requiring source verification.

### Semicoherent Gravitational-Wave Searches

Search targets:
- StackSlide.
- PowerFlux.
- Weave.
- Hough-transform CW searches.
- Radon-style track accumulation.
- Loosely coherent searches.
- Semicoherent metric template banks.

Questions:
1. Do these methods already optimize the number of coherent-chunk coordinates?
2. How do they choose `T_coh` and template-bank dimension?
3. Do they include compute cost, memory, and trials factors in the optimization?
4. Are there existing criteria equivalent to adding coordinates to `zeta`?

### Chirp And Polynomial-Phase Signal Processing

Search targets:
- fast chirp transform,
- polynomial-phase transform,
- high-order ambiguity function,
- generalized time-frequency transforms,
- chirplet transforms.

Questions:
1. Is a spectrum indexed by `{f, fdot, fddot, ...}` already standard?
2. What are the known computational scalings?
3. Are there known rules for selecting the polynomial order or dimension?
4. Are there fast algorithms relevant to `zeta` spectra?

### Reduced Basis, SVD/PCA, And ROQ

Search targets:
- reduced-basis methods for gravitational waveforms,
- SVD/PCA waveform compression,
- reduced-order quadrature,
- chirp-time coordinates,
- metric eigenbasis methods,
- compact-binary inspiral template-bank coordinates.

Questions:
1. Are optimal low-dimensional PN coordinates already known?
2. How do those coordinates compare with 0PN and higher-PN physical directions?
3. Can reduced-basis coordinates be used as axes for reusable coherent spectra, or only for matched-filter acceleration?

### MBTA And Multiband Inspiral Searches

Search targets:
- Multi-Band Template Analysis.
- Multibank compact-binary searches.
- Multiband or multirate inspiral filtering.

Questions:
1. Are these methods conceptually close to changing the coherent coordinate system by segment?
2. Do they provide reusable cost models?
3. Do they suggest a practical way to combine different local dimensionalities across frequency or time?

### Local Notes And Zotero

Required local inspection by the literature agent:
- `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md`

Required library check:
- Zotero status for candidate papers.

Until Zotero is checked, use label:

```text
Zotero status unknown.
```

## Required Simulations

### Simulation 1: Toy Polynomial-Phase Dimension Study

Purpose: isolate the generic signal-processing question.

Signal model:

```math
\phi(t)
=
2\pi
\left[
f t
+
\frac{1}{2}\dot f t^2
+
\frac{1}{6}\ddot f t^3
+
\cdots
\right].
```

Compare:
- `zeta = {f}`
- `zeta = {f, fdot}`
- `zeta = {f, fdot, fddot}`

Measure:
- coherent mismatch versus `T_coh`,
- grid size at fixed mismatch,
- runtime,
- memory,
- interpolation error,
- null distribution,
- detection probability at fixed false alarm,
- detectable amplitude at fixed compute.

Success criterion: determine whether extra polynomial-phase dimensions help after cost and threshold penalties.

### Simulation 2: Equal-Mass Non-Spinning 3.5PN SVD/PCA Study

Purpose: directly answer whether the optimal one-dimensional basis differs from the 0PN waveform direction.

Procedure:
1. Choose and record mass range, frequency range, chunk durations, phase convention, and inner product.
2. Generate equal-mass non-spinning 3.5PN phase families.
3. Remove nuisance components:
   - constant phase,
   - frequency if frequency is always included in `zeta`,
   - other explicitly included baseline coordinates.
4. Build a residual phase matrix.
5. Run SVD/PCA or metric eigenanalysis.
6. Compare:
   - frequency-only,
   - frequency plus 0PN,
   - frequency plus best one-dimensional basis,
   - frequency plus best two-dimensional basis,
   - frequency plus best three-dimensional basis,
   - selected PN-coordinate bases.
7. Validate on held-out waveforms.

Required outputs:
- singular value spectrum,
- overlap between 0PN direction and first SVD/PCA direction,
- mismatch versus dimension,
- required dimension versus `T_coh`,
- sensitivity-cost implication.

Success criterion: answer whether 0PN is effectively the best one-parameter choice for this waveform family and parameter range.

### Simulation 3: Equal-Mass Spinning 3.5PN Study With `chi <= 0.2`

Purpose: determine whether spin changes the useful `zeta` dimension.

Procedure:
1. Define spin convention clearly: aligned, anti-aligned, both signs, or another restricted model.
2. Use the same mass and frequency ranges as Simulation 2 when possible.
3. Generate spinning 3.5PN phases with `chi <= 0.2` in the chosen convention.
4. Repeat SVD/PCA analysis.
5. Compare:
   - non-spinning basis applied to spinning waveforms,
   - spinning basis,
   - PN-inspired spin coordinate additions,
   - one-, two-, three-, and higher-dimensional coordinate sets.
6. Validate on held-out spin values.

Required outputs:
- singular value spectrum with spin,
- overlap between spinning and non-spinning basis vectors,
- added mismatch from using the non-spinning basis,
- extra dimensions required to match non-spinning performance,
- whether one spin coordinate is enough.

Success criterion: decide whether the spinning case requires a genuinely different `zeta` space or a small extension.

### Simulation 4: End-To-End Semicoherent Cost-Sensitivity Comparison

Purpose: turn basis quality into search performance.

Compare:
- frequency-time baseline,
- 0PN-enhanced spectra,
- higher-PN coordinate spectra,
- SVD/PCA-coordinate spectra,
- local semicoherent matched-filter-bank baseline if feasible.

Measure:
- false-alarm threshold,
- effective trials factor,
- detection probability versus injected amplitude,
- detectable amplitude at fixed false alarm,
- runtime,
- memory,
- number of grid points,
- number of tracks,
- interpolation mismatch.

Success criterion: a richer `zeta` method is useful only if it lowers detectable amplitude at equal compute and equal false-alarm probability.

## Likely Bottlenecks

1. **Parameter-space definition**

The basis and cost results will be meaningless unless mass, frequency, spin, and chunk-duration ranges are specified and justified.

2. **Confusing `theta` and `zeta`**

Physical parameters `theta` define the waveform. Chunk-local coordinates `zeta` define the coherent spectra. They may be related but are not the same object.

3. **Trials factor**

Higher-dimensional spectra can recover more power while also increasing the threshold. The report must include this penalty.

4. **Memory**

High-dimensional spectra may be memory-limited before they are FLOP-limited.

5. **Interpolation**

Track summation through `zeta` grids may require interpolation. This can add both mismatch and runtime.

6. **Basis dependence**

SVD/PCA results depend on the chosen inner product, parameter range, waveform sampling, nuisance directions removed, and `T_coh`.

7. **Analytical tractability**

Analytical dimension rules may be possible for polynomial phase, but full 3.5PN spinning waveforms may require numerical bases.

8. **Spin convention**

The spin study can produce ambiguous conclusions unless the allowed spin range and alignment assumptions are explicit.

9. **Noise model**

Early dimension-selection studies may use phase-only mismatch. Final sensitivity claims need a noise model and calibrated null distributions.

10. **Prior-art overlap**

The strongest defensible contribution may be a costed dimension-selection criterion for this application, not a new class of semicoherent methods.

## Decision Criteria For Adding A `zeta` Dimension

An extra coordinate should be retained only if it satisfies all of the following in the relevant parameter range:

1. It reduces coherent mismatch by a measurable amount.

2. It improves expected semicoherent sensitivity, not just coherent recovered power.

3. The improvement survives the increased grid size.

4. The improvement survives memory and interpolation costs.

5. The improvement survives the larger effective trials factor.

6. It lowers detectable amplitude at fixed compute and fixed false-alarm probability.

7. It generalizes to held-out waveforms rather than overfitting the basis-construction sample.

## Proposed Report Structure

### 1. Executive Summary

State the main answer to:

```text
How many coherent-chunk coordinates should be used, and how should they be chosen?
```

Include only results supported by derivation, literature review, or simulation.

### 2. Problem Definition

Define:
- physical parameters `theta`,
- chunk-local coordinates `zeta`,
- coherent time `T_coh`,
- chunk index `k`,
- coherent spectra,
- semicoherent tracks,
- detection statistic,
- compute and memory budgets.

### 3. Prior Art And Novelty Risks

Review verified literature on:
- StackSlide,
- PowerFlux,
- Weave,
- Hough/Radon methods,
- fast chirp transforms,
- polynomial-phase transforms,
- reduced basis and SVD/PCA methods,
- chirp-time coordinates,
- MBTA and multiband searches.

Use `citation needed` for unverified claims.

### 4. Generic Mathematical Formalism

Derive the `zeta`-space semicoherent statistic and show the frequency-time special case.

### 5. Coherent Mismatch And Dimension Selection

Derive residual-phase mismatch, omitted-term scaling with `T_coh`, and the metric/SVD/PCA dimension criterion.

### 6. Computational Cost Model

Model spectrum construction, track summation, interpolation, memory, parallelization assumptions, and threshold penalties.

### 7. Sensitivity Model

Derive idealized false-alarm and detection behavior, then add mismatch, trials factor, and nonideal-noise corrections.

### 8. Analytical Low-Dimensional Bases

Address whether optimal coordinates can be derived analytically. Treat polynomial phase and PN phase separately. Explain when 0PN should or should not be expected to dominate.

### 9. Case Study A: Equal-Mass Non-Spinning 3.5PN Waveforms

Report:
- waveform and phase convention,
- parameter ranges,
- basis construction,
- singular value spectrum,
- 0PN versus best one-dimensional basis,
- mismatch versus dimension and `T_coh`,
- costed implication for semicoherent search design.

### 10. Case Study B: Equal-Mass Spinning 3.5PN Waveforms With `chi <= 0.2`

Report:
- spin assumptions,
- additional phase directions,
- comparison with non-spinning basis,
- dimensions required for fixed mismatch,
- costed implication for semicoherent search design.

### 11. End-To-End Semicoherent Comparison

Compare frequency-time accumulation, 0PN-enhanced spectra, higher-PN spectra, and SVD/PCA-coordinate spectra at equal compute and equal false-alarm probability.

### 12. Bottlenecks And Failure Modes

Discuss memory limits, interpolation, trials factors, parameter-range dependence, prior-art overlap, and noise realism.

### 13. Conclusions And Next Steps

Answer:
1. How should `dim(zeta)` be chosen in principle?
2. Is 0PN close to optimal for equal-mass non-spinning 3.5PN waveforms?
3. How much does spin up to `chi = 0.2` change the answer?
4. Is there a practical sensitivity-compute gain over frequency-time semicoherent accumulation?

## Immediate Workflow Tasks

1. Literature agent: verify prior-art overlap and collect real citations. Inspect local `CODEX.md` and check Zotero status.

2. Theory agent: derive residual-phase mismatch, dimension-selection criteria, cost scaling, and sensitivity at fixed false alarm.

3. Simulation agent: implement the equal-mass non-spinning 3.5PN SVD/PCA study first, because it directly tests the 0PN one-parameter conjecture.

4. Simulation agent: repeat the basis study with spin up to `chi = 0.2`.

5. Integration agent: combine mismatch results, cost models, and threshold calibration into an end-to-end sensitivity-compute comparison.