# Research Plan: Choosing the Dimension of Coherent-Chunk Coordinates in Semicoherent Searches

## Scope And Source Of Truth

This plan is based on `idea.md` and the human feedback provided for attempt 1. The central goal is not to claim that semicoherent searches are new, and not to compete with fully coherent matched filtering as the statistical optimum. The goal is to determine whether there is a better sensitivity-versus-computing tradeoff than ordinary frequency-time semicoherent accumulation, by choosing a higher-dimensional coherent-chunk coordinate space `zeta`.

The main target case is 3.5PN gravitational-wave chirps, with two case studies:

1. Equal-mass, non-spinning waveforms.
2. Equal-mass waveforms with spin up to `chi = 0.2`.

## Status Labels

Throughout the project, claims should be labelled as one of:

- **Established from project premise**: Directly stated in `idea.md` or in the existing project context.
- **Established from derivation**: Follows from assumptions written in the report.
- **Literature-supported**: Only after the literature agent verifies the claim from sources.
- **Conjecture**: Plausible but not yet demonstrated.
- **Open question**: Requires derivation, simulation, or literature review.

No citation should be invented. If a claim needs support, mark it as `citation needed`.

## Central Research Question

How should one choose the number and form of coherent-chunk coordinates `zeta` in a semicoherent search for modelled signals, so that the final statistic gives the best sensitivity at fixed computational cost?

More concretely:

Given a physical waveform family with phase model

```math
\phi(t; \theta),
```

and a semicoherent analysis made of coherent chunks of duration

```math
T_{\rm coh},
```

what is the optimal dimension and coordinate basis of

```math
\zeta = \zeta(\theta, t_k),
```

where `zeta` parametrizes the coherent analysis in each chunk?

## Core Framing

Standard semicoherent methods often compute spectra in frequency-time space and then sum power along model-predicted tracks. The proposed generalization is to compute spectra in a richer chunk-local coordinate space, for example

```math
\zeta = \{f\},
```

or

```math
\zeta = \{f, \dot f\},
```

or a PN-motivated coordinate set such as

```math
\zeta = \{f, \beta_0, \beta_1, \ldots\}.
```

A physical template `theta` then maps to a track through this space:

```math
\theta \mapsto \zeta_k(\theta)
```

for segment index `k`.

The central tradeoff is:

- Adding `zeta` dimensions can reduce coherent mismatch within each chunk.
- Adding `zeta` dimensions increases spectrum-generation cost, memory cost, interpolation cost, track-summing cost, and possibly the detection threshold through a larger trials factor.

The desired output is a criterion for when an extra coherent parameter is worth keeping.

## Established Facts From The Project Premise

The following are established from `idea.md` and the repository instructions, not from a new literature review:

1. The active research code already implements a phase-resampling/NUFFT workflow for modelled chirping signals.

2. The nearby example uses a coordinate set similar to

```math
\zeta = \{f, t, \beta\},
```

where `beta` is a 0PN-like chirp parameter.

3. The proposed research question is broader than this example: one could instead use 1PN or higher-PN local signal models inside coherent chunks.

4. Increasing the richness of the coherent chunk model can allow larger `T_coh`, but may increase computational cost.

5. The final report should include:
   - Literature review.
   - Mathematical formalism.
   - Sensitivity analysis.
   - Computational cost estimation.

6. The final report should separately consider:
   - Generic signal-processing principles for choosing `dim(zeta)`.
   - Equal-mass non-spinning 3.5PN gravitational waveforms.
   - Equal-mass spinning 3.5PN gravitational waveforms with `chi <= 0.2`.

## Main Conjectures To Test

These should not be presented as established results until supported.

1. **Optimal dimension conjecture**

For fixed observing time, parameter-space volume, allowed false-alarm probability, and compute budget, there is an optimal finite dimension

```math
d_\zeta^\star = \dim(\zeta)
```

that minimizes detectable signal amplitude.

2. **0PN one-parameter conjecture**

For equal-mass non-spinning 3.5PN waveforms, the best one-parameter reduced coordinate may be close to the physically motivated 0PN chirp coordinate. This is plausible but not guaranteed.

3. **Reduced-basis improvement conjecture**

SVD/PCA/reduced-basis coordinates may outperform raw PN coordinates once two or more intrinsic chunk parameters are allowed.

4. **Spin-dimensionality conjecture**

Allowing aligned spin up to `chi = 0.2` may increase the intrinsic dimension of the chunk-local phase family and therefore require extra `zeta` dimensions for the same mismatch.

5. **Compute-limited usefulness conjecture**

There may be a regime where a modestly higher-dimensional coherent representation, such as `{f, beta}` or `{f, fdot}`, beats ordinary frequency-time StackSlide at equal compute and equal false-alarm probability.

## Subquestions

### Generic Signal-Processing Questions

1. Given a phase family

```math
\phi(t; \theta),
```

what is the best low-dimensional chunk-local approximation

```math
\Phi_{\rm loc}(u; \zeta_k),
\qquad u = t - t_k,
```

over a chunk of duration `T_coh`?

2. Should `zeta` be chosen from physical parameters, Taylor coefficients, PN coefficients, chirp-time coordinates, or data-driven basis coefficients?

3. Is the proposed construction equivalent to, or a variant of, known methods such as:
   - StackSlide.
   - PowerFlux.
   - Weave.
   - Hough or Radon track accumulation.
   - Fast chirp transforms.
   - Polynomial-phase transforms.
   - Reduced-basis or SVD waveform methods.
   - Multibank or multiband inspiral methods.

Each of these is a literature-search target; no equivalence should be claimed until checked.

### Dimension-Selection Questions

1. How does the required dimension of `zeta` depend on `T_coh`?

2. How does it depend on the allowed coherent mismatch per chunk?

3. How does it depend on the full physical parameter-space volume?

4. How does it depend on the downstream semicoherent statistic?

5. Is the optimal dimension better predicted by:
   - PN order?
   - Taylor order in time?
   - Fisher-metric eigenvalues?
   - SVD/PCA singular values?
   - Empirical detection performance at fixed compute?

### Cost-Sensitivity Questions

1. How much sensitivity is gained by adding one more `zeta` dimension?

2. How much compute and memory does that dimension cost?

3. Does the larger `zeta` grid increase the effective trials factor enough to erase the sensitivity gain?

4. Does interpolation in higher-dimensional `zeta` space create unacceptable mismatch or runtime cost?

5. Is the limiting resource FLOPs, memory bandwidth, disk storage, or template-track summation?

### Equal-Mass Non-Spinning 3.5PN Questions

1. What is the intrinsic dimension of the equal-mass non-spinning 3.5PN phase family over relevant chunk durations?

2. How close is the best one-dimensional SVD/PCA basis to the 0PN phase direction?

3. Does adding a second or third basis direction significantly reduce mismatch?

4. For what `T_coh` does the 0PN coordinate cease to be enough?

5. Does a PN-coordinate basis or a data-driven basis produce lower total cost at fixed sensitivity?

### Equal-Mass Spinning 3.5PN Questions

1. How much extra phase variation appears when allowing spin up to

```math
\chi \le 0.2?
```

2. Is the spin-induced variation mostly aligned with the non-spinning basis directions, or does it introduce new directions?

3. How does the optimal `dim(zeta)` change relative to the non-spinning case?

4. Can spin be handled perturbatively using one additional coordinate, or does it require a qualitatively different basis?

5. Does the best low-dimensional basis remain interpretable in PN terms?

## Required Derivations

### 1. Semicoherent Statistic In A General `zeta` Space

Define the data model:

```math
x(t) = n(t) + h(t; \theta).
```

Split the data into chunks indexed by `k`, centered at time `t_k`, with local time

```math
u = t - t_k.
```

Define a coherent chunk output:

```math
X_k(\zeta)
=
\int_{I_k}
dt\,
W_k(t)\,
x(t)
\exp[-i \Phi_{\rm loc}(t - t_k; \zeta)].
```

Here:

- `W_k(t)` is a window.
- `Phi_loc` is the local phase model.
- `zeta` is the coordinate vector used to make the chunk-level spectrum.

Define a normalized chunk power or likelihood-like statistic:

```math
P_k(\zeta)
=
\frac{|X_k(\zeta)|^2}{\sigma_k^2(\zeta)}.
```

A physical template `theta` predicts a track through `zeta` space:

```math
\zeta_k = \zeta_k(\theta).
```

The semicoherent statistic is then

```math
S(\theta)
=
\sum_k
w_k(\theta)
P_k[\zeta_k(\theta)].
```

This derivation should explicitly show that ordinary frequency-time StackSlide is recovered when

```math
\zeta = \{f\}.
```

### 2. Coherent Mismatch From Residual Phase

Within chunk `k`, write the exact phase as

```math
\phi(t_k + u; \theta)
=
\phi_k
+
\Phi_{\rm loc}(u; \zeta_k)
+
R_k(u; \theta, \zeta_k),
```

where `R_k` is the residual phase after fitting the local model.

For small residual phase, derive the coherent mismatch approximately as the variance of the residual phase after removing an irrelevant constant phase:

```math
\mu_k
\approx
\left\langle R_k^2 \right\rangle
-
\left\langle R_k \right\rangle^2.
```

The weighting in the average must be specified, for example by the window function and noise weighting.

This derivation is essential because it connects the choice of `zeta` to sensitivity loss.

### 3. Scaling Of Omitted Phase Directions With `T_coh`

If the local model includes phase terms through order `p`, the first omitted term schematically scales like

```math
R_k(u) \sim \phi^{(p+1)}(t_k) u^{p+1}.
```

The mismatch then scales approximately as

```math
\mu_k
\sim
[\phi^{(p+1)}(t_k)]^2
T_{\rm coh}^{2p+2}
```

up to a window-dependent constant.

The report should derive the exact constants for simple windows if feasible, but it is acceptable to first use this as a scaling result.

### 4. Dimension Selection By Phase-Manifold Eigenvalues

Construct a phase-residual family after removing nuisance directions such as constant phase and possibly linear phase:

```math
r_i(u) =
\phi(u; \theta_i)
-
\phi_{\rm fitted}(u; \zeta_{\rm baseline}).
```

Define an inner product:

```math
\langle a, b \rangle
=
\int_{-T_{\rm coh}/2}^{T_{\rm coh}/2}
du\,
q(u)
a(u)b(u),
```

where `q(u)` encodes the window and possible noise weighting.

Build a covariance or Gram matrix over sampled waveforms and perform SVD/PCA:

```math
r_i(u)
\approx
\sum_{\alpha=1}^{d}
c_{i\alpha} e_\alpha(u).
```

A candidate rule is:

Retain the smallest `d` such that the residual mismatch from discarded modes is below a target:

```math
\mu_{\rm discarded}(d)
\le
\mu_{\rm max}.
```

This is only a mismatch criterion. It is not yet the final optimality criterion because it ignores computational cost.

### 5. Cost Model

Derive a cost model with at least the following components:

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

Let

```math
N_{\rm seg} = T_{\rm obs}/T_{\rm coh}.
```

Let `N_zeta` be the number of grid points in the coherent coordinate space. Then the spectrum-generation cost should be modelled as

```math
C_{\rm spectra}
\sim
N_{\rm seg}
N_\zeta
c_{\rm coh}.
```

The track-summing cost should be modelled as

```math
C_{\rm tracks}
\sim
N_\theta
N_{\rm seg}
c_{\rm lookup}.
```

Memory should be modelled as

```math
M_{\rm spectra}
\sim
N_{\rm seg}
N_\zeta
B,
```

where `B` is bytes per stored statistic.

For multilinear interpolation in `d_zeta` dimensions,

```math
c_{\rm lookup}
```

may scale at least like

```math
2^{d_\zeta}.
```

This should be checked empirically.

### 6. Sensitivity Model At Fixed False Alarm

For ideal independent normalized powers,

```math
P_k \sim \mathrm{Exp}(1)
```

under noise only, and

```math
S = \sum_k P_k
```

has a Gamma distribution. This is an idealized derivation and must be labelled as such.

The report should then add correction factors for:

- Mismatch.
- Weights.
- Non-independent bins.
- Interpolation.
- Coloured noise.
- Trials factor.
- Non-Gaussianity if real data is considered later.

The relevant figure of merit is not recovered power alone. It should be detectable amplitude at fixed:

- False-alarm probability.
- Detection probability.
- Observing time.
- Compute budget.
- Memory budget.

A possible optimization target is:

```math
(d_\zeta^\star, T_{\rm coh}^\star, \mathcal{B}^\star)
=
\arg\min
h_{\min}
```

subject to

```math
C_{\rm total} \le C_{\rm budget},
\qquad
M_{\rm total} \le M_{\rm budget},
\qquad
P_{\rm FA} \le P_{\rm FA}^{\rm target},
\qquad
\mu \le \mu_{\rm max}.
```

Here `B` or `mathcal{B}` denotes the chosen basis, not bytes.

### 7. Analytical Low-Dimensional Approximation

The human feedback asks whether the low-dimensional basis can be found analytically.

The report should derive what can be done analytically in at least two cases:

1. **Taylor/polynomial phase models**

For polynomial phase,

```math
\phi(u)
=
\sum_{j}
a_j u^j,
```

the natural local coordinates are polynomial coefficients. Orthogonal polynomial bases may diagonalize the mismatch approximately under simple windows.

2. **PN phase models**

For PN chirps, write the phase as a linear or nonlinear combination of PN basis functions. Investigate whether the 0PN term is the dominant first principal direction for equal-mass non-spinning waveforms.

This should be treated as an open derivation until performed. The expectation that 0PN is close to the best one-parameter coordinate is a conjecture, not an established result.

## Required Literature Searches

The literature agent should verify specific claims and produce citations. Until then, all items below are search targets.

### Semicoherent GW Search Methods

Search targets:

- StackSlide methods and optimization.
- PowerFlux weighted power sums.
- Weave semicoherent searches and template banks.
- Semicoherent metric construction.
- Loosely coherent searches.
- Hough-transform searches in continuous-wave or chirp contexts.

Questions to answer:

1. Have these methods already formulated a general coordinate choice problem for semicoherent chunk statistics?

2. Do they optimize coherent time and template-bank dimension jointly?

3. Do they include trials factors and computing cost in the optimization?

### Chirp And Polynomial-Phase Transforms

Search targets:

- Fast chirp transform.
- Polynomial-phase transform.
- High-order ambiguity function.
- Radon-transform track integration.
- Generalized time-frequency representations for chirping signals.

Questions to answer:

1. Is a spectrum over `{f, fdot, fddot, ...}` already a standard object?

2. Are there known computational scalings for such transforms?

3. Are there known optimal dimension-selection rules?

### Reduced Basis, SVD/PCA, And Reduced-Order Quadrature

Search targets:

- Reduced-basis methods for gravitational waveforms.
- SVD/PCA waveform compression.
- Reduced-order quadrature.
- Chirp-time coordinates for compact binary inspiral banks.
- Metric eigenbasis methods.

Questions to answer:

1. Are optimal low-dimensional waveform coordinates already known for PN inspiral families?

2. How do these coordinates compare to PN coefficients or chirp-time coordinates?

3. Can reduced bases be used as axes of reusable semicoherent spectra, or only as a way to accelerate matched filtering?

### MBTA And Multiband Inspiral Searches

Search targets:

- Multi-band template analysis.
- Multibank compact-binary searches.
- Methods that split waveform information by frequency or time scale.

Questions to answer:

1. Are these conceptually close to choosing different coherent coordinates for different chunks?

2. Do they offer cost models that can be reused here?

### Zotero And Local Notes

The note in `idea.md` says:

```text
/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md has useful information.
```

The literature agent should inspect that file.

The literature agent should also check Zotero before saying whether a paper is missing. Until Zotero is checked, use the label:

```text
Zotero status unknown.
```

## Required Simulations

All scripts should be run with:

```bash
conda run -n PBH python ...
```

Scratch scripts should be written under `/tmp/`, not inside the repository.

### Simulation 1: Toy Polynomial-Phase Study

Purpose:

Establish the generic signal-processing behavior in a controlled case.

Signal family:

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

Compare coherent spaces:

```math
\zeta_1 = \{f\},
```

```math
\zeta_2 = \{f, \dot f\},
```

```math
\zeta_3 = \{f, \dot f, \ddot f\}.
```

Measure:

- Coherent mismatch versus `T_coh`.
- Number of grid points required at fixed mismatch.
- Runtime for spectrum construction.
- Memory use.
- Track-summing cost.
- Null distribution.
- Detection probability at fixed false alarm.
- Sensitivity at fixed compute.

Acceptance criterion:

The simulation must show whether adding dimensions helps after cost and threshold penalties are included.

### Simulation 2: Equal-Mass Non-Spinning 3.5PN Basis Study

Purpose:

Answer whether 0PN is close to the best one-parameter chunk coordinate.

Inputs to specify:

- Mass range.
- Starting frequency range.
- Chunk durations.
- Sampling or frequency representation.
- Noise weighting, if any.
- Phase convention.
- Whether amplitude is ignored or included.

Procedure:

1. Generate equal-mass non-spinning 3.5PN phases over the chosen parameter range.

2. For each chunk center `t_k` and `T_coh`, remove nuisance components:
   - Constant phase.
   - Possibly linear phase if frequency is always included in `zeta`.
   - Any baseline 0PN component, depending on comparison.

3. Build a phase-residual matrix.

4. Run SVD/PCA.

5. Compare:
   - Frequency-only basis.
   - 0PN one-parameter basis.
   - Best one-dimensional SVD basis.
   - Best two-dimensional SVD basis.
   - Best three-dimensional SVD basis.
   - PN-coordinate bases of comparable dimension.

6. Record residual mismatch as a function of dimension and `T_coh`.

Key outputs:

- Singular value spectrum.
- Projection of the 0PN basis onto the first SVD basis vector.
- Mismatch versus `dim(zeta)`.
- Required dimension for chosen mismatch thresholds.
- Dependence on `T_coh`.

Acceptance criterion:

The result must directly answer:

```text
How different is the optimal one-parameter basis from the 0PN waveform?
```

### Simulation 3: Equal-Mass Spinning 3.5PN Basis Study With `chi <= 0.2`

Purpose:

Determine whether modest spin increases the required coherent-coordinate dimension.

Inputs:

- Equal masses.
- Spin range `chi <= 0.2`.
- Specify whether spins are aligned, anti-aligned, or both.
- Same frequency and mass ranges as the non-spinning case where possible.

Procedure:

Repeat the non-spinning SVD/PCA analysis with spin included.

Compare:

- Non-spinning basis applied to spinning signals.
- Spinning SVD/PCA basis.
- PN-inspired spin coordinate additions.
- One-, two-, three-, and higher-dimensional bases.

Key outputs:

- Singular value spectrum with spin.
- Overlap between spinning and non-spinning basis vectors.
- Additional mismatch caused by spin when using the non-spinning basis.
- Extra dimensions required to recover the same mismatch.
- Whether one spin-related coordinate is sufficient for `chi <= 0.2`.

Acceptance criterion:

The result must answer whether the spinning case requires a genuinely different `zeta` space or only a small extension of the non-spinning one.

### Simulation 4: Semicoherent End-To-End Cost-Sensitivity Comparison

Purpose:

Translate basis quality into actual semicoherent search performance.

Candidate methods:

1. Frequency-time StackSlide baseline:

```math
\zeta = \{f\}.
```

2. 0PN chunk spectra:

```math
\zeta = \{f, \beta_0\}.
```

3. Higher-PN coordinate spectra:

```math
\zeta = \{f, \beta_0, \beta_1, \ldots\}.
```

4. SVD/PCA coordinate spectra:

```math
\zeta = \{f, c_1, c_2, \ldots, c_d\}.
```

5. A local semicoherent matched-filter-bank baseline, if computationally feasible.

Measure:

- Detection probability versus injected amplitude.
- False-alarm threshold.
- Effective trials factor.
- Runtime.
- Memory.
- Number of templates.
- Interpolation mismatch.
- Total sensitivity at equal compute.

Acceptance criterion:

A higher-dimensional `zeta` method is only successful if it lowers detectable amplitude at equal compute and equal false-alarm probability.

## Likely Bottlenecks

1. **Parameter-space definition**

The basis study can give misleading results if the mass, frequency, and spin ranges are too broad or too narrow. The report must state the chosen ranges and justify them.

2. **Difference between `theta` and `zeta`**

The physical waveform parameters `theta` and chunk-local coordinates `zeta` are not the same. Confusing them will make the cost model wrong.

3. **Trials factor**

A higher-dimensional spectrum may recover more signal power but also produce a larger search space and a higher threshold.

4. **Memory**

High-dimensional spectra may become memory-limited before they become FLOP-limited.

5. **Interpolation**

Track accumulation through a high-dimensional grid may require interpolation. The mismatch and cost of interpolation must be measured.

6. **Basis dependence**

SVD/PCA bases depend on the chosen inner product, waveform sampling, parameter range, chunk duration, and whether frequency/phase nuisance directions are removed.

7. **Analytical tractability**

Analytical bases may be possible for polynomial phase models, but less obvious for full 3.5PN spinning waveforms.

8. **Spin**

For `chi <= 0.2`, spin effects may be small over short chunks but important over longer chunks. The answer may depend strongly on `T_coh`.

9. **Novelty risk**

The idea may overlap strongly with fast chirp transforms, polynomial-phase transforms, semicoherent template banks, or reduced-basis methods. The report should frame the contribution as a costed comparison and dimension-selection criterion unless literature review supports a stronger claim.

10. **Noise realism**

Early simulations can use ideal Gaussian noise, but final sensitivity claims require attention to coloured noise, nonuniform resampling effects, and bin covariance if the local PBH/NUFFT/5-vector machinery is used.

## Decision Criteria

An added coordinate dimension should be retained only if all of the following hold:

1. It reduces coherent mismatch enough to improve expected detection statistic.

2. The improvement survives the increased grid size, memory cost, interpolation cost, and track-summing cost.

3. The improvement survives the increased false-alarm threshold or effective trials factor.

4. It improves detectable amplitude at fixed compute, not merely recovered power at unlimited compute.

5. It remains robust across the relevant physical parameter range.

For the 3.5PN studies, the report should produce a table like:

| Case | `T_coh` | basis | `dim(zeta)` | residual mismatch | relative cost | threshold penalty | expected sensitivity |
|---|---:|---|---:|---:|---:|---:|---:|
| non-spinning | TBD | frequency only | 1 | TBD | TBD | TBD | TBD |
| non-spinning | TBD | 0PN | 2 | TBD | TBD | TBD | TBD |
| non-spinning | TBD | SVD/PCA | 2 | TBD | TBD | TBD | TBD |
| spinning `chi <= 0.2` | TBD | non-spinning basis | TBD | TBD | TBD | TBD | TBD |
| spinning `chi <= 0.2` | TBD | spin-extended basis | TBD | TBD | TBD | TBD | TBD |

## Proposed Report Structure

### 1. Executive Summary

State the narrowed research question:

```text
How many coherent-chunk coordinates should a semicoherent search use, and how should those coordinates be chosen, when optimizing sensitivity at fixed compute?
```

Summarize conclusions only after derivations, literature review, and simulations are complete.

### 2. Problem Definition

Define:

- Physical parameters `theta`.
- Chunk-local coordinates `zeta`.
- Coherent time `T_coh`.
- Segment index `k`.
- Coherent spectrum.
- Semicoherent track.
- Detection statistic.
- Compute and memory budgets.

### 3. Prior Art And Novelty Risks

Review verified literature on:

- StackSlide.
- PowerFlux.
- Weave.
- Hough/Radon methods.
- Fast chirp transforms.
- Polynomial-phase transforms.
- Reduced basis and SVD/PCA waveform methods.
- Chirp-time coordinates.
- MBTA or related multiband compact-binary methods.

Every citation must be checked. Mark unchecked items as `citation needed`.

### 4. Generic Mathematical Formalism

Derive the general `zeta`-space semicoherent statistic and show how ordinary frequency-time summing appears as a special case.

### 5. Coherent Mismatch And Dimension Selection

Develop the residual-phase mismatch expansion and the SVD/PCA or metric-eigenbasis criterion for choosing candidate dimensions.

Clearly distinguish:

- A mismatch-optimal basis.
- A compute-optimal basis.
- A physically interpretable basis.

### 6. Computational Cost Model

Include:

- Spectrum construction.
- Track summation.
- Interpolation.
- Memory.
- Trials factor.
- Parallelization assumptions.
- Scaling with `T_coh` and `dim(zeta)`.

### 7. Sensitivity Model

Derive the idealized semicoherent sensitivity scaling, then add penalties for mismatch, trials factor, and nonideal noise.

The main figure of merit should be detectable amplitude at fixed compute and false-alarm probability.

### 8. Analytical Low-Dimensional Bases

Address the human feedback directly:

- Can optimal low-dimensional expressions be derived analytically?
- Is 0PN expected to be the best one-parameter coordinate?
- Under what assumptions would that be true?
- When might SVD/PCA produce a better but less interpretable coordinate?

### 9. Case Study A: Equal-Mass Non-Spinning 3.5PN Waveforms

Report:

- Phase model.
- Parameter ranges.
- Basis construction.
- Singular value spectra.
- 0PN versus best one-dimensional basis.
- Required `dim(zeta)` versus `T_coh`.
- Costed semicoherent implications.

### 10. Case Study B: Equal-Mass Spinning 3.5PN Waveforms With `chi <= 0.2`

Report:

- Spin assumptions.
- Added phase directions.
- Comparison with non-spinning basis.
- Required extra dimensions.
- Costed semicoherent implications.

### 11. End-To-End Semicoherent Comparison

Compare at least:

- Frequency-time StackSlide.
- 0PN-enhanced spectra.
- Higher-PN spectra.
- SVD/PCA-coordinate spectra.

Use equal compute and equal false-alarm probability.

### 12. Bottlenecks And Failure Modes

List unresolved issues, including literature overlap, memory limits, trials factors, interpolation error, and dependence on parameter ranges.

### 13. Conclusions And Recommended Next Steps

The final conclusions should answer:

1. How should `dim(zeta)` be chosen in principle?
2. Is 0PN close to optimal for equal-mass non-spinning 3.5PN waveforms?
3. How much does spin up to `chi = 0.2` change the answer?
4. Is there a practical sensitivity-versus-compute gain over frequency-time semicoherent accumulation?

## Immediate Next Tasks For The Workflow

1. Literature agent: verify prior-art overlap and citations.

2. Theory agent: derive the mismatch, dimension-selection, cost, and sensitivity formulae.

3. Simulation agent: implement the non-spinning 3.5PN SVD/PCA basis study first.

4. Simulation agent: repeat the basis study with spin up to `chi = 0.2`.

5. Integration agent: combine basis mismatch results with a semicoherent cost model.

The first decisive technical result should be the non-spinning SVD/PCA comparison against the 0PN coordinate, because it directly tests the human feedback question about whether a physically motivated 0PN coordinate is already the best one-parameter choice.