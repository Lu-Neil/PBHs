# Synthesis: Higher-Dimensional Semicoherent `zeta` Searches

## Main Conclusion

The strongest defensible claim is not that “higher-dimensional spectra plus track summation” is new. That overlaps heavily with StackSlide, PowerFlux, Hough/Radon methods, semicoherent template banks, fast chirp transforms, and polynomial-phase transforms. The promising research question is narrower:

> Can one choose and optimize an intermediate segment-local phase-coordinate space `zeta`, including its dimension, to improve compute-limited semicoherent searches for modeled chirping gravitational-wave signals?

This should be framed as a cost-constrained detection problem, not as statistical optimality over matched filtering. At equal false-alarm probability and equal compute, the method must beat ordinary StackSlide and relevant chirp-transform or semicoherent matched-filter baselines.

Supporting detail: [plan](00_research_plan.md), [literature review](01_literature_review.md), [theory](02_theory_and_optimization.md), [referee report](03_referee_report.md).

## What Is Established

From the project premise, the active idea is to replace ordinary time-frequency semicoherent accumulation with spectra indexed by a richer coordinate set `zeta`, for example `{f, t, beta}` in the nearby PBH/NUFFT implementation. Segment time `t_k` should be treated as a segment label unless it is truly gridded as a searched coordinate.

From the literature artifact, the broad components are established prior art:

- StackSlide and related CW searches already optimize coherent time, segment count, template grids, mismatch, and compute.
- Weave is a particularly close GW analogue: semicoherent search with metric template banks and optimal lattices.
- PowerFlux establishes weighted power accumulation as more appropriate than naive power sums when antenna response and noise vary.
- Fast chirp transforms, polynomial-phase transforms, Hough/Radon methods, and high-order ambiguity-style methods are the main novelty risks for higher-dimensional chirp-coordinate spectra.
- Reduced-basis, SVD/PCA, chirp-time, and metric-coordinate template banks provide existing machinery for choosing better coordinates than raw physical parameters.

The literature pass did not inspect Zotero, local code, or full papers. Candidate papers to check or add if absent include Brady-Creighton StackSlide, Prix-Shaltev StackSlide optimization, Pletsch semicoherent metric, Weave, PowerFlux, Dergachev loosely coherent searches, Roulet et al. SVD template banks, Owen template metrics, and Jenet-Prince fast chirp transform. See the full table in [01_literature_review.md](01_literature_review.md).

## Formalism

A clean formulation is:

```math
x(t) = n(t) + h(t;\theta),
```

split into coherent chunks of duration `T_coh`, centered at `t_k`. In each chunk define a local demodulated amplitude

```math
X_k(\zeta)
=
\int_{I_k} dt\,
W_k(t)\,
x(t)\,
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

A scalar power statistic is then

```math
p_k(\zeta)= {|X_k(\zeta)|^2 \over \sigma_k^2(\zeta)}.
```

A physical template `theta` predicts a track through segment-local coordinates,

```math
\zeta_k(\theta),
```

and the semicoherent statistic is

```math
\mathcal{S}(\theta)
=
\sum_k w_k(\theta)\,
p_k[\zeta_k(\theta)].
```

Ordinary StackSlide is the special case `zeta = {f}` with `Phi_loc = 2 pi f u`. A chirp-aware extension might use

```math
\zeta = \{f,\dot f,\ddot f,\ldots\}
```

or PN-like local coordinates such as

```math
\zeta = \{f,\beta_0,\beta_1,\beta_{3/2},\ldots\}.
```

This formalism is derived in the theory artifact, not yet validated for the local code. For the existing 5-vector pipeline, the scalar power should likely be replaced by a covariance-weighted vector statistic, e.g.

```math
\widehat a =
{A^\dagger C^{-1}X \over A^\dagger C^{-1}A}.
```

That matters because the current roadmap already notes that NUFFT/resampling changes the noise PSD and that 5-vector sidebands may have nontrivial covariance.

## Sensitivity

Under ideal independent complex Gaussian noise,

```math
p_k \sim \mathrm{Exp}(1),
```

so an unweighted sum over `N_seg` chunks has

```math
\mathcal{S} \sim \Gamma(N_{\rm seg},1).
```

With signal, the expected excess is controlled by per-segment noncentralities `lambda_k`:

```math
E[\mathcal{S}] - E_0[\mathcal{S}]
=
\sum_k w_k \lambda_k.
```

For equal segments and a power-sum statistic, the usual semicoherent amplitude scaling is approximately

```math
h_{0,\min}
\propto
S_n^{1/2}
T_{\rm coh}^{-1/4}
T_{\rm obs}^{-1/4},
```

ignoring mismatch, weights, and trials factors.

The key benefit of adding `zeta` dimensions is reduced coherent mismatch. If the exact phase in a segment is

```math
\phi(t_k+u;\theta)
=
\phi_k + \Phi_{\rm loc}(u;\zeta_k) + R_k(u),
```

then the coherent mismatch is controlled, for small residual phase, by the variance of `R_k` after subtracting an irrelevant constant phase. If the first omitted Taylor term is order `p+2`,

```math
\mu_k
\sim
[\phi^{(p+2)} T_{\rm coh}^{p+2}]^2 C_p.
```

Thus higher-order `zeta` coordinates can permit longer `T_coh`, but only if the sensitivity gain survives increased cost and threshold penalties.

## Cost Model

The central optimization should be stated explicitly:

```math
(d_\zeta^*,T_{\rm coh}^*)
=
\arg\min C_{\rm total}(d_\zeta,T_{\rm coh})
```

subject to target detection probability, total false-alarm probability, parameter-space coverage, and maximum mismatch.

A useful decomposition is:

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

and `N_zeta = product_a N_a`, spectrum construction roughly scales as

```math
C_{\rm spectra}
\sim
N_{\rm seg} N_\zeta c_{\rm coh}.
```

Track accumulation scales as

```math
C_{\rm tracks}
\sim
N_\theta N_{\rm seg} c_{\rm lookup}.
```

If interpolation is multilinear, `c_lookup` can scale like `2^{d_zeta}`. Memory can become the dominant limit:

```math
M_{\rm spectra}
\sim
N_{\rm seg} N_\zeta B.
```

The referee artifact correctly stresses that trials factors belong inside the optimization. A higher-dimensional `zeta` grid may improve recovered power while raising the effective detection threshold enough to erase the gain.

## Resolved Tensions

The planner’s broad framing is useful as a research agenda, but the referee’s narrower framing is more defensible. The report should not claim a new class of semicoherent searches until fast chirp transform, polynomial-phase, Hough/Radon, Weave, and semicoherent matched-filter equivalences are checked.

The theory artifact provides a workable mathematical scaffold, but it uses idealized independent exponential powers. The local PBH/NUFFT/5-vector pipeline likely violates this through colored noise, resampling noise transfer, bin correlations, sideband covariance, gaps, and interpolation.

The literature artifact suggests PCA/SVD/metric coordinates as promising. The referee correctly notes that coordinate reduction itself is not novel. The potentially novel contribution is using such coordinates as axes of reusable intermediate spectra and optimizing the retained dimension.

## Assumptions And Conjectures

Assumptions:

- The local phase can be represented accurately by a low-dimensional `zeta`.
- Spectra can be computed once over a reusable `zeta` grid, not recomputed per full physical template.
- Noise covariance and thresholds can be calibrated.
- A practical fast transform or NUFFT strategy exists for the chosen coordinates.

Conjectures:

- The optimal `zeta` dimension increases with `T_coh`.
- PN/chirp-time/metric-eigenbasis coordinates will outperform raw physical parameters.
- There is an intermediate regime where `{f,beta}` or `{f,\dot f}` beats both ordinary StackSlide and fully local matched-filter banks at equal compute.
- Memory bandwidth, not FLOPs, may dominate high-dimensional spectra.

Open questions:

- Is `{f,t,beta}` mathematically a known chirp or polynomial-phase transform?
- What is the correct 5-vector covariance after NUFFT/resampling?
- What is the empirical effective trials factor?
- What parameter-space volume is being searched?
- Should the final statistic be power-only, loosely coherent, or covariance-weighted likelihood-like?

## Concrete Next Steps

1. Inspect the local `{f,t,beta}` implementation and define exactly what `zeta`, `theta`, `S(theta)`, weights, normalization, and units mean.

2. Audit the key novelty-risk papers in full: Weave, Pletsch semicoherent metric, Prix-Shaltev StackSlide optimization, Jenet-Prince fast chirp transform, and polynomial-phase/high-order ambiguity literature.

3. Build the minimal toy comparison:
   `{f}` versus `{f,\dot f}` versus `{f,\dot f,\ddot f}` on polynomial-phase signals.

4. For each candidate `zeta`, measure:
   coherent mismatch versus `T_coh`, runtime, memory, null distribution, trials factor, and detection probability at fixed false alarm.

5. Repeat for PN chirps:
   frequency-only, 0PN, 0PN+1PN, and reduced-basis phase coordinates.

6. Replace equal-power sums with covariance-aware statistics before claiming optimality for the PBH/5-vector pipeline.

7. Compare against baselines:
   ordinary StackSlide, local semicoherent matched-filter bank, fast chirp transform analogue if applicable, and fully coherent matched filtering on small parameter volumes.

The project should proceed as a falsifiable cost-sensitivity study. The decisive result is not higher recovered power, but lower detectable amplitude at equal compute and equal false-alarm probability.