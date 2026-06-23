# Final Synthesis: Choosing `zeta` Dimensions for Semicoherent Searches

## Main Conclusion

The useful number of coherent-chunk coordinates `dim(zeta)` should not be chosen by asking how many phase directions improve coherent mismatch. It should be chosen by minimizing detectable amplitude at fixed compute, memory, and total false-alarm probability.

A good operational rule is:

```math
(d_\zeta^\star,T_{\rm coh}^\star,\mathcal B^\star)
=
\arg\min h_{\min}(d,T_{\rm coh},\mathcal B)
```

subject to compute, memory, and false-alarm constraints. SVD/PCA, PN coordinates, metric eigenvectors, and polynomial-phase bases are tools for proposing candidate coordinates; they are not by themselves the final search-design criterion.

This directly responds to the iteration feedback: semicoherent methods are established prior art, and the target is not matched-filter optimality. The sharper question is whether richer coherent-chunk spectra give a better sensitivity-cost tradeoff than frequency-time power sums.

Supporting detail: [planner](00_research_plan.md), [literature](01_literature_review.md), [theory](02_theory_and_optimization.md), [referee](03_referee_report.md).

## What Is Established

Semicoherent track summing is not new. StackSlide, PowerFlux, Hough/Radon-like methods, Weave, semicoherent metrics, chirp transforms, polynomial-phase transforms, and reduced-basis waveform methods all overlap with parts of this proposal. The novelty cannot be “sum power along modeled tracks” or “add chirp-like coordinates.”

The defensible contribution is narrower:

> A costed rule for choosing reusable coherent-chunk coordinates `zeta`, demonstrated on PN chirps, with sensitivity compared at equal compute and equal false alarm.

Ordinary StackSlide is the limiting case `zeta = {f}`. Time is normally a segment label, not a searched coordinate, unless spectra are explicitly gridded over epoch.

## Generic Signal-Processing Answer: How Many Parameters Should `zeta` Have?

For each chunk centered at `t_k`, define a local phase model

```math
\Phi_{\rm loc}(u;\zeta_k),
\qquad u=t-t_k.
```

The coherent output is

```math
X_k(\zeta)
=
\int_{I_k} dt\,W_k(t)x_a(t)
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

A physical template `theta` predicts a track:

```math
\theta \mapsto \zeta_k(\theta).
```

The semicoherent statistic is then

```math
\mathcal S(\theta)
=
\sum_k w_k P_k[\zeta_k(\theta)].
```

The number of dimensions should be selected in three stages.

1. **Approximation stage.** Determine how many local phase directions are needed to keep coherent residual mismatch below tolerance.

2. **Grid stage.** Determine whether those directions can be gridded/interpolated affordably.

3. **Search stage.** Keep the dimension only if it lowers detectable amplitude after cost, memory, and trials-factor penalties.

The small-residual mismatch result derived in the theory artifact is:

```math
\mu_k
\approx
\langle R_k^2\rangle_k-\langle R_k\rangle_k^2,
```

where `R_k` is the residual phase after fitting the retained `zeta` coordinates. If the first omitted phase term is order `m`, then approximately

```math
\mu_k \propto T_{\rm coh}^{2m}.
```

This explains why adding a coordinate can allow longer coherent chunks. It does not prove that doing so improves the search.

## Sensitivity Versus Computing Cost

The relevant comparison is not recovered coherent power. It is detectable amplitude at fixed false alarm:

```math
h_{\min}^2
\approx
\frac{
\mathcal S_{\rm th}-E_0[\mathcal S]
}{
\sum_k w_k\kappa_k
}.
```

Adding a `zeta` dimension can improve the denominator by reducing mismatch or permitting larger `T_coh`. But it can also raise the threshold and cost through:

- more `zeta` grid points;
- higher spectrum-generation cost;
- higher memory use;
- more expensive interpolation;
- more track lookups;
- more correlated templates;
- larger effective trials factor;
- harder null calibration.

A schematic cost model is:

```math
C_{\rm total}
=
C_{\rm spectra}
+
C_{\rm tracks}
+
C_{\rm interp}
+
C_{\rm overhead},
```

with

```math
C_{\rm spectra}
\sim
N_{\rm seg}N_\zeta c_{\rm coh},
\qquad
C_{\rm tracks}
\sim
N_\theta N_{\rm seg}c_{\rm lookup}(d_\zeta),
```

and

```math
M_{\rm spectra}
\sim
N_{\rm seg}N_\zeta B.
```

The practical decision rule is therefore:

> Add a coordinate only if `h_min(d+1) < h_min(d)` under the same compute, memory, and total false-alarm constraints.

## Analytical Versus Numerical Bases

For polynomial phase models, analytic or semi-analytic bases are plausible. With

```math
\phi(u)=\sum_j a_j u^j,
```

the mismatch is quadratic in the coefficients, and orthogonal polynomials or Gram-matrix eigenvectors diagonalize the residual under simple windows/noise weights.

For PN waveforms, analytic guidance exists through 0PN, chirp-time, PN, or metric-adapted coordinates. But the optimal low-dimensional basis depends on the mass range, frequency range, chunk duration, window, noise weighting, and nuisance projections.

The human intuition is likely right in one limited sense: if only one additional non-spinning chirp coordinate is allowed, a 0PN-like direction is a strong candidate because the Newtonian term dominates accumulated phase. But after projecting out constant phase and local frequency, “dominates accumulated phase” is not identical to “dominates residual mismatch.” That has to be tested.

## Case A: Equal-Mass Non-Spinning 3.5PN Waveforms

For exactly equal-mass, non-spinning 3.5PN waveforms, the intrinsic physical family is effectively one-dimensional once mass ratio and spin are fixed. That makes this an important but forgiving test.

The key question is:

> After projecting out constant phase and local carrier frequency, is the best one-dimensional residual basis actually close to the 0PN direction?

Recommended comparison:

- frequency only;
- frequency plus 0PN;
- frequency plus best one SVD/PCA direction;
- frequency plus two or more SVD/PCA directions;
- frequency plus higher-PN inspired coordinates.

Required outputs:

- singular value spectrum;
- overlap between first SVD/PCA direction and 0PN;
- average and worst-case held-out mismatch versus dimension;
- dependence on `T_coh`;
- grid-size and memory implication;
- final `h_min` estimate at fixed compute.

Conjecture: 0PN will be close to optimal for narrow mass/frequency ranges and moderate `T_coh`. Broader ranges or longer chunks may need more than one local phase direction even though the physical family is one-dimensional.

## Case B: Equal-Mass Spinning 3.5PN Waveforms

The spinning case is not yet well-defined. The report must first choose one spin convention:

- equal aligned spins, `chi1 = chi2 in [0, 0.2]`;
- equal signed aligned spins, `chi1 = chi2 in [-0.2, 0.2]`;
- independent aligned spins;
- generic precessing spins.

These are different waveform families.

For the first pass, the cleanest auditable choice is equal aligned spin:

```math
\chi_1=\chi_2=\chi,
```

with either `0 <= chi <= 0.2` or `|chi| <= 0.2` explicitly stated.

Recommended comparison:

- non-spinning basis applied to spinning waveforms;
- spinning SVD/PCA basis;
- 0PN plus leading spin-PN coordinate;
- one-, two-, and three-coordinate bases;
- held-out mismatch versus dimension;
- final costed sensitivity.

Conjecture: modest equal aligned spin may add one dominant residual direction, partly degenerate with mass/chirp directions. Signed spin or independent spins may require more directions. Generic precession should be treated as a separate problem.

## Tensions Across Artifacts

The planner correctly broadened the research program, but the referee is right that the decisive objective must be narrower: lower detectable amplitude at fixed resources.

The literature review shows high novelty risk for broad claims. Fast chirp transforms and polynomial-phase transforms are close to `{f, fdot, ...}` spectra; Weave and semicoherent metrics are close to costed semicoherent template placement. The useful distinction to preserve is reusable chunk-local `zeta` spectra versus direct physical-template banks.

The theory artifact gives a coherent mathematical scaffold, but it remains an optimization scaffold, not a numerical result. The SVD/PCA study requested by the human is essential, but it answers only the representation part unless connected to grid cost, trials factor, and calibrated sensitivity.

## Assumptions To Make Explicit

Before quantitative claims, fix:

- PN phase convention;
- time-domain versus frequency-domain phase;
- mass range;
- frequency band;
- `T_coh` grid;
- spin convention;
- nuisance directions projected out;
- window and inner product;
- angular-frequency versus Hz convention;
- noise model;
- FFT/NUFFT normalization;
- null calibration method.

Without these, statements like “0PN is optimal” or “spin needs one more coordinate” are not auditable.

## Concrete Next Steps

1. Define the 3.5PN waveform convention, mass range, frequency band, and `T_coh` values.

2. Implement the residual-phase projection:
   project out constant phase and local frequency, then compare 0PN, PN-inspired, and SVD/PCA bases.

3. Run the equal-mass non-spinning SVD/PCA study first. Report singular values, 0PN overlap, and held-out mismatch.

4. Repeat with the chosen equal-aligned-spin convention up to `chi = 0.2`.

5. Convert mismatch curves into grid sizes, interpolation requirements, memory estimates, and runtime measurements.

6. Calibrate null distributions and effective trials factors for each candidate `dim(zeta)`.

7. Produce the decisive plot or table:
   detectable amplitude versus compute for frequency-only, 0PN, SVD/PCA, higher-PN, and relevant chirp-transform or semicoherent matched-filter baselines.

## Bottom Line

The right question is not “is there a better basis than 0PN?” in isolation. The right question is:

> For a specified waveform family, chunk duration, search volume, and compute budget, which `zeta` basis and dimension minimizes detectable amplitude at fixed false alarm?

For equal-mass non-spinning 3.5PN waveforms, 0PN is a strong one-coordinate candidate but must be tested against SVD/PCA after nuisance projection. For equal-mass spinning waveforms with `chi <= 0.2`, the answer depends on the spin convention and likely requires at least one additional spin-sensitive comparison. The project should proceed as a costed dimension-selection study, not as a claim of a new semicoherent paradigm.