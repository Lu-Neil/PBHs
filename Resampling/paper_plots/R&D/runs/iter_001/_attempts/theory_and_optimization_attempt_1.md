# Theory and Optimization: Choosing `zeta` and `T_coh`

## Audit Metadata

- Source of truth: `idea.md`, human feedback, planner artifact, and literature artifact.
- Agent: `theory_and_optimization`.
- Attempt: 1.
- Required output: `runs/iter_001/02_theory_and_optimization.md`.
- Status: mathematical formalism and optimization scaffold. This is not yet a numerical result.
- Citation policy: no new citations are introduced here. Prior-art claims are inherited only from the literature artifact and should remain auditable there.
- Main distinction: `theta` denotes physical signal parameters; `zeta` denotes chunk-local spectral coordinates used to build reusable coherent spectra.

## Executive Summary

The core decision is not whether a higher-dimensional coherent spectrum can recover more signal power. It usually can. The real decision is whether the recovered-power gain survives the extra cost, memory, interpolation error, and false-alarm threshold penalty.

The proposed optimization problem is:

```math
(d_\zeta^\star, T_{\rm coh}^\star, \mathcal{B}^\star)
=
\arg\min h_{\min}
```

subject to

```math
C_{\rm total}(d_\zeta,T_{\rm coh},\mathcal{B}) \le C_{\rm budget},
\qquad
M_{\rm total}(d_\zeta,T_{\rm coh},\mathcal{B}) \le M_{\rm budget},
\qquad
P_{\rm FA} \le P_{\rm FA}^{\rm target}.
```

Here `d_zeta` is the number of chunk-local spectral coordinates, `T_coh` is the coherent chunk duration, and `mathcal{B}` is the chosen coordinate basis.

The working rule is:

> Add a `zeta` dimension only if the coherent mismatch reduction lowers the final detectable amplitude at fixed compute, memory, and false-alarm probability.

This separates the proposed search from ordinary StackSlide, which corresponds to the limiting case `zeta = {f}` with segment time used only as a label.

## 1. Signal Parameters `theta`

### Established From The Problem

The physical signal family is modeled by parameters

```math
\theta.
```

These parameters determine the phase evolution

```math
\phi(t;\theta)
```

and, more generally, the strain

```math
h(t;\theta)
=
A(t;\theta)\cos[\phi(t;\theta)+\phi_0]
```

or its complex analytic representation

```math
h_a(t;\theta)
=
\mathcal{A}(t;\theta)\exp[i\phi(t;\theta)].
```

For this artifact, it is useful to split

```math
\theta = (\lambda, \alpha),
```

where:

- `lambda` are intrinsic phase-evolution parameters;
- `alpha` are amplitude, polarization, sky-response, and other extrinsic parameters.

For the PBH/chirping use case, examples of intrinsic parameters include frequency, chirp mass, chirp parameter `beta`, coalescence time, and possible PN or spin parameters.

For the later 3.5PN case studies:

```math
\theta_{\rm NS}
=
(M_c \ \text{or} \ M, \eta=1/4, \chi_1=\chi_2=0)
```

for equal-mass non-spinning systems, and a minimal aligned-spin extension could be

```math
\theta_{\rm spin}
=
(M_c \ \text{or} \ M, \eta=1/4, \chi_1=\chi_2=\chi),
\qquad
|\chi| \le 0.2
```

or

```math
0 \le \chi \le 0.2.
```

The spin convention is not fixed by `idea.md`. Any future simulation must specify whether spin is signed, aligned-only, equal-spin, independent-spin, or precessing.

### Assumption

This artifact treats amplitude variations inside each coherent chunk as either slowly varying or handled by weights. The main derivations focus on phase mismatch.

## 2. Spectral Coordinates `zeta`

The coherent chunk analysis does not need to use the physical parameters directly. Instead, each chunk has local spectral coordinates

```math
\zeta_k.
```

These coordinates parameterize the local demodulation phase

```math
\Phi_{\rm loc}(u;\zeta_k),
\qquad
u = t - t_k,
```

where `t_k` is the center time of chunk `k`.

Examples:

### Frequency-Only Coordinate

```math
\zeta = \{f\},
\qquad
\Phi_{\rm loc}(u;f)=2\pi f u.
```

This is the ordinary time-frequency / StackSlide-like case.

### Polynomial-Phase Coordinates

```math
\zeta = \{f,\dot f,\ddot f,\ldots\},
```

with

```math
\Phi_{\rm loc}(u;\zeta)
=
2\pi
\left[
f u
+
\frac{1}{2}\dot f u^2
+
\frac{1}{6}\ddot f u^3
+
\cdots
\right].
```

### PBH / Chirp Coordinates

A simplified local model might use

```math
\zeta = \{f,\beta\}
```

or

```math
\zeta = \{f,t,\beta\},
```

depending on whether segment time is treated as an explicit spectral coordinate or only as a segment label.

### PN-Oriented Coordinates

For PN waveform families, one may define

```math
\Phi_{\rm loc}(u;\zeta)
=
\sum_{a=1}^{d_\zeta}
\zeta^a e_a(u),
```

where `e_a(u)` may be:

- PN phase basis functions;
- chirp-time basis functions;
- Taylor coefficients;
- metric eigenvectors;
- SVD/PCA basis vectors.

### Important Distinction

`dim(theta)` is not the same as `dim(zeta)`.

A physical waveform family can be one-dimensional after restrictions, while its local phase residuals may still require multiple basis functions over long coherent chunks. Conversely, a high-dimensional physical family may have only a few dominant local phase directions over a restricted parameter range.

## 3. Mapping Between `theta` and `zeta`

A physical template `theta` predicts a track through the chunk-local spectral coordinates:

```math
\theta \mapsto \zeta_k(\theta).
```

The cleanest definition is a local projection problem.

Let the exact signal phase in chunk `k` be

```math
\phi_k(u;\theta)
=
\phi(t_k+u;\theta).
```

Choose `zeta_k(theta)` and an arbitrary constant phase `phi_{0,k}` by minimizing the residual phase norm:

```math
(\phi_{0,k},\zeta_k)
=
\arg\min_{\phi_0,\zeta}
\left\|
\phi_k(u;\theta)
-
\phi_0
-
\Phi_{\rm loc}(u;\zeta)
\right\|_k^2.
```

Define the inner product

```math
\langle a,b\rangle_k
=
\int_{-T_{\rm coh}/2}^{T_{\rm coh}/2}
du\,
q_k(u)\,
a(u)b(u),
```

where `q_k(u)` encodes the window, amplitude weighting, and possibly inverse noise weighting. A normalized version can impose

```math
\langle 1,1\rangle_k = 1.
```

If the local phase model is linear in coordinates,

```math
\Phi_{\rm loc}(u;\zeta)
=
\sum_{a=1}^{d_\zeta}
\zeta^a e_a(u),
```

then the mapping is given by normal equations after projecting out constant phase:

```math
G_{ab}^{(k)} \zeta_k^b
=
b_a^{(k)},
```

with

```math
G_{ab}^{(k)}
=
\langle \tilde e_a,\tilde e_b\rangle_k,
\qquad
b_a^{(k)}
=
\langle \tilde e_a,\tilde\phi_k\rangle_k.
```

Here tildes denote projection orthogonal to the constant phase direction:

```math
\tilde y
=
y
-
\frac{\langle y,1\rangle_k}{\langle 1,1\rangle_k}.
```

Thus

```math
\zeta_k^a(\theta)
=
(G^{-1})^{ab} b_b.
```

This is an established least-squares projection result, not a conjecture.

For Taylor coordinates, the mapping can instead be defined by local derivatives:

```math
f_k(\theta)
=
\frac{1}{2\pi}
\left.
\frac{d\phi}{dt}
\right|_{t_k},
```

```math
\dot f_k(\theta)
=
\frac{1}{2\pi}
\left.
\frac{d^2\phi}{dt^2}
\right|_{t_k},
```

and so on. The derivative definition and the projection definition agree only when the retained Taylor expansion is adequate over the chunk.

## 4. Coherent Chunk Statistic

Split the data into coherent chunks

```math
I_k =
[t_k-T_{\rm coh}/2,\ t_k+T_{\rm coh}/2].
```

Let the analytic strain be

```math
x_a(t)=n_a(t)+h_a(t;\theta).
```

Define a coherent demodulated chunk output

```math
X_k(\zeta)
=
\int_{I_k}
dt\,
W_k(t)
x_a(t)
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

A normalized scalar power statistic is

```math
P_k(\zeta)
=
\frac{|X_k(\zeta)|^2}{\sigma_k^2(\zeta)}.
```

For stationary Gaussian noise and slowly varying weighting, a schematic normalization is

```math
\sigma_k^2(\zeta)
=
E_0[|X_k(\zeta)|^2].
```

If the data are colored, gapped, resampled, or NUFFT-transformed, `sigma_k^2` must be measured or predicted. In the local project, this is especially important because the NUFFT/resampling operation changes the effective noise PSD.

### 5-Vector Generalization

For the sidereal 5-vector pipeline, the chunk output may be a vector

```math
\mathbf{X}_k(\zeta)
```

rather than a scalar. Then the appropriate statistic is likely covariance-weighted:

```math
\widehat a_k
=
\frac{
\mathbf{A}_k^\dagger C_k^{-1}\mathbf{X}_k
}{
\mathbf{A}_k^\dagger C_k^{-1}\mathbf{A}_k
},
```

with detection power proportional to

```math
|\widehat a_k|^2
\,
\mathbf{A}_k^\dagger C_k^{-1}\mathbf{A}_k.
```

This is a recommended generalization, not yet validated here.

## 5. Semicoherent Statistic

A physical template `theta` defines a track

```math
\zeta_k(\theta)
```

through the chunk-local spectra. The semicoherent statistic is

```math
\mathcal{S}(\theta)
=
\sum_{k=1}^{N_{\rm seg}}
w_k(\theta)
P_k[\zeta_k(\theta)].
```

The number of segments is

```math
N_{\rm seg}
=
\frac{T_{\rm obs}}{T_{\rm coh}}
```

when chunks are non-overlapping and cover the observation span.

If interpolation is needed because `zeta_k(theta)` does not fall exactly on the spectral grid, replace

```math
P_k[\zeta_k(\theta)]
```

by an interpolated value

```math
\mathcal{I}[P_k](\zeta_k(\theta)).
```

Interpolation error should be counted as mismatch.

## 6. Weights

### Equal Weights

The simplest statistic uses

```math
w_k = 1.
```

This is adequate only for ideal chunks with identical noise, identical antenna response, identical mismatch, and identical normalization.

### Inverse-Variance / PowerFlux-Like Weights

If each chunk has expected signal excess `lambda_k` and noise variance `v_k`, a locally optimal weak-signal linear power sum uses weights proportional to signal-to-noise contribution:

```math
w_k \propto \frac{\lambda_k}{v_k}.
```

In a simple scalar model,

```math
\lambda_k
\propto
h_0^2
T_{\rm coh}
\frac{A_k^2}{S_{n,k}}
(1-\mu_k),
```

where:

- `A_k` represents detector response and amplitude modulation;
- `S_{n,k}` is the effective noise level;
- `mu_k` is coherent mismatch.

For exponential normalized powers under the null,

```math
v_k = 1,
```

so

```math
w_k \propto \lambda_k.
```

For unknown amplitude/polarization, weights should either be marginalized, maximized, or replaced by a vector/covariance statistic.

### Assumption

The formulas above assume independent chunk powers. Correlations from overlapping windows, interpolation, gaps, or NUFFT sidebands modify both optimal weights and null distributions.

## 7. Null Distribution

### Ideal Scalar Case

Assume:

1. Gaussian stationary noise.
2. Correct normalization.
3. Independent chunks.
4. Independent selected spectral bins.
5. Complex coherent outputs.

Then under the noise-only hypothesis `H_0`,

```math
X_k/\sigma_k \sim \mathcal{CN}(0,1),
```

and

```math
P_k = |X_k|^2/\sigma_k^2 \sim \mathrm{Exp}(1).
```

For equal weights,

```math
\mathcal{S}
=
\sum_{k=1}^{N_{\rm seg}}P_k
\sim
\Gamma(N_{\rm seg},1).
```

The null mean and variance are

```math
E_0[\mathcal{S}]
=
N_{\rm seg},
```

```math
\mathrm{Var}_0[\mathcal{S}]
=
N_{\rm seg}.
```

For unequal weights,

```math
\mathcal{S}
=
\sum_k w_k P_k,
```

so

```math
E_0[\mathcal{S}]
=
\sum_k w_k,
```

```math
\mathrm{Var}_0[\mathcal{S}]
=
\sum_k w_k^2.
```

The full distribution is a weighted sum of exponentials, also called a hypoexponential distribution when weights are positive and distinct. For many segments, a Gaussian approximation may be acceptable:

```math
\mathcal{S}
\approx
\mathcal{N}
\left(
\sum_k w_k,\,
\sum_k w_k^2
\right).
```

This approximation must be calibrated for the real pipeline.

### Trials Factor

A search evaluates many templates. If the single-template false-alarm probability is `p_single`, then the total false-alarm probability is approximately

```math
P_{\rm FA,total}
\approx
1-(1-p_{\rm single})^{N_{\rm eff}}
\approx
N_{\rm eff}p_{\rm single}
```

for small `p_single`.

`N_eff` is the effective number of independent trials, not necessarily the raw number of templates. Higher-dimensional `zeta` spectra can increase `N_eff` through:

- more spectral bins;
- more template tracks;
- more interpolation locations;
- more candidate local phase models.

This threshold penalty must be included in the optimization.

## 8. Signal Expectation

Under a signal matching template `theta`, the demodulated output has nonzero mean:

```math
E[X_k(\zeta_k(\theta))]
=
H_k(\theta).
```

Define the noncentrality

```math
\lambda_k(\theta)
=
\frac{|H_k(\theta)|^2}{\sigma_k^2}.
```

Then in the ideal scalar case,

```math
P_k
```

has a noncentral chi-square-like exponential-power distribution with mean

```math
E_1[P_k]
=
1+\lambda_k.
```

Therefore

```math
E_1[\mathcal{S}]
=
\sum_k w_k(1+\lambda_k),
```

and the expected signal excess is

```math
E_1[\mathcal{S}]-E_0[\mathcal{S}]
=
\sum_k w_k\lambda_k.
```

Mismatch enters approximately as

```math
\lambda_k
=
\lambda_{k,\mathrm{ideal}}(1-\mu_k)
```

for small mismatch, or more exactly through the squared coherent overlap.

A schematic weak-signal scaling is

```math
\lambda_{k,\mathrm{ideal}}
\propto
h_0^2
T_{\rm coh}
\frac{A_k^2}{S_{n,k}^{\rm eff}}.
```

Thus longer `T_coh` increases per-chunk noncentrality, but only if coherent mismatch remains controlled.

## 9. Residual Phase and Mismatch Criterion

Write the exact phase as

```math
\phi(t_k+u;\theta)
=
\phi_{0,k}
+
\Phi_{\rm loc}(u;\zeta_k)
+
R_k(u;\theta).
```

The coherent overlap between the exact signal and local model is approximately

```math
\mathcal{O}_k
=
\frac{
\left|
\left\langle
e^{iR_k}
\right\rangle_k
\right|
}{
1
},
```

where the weighted average is

```math
\langle y\rangle_k
=
\frac{
\int du\,q_k(u)y(u)
}{
\int du\,q_k(u)
}.
```

For small residual phase,

```math
e^{iR_k}
\approx
1+iR_k-\frac{1}{2}R_k^2.
```

After maximizing over an arbitrary constant phase, the coherent mismatch is

```math
\mu_k
=
1-|\mathcal{O}_k|^2
\approx
\langle R_k^2\rangle_k
-
\langle R_k\rangle_k^2.
```

This is a derived small-residual result.

If the local coordinates are chosen by least-squares projection, then `R_k` is orthogonal to the retained basis functions and to the constant phase direction. The mismatch is then the weighted residual phase variance.

### Omitted-Term Scaling

Suppose the first omitted phase term is

```math
R_k(u)
\approx
a_m u^m.
```

For a rectangular window on

```math
u \in [-T_{\rm coh}/2,T_{\rm coh}/2],
```

with uniform weighting, the residual variance is

```math
\mu_k
\approx
a_m^2
\left[
\langle u^{2m}\rangle
-
\langle u^m\rangle^2
\right].
```

For odd `m`,

```math
\langle u^m\rangle=0,
```

and

```math
\mu_k
\approx
a_m^2
\frac{(T_{\rm coh}/2)^{2m}}{2m+1}.
```

For even `m`,

```math
\langle u^m\rangle
=
\frac{(T_{\rm coh}/2)^m}{m+1},
```

so

```math
\mu_k
\approx
a_m^2
(T_{\rm coh}/2)^{2m}
\left[
\frac{1}{2m+1}
-
\frac{1}{(m+1)^2}
\right].
```

Thus the generic scaling is

```math
\mu_k
\propto
a_m^2 T_{\rm coh}^{2m}.
```

If the local phase model retains Taylor terms through order `p`, the first omitted term has

```math
m=p+1
```

and

```math
a_m
=
\frac{1}{m!}
\left.
\frac{d^m\phi}{dt^m}
\right|_{t_k}.
```

Therefore

```math
\mu_k
\propto
\left[
\phi^{(p+1)}(t_k)
\right]^2
T_{\rm coh}^{2p+2}.
```

This shows why adding coherent `zeta` dimensions can allow longer `T_coh`.

## 10. Dimension Selection by Metric, SVD, or PCA

Define a candidate local basis

```math
\{e_a(u)\}_{a=1}^{d_{\rm max}}.
```

After projecting out constant phase and baseline coordinates, build residual phase functions

```math
r_i(u)
=
\phi(u;\theta_i)
-
\phi_{\rm fitted}(u;\zeta_{\rm baseline}).
```

The index `i` labels sampled physical templates.

Construct the residual covariance operator

```math
K(u,u')
=
\sum_i \pi_i r_i(u)r_i(u'),
```

where `pi_i` is the sampling weight over physical parameter space.

Equivalently, form a residual matrix and run SVD under the weighted inner product. The best rank-`d` approximation in this chosen norm retains the first `d` singular directions.

The discarded phase variance is

```math
\epsilon_d
=
\sum_{\alpha>d}s_\alpha^2,
```

where `s_alpha` are singular values in the chosen normalization.

A candidate dimension rule is

```math
\epsilon_d \le \mu_{\rm coh,max}.
```

But this is only a mismatch rule. It is not yet the search-optimal rule.

The search-optimal rule is:

```math
d_\zeta^\star
=
\arg\min_d h_{\min}(d)
```

subject to cost, memory, and false-alarm constraints.

### Important Caveat

SVD/PCA optimality is conditional on:

- the parameter range;
- the sampling measure over `theta`;
- the phase convention;
- the frequency or time interval;
- the window;
- the noise weighting;
- which nuisance directions were projected out;
- whether the basis is allowed to be linear or nonlinear.

Therefore, statements like “the first PCA basis is optimal” are incomplete unless all of the above choices are recorded.

## 11. Sensitivity Versus Computational-Cost Tradeoff

### Sensitivity Model

Let the threshold for total false-alarm probability be

```math
\mathcal{S}_{\rm th}(d,T_{\rm coh}).
```

For a weak signal, approximate detection by requiring

```math
E_1[\mathcal{S}]
-
E_0[\mathcal{S}]
\gtrsim
\kappa
\sqrt{\mathrm{Var}_0[\mathcal{S}]},
```

with `kappa` set by false-alarm and false-dismissal probabilities.

Using

```math
E_1[\mathcal{S}]-E_0[\mathcal{S}]
=
\sum_k w_k\lambda_k,
```

and

```math
\lambda_k
\propto
h_0^2
T_{\rm coh}
\frac{A_k^2}{S_{n,k}^{\rm eff}}
(1-\mu_k),
```

one obtains a schematic detectable amplitude

```math
h_{\min}^2
\propto
\frac{
\mathcal{S}_{\rm th}-E_0[\mathcal{S}]
}{
T_{\rm coh}
\sum_k
w_k
A_k^2
(1-\mu_k)
/S_{n,k}^{\rm eff}
}.
```

For identical chunks, equal weights, fixed trials factor, and small mismatch, this reduces to the usual semicoherent scaling

```math
h_{\min}
\propto
S_n^{1/2}
T_{\rm coh}^{-1/4}
T_{\rm obs}^{-1/4}.
```

This scaling is a limiting idealization. It does not include grid mismatch, changing trials factors, non-Gaussian noise, or real data gaps.

### Cost Model

Decompose total cost as

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
N_\zeta
=
\prod_{a=1}^{d_\zeta} N_a
```

be the number of grid points per coherent chunk.

A generic spectrum-construction cost is

```math
C_{\rm spectra}
\sim
N_{\rm seg}
N_\zeta
c_{\rm coh}(T_{\rm coh},d_\zeta).
```

Track accumulation cost is

```math
C_{\rm tracks}
\sim
N_\theta
N_{\rm seg}
c_{\rm lookup}(d_\zeta).
```

If multilinear interpolation is used,

```math
c_{\rm lookup}(d_\zeta)
\propto
2^{d_\zeta}
```

is a plausible scaling, but it must be measured in the actual implementation.

Stored spectra require memory

```math
M_{\rm spectra}
\sim
N_{\rm seg}
N_\zeta
B,
```

where `B` is bytes per stored statistic.

The number of grid points in each coordinate is set by a mismatch metric. If coordinate `zeta^a` has extent `L_a` and grid spacing `Delta zeta^a`, then

```math
N_a
\sim
\frac{L_a}{\Delta \zeta^a}.
```

The grid spacing is constrained by interpolation and discretization mismatch:

```math
\mu_{\rm grid}
\lesssim
\mu_{\rm grid,max}.
```

Metric-based spacing gives schematically

```math
\mu_{\rm grid}
\approx
g_{ab}^{(\zeta)}
\Delta\zeta^a\Delta\zeta^b.
```

Adding a dimension can reduce coherent model mismatch `mu_model`, but increases:

```math
N_\zeta,
\quad
C_{\rm spectra},
\quad
C_{\rm interp},
\quad
M_{\rm spectra},
\quad
N_{\rm eff}.
```

### Optimization Condition

An extra coordinate is beneficial only if

```math
h_{\min}(d+1,T_{\rm coh})
<
h_{\min}(d,T_{\rm coh})
```

under the same compute, memory, and false-alarm constraints.

Equivalently, the fractional sensitivity gain from reduced mismatch and possibly longer `T_coh` must exceed the fractional loss from:

- reduced number of templates affordable at fixed cost;
- higher threshold from larger trials factor;
- interpolation error;
- memory pressure;
- larger spectrum-construction cost.

## 12. How To Choose `T_coh`

For each candidate basis dimension `d`, there is a maximum useful coherent time set by residual phase mismatch:

```math
\max_k \mu_k(d,T_{\rm coh})
\le
\mu_{\rm coh,max}.
```

Since omitted-term mismatch often scales like

```math
\mu_k \propto T_{\rm coh}^{2m},
```

higher-dimensional local phase models can permit much larger `T_coh`.

However, longer `T_coh` also changes:

- number of segments `N_seg`;
- per-segment FFT or NUFFT cost;
- frequency-bin spacing;
- number of required templates;
- sensitivity scaling;
- data stationarity assumptions;
- gap handling;
- sidereal modulation behavior;
- trials factor.

The practical prescription is:

1. For each candidate `d`, find the largest `T_coh` satisfying coherent mismatch limits over the parameter space.
2. Compute the spectrum and track costs for that `d,T_coh`.
3. Compute the detection threshold including effective trials.
4. Estimate `h_min`.
5. Choose the pair with the lowest `h_min` within resource limits.

## 13. Analytical Basis Construction

### Polynomial Phase

For polynomial phase,

```math
\phi(u)
=
\sum_{j=0}^{J} a_j u^j,
```

the mismatch norm is quadratic in the coefficients after projecting out nuisance terms. With a simple window, the Gram matrix is

```math
G_{ij}
=
\langle u^i,u^j\rangle.
```

Orthogonal polynomials diagonalize this Gram matrix under the chosen weight. Therefore, for polynomial-phase families, an analytical or semi-analytical optimal basis exists in the sense of minimizing residual phase variance under that inner product.

This supports the idea that the generic signal-processing problem can be attacked analytically in idealized cases.

### PN Phase

For PN waveforms, the phase can often be written as a combination of known frequency-domain or time-domain basis functions with parameter-dependent coefficients. Schematically,

```math
\phi(u;\theta)
=
\sum_j c_j(\theta) \psi_j(u).
```

Given this form, one can build the weighted Gram matrix

```math
G_{ij}
=
\langle \tilde\psi_i,\tilde\psi_j\rangle,
```

after projecting out constant phase, frequency, and any other always-retained coordinates.

Then one can diagonalize either:

1. the function-space Gram matrix, or
2. the parameter-induced covariance

```math
K_{ij}
=
\sum_\theta \pi(\theta)c_i(\theta)c_j(\theta).
```

This gives metric/SVD coordinates.

### Is 0PN Analytically Expected To Be Best?

Established from PN intuition: the Newtonian or 0PN chirp term dominates the accumulated phase in many inspiral regimes.

Conjecture: for equal-mass non-spinning 3.5PN waveforms over restricted mass and frequency ranges, the best one-dimensional residual phase basis may be close to the 0PN direction.

Not established: that 0PN is the best one-dimensional `zeta` coordinate for the actual chunk lengths, frequency ranges, and projection conventions of this project.

Reason: once constant phase and frequency are removed, “dominant accumulated phase” is not automatically the same as “dominant residual phase variance inside a finite chunk.” The answer depends on the inner product and parameter range.

## 14. Equal-Mass Non-Spinning 3.5PN Case

### Physical Dimension

If the mass ratio is fixed exactly at

```math
\eta = 1/4
```

and spins are zero, the intrinsic physical family is effectively one-dimensional, controlled by total mass or chirp mass.

### Effective Local Spectral Dimension

Even with one intrinsic physical parameter, the local residual phase over a chunk may not be represented perfectly by one chosen coordinate such as 0PN `beta`. The mapping

```math
M_c \mapsto \zeta_k(M_c)
```

may trace a curved path through a higher-dimensional function space.

For short chunks, one coordinate may be enough. For longer chunks or wider mass ranges, higher-order PN curvature can require additional basis directions to keep mismatch below tolerance.

### Required Analysis

For each `T_coh` and frequency/mass range:

1. Generate 3.5PN equal-mass non-spinning phases.
2. Project out constant phase and frequency.
3. Compare residual bases:
   - 0PN direction;
   - best one-dimensional SVD/PCA direction;
   - best two-dimensional SVD/PCA basis;
   - PN-inspired bases.
4. Measure:
   - residual mismatch versus dimension;
   - overlap between 0PN and first SVD direction;
   - held-out waveform mismatch;
   - grid size required for each coordinate choice.
5. Feed mismatch and grid size into the cost-sensitivity model.

### Expected But Unproven Outcome

0PN may be close to optimal for one coordinate in narrow parameter ranges. It may fail for longer chunks or broader ranges where higher-PN terms create measurable residual curvature.

This must be treated as a conjecture until numerical or analytical metric results are available.

## 15. Equal-Mass Spinning 3.5PN Case

### Spin Convention Needed

The phrase “spinning up to `chi = 0.2`” is ambiguous. The minimal case for analysis should be explicitly defined, for example:

```math
\chi_1=\chi_2=\chi,
\qquad
0 \le \chi \le 0.2
```

or

```math
\chi_1=\chi_2=\chi,
\qquad
-0.2 \le \chi \le 0.2.
```

Generic precessing spins are a different problem and should not be mixed into this first study.

### Effective Dimension

Allowing spin adds physical phase variation. It may:

1. project mostly onto the non-spinning mass-like basis;
2. add one dominant spin-like residual direction;
3. add multiple directions if the spin range is broad or signed;
4. require amplitude/precession modeling if generic spins are allowed.

### Required Analysis

Repeat the non-spinning SVD/PCA study with spin included.

Compare:

- non-spinning basis applied to spinning waveforms;
- spinning SVD/PCA basis;
- 0PN plus leading spin-PN coordinate;
- one-, two-, and three-dimensional retained bases.

Measure:

```math
\mu_{\rm spin}(d,T_{\rm coh})
```

and determine the smallest `d` satisfying the coherent mismatch tolerance.

### Conjecture

For equal-mass equal-aligned spin with modest `|chi| <= 0.2`, one additional dominant direction may be enough. This is uncertain and must be tested. If signed spin is allowed, symmetry around zero spin may make the required basis different from the one-sided case.

## 16. Limiting Case: Ordinary StackSlide

Set

```math
\zeta = \{f\}
```

and

```math
\Phi_{\rm loc}(u;f)=2\pi f u.
```

For each chunk,

```math
X_k(f)
=
\int_{I_k}dt\,
W_k(t)x_a(t)e^{-i2\pi f(t-t_k)}.
```

This is the short-time Fourier coefficient up to phase convention.

The physical template predicts a frequency track

```math
f_k(\theta)
=
\frac{1}{2\pi}
\left.
\frac{d\phi}{dt}
\right|_{t_k}.
```

The semicoherent statistic becomes

```math
\mathcal{S}_{\rm SS}(\theta)
=
\sum_k w_k P_k[f_k(\theta)].
```

This is ordinary StackSlide-like power accumulation along a frequency-time track. Segment time `t_k` is a label, not an additional searched coordinate, unless spectra are explicitly gridded over segment time.

Thus the proposed `zeta` formalism strictly contains ordinary StackSlide as the `d_zeta = 1` frequency-only limit.

## 17. Practical Decision Rule

For each candidate coordinate family:

```math
\zeta^{(d)}
=
\{zeta^1,\ldots,zeta^d\},
```

compute or estimate:

```math
\mu_{\rm model}(d,T_{\rm coh})
```

from residual phase;

```math
\mu_{\rm grid}(d,T_{\rm coh})
```

from spectral grid spacing;

```math
\mu_{\rm interp}(d,T_{\rm coh})
```

from track interpolation;

```math
C_{\rm total}(d,T_{\rm coh})
```

from spectrum creation and track summation;

```math
M_{\rm total}(d,T_{\rm coh})
```

from stored spectra;

```math
N_{\rm eff}(d,T_{\rm coh})
```

from the effective number of trials.

Then estimate

```math
h_{\min}(d,T_{\rm coh})
```

using the weighted semicoherent statistic and threshold.

Choose the dimension and coherent time that minimize `h_min`, not the dimension that merely minimizes residual phase.

## 18. Established Results, Derived Results, Conjectures

### Established From Problem And Prior Art Artifact

- Ordinary semicoherent methods sum powers along template-predicted tracks.
- Frequency-time StackSlide corresponds to `zeta = {f}`.
- Weighted sums are preferable when noise and detector response vary.
- Metric, SVD, PCA, and reduced-basis tools are appropriate for identifying dominant phase directions.
- Fast chirp / polynomial-phase transforms are novelty risks for spectra indexed by chirp coordinates.

### Derived Here

- The mapping `theta -> zeta_k(theta)` can be defined as a local phase projection.
- Small residual phase gives coherent mismatch

```math
\mu_k
\approx
\langle R_k^2\rangle_k-\langle R_k\rangle_k^2.
```

- If the first omitted term is order `m`, mismatch scales like

```math
\mu_k \propto T_{\rm coh}^{2m}.
```

- The ideal null distribution is Gamma for equal-weight sums of independent exponential powers.
- The ordinary StackSlide statistic is the `zeta={f}` limiting case.

### Conjectures

- The optimal `dim(zeta)` increases with `T_coh`.
- 0PN is close to the best one-dimensional coordinate for equal-mass non-spinning 3.5PN waveforms in restricted regimes.
- Spin up to `chi = 0.2` adds at least one useful residual phase direction, but perhaps not many.
- There exists a practical regime where `{f,beta}` or an SVD/PCA coordinate improves sensitivity at equal compute over frequency-only StackSlide.

### Open Questions

- What mass, frequency, and spin ranges should define the 3.5PN basis study?
- Is the project’s `{f,t,beta}` construction mathematically equivalent to an existing chirp transform?
- What is the correct covariance of NUFFT/resampled 5-vector bins?
- How large is the effective trials factor for higher-dimensional `zeta` spectra?
- Does memory bandwidth dominate the cost before FLOPs?
- Does the best one-dimensional SVD/PCA basis align quantitatively with 0PN?

## 19. Immediate Next Steps For Simulation Agents

1. Define the exact 3.5PN phase convention and parameter ranges.
2. Implement phase-residual projection for fixed `T_coh`.
3. Compare 0PN against the first SVD/PCA basis for equal-mass non-spinning waveforms.
4. Repeat with equal aligned spin up to `chi = 0.2`, after fixing the spin convention.
5. Convert residual mismatch into required grid spacings.
6. Estimate cost, memory, and trials factor for each retained dimension.
7. Report detectable amplitude at fixed compute and false-alarm probability, not just recovered power.