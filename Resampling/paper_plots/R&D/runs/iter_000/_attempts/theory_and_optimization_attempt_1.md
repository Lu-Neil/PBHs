---
title: Mathematical formalism for zeta-space semicoherent searches and optimization of T_coh
source: idea.md
agent: theory_and_optimization
attempt: 1
status: draft theory artifact
required_output: outputs/02_theory_and_optimization.md
---

## Audit Status

This artifact treats `idea.md` as the source of truth.

No shell commands were run. No files, code, Zotero database, or external papers were inspected in this attempt. Literature facts are limited to the prior literature artifact and are not independently verified here. Where a supporting reference is needed but not audited, I mark `citation needed`.

The results below are separated into:

- **Established from `idea.md`**: project premises supplied by the user.
- **Derived here**: algebraic consequences of the stated model assumptions.
- **Assumptions**: modeling choices needed to make the formalism precise.
- **Conjectures**: plausible optimization principles requiring simulation or literature confirmation.
- **Open questions**: items that need later validation.

## Executive Summary

The proposed search can be written as a semicoherent statistic built from coherent segment spectra indexed by an intermediate coordinate vector `zeta`:

```math
\mathcal{S}(\theta)
=
\sum_{k=1}^{N_{\rm seg}}
w_k(\theta)\,
p_k\!\left[\zeta_k(\theta)\right].
```

Here `theta` denotes physical signal parameters, while `zeta_k(theta)` is the segment-local spectral coordinate predicted by the signal model during coherent chunk `k`.

Ordinary StackSlide is recovered when `zeta` contains only the local instantaneous frequency, together with the segment time. The proposed generalization adds local phase-evolution coordinates, for example

```math
\zeta_k = \{f_k,\dot f_k,\ddot f_k,\ldots\}
```

or PN/chirp coordinates such as

```math
\zeta_k = \{f_k,\beta_{0{\rm PN},k},\beta_{1{\rm PN},k},\ldots\}.
```

The main tradeoff is:

- More `zeta` dimensions reduce coherent residual phase error and permit larger `T_coh`.
- More `zeta` dimensions increase coherent-spectrum construction cost, memory cost, interpolation cost, track-summation cost, and trials factor.

A practical optimization target is therefore

```math
(d_\zeta^*,T_{\rm coh}^*)
=
\arg\min_{d_\zeta,T_{\rm coh}}
C_{\rm total}(d_\zeta,T_{\rm coh})
```

subject to fixed detection probability, false-alarm probability, and maximum allowed mismatch.

## Claim Taxonomy

### Established From `idea.md`

- Conventional semicoherent searches often build time-frequency spectra and sum power along model tracks.
- The proposed approach builds spectra in a higher-dimensional space `zeta`, which includes frequency and time and may include additional phase-evolution coordinates.
- A local implementation exists nearby using `zeta = {f,t,beta}`, where `beta` is a 0PN parameter.
- Increasing the number of `zeta` coordinates may increase the allowed coherent chunk duration.
- Increasing the number of `zeta` coordinates may also increase computational cost in both spectrum construction and semicoherent track summation.
- The target question is how to choose `zeta` and `T_coh` optimally.

### Derived Here

- A generic power-sum statistic has a central chi-square or weighted exponential null distribution under Gaussian noise, subject to independence assumptions.
- The signal expectation is controlled by a noncentrality parameter proportional to coherent SNR squared times a mismatch factor.
- The coherent mismatch from using an incomplete `zeta` model is controlled, to leading order, by the variance of the residual phase over a segment.
- Adding `zeta` dimensions is beneficial only if the sensitivity gain from increased `T_coh` or reduced mismatch exceeds the increased cost and trials threshold.

### Assumptions

- Noise is initially modeled as stationary Gaussian noise within each coherent segment.
- Segment powers are treated as independent unless overlapping windows, gaps, or spectral interpolation create correlations.
- The coherent statistic is initially taken to be a complex demodulated amplitude squared; later versions may replace this with an F-statistic, 5-vector statistic, or likelihood-ratio statistic.
- The segment-local phase model can be represented by a finite basis.
- The mapping `theta -> zeta_k(theta)` is smooth enough for metric and Taylor-expansion arguments.

### Conjectures

- The optimal `zeta` basis is likely not the raw physical parameter vector. It is more likely to be a local phase basis, PN coefficient basis, chirp-time basis, metric eigenbasis, or PCA/SVD basis.
- For PN chirps, the optimal `d_zeta` should increase with `T_coh` because longer coherent chunks resolve higher phase curvature.
- There may be regimes where adding a `zeta` coordinate reduces total cost by allowing a much larger `T_coh`, even though each coherent spectrum becomes more expensive.
- There may also be regimes where extra `zeta` dimensions are never worthwhile because spectrum arrays become memory-bandwidth limited.

### Open Questions

- Has the optimal model-order choice for polynomial-phase or chirp-transform searches already been solved in signal-processing literature? Search target: fast chirp transform, polynomial phase transform, high-order ambiguity function, chirplet transform.
- What is the best statistic for the current PBH/chirping-signal use case: power sum, 5-vector likelihood, complex coherent sum with loose phase constraints, or a marginalized likelihood?
- How strongly are adjacent `zeta` grid points correlated in realistic NUFFT spectra?
- What is the empirical trials factor for a high-dimensional `zeta` track search?
- Can the local `zeta` projection be made computationally cheap enough for a large template bank?

## 1. Signal Parameters `theta`

Let the measured strain be

```math
x(t) = n(t) + h(t;\theta).
```

A useful decomposition is

```math
\theta = (\lambda,\mathcal{A}),
```

where:

- `lambda` are intrinsic and phase-evolution parameters.
- `mathcal{A}` are extrinsic amplitude, phase, and detector-response parameters.

For a gravitational-wave chirp-like signal with detector modulation, a representative parameter set is

```math
\lambda =
\{f_{\rm ref}, \beta, \beta_1,\beta_{3/2},\ldots,
\alpha,\delta,\ldots\},
```

```math
\mathcal{A}
=
\{h_0,\phi_0,\psi,\iota,\eta,\ldots\}.
```

Here:

- `f_ref` is a reference frequency.
- `beta` may denote a 0PN chirp parameter, as in the project premise.
- Higher `beta_i` or PN coefficients represent additional phase-evolution terms.
- `alpha, delta` are sky coordinates if detector modulation or barycentric effects are included.
- `h_0, phi_0, psi, iota, eta` represent amplitude, initial phase, polarization, inclination or ellipticity-like parameters, depending on the signal model.

For theory it is convenient to use an analytic signal representation:

```math
h(t;\theta)
=
A(t;\mathcal{A},\lambda)
\exp\{i\phi(t;\lambda)+i\phi_0\}.
```

For a real strain channel, this should be interpreted as the positive-frequency or quadrature representation. The real-data normalization must be checked in implementation.

## 2. Spectral Coordinates `zeta`

Split the observation into `N_seg` coherent chunks of duration `T_coh`, centered at times `t_k`:

```math
I_k = [t_k - T_{\rm coh}/2,\ t_k + T_{\rm coh}/2].
```

Define the local time coordinate

```math
u = t - t_k.
```

In each segment, choose a local phase model

```math
\Phi_{\rm loc}(u;\zeta)
=
\sum_{a=0}^{d_\zeta-1}
\zeta^a b_a(u),
```

where `b_a(u)` are chosen basis functions. Examples:

### Ordinary StackSlide Basis

```math
\Phi_{\rm loc}(u;f) = 2\pi f u.
```

Then

```math
\zeta = \{f\}.
```

The segment time `t_k` labels the spectrum but is usually not a searched coordinate within one segment.

### Polynomial Phase Basis

```math
\Phi_{\rm loc}(u;\zeta)
=
2\pi
\left[
f u
+
{1\over 2}\dot f u^2
+
{1\over 6}\ddot f u^3
+
\cdots
\right],
```

with

```math
\zeta = \{f,\dot f,\ddot f,\ldots,f^{(p)}\}.
```

### PN/Chirp-Coefficient Basis

For a PN-inspired chirp, one may instead write

```math
\Phi_{\rm loc}(u;\zeta)
=
\Phi_{0{\rm PN}}(u;f,\beta_0)
+
\Phi_{1{\rm PN}}(u;\beta_1)
+
\Phi_{1.5{\rm PN}}(u;\beta_{3/2})
+
\cdots,
```

with

```math
\zeta =
\{f,\beta_0,\beta_1,\beta_{3/2},\ldots\}.
```

The current project example `zeta = {f,t,beta}` is consistent with this class, where `t` labels segment time and `beta` is a 0PN phase-evolution coordinate.

### Reduced-Basis or Metric-Eigenbasis Coordinates

More generally, choose basis functions `e_a(u)` from an SVD, PCA, greedy reduced basis, or metric eigenbasis:

```math
\Phi_{\rm loc}(u;\zeta)
=
\sum_{a=0}^{d_\zeta-1}
c_a e_a(u).
```

Then

```math
\zeta = \{c_0,c_1,\ldots,c_{d_\zeta-1}\}.
```

This is a promising but unverified coordinate strategy. It needs literature checking against reduced-basis, ROQ, and template-bank papers.

## 3. Mapping Between `theta` and `zeta`

The physical signal parameters predict a segment-local phase. The `zeta` track is the projection of the exact phase onto the chosen local basis:

```math
\zeta_k(\theta)
=
\Pi_{\zeta}
\left[
\phi(t_k+u;\theta)
\right].
```

There are two useful definitions of this projection.

### 3.1 Taylor-Derivative Map

If the basis is polynomial in `u`, then

```math
\phi(t_k+u;\theta)
=
\phi_k(\theta)
+
\sum_{r=1}^{\infty}
{1\over r!}
\phi_k^{(r)}(\theta)u^r,
```

where

```math
\phi_k^{(r)}(\theta)
=
\left.
{d^r\phi(t;\theta)\over dt^r}
\right|_{t=t_k}.
```

The local frequency derivatives are

```math
f_k(\theta)
=
{1\over 2\pi}\phi_k^{(1)}(\theta),
```

```math
\dot f_k(\theta)
=
{1\over 2\pi}\phi_k^{(2)}(\theta),
```

```math
f_k^{(r-1)}(\theta)
=
{1\over 2\pi}\phi_k^{(r)}(\theta).
```

Thus, for polynomial phase order `p`,

```math
\zeta_k(\theta)
=
\left\{
f_k(\theta),
\dot f_k(\theta),
\ldots,
f_k^{(p)}(\theta)
\right\}.
```

### 3.2 Least-Squares Phase Projection

For a general local basis, define the residual phase after projection:

```math
R_k(u;\theta,\zeta)
=
\phi(t_k+u;\theta)
-
\phi_k
-
\Phi_{\rm loc}(u;\zeta).
```

Choose `zeta_k(theta)` to minimize a weighted phase residual:

```math
\zeta_k(\theta)
=
\arg\min_\zeta
\int_{-T_{\rm coh}/2}^{T_{\rm coh}/2}
du\,
q_k(u)
\left[
R_k(u;\theta,\zeta)
-
\overline{R}_k
\right]^2.
```

Here `q_k(u)` is a weight, often related to the segment window, amplitude envelope, or inverse noise. The constant phase offset is removed because power statistics are insensitive to an overall phase.

For a linear basis, this gives normal equations:

```math
\sum_b G_{ab}\zeta_k^b
=
r_a,
```

with

```math
G_{ab}
=
\int du\,q_k(u)b_a(u)b_b(u),
```

```math
r_a
=
\int du\,q_k(u)b_a(u)
\left[
\phi(t_k+u;\theta)-\phi_k
\right].
```

This projection formalism makes clear that `zeta` need not equal a subset of physical parameters. It can be any coordinate system that captures the segment-local phase to sufficient accuracy.

## 4. Coherent Chunk Statistic

Define a windowed coherent demodulated amplitude

```math
X_k(\zeta)
=
\int_{I_k}
dt\,
W_k(t)
x(t)
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

For discrete samples, replace the integral by the appropriate weighted sum.

Let the segment noise variance of this complex amplitude be

```math
\sigma_k^2(\zeta)
=
E_0\left[|X_k(\zeta)|^2\right],
```

where `E_0` denotes expectation under noise only.

Define a normalized coherent power

```math
p_k(\zeta)
=
{|X_k(\zeta)|^2\over \sigma_k^2(\zeta)}.
```

Under ideal complex Gaussian noise,

```math
p_k(\zeta) \sim {\rm Exp}(1)
```

or equivalently

```math
2p_k(\zeta) \sim \chi^2_2.
```

If the statistic uses real-only Fourier powers, one-sided PSDs, overlapping windows, interpolation, NUFFT outputs, or 5-vector combinations, this normalization must be adjusted. The exponential model is the clean baseline, not automatically the final implementation.

### Coherent Likelihood Alternative

A coherent matched-filter statistic would retain the complex phase:

```math
\rho_k(\zeta)
=
{X_k(\zeta)\over \sigma_k(\zeta)}.
```

The power statistic discards the phase between chunks. This is robust but suboptimal when the inter-segment phase model is accurate. A later extension could use loosely coherent or complex-amplitude-preserving combinations. Citation needed for loosely coherent methods.

## 5. Semicoherent Statistic

Given a physical template `theta`, compute its predicted track through segment-local spectral space:

```math
\zeta_k(\theta),\qquad k=1,\ldots,N_{\rm seg}.
```

The generic semicoherent power statistic is

```math
\mathcal{S}(\theta)
=
\sum_{k=1}^{N_{\rm seg}}
w_k(\theta)
p_k\!\left(\zeta_k(\theta)\right).
```

The weights satisfy

```math
w_k(\theta) \ge 0.
```

A convenient normalization is

```math
\sum_k w_k = 1,
```

but for distributional calculations it is sometimes cleaner to keep arbitrary weights.

If interpolation is needed between `zeta` grid points,

```math
p_k[\zeta_k(\theta)]
\approx
\sum_j
a_{kj}(\theta)
p_k(\zeta_{kj}),
```

with interpolation weights `a_kj`. This creates correlations and changes the exact null distribution.

## 6. Weights

### 6.1 Equal Weights

The simplest choice is

```math
w_k = 1.
```

This is adequate only if all chunks have equal noise, equal antenna response, equal expected signal power, and equal data quality.

### 6.2 Inverse-Variance Weights

If

```math
p_k = 1 + \lambda_k + \epsilon_k,
```

where `lambda_k` is the expected signal contribution and `Var_0(p_k)=v_k`, then for estimating a known amplitude scale the optimal linear weights are proportional to

```math
w_k \propto {\lambda_k\over v_k}.
```

For exponential normalized powers under noise,

```math
v_k = 1.
```

Then

```math
w_k \propto \lambda_k.
```

### 6.3 Antenna and Noise Weights

For a signal with segment response amplitude `A_k(theta)` and noise PSD `S_{n,k}`, the coherent noncentrality usually scales like

```math
\lambda_k(\theta)
\propto
{A_k^2(\theta) T_{\rm coh}\over S_{n,k}}.
```

Therefore a natural power-sum weight is

```math
w_k(\theta)
\propto
{A_k^2(\theta)\over S_{n,k}}.
```

For a 5-vector or polarization-resolved statistic, the weight should be replaced by the appropriate covariance-weighted quadratic form:

```math
\widehat a
=
{A^\dagger C^{-1}X\over A^\dagger C^{-1}A},
```

or

```math
\Lambda
=
X^\dagger C^{-1}P_A X,
```

where `X` is the measured multi-bin vector, `A` is the template response vector, `C` is the noise covariance, and `P_A` projects onto the modeled signal subspace.

This is an extension of the scalar formalism and is more appropriate for the existing 5-vector pipeline.

## 7. Null Distribution

Assume independent segments and normalized powers

```math
p_k \sim {\rm Exp}(1)
```

under noise only.

### 7.1 Unweighted Sum

For

```math
\mathcal{S} = \sum_{k=1}^{N_{\rm seg}} p_k,
```

the null distribution is

```math
\mathcal{S} \sim \Gamma(N_{\rm seg},1),
```

or

```math
2\mathcal{S} \sim \chi^2_{2N_{\rm seg}}.
```

Thus

```math
E_0[\mathcal{S}] = N_{\rm seg},
```

```math
{\rm Var}_0[\mathcal{S}] = N_{\rm seg}.
```

### 7.2 Weighted Sum

For

```math
\mathcal{S} = \sum_k w_k p_k,
```

the mean and variance are

```math
E_0[\mathcal{S}] = \sum_k w_k,
```

```math
{\rm Var}_0[\mathcal{S}] = \sum_k w_k^2.
```

The exact distribution is a hypoexponential distribution if all weights are distinct. For many segments, a Gaussian approximation gives

```math
{\mathcal{S}-\sum_k w_k
\over
\sqrt{\sum_k w_k^2}}
\approx
\mathcal{N}(0,1).
```

For better tail accuracy, use a scaled chi-square approximation by matching the first two moments:

```math
\mathcal{S}
\approx
a\,\chi^2_\nu,
```

with

```math
a =
{ {\rm Var}_0(\mathcal{S}) \over 2E_0(\mathcal{S}) },
```

```math
\nu =
{2[E_0(\mathcal{S})]^2 \over {\rm Var}_0(\mathcal{S}) }.
```

### 7.3 Trials Factor

If `N_eff` effectively independent tracks are searched, the single-template false-alarm probability should satisfy approximately

```math
p_{\rm FA}^{\rm single}
\approx
{p_{\rm FA}^{\rm total}\over N_{\rm eff}}
```

for small probabilities.

Equivalently, the threshold `S_*` solves

```math
P_0(\mathcal{S}>S_*)
\approx
{p_{\rm FA}^{\rm total}\over N_{\rm eff}}.
```

The effective number of trials is not necessarily the raw number of templates because neighboring tracks are correlated. Estimating `N_eff(d_\zeta,T_coh)` is required for calibrated sensitivity.

## 8. Signal Expectation

Under a signal matching the template track, the coherent amplitude has nonzero mean:

```math
X_k[\zeta_k(\theta)]
=
s_k + \eta_k,
```

where

```math
E[\eta_k]=0,
```

and

```math
\lambda_k =
{|s_k|^2\over \sigma_k^2}
```

is the coherent noncentrality parameter.

Then

```math
E[p_k] = 1+\lambda_k
```

for the normalized exponential convention.

The semicoherent expectation is

```math
E_\theta[\mathcal{S}(\theta)]
=
\sum_k w_k(1+\lambda_k).
```

The excess above noise is

```math
\Delta \mathcal{S}
=
\sum_k w_k\lambda_k.
```

If the signal is mismatched because the chosen local `zeta` model does not capture the exact phase, then

```math
\lambda_k
=
\lambda_{k,{\rm opt}}(1-\mu_k),
```

where `mu_k` is the coherent mismatch in segment `k`.

Thus

```math
E_\theta[\mathcal{S}]
=
\sum_k w_k
+
\sum_k w_k\lambda_{k,{\rm opt}}(1-\mu_k).
```

For weak signals and many segments, an approximate semicoherent detection SNR is

```math
{\rm SNR}_{\rm semi}
\approx
{
\sum_k w_k\lambda_k
\over
\sqrt{\sum_k w_k^2}
}.
```

If equal weights and equal segment noncentralities are used,

```math
{\rm SNR}_{\rm semi}
\approx
\sqrt{N_{\rm seg}}\lambda.
```

Since single-segment coherent noncentrality usually scales as

```math
\lambda \propto h_0^2 T_{\rm coh}/S_n,
```

the semicoherent power-sum sensitivity scales roughly as

```math
h_{0,\min}
\propto
(S_n/T_{\rm coh})^{1/2}
N_{\rm seg}^{-1/4},
```

at fixed threshold and mismatch. This is the usual semicoherent amplitude scaling. Citation needed for a standard StackSlide sensitivity reference.

Using `N_seg = T_obs/T_coh`, this becomes

```math
h_{0,\min}
\propto
S_n^{1/2}
T_{\rm coh}^{-1/4}
T_{\rm obs}^{-1/4},
```

again ignoring trials factors and weights.

## 9. Residual Phase and Mismatch Criterion

Let the exact segment phase be decomposed as

```math
\phi(t_k+u;\theta)
=
\phi_k
+
\Phi_{\rm loc}(u;\zeta_k)
+
R_k(u;\theta).
```

Here `R_k` is the residual phase not represented by the chosen `zeta`.

The coherent matched amplitude is reduced by

```math
\mathcal{A}_k
=
{
\int du\,W_k(u)\,A_k(u)\,
e^{iR_k(u)}
\over
\int du\,W_k(u)\,A_k(u)
}.
```

The coherent mismatch is

```math
\mu_k
=
1-|\mathcal{A}_k|^2.
```

For small residual phase, after subtracting an irrelevant weighted mean residual phase,

```math
\delta R_k(u)
=
R_k(u)-\langle R_k\rangle,
```

the mismatch is approximately

```math
\mu_k
\approx
\left\langle \delta R_k^2 \right\rangle
-
\left\langle \delta R_k \right\rangle^2.
```

With the mean already removed,

```math
\mu_k
\approx
{\rm Var}_k[R_k].
```

Depending on normalization conventions, a factor of order unity may enter; this must be checked against the exact inner-product definition. The robust criterion is:

```math
{\rm rms}(R_k-\langle R_k\rangle)
\lesssim
\sqrt{\mu_{\rm max}}.
```

### Polynomial Truncation Estimate

Suppose the local basis keeps phase derivatives through order `p+1`, so the first omitted term is

```math
R_k(u)
\approx
{1\over (p+2)!}
\phi_k^{(p+2)}
u^{p+2}.
```

For a rectangular window, the residual variance scales as

```math
\mu_k
\sim
\left[
\phi_k^{(p+2)}
T_{\rm coh}^{p+2}
\right]^2
\times C_p,
```

where `C_p` is a dimensionless constant depending on the window and whether lower-order projections are refit.

Thus the maximum coherent time scales roughly as

```math
T_{\rm coh,max}
\propto
\left(
{\mu_{\rm max}^{1/2}\over |\phi^{(p+2)}|}
\right)^{1/(p+2)}.
```

This scaling is derived from Taylor truncation and is useful for comparing candidate `zeta` dimensions.

## 10. Metric View

For a small offset in physical parameters,

```math
\theta \rightarrow \theta+\Delta\theta,
```

the phase difference in segment `k` is

```math
\Delta\phi_k(u)
=
\partial_i\phi_k(u)\Delta\theta^i
+
O(\Delta\theta^2).
```

The coherent metric is approximately

```math
g_{ij}^{(k)}
=
\left\langle
\partial_i\phi\,\partial_j\phi
\right\rangle_k
-
\left\langle
\partial_i\phi
\right\rangle_k
\left\langle
\partial_j\phi
\right\rangle_k.
```

Then

```math
\mu_k
\approx
g_{ij}^{(k)}
\Delta\theta^i\Delta\theta^j.
```

For the `zeta` formalism, the local model projection splits the phase space into retained and residual directions.

Let `P_d` project onto the span of the retained `d_zeta` basis functions. Then

```math
\Delta\phi
=
P_d\Delta\phi
+
(1-P_d)\Delta\phi.
```

The retained component is controlled by gridding in `zeta`; the residual component is intrinsic model mismatch:

```math
\mu_{\rm total}
\approx
\mu_{\zeta{\rm -grid}}
+
\mu_{\rm residual}
+
\mu_{\rm semi-track}.
```

This decomposition is a derived modeling framework, not yet established as a validated theorem for the current pipeline.

The semicoherent metric for the summed statistic is approximately a weighted average:

```math
g_{ij}^{\rm semi}
\approx
{
\sum_k w_k \lambda_k g_{ij}^{(k)}
\over
\sum_k w_k\lambda_k
},
```

when mismatches are small and the statistic is dominated by the expected signal contribution.

Citation needed: semicoherent metric literature, StackSlide metric, Weave.

## 11. Computational Cost Model

The total cost is decomposed as

```math
C_{\rm total}(d_\zeta,T_{\rm coh})
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
N_{\rm seg} = {T_{\rm obs}\over T_{\rm coh}}.
```

Let the grid size in each retained `zeta` coordinate be `N_a`, so the number of coherent spectral grid points per segment is

```math
N_\zeta = \prod_{a=0}^{d_\zeta-1} N_a.
```

### 11.1 Spectrum Construction Cost

A generic scaling is

```math
C_{\rm spectra}
\sim
N_{\rm seg}\,
N_\zeta\,
c_{\rm coh}(T_{\rm coh},d_\zeta).
```

The per-grid coherent cost `c_coh` depends on implementation:

- FFT for `d_zeta=1` frequency spectra.
- NUFFT or chirp transform for nonuniform/local chirp coordinates.
- Brute-force demodulation if no fast transform exists.
- Reuse of intermediate transforms if `zeta` coordinates have structure.

For brute-force demodulation over `N_samp = f_s T_coh` samples,

```math
c_{\rm coh} \sim N_{\rm samp}.
```

For FFT-like transforms,

```math
c_{\rm coh} \sim \log N_{\rm samp}
```

per output bin after accounting for total transform cost. This distinction is decisive.

### 11.2 Track Accumulation Cost

Let `N_theta` be the number of physical semicoherent templates. Then

```math
C_{\rm tracks}
\sim
N_\theta
N_{\rm seg}
c_{\rm lookup},
```

where `c_lookup` includes evaluating `zeta_k(theta)`, accessing the spectrum, and adding to the statistic.

If interpolation over `m_interp` neighboring grid points is used,

```math
c_{\rm lookup} \sim m_{\rm interp}.
```

For multilinear interpolation in `d_zeta` dimensions,

```math
m_{\rm interp}=2^{d_\zeta}.
```

This exponential interpolation factor can become important.

### 11.3 Memory Cost

Storing all spectra costs

```math
M_{\rm spectra}
\sim
N_{\rm seg}N_\zeta B,
```

where `B` is bytes per stored power or complex value.

If spectra are streamed segment by segment, memory can be reduced to

```math
M_{\rm stream}
\sim
N_\zeta B
+
N_\theta B_{\rm accum},
```

but then track accumulation must be organized so that each segment updates all relevant templates.

A likely practical bottleneck is memory bandwidth, not floating-point arithmetic:

```math
C_{\rm memory}
\sim
{N_{\rm reads}B\over {\rm bandwidth}}.
```

This is a conjecture for the current pipeline and must be benchmarked.

### 11.4 Threshold and Trials Cost

The threshold increases with effective template count:

```math
S_*
=
F_0^{-1}
\left(
1-{p_{\rm FA}^{\rm total}\over N_{\rm eff}}
\right),
```

where `F_0` is the single-template null CDF.

Since `N_eff` generally grows with both `N_theta` and `N_\zeta`, higher-dimensional `zeta` can reduce sensitivity by increasing the detection threshold even if mismatch improves.

This effect belongs in the optimization objective, not just in final calibration.

## 12. Sensitivity Versus Cost Tradeoff

A useful optimization problem is:

```math
\min_{d_\zeta,T_{\rm coh},\Delta\zeta,\Delta\theta}
C_{\rm total}
```

subject to

```math
P_{\rm det}(h_0;\ d_\zeta,T_{\rm coh}) \ge P_{\rm det}^{\rm target},
```

```math
P_{\rm FA}^{\rm total} \le P_{\rm FA}^{\rm target},
```

```math
\mu_{\rm residual}
+
\mu_{\zeta{\rm -grid}}
+
\mu_{\rm semi-track}
\le
\mu_{\rm max}.
```

Equivalently, at fixed compute budget,

```math
\max_{d_\zeta,T_{\rm coh}}
P_{\rm det}
```

or minimize detectable amplitude:

```math
\min_{d_\zeta,T_{\rm coh}}
h_{0,\min}.
```

### Benefit of Increasing `d_zeta`

Adding one coordinate is worthwhile if it allows either:

1. Larger `T_coh`, increasing single-segment coherent SNR.
2. Lower residual mismatch at fixed `T_coh`.
3. Fewer physical templates because the track family is simpler in the new coordinates.
4. More accurate weights or covariance modeling.

### Cost of Increasing `d_zeta`

Adding one coordinate is harmful if it causes:

1. Larger `N_\zeta`.
2. More expensive coherent transforms.
3. Higher interpolation cost.
4. Larger memory footprint.
5. Larger effective trials factor.
6. More complicated track mapping.
7. Strong correlations that require empirical threshold calibration.

### Marginal Coordinate Criterion

Let `d` denote the current `zeta` dimension and `d+1` a candidate extension. The extension is favored if

```math
h_{0,\min}(d+1,T_{\rm coh}^{d+1})
<
h_{0,\min}(d,T_{\rm coh}^{d})
```

at equal total compute and false-alarm probability.

A rough analytic decision rule is:

```math
{\rm sensitivity\ gain}
>
{\rm threshold\ penalty}
+
{\rm cost\ penalty}.
```

More explicitly, require

```math
{
[1-\mu(d+1)]T_{\rm coh}^{d+1}
\over
[1-\mu(d)]T_{\rm coh}^{d}
}
```

to be large enough to compensate for the increased threshold from `N_eff` and any reduction in the number of segments.

This is a heuristic, not a final theorem.

## 13. Choosing `T_coh`

For fixed `d_zeta`, `T_coh` is limited by residual phase mismatch:

```math
\mu_{\rm residual}(d_\zeta,T_{\rm coh}) \le \mu_{\rm residual,max}.
```

It is also limited by cost:

```math
C_{\rm spectra}(d_\zeta,T_{\rm coh})
+
C_{\rm tracks}(d_\zeta,T_{\rm coh})
\le
C_{\rm budget}.
```

Increasing `T_coh` has competing effects:

- `N_seg` decreases as `1/T_coh`.
- Per-segment coherent SNR increases as `T_coh`.
- Semicoherent amplitude sensitivity improves roughly as `T_coh^{-1/4}` at fixed `T_obs`, ignoring trials.
- Frequency resolution improves as `1/T_coh`, increasing frequency grid count over a fixed band.
- Higher phase derivatives become resolvable, increasing the need for extra `zeta` coordinates.
- Long segments are more vulnerable to gaps, nonstationary noise, and detector modulation variation.

For polynomial residual phase with first omitted derivative order `p+2`,

```math
\mu_{\rm residual}
\sim
\left[
\phi^{(p+2)}T_{\rm coh}^{p+2}
\right]^2.
```

Thus the mismatch-limited coherent time is

```math
T_{\rm coh,max}(p)
\sim
\left(
{\mu_{\rm max}^{1/2}
\over
|\phi^{(p+2)}|}
\right)^{1/(p+2)}.
```

This gives a concrete way to compare coordinate sets:

- `zeta={f}`: limited by unmodeled chirp rate.
- `zeta={f,\dot f}` or `{f,beta_0}`: limited by unmodeled second chirp derivative or next PN correction.
- `zeta={f,\dot f,\ddot f}` or `{f,beta_0,beta_1}`: limited by the next omitted term.

## 14. Limiting Case: Ordinary StackSlide

Ordinary StackSlide is recovered by choosing

```math
\zeta_k = f_k
```

and

```math
\Phi_{\rm loc}(u;f_k)=2\pi f_k u.
```

The coherent statistic is just the normalized power in a short Fourier transform bin:

```math
p_k(f)
=
{|\tilde x_k(f)|^2\over \sigma_k^2(f)}.
```

The physical template predicts a frequency track:

```math
f_k(\theta)
=
{1\over 2\pi}
\left.
{d\phi(t;\theta)\over dt}
\right|_{t=t_k}.
```

The semicoherent statistic is

```math
\mathcal{S}_{\rm SS}(\theta)
=
\sum_k w_k p_k[f_k(\theta)].
```

The residual phase is

```math
R_k(u)
=
\phi(t_k+u;\theta)
-
\phi_k
-
2\pi f_k u.
```

If the signal frequency changes significantly over the segment, the leading residual term is

```math
R_k(u)
\approx
\pi \dot f_k u^2.
```

Therefore the ordinary StackSlide coherent time is limited by

```math
|\dot f_k|T_{\rm coh}^2 \lesssim O(1)
```

up to the chosen mismatch constant.

Adding a chirp coordinate such as `beta` or `dot f` generalizes StackSlide by replacing each Fourier spectrum with a locally dechirped spectrum and replacing the frequency track with a track through higher-dimensional `zeta` space.

## 15. Relation to the Existing PBH/NUFFT Direction

The project premise says the nearby implementation uses

```math
\zeta = \{f,t,\beta\},
```

where `beta` is a 0PN parameter.

In this language, each coherent chunk is not merely Fourier transformed at constant frequency. Instead, it is demodulated or resampled according to a 0PN chirp model controlled by `beta`, so signals with that local chirp behavior become approximately monochromatic in the resampled coordinate.

The residual phase is then not the full chirp curvature, but the difference between the true phase model and the 0PN local model:

```math
R_k(u)
=
\phi_{\rm true}(t_k+u;\theta)
-
\phi_{0{\rm PN}}(u;f_k,\beta_k).
```

The next theory task is to compute or numerically estimate

```math
\mu_{\rm residual}^{0{\rm PN}}(T_{\rm coh},\theta),
```

then compare it to

```math
\mu_{\rm residual}^{1{\rm PN}}(T_{\rm coh},\theta),
```

and higher-order variants.

This will reveal whether the added PN coordinate buys enough coherent time to offset its grid and transform cost.

## 16. Practical Optimization Workflow

A recommended workflow for the next agents is:

1. Choose candidate coordinate families:
   ```math
   \zeta_0=\{f\},
   ```
   ```math
   \zeta_1=\{f,\beta_0\},
   ```
   ```math
   \zeta_2=\{f,\beta_0,\beta_1\},
   ```
   or polynomial equivalents.

2. For each candidate and `T_coh`, compute residual phase mismatch:
   ```math
   \mu_{\rm residual}(d_\zeta,T_{\rm coh},\theta).
   ```

3. Determine grid spacing in `zeta` for a target coherent-grid mismatch:
   ```math
   \mu_{\zeta{\rm -grid}}\le\mu_{\zeta,\max}.
   ```

4. Determine semicoherent physical-template spacing:
   ```math
   \mu_{\rm semi-track}\le\mu_{\rm semi,\max}.
   ```

5. Estimate:
   ```math
   N_\zeta,\quad N_\theta,\quad N_{\rm eff}.
   ```

6. Estimate cost:
   ```math
   C_{\rm spectra},\quad C_{\rm tracks},\quad C_{\rm memory}.
   ```

7. Estimate sensitivity:
   ```math
   h_{0,\min}(d_\zeta,T_{\rm coh})
   ```

   including mismatch and threshold penalties.

8. Select the Pareto frontier in sensitivity versus compute.

9. Validate with toy polynomial-phase simulations before full PN simulations.

## 17. Required Validation Simulations

### 17.1 Toy Polynomial-Phase Study

Use signals

```math
\phi(t)
=
2\pi
\left[
f_0 t
+
{1\over 2}\dot f t^2
+
{1\over 6}\ddot f t^3
+
\cdots
\right].
```

Compare:

```math
\zeta=\{f\},
```

```math
\zeta=\{f,\dot f\},
```

```math
\zeta=\{f,\dot f,\ddot f\}.
```

Measure:

- residual coherent mismatch versus `T_coh`;
- detection statistic under signal;
- null distribution under noise;
- runtime and memory;
- empirical trials factor.

### 17.2 PN Chirp Study

Use the relevant 3.5PN phase model. Candidate local models:

- frequency only;
- 0PN;
- 0PN + 1PN;
- 0PN + 1PN + 1.5PN;
- reduced-basis phase coordinates.

Measure the same quantities and identify the best coordinate family at fixed compute.

### 17.3 Weighting and 5-Vector Study

Replace scalar powers with 5-vector statistics. Estimate covariance matrices for extracted bins and compare:

```math
\sum_k p_k
```

against

```math
\sum_k X_k^\dagger C_k^{-1}P_{A,k}X_k.
```

This is required before claiming optimal sensitivity for the detector-modulated pipeline.

## 18. Risks and Caveats

- The exponential null distribution is only exact for independent normalized complex Gaussian powers. Real implementation details can break this.
- NUFFT or resampled spectra may have nontrivial noise covariance across `zeta` bins.
- Interpolation creates correlations and modifies thresholds.
- The number of raw templates may badly overestimate or underestimate the true trials factor.
- The residual phase metric may fail for long chunks or large parameter offsets.
- Power summing discards inter-segment phase information and may be substantially suboptimal for well-modeled signals.
- Higher-dimensional spectra may become memory-limited before arithmetic-limited.
- The idea may overlap with fast chirp transform or polynomial-phase-transform literature. Citation and novelty audit needed.

## 19. Search Targets and Citation Gaps

The following citations or literature checks are needed before this artifact can be treated as a report-ready theory section:

- StackSlide sensitivity and cost scaling: citation needed.
- Semicoherent metric derivation: citation needed.
- Weave metric-bank implementation: citation needed.
- PowerFlux weighting and statistic distribution: citation needed.
- Fast chirp transform and polynomial-phase transform relation to `zeta`: citation needed.
- High-order ambiguity function and chirplet-transform model-order selection: citation needed.
- Reduced-basis/SVD phase-coordinate template-bank methods: citation needed.
- Loose coherence or partially coherent statistics: citation needed.

Specific search targets from the literature artifact remain relevant:

- `"fast chirp transform" "polynomial phase" complexity`
- `"polynomial phase signal" "model order selection" detection`
- `"high order ambiguity function" "polynomial phase" "detection"`
- `"semicoherent metric" "StackSlide" "coordinate"`
- `"Weave" "metric" "template bank" "StackSlide"`
- `"SVD phase" "template bank" "gravitational waves"`
- `"chirp time coordinates" "metric" "template bank"`

## 20. Recommended Next Steps

1. Inspect the local implementation using `zeta={f,t,beta}` and identify the exact phase model, normalization, and current statistic.
2. Compute the residual phase mismatch for frequency-only, 0PN, and higher-PN local models.
3. Build a small cost model using measured runtime for spectrum construction and track accumulation.
4. Calibrate the null distribution empirically for the current NUFFT/resampling statistic.
5. Compare analytic `T_coh,max` predictions against injection recovery.
6. Decide whether the next formalism should be scalar-power-based or directly 5-vector/covariance-based.