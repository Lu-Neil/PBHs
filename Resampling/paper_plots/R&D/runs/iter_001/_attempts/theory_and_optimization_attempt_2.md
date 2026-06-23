# Theory and Optimization: Choosing `zeta` and `T_coh`

## Audit Metadata

- Source of truth: `idea.md`, attempt-2 human feedback, planner artifact, and literature artifact.
- Agent: `theory_and_optimization`.
- Attempt: 2.
- Required artifact: `runs/iter_001/02_theory_and_optimization.md`.
- Status: mathematical formalism and optimization scaffold. This is not a numerical result.
- Citation policy: no new citations are introduced here. Prior-art claims should be audited through `01_literature_review.md`.
- Main distinction: `theta` denotes physical signal parameters; `zeta` denotes chunk-local spectral coordinates used to build reusable coherent spectra.

## Executive Summary

The key decision is not whether adding coherent-chunk coordinates improves recovered power. It usually does. The key decision is whether the recovered-power gain improves detectable amplitude after paying for extra grid points, memory, interpolation, track summation, and a larger false-alarm threshold.

The operational optimization is:

```math
(d_\zeta^\star,T_{\rm coh}^\star,\mathcal B^\star)
=
\arg\min_{d,T,\mathcal B}
h_{\min}(d,T,\mathcal B)
```

subject to

```math
C_{\rm total}(d,T,\mathcal B) \le C_{\rm budget},
\qquad
M_{\rm total}(d,T,\mathcal B) \le M_{\rm budget},
\qquad
P_{\rm FA,total} \le P_{\rm FA}^{\rm target}.
```

Here:

- `d_zeta` is the dimension of the coherent-chunk spectrum.
- `T_coh` is the coherent chunk duration.
- `mathcal B` is the chosen basis or coordinate family for `zeta`.

The practical rule is:

> Add a `zeta` coordinate only when its mismatch reduction lowers the final detectable amplitude at fixed compute, memory, and false-alarm probability.

Ordinary StackSlide is recovered as the limiting case `zeta = {f}`, with time represented by the segment index rather than an additional searched coordinate.

## 1. Signal Parameters `theta`

The measured data are modeled as

```math
x(t)=n(t)+h(t;\theta),
```

where `n(t)` is noise and `theta` are physical signal parameters.

For a phase-dominated signal, use the analytic representation

```math
h_a(t;\theta)
=
\mathcal A(t;\theta)
\exp[i\phi(t;\theta)].
```

It is useful to split

```math
\theta = (\lambda,\alpha),
```

where:

- `lambda` are intrinsic phase-evolution parameters;
- `alpha` are amplitude, polarization, sky response, and other extrinsic parameters.

For PBH-inspired or chirping signals, examples include initial frequency, chirp mass, a 0PN-like chirp parameter `beta`, coalescence time, and higher-PN or spin parameters.

For the requested 3.5PN case studies:

```math
\theta_{\rm NS}
=
(M \ \text{or} \ M_c,\ \eta=1/4,\ \chi_1=\chi_2=0)
```

for equal-mass non-spinning binaries, and a minimal equal-aligned-spin model is

```math
\theta_{\rm spin}
=
(M \ \text{or} \ M_c,\ \eta=1/4,\ \chi_1=\chi_2=\chi).
```

The spin range must be fixed before quantitative work. The phrase `chi <= 0.2` is ambiguous between:

```math
0 \le \chi \le 0.2
```

and

```math
-0.2 \le \chi \le 0.2.
```

This artifact treats spin conclusions as conditional until that convention is specified.

## 2. Spectral Coordinates `zeta`

The coherent analysis in each chunk uses local spectral coordinates

```math
\zeta_k.
```

These are not necessarily physical parameters. They parameterize a local demodulation phase

```math
\Phi_{\rm loc}(u;\zeta_k),
\qquad
u=t-t_k,
```

where `t_k` is the center of chunk `k`.

Examples:

### Frequency Only

```math
\zeta = \{f\},
\qquad
\Phi_{\rm loc}(u;f)=2\pi f u.
```

This is the ordinary time-frequency / StackSlide-like case.

### Polynomial Phase

```math
\zeta = \{f,\dot f,\ddot f,\ldots\},
```

with

```math
\Phi_{\rm loc}(u;\zeta)
=
2\pi
\left[
fu
+
\frac{1}{2}\dot f u^2
+
\frac{1}{6}\ddot f u^3
+\cdots
\right].
```

### PBH / Chirp Coordinates

A chirp-aware local model may use

```math
\zeta=\{f,\beta\}
```

or, if segment epoch is explicitly gridded,

```math
\zeta=\{f,t,\beta\}.
```

In most semicoherent searches, however, `t_k` is a segment label, not a searched spectral coordinate.

### Basis Coordinates

A general linear local basis is

```math
\Phi_{\rm loc}(u;\zeta)
=
\sum_{a=1}^{d_\zeta} \zeta^a e_a(u),
```

where `e_a(u)` may be PN functions, chirp-time functions, Taylor monomials, metric eigenvectors, or SVD/PCA basis vectors.

Important distinction:

```math
\dim(\theta) \ne \dim(\zeta)
```

in general. A one-dimensional physical family can require multiple local phase directions over long chunks, while a higher-dimensional physical family can have only a few dominant local directions over a restricted range.

## 3. Mapping Between `theta` and `zeta`

A physical template predicts a track through chunk-local spectral coordinates:

```math
\theta \mapsto \zeta_k(\theta).
```

A robust definition is local projection. For chunk `k`, choose `zeta_k` and an arbitrary phase offset by

```math
(\phi_{0,k},\zeta_k)
=
\arg\min_{\phi_0,\zeta}
\left\|
\phi(t_k+u;\theta)
-
\phi_0
-
\Phi_{\rm loc}(u;\zeta)
\right\|_k^2.
```

Define the weighted inner product

```math
\langle a,b\rangle_k
=
\int_{-T_{\rm coh}/2}^{T_{\rm coh}/2}
du\,
q_k(u)a(u)b(u),
```

where `q_k(u)` includes windowing, amplitude weighting, and optional inverse-noise weighting.

If

```math
\Phi_{\rm loc}(u;\zeta)=\sum_a \zeta^a e_a(u),
```

then the projection is a linear least-squares problem after removing constant phase. Let

```math
\tilde y
=
y
-
\frac{\langle y,1\rangle_k}{\langle 1,1\rangle_k}.
```

Then

```math
G_{ab}^{(k)}\zeta_k^b=b_a^{(k)},
```

with

```math
G_{ab}^{(k)}
=
\langle \tilde e_a,\tilde e_b\rangle_k,
\qquad
b_a^{(k)}
=
\langle \tilde e_a,\tilde \phi_k\rangle_k.
```

Thus

```math
\zeta_k^a(\theta)
=
(G^{-1})^{ab}b_b.
```

This is a derived least-squares result.

For Taylor coordinates one may instead define

```math
f_k(\theta)
=
\frac{1}{2\pi}
\left.
\frac{d\phi}{dt}
\right|_{t_k},
\qquad
\dot f_k(\theta)
=
\frac{1}{2\pi}
\left.
\frac{d^2\phi}{dt^2}
\right|_{t_k},
```

and similarly for higher derivatives. The derivative mapping and projection mapping agree only when the retained Taylor expansion is accurate over the chunk.

## 4. Coherent Chunk Statistic

Split the data into chunks

```math
I_k=
[t_k-T_{\rm coh}/2,\ t_k+T_{\rm coh}/2].
```

Define the coherent demodulated output

```math
X_k(\zeta)
=
\int_{I_k}
dt\,
W_k(t)
x_a(t)
\exp[-i\Phi_{\rm loc}(t-t_k;\zeta)].
```

A normalized scalar power is

```math
P_k(\zeta)
=
\frac{|X_k(\zeta)|^2}{\sigma_k^2(\zeta)},
```

where

```math
\sigma_k^2(\zeta)=E_0[|X_k(\zeta)|^2].
```

For colored, gapped, resampled, or NUFFT-transformed data, `sigma_k^2` must be measured or predicted. In the local project this is especially important because the NUFFT/resampling operation changes the effective noise PSD.

### 5-Vector Generalization

For the sidereal 5-vector pipeline, the chunk output may be a vector

```math
\mathbf X_k(\zeta),
```

with covariance

```math
C_k(\zeta)=E_0[\mathbf X_k\mathbf X_k^\dagger].
```

For a template response vector `A_k`, a covariance-weighted amplitude estimator is

```math
\widehat a_k
=
\frac{
\mathbf A_k^\dagger C_k^{-1}\mathbf X_k
}{
\mathbf A_k^\dagger C_k^{-1}\mathbf A_k
}.
```

A corresponding detection power is proportional to

```math
|\widehat a_k|^2
\,
\mathbf A_k^\dagger C_k^{-1}\mathbf A_k.
```

This is a recommended extension, not yet validated for the local code.

## 5. Semicoherent Statistic

A template `theta` defines a track `zeta_k(theta)`. The semicoherent statistic is

```math
\mathcal S(\theta)
=
\sum_{k=1}^{N_{\rm seg}}
w_k(\theta)
P_k[\zeta_k(\theta)].
```

For non-overlapping chunks,

```math
N_{\rm seg}=T_{\rm obs}/T_{\rm coh}.
```

If the predicted coordinate is off-grid, use an interpolation operator:

```math
P_k[\zeta_k(\theta)]
\rightarrow
\mathcal I[P_k](\zeta_k(\theta)).
```

Interpolation error contributes to the total mismatch.

## 6. Weights

Equal weights,

```math
w_k=1,
```

are appropriate only when all chunks have equal noise, equal detector response, equal mismatch, and equal normalization.

In the weak-signal scalar model, write the per-chunk noncentrality as

```math
\lambda_k
=
h_0^2 \kappa_k,
```

where schematically

```math
\kappa_k
\propto
T_{\rm coh}
\frac{A_k^2}{S_{n,k}^{\rm eff}}
(1-\mu_k).
```

For normalized exponential powers, the null variance of each `P_k` is 1. The weighted statistic has signal mean shift

```math
E_1[\mathcal S]-E_0[\mathcal S]
=
h_0^2\sum_k w_k\kappa_k
```

and null variance

```math
\mathrm{Var}_0(\mathcal S)=\sum_k w_k^2.
```

The weak-signal detection SNR of the power sum is therefore

```math
\frac{
h_0^2\sum_k w_k\kappa_k
}{
\sqrt{\sum_k w_k^2}
}.
```

Maximizing this over weights gives

```math
w_k \propto \kappa_k.
```

Thus, in the scalar idealization, chunks should be weighted by their expected signal contribution divided by their noise variance. For unknown amplitude and polarization, this should be replaced by marginalization, maximization, or a covariance-weighted vector statistic.

## 7. Null Distribution

Assume the ideal scalar case:

1. Gaussian stationary noise.
2. Correct normalization.
3. Independent chunks.
4. Independent selected spectral bins.
5. Complex coherent outputs.

Then

```math
X_k/\sigma_k \sim \mathcal{CN}(0,1)
```

and

```math
P_k\sim \mathrm{Exp}(1).
```

For equal weights,

```math
\mathcal S=\sum_k P_k
\sim
\Gamma(N_{\rm seg},1).
```

Thus

```math
E_0[\mathcal S]=N_{\rm seg},
\qquad
\mathrm{Var}_0[\mathcal S]=N_{\rm seg}.
```

For unequal positive weights,

```math
E_0[\mathcal S]=\sum_k w_k,
\qquad
\mathrm{Var}_0[\mathcal S]=\sum_k w_k^2.
```

The exact distribution is a weighted sum of exponentials. For many chunks, a Gaussian approximation is

```math
\mathcal S
\approx
\mathcal N
\left(
\sum_k w_k,\,
\sum_k w_k^2
\right).
```

This ideal null model is an analytical baseline. It is not automatically valid for NUFFT spectra, gapped data, overlapping windows, interpolated bins, 5-vector sidebands, or real detector noise.

### Trials Factor

If the search evaluates `N_eff` effectively independent templates, then

```math
P_{\rm FA,total}
=
1-(1-p_{\rm single})^{N_{\rm eff}}
\approx
N_{\rm eff}p_{\rm single}
```

for small `p_single`.

The threshold therefore depends on `d_zeta` through the number of spectral bins, number of tracks, interpolation locations, and correlations between templates. A higher-dimensional `zeta` grid can increase sensitivity per template while also increasing the threshold. Both effects must be included.

## 8. Signal Expectation

At the correct template, the coherent output has nonzero mean

```math
E[X_k(\zeta_k(\theta))]=H_k(\theta).
```

Define

```math
\lambda_k(\theta)
=
\frac{|H_k(\theta)|^2}{\sigma_k^2}.
```

Then

```math
E_1[P_k]=1+\lambda_k
```

in the ideal scalar model, and

```math
E_1[\mathcal S]-E_0[\mathcal S]
=
\sum_k w_k\lambda_k.
```

Mismatch enters through the coherent overlap. Write

```math
\lambda_k
=
\lambda_{k,\rm ideal}\,\eta_k,
```

where

```math
\eta_k=|\mathcal O_k|^2
```

and

```math
\mu_k=1-\eta_k.
```

For small mismatch,

```math
\lambda_k
\approx
\lambda_{k,\rm ideal}(1-\mu_k).
```

A useful weak-signal threshold estimate is

```math
h_{\min}^2
\approx
\frac{
\mathcal S_{\rm th}-E_0[\mathcal S]
}{
\sum_k w_k\kappa_k
},
```

where `S_th` includes the desired total false-alarm probability and trials factor.

For identical chunks, equal weights, fixed trials factor, and small mismatch, this gives the familiar semicoherent scaling

```math
h_{\min}
\propto
S_n^{1/2}
T_{\rm coh}^{-1/4}
T_{\rm obs}^{-1/4}.
```

This scaling is a limiting idealization.

## 9. Residual Phase and Mismatch Criterion

Write the exact phase in chunk `k` as

```math
\phi(t_k+u;\theta)
=
\phi_{0,k}
+
\Phi_{\rm loc}(u;\zeta_k)
+
R_k(u;\theta).
```

The coherent overlap is

```math
\mathcal O_k
=
\left|
\left\langle e^{iR_k}\right\rangle_k
\right|,
```

where the normalized average is

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

After maximizing over arbitrary constant phase,

```math
\mu_k
=
1-|\mathcal O_k|^2
\approx
\langle R_k^2\rangle_k-\langle R_k\rangle_k^2.
```

This is a derived small-residual result.

If the retained basis is chosen by least-squares projection, `R_k` is orthogonal to the retained basis and to the constant phase direction. The coherent mismatch is then the weighted residual phase variance.

### Omitted-Term Scaling

Suppose the first omitted term is

```math
R_k(u)\approx a_m u^m.
```

For a rectangular window on

```math
u\in[-T_{\rm coh}/2,T_{\rm coh}/2],
```

with uniform weighting,

```math
\mu_k
\approx
a_m^2
\left[
\langle u^{2m}\rangle-\langle u^m\rangle^2
\right].
```

For odd `m`,

```math
\mu_k
\approx
a_m^2
\frac{(T_{\rm coh}/2)^{2m}}{2m+1}.
```

For even `m`,

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

If the local phase model retains Taylor terms through order `p`, then

```math
m=p+1,
\qquad
a_m=\frac{1}{m!}\phi^{(m)}(t_k),
```

so

```math
\mu_k
\propto
[\phi^{(p+1)}(t_k)]^2T_{\rm coh}^{2p+2}.
```

This is the mathematical reason extra coherent coordinates can allow longer `T_coh`.

## 10. Dimension Selection

There are three distinct dimension questions.

### 10.1 Phase-Approximation Dimension

This asks: how many local basis functions are needed to make the residual phase mismatch small?

Define residual functions after projecting out baseline coordinates such as constant phase and frequency:

```math
r_i(u)
=
\phi(u;\theta_i)
-
\phi_{\rm fitted}(u;\zeta_{\rm baseline}).
```

Build a weighted residual covariance operator

```math
K(u,u')
=
\sum_i \pi_i r_i(u)r_i(u'),
```

or an equivalent residual matrix under the inner product `langle ., . rangle_k`.

SVD/PCA gives basis functions `e_alpha(u)` with singular values `s_alpha`. The best rank-`d` approximation in this norm discards

```math
\epsilon_d
=
\sum_{\alpha>d}s_\alpha^2.
```

A mismatch-only dimension rule is

```math
\epsilon_d \le \mu_{\rm coh,max}.
```

This is necessary but not sufficient for search optimality.

### 10.2 Grid/Search Dimension

Even if a direction matters physically, it should become a searched `zeta` coordinate only if it can be gridded affordably.

For a metric in `zeta` coordinates,

```math
\mu_{\rm grid}
\approx
g_{ab}^{(\zeta)}
\Delta\zeta^a\Delta\zeta^b.
```

The number of grid points scales schematically as

```math
N_\zeta
\sim
\mu_{\rm grid}^{-d_\zeta/2}
\int_{\mathcal Z}
d^{d_\zeta}\zeta\,
\sqrt{\det g^{(\zeta)}}.
```

Adding a coordinate reduces model mismatch but increases `N_zeta`, memory, interpolation cost, and the trials factor.

### 10.3 Operational Dimension

The final dimension is selected by detectable amplitude:

```math
d_\zeta^\star(T_{\rm coh})
=
\arg\min_d
h_{\min}(d,T_{\rm coh})
```

subject to compute, memory, and false-alarm constraints.

A coordinate should be retained only if

```math
h_{\min}(d+1,T_{\rm coh})
<
h_{\min}(d,T_{\rm coh})
```

under the same resource budget.

## 11. Sensitivity Versus Computational Cost

Use the decomposition

```math
C_{\rm total}
=
C_{\rm spectra}
+
C_{\rm tracks}
+
C_{\rm interp}
+
C_{\rm overhead}.
```

With

```math
N_{\rm seg}=T_{\rm obs}/T_{\rm coh},
\qquad
N_\zeta=\prod_{a=1}^{d_\zeta}N_a,
```

a generic spectral construction cost is

```math
C_{\rm spectra}
\sim
N_{\rm seg}N_\zeta c_{\rm coh}(T_{\rm coh},d_\zeta).
```

Track accumulation cost is

```math
C_{\rm tracks}
\sim
N_\theta N_{\rm seg}c_{\rm lookup}(d_\zeta).
```

If multilinear interpolation is used,

```math
c_{\rm lookup}(d_\zeta)\propto 2^{d_\zeta}
```

is plausible, but this must be measured.

Memory is

```math
M_{\rm spectra}
\sim
N_{\rm seg}N_\zeta B,
```

where `B` is bytes per stored statistic.

Higher `d_zeta` can improve sensitivity by reducing `mu_model` and allowing larger `T_coh`, but it can degrade sensitivity through:

- more spectral grid points;
- more expensive spectra;
- more expensive interpolation;
- larger memory footprint;
- fewer affordable templates at fixed compute;
- larger effective trials factor;
- larger threshold.

The optimization must compare final `h_min`, not recovered power.

## 12. Choosing `T_coh`

For each candidate basis dimension `d`, define total mismatch as

```math
\mu_{\rm total}
=
\mu_{\rm model}
+
\mu_{\rm grid}
+
\mu_{\rm interp}
+
\mu_{\rm other}.
```

A coherence-time constraint is

```math
\max_{\theta,k}\mu_{\rm total}(d,T_{\rm coh};\theta,k)
\le
\mu_{\max}.
```

Because omitted-term mismatch often scales as a high power of `T_coh`, increasing `d_zeta` can substantially increase the allowed `T_coh`.

However, increasing `T_coh` changes:

- `N_seg`;
- frequency resolution;
- grid spacings;
- stationarity assumptions;
- data-gap handling;
- transform cost;
- sideband treatment;
- threshold and trial factors.

The practical procedure is:

1. For each candidate basis `mathcal B` and dimension `d`, compute mismatch versus `T_coh`.
2. Reject pairs with unacceptable residual, grid, or interpolation mismatch.
3. Estimate cost, memory, and trials factor.
4. Estimate `h_min`.
5. Choose the feasible pair with the smallest `h_min`.

## 13. Analytical Basis Construction

### Polynomial Phase

For

```math
\phi(u)=\sum_j a_j u^j,
```

the residual mismatch is quadratic in the coefficients after projecting out nuisance terms. With a fixed window, the Gram matrix is

```math
G_{ij}=\langle u^i,u^j\rangle.
```

Orthogonal polynomials diagonalize this quadratic form under the chosen weight. Therefore, for polynomial phase under simple noise and window assumptions, an analytical or semi-analytical optimal basis exists in the sense of minimizing residual phase variance.

This is an established mathematical construction, but whether it solves the full search problem still depends on cost and threshold penalties.

### PN Phase

A PN phase can be represented schematically as

```math
\phi(u;\theta)=\sum_j c_j(\theta)\psi_j(u),
```

where `psi_j` are PN time- or frequency-dependent functions. After projecting out constant phase, frequency, and any always-retained coordinates, form

```math
G_{ij}=\langle \tilde\psi_i,\tilde\psi_j\rangle
```

and the parameter-induced covariance

```math
K_{ij}
=
\sum_\theta \pi(\theta)c_i(\theta)c_j(\theta).
```

Diagonalizing the induced phase covariance or local metric gives candidate optimal coordinates.

### Is 0PN Expected To Be Best?

Established from PN structure: the Newtonian or 0PN term often dominates accumulated inspiral phase.

Conjecture: for equal-mass non-spinning 3.5PN waveforms over restricted mass, frequency, and chunk-duration ranges, the best one-dimensional residual basis may be close to the 0PN direction.

Important caveat: after projecting out constant phase and frequency, “dominates accumulated phase” is not equivalent to “dominates residual phase variance.” The best one-dimensional coordinate depends on:

- mass range;
- frequency range;
- `T_coh`;
- time-domain versus frequency-domain convention;
- window;
- noise weighting;
- whether frequency is already included in `zeta`;
- whether the coordinate is allowed to be nonlinear.

Therefore 0PN near-optimality should be tested, not assumed.

## 14. Equal-Mass Non-Spinning 3.5PN Case

With

```math
\eta=1/4,
\qquad
\chi_1=\chi_2=0,
```

the intrinsic physical family is one-dimensional, controlled by total mass or chirp mass.

This does not automatically mean one `zeta` coordinate is enough. Over a finite chunk, the phase curve induced by changing mass can have curvature in function space. A one-parameter physical family can trace a curved path through a multi-dimensional local basis.

The required study is:

1. Choose mass range, frequency range, phase convention, and `T_coh`.
2. Generate equal-mass non-spinning 3.5PN phases.
3. Project out constant phase and frequency.
4. Compare:
   - frequency-only;
   - frequency plus 0PN;
   - frequency plus best one SVD/PCA coordinate;
   - frequency plus best two or more SVD/PCA coordinates;
   - frequency plus PN-inspired coordinates.
5. Measure:
   - singular value spectrum;
   - overlap between 0PN and the first SVD/PCA direction;
   - worst-case and average mismatch versus dimension;
   - held-out mismatch;
   - grid size implied by each coordinate choice.
6. Feed these results into the cost and threshold model.

Conjecture: 0PN will be close to optimal for narrow ranges and moderate `T_coh`, but higher dimensions may become useful for long chunks or broader parameter ranges.

## 15. Equal-Mass Spinning 3.5PN Case

For equal aligned spin, a minimal model is

```math
\chi_1=\chi_2=\chi.
```

The allowed spin range must be fixed as either one-sided or signed. Generic precessing spin is a different, higher-dimensional problem and should not be mixed into this first analysis.

Spin can affect dimension in several ways:

1. It may project mostly onto the non-spinning mass-like direction.
2. It may add one dominant spin-like residual direction.
3. It may add multiple directions if signed spin introduces curvature around zero.
4. It may require amplitude/precession treatment if generic spins are allowed.

The required study is:

1. Generate equal-mass 3.5PN phases with the chosen spin convention and `|chi|` or `chi` bounded by 0.2.
2. Apply the non-spinning basis to spinning phases.
3. Build a spinning SVD/PCA basis.
4. Compare 0PN, leading spin-PN coordinates, and SVD/PCA coordinates.
5. Measure added mismatch from spin and the number of retained directions needed to match the non-spinning tolerance.

Conjecture: modest equal aligned spin may add one dominant direction, but this is uncertain until tested.

## 16. Limiting Case: Ordinary StackSlide

Set

```math
\zeta=\{f\}
```

and

```math
\Phi_{\rm loc}(u;f)=2\pi f u.
```

Then

```math
X_k(f)
=
\int_{I_k}
dt\,
W_k(t)x_a(t)e^{-i2\pi f(t-t_k)}
```

is the short-time Fourier coefficient up to phase convention.

A physical template predicts

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
\mathcal S_{\rm SS}(\theta)
=
\sum_k w_kP_k[f_k(\theta)].
```

This is ordinary StackSlide-like power accumulation along a frequency-time track. Segment time is a label, not a searched coordinate, unless spectra are explicitly gridded over epoch.

Thus the `zeta` formalism strictly contains ordinary StackSlide as the `d_zeta=1` frequency-only limit.

## 17. Operational Decision Rule

For each candidate coordinate family

```math
\zeta^{(d)}=\{\zeta^1,\ldots,\zeta^d\},
```

estimate:

```math
\mu_{\rm model}(d,T_{\rm coh}),
\qquad
\mu_{\rm grid}(d,T_{\rm coh}),
\qquad
\mu_{\rm interp}(d,T_{\rm coh}),
```

```math
C_{\rm total}(d,T_{\rm coh}),
\qquad
M_{\rm total}(d,T_{\rm coh}),
\qquad
N_{\rm eff}(d,T_{\rm coh}),
```

and then

```math
h_{\min}(d,T_{\rm coh}).
```

Retain an extra coordinate only if it reduces `h_min` at fixed compute, memory, and false-alarm probability.

A useful checklist for adding one coordinate:

1. Does it reduce residual coherent mismatch?
2. Does it permit a useful increase in `T_coh`?
3. How many new grid points does it require?
4. How much memory does it require?
5. How much interpolation cost does it add?
6. How much does it increase the effective trials factor?
7. Does the final detectable amplitude improve?

Only the last question decides the search design.

## 18. Established, Derived, Conjectural, Open

### Established From Source Artifacts

- The project aims to generalize semicoherent track summation from frequency-time spectra to richer `zeta` spectra.
- Ordinary semicoherent methods already exist and are not the novelty target.
- The key question is how many coherent-chunk coordinates to use.
- The local codebase already motivates chirp-aware coordinates such as `{f,beta}`.
- Weighted or covariance-aware statistics are needed for realistic detector response and noise.

### Derived Here

- `theta -> zeta_k(theta)` can be defined as a local phase projection.
- The coherent mismatch for small residual phase is

```math
\mu_k
\approx
\langle R_k^2\rangle_k-\langle R_k\rangle_k^2.
```

- If the first omitted term is order `m`, then

```math
\mu_k\propto T_{\rm coh}^{2m}.
```

- The ideal equal-weight null distribution is Gamma.
- The ordinary StackSlide statistic is the `zeta={f}` limiting case.
- Weak-signal optimal scalar weights satisfy

```math
w_k\propto \kappa_k,
```

where `kappa_k` is the expected per-unit-`h_0^2` signal contribution.

### Conjectures

- The useful `zeta` dimension generally increases with `T_coh`.
- 0PN is close to the best one-dimensional coordinate for equal-mass non-spinning 3.5PN waveforms only in restricted regimes.
- Spin up to `chi=0.2` may add one dominant direction, but this is not established.
- There may be a compute-limited regime where `{f,beta}` or an SVD/PCA coordinate beats frequency-only StackSlide.

### Open Questions

- What exact 3.5PN phase convention and parameter ranges should define the basis study?
- Is the project’s `{f,beta}` construction equivalent to a known chirp-transform specialization?
- What is the correct covariance of NUFFT/resampled 5-vector bins?
- What is the empirical effective trials factor for high-dimensional `zeta` spectra?
- Does memory bandwidth dominate before FLOP cost?
- How close is the first SVD/PCA direction to the 0PN direction in the actual target regime?

## 19. Immediate Next Steps

1. Define mass, frequency, spin, and `T_coh` ranges for the 3.5PN studies.
2. Implement phase-residual projection using `conda run -n PBH python ...`.
3. Run the equal-mass non-spinning SVD/PCA comparison against 0PN.
4. Repeat with the chosen equal-aligned-spin convention up to `chi=0.2`.
5. Convert mismatch curves into grid sizes, memory, and runtime estimates.
6. Estimate `N_eff` and thresholds for each candidate dimension.
7. Compare final detectable amplitude at fixed compute and false alarm, not recovered power alone.