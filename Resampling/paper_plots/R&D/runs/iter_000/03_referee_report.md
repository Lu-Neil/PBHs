---
title: Referee report on higher-dimensional semicoherent zeta-space searches
source: idea.md
agent: referee
attempt: 1
status: skeptical review artifact
required_output: outputs/03_referee_report.md
---

## Audit Status

This report treats `idea.md` as the source of truth.

No shell commands were run. No files, code, Zotero entries, or external papers were inspected in this attempt. I did not independently verify the literature artifact. Literature overlap statements below are based on the prior artifacts and are marked as uncertain when not established.

## Bottom Line

The idea is plausible as a research direction, but currently underspecified and at high risk of being a relabeling of existing semicoherent matched-filter, StackSlide, Hough/Radon, fast-chirp-transform, or polynomial-phase-transform methods.

The strongest version of the claim should not be “construct higher-dimensional spectra and sum along tracks.” That is probably not novel. The defensible claim, if it survives review, is narrower:

> choose and optimize an intermediate local coherent coordinate system `zeta`, including its dimension, for semicoherent searches of modeled chirping gravitational-wave signals, with an explicit cost-sensitivity tradeoff.

That narrower claim still needs mathematical, statistical, and computational validation.

## Claim Taxonomy

### Established Facts From `idea.md`

- The proposed method forms spectra in a coordinate space `zeta` that includes `f` and `t`, and may include other model-derived parameters.
- A nearby implementation reportedly uses `zeta = {f, t, beta}`, where `beta` is a 0PN parameter.
- The desired question is computational: how many `zeta` dimensions should be used, as a function of coherent chunk length `T_coh`.
- Adding more model parameters to the coherent chunks may permit longer coherent integrations but may increase spectrum-building and track-summing cost.
- Related techniques include StackSlide, PowerFlux, Weave, MBTA, ROQ, and PCA.

### Derived Results

- If the method sums powers over segments, it is a semicoherent statistic and generally loses inter-segment phase information.
- If the full phase model is known and the data are Gaussian, coherent matched filtering is the optimal Neyman-Pearson statistic for known parameters.
- Therefore any power-summing `zeta` method must justify itself by computational cost, robustness, or model uncertainty, not by claiming statistical optimality over matched filtering.
- Increasing `dim(zeta)` has two opposing effects: it can reduce coherent mismatch but increases grid size, memory, interpolation cost, correlations, and likely trials factor.

### Assumptions

- The local phase model indexed by `zeta` can approximate the true phase well inside each coherent chunk.
- The chosen `zeta` coordinates can be gridded efficiently.
- The resulting spectra have a tractable noise distribution.
- The cost of computing higher-dimensional spectra is offset by reduced mismatch or larger `T_coh`.
- Effective template counts and false-alarm thresholds can be calibrated.

### Conjectures

- There may be an optimal intermediate `zeta` dimension between ordinary time-frequency StackSlide and full coherent matched filtering.
- PN coefficients, chirp-time coordinates, or metric/PCA eigen-coordinates may be better than raw physical parameters.
- The current `zeta = {f,t,beta}` example may be a useful special case for PBH-inspired chirps.

### Open Questions

- Is this already equivalent to a fast chirp transform or polynomial-phase transform?
- Is `beta` globally fixed, segment-local, or template-dependent?
- Are spectra computed once for a bank of `zeta`, or recomputed per physical template?
- What is the null distribution after NUFFT/resampling/interpolation?
- What exactly is optimized: detection probability, amplitude sensitivity, wall time, FLOPs, memory, or total search cost at fixed false alarm?

## Strongest Objections

### 1. The core idea may already be known

The phrase “construct spectra in a higher-dimensional space and sum along model tracks” sounds like a generalized Hough/Radon/chirp-transform construction. In gravitational-wave language, it also resembles StackSlide or semicoherent template-bank searches with a richer coherent statistic.

This is the most serious novelty risk. The project must show that it is not merely:

- StackSlide with extra spin-down/chirp dimensions;
- PowerFlux with a different local demodulation;
- Weave-style semicoherent template placement;
- a fast chirp transform;
- a polynomial-phase transform;
- a Radon/Hough transform in a higher-dimensional accumulator;
- a hierarchical matched-filter search with local Taylor-expanded phase.

Until that distinction is made explicitly, novelty is weak.

### 2. “Optimal” is undefined

The title asks for “optimal ways,” but `idea.md` does not define the objective function.

Possible objectives are not equivalent:

- minimize compute at fixed detection probability;
- maximize detection probability at fixed compute;
- minimize detectable amplitude at fixed false alarm;
- minimize total wall time;
- minimize memory;
- minimize number of templates;
- maximize robustness to phase-model error.

A reviewer would reject any claim of optimality without a precise objective and constraints.

### 3. The method may be statistically suboptimal by construction

If the statistic accumulates power, it discards phase between coherent chunks. For a modeled signal with known phase evolution, the optimal Gaussian-noise statistic is coherent matched filtering over the full observation, assuming feasible computation and accurate model.

Therefore this method is not “optimal” in the statistical sense unless “optimal” means compute-constrained or robustness-constrained. The report must state this directly.

### 4. The dimensionality tradeoff may be unfavorable

Adding a coordinate to `zeta` can increase the grid size multiplicatively. If `N_zeta = product_a N_a`, even moderate resolution in a new coordinate may dominate cost and memory.

The idea assumes that the added coordinate buys enough longer `T_coh` or mismatch reduction to compensate. That is plausible but not established. It must be demonstrated quantitatively.

### 5. Trials factors may erase sensitivity gains

Higher-dimensional spectra and more possible tracks likely increase the number of effective trials. Even if the raw signal power increases because chunks are better demodulated, the detection threshold may also rise.

A fair comparison must include false-alarm calibration. Comparing recovered power alone is insufficient.

### 6. Noise covariance is likely nontrivial

The active pipeline uses NUFFT/resampling and 5-vector sideband extraction. Those operations can correlate nearby bins and sidebands. A simple sum of normalized powers may have the wrong null distribution.

This is especially important because the roadmap already notes that the NUFFT changes the noise PSD. If the statistic uses the wrong noise normalization, any claimed sensitivity improvement may be artificial.

### 7. The current formulation mixes coordinates and labels

`zeta = {f,t,beta}` is potentially ambiguous. Segment time `t` may be an index labeling chunks, not a spectral coordinate searched within each chunk. If `t` is counted as a dimension of `zeta`, the cost model may double-count or misrepresent the transform space.

The formalism should distinguish:

- segment index or center time `t_k`;
- coherent spectral coordinates within a segment;
- physical template parameters `theta`;
- semicoherent track coordinates across segments.

## Unclear Assumptions

1. It is unclear whether `zeta` is a subset of physical parameters, a local Taylor expansion, a PN coefficient basis, a metric eigenbasis, or a learned reduced basis.

2. It is unclear whether each `zeta` spectrum is computed by a fast transform, by brute-force demodulation, by NUFFT, or by reusing previously computed arrays.

3. It is unclear whether `beta` is treated as constant over the observation, local per segment, or derived from global physical parameters.

4. It is unclear how amplitude modulation, detector response, gaps, and sidereal sidebands are handled in the semicoherent statistic.

5. It is unclear whether the statistic sums scalar powers, 5-vector matched powers, likelihood ratios, or covariance-weighted amplitudes.

6. It is unclear how interpolation in `zeta` space is performed and how interpolation bias is controlled.

7. It is unclear what parameter-space volume is searched. Without this, template counts and trials factors cannot be evaluated.

8. It is unclear what “higher-dimensional spectra” means operationally: stored arrays, implicit transforms, local matched-filter banks, or accumulator maps.

## Novelty Concerns

### High Risk

The general concept overlaps strongly with known classes of methods:

- semicoherent StackSlide-like searches;
- Hough/Radon accumulation along tracks;
- fast chirp transforms;
- polynomial-phase transforms;
- semicoherent metric template banks;
- local matched-filter banks followed by incoherent combination.

The next literature agent should specifically test whether the `zeta = {f,t,beta}` construction is mathematically a known chirp transform or polynomial-phase transform.

### Medium Risk

Using PCA, SVD, metric eigenvectors, chirp-time coordinates, or reduced bases to choose coordinates is also likely not novel by itself. The novelty would need to be in using such coordinates as axes of an intermediate semicoherent spectral representation and optimizing their retained dimension.

### Lower Risk

Applying the framework to PBH-inspired long chirps with sidereal 5-vector reconstruction may be more specialized and potentially publishable, but only if the statistical calibration and cost comparison are rigorous.

## Possible Normalization Errors

1. **Angular frequency versus Hz**  
   The repository instructions warn that internal frequencies are often angular frequencies in rad/s. A factor of `2 pi` error could shift phase, grid spacings, mismatch estimates, and NUFFT bin interpretation.

2. **Power versus amplitude normalization**  
   If the statistic sums `|X|^2`, the expected value under noise depends on the window, sampling, PSD convention, and one-sided versus two-sided normalization.

3. **NUFFT noise transfer**  
   The NUFFT/resampling operation does not preserve the input PSD. The effective noise should depend on the full chirp track, not only `S_n(f0)`. Any detection statistic normalized by `S_n(f0)` alone is suspect.

4. **5-vector covariance**  
   The five sidereal bins should not automatically be treated as independent equal-variance samples. The correct estimator should use a covariance matrix if bin correlations or unequal noise are present.

5. **Window and gap effects**  
   Segment windows and data gaps alter both signal response and noise covariance. A rectangular-window chi-square model may fail.

6. **Interpolation correlations**  
   Interpolating between `zeta` grid points mixes neighboring powers. This changes the effective distribution and invalidates naive independent-bin trials counts.

7. **Dirichlet/bin-mismatch corrections**  
   If the recovered carrier does not land exactly on a Fourier or NUFFT bin, the power loss must be corrected or included in mismatch. Otherwise coordinate choices may be compared unfairly.

8. **Double counting segment time**  
   Treating `t` as both segment label and search dimension may inflate or confuse dimensional cost estimates.

9. **Power-sum SNR scaling**  
   Semicoherent power statistics scale differently from coherent matched-filter SNR. Reports must distinguish amplitude sensitivity, power noncentrality, and detection-statistic SNR.

## Missing Tests

### Minimal Toy Tests

- Compare `{f}` versus `{f, fdot}` versus `{f, fdot, fddot}` on polynomial-phase signals.
- Measure coherent mismatch versus `T_coh` and compare with analytic residual phase predictions.
- Verify that added dimensions improve detection at equal false alarm, not merely recovered power.

### Null Calibration

- Run noise-only trials for each `zeta` dimension.
- Estimate empirical null distributions.
- Measure effective trials factors.
- Check whether the assumed gamma/chi-square distribution holds.

### Normalization Tests

- Inject white Gaussian noise with known variance and confirm expected spectrum normalization.
- Inject colored Gaussian noise and confirm the NUFFT/resampling noise transfer.
- Confirm one-sided/two-sided PSD conventions and angular-frequency conventions.

### Injection Tests

- Inject signals exactly matching the local `zeta` model.
- Inject signals with omitted higher-order phase terms.
- Sweep parameter offsets to measure metric/mismatch predictions.
- Compare detection probability at fixed false-alarm probability.

### Cost Tests

- Benchmark spectrum construction separately from track accumulation.
- Measure memory bandwidth and storage costs.
- Measure interpolation cost as `dim(zeta)` increases.
- Compare brute-force demodulation, NUFFT, and any fast-transform implementation fairly.

### 5-Vector Tests

- Estimate the 5-bin covariance matrix from noise.
- Compare equal-weight power sums against covariance-weighted 5-vector estimators.
- Test sidereal sideband recovery under gaps and nonstationary noise.

### Matched-Filtering Baselines

- Compare against fully coherent matched filtering on small parameter volumes where it is computationally feasible.
- Compare against ordinary StackSlide at equal compute.
- Compare against a local matched-filter bank plus incoherent summation, since that may be operationally identical to the proposed method.

## Is This Genuinely Different From Matched Filtering?

At present, only partially.

A coherent matched filter computes an inner product between the data and a full signal template over the observation. The proposed method appears to compute coherent local statistics in each segment and then sum powers along a model-predicted track. That is genuinely different from fully coherent matched filtering because it discards inter-segment phase.

However, it may not be different from a semicoherent matched-filter bank. If each `zeta` point corresponds to a local phase template and the final statistic sums local matched-filter powers, then the method is semicoherent matched filtering expressed in transformed coordinates.

The method becomes distinguishable only if the project contributes one or more of the following:

- a principled optimization over the retained local phase-coordinate dimension;
- a fast reusable transform that computes many local chirp-template powers more cheaply than a template bank;
- a cost model showing when intermediate `zeta` spaces beat both ordinary StackSlide and full local matched-filter banks;
- a calibrated statistic for PBH-like chirps with sidereal 5-vector structure.

Without those, the method is best described as semicoherent matched filtering or StackSlide in a different coordinate system.

## Required Clarifications Before Further Development

1. Define the exact statistic:
   ```math
   S(theta) = ?
   ```
   Include normalization, weights, and whether the inputs are scalar powers, complex amplitudes, or 5-vectors.

2. Define `zeta` operationally:
   ```math
   zeta = {what coordinates, with what units, gridded how?}
   ```

3. Define the optimization target:
   ```math
   minimize cost at fixed P_det and P_FA?
   maximize P_det at fixed cost?
   minimize h0_min?
   ```

4. Define the baseline methods:
   ordinary StackSlide, fully coherent matched filtering on small problems, local semicoherent matched-filter bank, and any fast chirp transform analogue.

5. Define the noise model and covariance:
   especially for NUFFT/resampled spectra and 5-vector bins.

6. Define the searched parameter-space volume:
   otherwise template counts and trials factors are meaningless.

## Reviewer Recommendation

Do not frame the project yet as a new optimal method. Frame it as an investigation of whether an intermediate local phase-coordinate representation can improve compute-limited semicoherent searches for modeled chirps.

The next artifact should be a falsifiable comparison plan:

- baseline methods;
- exact statistic;
- exact normalization;
- parameter-space volume;
- false-alarm calibration;
- cost model;
- toy simulation that can prove the idea wrong.

The project is worth pursuing only if it can show improved sensitivity at equal compute and equal false-alarm probability, not merely cleaner tracks or higher recovered power.