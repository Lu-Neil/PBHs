---
title: Research plan for optimal semicoherent searches in higher-dimensional spectral spaces
source: idea.md
attempt: 1
status: planning artifact
---

## Audit status

No external papers are claimed as read in this artifact. Literature items below are search targets unless explicitly marked as already established from `idea.md`. Citations must be filled by a later literature-review agent.

## Central research question

What is the computationally optimal way to construct semicoherent search statistics for modelled signals by choosing an intermediate spectral coordinate space `zeta`, potentially higher-dimensional than the usual time-frequency plane, so that the total cost of coherent spectrum construction plus semicoherent track accumulation is minimized at fixed sensitivity?

Equivalently: given a signal phase model `phi(theta)` and coherent segment duration `T_coh`, how should one choose the reduced or transformed parameter set `zeta`, including possible linear or nonlinear combinations of physical parameters `theta`, to maximize sensitivity per unit compute?

## Working distinction of claim types

### Established from `idea.md`

- Conventional semicoherent searches often form time-frequency spectra and sum power along template tracks.
- The proposed alternative is to form spectra over a higher-dimensional coordinate space `zeta`, where `zeta` includes at least frequency and time and may include additional phase-evolution parameters.
- The current nearby implementation contains an example using `zeta = {f, t, beta}`, where `beta` is a 0PN parameter.
- Increasing the number of parameters in `zeta` may increase allowable coherent time but also increases the cost of spectrum construction and track summation.
- Related techniques to investigate include StackSlide, PowerFlux, Weave, MBTA, reduced-order quadrature, and PCA.

### Assumptions to test

- A higher-dimensional spectral representation can reduce mismatch within each coherent chunk enough to justify its added computational cost.
- There is an optimal intermediate dimension of `zeta` for each search regime rather than a monotonic preference for either fully coherent templates or ordinary time-frequency StackSlide.
- Useful `zeta` coordinates may be linear combinations, principal components, or PN-order combinations of the physical parameters `theta`.

### Conjectures

- The optimal `zeta` dimension depends primarily on `T_coh`, intrinsic phase curvature over each segment, target mismatch, and available compute.
- For PN chirp models, including more PN-like coordinates in coherent chunks should increase the maximum useful `T_coh`, but with diminishing returns.
- Existing signal-processing or gravitational-wave template-bank literature may already contain an equivalent formulation under different terminology.

### Open questions

- What objective function should define "optimal": minimum cost at fixed sensitivity, maximum sensitivity at fixed cost, or minimum wall time for fixed false-alarm and false-dismissal probabilities?
- Should the semicoherent statistic sum power, likelihood ratios, marginalized coherent statistics, or noise-weighted matched-filter outputs?
- Can `zeta` be chosen analytically from the phase metric, or does it require empirical optimization?

## Subquestions

1. Formalism:
   - Given `phi(t; theta)`, how should a coherent segment statistic be written when spectra are indexed by `zeta` rather than only `{f, t}`?
   - What is the map from physical parameters `theta` to tracks through `zeta` space?
   - What mismatch is incurred when the signal model inside each chunk is approximated by the chosen `zeta` coordinates?

2. Sensitivity:
   - How does expected SNR scale with `T_coh`, number of segments, and model mismatch?
   - How does semicoherent power accumulation compare with coherent matched filtering and ordinary StackSlide?
   - How should false-alarm thresholds scale with the number of searched templates/tracks in `zeta` space?

3. Computational cost:
   - What is the cost of creating spectra over `zeta` as a function of dimension, grid spacing, segment count, and data length?
   - What is the cost of semicoherent track accumulation through the `zeta` grid?
   - When does adding one more `zeta` coordinate reduce the required number of semicoherent templates enough to be worthwhile?

4. Coordinate choice:
   - Are PN coefficients, chirp times, phase-metric eigenvectors, or PCA components better coordinates than physical masses/spins?
   - When are linear combinations of `theta` sufficient?
   - How does the optimal coordinate choice change with `T_coh`?

5. Relationship to existing methods:
   - Is this equivalent to known semicoherent template-bank optimization, hierarchical searches, Hough/StackSlide generalizations, Weave, MBTA, reduced-order quadrature, or reduced-basis methods?
   - Which parts are established in continuous-wave searches versus compact-binary searches?
   - Which literature results can be directly reused?

## Required derivations

1. General semicoherent statistic in `zeta` space:
   - Define data split into segments of duration `T_coh`.
   - Define coherent statistic `P_k(zeta)` for segment `k`.
   - Define a template track `zeta_k(theta)` through segment-indexed spectral space.
   - Define semicoherent statistic:
     ```math
     S(theta) = \sum_k w_k P_k(zeta_k(theta))
     ```
     or its likelihood-based analogue.
   - Mark whether the statistic is power-only, complex-amplitude-preserving, or noise-weighted.

2. Phase approximation and mismatch:
   - Expand the exact phase within a segment around a reference time:
     ```math
     phi(t; theta) = phi_0 + 2 pi f (t - t_0)
       + \sum_{n >= 1} a_n(theta) (t - t_0)^{n+1}
     ```
   - Define which coefficients are represented in `zeta`.
   - Derive residual phase error after truncating or projecting onto `zeta`.
   - Convert residual phase error into coherent mismatch.

3. Metric-based grid spacing:
   - Derive or quote with citation needed the coherent phase metric in physical parameters `theta`.
   - Derive induced metric in `zeta`.
   - Determine grid spacing for a target maximum mismatch.
   - Identify how grid density scales with `T_coh`.

4. Cost model:
   - Write total cost as:
     ```math
     C_total(d_zeta, T_coh)
       = C_spectra(d_zeta, T_coh)
       + C_tracks(d_zeta, T_coh)
       + C_overhead
     ```
   - Derive scaling of `C_spectra` with number of segments and number of `zeta` grid points.
   - Derive scaling of `C_tracks` with number of semicoherent templates and number of segments.
   - Include memory-bandwidth costs, not only floating-point counts.

5. Optimization condition:
   - For candidate coordinate sets `zeta_d`, derive criterion:
     ```math
     d*, T_coh* = argmin C_total
     ```
     subject to fixed sensitivity, mismatch, and false-alarm constraints.
   - Compare discrete choices: `{f,t}`, `{f,t,beta_0PN}`, `{f,t,beta_0PN,beta_1PN}`, etc.

6. PN waveform specialization:
   - Express the 3.5PN phase model in terms of coefficients suitable for segment-local approximation.
   - Identify which PN coefficients dominate residual phase over a segment.
   - Derive scaling of maximum allowed `T_coh` when using 0PN, 1PN, 1.5PN, etc. approximations inside coherent chunks.

7. Statistical distribution:
   - Derive the null distribution of the semicoherent statistic for power sums.
   - Include trials factor for number of searched tracks.
   - Derive detection threshold for target false-alarm probability.
   - Derive expected noncentrality under signal injection.

## Required literature searches

Searches should explicitly record query strings, databases used, papers found, whether each paper is in Zotero, and relevance notes.

1. StackSlide and semicoherent continuous-wave searches:
   - Search target: foundational StackSlide papers.
   - Search target: semicoherent metric and template-bank placement for continuous gravitational waves.
   - Search target: computational optimization of `T_coh` in semicoherent CW searches.
   - Citation needed.

2. PowerFlux:
   - Search target: PowerFlux method papers and cost/sensitivity analysis.
   - Search target: comparison of PowerFlux with StackSlide/Hough/F-statistic methods.
   - Citation needed.

3. Weave:
   - Search target: Weave semicoherent search method.
   - Search target: Weave template-bank construction and metric formalism.
   - Citation needed.

4. MBTA:
   - Search target: Multi-Band Template Analysis for compact binary coalescence.
   - Search target: how MBTA partitions waveform frequency bands to reduce cost.
   - Citation needed.

5. Reduced-order quadrature and reduced-basis methods:
   - Search target: ROQ for gravitational-wave likelihood evaluation.
   - Search target: reduced basis for compact-binary waveforms.
   - Determine whether coordinate compression ideas transfer to semicoherent spectra.
   - Citation needed.

6. PCA or coordinate-reduced template banks:
   - Search target: principal component analysis for gravitational-wave template banks.
   - Search target: chirp-time coordinates and metric eigen-coordinates.
   - Citation needed.

7. Signal-processing analogues:
   - Search target: polynomial phase signal detection.
   - Search target: chirp transforms, fast chirp transform, Radon/Hough transforms, generalized time-frequency representations.
   - Search target: computationally optimal search over polynomial phase tracks.
   - Citation needed.

8. Existing local references:
   - Read `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md`.
   - Inspect current implementation above this directory, especially the example using `zeta = {f,t,beta}`.
   - Note useful papers not already in Zotero.

## Required simulations

1. Toy polynomial-phase signals:
   - Generate signals with phase models containing frequency, chirp rate, and higher-order derivatives.
   - Compare ordinary `{f,t}` StackSlide against `{f,t,fdot}` and `{f,t,fdot,fddot}` spectral spaces.
   - Measure sensitivity versus cost at fixed false-alarm probability.

2. PN chirp signals:
   - Use a 3.5PN phase model.
   - Test coherent chunk models of increasing PN order: 0PN, 1PN, 1.5PN, etc.
   - Measure maximum useful `T_coh` before mismatch exceeds target values.

3. Track accumulation cost:
   - Benchmark spectrum creation and track summation separately.
   - Record scaling with number of `zeta` dimensions, grid sizes, and segment count.
   - Include memory use and wall time.

4. Noise-only calibration:
   - Run many noise-only trials.
   - Estimate null distribution of each statistic.
   - Compare with derived chi-square or Gaussian approximations.
   - Estimate trials-factor corrections empirically.

5. Signal injections:
   - Inject signals at controlled amplitudes and parameter offsets.
   - Measure detection probability versus amplitude.
   - Compare sensitivity of candidate `zeta` choices at equal compute.

6. Coordinate comparison:
   - Compare physical PN coefficients, chirp-time coordinates, metric eigenvectors, and PCA-derived coordinates.
   - Evaluate grid compactness, interpolation error, and track-summation complexity.

7. Realistic constraints:
   - Repeat a subset using colored Gaussian noise.
   - If relevant later, test gaps and detector modulation using the existing resampling pipeline.

## Likely bottlenecks

1. Dimensionality:
   - Adding `zeta` coordinates may cause spectrum grids to grow too quickly.
   - The main risk is moving cost from semicoherent tracking into spectrum construction without net gain.

2. Trials factor:
   - Higher-dimensional searches may create many more effective templates, raising detection thresholds.

3. Metric validity:
   - Quadratic mismatch approximations may fail for long coherent chunks or strongly chirping signals.

4. Coordinate degeneracies:
   - PN parameters may be highly correlated.
   - A naive coordinate choice may over-cover irrelevant directions.

5. Track-summation implementation:
   - Efficient accumulation through high-dimensional arrays may be memory-bandwidth limited.
   - Interpolation between grid points may dominate cost or introduce bias.

6. Literature overlap:
   - The core idea may already exist under names such as polynomial phase transforms, fast chirp transforms, or semicoherent metric optimization.

7. Statistical calibration:
   - Power sums are easier to compute but may be suboptimal relative to likelihood-based or complex-amplitude-preserving statistics.

## Proposed report structure

1. Executive summary:
   - State the research question.
   - Summarize whether higher-dimensional spectra are promising.
   - Identify the recommended `zeta` choices and `T_coh` regimes.

2. Background:
   - Conventional semicoherent searches.
   - Time-frequency StackSlide-style statistics.
   - Motivation for higher-dimensional spectral spaces.

3. Literature review:
   - StackSlide, PowerFlux, Weave.
   - MBTA and multiband compact-binary methods.
   - Reduced-order quadrature and reduced bases.
   - PCA, chirp-time coordinates, and metric template banks.
   - Polynomial phase/chirp-transform signal processing.
   - Zotero gap list.

4. Mathematical formalism:
   - Define `theta`, `zeta`, coherent segments, spectra, and tracks.
   - Derive semicoherent statistic.
   - Derive mismatch and metric expressions.

5. Coordinate choices:
   - Frequency-time baseline.
   - PN coefficient coordinates.
   - Chirp-time coordinates.
   - Metric eigen-coordinates/PCA coordinates.
   - Criteria for selecting `zeta`.

6. Sensitivity analysis:
   - SNR scaling.
   - Mismatch penalties.
   - Null and signal distributions.
   - False-alarm thresholds and trials factors.

7. Computational cost model:
   - Spectrum construction cost.
   - Track accumulation cost.
   - Memory and interpolation costs.
   - Optimization over `d_zeta` and `T_coh`.

8. Simulation study:
   - Toy polynomial-phase setup.
   - PN chirp setup.
   - Noise-only calibration.
   - Signal-injection sensitivity comparisons.
   - Runtime benchmarks.

9. Results:
   - Cost-sensitivity tradeoff plots.
   - Recommended regimes for each `zeta` dimension.
   - Cases where higher-dimensional spectra are not worthwhile.

10. Discussion:
   - Relation to existing methods.
   - Practical implementation constraints.
   - Open theoretical questions.

11. Conclusion:
   - Answer the central research question as far as supported.
   - State next steps for implementation or further study.

## Immediate next-agent tasks

1. Read `CODEX.md` and current local implementation of the `{f,t,beta}` example.
2. Perform the literature searches above without inventing citations.
3. Build a citation table with columns: method, paper, year, Zotero status, relevance, reusable result.
4. Draft the formal cost model before running large simulations.
5. Start with toy polynomial-phase simulations before using the full 3.5PN waveform.