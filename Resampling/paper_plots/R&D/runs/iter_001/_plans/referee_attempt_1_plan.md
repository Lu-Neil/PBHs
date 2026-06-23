# Scratchpad Plan: `referee`

## Artifact Goal

Produce a skeptical referee-style critique of the proposed higher-dimensional semicoherent `zeta` search idea, using `idea.md` and the human feedback as source of truth. The artifact should challenge the proposal’s novelty, mathematical framing, normalization, test plan, and distinction from matched filtering or known semicoherent/chirp-transform methods.

## Core Claims To Critique

1. Higher-dimensional coherent-chunk spectra can improve semicoherent sensitivity-cost tradeoffs relative to frequency-time spectra.
2. There is a principled way to choose `dim(zeta)` based on waveform family, `T_coh`, sensitivity, and compute.
3. A `{f,t,beta}` or related 0PN coordinate construction may be a useful intermediate representation for PN chirps.
4. SVD/PCA or reduced-basis analysis can reveal whether 0PN is close to optimal for equal-mass non-spinning 3.5PN waveforms.
5. Spin up to `chi = 0.2` may require additional `zeta` dimensions.
6. The method is not meant to beat matched filtering, but to occupy a better compute-sensitivity tradeoff.

## Strongest Objections To Develop

- The proposal may be mostly a reparameterized semicoherent matched-filter bank or chirp-transform search, not a distinct method.
- “Higher-dimensional spectra” may simply move cost from template filtering into spectrum construction and storage.
- Adding dimensions can improve local coherence while worsening total sensitivity through trials factors, interpolation cost, and memory limits.
- The proposal lacks a concrete objective function: recovered power, mismatch, detection probability, or detectable amplitude at fixed false alarm and compute.
- `t` should not be counted as a searched `zeta` dimension unless spectra are actually gridded over epoch.
- If equal-mass non-spinning 3.5PN is intrinsically one-dimensional, the “dimension selection” result may be trivial unless local phase curvature over chunks is the real target.
- SVD/PCA optimality is norm-dependent and may not correspond to detection optimality.
- The 0PN comparison may be ill-posed unless frequency, phase, time shift, and amplitude nuisance directions are projected consistently.
- The spinning case is ambiguous without specifying aligned/equal/signed/precessing spins.
- The proposed statistic risks incorrect noise normalization, especially under NUFFT/resampling and 5-vector sideband extraction.

## Established vs Conjectural

Mark as established:

- Semicoherent power summing along tracks is established prior art.
- Coherent time, mismatch, template count, and computing cost are already central to semicoherent searches.
- Chirp/polynomial-phase representations are known signal-processing ideas.
- Reduced bases/SVD/PCA can compress waveform families, subject to a chosen norm.

Mark as conjectural:

- A reusable `zeta` spectrum gives better sensitivity at equal compute.
- The optimal one-parameter basis is close to 0PN.
- Spin up to `chi = 0.2` adds only one important direction.
- A metric/SVD dimension rule remains valid after false-alarm and cost penalties.
- The NUFFT/5-vector statistic can be normalized with simple independent-bin assumptions.

## Possible Normalization Errors To Flag

- Confusing Hz and angular frequency.
- Missing factor of `T_coh`, window normalization, or PSD normalization in coherent powers.
- Treating NUFFT outputs as FFT-normalized without correcting the noise transfer function.
- Assuming independent exponential powers when bins, chunks, or sidebands are correlated.
- Ignoring covariance among the 5-vector sidereal bins.
- Ignoring interpolation-induced covariance and amplitude loss.
- Using `S_n(f0)` instead of the effective noise integrated along the chirp track.
- Treating trials factor as proportional to raw grid count without measuring correlations, or ignoring it entirely.

## Missing Tests To Demand

- Frequency-only StackSlide baseline at equal compute and false alarm.
- Local semicoherent matched-filter-bank baseline.
- Chirp-transform or polynomial-phase-transform analogue baseline if applicable.
- Null-distribution calibration for each `dim(zeta)`.
- Detection probability versus amplitude at fixed total false alarm.
- Runtime and memory measurements, not just asymptotic scaling.
- Interpolation mismatch and covariance tests.
- Held-out waveform tests for SVD/PCA bases.
- Equal-mass non-spinning 3.5PN: 0PN vs first SVD direction, with nuisance projections documented.
- Equal-mass spinning 3.5PN: define spin convention and test whether non-spinning basis fails.
- Sensitivity to mass range, frequency band, `T_coh`, window, and noise weighting.

## Shallow Reasoning Checks

- Does the artifact avoid claiming novelty without comparing to known methods?
- Does it separate basis approximation error from search sensitivity?
- Does it include false-alarm thresholds and trials factors?
- Does it ask whether extra `zeta` dimensions are affordable, not merely useful?
- Does it point out that PCA/SVD rank is not the same as optimal search dimension?
- Does it explicitly say what would falsify the idea?
- Does it avoid inventing citations?
- Does it identify exact ambiguities in the problem statement rather than vaguely calling it unclear?

## Artifact Structure

1. **Audit Metadata**
   - Source: `idea.md`, prior artifacts, human feedback.
   - Attempt: 1.
   - Status: skeptical review, not final literature conclusion.

2. **Summary Judgment**
   - Narrow defensible claim.
   - Main skepticism: novelty and cost-sensitivity proof burden.

3. **Strongest Objections**
   - Prior-art overlap.
   - Cost displacement.
   - Trials-factor penalty.
   - Ambiguous objective.
   - Basis optimality caveats.

4. **Unclear Assumptions**
   - Definition of `zeta`.
   - Whether `t` is a coordinate or segment label.
   - Waveform convention.
   - Spin convention.
   - Noise model.
   - Parameter ranges.

5. **Novelty Concerns**
   - StackSlide/PowerFlux/Weave-like searches.
   - Chirp and polynomial-phase transforms.
   - Reduced basis and template-bank metrics.
   - Matched filtering equivalence.

6. **Normalization Risks**
   - PSD, NUFFT, 5-vector covariance, bin correlations, interpolation, units.

7. **Missing Tests**
   - Baselines, null calibration, sensitivity curves, cost measurement, SVD/PCA validation.

8. **Is This Different From Matched Filtering?**
   - Explain possible equivalence.
   - Define what would make it genuinely different: reusable low-dimensional chunk spectra with demonstrably better cost-sensitivity tradeoff.

9. **Case-Specific Critique**
   - Equal-mass non-spinning 3.5PN.
   - Equal-mass spinning up to `chi = 0.2`.

10. **Falsification Criteria**
   - When the idea should be abandoned or reframed.

## Sections Needing Most Technical Depth

- Normalization risks.
- Difference from semicoherent matched filtering.
- Dimension selection versus PCA/SVD rank.
- Equal-mass 3.5PN non-spinning and spinning critiques.
- Concrete tests required before any claim of improvement.