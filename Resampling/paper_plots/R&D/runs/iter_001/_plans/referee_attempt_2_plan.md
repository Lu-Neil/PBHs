# Scratchpad Plan: Referee Attempt 2

## Audit Metadata

- Agent: `referee`
- Attempt: 2
- Source of truth: `idea.md` plus attempt-2 human feedback
- Purpose: skeptical critique of the research idea
- Final artifact target: referee report, not synthesis
- No new citations unless already verified by prior literature artifact
- No scripts needed for this referee pass

## Concrete Improvements Over Attempt 1

- Address the human feedback directly: the user is not claiming semicoherent searches are new and is not trying to beat fully coherent matched filtering.
- Reframe the critique around the sharper question: how to choose `dim(zeta)` for best sensitivity-compute tradeoff.
- Add more focused criticism of the proposed SVD/PCA analysis on equal-mass non-spinning 3.5PN waveforms.
- Add a skeptical section on whether analytical optimal low-dimensional bases are possible.
- Treat equal-mass spinning waveforms with `chi <= 0.2` as a required case, not a side note.
- Make the matched-filter comparison more nuanced: the question is not “is this optimal?” but “is this cheaper or more reusable than semicoherent matched filtering?”
- Strengthen normalization-error section around rank truncation, metric projection, PN units, and trials factors.

## Core Claims To Test/Criticize

1. Higher-dimensional coherent-chunk spectra are useful only if they reduce detectable amplitude at fixed compute, memory, and false-alarm probability.
2. Choosing `dim(zeta)` from SVD/PCA mismatch alone is insufficient.
3. The equal-mass non-spinning 3.5PN case may be too low-dimensional to prove general usefulness.
4. A 0PN-like coordinate may be close to optimal, but only after specifying projection, phase convention, mass/frequency range, and inner product.
5. Spin up to `chi=0.2` may or may not add a real dimension; the answer depends on spin convention.
6. The proposed method may be a cached/reparameterized semicoherent matched-filter bank unless reusable `zeta` spectra create a real cost advantage.

## Derivations/Formal Points To Demand

- Define scalar objective:
  ```math
  d_\zeta^\star = \arg\min_d h_{\min}(d)
  ```
  under compute, memory, and false-alarm constraints.

- Separate:
  - physical parameters `theta`;
  - chunk-local coordinates `zeta_k(theta)`;
  - stored grid coordinates;
  - interpolation coordinates.

- Demand residual-phase mismatch derivation after projecting out nuisance directions.

- Demand cost model including:
  - spectrum construction;
  - track summation;
  - interpolation;
  - memory;
  - trials factor;
  - calibration cost.

- Demand clear distinction between:
  - representation rank;
  - local coherent search dimension;
  - physical waveform dimension;
  - operational dimension at fixed compute.

## Literature/Novelty Questions To Raise

- Is `{f, fdot, ...}` just polynomial-phase/chirp-transform analysis?
- Is `{f, beta}` a nonlinear chirp-transform specialization?
- Is the semicoherent combination essentially StackSlide/PowerFlux with richer local templates?
- Does Weave or semicoherent metric-bank literature already answer the dimension/cost question?
- Does reduced-basis/ROQ literature already provide the relevant rank-selection machinery?
- Is the only novelty the implementation layout: reusable local spectra plus track summation?

Do not add new citations beyond prior artifacts. Mark unresolved literature overlap as “needs verification.”

## Assumptions To Flag

- `t` is not a searched `zeta` dimension unless explicitly gridded.
- Frequency units may be angular frequency in local code.
- The spin model is undefined: aligned, signed aligned, independent aligned, or precessing.
- The 3.5PN phase convention is undefined.
- The mass range, frequency band, and chunk durations are undefined.
- SVD/PCA basis depends on waveform sampling, weighting, window, and nuisance projections.
- Sensitivity claims require noise model and threshold calibration, not just phase mismatch.

## Likely Failure Modes

- Extra dimensions improve recovered power but lose after trials factor and memory cost.
- SVD/PCA overfits training waveforms and fails held-out validation.
- 0PN comparison is invalid because frequency/phase/time projections differ.
- Equal-mass non-spinning case gives a trivial one-dimensional result.
- Spin result is meaningless because `chi <= 0.2` is ambiguous.
- Null distribution is assumed exponential/gamma without calibration.
- NUFFT/resampling noise transfer invalidates PSD normalization.
- 5-vector sideband covariance is ignored.
- Method is equivalent to semicoherent matched filtering with cached local filters.

## Checks Against Shallow Reasoning

- Every “better” claim must specify metric: mismatch, recovered power, runtime, or detectable amplitude.
- Every dimension claim must specify waveform family and parameter range.
- Every SVD claim must include train/test validation and projection choices.
- Every sensitivity claim must include false-alarm threshold or explain why it is preliminary.
- Every novelty claim must name the nearest known method and the claimed distinction.
- Every normalization claim must state FFT/NUFFT convention, PSD convention, and window convention.
- Every spinning claim must define the spin model.

## Final Artifact Structure

1. **Audit Metadata**
   - Attempt, sources, status, citation policy.

2. **Summary Judgment**
   - Narrow defensible framing: costed dimension selection for semicoherent chunk-local coordinates.

3. **Strongest Objections**
   - Not clearly distinct from known methods.
   - Extra dimensions may move cost elsewhere.
   - Objective underdefined unless `h_min` at fixed compute/FAP is used.
   - SVD rank is not operational search dimension.
   - Equal-mass non-spinning case may be weak evidence.
   - Spinning case currently ambiguous.
   - Analytical-basis claim may be limited to idealized models.

4. **Unclear Assumptions**
   - `zeta`, `theta`, grid, interpolation, waveform convention, spin convention, noise model, search volume.

5. **Novelty Concerns**
   - StackSlide/PowerFlux, Weave/metrics, chirp transforms, polynomial-phase transforms, reduced basis, semicoherent matched filtering.

6. **Possible Normalization Errors**
   - `2pi`, PSD/window/FFT normalization, NUFFT noise transfer, independent-bin assumption, 5-vector covariance, interpolation covariance, trials factor.

7. **Missing Tests**
   - Fixed-FAP sensitivity.
   - Runtime/memory.
   - SVD/PCA validation.
   - Equal-mass non-spinning 3.5PN comparisons.
   - Equal-mass spinning comparisons.
   - Matched-filter and chirp-transform baselines.
   - Robustness over `T_coh`, mass range, frequency band, window, noise.

8. **Is This Genuinely Different From Matched Filtering?**
   - Answer: not yet clearly.
   - Clarify conditions under which it becomes distinct: reusable lower-dimensional spectra, cheaper track summation, controlled approximation loss, better detectable amplitude at equal compute/FAP.

9. **Case-Specific Critique**
   - Non-spinning: test 0PN vs SVD after nuisance projection.
   - Spinning: define spin convention and measure added dimension.

10. **Falsification Criteria**
   - Frequency-only or semicoherent matched filtering wins at equal compute/FAP.
   - Trials factor erases gain.
   - Memory/interpolation dominates.
   - Null distribution cannot be calibrated.
   - SVD basis fails held-out tests.

11. **Referee Recommendation**
   - Proceed only as a falsifiable cost-sensitivity study.
   - Avoid novelty claims until prior-art and baseline comparisons are resolved.