# Scratchpad Plan: `theory_and_optimization` Attempt 2

## Artifact Goal

Produce `runs/iter_001/02_theory_and_optimization.md` as a theory artifact, not a final report. It should formalize how to choose the spectral coordinate dimension `dim(zeta)` and coherent time `T_coh` for semicoherent searches, with explicit sensitivity-versus-cost criteria.

The artifact must preserve useful attempt-1 material, but extend it toward the human feedback: the key question is not whether semicoherent searches exist, but how many coherent-chunk coordinates should be used, and whether 0PN is close to the best one-parameter coordinate for 3.5PN waveforms.

## Concrete Improvements Over Existing Output

1. Make the dimension-selection rule more explicit:
   ```math
   d_\zeta^\star(T_{\rm coh}) = \arg\min_d h_{\min}(d,T_{\rm coh})
   ```
   under compute, memory, false-alarm, and mismatch constraints.

2. Separate three decision layers:
   - phase-approximation dimension from residual mismatch;
   - grid/search dimension from template-count and trials-factor penalties;
   - optimal operational dimension from detectable-amplitude minimization.

3. Add a clearer analytical framework for lower-dimensional bases:
   - metric eigenbasis;
   - SVD/PCA basis;
   - PN basis;
   - physically motivated 0PN basis.

4. Explicitly state that “best one-dimensional basis” is norm-, range-, and projection-dependent.

5. Add a more direct answer to:
   - why 0PN may be near-optimal;
   - why it may fail after projecting out phase/frequency;
   - how spinning waveforms can increase effective dimension.

6. Strengthen the limiting StackSlide section so it clearly recovers ordinary frequency-time spectra.

7. Mark conjectures and assumptions more visibly.

## Core Claims To Make

### Established / Derived

- `theta` are physical waveform parameters; `zeta` are coherent-chunk spectral coordinates.
- A template `theta` maps to a track `zeta_k(theta)` through chunk spectra.
- Ordinary StackSlide is recovered when `zeta = {f}` and chunk time is only a segment label.
- Small residual phase mismatch is approximately the weighted phase variance:
  ```math
  \mu_k \approx \langle R_k^2\rangle_k - \langle R_k\rangle_k^2.
  ```
- Adding a `zeta` dimension is useful only if reduced mismatch improves final detectable amplitude after cost and threshold penalties.
- Ideal scalar normalized powers have exponential null distributions; equal-weight sums are Gamma distributed.

### Conjectures

- The optimal `dim(zeta)` increases with `T_coh`.
- For equal-mass non-spinning 3.5PN waveforms, 0PN may be close to the best one-dimensional coordinate over restricted ranges.
- After projecting out constant phase and frequency, the dominant residual direction is not guaranteed to be 0PN.
- Equal aligned spin up to `chi = 0.2` may add one dominant phase direction, but this must be tested.
- There may be a compute-limited regime where `{f,beta}` or an SVD/PCA coordinate outperforms frequency-only StackSlide.

## Required Derivations

1. Coherent chunk statistic:
   ```math
   X_k(\zeta)=\int_{I_k}dt\,W_k(t)x_a(t)e^{-i\Phi_{\rm loc}(t-t_k;\zeta)}.
   ```

2. Semicoherent statistic:
   ```math
   \mathcal S(\theta)=\sum_k w_k P_k[\zeta_k(\theta)].
   ```

3. Projection definition of `theta -> zeta_k(theta)`:
   ```math
   (\phi_{0,k},\zeta_k)=\arg\min_{\phi_0,\zeta}
   \|\phi(t_k+u;\theta)-\phi_0-\Phi_{\rm loc}(u;\zeta)\|_k^2.
   ```

4. Residual mismatch:
   ```math
   \mu_k=1-\left|\langle e^{iR_k}\rangle_k\right|^2
   \approx \mathrm{Var}_k(R_k).
   ```

5. Omitted-term scaling:
   if `R_k(u) ~= a_m u^m`, then
   ```math
   \mu_k \propto a_m^2 T_{\rm coh}^{2m}.
   ```

6. Weighted statistic null:
   ```math
   E_0[\mathcal S]=\sum_k w_k,
   \qquad
   \mathrm{Var}_0[\mathcal S]=\sum_k w_k^2.
   ```

7. Signal expectation:
   ```math
   E_1[\mathcal S]-E_0[\mathcal S]=\sum_k w_k\lambda_k.
   ```

8. Cost model:
   ```math
   C_{\rm total}=C_{\rm spectra}+C_{\rm tracks}+C_{\rm interp}+C_{\rm memory}+C_{\rm threshold}.
   ```

9. Optimization criterion:
   ```math
   (d_\zeta^\star,T_{\rm coh}^\star,\mathcal B^\star)
   =
   \arg\min h_{\min}
   ```
   subject to compute, memory, false-alarm, and mismatch constraints.

## Literature Questions To Defer

Do not invent citations. The theory artifact may refer to the literature artifact only as prior audit context.

Questions to flag for literature follow-up:

- Does polynomial-phase transform literature already provide model-order selection criteria?
- Does Weave or semicoherent metric literature already imply a comparable dimension rule?
- Are chirp-time or metric coordinates analytically equivalent to the proposed optimal `zeta` basis in the PN case?
- Is `{f,beta}` equivalent to a known fast chirp transform specialization?

## Assumptions

- Noise is initially Gaussian and stationary for analytic derivations.
- Chunk powers are independent in the ideal null model.
- Amplitude variation inside a coherent chunk is slow or absorbed into weights.
- `zeta` grids are fine enough that interpolation can be treated as an added mismatch term.
- The 5-vector / NUFFT covariance problem is acknowledged but not solved here.
- Equal-mass non-spinning 3.5PN has one intrinsic physical dimension, but may have higher effective local phase dimension over finite chunks.
- Spinning case must define spin convention before quantitative conclusions.

## Likely Failure Modes

- Confusing physical dimension `dim(theta)` with spectral dimension `dim(zeta)`.
- Claiming 0PN is optimal without specifying inner product, projection, mass range, frequency range, and `T_coh`.
- Optimizing recovered power instead of detectable amplitude at fixed false alarm and compute.
- Ignoring trials-factor penalties from higher-dimensional spectra.
- Ignoring memory cost, which may dominate before FLOPs.
- Treating ideal Gamma null distributions as valid for NUFFT/resampled/gapped real data.
- Treating segment time as a searched `zeta` coordinate when it is only a chunk label.

## Checks Against Shallow Reasoning

- Every proposed extra coordinate must pass: mismatch gain, grid cost, memory cost, interpolation cost, trials factor, final `h_min`.
- Every “optimal basis” statement must identify the norm and projected-out nuisance directions.
- Every 3.5PN statement must be marked conjectural unless supported by future SVD/PCA or metric calculation.
- StackSlide must appear as an explicit limiting case.
- Null and signal distributions must be separated.
- Weights must be included, not only unweighted sums.
- Spin conclusions must remain conditional on aligned/equal/signed spin assumptions.

## Artifact Structure

1. **Audit Metadata**
   - Source of truth, attempt number, status, no-new-citations note.

2. **Executive Summary**
   - Main decision rule: add dimensions only if final sensitivity improves at fixed compute and false alarm.

3. **Definitions**
   - `theta`, `zeta`, `T_coh`, chunk index, `Phi_loc`, basis `B`.

4. **Mapping From `theta` To `zeta`**
   - Projection formulation and derivative/Taylor alternative.

5. **Coherent Chunk Statistic**
   - Scalar statistic and 5-vector/covariance-weighted generalization.

6. **Semicoherent Statistic And Weights**
   - Weighted power sum; weak-signal optimal weighting intuition.

7. **Null Distribution**
   - Exponential, Gamma, weighted hypoexponential/Gaussian approximation, trials factor.

8. **Signal Expectation**
   - Noncentrality, mismatch loss, amplitude scaling.

9. **Residual Phase And Mismatch**
   - Small-phase derivation and omitted-term scaling. High technical depth.

10. **Dimension Selection**
   - Metric/SVD/PCA criterion, then costed search criterion. High technical depth.

11. **Sensitivity-Cost Tradeoff**
   - Cost decomposition, memory, interpolation, threshold, `h_min`. High technical depth.

12. **Choosing `T_coh`**
   - Mismatch-limited and compute-limited coherent time.

13. **Analytical Basis Construction**
   - Polynomial phase, PN basis, metric eigenbasis, 0PN discussion. High technical depth.

14. **3.5PN Equal-Mass Non-Spinning Case**
   - One physical dimension but possible higher effective local dimension; 0PN conjecture and test.

15. **3.5PN Equal-Mass Spinning Case**
   - Spin convention required; possible added dimension; mark uncertain.

16. **Limiting Case: Ordinary StackSlide**
   - Explicit recovery with `zeta = {f}`.

17. **Decision Rule Checklist**
   - Operational procedure for future simulations.

18. **Established / Derived / Conjectural / Open**
   - Audit clarity.