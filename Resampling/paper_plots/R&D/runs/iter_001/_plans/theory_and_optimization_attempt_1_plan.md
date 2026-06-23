# Scratchpad Plan: `theory_and_optimization`

## Audit Metadata

- Agent: `theory_and_optimization`
- Attempt: 1
- Source of truth: `idea.md`, human feedback, planner output, literature output
- Intended final artifact: `runs/iter_001/02_theory_and_optimization.md`
- Status: plan only, not final artifact
- Citation policy: do not add new citations; rely only on verified literature artifact

## Core Claims To Develop

1. `theta` are physical signal parameters; `zeta` are chunk-local spectral coordinates. They are related but not identical.
2. Ordinary stack-slide is the limiting case `zeta = {f}` with segment time as a label.
3. Extra `zeta` dimensions reduce coherent mismatch but increase spectrum cost, memory, interpolation, track summation, and trials-factor thresholds.
4. The optimal `dim(zeta)` minimizes detectable amplitude at fixed compute, memory, and false-alarm probability, not residual phase alone.
5. 0PN near-optimality for equal-mass non-spinning 3.5PN waveforms is plausible but conjectural until shown by metric/SVD analysis.
6. Spin up to `chi <= 0.2` may add effective phase directions; the number required is unknown.

## Required Technical Content

- Define `theta`: intrinsic phase parameters, nuisance phase/time/amplitude parameters, and PN case parameters.
- Define `zeta_k`: local coordinates such as `{f}`, `{f, fdot}`, `{f, beta}`, PN coefficients, metric eigen-coordinates, or SVD/PCA coefficients.
- Derive `theta -> zeta_k(theta)` by locally fitting or projecting `phi(t_k + u; theta)` onto `Phi_loc(u; zeta_k)`.
- Define residual phase `R_k = phi - phi_k - Phi_loc`.
- Define coherent statistic:
  `X_k(zeta) = int W_k x exp[-i Phi_loc] dt`,
  `P_k = |X_k|^2 / sigma_k^2`.
- Define semicoherent statistic:
  `S(theta) = sum_k w_k(theta) P_k[zeta_k(theta)]`.
- Derive ideal weights, inverse-variance weighting, and note covariance-weighted vector generalization for 5-vector outputs.
- Derive null distribution under ideal complex Gaussian noise: exponential powers, Gamma sum for equal weights, weighted generalized chi-square otherwise.
- Derive signal expectation using noncentralities `lambda_k` and mismatch loss.
- Derive small-residual mismatch:
  `mu_k approx <R_k^2> - <R_k>^2`.
- Derive omitted-term scaling with `T_coh`.
- Build cost model: spectra, tracks, interpolation, memory, and threshold/trials penalty.
- Show stack-slide limit by setting `zeta = {f}`.

## Analytical Basis Questions

- For polynomial phase, formulate Gram/eigenbasis problem under a specified window.
- For PN phase, project out constant phase and frequency before comparing PN directions.
- Explain when 0PN dominance is expected and why it is not automatically proven.
- State that basis optimality depends on parameter range, `T_coh`, window, noise weighting, and sampling measure.

## 3.5PN Case Treatment

- Equal-mass non-spinning:
  - Physical intrinsic dimension is one after fixing equal mass ratio.
  - Effective local spectral dimension may still exceed one for broad ranges or long chunks.
  - Required result: formal criterion plus numerical SVD/PCA test.

- Equal-mass spinning:
  - Spin convention must be explicit; default minimal case is equal aligned spin.
  - Generic/precessing spin is outside scope unless specified.
  - Spin directions may be degenerate with mass-like directions or require new coordinates.

## Assumptions To Mark

- Gaussian stationary noise.
- Independent chunks and bins.
- Correct noise normalization.
- Small residual phase.
- Slowly varying amplitude or amplitude absorbed into weights.
- Calibratable effective trials factor.
- Interpolation error treated as mismatch.
- PN convention and valid frequency range are external inputs.

## Failure Modes And Checks

- Do not confuse `theta` dimension with useful `zeta` dimension.
- Do not call SVD/PCA “optimal” without specifying norm and sampling.
- Do not ignore trials-factor penalties.
- Do not count segment time as searched coordinate unless gridded.
- Do not assume 0PN optimality without proof.
- Do not treat `chi <= 0.2` as well-defined without spin convention.
- Check that stack-slide follows by direct substitution.
- Check every sensitivity claim includes mismatch, cost, and threshold.

## Proposed Artifact Structure

1. Audit Metadata
2. Definitions: `theta`, `zeta`, chunks
3. Mapping `theta -> zeta`
4. Coherent Chunk Statistic
5. Semicoherent Statistic And Weights
6. Null Distribution
7. Signal Expectation
8. Residual Phase And Mismatch
9. Cost Model
10. Choosing `dim(zeta)` And `T_coh`
11. Analytical Bases
12. Equal-Mass Non-Spinning 3.5PN Case
13. Equal-Mass Spinning 3.5PN Case
14. Stack-Slide Limiting Case
15. Open Problems And Numerical Checks