# Scratchpad Plan For Attempt 2

## Concrete Improvements Over Existing Output

- Remove all placeholder table entries that used `TBD`; use audit labels such as `not yet simulated`, `not yet derived`, or omit numeric tables until values exist.
- Put the human feedback at the center: the artifact must focus on choosing `dim(zeta)` for sensitivity-versus-compute tradeoff, not novelty or matched-filter optimality.
- Add stronger emphasis on two required 3.5PN studies:
  - equal-mass non-spinning
  - equal-mass spinning with `chi <= 0.2`
- Make the reduced-basis/SVD/PCA study a primary deliverable, not a secondary simulation.
- Separate established premises, conjectures, and unknowns in every major section.

## Core Claims To Preserve Or Test

- Established from `idea.md`: the goal is better semicoherent combinations than standard frequency-time track summation.
- Established from `idea.md`: richer coherent-chunk coordinates may allow longer `T_coh` but increase compute.
- Conjecture: optimal `dim(zeta)` is finite and depends on `T_coh`, mismatch tolerance, parameter-space volume, trials factor, and compute budget.
- Conjecture: for equal-mass non-spinning 3.5PN waveforms, the best one-dimensional basis may be close to the 0PN chirp direction.
- Conjecture: allowing spin up to `chi = 0.2` may introduce additional phase directions and raise the useful `dim(zeta)`.
- Unknown until literature review: whether this is already covered by polynomial-phase transforms, fast chirp transforms, Weave, StackSlide optimization, or reduced-basis methods.

## Required Derivations To Specify

- General semicoherent statistic in chunk-local coordinate space `zeta`.
- Reduction to ordinary frequency-time StackSlide when `zeta = {f}`.
- Residual-phase mismatch formula after fitting chunk-local coordinates.
- Scaling of omitted phase directions with `T_coh`.
- Dimension-selection rule using metric eigenvalues or SVD/PCA singular values.
- Cost model including spectrum generation, track summation, interpolation, memory, and threshold/trials factor.
- Sensitivity model at fixed false-alarm probability and fixed compute.
- Analytical low-dimensional basis discussion for polynomial phase and PN phase models.

## Required Literature Questions

- Do StackSlide, PowerFlux, Weave, Hough/Radon, or loosely coherent searches already optimize the number of coherent-chunk coordinates?
- Are `{f, fdot, fddot, ...}` spectra standard under fast chirp transforms, polynomial-phase transforms, or high-order ambiguity methods?
- Do reduced-basis, SVD/PCA, chirp-time, metric-coordinate, or ROQ methods provide an analytic or semi-analytic answer for low-dimensional PN phase coordinates?
- Do MBTA or multiband inspiral methods provide an applicable cost model?
- Inspect `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md`.
- Check Zotero status before saying any paper is absent.

## Required Simulations

- Toy polynomial-phase study comparing `zeta = {f}`, `{f, fdot}`, `{f, fdot, fddot}`.
- Equal-mass non-spinning 3.5PN SVD/PCA study:
  - compare 0PN direction to best one-dimensional basis
  - measure mismatch versus dimension and `T_coh`
- Equal-mass spinning 3.5PN SVD/PCA study with `chi <= 0.2`:
  - compare spinning basis to non-spinning basis
  - measure additional dimensions required
- End-to-end semicoherent cost-sensitivity comparison:
  - frequency-time baseline
  - 0PN-enhanced spectra
  - higher-PN spectra
  - SVD/PCA-coordinate spectra
- All scripts must run as `conda run -n PBH python ...`; scratch code goes under `/tmp/`.

## Assumptions To Mark Explicitly

- Phase-only mismatch is adequate for first-pass dimension selection.
- Local chunk spectra can be reused across physical templates.
- Noise is initially Gaussian and stationary unless a later simulation says otherwise.
- SVD/PCA basis depends on parameter ranges, chunk duration, inner product, and nuisance directions removed.
- Equal-mass cases reduce intrinsic dimensionality relative to generic compact-binary waveforms.
- Spin treatment must specify aligned/anti-aligned assumptions.

## Failure Modes And Checks

- Shallow novelty claim: caught by prior-art section requiring verified literature.
- Confusing `theta` with `zeta`: caught by formal definitions and mapping `theta -> zeta_k(theta)`.
- Ignoring trials factor: caught by cost/sensitivity optimization constraints.
- Reporting recovered power instead of detectable amplitude: caught by fixed false-alarm and fixed-compute metric.
- SVD basis overfitting chosen parameter range: caught by validation on held-out waveforms.
- 0PN comparison being unfair: caught by explicitly removing constant phase and frequency nuisance components before comparing basis directions.
- Spin conclusion depending on sign convention or spin alignment: caught by stating spin model and testing both signs if intended.
- Placeholder values in final artifact: avoid numeric tables unless populated; otherwise use prose stating which result is pending simulation.

## Artifact Structure

1. Scope and source of truth  
2. Central research question  
3. Established premises, conjectures, and open questions  
4. Subquestions  
5. Required derivations  
6. Required literature searches  
7. Required simulations  
8. Likely bottlenecks  
9. Decision criteria for adding a `zeta` dimension  
10. Proposed final report structure  
11. Immediate next tasks

## Sections Needing Most Technical Depth

- Residual-phase mismatch and `T_coh` scaling.
- SVD/PCA or metric-eigenbasis dimension-selection criterion.
- Cost model linking `dim(zeta)` to spectrum size, interpolation, memory, track summation, and trials factor.
- 3.5PN non-spinning basis comparison against 0PN.
- Spinning `chi <= 0.2` basis comparison.