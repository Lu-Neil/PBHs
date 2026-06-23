# Scratchpad Plan: Literature Agent Attempt 2

## Objective

Produce an auditable literature artifact that answers whether the proposed `zeta`-space semicoherent search has already been studied under another name, with special emphasis on how to choose the number of coherent-chunk dimensions for sensitivity versus compute.

Treat `idea.md` and human feedback as source of truth. Preserve useful attempt-1 findings, but extend them materially rather than polishing.

## Concrete Improvements Over Attempt 1

- Verify or explicitly mark citation uncertainty for all named methods.
- Add deeper discussion of the central question: choosing `dim(zeta)`, not merely identifying semicoherent prior art.
- Separate three levels of prior art:
  1. GW semicoherent methods.
  2. Signal-processing transforms for chirps / polynomial phase.
  3. Dimensionality reduction / reduced-basis methods.
- Add dedicated sections on equal-mass non-spinning 3.5PN and equal-mass spinning `chi <= 0.2` waveform bases.
- Discuss whether an analytical basis-selection rule exists, especially via metrics, Gram matrices, SVD/PCA, chirp-time coordinates, or orthogonalized PN phase functions.
- Strengthen novelty-risk assessment around fast chirp transform, polynomial-phase transforms, generalized Radon/Hough methods, Weave, and semicoherent template metrics.
- Inspect `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md` if possible.
- Check Zotero status if possible; otherwise mark all as `Zotero status unknown`.

## Core Claims To Evaluate

- Established: StackSlide, PowerFlux, Hough/Radon, Weave, and semicoherent metrics already cover broad classes of semicoherent track accumulation.
- Established: Fast chirp transform, polynomial-phase transform, high-order ambiguity functions, and chirplet transforms are likely close signal-processing analogues to higher-dimensional `zeta` spectra.
- Established: ROQ, reduced basis, SVD/PCA, chirp-time coordinates, and template-bank metrics provide ways to find low-dimensional waveform coordinates.
- Conjecture: The project’s novelty may lie in a costed rule for choosing the dimension and basis of reusable coherent-chunk spectra, not in semicoherent summing itself.
- Conjecture: For equal-mass non-spinning 3.5PN waveforms, the best one-dimensional chunk-local coordinate may be close to the 0PN direction, but this must be tested.
- Conjecture: Allowing aligned spin up to `chi = 0.2` may increase effective phase dimension or introduce a spin-like basis direction.

## Required Literature Questions

- StackSlide: how are coherent time, mismatch, template counts, and computing cost optimized?
- PowerFlux: how are noise and antenna-pattern weights incorporated, and does this imply a better statistic than naive power sums?
- Weave: does it already solve the semicoherent coordinate-count or metric-rank problem?
- MBTA: does multiband filtering provide an analogue of varying local model dimension across frequency/time?
- Reduced order quadrature: does basis-size selection map onto `zeta` dimension selection, or only matched-filter acceleration?
- PCA / dimensionality reduction: what existing GW work uses SVD/PCA bases for template compression or reduced coordinates?
- Semicoherent template metrics: can metric eigenvalues determine when a coordinate should be retained?
- Signal-processing analogues: do fast chirp transforms, polynomial-phase transforms, high-order ambiguity functions, chirplet transforms, or generalized Radon transforms already define spectra over `{f, fdot, fddot, ...}` and provide model-order selection rules?
- Novelty risks: which prior art could make `{f,t,beta}` or `{f, fdot}` spectra non-novel?

## Assumptions To State

- `zeta` means chunk-local coherent coordinates, not necessarily physical parameters.
- Segment time may be a label rather than a searched coordinate unless explicitly gridded.
- The goal is not full matched-filter optimality; it is better sensitivity at fixed compute, memory, and false-alarm probability.
- Literature alone cannot determine whether 0PN is the best one-dimensional basis for this project’s waveform range.
- Any statement about spinning 3.5PN waveforms depends on spin convention, approximant, parameter range, and inner product.

## Failure Modes

- Claiming novelty for semicoherent track summing.
- Treating `theta` and `zeta` as the same parameter space.
- Ignoring trials factor and memory cost when adding dimensions.
- Calling SVD/PCA “optimal” without specifying the norm, waveform range, and projected nuisance directions.
- Treating 0PN near-optimality as established without calculation.
- Citing Weave, PowerFlux, MBTA, or polynomial-phase methods without verification.
- Overlooking signal-processing names for the same idea.

## Checks Against Shallow Reasoning

- Every cited work must be either verified or marked as requiring verification.
- Every section should say whether the result is established, inferred, or conjectural.
- The artifact must explicitly answer: “Does this choose `dim(zeta)`?”
- The artifact must include a novelty-risk table.
- The artifact must distinguish reusable higher-dimensional spectra from ordinary physical-parameter template banks.
- The artifact must connect literature to the two required waveform cases:
  - equal-mass non-spinning 3.5PN,
  - equal-mass spinning up to `chi = 0.2`.

## Proposed Artifact Structure

1. **Audit Metadata**
   - Source of truth, attempt number, tool/browsing/Zotero/CODEX status.

2. **Executive Summary**
   - Main answer: this is likely a cost-constrained coordinate-design problem with substantial prior-art overlap.

3. **Terminology Map**
   - `theta`, `zeta`, coherent chunk, semicoherent track, spectrum dimension, metric dimension.

4. **GW Semicoherent Prior Art**
   - StackSlide.
   - PowerFlux.
   - Weave.
   - Hough/Radon.
   - Loosely coherent searches.
   - Semicoherent template metrics.

5. **CBC / Inspiral Prior Art**
   - MBTA.
   - Multiband/multirate filtering.
   - Chirp-time coordinates.
   - Compact-binary template-bank metrics.

6. **Reduced-Basis And Dimensionality Reduction**
   - ROQ.
   - Reduced basis.
   - PCA/SVD waveform compression.
   - How these relate to choosing `zeta`.

7. **Signal-Processing Analogues**
   - Fast chirp transform.
   - Polynomial-phase transform.
   - High-order ambiguity function.
   - Chirplet transforms.
   - Generalized Radon/Hough transforms.
   - Model-order/dimension-selection literature if found.

8. **Choosing `dim(zeta)`**
   - Literature-derived principles.
   - Metric eigenvalues.
   - SVD/PCA truncation.
   - Compute, memory, interpolation, and trials-factor penalties.
   - Distinguish representation error from detection sensitivity.

9. **3.5PN Equal-Mass Non-Spinning Case**
   - What literature suggests.
   - Why 0PN may dominate.
   - Why this remains empirical unless SVD/metric analysis is run.

10. **3.5PN Equal-Mass Spinning Case**
   - Relevant spin-phasing prior art.
   - Whether spin likely adds a new dimension.
   - Required assumptions for `chi <= 0.2`.

11. **Novelty Risks**
   - Ranked table: method, overlap, risk level, what must be checked.

12. **Candidate Papers To Verify / Add To Zotero**
   - Grouped by StackSlide, PowerFlux, Weave, MBTA, ROQ, PCA/SVD, semicoherent metrics, signal processing.

13. **Recommended Next Actions**
   - Full-paper checks.
   - Zotero check.
   - SVD/PCA study targets.
   - Analytical metric/Gram-matrix derivation targets.