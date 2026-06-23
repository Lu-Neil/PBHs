# Literature Scratchpad Plan: `literature`

## Core Questions

- What existing names cover “higher-dimensional coherent-chunk spectra plus semicoherent track summing”?
- Do existing methods answer how to choose `dim(zeta)` at fixed sensitivity/computing cost?
- Is this already covered by StackSlide, PowerFlux, Weave, MBTA, semicoherent template metrics, fast chirp/polynomial-phase transforms, ROQ, or PCA/SVD basis methods?
- For equal-mass non-spinning 3.5PN waveforms, is the 0PN coordinate known or expected to be the best one-parameter chunk coordinate?
- For equal-mass spinning 3.5PN waveforms with `chi <= 0.2`, does spin add genuinely new effective dimensions?

## Required Literature Targets

- **StackSlide**: coherent time optimization, semicoherent summing, template-count and mismatch tradeoffs.
- **PowerFlux**: weighted power accumulation, antenna/noise weighting, relevance beyond frequency-time spectra.
- **Weave**: semicoherent metric template banks, lattice placement, cost and mismatch control.
- **MBTA / multiband searches**: splitting inspiral information across bands/chunks; relation to local coordinate choice.
- **Reduced order quadrature / reduced basis**: whether bases only accelerate likelihoods or can define reusable `zeta` spectra.
- **PCA / dimensionality reduction**: waveform compression, phase bases, chirp-time coordinates, metric eigenvectors.
- **Semicoherent template metrics**: dimension, projection, mismatch accumulation, template-count scaling.
- **Signal-processing analogues**: fast chirp transform, polynomial-phase transform, high-order ambiguity function, chirplet transforms, Hough/Radon methods.
- **Novelty risks**: cases where the proposed method is an existing transform or semicoherent template-bank construction under different notation.

## Claims To Verify

- Established only if sourced: semicoherent GW searches already optimize `T_coh`, mismatch, template counts, and compute.
- Established only if sourced: Weave is a close analogue for metric-based semicoherent search design.
- Established only if sourced: fast chirp/polynomial-phase methods already construct spectra over chirp parameters.
- Conjecture until checked: the possible contribution is a costed rule for choosing coherent-chunk coordinate dimension and basis.
- Conjecture until checked: 0PN is near-optimal as a one-parameter coordinate for equal-mass non-spinning 3.5PN chunks.
- Conjecture until checked: spin up to `chi = 0.2` adds at least one useful phase direction.

## Extraction Table Fields

For each paper/method record:

- Method name
- Full verified citation
- DOI/arXiv/source link
- Problem solved
- Coordinates or parameters searched
- Treatment of `T_coh`
- Mismatch/metric model
- Compute/memory scaling
- False-alarm or trials-factor handling
- Relation to `zeta`-space spectra
- Zotero status
- Novelty risk

## Checks Against Shallow Reasoning

- Do not equate “semicoherent” with “answers dimension selection.”
- Distinguish physical parameters `theta` from chunk-local coordinates `zeta`.
- Distinguish reduced-basis likelihood acceleration from reusable intermediate spectra.
- Distinguish matched-filter optimality from compute-limited sensitivity tradeoffs.
- Mark unsupported claims explicitly.
- Do not infer 3.5PN or spin conclusions from unrelated waveform-compression papers without qualification.

## Final Artifact Structure

1. Executive summary: closest prior art and likely novelty risks.
2. Method comparison table: StackSlide, PowerFlux, Weave, MBTA, ROQ, PCA/SVD, semicoherent metrics, signal-processing analogues.
3. Semicoherent GW prior art: highest technical depth.
4. Inspiral/multiband prior art: MBTA, chirp-time, metric coordinates.
5. Reduced basis, ROQ, PCA/SVD: what they optimize and whether they define candidate `zeta` axes.
6. Signal-processing analogues: fast chirp, polynomial phase, ambiguity, chirplet, Hough/Radon.
7. Implications for choosing `dim(zeta)`: established rules versus open gaps.
8. 3.5PN relevance: equal-mass non-spinning and spinning up to `chi = 0.2`.
9. Novelty risks and defensible framing.
10. Papers to add/check in Zotero.

## Improvements Over Previous Report

- Focus on how prior work chooses the number of coherent-chunk dimensions.
- Treat 0PN near-optimality as an open literature question.
- Add explicit non-spinning and spinning 3.5PN relevance.
- Give signal-processing analogues equal priority with GW prior art.
- Require Zotero status and verified citations for auditability.