# Literature Review: Higher-Dimensional Semicoherent Searches and `zeta`-Space Coordinates

## Audit Metadata

- Source of truth: `idea.md` plus the attempt feedback in this run.
- Required output: `/home/neil-lu/Dropbox/PBHs/Codebase/Resampling/paper_plots/R&D/runs/iter_001/01_literature_review.md`
- Attempt: 1 for this agent.
- External access status: this artifact is based on prior domain knowledge and the supplied prompt. I did not browse, run shell commands, inspect `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md`, or check Zotero because the wrapper explicitly instructed this agent not to use tools.
- Zotero status: unknown for every citation below.
- Citation policy: citations are included only where I can identify real works with high confidence. Items marked `citation needed` require verification by a later literature pass.

## Executive Summary

The idea in `idea.md` is best framed as a **cost-constrained semicoherent coordinate-design problem**:

> Given a modeled signal family `theta -> phi(t; theta)`, choose the number and form of chunk-local coherent coordinates `zeta` so that semicoherent power accumulation gives the best sensitivity at fixed compute, memory, and false-alarm probability.

This problem overlaps strongly with established work under several names:

- **StackSlide / Hough / PowerFlux** in continuous-wave gravitational-wave searches.
- **Semicoherent template metrics** and metric-based template-bank placement.
- **Weave**, which is likely the closest gravitational-wave search framework because it explicitly combines coherent segments, semicoherent metrics, template placement, and computing-cost constraints.
- **Fast chirp transform**, **polynomial-phase transform**, **high-order ambiguity function**, **chirplet transforms**, and **Radon/Hough transforms** in signal processing.
- **Chirp-time coordinates**, **PCA/SVD waveform bases**, **reduced basis**, and **reduced order quadrature** in compact-binary matched filtering and waveform modeling.
- **MBTA / multiband inspiral searches**, which split inspiral filtering across frequency bands and use chirp-time or related coordinates, but are not obviously the same as reusable high-dimensional `zeta` spectra.

The likely novelty is not “semicoherent search” or “higher-dimensional spectral accumulation.” Those are established. The defensible gap is narrower:

> A systematic rule for choosing `dim(zeta)` and the basis of `zeta` for modeled chirping signals, including spectrum-generation cost, track-summing cost, interpolation, memory, trials factors, coherent mismatch, and sensitivity at fixed compute.

A later report should avoid claiming novelty until the fast chirp transform, polynomial-phase transform, Weave, and semicoherent metric literature are checked in full.

## Method Comparison Table

| Topic | Established result | Relation to `zeta` idea | Does it choose `dim(zeta)`? | Novelty risk |
|---|---|---|---|---|
| StackSlide | Semicoherent sum of coherent powers along template tracks; optimized coherent time and mismatch have been studied. | `zeta = {f}` or frequency-like local coordinates, with tracks through time. | Usually chooses `T_coh`, mismatch, and template grids, not arbitrary local coordinate dimension. | High for frequency-time track sums. |
| PowerFlux | Weighted semicoherent power sum using detector antenna pattern and noise weights. | Replaces naive equal-weight track power by weighted accumulation. | No evidence that it optimizes arbitrary coherent-coordinate dimension. | Moderate: weighting/statistic design is prior art. |
| Weave | Semicoherent continuous-wave framework using metric template banks and optimal lattice placement. | Very close to costed semicoherent template-bank design. | Likely optimizes grids in physical parameter space, not necessarily reusable higher-dimensional chunk spectra. Needs verification. | High. |
| MBTA | Multiband compact-binary inspiral filtering splits waveform/filtering across frequency bands. | Related to choosing different local representations across signal evolution. | Not obviously a rule for `dim(zeta)` spectra. | Moderate. |
| Reduced order quadrature | Accelerates likelihood/matched filtering using reduced bases and empirical interpolation. | Provides low-dimensional waveform bases; candidate way to define `zeta` axes. | Chooses basis size for waveform accuracy, not semicoherent spectrum dimension at fixed search cost. | Moderate. |
| PCA/SVD dimensionality reduction | Compresses waveform/template spaces and reveals dominant phase directions. | Directly relevant to choosing low-dimensional chunk coordinates. | Usually based on representation error or metric eigenvalues, not full semicoherent cost. | Moderate to high. |
| Semicoherent template metrics | Predict mismatch and template counts for coherent/semicoherent searches. | Natural mathematical tool for deciding which directions matter over `T_coh`. | Gives metric dimension/rank and template density, but may not optimize reusable spectra. | High. |
| Signal-processing analogues | Chirp/polynomial-phase transforms produce spectra over frequency derivatives or chirp parameters. | Very close to `zeta = {f, fdot, fddot, ...}` spectra. | Polynomial order/model order selection exists in parts of this literature; exact cost-sensitivity rule must be checked. | Very high. |

## Semicoherent GW Prior Art

### StackSlide

**Established.** StackSlide is a standard semicoherent continuous-wave method: compute coherent statistics over shorter segments, then sum powers or detection statistics along a template-predicted frequency evolution. It is directly related to the baseline described in `idea.md`.

High-confidence citations:

- Brady, P. R. and Creighton, T., “Searching for periodic sources with LIGO. II. Hierarchical searches,” *Physical Review D* 61, 082001, 2000. DOI: `10.1103/PhysRevD.61.082001`. Zotero status unknown.
- Prix, R. and Shaltev, M., “Search for continuous gravitational waves: optimal StackSlide method at fixed computing cost,” *Physical Review D* 85, 084010, 2012. DOI: `10.1103/PhysRevD.85.084010`. Zotero status unknown.

**Relation to `zeta`.** Ordinary StackSlide corresponds to a chunk-local spectrum indexed mainly by frequency, with time represented by the segment index. The physical template predicts a frequency bin in each segment.

**What it likely answers.** StackSlide optimization literature addresses `T_coh`, number of segments, mismatch, template counts, and fixed computing cost. This is directly relevant to the user’s sensitivity-versus-compute question.

**What it probably does not fully answer.** It does not obviously ask whether the coherent chunk output should be a spectrum over `{f}`, `{f, fdot}`, `{f, beta}`, `{f, beta_1, beta_2}`, or a PCA basis. That is the gap to verify.

**Novelty risk.** Any claim of “new semicoherent summing along tracks” is not defensible. The possible novelty is choosing a richer reusable local coordinate space and optimizing its dimension.

### PowerFlux

**Established.** PowerFlux is a semicoherent continuous-wave method that uses weighted sums of power, accounting for detector noise and antenna-pattern modulation.

Representative citations/search targets:

- Abbott et al., LIGO all-sky continuous-wave PowerFlux searches, e.g. early S5/S6 all-sky CW search papers. Exact PowerFlux methods paper citation needed.
- Dergachev, V., PowerFlux / loosely coherent search papers. Exact citations needed. Zotero status unknown.

**Relation to `zeta`.** PowerFlux warns against treating all segment powers equally. For the proposed `zeta` spectra, the semicoherent combination should probably include noise and antenna weights, and for the local 5-vector pipeline possibly a covariance-weighted statistic rather than a scalar equal-weight power sum.

**Does it choose `dim(zeta)`?** No verified evidence here. It is more about weighting, robustness, upper limits, and semicoherent detection statistics than choosing the dimensionality of local coherent spectra.

**Novelty risk.** If the proposed method uses weighted power accumulation, that part is not novel. The coordinate-dimension selection question remains potentially distinct.

### Weave

**Established with citation gap.** Weave is a semicoherent continuous-wave search framework used in modern all-sky or directed CW searches. It uses metric template banks and lattice placement. Exact citation needed.

Search targets:

- Wette, K. and collaborators, Weave semicoherent CW search implementation.
- LIGO/Virgo continuous-wave papers using Weave.
- Papers on `lalpulsar` Weave, semicoherent template banks, and supersky metrics.

Zotero status unknown.

**Relation to `zeta`.** Weave is one of the closest GW analogues because it formalizes semicoherent metric placement and computing tradeoffs. It may already contain the machinery needed to choose how many physical dimensions to search and how finely.

**Important distinction.** Weave likely grids physical CW parameters such as sky position, frequency, and spin-down parameters. The proposed `zeta` idea instead asks whether to precompute reusable coherent spectra in a chosen local coordinate basis that may be lower-dimensional or transformed relative to physical parameters. Whether this is genuinely different from Weave’s semicoherent template-bank machinery requires a careful reading.

**Novelty risk.** High. A later agent should inspect Weave first.

### Semicoherent Template Metrics

**Established.** Metric methods approximate fractional loss in detection statistic due to parameter mismatch. They are central to template-bank construction in coherent and semicoherent GW searches.

High-confidence citations:

- Owen, B. J., “Search templates for gravitational waves from inspiraling binaries: Choice of template spacing,” *Physical Review D* 53, 6749, 1996. DOI: `10.1103/PhysRevD.53.6749`. Zotero status unknown.
- Balasubramanian, R., Sathyaprakash, B. S., and Dhurandhar, S. V., “Gravitational waves from coalescing binaries: Detection strategies and Monte Carlo estimation of parameters,” *Physical Review D* 53, 3033, 1996. DOI likely `10.1103/PhysRevD.53.3033`; verify. Zotero status unknown.
- Owen, B. J. and Sathyaprakash, B. S., “Matched filtering of gravitational waves from inspiraling compact binaries: Computational cost and template placement,” *Physical Review D* 60, 022002, 1999. DOI: `10.1103/PhysRevD.60.022002`. Zotero status unknown.
- Pletsch, H. J., semicoherent CW metric papers, including work around 2010 on parameter-space correlations and semicoherent searches. Exact citation details should be verified. Zotero status unknown.
- Prix, R., “Search for continuous gravitational waves: Metric of the multidetector F-statistic,” *Physical Review D* 75, 023004, 2007. DOI: `10.1103/PhysRevD.75.023004`. Zotero status unknown.

**Relation to `zeta`.** The metric gives a local quadratic mismatch model. Its eigenvalues and eigenvectors can define important and unimportant phase directions. This is directly relevant to choosing the number of `zeta` dimensions.

**Established implication.** For small mismatches, directions with large metric eigenvalues require finer grids or explicit coordinates; small-eigenvalue directions can be neglected, projected out, or treated semicoherently.

**Open issue.** The metric alone selects dimensions by mismatch, not by full costed sensitivity. The user’s question requires adding compute, memory, interpolation, and trials factors.

## MBTA and Multiband Inspiral Searches

### MBTA

**Established.** MBTA stands for Multi-Band Template Analysis and is used in compact-binary coalescence searches, especially low-latency pipelines. It splits the matched filtering into frequency bands to reduce computational cost.

Representative citations/search targets:

- Beauville et al., early MBTA paper. Exact title and bibliographic details needed.
- Adams et al., “Low-latency analysis pipeline for compact binary coalescences in the advanced gravitational wave detector era,” *Classical and Quantum Gravity* 33, 175012, 2016. DOI: `10.1088/0264-9381/33/17/175012`. This paper discusses low-latency CBC pipelines; verify exact MBTA content. Zotero status unknown.

**Relation to `zeta`.** MBTA is relevant because it changes the computational representation of modeled inspiral signals. It effectively exploits the fact that different frequency bands contribute differently to the phase and SNR.

**Not the same problem.** MBTA is closer to a multiband matched-filter acceleration than a general semicoherent sum of powers through a higher-dimensional local spectrum. It should be reviewed for cost models and coordinate choices, but it is not obviously a direct solution to choosing `dim(zeta)`.

**Novelty risk.** Moderate. The risk is mostly around cost-efficient decomposition of chirping signals, not around high-dimensional semicoherent track sums.

### Chirp-Time Coordinates

**Established.** Chirp-time coordinates are used to parameterize inspiral template banks more conveniently than raw masses. They are physically motivated low-dimensional coordinates for PN phase evolution.

Representative citations/search targets:

- Sathyaprakash, B. S. and Dhurandhar, S. V., “Choice of filters for the detection of gravitational waves from coalescing binaries,” *Physical Review D* 44, 3819, 1991. DOI: `10.1103/PhysRevD.44.3819`. Zotero status unknown.
- Owen and Sathyaprakash 1999, cited above.
- Tanaka and Tagoshi, template bank coordinate papers. Citation needed.

**Relation to `zeta`.** Chirp-time coordinates are a strong precedent for replacing raw physical parameters with combinations that better describe waveform phase. They may be the compact-binary analogue of choosing `beta` or PN-combination coordinates for `zeta`.

**Implication for 0PN.** The 0PN chirp term is physically dominant in inspiral phase at low PN order. However, the statement “0PN is the best one-dimensional chunk coordinate for equal-mass non-spinning 3.5PN waveforms” is not established by this literature review. It is a conjecture to test via metric eigenvectors or SVD/PCA over the project’s mass/frequency/chunk-duration range.

## Reduced Basis, ROQ, PCA, and Dimensionality Reduction

### Reduced Basis and Reduced Order Quadrature

**Established.** Reduced-basis methods and reduced order quadrature compress families of gravitational waveforms and accelerate likelihood evaluation.

High-confidence citations:

- Field, S. E., Galley, C. R., Hesthaven, J. S., Kaye, J., and Tiglio, M., “Fast prediction and evaluation of gravitational waveforms using surrogate models,” *Physical Review X* 4, 031006, 2014. DOI: `10.1103/PhysRevX.4.031006`. Zotero status unknown.
- Canizares, P., Field, S. E., Gair, J. R., and Tiglio, M., “Gravitational wave parameter estimation with compressed likelihood evaluations,” *Physical Review D* 87, 124005, 2013. DOI: `10.1103/PhysRevD.87.124005`. Zotero status unknown.
- Canizares, P., Field, S. E., Gair, J. R., Raymond, V., Smith, R., and Tiglio, M., “Accelerated gravitational wave parameter estimation with reduced order modeling,” *Physical Review Letters* 114, 071104, 2015. DOI: `10.1103/PhysRevLett.114.071104`. Zotero status unknown.
- Pürrer, M., “Frequency domain reduced order models for gravitational waves from aligned-spin compact binaries,” *Classical and Quantum Gravity* 31, 195010, 2014. DOI: `10.1088/0264-9381/31/19/195010`. Zotero status unknown.

**Relation to `zeta`.** Reduced bases identify low-dimensional waveform subspaces. In principle, basis coefficients or empirical interpolation nodes could inspire `zeta` coordinates.

**Key distinction.** ROQ typically accelerates likelihood evaluations for matched filtering or parameter estimation. It does not automatically create reusable coherent spectra indexed by basis coefficients, nor does it directly optimize semicoherent power summing.

**Use for this project.** ROQ/reduced basis gives algorithms for selecting dimensions from waveform approximation error. The project needs to extend that to a search statistic and cost model.

### PCA / SVD Waveform Compression

**Established.** SVD/PCA has been used to compress template banks, reduce matched-filter cost, and identify dominant waveform directions.

Representative citations/search targets:

- Cannon et al., SVD compression of compact-binary template banks. Exact paper citation needed.
- Cannon et al., “Toward Early-Warning Detection of Gravitational Waves from Compact Binary Coalescence,” *The Astrophysical Journal* 748, 136, 2012. DOI: `10.1088/0004-637X/748/2/136`; verify relevance to SVD compression. Zotero status unknown.
- Roulet et al., SVD/reduced template-bank methods for gravitational waves. Exact citation needed.
- Brown et al. or Cannon et al. low-latency filtering/SVD papers. Citation needed.

**Relation to `zeta`.** SVD/PCA is directly relevant to the feedback item:

> Run reduced-basis, SVD/PCA analysis on equal mass-ratio, non-spinning 3.5PN waveforms. How different are the optimal bases from the 0PN waveform?

A PCA/SVD basis can answer whether the leading residual phase direction is aligned with the 0PN phase direction after nuisance terms such as constant phase and frequency are removed.

**Established vs conjecture.**

- Established: SVD/PCA can find low-rank approximations to waveform families.
- Conjecture: for equal-mass non-spinning 3.5PN phase evolution over the project’s frequency and mass range, the first SVD direction is close to the 0PN coordinate.
- Conjecture: allowing spin up to `chi = 0.2` adds at least one new significant direction.

**Important caution.** The optimal PCA basis depends on:
- mass range,
- frequency range,
- chunk duration,
- time or frequency-domain convention,
- whitening/noise weighting,
- whether amplitude is included,
- whether phase, frequency, and coalescence time are projected out,
- sampling distribution over parameter space.

A literature citation alone cannot settle the 0PN-vs-SVD question without reproducing the basis under this project’s assumptions.

## Signal-Processing Analogues

### Fast Chirp Transform

**Established.** The fast chirp transform computes transforms over chirp-like phase parameters more efficiently than brute-force searches in some polynomial or chirp families.

High-confidence citation:

- Jenet, F. A. and Prince, T. A., “Detection of variable frequency signals using a fast chirp transform,” *Physical Review D* 62, 122001, 2000. DOI: `10.1103/PhysRevD.62.122001`. Zotero status unknown.

**Relation to `zeta`.** This is a very close analogue to spectra indexed by chirp parameters. A `zeta = {f, fdot}` or `{f, beta}` coherent spectrum may be a variant of a chirp transform.

**Novelty risk.** Very high. Any claim that higher-dimensional chirp-coordinate spectra are new must be checked against this literature.

**Gap.** Fast chirp transform papers may focus on fully coherent detection or transform algorithms, not semicoherent chunk dimension selection at fixed compute. This distinction should be preserved.

### Polynomial-Phase Transform and High-Order Ambiguity Function

**Established.** Polynomial-phase signals are often analyzed by transforms indexed by polynomial phase coefficients, including high-order ambiguity functions.

Representative citations/search targets:

- Peleg, S. and Porat, B., polynomial-phase signal parameter estimation and high-order ambiguity function papers. Exact citations needed.
- Barbarossa, S., Scaglione, A., and Giannakis, G. B., product high-order ambiguity function / polynomial-phase signal papers. Exact citations needed.
- Boashash, B., time-frequency signal analysis references on polynomial-phase signals. Citation needed.

**Relation to `zeta`.** This literature is almost exactly the generic signal-processing version of choosing `zeta = {f, fdot, fddot, ...}`.

**Important question for next pass.** Does this literature provide model-order selection rules or cost-constrained dimension rules? If yes, it may already answer part of the user’s central question.

### Chirplet Transforms

**Established.** Chirplet transforms generalize wavelet/time-frequency methods to chirping atoms. They provide overcomplete dictionaries indexed by time, frequency, chirp rate, duration, and sometimes higher-order parameters.

Representative citations/search targets:

- Mann, S. and Haykin, S., “The chirplet transform: physical considerations,” citation needed.
- Baraniuk and Jones, chirplet transform papers. Citation needed.

**Relation to `zeta`.** Chirplet dictionaries are an established way to create higher-dimensional time-frequency-chirp representations. They are a strong novelty risk for the generic signal-processing framing.

**Difference.** Chirplet methods are often dictionary/time-frequency analysis tools, not necessarily cost-optimized semicoherent track-summing searches for PN-modeled gravitational waveforms.

### Hough and Radon Transforms

**Established.** Hough-transform methods are used in continuous-wave searches to identify tracks in time-frequency maps. Radon-style transforms integrate along curves or lines in images and time-frequency planes.

Representative citations/search targets:

- Krishnan et al., Hough-transform continuous-wave search papers. Exact citation needed.
- Astone et al., peakmap/Hough methods for CW searches. Citation needed.
- Brady and Creighton StackSlide/Hough comparison. Citation needed.

**Relation to `zeta`.** Hough/Radon methods are the established “sum along tracks” analogy. Higher-dimensional `zeta` spectra may correspond to higher-dimensional generalized Radon transforms.

**Novelty risk.** High for the broad “track accumulation in transformed coordinates” idea.

## Implications for Choosing `dim(zeta)`

### Established Tools

The literature provides several partial answers:

1. **Mismatch metrics** tell which phase directions matter locally and how template counts scale with coherent time.
2. **StackSlide optimization** gives fixed-computing-cost tradeoffs among `T_coh`, number of segments, mismatch, and template count.
3. **PowerFlux** shows that weighted semicoherent sums can outperform unweighted power sums when detector noise and response vary.
4. **Reduced bases/SVD/PCA** provide numerical low-dimensional representations of waveform families.
5. **Polynomial-phase/chirp transforms** provide algorithms for spectra over chirp coefficients.
6. **Chirp-time coordinates** provide physically motivated compact parameterizations of inspiral phase.

### What Remains Open

The specific question in `idea.md` is not fully answered by any single method identified here:

> How many dimensions should the coherent chunk spectrum have, and which coordinates should they be, when the objective is semicoherent sensitivity at fixed computing cost?

A useful decision rule should combine:

```math
\Delta \mu_d(T_{\rm coh})
```

the mismatch reduction from adding dimension `d`, with

```math
\Delta C_d,\quad \Delta M_d,\quad \Delta N_{\rm trials,d}
```

the extra compute, memory, and trials-factor cost.

A practical criterion is:

```math
\text{keep dimension } d
\quad \text{only if} \quad
h_{\min}(d) < h_{\min}(d-1)
```

at fixed false-alarm probability and fixed compute/memory budgets.

This criterion is a synthesis, not a directly verified literature result.

## Relevance to Equal-Mass Non-Spinning 3.5PN Waveforms

### Established Background

For compact-binary inspiral phases, PN expansions provide phase terms with different frequency dependence. The leading Newtonian/0PN term dominates the accumulated phase over many regimes. Chirp-time and metric-coordinate literature already exploits physically meaningful combinations of mass parameters.

### Open Project-Specific Question

The user’s key question is sharper:

> For equal mass-ratio, non-spinning 3.5PN waveforms, is the best one-parameter chunk-local basis meaningfully different from the 0PN waveform direction?

This is not settled by the literature pass. It should be tested numerically and, if possible, analytically.

Recommended procedure:

1. Generate equal-mass non-spinning 3.5PN phases over the project’s mass/frequency range.
2. Restrict to coherent chunks of duration `T_coh`.
3. Remove nuisance directions: constant phase, possibly frequency, and any baseline coordinate included in all searches.
4. Build residual phase vectors.
5. Compute SVD/PCA or metric eigenvectors.
6. Compare the first principal direction to the 0PN direction using overlap under the same inner product.
7. Report mismatch versus dimension.

### Conjecture

It is plausible that the 0PN direction is close to the best one-dimensional coordinate when:
- the parameter range is narrow,
- the mass ratio is fixed,
- spins are absent,
- chunks are short enough that higher PN residuals are small.

But this is a conjecture. It may fail at higher frequencies, longer `T_coh`, broader mass ranges, or after projecting out frequency and time shifts.

## Relevance to Equal-Mass Spinning 3.5PN Waveforms with `chi <= 0.2`

### Established Background

Aligned-spin PN phasing introduces spin-orbit and spin-spin terms at PN orders above leading Newtonian order. For equal-mass systems with modest aligned spin, spin-dependent phase variation may be smaller than mass-driven variation but can still accumulate coherently.

Exact PN spin-phase citations needed. Search targets:
- Blanchet PN inspiral review.
- Kidder spin effects in compact binary inspiral.
- Ajith / LALSimulation TaylorF2 aligned-spin phasing references.
- Buonanno, Chen, Vallisneri effective spin / aligned-spin template-bank work.

Zotero status unknown.

### Open Project-Specific Question

The user asks:

> What happens if we allow waveforms to be spinning up to `chi = 0.2`?

This requires specifying:
- aligned spin only or both aligned and anti-aligned,
- single effective spin or equal component spins,
- whether `0 <= chi <= 0.2` or `|chi| <= 0.2`,
- waveform approximant and PN convention.

### Conjecture

Spin likely adds at least one additional phase direction, but for small aligned spins it may project significantly onto existing non-spinning mass directions. The answer depends on the chosen parameter range and inner product.

Recommended basis comparison:

| Basis | Question answered |
|---|---|
| Non-spinning SVD basis applied to spinning waveforms | Does spin live mostly in existing directions? |
| Spinning SVD basis | What is the true effective dimension with spin? |
| 0PN plus leading spin-PN coordinate | Does a physical spin coordinate match SVD performance? |
| Higher-dimensional SVD basis | How many coordinates are needed at target mismatch? |

## Mathematical and Analytical Connections

### Polynomial Phase

For polynomial phase models,

```math
\phi(u) = \sum_{j=0}^{p} a_j u^j,
```

there is an analytical path to dimension selection. Under a simple windowed inner product, orthogonal polynomial bases diagonalize the residual phase space more naturally than raw monomials. This suggests that the best `zeta` coordinates may be orthogonalized combinations of `{f, fdot, fddot, ...}`, not the raw derivatives.

This is an inference from standard approximation theory and signal processing, not a verified citation-specific result in this artifact.

### PN Phase

For PN phases, the analytic analogue is less direct. A possible path is:

1. Write the phase as a linear combination of PN basis functions over a chunk.
2. Define a noise/window-weighted inner product.
3. Project out nuisance directions.
4. Compute the Gram matrix of PN basis functions.
5. Diagonalize the Gram matrix over the chosen parameter distribution.

This would produce analytic or semi-analytic metric eigen-coordinates. It may show whether the 0PN coordinate is the leading one-dimensional direction.

This derivation should be treated as proposed theory, not established literature.

## Novelty Risks

### Highest-Risk Overlaps

1. **Fast chirp transform**
   - Risk: `zeta` spectra over chirp parameters may already be known chirp transforms.

2. **Polynomial-phase transforms**
   - Risk: choosing dimensions like `{f, fdot, fddot}` may already be model-order selection for polynomial-phase signals.

3. **Weave and semicoherent metrics**
   - Risk: metric-based semicoherent template banks may already answer the coordinate-count question for CW-like searches.

4. **StackSlide optimization**
   - Risk: fixed-compute optimization of semicoherent searches is established, even if not in the proposed `zeta` language.

5. **Chirplet / Radon / Hough methods**
   - Risk: higher-dimensional track integration in time-frequency-like spaces is a known signal-processing construction.

### Lower-Risk But Relevant Overlaps

1. **ROQ / reduced basis**
   - These optimize waveform representation and likelihood cost, but not necessarily semicoherent power spectra.

2. **MBTA**
   - Relevant for chirping signals and multiband cost reduction, but not obviously the same intermediate-statistic design.

3. **PCA/SVD template-bank compression**
   - Relevant for basis choice, but not necessarily for choosing reusable coherent spectra and semicoherent tracks.

## Defensible Framing

A defensible final report should say:

1. Semicoherent track summing is established.
2. Higher-dimensional chirp/polynomial-phase transforms are established.
3. Metric and reduced-basis coordinate choices are established.
4. What remains potentially useful is a **unified cost-sensitivity framework** for selecting chunk-local coordinate dimension `dim(zeta)` for modeled chirping GW signals.
5. The framework should be tested against:
   - StackSlide,
   - PowerFlux-style weighted sums,
   - Weave/metric template-bank expectations,
   - fast chirp or polynomial-phase transforms,
   - local semicoherent matched-filter banks,
   - reduced-basis/SVD coordinate choices.

## Papers and Sources to Verify or Add to Zotero

Zotero status unknown for all items.

### Highest Priority

1. Brady and Creighton, 2000, StackSlide/hierarchical CW searches.
2. Prix and Shaltev, 2012, optimal StackSlide at fixed computing cost.
3. Weave methods paper or `lalpulsar` Weave documentation/paper.
4. Pletsch semicoherent metric papers.
5. Jenet and Prince, 2000, fast chirp transform.
6. Polynomial-phase transform / high-order ambiguity papers by Peleg, Porat, Barbarossa, Giannakis, Boashash.
7. Owen 1996 and Owen/Sathyaprakash 1999 template-bank metric papers.
8. Chirp-time coordinate papers by Sathyaprakash, Dhurandhar, Owen, Tanaka, Tagoshi.
9. ROQ/reduced-basis papers by Field, Galley, Tiglio, Canizares, Pürrer.
10. MBTA/multiband CBC search papers.

### Secondary Priority

1. PowerFlux method papers by Dergachev and LIGO/Virgo CW collaborations.
2. Loosely coherent searches.
3. Hough and Radon transform CW search papers.
4. Chirplet transform papers.
5. SVD/PCA template-bank compression papers by Cannon and collaborators.
6. Aligned-spin PN phasing references for TaylorF2 / 3.5PN spin effects.

## Concrete Recommendations for the Next Agent

1. Inspect Weave and StackSlide fixed-cost papers first. These are the closest GW prior art.
2. Separately inspect fast chirp transform and polynomial-phase transform literature. These are the closest signal-processing prior art.
3. Do not claim novelty for `{f, fdot}` or `{f, beta}` spectra until those are compared against chirp-transform methods.
4. Treat 0PN near-optimality as an empirical/theoretical question, not a literature fact.
5. For the 3.5PN non-spinning case, run SVD/PCA on residual phase after projecting out constant phase and frequency.
6. For the spinning case, define the spin convention before drawing conclusions.
7. In the final synthesis, use the literature to motivate a costed decision rule:

```math
d_\zeta^\star
=
\arg\min_d h_{\min}(d)
\quad
\text{subject to compute, memory, and false-alarm constraints.}
```

8. Mark all Zotero statuses until checked.