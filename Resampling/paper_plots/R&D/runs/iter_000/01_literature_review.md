---
title: Literature review for optimal semicoherent searches in higher-dimensional spectral spaces
source: idea.md
agent: literature
attempt: 1
status: auditable literature artifact
required_output: outputs/01_literature_review.md
---

## Audit status

This artifact treats `idea.md` as the source of truth.

No shell commands were run. `CODEX.md`, Zotero, and local implementation files were not inspected in this attempt because the instruction for this attempt explicitly prohibited shell commands and file creation/editing. Zotero status is therefore marked as `not checked`.

External claims below are based on web-accessible metadata and abstracts/snippets, not full-paper reading unless explicitly stated otherwise. I do not claim to have read the full papers. Items needing confirmation from the full text are marked accordingly.

## Executive summary

The core idea has several close precedents, but I did not find evidence from this limited search that the exact formulation has already been named in the gravitational-wave literature as "choose an optimal intermediate spectral coordinate space `zeta` of variable dimension for semicoherent modelled-signal searches."

The closest established gravitational-wave antecedents are:

- StackSlide and optimized semicoherent continuous-wave searches: these already optimize coherent time, segment count, coarse/fine template grids, mismatch, and computing cost.
- Weave: directly relevant because it implements semicoherent CW searches using metric template banks and optimal lattices.
- Semicoherent metric work: directly relevant because it formalizes parameter-space resolution in the incoherent combination step.
- PowerFlux and Hough searches: closely related as weighted or voting-based power accumulation along time-frequency tracks.
- Fast chirp transform and polynomial-phase signal methods: probably the highest novelty risk for the specific idea of replacing a 2D time-frequency map with a higher-dimensional transform space for chirping/modelled signals.

The most likely "other names" for the idea are:

- semicoherent StackSlide with metric template banks
- generalized Hough/Radon transform over signal tracks
- fast chirp transform
- polynomial phase transform / polynomial-phase signal detection
- higher-order ambiguity function
- reduced-coordinate or metric-eigenbasis template placement
- reduced-basis / SVD phase-coordinate template bank construction

The likely novelty is not "semicoherent summing along tracks" or "optimizing `T_coh` at fixed compute"; those are established. The possible novelty is the explicit optimization over the dimension and coordinates of the intermediate spectral representation `zeta`, especially for PN chirps where each coherent chunk is transformed with a low-order local phase model and then semicoherently connected across chunks.

## Claim taxonomy

### Established facts from `idea.md`

- The proposed statistic constructs spectra in a coordinate space `zeta`, which includes at least frequency and time and may include additional phase-evolution parameters.
- The local implementation uses an example with `zeta = {f, t, beta}`, where `beta` is a 0PN parameter.
- Increasing the number of `zeta` coordinates may increase maximum coherent time, but also increases the cost of spectrum creation and track accumulation.
- The desired question is computational: choose the number and definition of `zeta` parameters optimally as a function of coherent chunk length and signal model.

### Established from literature metadata/abstracts

- StackSlide and related semicoherent methods were developed because fully coherent wide-parameter searches are computationally prohibitive.
- StackSlide optimization at fixed computing cost has been studied analytically under ideal conditions and numerically under more realistic data conditions.
- Weave uses parameter-space metrics and optimal lattices for semicoherent continuous-wave template-bank construction.
- Semicoherent parameter-space metrics for continuous-wave searches have been derived.
- PowerFlux is a weighted power-summing method related to StackSlide.
- MBTA is a multiband compact-binary pipeline, relevant as a cost-reduction strategy but not obviously the same as higher-dimensional semicoherent spectra.
- Reduced-basis, ROQ, surrogate, SVD, PCA-like, and chirp-time coordinate methods exist for waveform compression, likelihood acceleration, and template-bank construction.
- Fast chirp transform methods were proposed for variable-frequency signal detection, motivated partly by gravitational-wave chirps.

### Conjectures for this project

- The optimal `zeta` dimension should balance residual coherent mismatch against the combinatorial/memory cost of constructing and searching spectra.
- For PN chirps, `zeta` coordinates based on local PN phase coefficients, chirp times, or metric eigenvectors may outperform raw physical parameters.
- The project may be expressible as a generalized semicoherent Radon/Hough transform in a reduced phase-coordinate space.
- A useful cost model likely needs both FLOP scaling and memory-bandwidth scaling, because higher-dimensional spectral arrays may become bandwidth dominated.

### Open questions

- Has a signal-processing paper already solved the optimal dimension-selection problem for polynomial-phase/chirp transforms?
- Has the CW literature considered a hierarchy where coherent chunk transforms are computed over progressively richer local phase models, rather than over standard CW coordinates?
- Does the proposed `zeta = {f,t,beta}` approach reduce to a known fast chirp transform or polynomial phase transform under a change of variables?
- Can reduced-basis/SVD phase coordinates be used to define `zeta` directly, or only to place templates after spectra are computed?

## Citation table

| Method | Paper/source found | Year | Zotero status | Relevance | Reusable result |
|---|---:|---:|---|---|---|
| StackSlide / hierarchical CW | Patrick R. Brady and Teviet Creighton, "Searching for periodic sources with LIGO. II: Hierarchical searches", arXiv:gr-qc/9812014, https://arxiv.org/abs/gr-qc/9812014 | 1998 arXiv | not checked | Foundational semicoherent stacked-power search | Number of corrections/templates, sensitivity/cost comparison of coherent vs incoherent stacks |
| StackSlide optimization | Curt Cutler, Iraj Gholami, Badri Krishnan, "Improved Stack-Slide Searches for Gravitational-Wave Pulsars", arXiv:gr-qc/0505082, https://arxiv.org/abs/gr-qc/0505082 | 2005 | not checked | Multi-stage optimized StackSlide | Computational search strategy and sensitivity vs computing power |
| StackSlide fixed-cost optimization | Reinhard Prix and Miroslav Shaltev, "Search for Continuous Gravitational Waves: Optimal StackSlide method at fixed computing cost", arXiv:1201.4321, https://arxiv.org/abs/1201.4321 | 2012 | not checked | Directly relevant to optimizing `T_coh` and segment count | Analytical fixed-compute optimization; coarse/fine mismatch treatment |
| Realistic StackSlide optimization | Miroslav Shaltev, "Optimizing StackSlide setup and data selection...", arXiv:1510.06427, https://arxiv.org/abs/1510.06427 | 2015 | not checked | Realistic gaps/noise case | Numerical optimization and data selection |
| Hough CW search | Badri Krishnan et al., "The Hough transform search for continuous gravitational waves", arXiv:gr-qc/0407001, https://arxiv.org/abs/gr-qc/0407001 | 2004 | not checked | Track voting in time-frequency plane | Hough statistic, sensitivity, and statistical properties |
| Large-scale correlations | Holger J. Pletsch and Bruce Allen, "Exploiting Large-Scale Correlations to Detect Continuous Gravitational Waves", arXiv:0906.0023, https://arxiv.org/abs/0906.0023 | 2009 | not checked | Optimizes incoherent combination using parameter-space correlations | Strongly relevant to track accumulation through transformed coordinates |
| Semicoherent metric | Holger J. Pletsch, "Parameter-space metric of semicoherent searches for continuous gravitational waves", arXiv:1005.0395, https://arxiv.org/abs/1005.0395 | 2010 | not checked | Direct metric formalism for semicoherent combination | Semicoherent metric and resolution scaling |
| Weave | K. Wette, S. Walsh, R. Prix, M. A. Papa, "Implementing a semicoherent search for continuous gravitational waves using optimally-constructed template banks", arXiv:1804.03392, https://arxiv.org/abs/1804.03392 | 2018 | not checked | Closest GW implementation analogue | Metric banks plus optimal lattices |
| Lattice placement | Karl Wette, "Lattice template placement for coherent all-sky searches...", arXiv:1410.6882, https://arxiv.org/abs/1410.6882 | 2014 | not checked | Template-grid efficiency | Constant-density metric lattice placement |
| BinaryWeave | Arunava Mukherjee, Reinhard Prix, Karl Wette, "Implementation of a new weave-based search pipeline...", arXiv:2207.09326, https://arxiv.org/abs/2207.09326 | 2022 | not checked | Extension to known binary CWs | Timing model and semicoherent StackSlide F-statistic with lattice banks |
| PowerFlux | LIGO Scientific Collaboration, "All-sky search for periodic gravitational waves in LIGO S4 data", arXiv:0708.3818, https://arxiv.org/abs/0708.3818 | 2007 | not checked | Weighted power-summing method | Comparison of StackSlide, weighted Hough, and PowerFlux |
| PowerFlux S5 | LIGO Scientific Collaboration, "All-sky LIGO Search for Periodic Gravitational Waves in the Early S5 Data", arXiv:0810.0283, https://arxiv.org/abs/0810.0283 | 2008 | not checked | Mature PowerFlux application | Semicoherent power summing and upper-limit procedure |
| Modern PowerFlux | Aashish Tripathee and Keith Riles, "Probing More Deeply in an All-Sky Search...", arXiv:2311.04985, https://arxiv.org/abs/2311.04985 | 2023 | not checked | Modern PowerFlux/loose-coherence search | Practical sensitivity and loose-coherence implementation context |
| Loosely coherent | Vladimir Dergachev, "Loosely coherent searches for sets of well-modeled signals", arXiv:1110.3297, https://arxiv.org/abs/1110.3297 | 2011 | not checked | Intermediate between coherent and incoherent | High-performance statistic for finite-dimensional signal manifolds |
| MBTA | T. Adams, "Low latency search for compact binary coalescences using MBTA", arXiv:1507.01787, https://arxiv.org/abs/1507.01787 | 2015 | not checked | Multiband CBC pipeline | Cost reduction via frequency-band decomposition |
| MBTA O3 | Florian Aubin et al., "The MBTA Pipeline for Detecting Compact Binary Coalescences in the Third LIGO-Virgo Observing Run", arXiv:2012.11512, https://arxiv.org/abs/2012.11512 | 2020 | not checked | Current pipeline details | Architecture, ranking, background estimation |
| MBTA O4 | Christopher Allene et al., "The MBTA Pipeline... Fourth LIGO-Virgo-KAGRA Observing Run", arXiv:2501.04598, https://arxiv.org/abs/2501.04598 | 2025 | not checked | Latest MBTA source found | O4 configuration and evolution |
| Reduced basis | Scott E. Field et al., "Reduced basis catalogs for gravitational wave templates", arXiv:1101.3765, https://arxiv.org/abs/1101.3765 | 2011 | not checked | Waveform-space compression | Greedy reduced bases; fewer basis elements than standard catalogs |
| Surrogates | Scott E. Field et al., "Fast prediction and evaluation of gravitational waveforms using surrogate models", arXiv:1308.3565, https://arxiv.org/abs/1308.3565 | 2013 | not checked | Reduced-order waveform generation | Greedy basis, empirical interpolation, low online cost |
| ROQ | Priscilla Canizares et al., "Accelerated gravitational-wave parameter estimation with reduced order modeling", arXiv:1404.6284, https://arxiv.org/abs/1404.6284 | 2014 | not checked | Reduced-order quadrature | Likelihood speedups via ROQ |
| ROM/SVD | Michael Purrer, "Frequency domain reduced order models for gravitational waves from aligned-spin compact binaries", arXiv:1402.4146, https://arxiv.org/abs/1402.4146 | 2014 | not checked | SVD-based reduced model | Frequency-domain reduced-order modeling |
| Review | Manuel Tiglio and Aaron Villanueva, "Reduced Order and Surrogate Models for Gravitational Waves", arXiv:2101.11608, https://arxiv.org/abs/2101.11608 | 2021 | not checked | Review of PCA, POD, RB, EIM, ROQ | Conceptual map for dimensionality reduction |
| Template metric | Benjamin J. Owen, "Search templates for gravitational waves from inspiraling binaries: Choice of template spacing", arXiv:gr-qc/9511032, https://arxiv.org/abs/gr-qc/9511032 | 1995 | not checked | Foundational metric placement | Differential geometry for template spacing and cost |
| Inspiral cost/template placement | Benjamin J. Owen and B. S. Sathyaprakash, "Matched filtering of gravitational waves from inspiraling compact binaries...", arXiv:gr-qc/9808076, https://arxiv.org/abs/gr-qc/9808076 | 1998 | not checked | CBC template-bank cost | Template counts, computational power, placement |
| IMR chirp-time-like coordinates | Chinmay Kalaghatgi, Parameswaran Ajith, K. G. Arun, "Template-space metric...", arXiv:1501.04418, https://arxiv.org/abs/1501.04418 | 2015 | not checked | Coordinate choice for metric stability | Modified PN chirp-time coordinates |
| SVD/PCA-like template banks | Javier Roulet et al., "Template Bank for Compact Binary Coalescence Searches... General Geometric Placement Algorithm", arXiv:1904.01683, https://arxiv.org/abs/1904.01683 | 2019 | not checked | Very relevant coordinate-reduction analogue | SVD of phase profiles; regular grid in reduced coefficient space |
| Fast chirp transform | F. A. Jenet and T. A. Prince, "Detection of variable frequency signals using a fast chirp transform", arXiv:gr-qc/0012029, https://arxiv.org/abs/gr-qc/0012029 | 2000 | not checked | Highest signal-processing novelty risk | Fast transform over variable-frequency/chirp signals |
| Chirp-Z transform | Rabiner, Schafer, Rader, "The chirp z-transform algorithm and its application", IEEE Trans. Audio Electroacoustics, 1969 | 1969 | citation from secondary source; verify | Spectral zoom/chirp transform analogue | Fast evaluation of generalized z-transform contours |
| Generalized Hough/Radon | Duda and Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures", Comm. ACM, 1972 | 1972 | citation from secondary source; verify | Parameter-space voting analogue | Accumulator-space interpretation of curve detection |

## StackSlide

StackSlide is directly relevant and is a major prior art source.

Established from the Brady-Creighton abstract: the method divides a demodulated time series into `N` segments of length `Delta T`, FFTs each segment, computes power, and sums spectra. It estimates the number of independent corrections required for stacked-power searches and compares sensitivity under computational constraints.

For this project, ordinary StackSlide corresponds roughly to choosing a low-dimensional spectral representation, usually per-segment frequency-time power, and summing along tracks defined by source parameters. The project's `zeta` generalization can be viewed as StackSlide where the per-segment coherent statistic is indexed by additional local phase-evolution coordinates.

Important carry-over results:

- Sensitivity/cost tradeoff for coherent segment length.
- Template-count scaling for incoherent stacks.
- Hierarchical refinement concepts.
- Distinction between coarse coherent grids and finer semicoherent grids.
- Trials-factor and threshold implications of many tracks.

Directly relevant follow-ups:

- Cutler, Gholami, and Krishnan generalize Brady-Creighton to multi-stage StackSlide and optimize computational search strategies.
- Prix and Shaltev give an analytical fixed-computing-cost optimization under ideal conditions.
- Shaltev extends setup/data-selection optimization to gaps and varying noise.

Novelty implication: optimizing `T_coh`, segment count, and coarse/fine-grid mismatches is not novel. Any project report must position the new contribution as optimization over the dimension and coordinate choice of `zeta`, not generic StackSlide optimization.

## PowerFlux

PowerFlux is a close prior art for weighted power accumulation.

Established from the S4 all-sky search abstract: PowerFlux is described as a variant of StackSlide in which power is weighted before summing; weights are chosen using noise and detector antenna pattern to maximize SNR. The same S4 paper compares StackSlide, weighted Hough, and PowerFlux.

PowerFlux relevance to `zeta`:

- It already treats semicoherent power summation as a weighted statistical estimator rather than a simple unweighted sum.
- It highlights the importance of detector antenna patterns and noise weights in any power-summing statistic.
- It is a warning that the proposed statistic should not stop at unweighted accumulated power unless justified.

Uncertain claim: PowerFlux may contain implementation tricks for efficient track accumulation and upper-limit construction that transfer to higher-dimensional spectra. This needs full-paper inspection.

Novelty implication: a higher-dimensional `zeta` statistic that only sums weighted power could be viewed as a PowerFlux/StackSlide variant unless the coordinate-construction and dimension-optimization piece is genuinely new.

## Weave

Weave is one of the strongest gravitational-wave novelty risks.

Established from the Weave abstract: Weave is a semicoherent CW search implementation that uses a parameter-space metric to generate template banks at the correct resolution and combines this with optimal lattices to minimize template count and computational cost.

Relevance to this project:

- Weave already integrates semicoherent search, metric template banks, and computational-cost reduction.
- It is likely the best reference for formalizing template placement in a high-dimensional signal parameter space.
- BinaryWeave extends the same style to continuous waves from known binary systems, which is closer to signals with additional orbital/evolution parameters.

Difference from the current idea, as currently understood:

- Weave appears to optimize template banks in the physical or semicoherent CW parameter space, not necessarily to choose a variable-dimensional intermediate spectral coordinate system `zeta`.
- The proposed idea asks whether the coherent chunk statistic itself should be indexed by additional local model parameters such as PN coefficients, and how many such dimensions are optimal.

Uncertain claim: Weave may already support multiple coherent-statistic coordinate choices internally. Needs full-paper and code inspection.

## MBTA

MBTA is relevant but less directly overlapping.

Established from MBTA abstracts: MBTA is a low-latency compact-binary coalescence pipeline using multiband analysis. Modern MBTA papers describe architecture, configuration, ranking statistics, false-alarm-rate evaluation, and O3/O4 operation.

Relevance:

- MBTA reduces cost by splitting waveform filtering across frequency bands.
- The conceptual analogue is trading exact full-band filtering for a cheaper decomposition that preserves detection power.
- For the proposed `zeta` framework, MBTA suggests that "optimal representation" may involve partitioning data or waveform phase by frequency band, not only by time segment.

Difference:

- MBTA is fundamentally matched filtering for CBC transients, not semicoherent power accumulation along long-duration tracks.
- MBTA's multibanding is a frequency-domain cost decomposition, while `zeta` is an intermediate spectral coordinate space.

Novelty implication: MBTA is not the same problem, but it weakens claims that "using lower-dimensional or approximate representations for modelled GW signals to reduce cost" is new.

## Reduced order quadrature and reduced basis methods

Reduced-basis and ROQ methods are important conceptual analogues for choosing efficient coordinates/bases.

Established from abstracts:

- Field et al. 2011 introduce reduced-basis catalogs for GW templates and report large reductions in template/basis counts relative to standard placement.
- Field et al. 2013 use greedy basis construction and empirical interpolation to build fast waveform surrogates.
- Canizares et al. 2014 implement ROQ in LAL and report speedups for Bayesian inference.
- Purrer 2014 uses SVD-based frequency-domain reduced-order models.
- Tiglio and Villanueva 2021 review PCA, POD, reduced basis, empirical interpolation, ROQ, and compressed likelihoods.

Relevance to `zeta`:

- These methods provide a principled way to identify low-dimensional waveform subspaces.
- They may suggest coordinates for `zeta`: greedy basis coefficients, empirical interpolation nodes, SVD coefficients, or reduced phase coefficients.
- They offer error-control language that could be translated into coherent mismatch bounds.

Difference:

- ROQ accelerates likelihood evaluations once a waveform family and basis are built; it does not directly construct semicoherent spectra and sum power along tracks.
- Reduced bases often preserve complex waveform information, whereas the proposed statistic may discard phase by summing power.

Potential reusable result:

If the phase model inside each coherent chunk is approximated by
`phi(t; theta) approx sum_i c_i(theta) e_i(t)`,
then a natural `zeta` could be the leading coefficients `c_i`. This connects the project's "linear combinations of theta" language to established reduced-basis/SVD machinery.

Novelty implication: claiming "use PCA or reduced bases to choose coordinates" is not novel. Applying such coordinates as axes of a semicoherent spectral accumulator may still be novel, but must be checked against polynomial-phase-transform literature.

## PCA and dimensionality reduction

The Roulet et al. 2019 template-bank paper is especially relevant.

Established from the abstract: Roulet et al. exploit smooth dependence of frequency-domain waveform amplitude and unwrapped phase on binary parameters. They group similar amplitude profiles and perform SVD of phase profiles to obtain an orthonormal basis. Leading basis functions span a lower-dimensional linear space in which physical waveform phases are well approximated. Template placement is then a regular grid in the space of linear coefficients.

This is very close to the coordinate-choice part of `idea.md`.

Relevance:

- Supports using phase-basis coefficients rather than physical masses/spins.
- Provides a concrete precedent for linear coefficient coordinates derived from SVD/PCA of waveform phase.
- Could be adapted segment-wise: compute the SVD of local phase residuals over chunk duration `T_coh`, then choose the first `d_zeta` coefficients as spectral axes.

Related metric-coordinate work:

- Owen's template-spacing metric provides the older differential-geometric basis for mismatch-controlled grids.
- Owen and Sathyaprakash estimate template counts and computational costs for inspiral searches.
- Kalaghatgi, Ajith, and Arun propose modified PN chirp-time coordinates for a slowly varying IMR metric.

Novelty implication: reduced phase-coordinate placement is established. The remaining possible novelty is to use these coordinates as coherent-statistic transform axes and optimize how many are included before semicoherent summation.

## Semicoherent template metrics

The semicoherent metric literature is central.

Established from Pletsch 2010 abstract: the paper derives an analytical parameter-space metric for the incoherent combination step in semicoherent CW searches and studies additional metric resolution from combining segments.

Established from Pletsch and Allen 2009 abstract: optimal incoherent combination exploits large-scale parameter-space correlations in the coherent detection statistic, improving sensitivity and reducing cost relative to ad hoc methods.

Established from Weave abstract: Weave uses parameter-space metrics and optimal lattices to minimize templates and cost.

Relevance to project formalism:

For a semicoherent statistic
`S(theta) = sum_k P_k(zeta_k(theta))`,
a local mismatch expansion should yield something like
`mu(theta, Delta theta) approx g^semi_ij Delta theta_i Delta theta_j`,
where `g^semi` depends on the coherent statistic and the map `theta -> zeta_k(theta)`.

The project's additional layer is choosing `zeta` itself. A plausible route:

1. Define a full coherent phase metric over physical parameters `theta`.
2. Define a projection onto retained local coordinates `zeta_d`.
3. Decompose mismatch into retained-grid mismatch plus residual model mismatch:
   `mu_total(d, T_coh) = mu_grid(zeta_d) + mu_residual(theta | zeta_d) + mu_semi(track grid)`.
4. Optimize cost over discrete coordinate sets and `T_coh`.

Uncertain claim: the above decomposition may already exist in the semicoherent metric literature under another notation. Needs full-text search.

## Signal-processing analogues

### Fast chirp transform

The fast chirp transform is the strongest non-GW prior-art risk.

Established from Jenet and Prince abstract: the fast chirp transform was defined as analogous to the FFT for detecting signals with variable frequency. It was motivated partly by gravitational-wave binary inspiral detection and was proposed as a way to avoid generating complicated matched-filter families.

Relevance:

- A higher-dimensional spectral space indexed by chirp parameters is essentially the conceptual territory of chirp transforms.
- The project's `zeta = {f,t,beta}` may be a special case of a chirp-transform-like representation if `beta` controls a polynomial or PN chirp rate.
- Fast chirp transform literature may already address computational scaling with chirp dimension/order.

Action for next agent: search full text and citations of Jenet & Prince for "polynomial phase", "order", "dimension", "cost", "Radon", and "Hough".

### Polynomial-phase transforms

Search target; not sufficiently audited in this attempt.

Likely relevant terms:

- polynomial phase signal detection
- polynomial phase transform
- high-order ambiguity function
- product high-order ambiguity function
- Wigner-Hough transform
- Radon-Wigner transform
- chirplet transform
- fractional Fourier transform
- generalized time-frequency distributions

Uncertain but plausible: the problem of choosing how many polynomial phase coefficients to include at a given coherent integration time has been studied in radar/sonar signal processing. This is a major open search target.

### Hough/Radon transforms

The Hough transform CW paper is directly relevant because it maps time-frequency tracks into a parameter-space accumulator. Generalized Hough/Radon transforms are the generic signal-processing version of "sum along parameterized tracks."

Relevance:

- The proposed statistic can be interpreted as an accumulator over template tracks in a higher-dimensional feature space.
- Hough/Radon literature likely contains results on curse-of-dimensionality, accumulator binning, and optimal parameterization.
- The Hough transform becomes inefficient at high parameter dimension; this is directly analogous to adding too many `zeta` axes.

### Chirp-Z transform

The chirp-Z transform is less directly relevant: it evaluates the z-transform along spiral contours and is often used for spectral zooming. It should not be conflated with fast chirp transforms for arbitrary variable-frequency tracks. It is still worth mentioning as a related spectral-transform analogue, but likely not the main prior art.

## Relationship to the proposed `zeta` formalism

The proposed method can be cast as:

```math
x_k(t) = x(t_k + t), \quad t \in [-T_{\rm coh}/2, T_{\rm coh}/2]
```

```math
P_k(\zeta) = \left| \int_{k} x(t)\,\exp[-i \Phi_{\rm local}(t; \zeta)]\,dt \right|^2
```

```math
S(\theta) = \sum_k w_k(\theta)\,P_k(\zeta_k(\theta)).
```

Ordinary StackSlide is approximately the special case where `zeta` contains only local frequency, plus the segment time index. The current project's example adds a 0PN chirp parameter `beta`.

A general phase expansion inside each coherent chunk is:

```math
\phi(t;\theta)
= \phi_k
+ 2\pi f_k(\theta)(t-t_k)
+ \sum_{n=1}^{p} a_{n,k}(\theta)(t-t_k)^{n+1}
+ R_{p,k}(t;\theta).
```

One possible `zeta_p` is:

```math
\zeta_p = \{ f_k, a_{1,k}, a_{2,k}, \ldots, a_{p,k} \}.
```

The coherent residual mismatch is controlled by `R_{p,k}`. The spectrum cost grows with the number of grid points in `zeta_p`. The track cost depends on how many physical templates `theta` must be followed through the `zeta_p` arrays.

A first cost model target is:

```math
C_{\rm total}(p,T_{\rm coh})
=
C_{\rm spectra}(p,T_{\rm coh})
+
C_{\rm tracks}(p,T_{\rm coh})
+
C_{\rm thresholds}(p,T_{\rm coh})
+
C_{\rm memory}(p,T_{\rm coh}).
```

The central optimization is:

```math
(p^*, T_{\rm coh}^*)
=
\arg\min_{p,T_{\rm coh}}
C_{\rm total}(p,T_{\rm coh})
```

subject to:

```math
\mu_{\rm residual}(p,T_{\rm coh})
+
\mu_{\rm grid}(p,T_{\rm coh})
+
\mu_{\rm semi}(p,T_{\rm coh})
\leq
\mu_{\rm max}
```

and a fixed false-alarm / false-dismissal requirement.

This formalism is consistent with StackSlide and semicoherent metrics, but the explicit optimization over `p` or `dim(zeta)` is the piece that needs novelty checking.

## Novelty risks

### High risk: fast chirp transform / polynomial phase transform

The exact idea of building a higher-dimensional transform for variable-frequency signals is probably established in signal processing. The next literature pass should deeply inspect fast chirp transform, polynomial phase transform, high-order ambiguity function, chirplet transform, and Wigner-Hough/Radon-Wigner literature.

Risk statement: `zeta = {f,t,beta}` may already be a known chirp-transform coordinate choice.

### High risk: semicoherent CW metric/template-bank optimization

StackSlide, semicoherent metrics, Pletsch-Allen correlations, Prix-Shaltev optimization, and Weave already cover much of the GW semicoherent optimization space.

Risk statement: if the project only optimizes `T_coh`, segment count, and template spacing, it is likely not novel.

### Medium risk: reduced phase-coordinate template banks

Roulet et al. already use SVD phase bases and grids in reduced coefficient space for CBC template banks.

Risk statement: using PCA/SVD to choose coordinates is not novel. The novel angle would need to be "coordinates as axes of an intermediate semicoherent spectral transform."

### Medium risk: Hough/Radon accumulator formulation

Summing evidence along parameterized curves in a data-derived map is exactly the Hough/Radon family.

Risk statement: the project may be a generalized Hough/Radon transform with a GW-specific coherent statistic.

### Medium risk: loosely coherent searches

Dergachev's loosely coherent searches target finite-dimensional well-modelled signal manifolds and may occupy the space between incoherent power sums and full coherent matched filtering.

Risk statement: if the proposed statistic preserves partial phase coherence between segments, it may overlap strongly with loosely coherent methods.

### Lower risk: MBTA and ROQ

MBTA, ROQ, and reduced-basis methods are conceptually related but not obvious duplicates. They are more likely sources of transferable ideas than direct novelty blockers.

## Useful search targets for next attempt

Search strings to run in ADS/arXiv/Google Scholar/INSPIRE/IEEE:

- `"fast chirp transform" "polynomial phase" complexity`
- `"fast chirp transform" gravitational waves citations`
- `"polynomial phase transform" "order" "computational complexity"`
- `"polynomial phase signal" "model order selection" detection`
- `"high order ambiguity function" "polynomial phase" "detection"`
- `"Wigner Hough transform" "chirp" "polynomial phase"`
- `"Radon Wigner transform" "polynomial phase signal"`
- `"chirplet transform" "signal detection" "computational complexity"`
- `"semicoherent metric" "StackSlide" "coordinate"`
- `"Weave" "metric" "template bank" "StackSlide" full text`
- `"PowerFlux" "loose coherence" "StackSlide"`
- `"reduced basis" "semicoherent" "continuous waves"`
- `"SVD phase" "template bank" "gravitational waves"`
- `"chirp time coordinates" "metric" "template bank"`

## Papers likely worth adding to Zotero if absent

Zotero was not checked. Candidate additions:

- Brady & Creighton, arXiv:gr-qc/9812014
- Cutler, Gholami & Krishnan, arXiv:gr-qc/0505082
- Prix & Shaltev, arXiv:1201.4321
- Shaltev, arXiv:1510.06427
- Pletsch & Allen, arXiv:0906.0023
- Pletsch, arXiv:1005.0395
- Wette et al., arXiv:1804.03392
- Mukherjee, Prix & Wette, arXiv:2207.09326
- LIGO S4 PowerFlux/StackSlide/Hough paper, arXiv:0708.3818
- LIGO early S5 PowerFlux paper, arXiv:0810.0283
- Dergachev, arXiv:1110.3297
- Adams MBTA, arXiv:1507.01787
- Aubin et al. MBTA O3, arXiv:2012.11512
- Field et al. reduced basis, arXiv:1101.3765
- Field et al. surrogate models, arXiv:1308.3565
- Canizares et al. ROQ, arXiv:1404.6284
- Tiglio & Villanueva review, arXiv:2101.11608
- Roulet et al. SVD/geometric template bank, arXiv:1904.01683
- Owen template spacing, arXiv:gr-qc/9511032
- Owen & Sathyaprakash, arXiv:gr-qc/9808076
- Kalaghatgi, Ajith & Arun, arXiv:1501.04418
- Jenet & Prince fast chirp transform, arXiv:gr-qc/0012029

## Recommended next-agent actions

1. Inspect `CODEX.md`, local `zeta = {f,t,beta}` implementation, and Zotero.
2. Read Weave, Pletsch 2010, Prix-Shaltev 2012, and Jenet-Prince 2000 closely.
3. Search IEEE/Google Scholar for polynomial-phase transform, high-order ambiguity function, Wigner-Hough, and chirplet transform.
4. Build a mapping table from this project's notation to known methods:
   - `P_k(zeta)` vs SFT power, F-statistic atoms, chirp transform output, Hough accumulator.
   - `zeta_k(theta)` vs template track, chirp-rate parameterization, phase-metric coordinates.
5. Decide whether the report should frame the project as:
   - a new semicoherent metric/cost optimization over intermediate coherent-statistic coordinates, or
   - an application/adaptation of fast chirp/polynomial-phase transforms to PBH-inspired GW chirps with detector modulation.