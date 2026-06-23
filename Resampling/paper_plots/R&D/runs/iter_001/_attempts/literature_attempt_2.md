# Literature Review: Semicoherent `zeta`-Space Searches and Dimension Selection

## Audit Metadata

- Source of truth: `idea.md` and attempt-2 human feedback.
- Required artifact: `runs/iter_001/01_literature_review.md`.
- Agent: `literature`.
- Attempt: 2.
- External literature access: web search was used for bibliographic verification and source links.
- Local file access: `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md` was not inspected because the wrapper instruction for this agent explicitly said not to use shell commands or edit files.
- Zotero status: unknown for all papers. No Zotero check was possible under the wrapper instruction.
- Citation policy: works are cited only when a real source could be identified with high confidence. Items marked `citation needed` require follow-up verification.

## Executive Summary

The problem in `idea.md` has almost certainly been studied in pieces, but not obviously as one unified optimization problem in the exact proposed form.

The broad idea of semicoherently summing power along modeled tracks is established prior art: StackSlide, PowerFlux, Hough/Radon methods, loosely coherent searches, Weave, and semicoherent template metrics all overlap with this. The idea of coherent spectra indexed by chirp parameters is also established in signal processing under names such as fast chirp transform, polynomial-phase transform, high-order ambiguity function, chirplet transform, and generalized Radon/Hough transforms.

The defensible research question is narrower:

> Given a modeled chirping signal family, how should one choose the number and basis of chunk-local coherent coordinates `zeta` to maximize sensitivity at fixed compute, memory, and false-alarm probability?

The strongest novelty risk is that `{f, fdot, fddot, ...}` or `{f, beta}` spectra may be a version of a chirp or polynomial-phase transform. The strongest GW-specific novelty risk is Weave plus semicoherent metric literature, which already optimizes semicoherent template banks and computing cost.

The likely remaining contribution is a costed dimension-selection criterion for reusable coherent-chunk spectra, especially for PN chirps, rather than a new detection statistic in the abstract.

## Terminology Map

- `theta`: physical signal parameters, e.g. masses, spins, sky position, coalescence phase/time, polarization, or PBH-inspired chirp parameters.
- `zeta`: chunk-local coherent coordinates used to construct intermediate spectra. These need not be physical parameters.
- `dim(zeta)`: number of non-time coordinates indexed in each coherent chunk output, e.g. `{f}`, `{f, fdot}`, `{f, beta}`, or SVD/PCA coefficients.
- Segment time: in many StackSlide-like methods this is a segment label, not a searched `zeta` dimension. It becomes a `zeta` coordinate only if spectra are explicitly gridded over time/epoch.
- Central optimization: retain an extra `zeta` dimension only if the coherent-mismatch reduction improves the final detectable amplitude after paying spectrum-generation cost, track-summing cost, interpolation, memory, and trials-factor penalties.

## Prior-Art Map

| Area | Established result | Relation to `zeta` idea | Does it choose `dim(zeta)`? | Novelty risk |
|---|---|---|---|---|
| StackSlide | Sum coherent powers from segments along modeled frequency tracks. | Baseline case, roughly `zeta = {f}`. | Optimizes coherent time, mismatch, and template counts, but not arbitrary local coordinate dimension. | High for time-frequency track sums. |
| PowerFlux | Weighted semicoherent power sums using noise and detector response weights. | Suggests `zeta` track sums should be weighted, not naive sums. | No evidence it chooses arbitrary coherent-coordinate dimension. | Moderate. |
| Weave | Semicoherent CW search implementation using metric template banks and optimal lattices. | Closest GW framework for costed semicoherent template placement. | Chooses template-bank dimensions in physical CW parameter space; reusable `zeta` spectra distinction needs checking. | Very high. |
| MBTA | Multiband compact-binary matched filtering. | Related to changing representation across frequency bands. | Not a direct `dim(zeta)` rule. | Moderate. |
| Reduced order quadrature | Reduced bases and empirical interpolation for fast likelihoods. | Basis-size selection is relevant to `zeta` coordinates. | Chooses waveform/likelihood approximation dimension, not semicoherent spectrum dimension. | Moderate. |
| PCA / dimensionality reduction | SVD/PCA compress waveform banks and identify dominant directions. | Direct tool for testing whether 0PN is close to the best one-dimensional basis. | Chooses rank by representation error, not full search sensitivity. | Moderate. |
| Semicoherent template metrics | Predict mismatch and template density for coherent/semicoherent searches. | Natural way to decide which phase directions matter. | Gives local metric rank/eigenvalues, but not alone a full costed dimension rule. | High. |
| Signal-processing analogues | Chirp/polynomial-phase transforms index spectra by chirp parameters. | Very close to `zeta = {f, fdot, ...}`. | Model-order selection exists in parts of the literature, but cost-sensitivity mapping must be checked. | Very high. |

## StackSlide

**Established.** StackSlide is a standard semicoherent continuous-wave method. Brady and Creighton describe dividing data into segments, Fourier transforming each segment, computing powers, and summing spectra after correcting for modeled frequency evolution. Source: Brady & Creighton, “Searching for periodic sources with LIGO. II: Hierarchical searches,” arXiv:gr-qc/9812014, later Phys. Rev. D 61, 082001.  
Link: https://arxiv.org/abs/gr-qc/9812014

Prix and Shaltev directly address optimal StackSlide parameters at fixed computing cost under ideal data conditions. Their analysis includes coherent time, number of segments, coarse/fine grid mismatch, and sensitivity scaling. Source: Prix & Shaltev, “Search for Continuous Gravitational Waves: Optimal StackSlide method at fixed computing cost,” arXiv:1201.4321, Phys. Rev. D 85, 084010.  
Link: https://arxiv.org/abs/1201.4321

Shaltev later extends this style of optimization to realistic data gaps and changing noise.  
Link: https://arxiv.org/abs/1510.06427

**Relation to `zeta`.** Ordinary StackSlide is the `zeta = {f}` limiting case, with time represented by segment index and a physical template specifying which frequency bin to read in each segment.

**Does it answer the user’s question?** Partially. It answers how to choose `T_coh`, mismatch, grid densities, and data selection for a fixed-compute semicoherent frequency-track search. It does not obviously answer whether the coherent chunk output should be one-dimensional in frequency or higher-dimensional in chirp coordinates.

**Implication.** Any final report should use StackSlide optimization as the baseline cost-sensitivity framework. A proposed higher-dimensional `zeta` method must beat StackSlide at equal false alarm and equal compute, not merely recover more coherent power.

## PowerFlux

**Established.** PowerFlux is a semicoherent CW method based on weighted sums of short Fourier transform powers. In the LIGO S4 all-sky CW paper, StackSlide, weighted Hough, and PowerFlux are compared; PowerFlux is described as a StackSlide variant where powers are weighted using detector noise and antenna pattern to improve SNR.  
Link: https://arxiv.org/abs/0708.3818

Mendell and Wette discuss generalized PowerFlux methods for parameter estimation from Fourier-transformed strain segments.  
Link: https://arxiv.org/abs/0710.4362

**Relation to `zeta`.** PowerFlux is important because the `zeta` statistic should probably not be an unweighted sum of powers. For this codebase’s 5-vector setting, a covariance-weighted vector statistic may be the correct analogue, especially after NUFFT/resampling changes the noise covariance.

**Does it choose `dim(zeta)`?** Not in the sense asked here. It improves the semicoherent combination weights and amplitude/polarization handling, not the dimensionality of local coherent spectra.

**Implication.** Treat equal-power sums as a baseline only. The serious comparison should include PowerFlux-like weighting or a covariance-weighted likelihood statistic.

## Weave

**Established.** Weave is a semicoherent CW search implementation using parameter-space metrics and optimal template-bank lattices. Wette, Walsh, Prix, and Papa describe Weave as combining semicoherent search methods with metric template banks at the correct resolution and optimal lattices to minimize template count and computational cost.  
Link: https://arxiv.org/abs/1804.03392

**Relation to `zeta`.** This is one of the closest GW prior-art overlaps. It already addresses semicoherent template placement, metric resolution, and computational cost.

**Important distinction.** Weave appears to grid the physical CW parameter space and uses coherent/semicoherent metrics to place templates efficiently. The `idea.md` proposal asks a different but adjacent question: can one precompute reusable spectra in a lower- or higher-dimensional chunk-local coordinate space `zeta`, then sum along tracks induced by physical templates `theta`?

**Does it choose `dim(zeta)`?** It chooses dimensions and resolutions of a semicoherent physical template bank. It does not obviously optimize the dimension of reusable coherent-chunk spectra independent of physical parameters. This needs full-paper review.

**Novelty risk.** Very high. Any claim of novelty must explicitly distinguish `zeta`-space intermediate spectra from Weave’s metric template banks.

## Semicoherent Template Metrics

**Established.** Semicoherent metrics quantify mismatch and template density for searches that combine coherent segments incoherently.

Pletsch derives an analytical semicoherent metric for CW searches using new parameter-space coordinates and studies how incoherent combination increases metric resolution, especially in frequency derivatives.  
Link: https://arxiv.org/abs/1005.0395

Owen introduced differential-geometric template spacing for inspiral searches.  
Link: https://arxiv.org/abs/gr-qc/9511032

Owen and Sathyaprakash estimate template counts, computing power, and storage for inspiral matched filtering.  
Link: https://arxiv.org/abs/gr-qc/9808076

Balasubramanian, Sathyaprakash, and Dhurandhar apply differential geometry to chirp detection and parameter estimation.  
Link: https://arxiv.org/abs/gr-qc/9508011

**Relation to `zeta`.** Metrics are the cleanest established mathematical tool for choosing which directions matter. If the coherent-chunk residual phase has a metric with rapidly decaying eigenvalues, that gives a natural candidate dimension.

**Does it choose `dim(zeta)`?** It can identify local important directions and grid densities, but metric truncation alone is not the final answer. The user’s question requires adding compute, memory, interpolation, and false-alarm threshold penalties.

**Actionable criterion from this literature.** Use the metric or SVD to propose candidate dimensions, then keep dimension `d` only if the final search sensitivity improves at fixed computational budget.

## Hough, Radon, and Loosely Coherent Searches

**Established.** Hough-transform CW searches and Radon-like track accumulation are established ways to detect tracks in time-frequency maps. The LIGO S4 paper explicitly includes weighted Hough alongside StackSlide and PowerFlux.  
Link: https://arxiv.org/abs/0708.3818

Dergachev’s loosely coherent method is especially relevant because it targets sets of well-modeled signals spanning finite-dimensional manifolds and is motivated by computationally limited CW searches.  
Link: https://arxiv.org/abs/1110.3297

**Relation to `zeta`.** These methods support the view that “sum along modeled tracks in a transformed space” is established. Higher-dimensional `zeta` spectra may be a generalized Radon/Hough construction.

**Does it choose `dim(zeta)`?** Not obviously. Loosely coherent searches relax phase coherence across a finite-dimensional manifold, which is related but not identical to deciding the dimension of a reusable coherent spectrum.

## MBTA and Multiband Inspiral Searches

**Established.** MBTA is a compact-binary coalescence search pipeline that uses a multiband decomposition to reduce filtering cost. A recent O4-era MBTA paper describes the Multi-Band Template Analysis pipeline and its observing-run configuration.  
Link: https://arxiv.org/abs/2501.04598

Earlier MBTA references should be verified. Search targets include Beauville et al. and LVK low-latency CBC pipeline papers. Zotero status unknown.

**Relation to `zeta`.** MBTA is relevant because it exploits the structure of chirping inspirals to reduce cost by changing representation across frequency bands. This is conceptually adjacent to choosing different chunk-local coordinates or dimensionalities as the signal evolves.

**Does it choose `dim(zeta)`?** No direct evidence. It is closer to matched-filter acceleration than semicoherent power accumulation in a higher-dimensional local spectrum.

**Implication.** MBTA may provide useful cost-model ideas, but it is not currently the primary novelty blocker.

## Reduced Order Quadrature

**Established.** Reduced order quadrature and reduced-basis methods accelerate waveform generation and likelihood evaluation.

Canizares, Field, Gair, and Tiglio introduce ROQ for compressed likelihood evaluations in GW parameter estimation.  
Link: https://arxiv.org/abs/1304.0462

Field, Galley, Hesthaven, Kaye, and Tiglio describe reduced-basis/surrogate waveform construction using greedy bases and empirical interpolation.  
Link: https://arxiv.org/abs/1308.3565

Pürrer builds SVD-based frequency-domain reduced order models for aligned-spin compact binaries.  
Link: https://arxiv.org/abs/1402.4146

**Relation to `zeta`.** ROQ chooses a reduced basis size for waveform accuracy. This is directly relevant to “how many basis directions are needed,” but its objective is fast waveform/likelihood evaluation, not semicoherent spectrum construction.

**Does it choose `dim(zeta)`?** Not directly. It chooses representation dimension under an approximation norm. A `zeta` search needs a detection-sensitivity objective that includes trials factors and search cost.

**Implication.** ROQ provides algorithms and error-control language, but the project must translate basis truncation error into coherent mismatch and semicoherent detection loss.

## PCA / Dimensionality Reduction

**Established.** SVD/PCA has been used to compress compact-binary template banks.

Cannon et al. apply SVD to compact-binary coalescence GW signals and find that truncated SVD can reduce the number of filters required for a region of parameter space while controlling reconstruction loss.  
Link: https://arxiv.org/abs/1005.0012

Cannon, Hanna, and Keppel discuss SVD-based interpolation of compact-binary waveforms.  
Link: https://arxiv.org/abs/1108.5618

Cannon et al. also use computationally practical filtering strategies for early-warning CBC detection.  
Link: https://arxiv.org/abs/1107.2665

**Relation to `zeta`.** This is the most direct literature support for the human feedback item: run reduced-basis/SVD/PCA analysis on equal-mass, non-spinning 3.5PN waveforms and compare the optimal one-dimensional basis against the 0PN direction.

**Established vs conjectural.**

- Established: SVD/PCA can identify low-rank waveform or phase subspaces.
- Established: truncation error can be related to expected signal loss in at least some matched-filter settings.
- Conjecture: for this project’s equal-mass non-spinning 3.5PN chunks, the first residual phase basis is close to the 0PN phase direction.
- Conjecture: spin up to `chi = 0.2` introduces one or more additional significant directions.

**Caution.** SVD/PCA “optimality” is norm-dependent. The basis depends on the waveform family, mass/spin range, sampling distribution, time/frequency interval, whitening, windowing, and nuisance directions projected out.

## Signal-Processing Analogues

### Fast Chirp Transform

**Established.** Jenet and Prince define a fast chirp transform for detecting variable-frequency signals, motivated partly by GW inspirals.  
Link: https://arxiv.org/abs/gr-qc/0012029

**Relation to `zeta`.** This is very close to coherent spectra indexed by chirp parameters. A `zeta = {f, fdot}` or `{f, beta}` coherent spectrum may be a special case or approximation to a chirp transform.

**Does it choose `dim(zeta)`?** The fast chirp transform literature should be checked for model-order and cost-scaling rules. From the verified source alone, it establishes the transform analogue but not the full semicoherent dimension-selection rule.

### Polynomial-Phase Transforms and High-Order Ambiguity Functions

**Established.** Polynomial-phase signal estimation is a mature signal-processing topic. Relevant names include polynomial-phase transform, high-order ambiguity function, product high-order ambiguity function, and polynomial Wigner-Ville distribution.

Search targets requiring full verification:

- Peleg and Porat, polynomial-phase signal estimation/classification. Citation needed.
- Barbarossa, Scaglione, and Giannakis, “Product high-order ambiguity function for multicomponent polynomial-phase signal modeling,” IEEE Transactions on Signal Processing, 1998. Citation details need verification.
- Boashash and O’Shea, polynomial Wigner-Ville distributions. Citation details need verification.
- Cohen, *Time-Frequency Analysis*, Prentice Hall, 1995. Citation details known but not inspected here.
- Boashash, ed., *Time-Frequency Signal Analysis and Processing*. Citation details need edition verification.

**Relation to `zeta`.** This is almost exactly the generic signal-processing version of spectra over `{f, fdot, fddot, ...}`.

**Does it choose `dim(zeta)`?** Likely partly. Polynomial-phase literature includes order estimation and parameter estimation, but the next agent must verify whether it contains a cost-constrained detection-sensitivity rule comparable to the one desired here.

### Chirplet Transforms

**Established.** Chirplet transforms are dictionaries indexed by time, frequency, chirp rate, duration, and sometimes higher-order deformations.

Search targets:

- Mann and Haykin, “The Chirplet Transform: Physical Considerations,” IEEE Transactions on Signal Processing, 1995. Citation details should be verified.
- Baraniuk and Jones chirplet transform papers. Citation needed.
- Mihovilovic and Bracewell, adaptive chirplet representation. Citation needed.

**Relation to `zeta`.** Chirplets are a high-dimensional time-frequency-chirp representation. They are a novelty risk for broad claims about higher-dimensional spectra.

**Difference.** Chirplet methods are often signal representation/dictionary methods, not necessarily semicoherent PN-template track sums with fixed false-alarm and compute constraints.

## Compact-Binary PN and Chirp-Time Coordinates

**Established.** Inspiral template-bank coordinates have long used physically motivated combinations such as chirp times and metric-adapted coordinates.

Tanaka and Tagoshi propose new coordinates for hierarchical searches of inspiraling binaries to reduce computational cost.  
Link: https://arxiv.org/abs/gr-qc/0001090

Owen and Sathyaprakash treat computational cost and template placement for inspiral matched filtering.  
Link: https://arxiv.org/abs/gr-qc/9808076

Blanchet’s review covers PN phasing and compact-binary inspiral theory through high PN order.  
Link: https://arxiv.org/abs/gr-qc/0202016

Arun, Iyer, Sathyaprakash, and Sundararajan study 3.5PN non-spinning phasing and parameter estimation.  
Link: https://arxiv.org/abs/gr-qc/0411146

**Relation to `zeta`.** Chirp-time and PN metric coordinates are a strong precedent for replacing raw physical parameters with better-conditioned phase coordinates. They support the idea that `zeta` should probably be an orthogonalized or metric-adapted phase basis, not necessarily raw PN coefficients.

**Does it choose `dim(zeta)`?** For matched filtering, the intrinsic parameter-space dimension is set by the waveform model. For the proposed semicoherent spectra, the question is different: how many local phase directions should be indexed coherently before incoherent combination? That specific costed choice remains open.

## Equal-Mass Non-Spinning 3.5PN Case

**Established background.** For non-spinning equal-mass binaries, the intrinsic physical parameter family is effectively one-dimensional if the mass ratio is fixed exactly at `eta = 1/4`: the total mass or chirp mass controls the inspiral phasing, aside from extrinsic time and phase shifts. However, the waveform’s local phase curve in a chunk can still be poorly approximated by a chosen one-parameter coordinate if the coordinate is not aligned with the dominant residual direction over the selected mass/frequency/chunk range.

**Literature implication.** PN theory suggests that the 0PN/Newtonian chirp term is dominant in accumulated phase, and chirp-mass-like coordinates are physically privileged. This makes the user’s intuition plausible: for a one-parameter non-spinning equal-mass family, a 0PN-like coordinate may be near-optimal.

**But this is not established for the project.** The statement “0PN is the best one-dimensional chunk-local `zeta` coordinate” depends on:

- whether frequency is already included in `zeta`;
- whether constant phase and linear frequency are projected out;
- the chunk duration `T_coh`;
- the frequency range;
- the mass range;
- the time-domain versus frequency-domain convention;
- the inner product/window/noise weighting;
- whether the basis is required to be linear in parameters or may be nonlinear.

**Recommended test.** Build phase residuals from equal-mass non-spinning 3.5PN waveforms after projecting out constant phase and whatever baseline coordinates are always included, then compare:

- 0PN residual direction;
- first SVD/PCA residual direction;
- metric eigenvector direction;
- higher PN-inspired directions;
- mismatch versus retained dimension.

**Expected but conjectural outcome.** For narrow mass/frequency ranges and moderate `T_coh`, the first SVD direction may align strongly with 0PN. For broader ranges or longer chunks, curvature of the 3.5PN phase family may require additional directions even though the physical family is one-dimensional.

## Equal-Mass Spinning 3.5PN Case with `chi <= 0.2`

**Established background.** Spin affects compact-binary inspiral phasing through spin-orbit and spin-spin terms. Kidder and Kidder-Will-Wiseman are core early references on spin effects.  
Links: https://arxiv.org/abs/gr-qc/9506022 and https://arxiv.org/abs/gr-qc/9211025

Poisson and Will study parameter estimation including spin-related parameters in 2PN waveforms.  
Link: https://arxiv.org/abs/gr-qc/9502040

Brown, Harry, Lundgren, and Nitz show that neglecting spin in BNS searches can cause SNR loss and present aligned-spin template-bank methods.  
Link: https://arxiv.org/abs/1207.6406

Dal Canton et al. study aligned-spin NSBH search implementation and find improved sensitive volume relative to non-spinning templates in relevant spin ranges.  
Link: https://arxiv.org/abs/1405.6731

**Relation to `zeta`.** Allowing spin adds at least one physical parameter if equal component aligned spin is varied. If spins may be anti-aligned or unequal, the dimensionality can increase further.

**Assumption needing confirmation.** “Spinning up to `chi = 0.2`” must be defined as one of:

- equal aligned component spins, `chi1 = chi2 in [0, 0.2]`;
- equal signed aligned spins, `chi1 = chi2 in [-0.2, 0.2]`;
- independent aligned spins;
- generic precessing spins.

The dimension-selection answer changes substantially across these cases.

**Conjecture.** For equal-mass binaries with equal aligned spins limited to `0 <= chi <= 0.2`, spin may add one dominant residual direction, but it may be partially degenerate with mass/chirp-time directions. For `|chi| <= 0.2`, the basis may need to capture antisymmetric curvature around zero spin. For precessing spins, this literature review is not sufficient; the problem becomes higher-dimensional and amplitude/phase modulations matter.

**Recommended test.** Compare:

- non-spinning SVD basis applied to spinning waveforms;
- spinning SVD basis;
- 0PN plus leading spin-PN coordinate;
- first two or three metric eigenvectors;
- mismatch and cost versus retained dimension.

## How To Choose `dim(zeta)`: Literature-Derived Rule

No reviewed source provides the exact desired rule in one package. The synthesis from StackSlide optimization, metrics, PowerFlux weighting, ROQ/SVD, and chirp transforms is:

1. Define a chunk-local phase model `Phi_loc(u; zeta)`.
2. For each candidate dimension `d`, compute coherent mismatch from residual phase after fitting the best `d` coordinates.
3. Estimate the number of required grid points in `zeta` using a metric, SVD truncation error, or empirical mismatch maps.
4. Estimate spectrum-generation cost, track-summing cost, interpolation cost, memory, and effective trials factor.
5. Convert mismatch and threshold penalty into detectable amplitude at fixed false-alarm probability.
6. Choose the dimension minimizing detectable amplitude under compute and memory constraints.

In shorthand:

```math
d_\zeta^\star
=
\arg\min_d h_{\min}(d)
\quad
\text{subject to}
\quad
C(d) \le C_{\rm budget},
\quad
M(d) \le M_{\rm budget},
\quad
P_{\rm FA}(d) \le P_{\rm FA}^{\rm target}.
```

This rule is a synthesis, not a directly established result from one cited paper.

## Can The Basis Be Derived Analytically?

**Polynomial phase.** Likely yes in idealized cases. For

```math
\phi(u) = \sum_j a_j u^j,
```

and a simple window/noise inner product, orthogonal polynomials or Gram-matrix eigenvectors provide analytic or semi-analytic basis directions. This connects directly to polynomial-phase transforms and high-order ambiguity methods.

**PN phase.** Partly. If the PN phase is written as a linear combination of basis functions over a chunk, one can construct a weighted Gram matrix of PN phase functions, project out nuisance directions, and diagonalize it. That gives metric eigen-coordinates. For equal-mass non-spinning waveforms, the physical family is only one-dimensional, but the best local coordinate depends on the chosen projection and approximation class.

**0PN expectation.** The 0PN direction is physically motivated and likely dominant in many regimes, but “best one-dimensional `zeta` coordinate” is not guaranteed. It must be tested against a metric/SVD basis under the exact chunk, frequency, and mass ranges used by the project.

## Novelty Risks

| Risk | Overlap | Severity | What to check |
|---|---|---:|---|
| Fast chirp transform | Coherent spectra over variable-frequency/chirp parameters. | Very high | Compare `{f, beta}` and `{f, fdot}` spectra to FCT definitions and scaling. |
| Polynomial-phase transforms | Spectra/estimators over polynomial phase coefficients. | Very high | Check model-order selection and computational scaling literature. |
| Weave | Semicoherent metric template banks with cost-aware optimal lattices. | Very high | Determine whether reusable `zeta` spectra differ materially from Weave’s template-bank machinery. |
| StackSlide optimization | Fixed-compute semicoherent sensitivity optimization. | High | Use as baseline; do not claim novelty for semicoherent track sums. |
| Semicoherent metrics | Dimension/resolution of parameter-space directions. | High | Determine whether metric eigenvalue truncation already answers `dim(zeta)` in CW literature. |
| Hough/Radon/chirplet methods | Track integration in higher-dimensional time-frequency-chirp spaces. | High | Check whether the proposed construction is a generalized Radon/chirplet transform. |
| ROQ/reduced basis/SVD | Low-dimensional waveform bases and rank truncation. | Moderate | Distinguish representation-rank selection from semicoherent search-coordinate selection. |
| MBTA/multiband filtering | Cost-efficient inspiral decomposition. | Moderate | Borrow cost ideas; probably not direct prior art for `zeta` spectra. |
| PowerFlux | Weighted power combination. | Moderate | Use weighted/covariance statistic; do not claim novelty for weighting. |

## Candidate Papers To Verify / Add To Zotero

Zotero status unknown for all.

### Highest Priority

1. Brady & Creighton, StackSlide/hierarchical CW search: https://arxiv.org/abs/gr-qc/9812014
2. Prix & Shaltev, optimal StackSlide at fixed cost: https://arxiv.org/abs/1201.4321
3. Wette, Walsh, Prix, Papa, Weave: https://arxiv.org/abs/1804.03392
4. Pletsch, semicoherent CW metric: https://arxiv.org/abs/1005.0395
5. Jenet & Prince, fast chirp transform: https://arxiv.org/abs/gr-qc/0012029
6. Cannon et al., SVD for CBC waveforms: https://arxiv.org/abs/1005.0012
7. Canizares et al., ROQ: https://arxiv.org/abs/1304.0462
8. Field et al., reduced-basis/surrogate models: https://arxiv.org/abs/1308.3565
9. Arun et al., 3.5PN non-spinning phasing: https://arxiv.org/abs/gr-qc/0411146
10. Kidder spin effects: https://arxiv.org/abs/gr-qc/9506022

### Secondary Priority

1. LIGO S4 all-sky CW comparison of StackSlide/Hough/PowerFlux: https://arxiv.org/abs/0708.3818
2. Mendell & Wette generalized PowerFlux: https://arxiv.org/abs/0710.4362
3. Dergachev loosely coherent searches: https://arxiv.org/abs/1110.3297
4. Tanaka & Tagoshi hierarchical inspiral coordinates: https://arxiv.org/abs/gr-qc/0001090
5. Owen template spacing: https://arxiv.org/abs/gr-qc/9511032
6. Owen & Sathyaprakash cost/template placement: https://arxiv.org/abs/gr-qc/9808076
7. Pürrer aligned-spin reduced order models: https://arxiv.org/abs/1402.4146
8. Brown et al. aligned-spin BNS banks: https://arxiv.org/abs/1207.6406
9. Dal Canton et al. aligned-spin NSBH search: https://arxiv.org/abs/1405.6731
10. MBTA O4 pipeline: https://arxiv.org/abs/2501.04598

### Signal-Processing Search Targets

These require full bibliographic verification:

1. Peleg and Porat, polynomial-phase signal estimation/classification.
2. Barbarossa, Scaglione, and Giannakis, product high-order ambiguity function.
3. Boashash and O’Shea, polynomial Wigner-Ville distributions.
4. Mann and Haykin, chirplet transform.
5. Baraniuk and Jones, chirplet/time-frequency dictionaries.
6. Cohen, *Time-Frequency Analysis*.
7. Boashash, *Time-Frequency Signal Analysis and Processing*.

## Recommendations For Next Agents

1. Read Weave, Pletsch semicoherent metrics, and Prix-Shaltev StackSlide before making novelty claims.
2. Read fast chirp transform and polynomial-phase transform literature before claiming that `{f, beta}` spectra are new.
3. Treat 0PN near-optimality as a conjecture. Test it with SVD/PCA or metric eigenvectors on the exact equal-mass non-spinning 3.5PN phase family.
4. Define the spin convention before studying `chi <= 0.2`.
5. Compare dimensions using detectable amplitude at fixed compute and false alarm, not recovered power alone.
6. Inspect `CODEX.md` and Zotero in a later pass with filesystem/library access.