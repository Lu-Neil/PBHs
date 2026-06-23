# Scratchpad Plan: Semicoherent `zeta`-Space Search Design

## Core Aim

Turn `idea.md` plus human feedback into a concrete research plan for answering:

> How should one choose the number and form of coherent-chunk parameters `zeta` in a semicoherent search, to optimize sensitivity versus computational cost for modeled gravitational-wave chirps?

The plan should not claim novelty for semicoherent searches. It should focus on the tradeoff between richer coherent-chunk spectra and downstream semicoherent cost.

## Core Claims To Test

Established from `idea.md` and prior work summary:

- Standard semicoherent searches often combine power along tracks in frequency-time representations.
- Richer coherent-chunk coordinates `zeta`, such as `{f, beta}` or PN-like local phase parameters, may reduce coherent mismatch.
- Adding `zeta` dimensions increases spectrum-generation cost, memory cost, interpolation cost, and trials factors.

Conjectures to mark explicitly:

- There is an optimal `dim(zeta)` for a fixed compute budget and target sensitivity.
- For equal-mass non-spinning 3.5PN waveforms, the best one-parameter reduced coordinate may be close to the 0PN chirp parameter.
- SVD/PCA or reduced-basis coordinates may outperform physically motivated 0PN coordinates once more than one parameter is allowed.
- Allowing spin up to `chi = 0.2` may increase intrinsic dimensionality and shift the optimal `zeta`.

## Central Research Question

How can we choose the dimensionality and coordinate basis of `zeta` for coherent chunks in a semicoherent search so that the resulting statistic gives the best sensitivity at fixed computational cost?

## Subquestions

1. Generic signal-processing question:
   - Given a parameterized phase model `phi(t; theta)`, how should one choose local coherent coordinates `zeta(theta, t_k)`?
   - How does the optimal `dim(zeta)` depend on `T_coh`, parameter-space volume, mismatch tolerance, and compute budget?
   - Is this equivalent to known chirp-transform, polynomial-phase, Radon/Hough, or reduced-basis constructions?

2. Cost-sensitivity tradeoff:
   - How does adding each extra `zeta` dimension affect coherent mismatch?
   - How does it affect spectrum construction cost, memory, interpolation, track summation, and false-alarm threshold?
   - What criterion should decide whether an extra dimension is worth keeping?

3. 3.5PN equal-mass non-spinning case:
   - What is the intrinsic dimensionality of the local phase family over relevant `T_coh`?
   - How close is the optimal one-dimensional basis to the 0PN waveform parameter?
   - Do SVD/PCA/reduced-basis coordinates produce materially better low-dimensional approximations?

4. 3.5PN equal-mass spinning case with `chi <= 0.2`:
   - How much additional phase variation is introduced by spin?
   - Does spin require extra `zeta` dimensions for the same mismatch?
   - Are spin effects captured by perturbations of the non-spinning basis or by genuinely new basis directions?

## Required Derivations

1. Formal semicoherent statistic:
   - Define data `x(t) = n(t) + h(t; theta)`.
   - Split into chunks centered at `t_k`.
   - Define coherent chunk output:
     ```math
     X_k(\zeta) = \int_{I_k} x(t) e^{-i \Phi_{\rm loc}(t-t_k; \zeta)} dt
     ```
   - Define semicoherent track statistic:
     ```math
     S(\theta) = \sum_k w_k P_k[\zeta_k(\theta)]
     ```

2. Mismatch expansion:
   - Expand exact phase in each chunk around local model.
   - Express coherent mismatch in terms of residual phase variance.
   - Show how omitted phase directions scale with `T_coh`.

3. Dimensionality criterion:
   - Derive a rule based on eigenvalues or singular values of the phase family.
   - Candidate criterion: retain basis directions until residual mismatch is below tolerance.
   - Then add compute penalty and trials-factor penalty.

4. Cost model:
   - Spectrum cost:
     ```math
     C_{\rm spectra} \sim N_{\rm seg} N_\zeta c_{\rm coh}
     ```
   - Track cost:
     ```math
     C_{\rm tracks} \sim N_\theta N_{\rm seg} c_{\rm lookup}
     ```
   - Memory:
     ```math
     M \sim N_{\rm seg} N_\zeta B
     ```
   - Include interpolation scaling and trials-factor effects.

5. Optimization statement:
   ```math
   (d_\zeta, T_{\rm coh}, \text{basis}) =
   \arg\min h_{\min}
   ```
   subject to fixed compute, memory, false-alarm probability, and allowed mismatch.

## Required Literature Searches

Do not invent citations. Search and verify:

- StackSlide optimization and semicoherent metric literature.
- PowerFlux and weighted semicoherent power sums.
- Weave and semicoherent template-bank construction.
- Fast chirp transform and polynomial-phase transform literature.
- Hough/Radon-style track accumulation.
- Reduced-order quadrature, reduced-basis, SVD/PCA waveform bases.
- Chirp-time coordinates for compact binary inspirals.
- MBTA and multiband/multibank compact-binary search methods.
- Literature on intrinsic dimensionality of aligned-spin PN waveform families.

Also inspect `/home/neil-lu/Dropbox/PBHs/Codebase/CODEX.md` and note papers not already in Zotero, without claiming Zotero status unless actually checked.

## Required Simulations

1. Toy polynomial-phase study:
   - Compare `zeta = {f}`, `{f, fdot}`, `{f, fdot, fddot}`.
   - Measure mismatch, runtime, memory, and detection probability versus `T_coh`.

2. Equal-mass non-spinning 3.5PN study:
   - Generate waveform phase family over chosen mass/frequency ranges.
   - Build local phase residual matrix after removing constant phase and time shift terms.
   - Run SVD/PCA.
   - Compare:
     - 0PN one-parameter coordinate
     - best one-dimensional SVD coordinate
     - best two- and three-dimensional bases
   - Measure mismatch versus `T_coh`.

3. Equal-mass spinning `chi <= 0.2` study:
   - Repeat SVD/PCA with spin included.
   - Compare basis directions to non-spinning case.
   - Estimate extra dimension required for same mismatch.

4. Semicoherent cost-sensitivity experiment:
   - For candidate bases and dimensions, compute expected sensitivity at fixed compute.
   - Include empirical null distribution or trials-factor estimate where feasible.
   - Compare against ordinary frequency-time StackSlide.

## Likely Bottlenecks

- Defining the waveform parameter range narrowly enough for a meaningful basis study.
- Avoiding confusion between physical parameters `theta` and coherent coordinates `zeta`.
- Trials factors may erase apparent gains from richer spectra.
- Memory bandwidth may dominate before FLOP count does.
- SVD/PCA basis may depend strongly on chosen inner product, frequency range, and `T_coh`.
- Spin effects may be small over short chunks but important over longer `T_coh`.
- Reduced-basis coordinates may be accurate but hard to evaluate as reusable spectra.
- Need to avoid claiming optimality relative to full matched filtering.

## Failure Modes And Checks

- Shallow claim: “more dimensions improves sensitivity.”
  - Check: include cost, memory, and false-alarm threshold.

- Shallow claim: “0PN is optimal.”
  - Check: compare directly against SVD/PCA basis.

- Shallow claim: “SVD basis is better.”
  - Check: compare at equal grid density and compute cost.

- Wrong novelty claim.
  - Check: explicitly compare with chirp transforms, polynomial-phase transforms, Weave, StackSlide, and reduced-basis methods.

- Misleading simulations.
  - Check: report parameter ranges, waveform assumptions, noise assumptions, and mismatch definitions.

## Proposed Report Structure

1. Executive summary
   - State the narrowed research question and main proposed tests.

2. Problem formulation
   - Define `theta`, `zeta`, coherent chunks, spectra, tracks, and semicoherent statistic.

3. Prior art and novelty risks
   - StackSlide, PowerFlux, Weave, chirp transforms, polynomial-phase methods, reduced basis, MBTA.

4. Generic dimensionality criterion
   - Mismatch expansion, local phase manifold, SVD/PCA/eigenbasis interpretation.

5. Computational cost model
   - Spectrum generation, track summation, interpolation, memory, trials factor.

6. Sensitivity model
   - Semicoherent detection scaling, mismatch losses, threshold penalties.

7. Case study A: equal-mass non-spinning 3.5PN waveforms
   - 0PN versus SVD/PCA basis.
   - Required `dim(zeta)` versus `T_coh`.

8. Case study B: equal-mass spinning 3.5PN waveforms with `chi <= 0.2`
   - Added intrinsic dimensionality from spin.
   - Whether spin requires new coherent coordinates.

9. Simulation plan and validation metrics
   - Toy polynomial phase, PN phase studies, semicoherent detection experiments.

10. Bottlenecks, assumptions, and decision criteria
   - What evidence would support or falsify using higher-dimensional `zeta`.

11. Recommended next steps
   - Implement basis study first, then costed semicoherent comparison.