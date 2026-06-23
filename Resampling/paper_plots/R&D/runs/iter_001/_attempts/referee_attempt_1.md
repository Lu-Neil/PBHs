# Referee Report: Higher-Dimensional Semicoherent `zeta` Searches

## Audit Metadata

- Agent: `referee`
- Attempt: 1
- Source of truth: `idea.md` plus the human feedback embedded in the task prompt
- Prior artifacts considered: planner, literature review, theory/optimization reports included in the prompt
- External literature verification in this artifact: none
- Shell/script execution: none, per wrapper instruction
- Status: skeptical review, not a final scientific assessment

## Summary Judgment

The defensible idea is narrower than the current framing.

A plausible research question is:

> For a modeled signal family, what chunk-local phase coordinates `zeta` should be retained so that a semicoherent search minimizes detectable amplitude at fixed compute, memory, and false-alarm probability?

That is a useful question. But the proposal is not yet clearly distinct from existing semicoherent matched-filter banks, StackSlide-like searches, chirp transforms, polynomial-phase transforms, metric template banks, or reduced-basis waveform compression. The burden of proof is not to show that extra `zeta` dimensions recover more coherent power. That is expected. The burden is to show that they improve the final sensitivity-compute tradeoff after trials factors, grid costs, memory costs, interpolation loss, and noise normalization are included.

## Strongest Objections

### 1. The proposal may be a reparameterized known method

The idea of summing semicoherent power along modeled tracks is established prior art. The idea of coherent transforms indexed by chirp or polynomial-phase parameters is also established in signal processing. The proposed `zeta` spectra could be:

- StackSlide with extra local phase parameters;
- a semicoherent matched-filter bank written in intermediate coordinates;
- a fast chirp transform or polynomial-phase transform variant;
- a generalized Radon/Hough accumulation in a higher-dimensional transform space;
- a metric-template-bank construction with a different implementation layout.

This does not make the project uninteresting, but it weakens any novelty claim. The report should avoid claiming a new detection paradigm unless it first distinguishes the method from these baselines.

### 2. Extra dimensions may only move the cost elsewhere

Adding `zeta` dimensions can reduce local phase mismatch, but it also increases:

- number of coherent-spectrum grid points;
- memory footprint;
- interpolation cost;
- track-lookup cost;
- effective number of trials;
- calibration burden;
- implementation complexity.

The method could look better in recovered power but worse in detectable amplitude at fixed false alarm. A high-dimensional spectrum is not automatically cheaper than a semicoherent matched-filter bank; it may simply precompute many local filters and store their outputs.

### 3. The objective function is still underdefined

The human feedback correctly says the target is not full matched-filter optimality, but a better sensitivity-versus-computing-cost tradeoff. That still needs a precise scalar objective.

A referee would require something like:

```math
d_\zeta^\star
=
\arg\min_d h_{\min}(d)
```

subject to:

```math
C(d) \le C_{\rm budget},
\qquad
M(d) \le M_{\rm budget},
\qquad
P_{\rm FA,total}(d) \le P_{\rm FA}^{\rm target}.
```

Without this, “better” is ambiguous. Better recovered coherent power, lower mismatch, lower runtime, and lower detectable amplitude are different claims.

### 4. PCA/SVD rank is not the same as optimal search dimension

Reduced-basis, SVD, or PCA analysis can identify low-rank phase structure under a chosen norm. That is useful. But the number of retained singular vectors is not automatically the number of `zeta` dimensions a search should use.

The search dimension also depends on:

- grid density in each retained coordinate;
- parameter-space prior or coverage requirement;
- false-alarm threshold;
- correlations between templates;
- memory bandwidth;
- interpolation scheme;
- noise weighting;
- whether the basis generalizes to held-out waveforms.

A basis that is optimal for phase reconstruction may be suboptimal for detection at fixed compute.

### 5. The equal-mass non-spinning case may be less informative than it appears

For exactly equal-mass, non-spinning 3.5PN waveforms, the intrinsic physical family is essentially one-dimensional after fixing mass ratio and spin. If the study finds that one dominant coordinate suffices, that may be unsurprising.

The real question is subtler: after projecting out phase, time, and local frequency, does the residual 3.5PN phase family require more than a 0PN-like direction over realistic chunk durations and frequency ranges?

If not, the result supports the 0PN construction but does not establish a general dimension-selection method.

### 6. The spinning case is currently ambiguous

The phrase `chi <= 0.2` is underspecified. The dimension answer depends on whether this means:

- equal aligned spins, `chi1 = chi2 in [0, 0.2]`;
- equal signed aligned spins, `chi1 = chi2 in [-0.2, 0.2]`;
- independent aligned component spins;
- generic precessing spins.

These are different waveform families with different intrinsic dimensions. Any report that does not define this convention cannot make a meaningful claim about the spinning case.

## Unclear Assumptions

### Definition of `zeta`

It is unclear whether `zeta` means:

- a coordinate system for coherent demodulation;
- a set of spectral axes actually gridded and stored;
- a local projection of physical parameters;
- a reduced-basis coefficient vector;
- a mixture of segment label, frequency, and chirp parameters.

This must be fixed. In particular, `t` should not be counted as a searched `zeta` dimension unless the method actually computes spectra over multiple candidate epochs. In ordinary semicoherent searches, time is a segment label.

### Relationship between `theta` and `zeta`

The proposal should clearly separate:

- `theta`: physical signal parameters;
- `zeta_k(theta)`: chunk-local coordinates predicted by a physical template;
- grid coordinates used to store coherent spectra;
- interpolation coordinates used during track summation.

Confusing these will lead to incorrect template counts and cost estimates.

### Waveform convention

The 3.5PN study needs a precise phase convention:

- time-domain or frequency-domain phasing;
- PN terms included;
- reference frequency or reference time;
- treatment of coalescence time and phase;
- mass range;
- frequency band;
- termination frequency;
- units.

Without these choices, statements about 0PN optimality are not auditable.

### Nuisance projections

The comparison between 0PN and SVD/PCA basis vectors is ill-posed unless the same nuisance directions are projected out in all cases. At minimum, this likely includes:

- constant phase;
- time shift or linear phase;
- local carrier frequency if frequency is always included in `zeta`.

A 0PN direction can look artificially good or bad depending on this projection.

### Noise model

The proposal currently mixes phase-only mismatch reasoning with detection-statistic claims. A phase-only study is acceptable as a first diagnostic, but sensitivity claims require a noise model and a calibrated null distribution.

### Search volume

The optimal dimension depends on parameter-space volume. A narrow mass and spin range may favor low dimension; a broad range may require more coordinates. The proposal should not generalize from one range without stating the range.

## Novelty Concerns

### StackSlide and PowerFlux overlap

Frequency-time power summation along modeled tracks is established. Weighted power summation is also established. A `zeta={f}` limit is therefore not new, and `zeta={f,beta}` must be presented as an extension of that family, not as a new semicoherent principle.

### Weave and semicoherent metric overlap

Metric-based semicoherent searches already optimize template placement, coherent time, mismatch, and computing cost in gravitational-wave contexts. The possible distinction is that the proposed method precomputes reusable chunk-local spectra in `zeta` rather than directly evaluating physical templates. That distinction must be made explicit and tested.

### Chirp-transform and polynomial-phase overlap

A coherent spectrum indexed by `{f, fdot, fddot, ...}` is close to polynomial-phase or chirp-transform methods. A spectrum indexed by `{f, beta}` may be a nonlinear chirp-transform specialization. This is a major novelty risk.

### Reduced-basis overlap

SVD/PCA basis construction for waveform families is established in spirit. The novel part, if any, is not “use SVD.” It is using reduced phase coordinates as axes of reusable coherent spectra and proving that this improves the semicoherent search tradeoff.

### Matched-filter-bank overlap

If each `zeta` grid point corresponds to a local phase model and each physical template selects one grid point per segment, the method may be mathematically equivalent to a semicoherent matched-filter bank with cached segment likelihoods. That may still be computationally useful, but it should be described honestly.

## Possible Normalization Errors

### Frequency units

The local codebase reportedly often uses angular frequency in radians per second. The report must not mix:

```math
\omega
```

and

```math
f
```

without factors of `2 pi`. This is especially dangerous for `beta`, PN chirp parameters, and NUFFT bin definitions.

### Coherent power normalization

The statistic must specify all factors of:

- `T_coh`;
- sampling interval;
- window normalization;
- one-sided versus two-sided PSD;
- real versus complex analytic signal convention;
- FFT/NUFFT normalization.

Otherwise the recovered power cannot be compared across dimensions or coherent times.

### NUFFT noise transfer

The project context explicitly warns that the NUFFT on nonuniform `tau` changes the effective noise PSD. Using `S_n(f0)` alone is likely wrong for chirping tracks. The relevant effective noise should depend on the whole track through the resampling map.

### Independent-bin assumption

The ideal result

```math
P_k \sim \mathrm{Exp}(1)
```

is only valid under restrictive assumptions. It may fail because of:

- colored noise;
- window overlap;
- interpolated bins;
- gapped data;
- nonuniform resampling;
- nearby correlated `zeta` grid points;
- sidereal sideband extraction;
- shared data across templates.

Any threshold based on independent exponentials must be validated empirically.

### 5-vector covariance

For the PBH/5-vector pipeline, the five sidereal bins should not automatically be treated as independent equal-variance measurements. A covariance matrix is needed, especially after NUFFT/resampling and gaps.

### Interpolation normalization

If tracks pass between grid points in `zeta`, interpolation changes both signal amplitude and noise covariance. Interpolation cannot be treated only as a small deterministic mismatch unless its effect on the null distribution is measured.

### Trials factor

The trials factor is not simply the number of physical templates or raw `zeta` bins. Correlations matter. But ignoring the trials factor is worse: extra dimensions can raise the detection threshold enough to erase the gain.

### Window and leakage corrections

Changing `T_coh`, using chirped local templates, or applying nonuniform sampling changes leakage and bin-mismatch behavior. Any Dirichlet-kernel correction used for FFT-like spectra may not carry over unchanged.

## Missing Tests

### Baseline comparisons

The method must be compared against:

1. Frequency-only StackSlide-like accumulation.
2. Weighted PowerFlux-like accumulation where applicable.
3. A semicoherent matched-filter-bank baseline.
4. A chirp-transform or polynomial-phase-transform analogue if the chosen `zeta` is `{f, fdot, ...}` or `{f, beta}`.
5. A fully coherent matched-filter baseline on small parameter volumes, not as the target method, but as a sanity bound.

### Fixed-false-alarm sensitivity

For each candidate `dim(zeta)`, measure:

- null distribution;
- effective trials factor;
- detection threshold;
- detection probability versus injection amplitude;
- detectable amplitude at fixed total false alarm.

Recovered power alone is insufficient.

### Runtime and memory

The proposal needs measured costs, not only asymptotic scaling:

- spectrum construction time;
- track summation time;
- interpolation time;
- peak memory;
- memory bandwidth sensitivity;
- scaling with `T_coh`;
- scaling with `dim(zeta)`.

### Interpolation tests

Required tests:

- signal loss from off-grid tracks;
- covariance introduced by interpolation;
- threshold changes from interpolation;
- dependence on grid spacing and interpolation order.

### SVD/PCA validation

For the reduced-basis analysis:

- train/test split over waveforms;
- documented mass, spin, and frequency ranges;
- documented inner product;
- nuisance directions projected out;
- singular value spectrum;
- mismatch versus retained dimension;
- comparison of first SVD direction with 0PN;
- held-out worst-case mismatch.

### Equal-mass non-spinning 3.5PN tests

Required comparisons:

- frequency only;
- frequency plus 0PN;
- frequency plus best one SVD/PCA direction;
- frequency plus two or more SVD/PCA directions;
- frequency plus PN-inspired higher-order coordinates.

The key output is not just basis overlap. It is whether the improved mismatch justifies the extra grid and threshold cost.

### Equal-mass spinning 3.5PN tests

Before testing, define the spin model. Then compare:

- non-spinning basis applied to spinning waveforms;
- spinning SVD/PCA basis;
- 0PN plus leading spin-related coordinate;
- higher-dimensional PN-inspired bases;
- mismatch and cost versus dimension.

The test should answer whether spin up to `chi=0.2` adds a genuinely new useful coordinate or only perturbs the mass-like direction.

### Robustness tests

Repeat the basis and sensitivity study across:

- multiple `T_coh`;
- multiple mass ranges;
- multiple frequency bands;
- different windows;
- white and colored noise;
- noiseless phase mismatch and noisy detection;
- gapped data if relevant to the final application.

## Is This Genuinely Different From Matched Filtering?

At present, not clearly.

A semicoherent matched-filter bank would compute coherent matched-filter outputs for many local templates in each segment, then combine them incoherently along physical-template tracks. The proposed `zeta` method appears to do something very similar, except that the local templates are organized by coordinates such as `{f,beta}` or SVD/PCA coefficients.

The method becomes genuinely distinct only if the following are true:

1. The local `zeta` spectra are reusable across many physical templates.
2. The `zeta` dimension is lower than the full physical template dimension needed for a comparable local matched-filter bank.
3. The mapping `theta -> zeta_k(theta)` enables cheaper track summation than direct semicoherent filtering.
4. The approximation loss is controlled.
5. The final detectable amplitude is better at equal compute and false alarm.

If these conditions are not demonstrated, the proposal should be framed as an implementation strategy for semicoherent matched filtering, not as a distinct search method.

## Case-Specific Critique

### Equal-Mass Non-Spinning 3.5PN

Established from the problem setup: this case is intended as a test of whether the best one-dimensional chunk-local basis differs from the 0PN waveform.

Skeptical view:

- Since the physical family is one-dimensional after fixing equal mass ratio and zero spin, a dominant one-dimensional basis is expected.
- The meaningful question is whether the 0PN coordinate remains near-optimal after projecting out phase and frequency over the chosen finite chunks.
- If the first SVD vector differs from 0PN, the result may still be range-dependent rather than universal.
- If the first SVD vector matches 0PN, that supports the local implementation but does not establish a general dimension-selection theory.

The report should avoid overclaiming from this case.

### Equal-Mass Spinning With `chi <= 0.2`

Established from the prompt: the final report should consider spinning waveforms up to `chi=0.2`.

Skeptical view:

- The spin convention is undefined.
- Equal aligned spin is a very restricted spin model.
- Signed spin may require different basis behavior from one-sided positive spin.
- Generic precession would be a different problem and should not be implied.
- Spin effects may be degenerate with mass-like PN directions over narrow ranges, so an apparent low dimension may not generalize.

The useful result would be a controlled statement such as:

> For equal aligned spins `chi1=chi2=chi` in a specified interval, over this mass and frequency range, the spinning residual phase family requires `d` retained directions to keep worst-case coherent mismatch below `mu`.

Anything broader would be premature.

## Falsification Criteria

The idea should be abandoned or reframed if any of the following occur:

1. At equal compute and false alarm, frequency-only StackSlide or a standard semicoherent matched-filter bank has equal or better detectable amplitude.
2. The gain from extra `zeta` dimensions disappears after trials-factor calibration.
3. Memory or interpolation costs dominate before useful sensitivity gains appear.
4. The `zeta` construction is equivalent to an existing chirp-transform or semicoherent metric-bank method without a clear implementation advantage.
5. SVD/PCA coordinates do not generalize to held-out waveforms.
6. The null distribution cannot be calibrated robustly under realistic colored/gapped/resampled data.
7. The 5-vector covariance or NUFFT noise transfer invalidates the assumed statistic normalization.

## Referee Recommendation

Proceed, but narrow the claim.

The project should not be sold as a new semicoherent method yet. It should be framed as a costed study of chunk-local coordinate dimension for semicoherent searches. The decisive deliverable is a plot or table of detectable amplitude versus compute for competing `zeta` dimensions and bases, including frequency-only, 0PN, SVD/PCA, and relevant chirp-transform or matched-filter baselines.

The strongest possible positive outcome would be:

> For a specified 3.5PN waveform family, frequency band, spin range, and computing budget, a particular low-dimensional `zeta` basis gives lower detectable amplitude than frequency-time StackSlide and a direct semicoherent matched-filter baseline at the same total false-alarm probability.

Until that result exists, the central claims remain conjectural.