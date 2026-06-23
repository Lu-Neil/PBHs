# Referee Report: Semicoherent `zeta`-Space Dimension Selection

## Audit Metadata

- Agent: `referee`
- Attempt: 2
- Source of truth: `idea.md` and attempt-2 human feedback
- Prior artifacts considered: planner, literature review, theory/optimization reports included in prompt
- External verification in this artifact: none
- Script execution: none
- Status: skeptical review, not a final scientific assessment

## Attempt-2 Changes

This version addresses the human feedback more directly. I do not criticize the project for failing to be fully coherent matched filtering, because that is not the goal. The relevant question is narrower:

> How should one choose the number and basis of coherent-chunk coordinates `zeta` to optimize sensitivity versus computing cost in a semicoherent search?

The main critique is that this question is still under-specified unless “better” means lower detectable amplitude at fixed compute, memory, and total false-alarm probability.

## Summary Judgment

The proposal is scientifically plausible, but its current framing is still too broad. The strongest defensible project is a falsifiable cost-sensitivity study of semicoherent searches that use richer coherent-chunk coordinates than ordinary frequency-time power spectra.

Established from `idea.md`: the target is not novelty of semicoherent searches in general. The target is whether there are better ways to build semicoherent combinations than using only frequency-time spectra, and how to decide the useful dimension of `zeta`.

Conjecture: low-dimensional PN-, SVD-, or PCA-inspired `zeta` coordinates may outperform frequency-only spectra in some compute-limited regimes.

Open question: whether this remains true after grid cost, interpolation, memory, trials factors, and realistic noise normalization are included.

## Strongest Objections

### 1. The operational objective is still not fully defined

The project should not optimize “dimension,” “mismatch,” or “recovered power” in isolation. The auditable objective should be something like:

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

Without this, different agents may optimize incompatible quantities. A PCA rank, a metric dimension, a number of PN terms, and an operational search dimension are not the same thing.

### 2. Extra `zeta` dimensions may improve mismatch while worsening search sensitivity

Adding a coherent-chunk coordinate usually reduces model mismatch. That is the easy part. The hard part is showing that the gain survives:

- more spectral grid points;
- larger memory footprint;
- more interpolation work;
- higher track-summation cost;
- larger effective trials factor;
- more difficult null calibration;
- possible covariance between neighboring bins.

A referee would not accept “more recovered power” as evidence. The required claim is lower detectable amplitude at equal compute and equal false alarm.

### 3. SVD/PCA dimension is not the final answer

The human-requested SVD/PCA analysis is valuable, especially for testing whether the best one-parameter basis differs from 0PN. But SVD/PCA answers a representation question under a chosen norm. It does not by itself answer the search-design question.

A retained SVD direction should become a searched `zeta` dimension only if its mismatch reduction is worth the extra grid, memory, interpolation, and threshold cost. Otherwise it may be useful for waveform compression but not for a semicoherent search.

### 4. The equal-mass non-spinning case may be too forgiving

For exactly equal-mass, non-spinning 3.5PN waveforms, the intrinsic physical family is essentially one-dimensional after fixing mass ratio and spin. If one dominant coordinate works well, that may be expected.

The meaningful test is more specific:

> After projecting out constant phase, time shift, and local carrier frequency, is the dominant residual phase direction actually close to the 0PN coordinate over the chosen mass range, frequency band, and `T_coh`?

If yes, that supports the 0PN construction in that regime. It does not establish a general dimension-selection principle.

### 5. The spinning case is currently ambiguous

The phrase `chi <= 0.2` is not enough. The answer changes depending on whether the model is:

- equal aligned spins, `chi1 = chi2 in [0, 0.2]`;
- equal signed aligned spins, `chi1 = chi2 in [-0.2, 0.2]`;
- independent aligned spins;
- generic precessing spins.

These are different waveform families. Any report that discusses “the spinning case” without fixing this convention is not auditable.

### 6. The analytical-basis hope may be limited

For polynomial phase models with simple windows and noise, analytic orthogonal bases may be possible. For 3.5PN spinning waveforms, an exact globally optimal analytic low-dimensional basis is much less likely.

Conjecture: analytic work can provide good candidate coordinates, such as 0PN/chirp-time/metric eigen-directions. But final dimension selection probably still requires numerical validation over the actual mass, frequency, spin, and chunk-duration ranges.

## Unclear Assumptions

### Meaning of `zeta`

The report must distinguish:

- physical parameters `theta`;
- chunk-local coordinates `zeta_k(theta)`;
- axes of stored coherent spectra;
- interpolation coordinates used during track summation;
- segment time labels.

In ordinary semicoherent searches, time is a segment label. It should not be counted as a searched `zeta` dimension unless spectra are explicitly gridded over candidate epochs.

### Phase convention

The 3.5PN study needs a precise convention:

- time-domain or frequency-domain phase;
- PN terms included;
- reference frequency or reference time;
- treatment of coalescence time and phase;
- mass range;
- frequency band;
- termination frequency;
- units.

Without these, “0PN is near optimal” is not a well-defined claim.

### Nuisance projections

The comparison between 0PN and SVD/PCA is invalid unless all bases project out the same nuisance directions. At minimum, these likely include constant phase and local frequency. Depending on convention, time shift or linear-in-frequency phase may also need projection.

### Search volume

The useful dimension depends on the searched region. A narrow mass/spin/frequency range can look low-dimensional even when a broader search is not. Dimension claims must always state the range.

### Noise model

Phase-only mismatch is a useful first diagnostic. Sensitivity claims require a null distribution and noise normalization. For the local NUFFT/resampling/5-vector pipeline, this is especially risky because the transformation changes the effective noise covariance.

## Novelty Concerns

Established from prior artifacts: semicoherent track sums, weighted power sums, metric template banks, chirp transforms, polynomial-phase transforms, and reduced-basis waveform compression all overlap with parts of this idea.

The novelty, if any, is not:

- semicoherent summing;
- using frequency-time tracks;
- adding chirp-like coordinates;
- using SVD/PCA;
- using PN-inspired coordinates.

The possible novelty is:

> A costed rule for choosing reusable coherent-chunk coordinate dimensions `zeta`, and demonstrating improved detectable amplitude over relevant baselines at equal compute and false alarm.

Nearest-method comparisons that still need explicit resolution:

- StackSlide / PowerFlux-style frequency-track sums;
- Weave and semicoherent metric template banks;
- fast chirp transform and polynomial-phase transform analogues;
- reduced-basis / ROQ / SVD waveform compression;
- direct semicoherent matched-filter banks with cached segment outputs.

## Possible Normalization Errors

### Frequency units

The local codebase reportedly often uses angular frequency. Any formulas mixing `f`, `omega`, `beta`, PN chirp parameters, NUFFT bins, or phase derivatives must track factors of `2 pi`.

### Coherent statistic normalization

The report must specify:

- `T_coh` factors;
- sample spacing;
- window normalization;
- FFT/NUFFT normalization;
- real versus analytic signal convention;
- one-sided versus two-sided PSD convention.

Otherwise comparisons across `dim(zeta)` or `T_coh` can be numerically meaningless.

### NUFFT noise transfer

Using `S_n(f0)` alone is likely wrong for a chirping resampled coordinate. The effective noise should depend on the whole instantaneous-frequency track induced by the `tau` map.

### Independent-bin assumption

The ideal assumption

```math
P_k \sim \mathrm{Exp}(1)
```

is not guaranteed for:

- colored noise;
- gapped data;
- overlapping windows;
- interpolation;
- nonuniform resampling;
- correlated nearby `zeta` bins;
- 5-vector sideband extraction;
- repeated use of the same data across many templates.

Any Gamma or exponential threshold should be treated as a starting approximation, not a calibrated result.

### 5-vector covariance

The five sidereal bins should not automatically be treated as equal-variance independent samples. A covariance-weighted statistic is probably required for the PBH/5-vector pipeline.

### Interpolation covariance

Interpolating through a `zeta` grid changes both signal response and noise covariance. It is not enough to call this a deterministic mismatch unless the null distribution is measured.

### Trials factor

The effective trials factor is not simply the raw number of templates, but ignoring it is worse. Extra dimensions can raise the detection threshold enough to erase their coherent-mismatch gain.

## Missing Tests

### Baseline comparisons

Required baselines:

1. Frequency-only StackSlide-like accumulation.
2. Weighted PowerFlux-like accumulation where applicable.
3. Direct semicoherent matched-filter bank.
4. Chirp-transform or polynomial-phase analogue for `{f, fdot, ...}` or `{f, beta}`.
5. Fully coherent matched filtering on small volumes as a sanity bound, not as the target method.

### Fixed-false-alarm sensitivity

For every candidate `dim(zeta)`:

- measure the null distribution;
- estimate effective trials factor;
- set a total false-alarm threshold;
- inject signals across amplitudes;
- report detectable amplitude at fixed detection probability and false alarm.

### Runtime and memory

Asymptotic scaling is insufficient. Measure:

- coherent spectrum construction time;
- track summation time;
- interpolation time;
- peak memory;
- memory bandwidth sensitivity;
- scaling with `T_coh`;
- scaling with `dim(zeta)`.

### SVD/PCA validation

Required for both non-spinning and spinning studies:

- train/test waveform split;
- documented parameter sampling;
- inner product and window definition;
- nuisance projection definition;
- singular value spectrum;
- mismatch versus retained dimension;
- worst-case held-out mismatch;
- overlap of first SVD direction with 0PN.

### Equal-mass non-spinning 3.5PN comparisons

Compare:

- frequency only;
- frequency plus 0PN;
- frequency plus best one SVD/PCA direction;
- frequency plus two or more SVD/PCA directions;
- frequency plus higher-PN inspired coordinates.

The decisive output is not basis overlap alone. It is whether any basis lowers detectable amplitude after cost and thresholds.

### Equal-mass spinning 3.5PN comparisons

First define the spin model. Then compare:

- non-spinning basis applied to spinning waveforms;
- spinning SVD/PCA basis;
- 0PN plus leading spin coordinate;
- higher-dimensional PN-inspired bases;
- mismatch and cost versus dimension.

The key result should be a bounded statement: for a specified spin family and range, how many directions are needed below a stated mismatch tolerance.

## Is This Genuinely Different From Matched Filtering?

Not clearly yet.

A semicoherent matched-filter bank can compute coherent outputs for many local templates in each segment and then combine them incoherently along physical-template tracks. The proposed `zeta` method appears close to that, with local templates organized by coordinates such as `{f, beta}` or SVD/PCA coefficients.

It becomes genuinely distinct only if:

1. the local `zeta` spectra are reusable across many physical templates;
2. the useful `zeta` dimension is lower than the direct local physical-template dimension;
3. `theta -> zeta_k(theta)` track summation is cheaper than direct semicoherent filtering;
4. approximation loss is controlled;
5. detectable amplitude improves at equal compute and equal false alarm.

If these are not demonstrated, the honest framing is “an implementation strategy for semicoherent matched filtering,” not a distinct search method.

## Falsification Criteria

The idea should be abandoned or reframed if:

- frequency-only StackSlide wins at equal compute and false alarm;
- direct semicoherent matched filtering wins at equal compute and false alarm;
- trials-factor penalties erase the gain from extra dimensions;
- memory or interpolation dominates before sensitivity improves;
- SVD/PCA bases fail held-out validation;
- the spinning result depends strongly on an arbitrary spin convention;
- the null distribution cannot be calibrated under colored/gapped/resampled data;
- the method is equivalent to known chirp-transform or metric-bank machinery without a measurable implementation advantage.

## Referee Recommendation

Proceed, but narrow the claim.

The project should be framed as a costed dimension-selection study for semicoherent chunk-local coordinates. The decisive deliverable is a table or plot of detectable amplitude versus compute for competing choices of `zeta`, including frequency-only, 0PN, SVD/PCA, higher-PN, and relevant matched-filter or chirp-transform baselines.

Until that exists, the main claims remain conjectural.