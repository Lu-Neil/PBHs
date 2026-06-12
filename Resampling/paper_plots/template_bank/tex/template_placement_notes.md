# Template Placement Notes

This note is for future agents editing `main.tex`, especially the
semicoherent template-placement discussion around lines 354-584.

## Relevant Paper Section

The section begins at `\subsubsection{Template placement}` and defines the
stack-slide mismatch geometry for parameters

```tex
\theta = \{\ln M_c, t_{20}\}.
```

The intended logic is:

1. A nearby template changes the predicted track frequency in each coherent
   chunk:

   ```tex
   \Delta f_i =
   \frac{\partial f_i}{\partial\theta^a}\Delta\theta^a .
   ```

2. That physical frequency error must be converted into a dimensionless offset
   in Fourier-bin units:

   ```tex
   d_i = \Delta f_i / \Delta f_{{\rm bin},i}.
   ```

3. The bin offset feeds the coherent-bin response function `B(d)`, which gives
   the retained power for a track offset.

4. Expanding the retained power for small offsets gives the semicoherent metric

   ```tex
   g^{\rm sc}_{ab}
   =
   \kappa
   \frac{
   \sum_i w_i
   [\Delta f_{{\rm bin},i}^{-1}\partial_a f_i]
   [\Delta f_{{\rm bin},i}^{-1}\partial_b f_i]
   }{\sum_i w_i}.
   ```

## Important Clarification Added

The bin width `\Delta f_{{\rm bin},i}` is the Fourier-mode spacing in the
**resampled frequency coordinate**, not the original detector-frame frequency
axis. This distinction is easy to miss because the track frequency `f_i` is
still discussed in Hz.

The paper now states this near the definition

```tex
\Delta f_{{\rm bin},i} = 1/\Delta\tau_i,
```

where `\Delta\tau_i` is the duration of the coherent chunk in the resampled
coordinate. This definition should be kept because it is the normalization that
turns Hz track errors into dimensionless bin offsets.

## Connection To `geometric_metric.py`

The implementation is in:

```text
template_bank/semicoherent_combination/geometric_metric.py
```

The key correspondences are:

- `tau_0pn_duration(...)` implements the paper's `\Delta\tau_i` formula.
- `coherent_bin_width_hz(...)` computes `bin_width_hz = 1.0 / tau_duration`,
  corresponding to `\Delta f_{{\rm bin},i}`.
- `collect_stack_samples(...)` computes TaylorF2 frequency derivatives in Hz:
  `dfdlogm_hz` and `dfdt20_hz`.
- The same function divides those Hz derivatives by `bin_width_hz`, producing
  `dfdlogm_bins` and `dfdt20_bins`. These are the bracketed terms
  `\Delta f_{{\rm bin},i}^{-1}\partial f_i/\partial\theta^a` in the paper.
- `stack_slide_metric_terms(...)` forms the metric entries from weighted sums
  of those bin-normalized derivatives.
- `finite_offset_retained_power(...)` uses the same bin-normalized derivatives
  to compute finite-offset retained power through the response function.

So if future edits alter the definition of `\Delta f_{{\rm bin},i}` or the
metric normalization in the paper, `coherent_bin_width_hz(...)` and
`collect_stack_samples(...)` are the implementation points that must be checked.

## Editing Guidance

When editing this section:

- Preserve the distinction between detector-frame Hz quantities and resampled
  Fourier-bin units.
- Keep the metric expressed in terms of chunk-wise sums over actual coherent
  samples, matching the script.
- Avoid replacing the chunk-wise sum with a continuous time average unless the
  implementation is also changed.
- If the weight model is generalized in the paper, check
  `stack_slide_weights(...)`; it currently supports only uniform weights.
- If a different coherent window is introduced, update both the paper's
  response function `B(d)` and the script's `response_power(...)` /
  `response_curvature(...)`.

