# Template Bank

We use a semicoherent search.  The data, with total observing time `T_obs`, are
split into coherent chunks of duration `t`.  Each chunk is analyzed with a
0PN demodulation template labelled by `beta`; the demodulated samples are
Fourier transformed with a NUFFT, which returns all carrier-frequency bins in
the resampled time coordinate.  The coherent powers are then combined along
long-timescale inspiral tracks.  These stack-slide tracks are defined by the
3.5PN frequency evolution, not by the 0PN model used inside a single chunk.

The separation is deliberate.  The 0PN model is accurate enough over short
coherent chunks, provided the chunk duration and beta spacing are controlled.
Over the full observing time, however, the phase drift between 0PN and 3.5PN
would be large, so the semicoherent tracks should follow the 3.5PN frequency
evolution.  For a fixed chirp mass and reference crossing time, the track
selects one frequency bin in each coherent chunk.

The fiducial search discussed here uses

```text
20 Hz <= f <= 64 Hz,
5e-4 <= Mc/Msun <= 1e-1,
T_obs = 1 yr.
```

The PN calculations use the equal-mass symmetric mass ratio, `eta = 1/4`.
If the manuscript denotes the equal-mass case by a mass ratio `q = 1`, this is
the same physical choice.

## Coherent 0PN Approximation

Within a coherent chunk starting at physical frequency `f`, the 0PN phase model
can be written in terms of the resampled time

```text
tau_0PN(t; beta) =
    3/(5 beta) [1 - (1 - 8 beta t / 3)^(5/8)]
```

so that

```text
Phi_0PN(t; f, beta) = 2 pi f tau_0PN(t; beta).
```

The true signal phase over the same chunk is taken to be the 3.5PN phase,
`Phi_3.5PN(t; f, Mc, eta)`.  The chunk duration is chosen so that the
accumulated 0PN modelling error is bounded by a fixed phase budget.  We use

```text
max_{f,Mc} |Phi_3.5PN(t; f, Mc, eta)
          - Phi_0PN(t; f, beta_0PN(f,Mc))|
    <= pi/2.
```

The maximum is evaluated over the coherent search band and chirp-mass range.
For the fiducial bank, the most restrictive point is the high-frequency,
high-mass corner.  At `f = 64 Hz`, `Mc = 0.1 Msun`, and `t = 30 s`, the
0PN--3.5PN dephasing is approximately `1.25 rad`, below `pi/2`.  The time to
reach a `pi` dephasing at the same corner is approximately `46.6 s`.  We
therefore use

```text
t = T_chunk = 30 s
```

as the fiducial coherent chunk duration.

Halving the chunk duration to 15 s would roughly double the number of chunks.
The semicoherent distance sensitivity then changes by

```text
D_15 / D_30 ~= (N_30 / N_15)^(1/4) ~= 2^(-1/4) = 0.841,
```

corresponding to a 15.9% loss in distance reach before accounting for any
reduction in 0PN modelling error or any change in template count.

## Coherent Beta Grid

The second coherent requirement is that finite beta spacing not introduce more
than a comparable phase error.  For a beta-grid template
`beta = beta_true + delta beta`, we require

```text
|Phi_0PN(t; f, beta_true + delta beta)
 - Phi_0PN(t; f, beta_true)| <= pi/2
```

at the edge of a beta cell.  Linearizing,

```text
|delta beta| <= (pi/2) / |d Phi_0PN / d beta|.
```

Since

```text
Phi_0PN = 2 pi f tau_0PN,
```

this is equivalently

```text
|delta beta| <= 1 / [4 f |d tau_0PN / d beta|].
```

In the small-`beta t` limit, `tau_0PN = t + beta t^2/2 + O(beta^2 t^3)`, so

```text
d Phi_0PN / d beta ~= pi f t^2,
|delta beta| <= 1 / (2 f t^2).
```

The full grid spacing is twice the allowed edge error,

```text
Delta beta <= 1 / (f t^2)
```

in this leading-order limit.  Thus, at fixed beta range, the number of coherent
beta templates scales approximately as

```text
N_beta proportional to t^2.
```

Shorter coherent chunks therefore rapidly reduce the beta-bank size.

It is also useful to express the same condition in NUFFT-bin units.  A beta
error produces an approximate tau-frequency shift

```text
Delta f_tau ~=
    f delta beta t / (1 - 8 beta t / 3),
```

which is compared with the coherent bin width

```text
Delta f_bin = 1 / tau_0PN(t; beta).
```

For the current Galactic-Center-optimized bank, the coherent beta edge is
`0.4517` coherent bins.  The resulting coherent beta bank contains

```text
N_beta = 56
```

templates for the fiducial `20--64 Hz`, `5e-4--1e-1 Msun`, `t = 30 s` setup.

## Joint 0PN and Beta Error

The two coherent errors are not strictly independent.  The relevant comparison
for a real signal is not a 0PN signal with the wrong beta against a 0PN signal
with the right beta.  It is a 3.5PN signal with the true parameters analyzed by
a 0PN template with beta error.  The combined coherent residual is

```text
Delta Phi_joint(t)
  = Phi_3.5PN(t; f, Mc, eta)
    - Phi_0PN(t; f, beta_0PN(f,Mc) + delta beta).
```

The robust placement condition is therefore

```text
max_{f,Mc, |delta beta| <= Delta beta/2}
    |Delta Phi_joint(t)| <~ pi.
```

The separate `pi/2` budgets for the 0PN approximation and the beta-grid edge
are a conservative way of satisfying this combined `~pi` condition by the
triangle inequality.  The actual residual can be smaller or larger depending on
the sign of the beta error relative to the PN modelling error, so the combined
condition should be checked directly on a validation grid.  For the fiducial
`t = 30 s` bank, the worst-corner 0PN modelling error alone is `1.25 rad`, and
the beta-grid edge is chosen so that the combined residual remains of order one
radian over the validation grid.

## 3.5PN Frequency Tracks

The semicoherent stack-slide stage should use 3.5PN frequency tracks.  We
parameterize the track by chirp mass and by a reference crossing time `t20`,
defined by

```text
f_3.5PN(t20; Mc, eta) = 20 Hz.
```

For a trial `(Mc, t20)`, the template frequency in a chunk centered at time
`t_i` is

```text
f_i = f_3.5PN(t_i - t20; Mc, eta).
```

If `f_i` lies outside the analysis band, that chunk does not contribute to the
track.  If `f_i` lies inside the band, the stack-slide track reads the nearest
coherent frequency bin from the NUFFT map associated with the appropriate beta
template.

There is no additional independent `f0` template once `t20` is defined relative
to a fixed reference frequency.  The range `20--64 Hz` specifies the in-band
portion of each 3.5PN track; it is not a separate dimension multiplying the
number of stack-slide tracks.

## Time and Mass Grids for Tracks

The track grid is placed so that neighbouring 3.5PN tracks do not move the
selected coherent frequency bins too far.  For small parameter offsets,

```text
Delta f_i
  ~= (partial f_3.5PN / partial log Mc)_i Delta log Mc
     - fdot_3.5PN(t_i) Delta t20.
```

A conservative local metric requires the two contributions to be bounded by
allocated fractions of the coherent bin width,

```text
|partial f_3.5PN / partial log Mc| Delta log Mc / 2
    <= epsilon_M Delta f_bin,
```

and

```text
|fdot_3.5PN| Delta t20 / 2
    <= epsilon_t Delta f_bin,
```

where `Delta f_bin = 1 / tau_0PN(t; beta)` is the coherent bin width and
`epsilon_M + epsilon_t` is the stack-slide bin-offset budget.  The derivatives
should be evaluated along the 3.5PN track and maximized over the part of the
track in band.

The crossing-time grid can be relaxed relative to this worst-case criterion if
one models the loss as an average over coherent-bin offsets.  For a crossing
time error `epsilon`, the induced offset in bin units is

```text
d_i = fdot_3.5PN(t_i) epsilon / Delta f_bin,i.
```

The averaged nearest-bin power is

```text
B(d) = integral_{-1/2}^{1/2} sinc^2(x + d) dx,
```

and the track-averaged crossing-time loss is

```text
R_t20(epsilon, Mc) = < B(d_i) >_i / B(0).
```

The corresponding distance loss is `sqrt(R_t20)`.  This criterion is more
appropriate for a Galactic-Center-optimized search, because large losses in
regions with large baseline distance margin may still leave the source
detectable at 8 kpc.

Using the current conservative proxy bank with `t = 30 s`, the fiducial
parameter space gives

```text
N_Mc = 310,834,
N_t20 tracks = 4.03e12,
```

where the second number is the total number of `(Mc, t20)` tracks over the
one-year observing time.  The number of crossing-time templates per mass ranges
from approximately `5.19e4` to `1.14e8`, with a median of `1.42e6`.  These
numbers should be understood as conservative bank-count estimates; a final
production bank should recompute the mass and time grids using the 3.5PN
frequency derivatives directly.

## Computational Cost

The computational cost has two pieces: coherent NUFFT transforms and
semicoherent stack-slide track summation.  Let `T` be the total observing time
and `t` the coherent chunk duration.  Up to sampling-rate and implementation
constants, the coherent cost is

```text
C_coh(t) ~= (T/t) N_beta(t) [t log t]
         = T log t N_beta(t).
```

The naive stack-slide cost is

```text
C_stack(t) ~= N_Mc(t) N_t20(t) T/t,
```

where `T/t` is the number of chunk samples read by a typical year-long track.
Thus the total cost model is

```text
C(t) = t log t * T/t * N_beta(t)
       + N_Mc(t) N_t20(t) T/t.
```

This expression is the useful bookkeeping form because each factor responds
differently to the coherent chunk duration:

```text
N_beta(t) decreases rapidly for shorter chunks, approximately as t^2.
The number of chunks T/t increases for shorter chunks.
The coherent transform size t log t decreases for shorter chunks.
The product N_Mc(t) N_t20(t) depends on the track-placement rule.
```

For a non-redundant stack-slide covering of the 3.5PN time-frequency track
manifold, the factor `N_Mc(t) N_t20(t)` is expected to scale approximately like
`t`, so that the naive stack-slide cost is roughly independent of `t`:

```text
C_stack(t) ~ t * T/t ~ T.
```

This cancellation is not automatic if one imposes an overly conservative metric
or an externally fixed `t20` spacing; it should be verified with the final
placement rule.

For the fiducial parameter space

```text
5e-4 <= Mc/Msun <= 1e-1,
20 Hz <= f <= 64 Hz,
T_obs = 1 yr,
equal masses: q = 1, eta = 1/4,
t = 30 s,
```

the current bank-count artifacts give

```text
N_chunks = 1,051,920,
N_beta = 56,
N_NUFFT = 58,907,520,
N_Mc = 310,834,
N_t20 tracks = 4.03e12.
```

The coherent NUFFT stage is therefore not the dominant template-count problem.
The dominant cost is the stack-slide summation over 3.5PN tracks.  A search
limited to approximately `1e5` CPU hours will likely require a
GC-margin-weighted track grid, a hierarchical first pass with follow-up, or an
accumulator-based stack-slide implementation that avoids explicitly reading
every track independently.
