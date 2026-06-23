# Searching over an unknown start time

Suppose the chirp mass `Mc` is fixed, but the epoch at which the signal crosses
the reference frequency `f0` is unknown. The unknown parameter should be treated
as the crossing time

```text
t0 such that f(t0) = f0.
```

For a fixed `Mc` and reference frequency `f0`, the template can be written as

```text
f(t; t0, Mc) = f0 [1 - (8 beta0 / 3) (t - t0)]^(-3/8),
```

where `beta0` is fixed by `Mc` and `f0`.

The corresponding resampled coordinate is

```text
tau(t; t0) = integral [f(t; t0) / f0] dt.
```

For the correct trial `t0`, the signal becomes approximately monochromatic at
`f0` in the `tau` coordinate. This remains true even if the data window begins
after the crossing time, in which case the signal is already at a physical
frequency above `f0`.

## Search strategy

Do not assume that the signal crosses `f0` at the start of each 1 s spectrum.
That only catches signals whose crossing time happens to align with the window
start. Instead, evaluate the detection statistic over trial crossing times:

```text
for each data window:
    for each trial crossing time t0:
        build tau(t; t0)
        compute the NUFFT spectrum in tau
        extract the carrier and sidereal sidebands
        compute the 5-vector detection statistic
```

Equivalently, for a window starting at time `T`, search over the signal age

```text
Delta = T - t0.
```

If `Delta > 0`, the signal crossed `f0` before the window began. The starting
frequency in that window is then

```text
f_start = f0 [1 - (8 beta0 / 3) Delta]^(-3/8).
```

This is not a different chirp-mass template. It is the same physical template
with a different time offset.

## Grid spacing in crossing time

The `t0` spacing should be set by mismatch, not automatically by the 1 s
spectrum cadence. A crossing-time error `delta_t0` produces a local frequency
track error of roughly

```text
delta_f ~ fdot * delta_t0.
```

A simple conservative requirement is that this induced frequency error remain
smaller than about one coherent Fourier bin over the analyzed segment:

```text
delta_f <~ 1 / T_obs,
```

or

```text
delta_t0 <~ 1 / (T_obs * max(fdot)).
```

Here `max(fdot)` should be taken over the portion of the chirp track covered by
the coherent window and over the relevant parameter-space region. Slow chirps
may allow `t0` spacings of order 1 s, while fast chirps require subsecond
crossing-time templates.

The practical detection statistic is therefore

```text
Lambda = Lambda(t0; Mc, f0, sky, ...)
```

and candidate events appear as peaks in `Lambda` as a function of the trial
crossing time.
