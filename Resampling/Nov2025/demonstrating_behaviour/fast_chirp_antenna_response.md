# Fast chirps and the antenna-response 5-vector

## Problem

The current 5-vector pipeline assumes that the detector response is a slow
sidereal amplitude modulation multiplying an approximately monochromatic signal.
In the time domain the analytic signal is treated as

```text
h(t) = h0 [H_+ F_+(t) + H_x F_x(t)] exp(i phi(t)),
```

where `F_+(t)` and `F_x(t)` are the detector antenna patterns. In
`five_vec.py`, these are represented as finite sums of sidereal harmonics:

```text
F(t) = sum_{k=-2}^{2} A_k exp(i k Omega_sid t).
```

The resampler chooses a nonuniform time coordinate `tau` such that the source
phase becomes approximately monochromatic,

```text
phi(t) = omega0 tau(t).
```

If `tau(t)` is close to linear in `t`, the sidereal modulation remains close to
five discrete Fourier components around the carrier. This is the usual
5-vector picture:

```text
omega0 + k Omega_sid,    k = -2, -1, 0, 1, 2.
```

For a rapidly chirping signal, however, `tau(t)` is strongly nonlinear. After
resampling, the antenna response becomes

```text
F(t(tau)) = sum_{k=-2}^{2} A_k exp(i k Omega_sid t(tau)),
```

which is not a sum of five pure sinusoids in `tau`. Each sidereal sideband is
itself chirped in the resampled coordinate. The power is therefore no longer
confined to the five bins extracted by `Resampler.extract_5vec`.

This is a limitation of the five-bin approximation, not of the source phase
resampling itself.

## Can the antenna phase be absorbed into `tau`?

Partly, but only for a fixed assumed detector response. Define the complex
combined antenna response

```text
M(t) = H_+ F_+(t) + H_x F_x(t).
```

Then the signal can be written as

```text
h(t) = h0 |M(t)| exp(i [phi(t) + arg M(t)]).
```

For a single assumed sky position and polarization, one could define an
effective resampling coordinate

```text
tau_eff(t) = [phi(t) + arg M(t)] / omega0.
```

This absorbs the phase of the antenna response into the demodulation. The
remaining signal is

```text
h(t) = h0 |M(t)| exp(i omega0 tau_eff(t)).
```

However, this is not a clean replacement for the current pipeline:

- `arg M(t)` depends on the polarization parameters `eta` and `psi`, which are
  not known a priori.
- `M(t)` is detector-dependent, so this coordinate would differ between H1, L1,
  etc.
- `M(t)` can pass near zero, making `arg M(t)` discontinuous or numerically
  unstable.
- It collapses the useful linear separation between plus and cross templates.
- It absorbs only the complex phase response. The amplitude factor `|M(t)|`
  still remains as a time-dependent modulation.

So this approach is possible as a template-specific coherent demodulation, but
it is probably not the best generalization of the 5-vector method.

## Better interpretation: put the antenna response in the matched-filter kernel

A more robust route is to keep the source phase resampling

```text
tau(t) = phi(t) / omega0
```

and stop assuming that the antenna response remains a five-bin object after the
nonlinear time remapping.

Instead, build the detector response templates directly in the original time
samples and project the data against them after demodulating the source phase:

```text
z_+(omega0) = sum_i x(t_i) conj(F_+(t_i)) exp(-i omega0 tau_i),
z_x(omega0) = sum_i x(t_i) conj(F_x(t_i)) exp(-i omega0 tau_i).
```

Equivalently, the matched-filter basis functions are

```text
q_+(t_i) = F_+(t_i) exp(i omega0 tau_i),
q_x(t_i) = F_x(t_i) exp(i omega0 tau_i).
```

This directly includes the detector response in the phase-demodulated search
kernel. It does not require the response to occupy exactly five Fourier bins in
`tau`.

The current helper path in `fiveVec_resampler_utils.py` already points in this
direction: `_time_domain_5vec` constructs time-domain plus/cross sidereal
templates and passes them through the same resampler. For fast chirps, the next
step is to generalize this from "resample the template and extract five bins" to
"project against the full resampled antenna-response template".

## Consequences for the pipeline

The fast-chirp regime suggests replacing the five-bin estimator

```text
X_5 . A_5 / (A_5^\dagger A_5)
```

with a two-template coherent estimator over the full sampled time series, or
over a sufficiently wide set of NUFFT bins:

```text
d = h_+ q_+ + h_x q_x + n.
```

With white noise, the least-squares estimator is

```text
h_hat = (Q^\dagger Q)^{-1} Q^\dagger d,
```

where the columns of `Q` are `q_+` and `q_x`. With coloured or resampling-shaped
noise, this should become the noise-weighted estimator

```text
h_hat = (Q^\dagger C^{-1} Q)^{-1} Q^\dagger C^{-1} d.
```

This also connects naturally to the existing roadmap item on noise-weighted
matched filtering. The covariance `C` should describe the noise in whichever
representation is used: time-domain samples, resampled samples, or a selected
NUFFT-bin vector.

## Practical recommendation

For slowly chirping signals, the current 5-vector extraction remains useful and
cheap.

For rapidly chirping signals, do not try to force the antenna response into five
bins after nonlinear resampling. Keep `tau` tied to the source phase, then use
time-domain or resampled-domain plus/cross detector-response templates in the
matched filter. This preserves the physical separation between source phase,
detector response, and polarization while avoiding the false assumption that the
sidereal response remains monochromatic in `tau`.

