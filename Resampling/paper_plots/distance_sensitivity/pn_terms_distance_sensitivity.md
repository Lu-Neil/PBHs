# Why post-0PN terms have little effect on distance sensitivity

This note explains why replacing the 0PN chirp by a nonspinning TaylorF2
3.5PN frequency evolution changes `distance_sensitivity/maximum_sensitivity.py`
only at the percent level.

## Sensitivity functional

The distance sensitivity used in the plotting scripts can be written as

```math
d(f_0, M_c)
=
K
\frac{\beta(f_0,M_c)}{f_0^2}
\sqrt{I}
```

where `K` contains constants and the detection threshold. The part affected by
the frequency evolution is

```math
I
=
\int_0^T
\frac{\left[f(t)/f_0\right]^{4/3}}
{S_n(f(t))}
\, dt .
```

Here `S_n(f) = ASD(f)^2`. The factor `[f(t)/f_0]^{4/3}` is the same amplitude
evolution factor that appears as

```math
\left(1-\frac{8}{3}\beta t\right)^{-1/2}
```

in the 0PN expression, because

```math
\frac{f_{\rm 0PN}(t)}{f_0}
=
\left(1-\frac{8}{3}\beta t\right)^{-3/8}.
```

For any monotonic chirp, change variables from `t` to `f`:

```math
I_{\rm PN}
=
f_0^{-4/3}
\int_{f_0}^{f_{\rm end,PN}}
\frac{f^{4/3}}{S_n(f)}
\left(\frac{dt_{\rm PN}}{df}\right)
df .
```

At 0PN,

```math
\frac{dt_0}{df}
=
\frac{5}{96}
\pi^{-8/3}
\left(\frac{G M_c}{c^3}\right)^{-5/3}
f^{-11/3},
```

so

```math
I_0
=
A f_0^{-4/3}
\int_{f_0}^{f_{\rm end,0}}
\frac{df}{f^{7/3}S_n(f)}
```

for a constant `A` independent of frequency.

## Perturbative PN correction

Write the PN time-frequency derivative as a correction to 0PN:

```math
\frac{dt_{\rm PN}}{df}
=
\frac{dt_0}{df}
\left[1+\epsilon_{\rm PN}(f)\right].
```

For a nonspinning TaylorF2-like expansion,

```math
\epsilon_{\rm PN}(f)
=
a_2 v^2 + a_3 v^3 + a_4 v^4 + a_5 v^5
+ a_6 v^6 + a_7 v^7 + \cdots,
```

where

```math
v = \left(\frac{\pi G M f}{c^3}\right)^{1/3}.
```

The PN terms are not added directly to the distance. They enter only through a
weighted average inside the integral:

```math
I_{\rm PN}
=
I_0
+
\int_{f_0}^{f_{\rm end,0}} W_0(f)\epsilon_{\rm PN}(f)df
+ W_0(f_{\rm end,0})\Delta f_{\rm end}
+ O(\epsilon_{\rm PN}^2),
```

with

```math
W_0(f)
=
f_0^{-4/3}
\frac{f^{4/3}}{S_n(f)}
\frac{dt_0}{df}
\propto
\frac{1}{f^{7/3}S_n(f)}.
```

Therefore the fractional change in sensitivity is

```math
\frac{\Delta d}{d_0}
=
\sqrt{\frac{I_{\rm PN}}{I_0}} - 1
\simeq
\frac{1}{2}
\left[
\langle \epsilon_{\rm PN}\rangle_{W_0}
+
\frac{W_0(f_{\rm end,0})\Delta f_{\rm end}}{I_0}
\right],
```

where

```math
\langle \epsilon_{\rm PN}\rangle_{W_0}
=
\frac{\int W_0(f)\epsilon_{\rm PN}(f)df}
{\int W_0(f)df}.
```

This equation is the main reason the effect is small:

1. The distance is proportional to `sqrt(I)`, so an integral-level change is
   halved in distance.
2. The PN correction is weighted by `W_0(f)`, not by the pointwise largest PN
   term. Since `W_0(f) \propto 1/[f^{7/3}S_n(f)]`, high-frequency parts of the
   chirp are strongly downweighted.
3. The PN expansion parameter `v` is largest near the high-frequency endpoint,
   but that is also where the `f^{-7/3}` factor and detector noise suppress the
   contribution to the sensitivity integral.
4. For tracks that reach `F_MAX`, the endpoint is fixed at the same frequency.
   For tracks limited by `MAX_OBS_TIME`, the final-frequency shifts are small
   over this grid.

## Numerical size on the plotting grid

Using the current scripts with common LAL constants,

```text
distance_sensitivity/maximum_sensitivity.py
distance_sensitivity/maximum_sensitivity_35PN.py
```

on the grid

```text
f0 in [20, 200] Hz, 51 points
Mc in [1e-5, 1] Msun, 49 points
F_MAX = 2000 Hz
MAX_OBS_TIME = 3e7 s
eta = 0.25
```

the TaylorF2 3.5PN to 0PN ratios are:

```text
D_35PN / D_0PN:
  min    = 0.99070471
  median = 1.0009838
  mean   = 1.0026564
  max    = 1.010396

I_35PN / I_0PN:
  min    = 0.98149582
  median = 1.0019686
  mean   = 1.005331
  max    = 1.0209001

T_35PN / T_0PN:
  min    = 0.99584991
  median = 1.0016169
  mean   = 1.0052841
  max    = 1.0213102
```

The largest distance change is about `1.04%`. This is consistent with the
perturbative estimate above: the PN terms change the time-frequency measure
inside the integral by at most a few percent over the relevant weighted band,
and the distance response is the square root of that integral.

## Conclusion

The post-0PN terms do affect the chirp time and the detailed mapping `f(t)`.
However, distance sensitivity is not very sensitive to those details because it
depends on a noise-weighted frequency integral and then on the square root of
that integral. In the current search band and mass range, the 3.5PN TaylorF2
correction to the maximum distance sensitivity is therefore only percent-level.
