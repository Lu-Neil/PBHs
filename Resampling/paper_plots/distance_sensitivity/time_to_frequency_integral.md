# Time Integral to Frequency Integral

`Analytic_sensitivity.py` originally evaluated the sensitivity integral in time:

```math
I(T; f_0, \beta)
= \int_0^T
\frac{\left(1 - \frac{8}{3}\beta t\right)^{-1/2}}
{\mathrm{ASD}(f(t))^2}
\, dt
```

with

```math
f(t) = f_0 \left(1 - \frac{8}{3}\beta t\right)^{-3/8}.
```

The adaptive `quad` integrator is slow here because `ASD(f)` is a linear interpolation of tabulated ASD samples. That makes the time-domain integrand smooth between ASD knots but kinked at every frequency where the chirp crosses a tabulated sample. The integrator spends many subdivisions resolving those interpolation kinks.

## Change of Variables

Use frequency as the integration variable. From the chirp model,

```math
\frac{f}{f_0}
= \left(1 - \frac{8}{3}\beta t\right)^{-3/8},
```

so

```math
1 - \frac{8}{3}\beta t
= \left(\frac{f_0}{f}\right)^{8/3}.
```

Solving for `t` gives

```math
t(f)
= \frac{3}{8\beta}
\left[
1 - \left(\frac{f_0}{f}\right)^{8/3}
\right].
```

Differentiating,

```math
\frac{dt}{df}
= \frac{f_0^{8/3}}{\beta f^{11/3}}.
```

The chirp prefactor transforms as

```math
\left(1 - \frac{8}{3}\beta t\right)^{-1/2}
= \left(\frac{f}{f_0}\right)^{4/3}.
```

Therefore,

```math
\left(1 - \frac{8}{3}\beta t\right)^{-1/2} dt
=
\frac{f_0^{4/3}}{\beta f^{7/3}} df.
```

The time integral becomes

```math
I(T; f_0, \beta)
=
\frac{f_0^{4/3}}{\beta}
\int_{f_0}^{f(T)}
\frac{1}{f^{7/3}\mathrm{ASD}(f)^2}
\, df.
```

where

```math
f(T) = f_0 \left(1 - \frac{8}{3}\beta T\right)^{-3/8}.
```

## Numerical Implementation

The optimized code precomputes the cumulative integral

```math
F(f) = \int^{f}
\frac{1}{\nu^{7/3}\mathrm{ASD}(\nu)^2}
\, d\nu
```

on the ASD sample grid using trapezoidal integration. Then each sensitivity evaluation only needs

```math
I(T; f_0, \beta)
=
\frac{f_0^{4/3}}{\beta}
\left[ F(f(T)) - F(f_0) \right].
```

This changes the expensive part from thousands of adaptive `quad` calls to interpolation lookups on a single precomputed cumulative integral. It also aligns the numerical integration with the ASD table, avoiding the subdivision warnings caused by crossing many piecewise-linear interpolation knots in time.
