# Semicoherent Chunk Weights

This note records the intended normalization for semicoherent stack-slide
weights. The main point is that the apparent PSD power depends on whether the
quantity being summed is raw coherent power or PSD-normalized coherent power.

## Setup

Let $i = (X, I)$ label detector $X$ and coherent chunk $I$. For a trial
signal track with parameters $\lambda$, let

$$
\begin{aligned}
P_i
    &= \text{coherent power read from the selected frequency bin}, \\
A_i(\lambda)
    &= \text{expected signal amplitude in that detector/chunk}, \\
\Gamma_i(\lambda)
    &= \text{live-time, windowing, and data-quality efficiency factor}, \\
S_i^{\rm eff}\!\left[f_i(\lambda)\right]
    &= \text{effective noise PSD after the NUFFT/resampling operation}.
\end{aligned}
$$

For a raw periodogram-like coherent power, the noise-only mean scales as

$$
\mathbb{E}_0[P_i] \sim S_i^{\rm eff}
$$

and the noise-only variance scales as

$$
\operatorname{Var}_0(P_i) \sim \left(S_i^{\rm eff}\right)^2 .
$$

The expected signal excess in raw power is

$$
s_i
    =
    \mathbb{E}_1[P_i] - \mathbb{E}_0[P_i]
    \sim
    A_i^2(\lambda)\,\Gamma_i(\lambda).
$$

## Optimal Weights For Raw Powers

For a weak signal and independent chunk powers, the locally optimal linear
statistic

$$
\Lambda = \sum_i \alpha_i P_i
$$

uses inverse-variance weights matched to the expected signal excess:

$$
\alpha_i \propto \frac{s_i}{\operatorname{Var}_0(P_i)} .
$$

Therefore, when summing raw powers directly,

$$
\alpha_i
    \propto
    \frac{A_i^2(\lambda)\,\Gamma_i(\lambda)}
         {\left(S_i^{\rm eff}\right)^2}.
$$

The squared PSD appears because raw power fluctuations have variance
proportional to the square of the noise floor.

## Optimal Weights For PSD-Normalized Powers

If each coherent power is first PSD-normalized,

$$
\rho_i^2 = \frac{P_i}{S_i^{\rm eff}},
$$

then the noise-only variance of $\rho_i^2$ is approximately constant across
chunks. The expected signal excess in the normalized power is instead

$$
\mathbb{E}_1[\rho_i^2] - \mathbb{E}_0[\rho_i^2]
    \sim
    \frac{A_i^2(\lambda)\,\Gamma_i(\lambda)}
         {S_i^{\rm eff}} .
$$

Thus a semicoherent statistic formed from normalized powers,

$$
\Lambda = \sum_i w_i \rho_i^2,
$$

should use

$$
w_i
    \propto
    \frac{A_i^2(\lambda)\,\Gamma_i(\lambda)}
         {S_i^{\rm eff}} .
$$

With weights normalized to sum to one, this is

$$
w_i(\lambda)
    =
    \frac{
        A_i^2(\lambda)\,\Gamma_i(\lambda)
        /
        S_i^{\rm eff}\!\left[f_i(\lambda)\right]
    }{
        \sum_j
        A_j^2(\lambda)\,\Gamma_j(\lambda)
        /
        S_j^{\rm eff}\!\left[f_j(\lambda)\right]
    } .
$$

Equivalently, substituting $\rho_i^2 = P_i / S_i^{\rm eff}$ gives

$$
w_i \rho_i^2
    \propto
    \frac{
        A_i^2(\lambda)\,\Gamma_i(\lambda)\,P_i
    }{
        \left(S_i^{\rm eff}\right)^2
    },
$$

which is the same raw-power statistic as above.

## Practical Convention

Use the $1 / S_i^{\rm eff}$ weight form only when the coherent powers being
combined have already been divided by their effective PSD/noise floor. If the
implementation sums raw coherent powers, use the $1 / (S_i^{\rm eff})^2$
form.

For correlated chunks or detectors, replace the scalar inverse-variance rule by
the covariance-weighted form

$$
\boldsymbol{\alpha} \propto C^{-1}\boldsymbol{s},
$$

where $s_i$ is the expected signal excess vector and $C$ is the noise
covariance matrix of the chunk powers being summed.
