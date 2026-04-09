"""
Noise-weighted 5-vector matched-filter estimator with NUFFT noise transfer correction.

Demonstrates:
  1. Computing the effective NUFFT noise PSD at each sidereal sideband bin via the
     stationary-phase time-average:
         S_eff(f_out) = (1/T) integral_0^T S_n( f_out * dtau/dt(t) ) dt
     where dtau/dt = (1 - 8*beta*t/3)^{-3/8}
  2. A noise-weighted joint matched filter for the two GW polarisations:
         [h_p, h_c] = (A† C^{-1} A)^{-1} A† C^{-1} X
         SNR² = X† C^{-1} A (A† C^{-1} A)^{-1} A† C^{-1} X
  3. Calibrated SNR² output vs the old unweighted _detection_stat
  4. Bias that results from using the naive S_n(f_k) instead of S_eff

Parameters are chosen so the noise transfer correction is ~17%:
  - M_c = 16 solar masses  ->  beta ~3.65e-7 s^-1,  8/3*beta*T_obs ~0.167
  - PSD slope alpha = -6  (steeply red power-law in the sub-Hz band)
  -> S_eff / S_n(f_0) ~ 0.825  (signal chirps up into quieter noise)

For real data replace `Sn` with an interpolant of the measured ASD.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import matplotlib.pyplot as plt

NOV2025_DIR = Path(__file__).resolve().parent.parent
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import (
    _create_PBH_signal,
    _resample_and_extract_5vec,
    _time_domain_5vec,
    _joint_estimator,
    _detection_stat,
)

# ------------------------------------------------------------------ #
# Global parameters                                                    #
# ------------------------------------------------------------------ #

SEED = 42
MC_SOLAR = 16.0  # chirp mass [solar masses]
MC = MC_SOLAR * 2e30  # kg
ALPHA_NOISE = -6.0  # power-law PSD slope
TARGET_SNR = 15.0  # nominal SNR ~ 15 with flat noise (SNR^2 ~ 225)
N_REAL = 100  # Monte Carlo noise realisations
F_S = 1.0  # Hz  (set by _create_PBH_signal)

SIDE_DAY = 86164.09053083288  # sidereal day [s]
c, G = 3e8, 6.67e-11
_CHIRP_CONST = 96 / 5 * np.pi ** (8 / 3) * (G / c**3) ** (5 / 3)


# ------------------------------------------------------------------ #
# 1. Signal and template generation                                    #
# ------------------------------------------------------------------ #


def build_signal_and_templates():
    """Generate PBH signal, resampling grid, and 5-vector templates.

    Returns
    -------
    dict with keys: signal, tau, omega0, sidereal, h0, gamma, t,
                    data_X_signal, resampler,
                    template_Xp, template_Xc,
                    f0, beta, n_samples, T_obs
    """
    np.random.seed(SEED)
    signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(f0_setting="midpoint", Mc=MC)

    n_samples = len(signal)
    T_obs = n_samples / F_S
    f0 = omega0 / (2 * np.pi)
    beta = _CHIRP_CONST * f0 ** (8 / 3) * MC ** (5 / 3)

    data_X_signal, resampler = _resample_and_extract_5vec(signal, tau, omega0)
    _, template_Xp, template_Xc = _time_domain_5vec(sidereal, t, tau)

    return dict(
        signal=signal,
        tau=tau,
        omega0=omega0,
        sidereal=sidereal,
        h0=h0,
        gamma=gamma,
        t=t,
        data_X_signal=data_X_signal,
        resampler=resampler,
        template_Xp=template_Xp,
        template_Xc=template_Xc,
        f0=f0,
        beta=beta,
        n_samples=n_samples,
        T_obs=T_obs,
    )


# ------------------------------------------------------------------ #
# 2. Synthetic power-law noise PSD and noise generator                 #
# ------------------------------------------------------------------ #


def make_Sn(f0, T_obs, P_sig):
    """Return a power-law PSD callable scaled so flat-noise SNR ≈ TARGET_SNR.

    Convention matches the NUFFT normalization:
        <|weights_normalized[k]|^2> * T_obs = Sn(f_k)
    """
    S0 = P_sig * T_obs / TARGET_SNR**2  # PSD amplitude at f0 [strain^2 / Hz]

    def Sn(f):
        f = np.asarray(f, dtype=float)
        out = np.zeros_like(f)
        pos = f > 0
        out[pos] = S0 * (f[pos] / f0) ** ALPHA_NOISE
        return out

    return Sn


def make_noise_generator(n_samples, T_obs, Sn):
    """Return a callable that draws one complex coloured-noise realisation.

    Normalization: the NUFFT of the generated noise satisfies
        <|NUFFT(noise)[k] / N|^2> * T_obs  ~  Sn(f_k)

    Implementation: draw noise in the frequency domain so that
        E[|FFT(noise)[k]|^2] = N * F_S * Sn(|f_k|)
    which implies the above through the equivalence of NUFFT and FFT on
    a uniform grid.
    """
    freqs = np.fft.fftfreq(n_samples, d=1.0 / F_S)  # Hz
    psd = Sn(np.abs(freqs))
    # std of each complex FFT component (real and imag independently)
    std = np.sqrt(n_samples * F_S * psd / 2.0)
    std[0] = 0.0  # zero DC

    rng = np.random.default_rng(SEED)

    def draw():
        noise_fd = std * (rng.standard_normal(n_samples) + 1j * rng.standard_normal(n_samples))
        return np.fft.ifft(noise_fd)  # complex coloured noise

    return draw


# ------------------------------------------------------------------ #
# 3. Effective NUFFT noise PSD at the 5 sideband bins                 #
# ------------------------------------------------------------------ #


def analytical_S_eff(freqs_out_hz, beta, T_obs, Sn, n_t=2000):
    """
    Stationary-phase time-average of S_n along the chirp track.

        S_eff(f_out) = (1/T) * integral_0^T S_n( f_out * dtau/dt(t) ) dt

    where  dtau/dt = (1 - 8*beta*t/3)^{-3/8}

    Parameters
    ----------
    freqs_out_hz : array  Output frequencies (Hz) where S_eff is needed
    beta         : float  Chirp parameter (s^-1)
    T_obs        : float  Observation time (clock seconds)
    Sn           : callable  Noise PSD function
    n_t          : int    Quadrature points

    Returns
    -------
    S_eff : array, same shape as freqs_out_hz
    """
    freqs_out_hz = np.asarray(freqs_out_hz, dtype=float)
    t_q = np.linspace(0, T_obs, n_t, endpoint=False)
    dtau_dt = (1.0 - (8.0 / 3.0) * beta * t_q) ** (-3.0 / 8.0)  # (n_t,)
    f_in = freqs_out_hz[:, None] * dtau_dt[None, :]  # (N_f, n_t)
    return np.mean(Sn(f_in), axis=1)  # (N_f,)


# ------------------------------------------------------------------ #
# 4. Noise-weighted joint matched filter                               #
# ------------------------------------------------------------------ #


def noise_weighted_estimator(data_X, template_Xp, template_Xc, sigma2):
    """
    Noise-weighted joint matched filter for two GW polarisations.

    Solves the normal equations
        (A† C^{-1} A) [h_p, h_c]^T = A† C^{-1} X
    where  C = diag(sigma2),  A = [template_Xp | template_Xc]  (5x2)

    Parameters
    ----------
    data_X                   : (5,) complex  Data 5-vector
    template_Xp, template_Xc : (5,) complex  Plus/cross-pol templates
    sigma2                   : (5,) real     Noise variance per NUFFT bin

    Returns
    -------
    hp, hc : complex  Estimated GW amplitudes
    snr_sq : float    Matched-filter SNR^2  =  h† (A† C^{-1} A) h
    """
    C_inv = 1.0 / sigma2  # (5,)
    A = np.column_stack([template_Xp, template_Xc])  # (5, 2)
    AtC = A.conj().T * C_inv  # (2, 5)
    AtCA = AtC @ A  # (2, 2)
    AtCX = AtC @ data_X  # (2,)
    h = np.linalg.solve(AtCA, AtCX)  # [hp, hc]
    snr_sq = float(np.real(np.conj(AtCX) @ h))
    return h[0], h[1], snr_sq


# ------------------------------------------------------------------ #
# 5. Main                                                              #
# ------------------------------------------------------------------ #


def main():
    print("=" * 60)
    print("Noise-weighted 5-vector estimator")
    print("=" * 60)

    # ----- Build signal and templates -----
    D = build_signal_and_templates()
    f0 = D["f0"]
    beta = D["beta"]
    T_obs = D["T_obs"]
    n_samples = D["n_samples"]
    signal = D["signal"]
    tau = D["tau"]
    omega0 = D["omega0"]
    template_Xp = D["template_Xp"]
    template_Xc = D["template_Xc"]
    data_X_signal = D["data_X_signal"]

    chirp_param = 8 / 3 * beta * T_obs
    print(f"\nf0       = {f0:.6f} Hz")
    print(f"beta     = {beta:.3e} s^-1")
    print(f"T_obs    = {T_obs:.0f} s")
    print(f"8/3*beta*T_obs = {chirp_param:.4f}  (chirp parameter)")
    print(f"h0 mean  = {np.mean(D['h0']):.3e}")

    # ----- PSD and noise -----
    P_sig = np.sum(np.abs(data_X_signal) ** 2)
    Sn = make_Sn(f0, T_obs, P_sig)
    make_noise = make_noise_generator(n_samples, T_obs, Sn)

    # ----- Effective PSD at the 5 sideband bins -----
    freqs_5vec_hz = np.array([f0 + k / SIDE_DAY for k in range(-2, 3)])
    S_eff = analytical_S_eff(freqs_5vec_hz, beta, T_obs, Sn)
    S_naive = Sn(freqs_5vec_hz)  # naive: just Sn evaluated at each bin centre

    sigma2_eff = S_eff / T_obs
    sigma2_naive = S_naive / T_obs

    print("\nEffective vs naive PSD at the 5 sideband bins:")
    for k in range(5):
        ratio = S_eff[k] / S_naive[k]
        print(f"  bin k={k - 2:+d}: S_eff={S_eff[k]:.4e}  S_naive={S_naive[k]:.4e}  ratio={ratio:.4f}")
    mean_ratio = np.mean(S_eff / S_naive)
    print(f"  Mean S_eff/S_naive = {mean_ratio:.4f}  (< 1: signal chirps into quieter noise)")

    # ----- Theoretical SNR from signal-only 5-vector -----
    _, _, snr_sq_theory_eff = noise_weighted_estimator(data_X_signal, template_Xp, template_Xc, sigma2_eff)
    _, _, snr_sq_theory_naive = noise_weighted_estimator(data_X_signal, template_Xp, template_Xc, sigma2_naive)

    print(f"\nTheoretical SNR^2 (corrected, S_eff)  = {snr_sq_theory_eff:.2f}")
    print(f"Theoretical SNR^2 (naive,     S_n(fk)) = {snr_sq_theory_naive:.2f}")
    print(
        f"Ratio naive/corrected = {snr_sq_theory_naive / snr_sq_theory_eff:.4f}  "
        f"(naive underestimates SNR by {100 * (1 - snr_sq_theory_naive / snr_sq_theory_eff):.1f}%)"
    )

    # ----- Single noisy realisation -----
    noise0 = make_noise()
    data_X_noisy, _ = _resample_and_extract_5vec(signal + noise0, tau, omega0)

    hp_eff, hc_eff, snr_sq_eff = noise_weighted_estimator(data_X_noisy, template_Xp, template_Xc, sigma2_eff)
    hp_naive, hc_naive, snr_sq_naive = noise_weighted_estimator(data_X_noisy, template_Xp, template_Xc, sigma2_naive)

    # Unweighted (old pipeline)
    hp_old, hc_old = _joint_estimator(data_X_noisy, template_Xp, template_Xc)
    old_stat = _detection_stat(template_Xp, template_Xc, hp_old, hc_old)

    print(f"\nSingle noisy realisation:")
    print(f"  Old _detection_stat                  = {old_stat:.4e}  (arbitrary units)")
    print(f"  SNR^2 (corrected, S_eff)             = {snr_sq_eff:.2f}")
    print(f"  SNR^2 (naive,     S_n(fk))           = {snr_sq_naive:.2f}")
    print(f"  Theory (corrected, + 2 dof expected) = {snr_sq_theory_eff + 2:.2f}")

    # ----- Monte Carlo -----
    print(f"\nMonte Carlo ({N_REAL} realisations) ...")
    snr_eff_mc = np.zeros(N_REAL)
    snr_naive_mc = np.zeros(N_REAL)

    for i in range(N_REAL):
        n_i = make_noise()
        dX, _ = _resample_and_extract_5vec(signal + n_i, tau, omega0)
        _, _, snr_eff_mc[i] = noise_weighted_estimator(dX, template_Xp, template_Xc, sigma2_eff)
        _, _, snr_naive_mc[i] = noise_weighted_estimator(dX, template_Xp, template_Xc, sigma2_naive)

    # Expected mean: theory + 2 (2 real DOF from 2 complex polarisation amplitudes)
    expected_eff = snr_sq_theory_eff + 2
    expected_naive = snr_sq_theory_naive + 2

    print(
        f"  SNR^2_eff   mean={snr_eff_mc.mean():.2f}  expected={expected_eff:.2f}  "
        f"ratio={snr_eff_mc.mean() / expected_eff:.4f}"
    )
    print(
        f"  SNR^2_naive mean={snr_naive_mc.mean():.2f}  expected={expected_naive:.2f}  "
        f"ratio={snr_naive_mc.mean() / expected_naive:.4f}"
    )

    # ----- Plots -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)

    # Panel 1: S_eff vs S_naive at the 5 bins
    ax = axes[0]
    x = np.arange(5)
    w = 0.35
    ax.bar(x - w / 2, S_naive, w, label=r"$S_n(f_k)$  naive", color="C0", alpha=0.85)
    ax.bar(x + w / 2, S_eff, w, label=r"$S_\mathrm{eff}$  corrected", color="C1", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={k:+d}" for k in range(-2, 3)])
    ax.set_ylabel(r"PSD [strain$^2$ Hz$^{-1}$]")
    ax.set_title("Effective vs naive PSD\nat the 5 sideband bins")
    ax.legend(fontsize=9)
    ax.text(
        0.05,
        0.95,
        rf"$\langle S_{{\rm eff}}/S_n(f_k) \rangle = {mean_ratio:.3f}$",
        transform=ax.transAxes,
        va="top",
        fontsize=9,
    )

    # Panel 2: SNR^2 distributions from Monte Carlo
    ax = axes[1]
    all_vals = np.concatenate([snr_eff_mc, snr_naive_mc])
    bins = np.linspace(all_vals.min() * 0.8, all_vals.max() * 1.15, 35)
    ax.hist(snr_eff_mc, bins=bins, alpha=0.6, color="C1", label=r"SNR$^2$ ($S_\mathrm{eff}$ corrected)")
    ax.hist(snr_naive_mc, bins=bins, alpha=0.6, color="C0", label=r"SNR$^2$ ($S_n(f_k)$ naive)")
    ax.axvline(expected_eff, color="C1", lw=2, ls="--", label=rf"Theory (corr.) = {expected_eff:.1f}")
    ax.axvline(expected_naive, color="C0", lw=2, ls=":", label=rf"Theory (naive) = {expected_naive:.1f}")
    ax.set_xlabel(r"SNR$^2$")
    ax.set_ylabel("Count")
    ax.set_title(rf"SNR$^2$ distribution  ({N_REAL} realisations)")
    ax.legend(fontsize=8)

    # Panel 3: Recovered vs true amplitudes for single realisation
    ax = axes[2]
    # True amplitudes from the signal-only estimator
    hp_true, hc_true, _ = noise_weighted_estimator(data_X_signal, template_Xp, template_Xc, sigma2_eff)
    for h_true_i, h_est_i, lbl, col in [
        (hp_true, hp_eff, r"$h_+$", "C2"),
        (hc_true, hc_eff, r"$h_\times$", "C3"),
    ]:
        ax.plot(h_true_i.real, h_true_i.imag, "o", color=col, ms=9, label=f"{lbl} true")
        ax.plot(h_est_i.real, h_est_i.imag, "x", color=col, ms=9, mew=2.5, label=f"{lbl} estimated")
        ax.annotate(
            "",
            xy=(h_est_i.real, h_est_i.imag),
            xytext=(h_true_i.real, h_true_i.imag),
            arrowprops=dict(arrowstyle="->", color=col, lw=1.5),
        )
    ax.axhline(0, color="gray", lw=0.5)
    ax.axvline(0, color="gray", lw=0.5)
    ax.set_xlabel(r"Re$(\hat{h})$")
    ax.set_ylabel(r"Im$(\hat{h})$")
    ax.set_title("Recovered vs true amplitudes\n(corrected MF, single realisation)")
    ax.legend(fontsize=8)
    ax.set_aspect("equal")

    fig.suptitle(
        rf"Noise-weighted 5-vector estimator  —  "
        rf"$f_0={f0:.4f}$ Hz,  $\beta={beta:.2e}$ s$^{{-1}}$,  "
        rf"$\alpha={ALPHA_NOISE:.0f}$,  $M_c={MC_SOLAR:.0f}\,M_\odot$",
        fontsize=10,
    )

    out_path = Path(__file__).resolve().parent / "figs" / "noise_weighted_estimator.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()
