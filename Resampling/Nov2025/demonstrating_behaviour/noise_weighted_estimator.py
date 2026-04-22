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

Parameters:
  - f_0 = 20 Hz, F_S = 100 Hz  (Nyquist = 50 Hz; signal stays in band throughout chirp)
  - M_c = 0.01 solar masses  ->  beta ~8.0e-7 s^-1,  8/3*beta*T_obs ~0.37  (2 sidereal days)
  - Signal chirps 20 Hz -> ~23.7 Hz over 2 sidereal days  (Delta_f ~ 3.7 Hz)
  - T_coal ~ 5.5 sidereal days: signal does not merge during the observation
  - Noise from bilby H1 design PSD (finite above ~10 Hz; f0=20 Hz is in the sensitive band)

Noise is drawn from the bilby H1 interferometer design PSD. The Sn interpolant is built
from the same PSD and passed to analytical_S_eff for the noise-transfer correction.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import bilby
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

bilby.core.utils.logger.setLevel("WARNING")

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
MC_SOLAR = 0.01  # chirp mass [solar masses]; T_coal ~ 5.5 days at f0=20 Hz
F0 = 20
MC = MC_SOLAR * 2e30  # kg
N_REAL = 10  # Monte Carlo noise realisations
F_S = 100.0  # Hz (Nyquist = 50 Hz; signal chirps 20->24 Hz, well within band)

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
    signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(f0_setting=20.0, Mc=MC, f_signal=F_S, n_days=2)
    signal = 3e-2 * signal  # Decrease the SNR to reasonable numbers

    n_samples = len(signal)
    T_obs = n_samples / F_S
    f0 = omega0 / (2 * np.pi)
    beta = _CHIRP_CONST * f0 ** (8 / 3) * MC ** (5 / 3)

    f_start = f0
    f_end = f0 * (1.0 - (8.0 / 3.0) * beta * T_obs) ** (-3.0 / 8.0)
    print(f"Chirp: f_start = {f_start:.1f} Hz -> f_end = {f_end:.1f} Hz")

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
# 2. Bilby H1 noise PSD and noise generator                            #
# ------------------------------------------------------------------ #


def make_bilby_noise_and_psd(n_samples, T_obs):
    """Set up bilby H1 noise generator and PSD interpolant.

    Returns
    -------
    Sn : callable  Two-sided noise PSD (Hz → strain² Hz⁻¹).
                   Convention: E[|weights_normalized[k]|²] * T_obs = Sn(|f_k|).
                   Equals bilby's one-sided design PSD divided by 2 (real signal).
    draw : callable  Returns one real noise realisation as a numpy array
    """
    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
    ifo.set_strain_data_from_power_spectral_density(sampling_frequency=F_S, duration=T_obs, start_time=-T_obs / 2)
    f_design = ifo.strain_data.frequency_array
    psd_design = ifo.power_spectral_density_array

    # bilby provides the one-sided PSD; divide by 2 to get the two-sided PSD that
    # matches the NUFFT convention:  E[|weights_normalized[k]|²] * T_obs = S_two_sided.
    # (nufft_noise_psd.py confirms: 2 * psd_fft = S_bilby_one_sided.)
    finite = np.isfinite(psd_design) & (psd_design > 0) & (f_design > 0)
    log_Sn = interp1d(
        np.log(f_design[finite]),
        np.log(psd_design[finite] / 2),  # /2: one-sided → two-sided
        kind="linear",
        bounds_error=False,
        fill_value=-np.inf,
    )

    def Sn(f):
        f = np.asarray(f, dtype=float)
        log_val = log_Sn(np.where(f > 0, np.log(np.maximum(f, 1e-30)), -np.inf))
        out = np.exp(log_val)
        out[f <= 0] = 0.0
        return out

    def draw():
        ifo.set_strain_data_from_power_spectral_density(sampling_frequency=F_S, duration=T_obs, start_time=-T_obs / 2)
        return ifo.strain_data.time_domain_strain  # real numpy array

    return Sn, draw


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
    Sn, make_noise = make_bilby_noise_and_psd(n_samples, T_obs)

    # ----- Effective PSD at the 5 sideband bins -----
    freqs_5vec_hz = np.array([f0 + k / SIDE_DAY for k in range(-2, 3)])
    S_eff = analytical_S_eff(freqs_5vec_hz, beta, T_obs, Sn)
    S_naive = Sn(freqs_5vec_hz)  # naive: just Sn evaluated at each bin centre

    sigma2_eff = S_eff / T_obs
    sigma2_naive = S_naive / T_obs

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
    # old_stat = _detection_stat(template_Xp, template_Xc, hp_old, hc_old)

    print(f"\nSingle noisy realisation:")
    # print(f"  Old _detection_stat                  = {old_stat:.4e}  (arbitrary units)")
    print(f"  SNR^2 (corrected, S_eff)             = {snr_sq_eff:.2f}")
    print(f"  SNR^2 (naive,     S_n(fk))           = {snr_sq_naive:.2f}")
    print(f"  Theory (corrected, + 2 dof expected) = {snr_sq_theory_eff + 2:.2f}")

    # ----- Monte Carlo -----
    print(f"\n({N_REAL} realisations) ...")
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

    # ----- Plot: SNR^2 distributions from Monte Carlo -----
    fig, ax = plt.subplots(1, 1, figsize=(6, 4.5), constrained_layout=True)
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

    fig.suptitle(
        rf"Noise-weighted 5-vector estimator  —  "
        rf"$f_0={f0:.4f}$ Hz,  $\beta={beta:.2e}$ s$^{{-1}}$,  "
        rf"$M_c={MC_SOLAR:.1e}\,M_\odot$",
        fontsize=10,
    )

    out_path = Path(__file__).resolve().parent / "figs/noise_weighted_estimator.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot -> {out_path}")


if __name__ == "__main__":
    main()
