"""
Compare the PSD of H1 design-PSD noise estimated via a standard FFT against
a NUFFT on the resampled time coordinate tau.

For white noise the two agree.  For coloured (H1) noise they differ: the
NUFFT at each output bin averages S_n along the chirp frequency track rather
than reading off S_n at that bin's frequency.

Parameters
----------
F_S   = 64 Hz,  T_OBS = 512 s  ->  N = 32 768 samples
F0    = 15 Hz,  BETA = 5.449e-4  ->  f sweeps 15 -> 25 Hz over T_OBS
Both endpoints lie in H1's sensitive band (finite PSD from ~10 to 32 Hz).
bilby requires F_S * T_OBS to be an integer (32768 ✓).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d  # used for Sn interpolant below

bilby.core.utils.logger.setLevel("WARNING")

NOV2025_DIR = Path(__file__).resolve().parents[1]
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from resampler import Resampler

# ------------------------------------------------------------------ #
# Parameters                                                           #
# ------------------------------------------------------------------ #

SEED = 1
N_REALIZATIONS = 5
T_OBS = 512  # seconds (integer)
F_S = 2048.0  # Hz
N_SAMPLES = int(F_S * T_OBS)  # 32 768
F0 = 25.0  # Hz  (start of chirp, in H1 sensitive band)
BETA = 4e-4  # s^-1; gives f_end ≈ 25 Hz at t = T_OBS
OMEGA0 = 2 * np.pi * F0
f_end = F0 * (1 - 8 / 3 * BETA * T_OBS) ** (-3 / 8)
print(f_end)

t = np.linspace(0, T_OBS, N_SAMPLES, endpoint=False)
tau = -(3 / (5 * BETA)) * (1 - 8 / 3 * BETA * t) ** (5 / 8)
tau -= tau[0]


# ------------------------------------------------------------------ #
# Run                                                                  #
# ------------------------------------------------------------------ #


def run():
    np.random.seed(SEED)

    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]

    psd_fft_acc = np.zeros(N_SAMPLES)
    psd_nufft_acc = None
    freqs_nufft = None

    f_fft = np.fft.fftfreq(N_SAMPLES, d=1 / F_S)
    highpass = np.abs(f_fft) >= 20.0  # mask below 20 Hz

    for _ in range(N_REALIZATIONS):
        ifo.set_strain_data_from_power_spectral_density(sampling_frequency=F_S, duration=T_OBS, start_time=-T_OBS / 2)
        strain = ifo.strain_data.time_domain_strain  # real, shape (N_SAMPLES,)

        # Zero out frequencies below 20 Hz for both FFT and NUFFT
        strain_fft = np.fft.fft(strain)
        strain_fft[~highpass] = 0.0
        strain = np.fft.ifft(strain_fft).real

        # FFT PSD
        W = strain_fft / N_SAMPLES
        psd_fft_acc += np.abs(W) ** 2 * T_OBS

        # NUFFT PSD
        r = Resampler()
        r.timeseries = strain.astype(complex)
        r.resampled_time = tau
        r.nufft()
        power = np.abs(r.weights_normalized) ** 2 * T_OBS
        if psd_nufft_acc is None:
            psd_nufft_acc = power
            freqs_nufft = r.freq_in_hz
        else:
            psd_nufft_acc += power

    psd_fft = psd_fft_acc / N_REALIZATIONS
    psd_nufft = psd_nufft_acc / N_REALIZATIONS
    freqs_fft = f_fft

    # H1 design PSD for reference
    f_design = ifo.strain_data.frequency_array
    psd_design = ifo.power_spectral_density_array

    return dict(
        freqs_fft=freqs_fft,
        psd_fft=psd_fft,
        freqs_nufft=freqs_nufft,
        psd_nufft=psd_nufft,
        f_design=f_design,
        psd_design=psd_design,
    )


# ------------------------------------------------------------------ #
# Analytical NUFFT PSD prediction                                      #
# ------------------------------------------------------------------ #
# From the phase-mixing kernel derivation (stationary-phase approx):
#
#   <S_NUFFT(f_out)> = (1/T) ∫₀ᵀ S_n( f_out · τ'(t) ) dt
#
# where τ'(t) = dτ/dt = (1 - 8β t/3)^{-3/8} maps each output frequency
# to the instantaneous input frequency being sampled at time t.
# The highpass at 20 Hz means S_n_filtered = 0 for f < 20 Hz.


def analytical_nufft_psd(freqs_out, f_design, psd_design, highpass_hz=20.0, n_t=2000):
    """
    Predicted NUFFT PSD at each output frequency via stationary-phase formula.

    Parameters
    ----------
    freqs_out : array (N_f,)   Output frequencies to evaluate (Hz, positive)
    f_design  : array          Bilby PSD frequency axis
    psd_design: array          Bilby one-sided PSD values
    highpass_hz: float         High-pass cutoff applied to input noise (Hz)
    n_t       : int            Number of quadrature points for the time integral

    Returns
    -------
    psd_pred : array (N_f,)    Predicted NUFFT PSD (same units as psd_design)
    """
    # Build log-space interpolant of S_n; extrapolate with zero outside range
    finite = np.isfinite(psd_design) & (psd_design > 0) & (f_design >= highpass_hz)
    log_Sn = interp1d(
        np.log(f_design[finite]),
        np.log(psd_design[finite]),
        kind="linear",
        bounds_error=False,
        fill_value=-np.inf,  # log(0) outside range -> Sn = 0
    )

    def Sn(f):
        """Evaluate S_n with highpass applied; shape preserved."""
        f = np.asarray(f)
        log_val = log_Sn(np.where(f > 0, np.log(np.maximum(f, 1e-30)), -np.inf))
        result = np.exp(log_val)
        result[f < highpass_hz] = 0.0
        return result

    # Coarse time grid for quadrature (integrand is smooth)
    t_quad = np.linspace(0, T_OBS, n_t, endpoint=False)
    dtau_dt = (1.0 - (8.0 / 3.0) * BETA * t_quad) ** (-3.0 / 8.0)  # shape (n_t,)

    # f_in[i, j] = freqs_out[i] * dtau_dt[j]  ->  shape (N_f, n_t)
    f_in = freqs_out[:, None] * dtau_dt[None, :]  # (N_f, n_t)
    Sn_track = Sn(f_in)  # (N_f, n_t)

    # Time-average  = (1/T) ∫ S_n(f_out * τ'(t)) dt  ≈ mean over quadrature points
    psd_pred = np.mean(Sn_track, axis=1)  # (N_f,)
    return psd_pred


# ------------------------------------------------------------------ #
# Plot                                                                 #
# ------------------------------------------------------------------ #


def logbin(f, y, n_bins=2000):
    """Average (f, y) into n_bins logarithmically-spaced bins.

    Reduces millions of points to a few thousand for fast plotting on a log
    frequency axis with no visible loss of information.
    """
    f = np.asarray(f)
    y = np.asarray(y)
    edges = np.geomspace(f[f > 0].min(), f.max(), n_bins + 1)
    idx = np.searchsorted(edges, f) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    f_out, y_out = [], []
    for i in range(n_bins):
        mask = idx == i
        if mask.any():
            f_out.append(f[mask].mean())
            y_out.append(y[mask].mean())
    return np.array(f_out), np.array(y_out)


def make_plot(R):
    pos_fft = R["freqs_fft"] > 0
    pos_nufft = R["freqs_nufft"] > 0
    finite_des = np.isfinite(R["psd_design"]) & (R["f_design"] > 0) & (R["f_design"] < 700)

    # Downsample first, then compute analytical prediction only on the small grid.
    # This avoids building a (524K × 2000) matrix inside analytical_nufft_psd.
    # Multiply by 2 to convert two-sided PSD (from FFT/NUFFT of real signal) to
    # one-sided, matching the convention of bilby's psd_design and analytical_nufft_psd.
    _c1 = logbin(R["freqs_fft"][pos_fft], 2 * R["psd_fft"][pos_fft])
    _c2 = logbin(R["freqs_nufft"][pos_nufft], 2 * R["psd_nufft"][pos_nufft])
    _c4 = logbin(R["f_design"][finite_des], R["psd_design"][finite_des])
    # Analytical prediction evaluated on the already-downsampled NUFFT frequency grid
    psd_pred_binned = analytical_nufft_psd(_c2[0], R["f_design"], R["psd_design"])
    print(
        f"Plot lengths — FFT PSD: {len(_c1[0])}, NUFFT PSD: {len(_c2[0])}, Predicted: {len(psd_pred_binned)}, Design PSD: {len(_c4[0])}"
    )

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True, sharex=True)

    ax.loglog(*_c1, lw=1.2, label="FFT PSD")
    ax.loglog(*_c2, lw=1.2, label="NUFFT PSD (resampled τ)")
    ax.loglog(
        _c2[0],
        psd_pred_binned,
        lw=1.5,
        ls="-.",
        color="C2",
        label=r"Predicted NUFFT PSD: $\frac{1}{T}\int S_n(f_\mathrm{out}\,\dot\tau(t))\,dt$",
    )
    ax.loglog(
        *_c4,
        lw=1.5,
        ls="--",
        color="k",
        label="H1 design PSD (bilby)",
    )

    xlim = (20, 700)
    ax.set_xlim(xlim)
    # Set ylim based on the H1 design PSD within the frequency band of interest
    in_band = finite_des & (R["f_design"] >= xlim[0]) & (R["f_design"] <= xlim[1])
    psd_min = R["psd_design"][in_band].min()
    psd_max = R["psd_design"][in_band].max()
    ax.set_ylim(psd_min * 0.3, psd_max * 3)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD [strain² / Hz]")
    ax.set_title(f"H1 noise: FFT vs NUFFT PSD  —  β={BETA:.3e}  ({N_REALIZATIONS} realisations)")
    ax.legend(fontsize=8)

    # --- Whitening verification ---
    # Dividing NUFFT weights by sqrt(psd_pred / T_OBS) should whiten the noise.
    # Equivalently, the ratio psd_nufft / psd_pred should be flat at 1.
    # Use the already-downsampled grids to avoid recomputing on 524K points.
    valid_binned = psd_pred_binned > 0
    ratio = _c2[1][valid_binned] / psd_pred_binned[valid_binned]
    f_ratio = _c2[0][valid_binned]

    print(f"Plot lengths — Ratio: {len(ratio)}")
    ax2.semilogx(f_ratio, ratio, lw=1.2, color="C1", label="NUFFT PSD / Predicted PSD")
    ax2.axhline(np.mean(ratio), color="C2", lw=1.0, ls=":", label=f"Mean = {np.mean(ratio):.3f}")

    ax2.set_xlabel("Frequency (Hz)")
    ax2.set_ylabel("Ratio")
    ax2.set_title("Whitening check: NUFFT PSD / Predicted PSD  (flat ≈ 1 confirms whitening works)")
    ax2.legend(fontsize=8)

    return fig


def main():
    print(f"N_SAMPLES={N_SAMPLES},  N_REALIZATIONS={N_REALIZATIONS}")
    print(f"f0={F0} Hz  ->  f_end={f_end:.3f} Hz  (Nyquist={F_S / 2:.0f} Hz)")

    R = run()
    print("Saving plot")

    out = Path(__file__).resolve().parent / "figs" / "nufft_noise_psd.png"
    make_plot(R).savefig(out, dpi=200)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
