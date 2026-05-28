from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nov2025.resampler import Resampler


G = 6.67430e-11
c = 299792458.0
MSUN = 2.0e30


def pbh_beta(f0_hz, mc_msun):
    """Return the 0PN chirp beta used by the active Nov2025 resampler tests."""
    mc_si = mc_msun * MSUN
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * (G / c**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
        * mc_si ** (5.0 / 3.0)
    )


def build_signal(mc_msun=1e-1, f0_hz=20.0, t_obs=500.0, sample_rate=128.0):
    n_samples = round(sample_rate * t_obs)
    t = np.arange(n_samples, dtype=float) / sample_rate

    beta = pbh_beta(f0_hz, mc_msun)
    chirp_factor = 1.0 - (8.0 / 3.0) * beta * t
    if np.any(chirp_factor <= 0.0):
        raise ValueError("Requested signal reaches coalescence during the observation.")

    phase = (
        -6.0
        * np.pi
        / 5.0
        * f0_hz
        * chirp_factor ** (5.0 / 8.0)
        / beta
    )
    signal = np.exp(1j * (phase - phase[0]))

    tau = -(3.0 / (5.0 * beta)) * chirp_factor ** (5.0 / 8.0)
    tau -= tau[0]

    instantaneous_frequency = f0_hz * chirp_factor ** (-3.0 / 8.0)
    return t, signal, tau, instantaneous_frequency


def naive_fft_spectrum(signal, sample_rate):
    freqs = np.fft.fftshift(np.fft.fftfreq(signal.size, d=1.0 / sample_rate))
    weights = np.fft.fftshift(np.fft.fft(signal)) / signal.size

    # Optional Hann-windowed version for suppressing sidelobe ringing:
    # window = np.hanning(signal.size)
    # weights = np.fft.fftshift(np.fft.fft(signal * window)) / np.sum(window)

    return freqs, np.abs(weights) ** 2


def resampled_spectrum(signal, tau):
    resampler = Resampler()
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.nufft()
    return resampler.freq_in_hz, resampler.power_normalized


def main():
    mc_msun = 1e-1
    f0_hz = 20.0
    t_obs = 500.0
    sample_rate = 128.0

    _, signal, tau, instantaneous_frequency = build_signal(
        mc_msun=mc_msun,
        f0_hz=f0_hz,
        t_obs=t_obs,
        sample_rate=sample_rate,
    )

    fft_freqs, fft_power = naive_fft_spectrum(signal, sample_rate)
    nufft_freqs, nufft_power = resampled_spectrum(signal, tau)

    f_min = f0_hz - 0.12
    f_max = instantaneous_frequency[-1] + 0.12
    fft_mask = (fft_freqs >= f_min) & (fft_freqs <= f_max)
    nufft_mask = (nufft_freqs >= f_min) & (nufft_freqs <= f_max)

    fig, (ax_fft, ax_nufft) = plt.subplots(
        2,
        1,
        figsize=(7.2, 6.0),
        sharex=True,
        constrained_layout=True,
    )

    ax_fft.plot(fft_freqs[fft_mask], fft_power[fft_mask], color="tab:blue", lw=1.4)
    # ax_fft.axvspan(
    #     instantaneous_frequency[0],
    #     instantaneous_frequency[-1],
    #     color="tab:blue",
    #     alpha=0.12,
    #     linewidth=0,
    #     label="chirp track",
    # )
    ax_fft.axvline(
        f0_hz,
        color="black",
        ls="--",
        lw=1.0,
        # label=rf"$f_0={f0_hz:g}\,\mathrm{{Hz}}$",
    )
    ax_fft.set_ylabel("Normalized power")
    ax_fft.set_title("FFT")
    ax_fft.legend(loc="upper right", frameon=False)
    ax_fft.grid(True, alpha=0.25)

    ax_nufft.plot(
        nufft_freqs[nufft_mask],
        nufft_power[nufft_mask],
        color="tab:orange",
        lw=1.4,
    )
    ax_nufft.axvline(
        f0_hz,
        color="black",
        ls="--",
        lw=1.0,
        label=rf"Injected $f_0={f0_hz:g}\,\mathrm{{Hz}}$",
    )
    ax_nufft.set_xlabel("Frequency [Hz]")
    ax_nufft.set_ylabel("Normalized power")
    ax_nufft.set_title("Non-Uniform FFT")
    ax_nufft.legend(loc="upper right", frameon=False)
    ax_nufft.grid(True, alpha=0.25)

    # fig.suptitle(
    #     rf"PBH chirp resampling example: $M_c={mc_msun:.0e}M_\odot$, "
    #     rf"$T_\mathrm{{obs}}={t_obs:g}\,\mathrm{{s}}$"
    # )

    output_dir = Path(__file__).resolve().parent / "figs"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "resampling_example.png"
    fig.savefig(output_path, dpi=220)

    fft_peak = fft_power[fft_mask].max()
    nufft_peak = nufft_power[nufft_mask].max()
    print(f"Saved {output_path}")
    print(f"Naive FFT peak normalized power: {fft_peak:.4g}")
    print(f"Resampled NUFFT peak normalized power: {nufft_peak:.4g}")


if __name__ == "__main__":
    main()
