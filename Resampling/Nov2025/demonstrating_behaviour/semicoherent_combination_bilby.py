"""
Semicoherent combination of a 4-day PBH chirp split into two 2-day chunks.

This version uses no time-domain noise realisations. Instead, the noise
covariance C is built from the theoretically expected NUFFT noise transfer,
following the stationary-phase / point-spreading argument used in
nufft_noise_psd.py:

    S_eff(f_out) = (1/T) ∫ S_n(f_out * dtau/dt) dt

For each analysis segment we:

1. compute the signal-only 5-vector
2. build C = diag(sigma2) from the expected NUFFT PSD in the 5 sideband bins
3. evaluate the calibrated signal-only SNR^2
4. derive the noise-only mean/std analytically from the corresponding matched
   filter projector, with no explicit noise injection

The semicoherent statistic is the sum of the two 2-day SNR^2 values. If the
signal contribution stays roughly unchanged while the chunk noises add in
quadrature, the detectability ratio should approach 1/sqrt(2).
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import bilby
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

bilby.core.utils.logger.setLevel("WARNING")

NOV2025_DIR = Path(__file__).resolve().parent.parent
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import _create_PBH_signal, _resample_and_extract_5vec, _time_domain_5vec


SEED = 7
TOTAL_DAYS = 4
N_SEGMENTS = 2
SEGMENT_DAYS = TOTAL_DAYS // N_SEGMENTS
MC_SOLAR = 0.005
MC = MC_SOLAR * 2e30
F_S = 48.0
SIGNAL_SCALE = float(os.getenv("PBH_SEMICOHERENT_SIGNAL_SCALE", "0.03"))
SIDE_DAY = 86164.09053083288
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "semicoherent_combination.png"

c, G = 3e8, 6.67e-11
CHIRP_CONST = 96 / 5 * np.pi ** (8 / 3) * (G / c**3) ** (5 / 3)


def _segment_slices(n_samples, n_segments):
    edges = np.linspace(0, n_samples, n_segments + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_segments)]


TARGET_F0 = 20.0  # Hz; carrier sits in H1 sensitive band


def _bin_aligned_f0(target_f0):
    """Solve for f0 near target_f0 such that f0 lands on a NUFFT bin in the
    coherent tau-span AND on a NUFFT bin in each segment (carrier_bin is
    constrained to be a multiple of N_SEGMENTS so it halves cleanly).
    """
    T_obs = TOTAL_DAYS * SIDE_DAY
    n_samples = round(F_S * T_obs)
    t_last = T_obs * (n_samples - 1) / n_samples

    def tau_span(f0):
        beta = CHIRP_CONST * f0 ** (8 / 3) * MC ** (5 / 3)
        return (3 / (5 * beta)) * (1 - (1 - 8 / 3 * beta * t_last) ** (5 / 8))

    carrier_bin = N_SEGMENTS * round(target_f0 * t_last / N_SEGMENTS)
    f0 = carrier_bin / t_last
    for _ in range(20):
        f0_next = carrier_bin / tau_span(f0)
        if np.isclose(f0_next, f0, rtol=0.0, atol=1e-14):
            break
        f0 = f0_next
    return float(f0)


def build_signal():
    """Create one fixed 4-day PBH chirp in the detector band."""
    np.random.seed(SEED)
    f0_aligned = _bin_aligned_f0(TARGET_F0)
    signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(
        f0_setting=f0_aligned,
        Mc=MC,
        f_signal=F_S,
        n_days=TOTAL_DAYS,
    )
    signal = SIGNAL_SCALE * signal
    f0 = omega0 / (2.0 * np.pi)
    beta = CHIRP_CONST * f0 ** (8 / 3) * MC ** (5 / 3)
    return {
        "signal": signal,
        "tau": tau,
        "omega0": omega0,
        "sidereal": sidereal,
        "h0": SIGNAL_SCALE * h0,
        "gamma": gamma,
        "t": t,
        "n_samples": len(signal),
        "T_obs": len(signal) / F_S,
        "f0": f0,
        "beta": beta,
    }


def make_bilby_psd(duration):
    """Return the two-sided H1 design-PSD interpolant used by the NUFFT transfer."""
    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
    ifo.set_strain_data_from_power_spectral_density(
        sampling_frequency=F_S,
        duration=duration,
        start_time=-duration / 2.0,
    )
    f_design = ifo.strain_data.frequency_array
    psd_design = ifo.power_spectral_density_array

    finite = np.isfinite(psd_design) & (psd_design > 0) & (f_design > 0)
    log_Sn = interp1d(
        np.log(f_design[finite]),
        np.log(psd_design[finite] / 2.0),
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

    return Sn


def analytical_S_eff(freqs_out_hz, t_start, t_end, beta, Sn, n_t=1000):
    """Expected NUFFT PSD in the 5-vector bins from the stationary-phase transfer.

    t_start, t_end are measured from the chirp start (so dtau/dt has the right
    reference). n_t is the quadrature grid size; the integrand is smooth so a
    coarse grid suffices.
    """
    freqs_out_hz = np.asarray(freqs_out_hz, dtype=float)
    t_quad = np.linspace(t_start, t_end, n_t, endpoint=False)
    dtau_dt = (1.0 - (8.0 / 3.0) * beta * t_quad) ** (-3.0 / 8.0)
    f_in = freqs_out_hz[:, None] * dtau_dt[None, :]
    return np.mean(Sn(f_in), axis=1)


def noise_weighted_estimator(data_X, template_Xp, template_Xc, sigma2):
    """Calibrated two-polarisation matched-filter SNR^2."""
    C_inv = 1.0 / sigma2
    A = np.column_stack([template_Xp, template_Xc])
    AtC = A.conj().T * C_inv
    AtCA = AtC @ A
    AtCX = AtC @ data_X
    h = np.linalg.solve(AtCA, AtCX)
    snr_sq = float(np.real(np.conj(AtCX) @ h))
    return h[0], h[1], snr_sq


def projector_noise_moments(template_Xp, template_Xc, sigma2):
    """
    Noise-only mean/std of the SNR^2 statistic implied by C.

    With y = C^{-1/2} X and W = C^{-1/2} A, the statistic is y† P y where
    P = W (W†W)^{-1} W†. For circular complex Gaussian noise, the first two
    moments are:

        mean = Tr(P)
        var  = Tr(P^2)
    """
    A = np.column_stack([template_Xp, template_Xc])
    W = A / np.sqrt(sigma2)[:, None]
    projector = W @ np.linalg.inv(W.conj().T @ W) @ W.conj().T
    mean_noise = float(np.real(np.trace(projector)))
    var_noise = float(np.real(np.trace(projector @ projector)))
    return mean_noise, np.sqrt(var_noise)


def analyze_segment(signal_seg, t_seg, tau_seg, sidereal, omega0, f0, beta, t_start, t_end, Sn):
    """Run the signal-only 5-vector analysis for one contiguous segment."""
    tau_local = tau_seg - tau_seg[0]
    data_X, _ = _resample_and_extract_5vec(signal_seg, tau_local, omega0)
    _, template_Xp, template_Xc = _time_domain_5vec(sidereal, t_seg, tau_local)

    duration = len(signal_seg) / F_S
    freqs_5vec_hz = np.array([f0 + k / SIDE_DAY for k in range(-2, 3)])
    sigma2 = analytical_S_eff(freqs_5vec_hz, t_start, t_end, beta, Sn) / duration
    _, _, snr_sq = noise_weighted_estimator(data_X, template_Xp, template_Xc, sigma2)
    noise_mean, noise_std = projector_noise_moments(template_Xp, template_Xc, sigma2)
    return {
        "snr_sq": snr_sq,
        "sigma2": sigma2,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
    }


def summarize_case(signal_snr_sq, noise_mean, noise_std):
    return {
        "signal_snr_sq": signal_snr_sq,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "signal_plus_noise_mean": signal_snr_sq + noise_mean,
        "detectability": signal_snr_sq / noise_std,
    }


def analyze_observation(signal, t, tau, sidereal, omega0, f0, beta, Sn):
    """Return coherent and semicoherent summaries using theoretical C only."""
    t_offset = t.gps - t.gps[0]
    dt = 1.0 / F_S
    coherent_seg = analyze_segment(
        signal, t, tau, sidereal, omega0, f0, beta,
        float(t_offset[0]), float(t_offset[-1] + dt), Sn,
    )
    coherent = summarize_case(
        coherent_seg["snr_sq"],
        coherent_seg["noise_mean"],
        coherent_seg["noise_std"],
    )

    chunk_results = []
    for seg in _segment_slices(signal.size, N_SEGMENTS):
        t_off_seg = t_offset[seg]
        chunk_results.append(
            analyze_segment(
                signal[seg],
                t[seg],
                tau[seg],
                sidereal,
                omega0,
                f0,
                beta,
                float(t_off_seg[0]),
                float(t_off_seg[-1] + dt),
                Sn,
            )
        )

    semi_signal_snr_sq = float(sum(chunk["snr_sq"] for chunk in chunk_results))
    semi_noise_mean = float(sum(chunk["noise_mean"] for chunk in chunk_results))
    semi_noise_std = float(np.sqrt(sum(chunk["noise_std"] ** 2 for chunk in chunk_results)))
    semicoherent = summarize_case(semi_signal_snr_sq, semi_noise_mean, semi_noise_std)
    semicoherent["chunks"] = chunk_results

    return coherent, semicoherent


def make_plot(coherent, semicoherent):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)

    labels = ["Coherent\n4 d", "Semicoherent\n2 x 2 d"]
    signal_vals = [coherent["signal_snr_sq"], semicoherent["signal_snr_sq"]]
    detectability_vals = [coherent["detectability"], semicoherent["detectability"]]
    expected_ratio = 1.0 / np.sqrt(N_SEGMENTS)
    measured_ratio = semicoherent["detectability"] / coherent["detectability"]

    ax = axes[0]
    ax.bar(labels, signal_vals, color=["C0", "C1"], alpha=0.85)
    ax.set_ylabel(r"Signal-only SNR$^2$")
    ax.set_title("Signal Contribution")

    ax = axes[1]
    x = np.arange(2)
    width = 0.34
    noise_means = [coherent["noise_mean"], semicoherent["noise_mean"]]
    signal_means = [coherent["signal_plus_noise_mean"], semicoherent["signal_plus_noise_mean"]]
    noise_stds = [coherent["noise_std"], semicoherent["noise_std"]]
    ax.bar(x - width / 2, noise_means, width=width, color="C0", alpha=0.8, label="Noise-only mean")
    ax.bar(x + width / 2, signal_means, width=width, color="C1", alpha=0.8, label="Signal+noise mean")
    ax.errorbar(x - width / 2, noise_means, yerr=noise_stds, fmt="none", ecolor="k", capsize=4, lw=1.2)
    ax.set_xticks(x, labels)
    ax.set_ylabel(r"Expected SNR$^2$")
    ax.set_title("Theoretical Means and H0 Scatter")
    ax.legend(fontsize=8)

    fig.suptitle(
        (
            f"No-noise semicoherent test from theoretical C\n"
            f"detectability ratio={measured_ratio:.3f} (expected {expected_ratio:.3f}), "
            f"signal SNR^2 ratio={semicoherent['signal_snr_sq'] / coherent['signal_snr_sq']:.3f}"
        ),
        fontsize=10,
    )

    text = (
        f"coh detectability = {detectability_vals[0]:.3f}\n"
        f"semi detectability = {detectability_vals[1]:.3f}\n"
        f"semi/coh = {measured_ratio:.3f}"
    )
    fig.text(0.5, 0.02, text, ha="center", va="bottom", fontsize=9)
    return fig


def main():
    print("=" * 60)
    print("Semicoherent combination from theoretical noise covariance")
    print("=" * 60)
    print(f"seed={SEED}, f0=auto (midpoint), Mc={MC_SOLAR:.3g} Msun, F_S={F_S:.1f} Hz")
    print("Using no time-domain noise injection; C comes from the expected NUFFT PSD transfer.")

    D = build_signal()
    signal = D["signal"]
    tau = D["tau"]
    t = D["t"]
    sidereal = D["sidereal"]
    omega0 = D["omega0"]
    f0 = D["f0"]
    beta = D["beta"]

    print(f"n_samples={D['n_samples']}, T_obs={D['T_obs'] / 86400.0:.6f} days, segment={SEGMENT_DAYS} days")
    print(f"signal scale={SIGNAL_SCALE:.3g}")
    f_end = f0 * (1 - 8 / 3 * beta * D["T_obs"]) ** (-3 / 8)
    print(f"chirp track: f_start={f0:.6f} Hz -> f_end={f_end:.6f} Hz")

    Sn = make_bilby_psd(D["T_obs"])
    coherent, semicoherent = analyze_observation(signal, t, tau, sidereal, omega0, f0, beta, Sn)

    print("\nCoherent 4-day:")
    print(f"  signal-only SNR^2         = {coherent['signal_snr_sq']:.6f}")
    print(f"  noise-only mean           = {coherent['noise_mean']:.6f}")
    print(f"  noise-only std            = {coherent['noise_std']:.6f}")
    print(f"  detectability             = {coherent['detectability']:.6f}")

    print("\nSemicoherent 2 x 2-day:")
    print(f"  summed signal-only SNR^2  = {semicoherent['signal_snr_sq']:.6f}")
    print(f"  noise-only mean           = {semicoherent['noise_mean']:.6f}")
    print(f"  noise-only std            = {semicoherent['noise_std']:.6f}")
    print(f"  detectability             = {semicoherent['detectability']:.6f}")

    detectability_ratio = semicoherent["detectability"] / coherent["detectability"]
    expected_ratio = 1.0 / np.sqrt(N_SEGMENTS)
    print("\nComparison:")
    print(f"  semi/coh signal SNR^2     = {semicoherent['signal_snr_sq'] / coherent['signal_snr_sq']:.6f}")
    print(f"  semi/coh noise std        = {semicoherent['noise_std'] / coherent['noise_std']:.6f}")
    print(f"  semi/coh detectability    = {detectability_ratio:.6f}")
    print(f"  expected 1/sqrt(2)        = {expected_ratio:.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = make_plot(coherent, semicoherent)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"\nSaved plot -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
