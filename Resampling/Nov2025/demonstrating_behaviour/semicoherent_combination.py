"""
Compare a fully coherent 4-day 5-vector analysis against a semicoherent sum of
two 2-day chunks.

The key point being tested is not the raw summed detection statistic itself, but
its detectability:

    detectability = (mean(signal + noise) - mean(noise)) / std(noise)

For two equal chunks, the semicoherent signal contribution should stay close to
the coherent one, while the noise fluctuations grow by about sqrt(2). This
predicts

    detectability_semi / detectability_coh ~= 1 / sqrt(2)

The script generates one 4-day PBH chirp, analyzes it coherently across the full
span, then re-analyzes the same data as two independent 2-day chunks whose
detection statistics are added. A small Monte Carlo with bilby H1 design noise is
used to check the expected sensitivity loss numerically.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import bilby
import matplotlib.pyplot as plt
import numpy as np

bilby.core.utils.logger.setLevel("WARNING")

NOV2025_DIR = Path(__file__).resolve().parent.parent
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import (
    _create_PBH_signal,
    _detection_stat,
    _estimator,
    _joint_estimator,
    _resample_and_extract_5vec,
    _time_domain_5vec,
)


SEED = 7
TOTAL_DAYS = 4
N_SEGMENTS = 2
SEGMENT_DAYS = TOTAL_DAYS // N_SEGMENTS
MC_SOLAR = 0.01
MC = MC_SOLAR * 2e30
F_S = 32.0
F0 = 12.0
SIGNAL_SCALE = float(os.getenv("PBH_SEMICOHERENT_SIGNAL_SCALE", "0.03"))
N_REAL = int(os.getenv("PBH_SEMICOHERENT_NREAL", "16"))
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "semicoherent_combination.png"


def _segment_slices(n_samples, n_segments):
    edges = np.linspace(0, n_samples, n_segments + 1, dtype=int)
    return [slice(edges[i], edges[i + 1]) for i in range(n_segments)]


def build_signal():
    """Create one fixed 4-day PBH chirp in the detector band."""
    np.random.seed(SEED)
    signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(
        n_days=TOTAL_DAYS,
        f0_setting=F0,
        Mc=MC,
        f_signal=F_S,
    )
    return {
        "signal": SIGNAL_SCALE * signal,
        "tau": tau,
        "omega0": omega0,
        "sidereal": sidereal,
        "h0": SIGNAL_SCALE * h0,
        "gamma": gamma,
        "t": t,
        "n_samples": len(signal),
        "T_obs": len(signal) / F_S,
        "f0": omega0 / (2.0 * np.pi),
    }


def analyze_segment(signal_seg, t_seg, tau_seg, sidereal, omega0):
    """Run the standard 5-vector pipeline on one contiguous segment."""
    tau_local = tau_seg - tau_seg[0]
    data_X, _ = _resample_and_extract_5vec(signal_seg, tau_local, omega0)
    template_X, template_Xp, template_Xc = _time_domain_5vec(sidereal, t_seg, tau_local)
    h_est = _estimator(data_X, template_X)
    hp_est, hc_est = _joint_estimator(data_X, template_Xp, template_Xc)
    det_stat = float(_detection_stat(template_Xp, template_Xc, hp_est, hc_est))
    return {
        "data_X": data_X,
        "template_X": template_X,
        "template_Xp": template_Xp,
        "template_Xc": template_Xc,
        "h_est": h_est,
        "hp_est": hp_est,
        "hc_est": hc_est,
        "power": float(np.abs(h_est) ** 2),
        "det_stat": det_stat,
    }


def analyze_observation(signal, t, tau, sidereal, omega0):
    """Return coherent and two-chunk semicoherent summaries for one dataset."""
    coherent = analyze_segment(signal, t, tau, sidereal, omega0)
    chunk_results = []
    for seg in _segment_slices(signal.size, N_SEGMENTS):
        chunk_results.append(analyze_segment(signal[seg], t[seg], tau[seg], sidereal, omega0))

    semicoherent = {
        "power_sum": float(sum(chunk["power"] for chunk in chunk_results)),
        "det_stat_sum": float(sum(chunk["det_stat"] for chunk in chunk_results)),
        "chunks": chunk_results,
    }
    return coherent, semicoherent


def make_bilby_noise_drawer(n_samples, duration):
    """Return a callable that draws one H1 design-noise realization."""
    ifo = bilby.gw.detector.InterferometerList(["H1"])[0]

    def draw():
        ifo.set_strain_data_from_power_spectral_density(
            sampling_frequency=F_S,
            duration=duration,
            start_time=-duration / 2.0,
        )
        noise = ifo.strain_data.time_domain_strain
        if noise.size != n_samples:
            raise RuntimeError(f"Expected {n_samples} samples from bilby, got {noise.size}")
        return noise

    return draw


def summarise_distribution(signal_plus_noise, noise_only):
    signal_contribution = float(np.mean(signal_plus_noise) - np.mean(noise_only))
    noise_std = float(np.std(noise_only, ddof=1))
    detectability = signal_contribution / noise_std
    return {
        "signal_contribution": signal_contribution,
        "noise_std": noise_std,
        "detectability": detectability,
        "mean_noise": float(np.mean(noise_only)),
        "mean_signal": float(np.mean(signal_plus_noise)),
    }


def make_plot(coh_noise, coh_signal, semi_noise, semi_signal, coh_summary, semi_summary, signal_only):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), constrained_layout=True)

    panels = [
        ("Coherent 4-day", axes[0], coh_noise, coh_signal, coh_summary),
        ("Semicoherent 2 x 2-day", axes[1], semi_noise, semi_signal, semi_summary),
    ]
    for title, ax, noise_only, signal_plus_noise, summary in panels:
        all_vals = np.concatenate([noise_only, signal_plus_noise])
        bins = np.linspace(all_vals.min(), all_vals.max(), 24)
        ax.hist(noise_only, bins=bins, alpha=0.65, label="Noise only")
        ax.hist(signal_plus_noise, bins=bins, alpha=0.65, label="Signal + noise")
        ax.axvline(summary["mean_noise"], color="C0", lw=2, ls="--")
        ax.axvline(summary["mean_signal"], color="C1", lw=2, ls="--")
        ax.set_title(title)
        ax.set_xlabel("Detection statistic")
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)

    expected_ratio = 1.0 / np.sqrt(N_SEGMENTS)
    measured_ratio = semi_summary["detectability"] / coh_summary["detectability"]
    fig.suptitle(
        (
            f"Semicoherent combination for one {TOTAL_DAYS}-day PBH chirp\n"
            f"signal-only stat ratio={signal_only['det_stat_ratio']:.3f}, "
            f"noise-std ratio={semi_summary['noise_std'] / coh_summary['noise_std']:.3f}, "
            f"detectability ratio={measured_ratio:.3f} (expected {expected_ratio:.3f})"
        ),
        fontsize=10,
    )
    return fig


def main():
    print("=" * 60)
    print("Semicoherent combination of two 2-day chunks")
    print("=" * 60)
    print(f"seed={SEED}, f0={F0:.1f} Hz, Mc={MC_SOLAR:.3g} Msun, F_S={F_S:.1f} Hz")
    print(f"signal scale={SIGNAL_SCALE:.3g}, N_real={N_REAL}")

    D = build_signal()
    signal = D["signal"]
    tau = D["tau"]
    t = D["t"]
    sidereal = D["sidereal"]
    omega0 = D["omega0"]

    print(f"n_samples={D['n_samples']}, T_obs={D['T_obs'] / 86400.0:.6f} days, segment={SEGMENT_DAYS} days")

    coherent_signal, semicoherent_signal = analyze_observation(signal, t, tau, sidereal, omega0)
    signal_only = {
        "coh_power": coherent_signal["power"],
        "semi_power_sum": semicoherent_signal["power_sum"],
        "power_ratio": semicoherent_signal["power_sum"] / coherent_signal["power"],
        "coh_det_stat": coherent_signal["det_stat"],
        "semi_det_stat_sum": semicoherent_signal["det_stat_sum"],
        "det_stat_ratio": semicoherent_signal["det_stat_sum"] / coherent_signal["det_stat"],
    }

    print("\nSignal only:")
    print(f"  coherent power              = {signal_only['coh_power']:.6e}")
    print(f"  semicoherent summed power   = {signal_only['semi_power_sum']:.6e}")
    print(f"  summed-power ratio          = {signal_only['power_ratio']:.6f}")
    print(f"  coherent detection stat     = {signal_only['coh_det_stat']:.6e}")
    print(f"  semicoherent summed stat    = {signal_only['semi_det_stat_sum']:.6e}")
    print(f"  summed-stat ratio           = {signal_only['det_stat_ratio']:.6f}")

    draw_noise = make_bilby_noise_drawer(D["n_samples"], D["T_obs"])

    coh_noise = np.empty(N_REAL)
    coh_signal = np.empty(N_REAL)
    semi_noise = np.empty(N_REAL)
    semi_signal = np.empty(N_REAL)

    print(f"\nRunning {N_REAL} bilby noise realizations ...")
    for i in range(N_REAL):
        noise = draw_noise()
        coherent_noise, semicoherent_noise = analyze_observation(noise, t, tau, sidereal, omega0)
        coherent_sn, semicoherent_sn = analyze_observation(signal + noise, t, tau, sidereal, omega0)
        coh_noise[i] = coherent_noise["det_stat"]
        coh_signal[i] = coherent_sn["det_stat"]
        semi_noise[i] = semicoherent_noise["det_stat_sum"]
        semi_signal[i] = semicoherent_sn["det_stat_sum"]

        print(
            f"  {i + 1:02d}/{N_REAL}: "
            f"coh={coh_signal[i]:.3e}, semi={semi_signal[i]:.3e}, "
            f"semi/coh={semi_signal[i] / coh_signal[i]:.3f}"
        )

    coh_summary = summarise_distribution(coh_signal, coh_noise)
    semi_summary = summarise_distribution(semi_signal, semi_noise)
    detectability_ratio = semi_summary["detectability"] / coh_summary["detectability"]
    expected_ratio = 1.0 / np.sqrt(N_SEGMENTS)

    print("\nNoise summary:")
    print(f"  coherent noise std          = {coh_summary['noise_std']:.6e}")
    print(f"  semicoherent noise std      = {semi_summary['noise_std']:.6e}")
    print(f"  noise-std ratio             = {semi_summary['noise_std'] / coh_summary['noise_std']:.6f}")

    print("\nDetectability:")
    print(f"  coherent shift/std          = {coh_summary['detectability']:.6f}")
    print(f"  semicoherent shift/std      = {semi_summary['detectability']:.6f}")
    print(f"  semi/coh detectability      = {detectability_ratio:.6f}")
    print(f"  expected 1/sqrt(2)          = {expected_ratio:.6f}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig = make_plot(coh_noise, coh_signal, semi_noise, semi_signal, coh_summary, semi_summary, signal_only)
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(f"\nSaved plot -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
