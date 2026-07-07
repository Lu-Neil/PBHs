"""Inject a 3.5PN chirp at the semicoherent sensitivity and recover track power."""

import argparse
import sys
from pathlib import Path

import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, norm


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = SCRIPT_DIR / "semicoherent_track_injection.png"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from five_vec import five_vec  # noqa: E402
from distance_sensitivity.maximum_sensitivity_35PN import (  # noqa: E402
    LAMBDA_THRESHOLD,
    power_at_distance,
)
from distance_sensitivity.semicoherent_sensitivity import (  # noqa: E402
    semicoherent_distance_sensitivity,
)
from resampler import Resampler  # noqa: E402
from signal_generators import (  # noqa: E402
    C,
    G,
    MSUN,
    NoiseCurve,
    TaylorF2FrequencyModel,
    beta_0pn,
    make_35pn_track,
    tau_0pn,
)


PARSEC_M = lal.PC_SI
MAX_OBS_TIME = 3.0e7
SOLAR_DAY_S = 86400.0

GALACTIC_CENTER_RA = np.deg2rad(266.416833)
GALACTIC_CENTER_DEC = np.deg2rad(-29.007806)

LLO_LAT = np.deg2rad(30.562894333574896)
LLO_LNG = np.deg2rad(269.2257596112789)
LLO_AZ = np.deg2rad(72.28350084422942)

INJECTION_ETA = 0.0
INJECTION_PSI = 1.0

INJECTION_EPOCH_TM_UTC = (2023, 5, 24, 0, 0, 0, 0, 0, 0)
INJECTION_EPOCH_JD = lal.ConvertCivilTimeToJD(INJECTION_EPOCH_TM_UTC)

DEFAULTS = {
    "f0": 40.0,
    "mchirp": 1.0e-1,
    "f_min": 40.0,
    "f_max": 60.0,
    "chunk_duration": 30.0,
    "chunk_overlap": 0.0,
    "sample_rate": 512.0,
    "lambda_threshold": LAMBDA_THRESHOLD,
}


def default_arg_type(name, default):
    if name == "lambda_threshold":
        return float
    return type(default)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject a 3.5PN chirp at the semicoherent distance sensitivity and "
            "sum Hann-windowed NUFFT power along the track."
        )
    )
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=default_arg_type(name, default),
            default=default,
        )
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def validate_args(args):
    if args.f0 <= 0.0:
        raise ValueError("f0 must be positive")
    if args.mchirp <= 0.0:
        raise ValueError("mchirp must be positive")
    if args.f_min > args.f0:
        raise ValueError("f_min must not exceed f0")
    if args.f_max <= args.f0:
        raise ValueError("f_max must be larger than f0")
    if args.sample_rate <= 2.0 * args.f_max:
        raise ValueError("sample_rate must exceed twice f_max")
    if args.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be positive")
    if not 0.0 <= args.chunk_overlap < 1.0:
        raise ValueError("chunk_overlap must be in [0, 1)")
    if args.lambda_threshold <= 0.0:
        raise ValueError("lambda_threshold must be positive")


def chunk_count(duration, chunk_duration):
    return max(1, int(np.ceil(duration / chunk_duration)))


def semicoherent_frequency_model(args):
    return TaylorF2FrequencyModel(
        f0_hz=args.f0,
        f_stop_hz=args.f_max,
        mchirp_msun=args.mchirp,
    )


def observation_span(frequency_model):
    if frequency_model.t_end <= MAX_OBS_TIME:
        return frequency_model.t_end, frequency_model.f_stop_hz
    return MAX_OBS_TIME, float(frequency_model.frequency(MAX_OBS_TIME))


def integrated_chirp_power_35pn(f_start, f_end, frequency_model, noise):
    in_band = (
        (frequency_model.frequency_grid > f_start)
        & (frequency_model.frequency_grid < f_end)
    )
    frequencies = np.concatenate(
        ([f_start], frequency_model.frequency_grid[in_band], [f_end])
    )
    elapsed_times = np.interp(
        frequencies,
        frequency_model.frequency_grid,
        frequency_model.elapsed_time_grid,
    )
    power_integrand = frequencies ** (4.0 / 3.0) / noise.psd_at(frequencies)
    return f_start ** (-4.0 / 3.0) * float(
        np.trapezoid(power_integrand, elapsed_times)
    )


def h0_amplitude(distance_m, frequency, mchirp_msun):
    mchirp_kg = mchirp_msun * MSUN
    return (
        4.0
        / distance_m
        * (G * mchirp_kg / C**2) ** (5.0 / 3.0)
        * (np.pi * frequency / C) ** (2.0 / 3.0)
    )


def injection_gmst(t):
    jd = INJECTION_EPOCH_JD + np.asarray(t, dtype=float) / SOLAR_DAY_S
    template = five_vec()
    return template.gmst(jd)


def antenna_pattern_modulation(t):
    gmst = injection_gmst(t)
    template = five_vec(
        ra=GALACTIC_CENTER_RA,
        dec=GALACTIC_CENTER_DEC,
        eta=INJECTION_ETA,
        psi=INJECTION_PSI,
        lat=LLO_LAT,
        lng=LLO_LNG,
        az=LLO_AZ,
    )
    template.compute_H()
    template.compute_A(gmst)
    return template.amp_modulation, INJECTION_ETA, INJECTION_PSI


def make_injection(args, distance_m, duration):
    dt = 1.0 / args.sample_rate
    t = dt * np.arange(int(np.ceil(duration * args.sample_rate)))
    track = make_35pn_track(t, args.f0, args.f_max, args.mchirp)
    antenna_modulation, eta, psi = antenna_pattern_modulation(t)
    strain = (
        0.5
        * h0_amplitude(distance_m, track.frequency, args.mchirp)
        * antenna_modulation
        * track.signal
    )
    return t, strain, track.frequency, eta, psi


def chunk_starts(n_samples, chunk_samples, hop_samples):
    if n_samples < chunk_samples:
        return np.array([], dtype=int)
    return np.arange(0, n_samples - chunk_samples + 1, hop_samples)


def effective_nufft_psd(f_out, t_rel, beta, window, noise):
    tau_dot = (1.0 - 8.0 / 3.0 * beta * t_rel) ** (-3.0 / 8.0)
    frequencies = f_out * tau_dot
    weights = window**2 / np.sum(window**2)
    return float(np.sum(weights * noise.psd_at(frequencies)))


def track_power_for_chunk(signal, window, dt, f_start, beta, noise, resampler):
    t_rel = dt * np.arange(signal.size)
    tau = tau_0pn(t_rel, beta)

    resampler.timeseries = signal * window
    resampler.resampled_time = tau
    resampler.nufft()

    idx = int(np.argmin(np.abs(resampler.freq_in_hz - f_start)))
    carrier_power = resampler.power_normalized[idx] / np.mean(window**2)
    psd = effective_nufft_psd(f_start, t_rel, beta, window, noise)
    return float(4.0 * signal.size * dt * carrier_power / psd)


def effective_chi_squared_params(n_chunks, window, hop_samples, statistic_scale):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    window_norm = float(np.sum(window**2))
    variance_inflation = 1.0
    for lag in range(1, int(np.ceil(window.size / hop_samples))):
        shift = lag * hop_samples
        if shift >= window.size:
            break
        correlation = float(np.sum(window[:-shift] * window[shift:]) / window_norm)
        variance_inflation += 2.0 * (1.0 - lag / n_chunks) * correlation**2

    mean = 2.0 * statistic_scale * n_chunks
    variance = 4.0 * statistic_scale**2 * n_chunks * variance_inflation
    dof = 2.0 * mean**2 / variance
    scale = variance / (2.0 * mean)
    return mean, variance, dof, scale, variance_inflation


def null_significance(statistic, dof, scale):
    p_value = chi2.sf(statistic / scale, dof)
    sigma = norm.isf(p_value) if p_value > 0.0 else np.inf
    return p_value, sigma


def recover_track_power(t, strain, frequency_track, args, noise):
    chunk_samples = int(round(args.chunk_duration * args.sample_rate))
    hop_samples = int(round(chunk_samples * (1.0 - args.chunk_overlap)))
    if chunk_samples < 2 or hop_samples < 1:
        raise ValueError("invalid chunking configuration")

    dt = 1.0 / args.sample_rate
    window = np.hanning(chunk_samples)
    resampler = Resampler(nthreads=4, eps=1e-2)

    times = []
    frequencies = []
    powers = []
    for start in chunk_starts(strain.size, chunk_samples, hop_samples):
        stop = start + chunk_samples
        f_start = frequency_track[start]
        if not args.f_min <= f_start <= args.f_max:
            continue

        beta = beta_0pn(f_start, args.mchirp)
        power = track_power_for_chunk(
            strain[start:stop],
            window,
            dt,
            f_start,
            beta,
            noise,
            resampler,
        )
        times.append(t[start + chunk_samples // 2])
        frequencies.append(f_start)
        powers.append(power)

    return np.asarray(times), np.asarray(frequencies), np.asarray(powers)


def plot_chunks(times, frequencies, powers, output):
    fig, (ax_freq, ax_power) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True
    )

    ax_freq.plot(times, frequencies, marker="o", ms=2.8, lw=1.0)
    ax_freq.set_ylabel("Track frequency [Hz]")
    ax_freq.grid(True, alpha=0.25)

    ax_power.plot(times, powers, marker="o", ms=2.8, lw=1.0)
    ax_power.set_xlabel("Chunk center time [s]")
    ax_power.set_ylabel("Resampled ASD-weighted power")
    ax_power.grid(True, alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)

    noise = NoiseCurve.from_asd_file(args.asd)
    frequency_model = semicoherent_frequency_model(args)
    duration, f_end = observation_span(frequency_model)
    chirp_power = integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )
    distance_m = semicoherent_distance_sensitivity(
        args.f0,
        args.mchirp,
        chirp_power,
        duration,
        chunk_duration=args.chunk_duration,
        lambda_thresh=args.lambda_threshold,
    )

    t, strain, frequency_track, eta, psi = make_injection(
        args, distance_m, duration
    )
    times, frequencies, powers = recover_track_power(
        t, strain, frequency_track, args, noise
    )
    plot_chunks(times, frequencies, powers, args.output)

    overlap_scale = 1.0 - args.chunk_overlap
    recovered_power = overlap_scale * float(np.sum(powers))
    n_chunks = chunk_count(duration, args.chunk_duration)
    expected_sky_averaged_power = args.lambda_threshold * np.sqrt(n_chunks)
    response_weights = frequency_track ** (4.0 / 3.0) / noise.psd_at(frequency_track)
    expected_power = power_at_distance(
        distance_m,
        args.f0,
        args.mchirp,
        chirp_power,
        GALACTIC_CENTER_RA,
        GALACTIC_CENTER_DEC,
        eta,
        psi,
        injection_gmst(t),
        weights=response_weights,
    )
    window = np.hanning(int(round(args.chunk_duration * args.sample_rate)))
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)
    window_normalized_power = recovered_power / hann_power_factor
    null_mean, null_variance, dof, scale, variance_inflation = (
        effective_chi_squared_params(
            powers.size,
            window,
            int(round(window.size * overlap_scale)),
            overlap_scale,
        )
    )
    expected_statistic = null_mean + recovered_power
    p_value, sigma = null_significance(expected_statistic, dof, scale)

    print("-" * 9 + "Injection parameters" + "-" * 9)
    print("No noise injection")
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(
        "sky location: Galactic Center "
        f"(ra={np.rad2deg(GALACTIC_CENTER_RA):.6f} deg, "
        f"dec={np.rad2deg(GALACTIC_CENTER_DEC):.6f} deg)"
    )
    print(
        "detector: LIGO Livingston "
        f"(lat={np.rad2deg(LLO_LAT):.6f} deg, "
        f"lng={np.rad2deg(LLO_LNG):.6f} deg, "
        f"az={np.rad2deg(LLO_AZ):.6f} deg)"
    )
    print("t0: 2023-05-24 00:00:00 UTC")
    print(
        "injection polarization: "
        f"eta={eta:.6g}, psi={psi:.6g} rad"
    )
    print(f"signal duration: {duration:.2f} s")
    print(f"final semicoherent-path frequency: {f_end:.6g} Hz")
    print(f"distance sensitivity: {distance_m / PARSEC_M:.2e} pc")
    print(f"lambda threshold: {args.lambda_threshold:.12g}")
    print(
        "chunks in sensitivity formula: "
        f"{chunk_count(duration, args.chunk_duration)}"
    )
    print(f"analysis chunk duration: {args.chunk_duration:g} s")
    print(f"analysis chunk overlap: {args.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {powers.size}")
    print("-" * 9 + "Analysis results" + "-" * 9)
    print(f"overlap/window-normalized recovered power: {window_normalized_power:.2e}")
    print(f"expected power: {expected_power:.2e}")
    print(f"expected sky-averaged power: {expected_sky_averaged_power:.2e}")
    print(f"effective chi2 dof: {dof:.2e}")
    print(f"effective chi2 scale: {scale:.2e}")
    print(f"overlap variance inflation: {variance_inflation:.2e}")
    print(f"null mean statistic: {null_mean:.2e}")
    print(f"null std statistic: {np.sqrt(null_variance):.2e}")
    print(f"expected statistic if in white noise: {expected_statistic:.2e}")
    print(f"null p-value for expected statistic: {p_value:.2e}")
    print(f"Gaussian-equivalent significance: {sigma:.2e} sigma")


if __name__ == "__main__":
    main()
