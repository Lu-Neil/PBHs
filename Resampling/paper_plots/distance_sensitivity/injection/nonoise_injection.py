"""Inject a 3.5PN chirp at the semicoherent sensitivity and recover track power."""

import argparse
import sys
from pathlib import Path

import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2, norm


SCRIPT_DIR = Path(__file__).resolve().parent
FIGS_DIR = SCRIPT_DIR / "figs"
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = FIGS_DIR / "nonoise_injection.png"
Z_DISTANCE_OUTPUT_PATH = FIGS_DIR / "z_vs_distance_nonoise.png"

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
ANDROMEDA_RA = np.deg2rad(10.684708333333333)
ANDROMEDA_DEC = np.deg2rad(41.26875)

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

DEFAULT_N_DISTANCES = 20
DEFAULT_MIN_DISTANCE_RATIO = 0.1
DEFAULT_MAX_DISTANCE_RATIO = 100.0
DEFAULT_FALSE_ALARM_PROBABILITY = 1.0e-6
DEFAULT_DETECTION_PROBABILITY = 0.95


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
    parser.add_argument("--distance-output", type=Path, default=Z_DISTANCE_OUTPUT_PATH)
    parser.add_argument(
        "--false-alarm-probability",
        type=float,
        default=DEFAULT_FALSE_ALARM_PROBABILITY,
    )
    parser.add_argument(
        "--detection-probability",
        type=float,
        default=DEFAULT_DETECTION_PROBABILITY,
    )
    parser.add_argument("--n-distances", type=int, default=DEFAULT_N_DISTANCES)
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=DEFAULT_MIN_DISTANCE_RATIO,
        help="Smallest plotted distance as a multiple of the semicoherent distance.",
    )
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=DEFAULT_MAX_DISTANCE_RATIO,
        help="Largest plotted distance as a multiple of the semicoherent distance.",
    )
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


def validate_distance_scan_args(args):
    if not 0.0 < args.false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be in (0, 1)")
    if not 0.0 < args.detection_probability < 1.0:
        raise ValueError("detection_probability must be in (0, 1)")
    if args.n_distances < 2:
        raise ValueError("n_distances must be at least 2")
    if args.min_distance_ratio <= 0.0:
        raise ValueError("min_distance_ratio must be positive")
    if args.max_distance_ratio <= args.min_distance_ratio:
        raise ValueError("max_distance_ratio must exceed min_distance_ratio")


def chunk_count(duration, chunk_duration):
    return int(np.floor(duration / chunk_duration))


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


def analysis_span(args, frequency_model):
    duration, _ = observation_span(frequency_model)
    n_chunks = chunk_count(duration, args.chunk_duration)
    if n_chunks < 1:
        raise ValueError("observation span is shorter than one analysis chunk")

    duration = n_chunks * args.chunk_duration
    f_end = float(frequency_model.frequency(duration))
    return duration, f_end, n_chunks


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
        ra=ANDROMEDA_RA,
        dec=ANDROMEDA_DEC,
        eta=INJECTION_ETA,
        psi=INJECTION_PSI,
        lat=LLO_LAT,
        lng=LLO_LNG,
        az=LLO_AZ,
    )
    template.compute_H()
    template.compute_A(gmst)
    return template.amp_modulation, INJECTION_ETA, INJECTION_PSI


def sample_times(sample_rate, duration):
    dt = 1.0 / sample_rate
    return dt * np.arange(int(np.ceil(duration * sample_rate)))


def make_injection(args, distance_m, duration):
    t = sample_times(args.sample_rate, duration)
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


def chunk_config(args):
    chunk_samples = int(round(args.chunk_duration * args.sample_rate))
    hop_samples = int(round(chunk_samples * (1.0 - args.chunk_overlap)))
    if chunk_samples < 2 or hop_samples < 1:
        raise ValueError("invalid chunking configuration")
    return chunk_samples, hop_samples


def count_summed_chunks(frequency_track, args):
    chunk_samples, hop_samples = chunk_config(args)
    starts = chunk_starts(frequency_track.size, chunk_samples, hop_samples)
    frequencies = frequency_track[starts]
    in_band = (args.f_min <= frequencies) & (frequencies <= args.f_max)
    return int(np.count_nonzero(in_band))


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


def required_noncentrality(false_alarm_probability, detection_probability, dof):
    statistic_threshold = chi2.isf(false_alarm_probability, dof)

    def detection_probability_error(noncentrality):
        return (
            ncx2.sf(statistic_threshold, dof, noncentrality)
            - detection_probability
        )

    high = max(1.0, statistic_threshold)
    while detection_probability_error(high) < 0.0:
        high *= 2.0

    return float(brentq(detection_probability_error, 0.0, high))


def detection_thresholds(false_alarm_probability, detection_probability, dof, scale):
    false_alarm_statistic = float(scale * chi2.isf(false_alarm_probability, dof))
    noncentrality = required_noncentrality(
        false_alarm_probability,
        detection_probability,
        dof,
    )
    return {
        "false_alarm_statistic": false_alarm_statistic,
        "false_alarm_gaussian_z": float(norm.isf(false_alarm_probability)),
        "required_noncentrality": noncentrality,
        "required_signal_statistic": float(scale * noncentrality),
    }


def recover_track_power(t, strain, frequency_track, args, noise):
    chunk_samples, hop_samples = chunk_config(args)
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


def distance_scan(distance_m, reference_signal_statistic, sigma0, args):
    ratios = np.linspace(
        args.min_distance_ratio,
        args.max_distance_ratio,
        args.n_distances,
    )
    z = reference_signal_statistic / sigma0 / ratios**2
    return distance_m * ratios, z


def plot_z_vs_distance(result, output):
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    distance_pc = result["distances_m"] / PARSEC_M

    ax.plot(
        distance_pc,
        result["expected_z"],
        marker="o",
        ms=3.5,
        lw=1.5,
        label="Expected signal",
    )
    ax.plot(
        distance_pc,
        result["recovered_z"],
        marker="s",
        ms=3.5,
        lw=1.2,
        ls="--",
        label="Recovered no-noise scaling",
    )
    ax.axvline(
        result["distance_m"] / PARSEC_M,
        color="k",
        ls=":",
        lw=1.0,
        label="Semicoherent distance",
    )
    ax.axhline(
        result["threshold_z"],
        color="0.35",
        ls=":",
        lw=1.0,
        label="95% detection threshold",
    )
    ax.set_xlabel("Injection distance [pc]")
    ax.set_ylabel(r"$z = (\Lambda - \mu_0) / \sigma_0$")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def expected_power_for_track(
    args,
    distance_m,
    chirp_power,
    t,
    frequency_track,
    eta,
    psi,
    noise,
):
    response_weights = frequency_track ** (4.0 / 3.0) / noise.psd_at(frequency_track)
    return power_at_distance(
        distance_m,
        args.f0,
        args.mchirp,
        chirp_power,
        ANDROMEDA_RA,
        ANDROMEDA_DEC,
        eta,
        psi,
        injection_gmst(t),
        weights=response_weights,
    )


def main():
    args = parse_args()
    validate_args(args)
    validate_distance_scan_args(args)

    noise = NoiseCurve.from_asd_file(args.asd)
    frequency_model = semicoherent_frequency_model(args)
    duration, f_end, n_chunks = analysis_span(args, frequency_model)
    chirp_power = integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )
    reference_distance_m = semicoherent_distance_sensitivity(
        args.f0,
        args.mchirp,
        chirp_power,
        duration,
        chunk_duration=args.chunk_duration,
        false_alarm_probability=args.false_alarm_probability,
        detection_probability=args.detection_probability,
        mismatch_bank=0.0,
        mismatch_coh=0.0,
    )

    overlap_scale = 1.0 - args.chunk_overlap
    chunk_samples, hop_samples = chunk_config(args)
    window = np.hanning(chunk_samples)
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)

    t = sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t), dtype=float)
    eta = INJECTION_ETA
    psi = INJECTION_PSI
    n_summed_chunks = count_summed_chunks(frequency_track, args)
    null_mean, null_variance, dof, scale, variance_inflation = (
        effective_chi_squared_params(
            n_summed_chunks,
            window,
            hop_samples,
            overlap_scale,
        )
    )
    sigma0 = float(np.sqrt(null_variance))
    thresholds = detection_thresholds(
        args.false_alarm_probability,
        args.detection_probability,
        dof,
        scale,
    )
    required_signal_statistic = thresholds["required_signal_statistic"]
    threshold_z = required_signal_statistic / sigma0
    false_alarm_z = (
        thresholds["false_alarm_statistic"] - null_mean
    ) / sigma0
    reference_expected_power = expected_power_for_track(
        args,
        reference_distance_m,
        chirp_power,
        t,
        frequency_track,
        eta,
        psi,
        noise,
    )
    reference_expected_signal_statistic = (
        hann_power_factor * reference_expected_power
    )
    if reference_expected_signal_statistic <= 0.0:
        raise ValueError("expected signal statistic must be positive")
    distance_m = reference_distance_m * np.sqrt(
        reference_expected_signal_statistic / required_signal_statistic
    )

    t, strain, frequency_track, eta, psi = make_injection(
        args, distance_m, duration
    )
    times, frequencies, powers = recover_track_power(
        t, strain, frequency_track, args, noise
    )
    plot_chunks(times, frequencies, powers, args.output)

    recovered_power = overlap_scale * float(np.sum(powers))
    threshold_unwindowed_power = required_signal_statistic / hann_power_factor
    expected_power = expected_power_for_track(
        args,
        distance_m,
        chirp_power,
        t,
        frequency_track,
        eta,
        psi,
        noise,
    )
    window_normalized_power = recovered_power / hann_power_factor
    expected_signal_statistic = hann_power_factor * expected_power
    expected_statistic = null_mean + expected_signal_statistic
    recovered_z_at_reference = recovered_power / sigma0
    expected_z_at_reference = expected_signal_statistic / sigma0
    distances_m, recovered_z = distance_scan(
        distance_m,
        recovered_power,
        sigma0,
        args,
    )
    _, expected_z = distance_scan(
        distance_m,
        expected_signal_statistic,
        sigma0,
        args,
    )
    plot_z_vs_distance(
        {
            "distance_m": distance_m,
            "distances_m": distances_m,
            "recovered_z": recovered_z,
            "expected_z": expected_z,
            "threshold_z": threshold_z,
        },
        args.distance_output,
    )
    p_value, sigma = null_significance(expected_statistic, dof, scale)
    expected_detection_probability = ncx2.sf(
        thresholds["false_alarm_statistic"] / scale,
        dof,
        expected_signal_statistic / scale,
    )

    print("-" * 9 + "Injection parameters" + "-" * 9)
    print("No noise injection")
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(
        "sky location: Andromeda (M31) "
        f"(ra={np.rad2deg(ANDROMEDA_RA):.6f} deg, "
        f"dec={np.rad2deg(ANDROMEDA_DEC):.6f} deg)"
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
    print(
        "zero-mismatch semicoherent distance estimate: "
        f"{reference_distance_m / PARSEC_M:.2e} pc"
    )
    print(f"distance sensitivity: {distance_m / PARSEC_M:.2e} pc")
    print(f"legacy single-bin lambda threshold: {args.lambda_threshold:.12g}")
    print(f"false alarm probability: {args.false_alarm_probability:.2e}")
    print(f"detection probability target: {args.detection_probability:.3g}")
    print(
        "chunks in sensitivity formula: "
        f"{n_chunks}"
    )
    print(f"analysis chunk duration: {args.chunk_duration:g} s")
    print(f"analysis chunk overlap: {args.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {powers.size}")
    print("-" * 9 + "Analysis results" + "-" * 9)
    print(f"overlap/window-normalized recovered power: {window_normalized_power:.2e}")
    print(f"expected power: {expected_power:.2e}")
    print(f"unwindowed power required for threshold z: {threshold_unwindowed_power:.2e}")
    print(f"effective chi2 dof: {dof:.2e}")
    print(f"effective chi2 scale: {scale:.2e}")
    print(f"overlap variance inflation: {variance_inflation:.2e}")
    print(f"null mean statistic: {null_mean:.2e}")
    print(f"null std statistic: {sigma0:.2e}")
    print(
        "false-alarm statistic threshold: "
        f"{thresholds['false_alarm_statistic']:.2e}"
    )
    print(f"false-alarm Gaussian z: {thresholds['false_alarm_gaussian_z']:.2e}")
    print(f"false-alarm threshold in script z: {false_alarm_z:.2e}")
    print(
        "required noncentrality for detection probability: "
        f"{thresholds['required_noncentrality']:.2e}"
    )
    print(f"required signal statistic: {required_signal_statistic:.2e}")
    print(f"expected statistic if in white noise: {expected_statistic:.2e}")
    print(f"required signal z: {threshold_z:.2e}")
    print(f"recovered z at distance sensitivity: {recovered_z_at_reference:.2e}")
    print(f"expected z at distance sensitivity: {expected_z_at_reference:.2e}")
    print(f"detection probability at expected statistic: {expected_detection_probability:.3g}")
    print(f"null p-value for expected statistic: {p_value:.2e}")
    print(f"Gaussian-equivalent significance: {sigma:.2e} sigma")
    print(f"z(d) plot: {args.distance_output}")


if __name__ == "__main__":
    main()
