"""Inject a 3.5PN chirp into one-detector O4a strain data."""

import argparse
import sys
from pathlib import Path

import lal
from matplotlib.axes import Axes
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
PAPER_STYLE_PATH = PAPER_PLOTS_DIR / "paper.mplstyle"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402
from distance_sensitivity.maximum_sensitivity_35PN import (  # noqa: E402
    FOURIER_BIN_POWER_AVERAGE,
)


base.plt.style.use(PAPER_STYLE_PATH)

OUTPUT_PATH = base.FIGS_DIR / "semicoherent_track_injection_detector_noise.png"
Z_DISTANCE_OUTPUT_PATH = base.FIGS_DIR / "z_vs_distance_detector_noise.png"

DETECTOR = "H1"
DATA_QUALITY_FLAG = "H1_DATA"
O4A_START = "2023-05-24 15:00:00 UTC"
O4A_END = "2024-01-16 16:00:00 UTC"
NULL_TRIALS = 12
INJECTION_WINDOW_INDEX = 0
ASD_FFTLENGTH = 8.0
ASD_OVERLAP = 4.0
RESPONSE_SAMPLE_SPACING = 1.0

DETECTOR_INDEX = {
    "H1": lal.LALDetectorIndexLHODIFF,
    "L1": lal.LALDetectorIndexLLODIFF,
    "V1": lal.LALDetectorIndexVIRGODIFF,
    "K1": lal.LALDetectorIndexKAGRADIFF,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find a continuous O4a open-data section, estimate its ASD and null "
            "statistic numerically, inject a 3.5PN chirp into one detector, and "
            "plot expected versus recovered signal excess."
        )
    )
    for name, default in base.DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=base.default_arg_type(name, default),
            default=default,
        )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--distance-output", type=Path, default=Z_DISTANCE_OUTPUT_PATH)
    parser.add_argument(
        "--false-alarm-probability",
        type=float,
        default=base.DEFAULT_FALSE_ALARM_PROBABILITY,
    )
    parser.add_argument(
        "--detection-probability",
        type=float,
        default=base.DEFAULT_DETECTION_PROBABILITY,
    )
    parser.add_argument("--n-distances", type=int, default=base.DEFAULT_N_DISTANCES)
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=base.DEFAULT_MIN_DISTANCE_RATIO,
    )
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=base.DEFAULT_MAX_DISTANCE_RATIO,
    )
    return parser.parse_args()


def import_gwpy():
    try:
        from gwpy.segments import DataQualityFlag
        from gwpy.time import to_gps
        from gwpy.timeseries import TimeSeries
    except ImportError as exc:
        raise ImportError(
            "detector_noise_injection.py requires GWpy. Install it in the PBH "
            "environment before running this script."
        ) from exc
    return DataQualityFlag, TimeSeries, to_gps


def segment_bounds(segment):
    try:
        start, end = segment
        return float(start), float(end)
    except TypeError:
        return float(segment.start), float(segment.end)


def find_continuous_segment(duration, DataQualityFlag, to_gps):
    query_start = float(to_gps(O4A_START))
    query_end = float(to_gps(O4A_END))
    flag = DataQualityFlag.fetch_open_data(DATA_QUALITY_FLAG, query_start, query_end)

    for segment in flag.active:
        start, end = segment_bounds(segment)
        if end - start >= duration:
            return start, start + duration

    raise RuntimeError(
        f"No {DATA_QUALITY_FLAG} active segment within O4a is at least {duration:.1f} s."
    )


def quantity_to_float(value):
    try:
        return float(value.to_value("Hz"))
    except AttributeError:
        pass
    try:
        return float(value.value)
    except AttributeError:
        return float(value)


def fetch_open_strain(args, start, end, TimeSeries):
    data = TimeSeries.fetch_open_data(DETECTOR, start, end, cache=True)

    actual_rate = quantity_to_float(data.sample_rate)
    if not np.isclose(actual_rate, args.sample_rate):
        data = data.resample(args.sample_rate)
    return data.detrend("constant")


def noise_curve_from_detector_data(data, args):
    asd = data.asd(
        fftlength=ASD_FFTLENGTH,
        overlap=ASD_OVERLAP,
        method="welch",
        window="hann",
    )
    frequency = np.asarray(asd.frequencies.value, dtype=float)
    asd_values = np.asarray(asd.value, dtype=float)
    valid = (
        np.isfinite(frequency)
        & np.isfinite(asd_values)
        & (frequency > 0.0)
        & (asd_values > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("GWpy Welch ASD did not produce a usable frequency band.")
    return base.NoiseCurve(frequency[valid], asd_values[valid] ** 2)


def detector_response(detector, gps_start, t, eta, psi, spacing):
    detector_response_tensor = lal.CachedDetectors[DETECTOR_INDEX[detector]].response
    response_times = np.arange(0.0, float(t[-1]) + spacing, spacing)
    if response_times[-1] < t[-1]:
        response_times = np.append(response_times, float(t[-1]))

    response_grid = np.empty(response_times.size, dtype=complex)
    for idx, offset in enumerate(response_times):
        gps = lal.LIGOTimeGPS(float(gps_start + offset))
        gmst = lal.GreenwichMeanSiderealTime(gps)
        f_plus, f_cross = lal.ComputeDetAMResponse(
            detector_response_tensor,
            base.GALACTIC_CENTER_RA,
            base.GALACTIC_CENTER_DEC,
            psi,
            gmst,
        )
        response_grid[idx] = (f_plus + 1j * eta * f_cross) / np.sqrt(1.0 + eta**2)

    return np.interp(t, response_times, response_grid.real) + 1j * np.interp(
        t,
        response_times,
        response_grid.imag,
    )


def make_detector_injection(args, distance_m, duration, gps_start):
    t = base.sample_times(args.sample_rate, duration)
    track = base.make_35pn_track(t, args.f0, args.f_max, args.mchirp)
    eta = base.INJECTION_ETA
    psi = base.INJECTION_PSI
    response = detector_response(
        DETECTOR,
        gps_start,
        t,
        eta,
        psi,
        RESPONSE_SAMPLE_SPACING,
    )
    strain = (
        0.5
        * base.h0_amplitude(distance_m, track.frequency, args.mchirp)
        * response
        * track.signal
    )
    return t, strain, track.frequency, response, eta, psi


def expected_power_for_detector_track(
    args,
    distance_m,
    chirp_power,
    frequency_track,
    response,
    noise,
):
    response_weights = frequency_track ** (4.0 / 3.0) / noise.psd_at(frequency_track)
    response_power = FOURIER_BIN_POWER_AVERAGE * np.average(
        np.abs(response) ** 2,
        weights=response_weights,
    )
    beta = base.beta_0pn(args.f0, args.mchirp)
    prefactor_squared = response_power * 16.0 / np.pi**4 * (5.0 / 96.0) ** 2
    return (
        prefactor_squared
        * (base.C * beta / args.f0**2) ** 2
        * chirp_power
        / distance_m**2
    )


def recover_statistic(strain, frequency_track, args, noise):
    t = base.sample_times(args.sample_rate, strain.size / args.sample_rate)
    times, frequencies, powers = base.recover_track_power(
        t,
        strain,
        frequency_track,
        args,
        noise,
    )
    statistic = (1.0 - args.chunk_overlap) * float(np.sum(powers))
    return times, frequencies, powers, statistic


def null_statistics(detector_strain, frequency_track, analysis_samples, args, noise):
    statistics = []
    for idx in range(NULL_TRIALS + 1):
        if idx == INJECTION_WINDOW_INDEX:
            continue
        start = idx * analysis_samples
        stop = start + analysis_samples
        _, _, _, statistic = recover_statistic(
            detector_strain[start:stop],
            frequency_track,
            args,
            noise,
        )
        statistics.append(statistic)

    statistics = np.asarray(statistics, dtype=float)
    if statistics.size < 2:
        raise ValueError("Need at least two null statistics.")
    return statistics, float(np.mean(statistics)), float(np.std(statistics, ddof=1))


def scaled_chi2_from_moments(mean, std):
    variance = std**2
    dof = 2.0 * mean**2 / variance
    scale = variance / (2.0 * mean)
    return dof, scale


def distance_scan(
    distance_m,
    signal,
    detector_noise,
    frequency_track,
    expected_signal_statistic,
    mu0,
    sigma0,
    args,
    noise,
):
    ratios = np.logspace(
        np.log10(args.min_distance_ratio),
        np.log10(args.max_distance_ratio),
        args.n_distances,
    )
    distances_m = distance_m * ratios
    expected_z = expected_signal_statistic / sigma0 / ratios**2
    recovered_z = np.empty_like(expected_z)

    for idx, ratio in enumerate(ratios):
        strain = signal / ratio + detector_noise
        _, _, _, statistic = recover_statistic(strain, frequency_track, args, noise)
        recovered_z[idx] = (statistic - mu0) / sigma0

    return distances_m, expected_z, recovered_z


def plot_chunks(times, frequencies, powers, output):
    fig = base.plt.figure(figsize=(7.2, 5.6), constrained_layout=True)
    ax_freq = fig.add_subplot(2, 1, 1, axes_class=Axes)
    ax_power = fig.add_subplot(2, 1, 2, sharex=ax_freq, axes_class=Axes)

    ax_freq.plot(times, frequencies, marker="o", ms=2.8, lw=1.0)
    ax_freq.set_ylabel("Track frequency [Hz]")
    ax_freq.grid(True, alpha=0.25)

    ax_power.plot(times, powers, marker="o", ms=2.8, lw=1.0)
    ax_power.set_xlabel("Chunk center time [s]")
    ax_power.set_ylabel("Resampled ASD-weighted power")
    ax_power.grid(True, alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    base.plt.close(fig)


def plot_z_vs_distance(result, output):
    fig = base.plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, axes_class=Axes)
    distance_pc = result["distances_m"] / base.PARSEC_M

    ax.plot(
        distance_pc,
        result["expected_z"],
        ms=3.5,
        lw=1.5,
        ls="-",
        label="Expected signal",
    )
    ax.plot(
        distance_pc,
        result["recovered_z"],
        marker="x",
        ms=5,
        ls="",
        label="Recovered detector-noise excess",
    )
    ax.axvline(
        result["distance_m"] / base.PARSEC_M,
        color="k",
        ls=":",
        lw=1.0,
        label="Semicoherent distance threshold",
    )
    ax.axhline(
        result["threshold_z"],
        color="0.35",
        ls=":",
        lw=1.0,
        label="95% detection threshold",
    )
    ax.set_xlabel("Injection distance [pc]")
    ax.set_ylabel(r"$n_\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("symlog")
    ax.set_ylim(-1)
    ax.grid(True, alpha=0.25)
    ax.legend(handlelength=2)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    base.plt.close(fig)


def main():
    args = parse_args()
    base.validate_args(args)
    base.validate_distance_scan_args(args)
    DataQualityFlag, TimeSeries, to_gps = import_gwpy()

    frequency_model = base.semicoherent_frequency_model(args)
    duration, f_end, n_chunks = base.analysis_span(args, frequency_model)
    analysis_samples = int(np.ceil(duration * args.sample_rate))
    section_duration = duration * (NULL_TRIALS + 1)
    section_start, section_end = find_continuous_segment(
        section_duration,
        DataQualityFlag,
        to_gps,
    )
    data = fetch_open_strain(args, section_start, section_end, TimeSeries)
    detector_strain = np.asarray(data.value, dtype=float)
    required_samples = analysis_samples * (NULL_TRIALS + 1)
    if detector_strain.size < required_samples:
        raise RuntimeError(
            f"Fetched {detector_strain.size} samples, need {required_samples}."
        )
    detector_strain = detector_strain[:required_samples]

    noise = noise_curve_from_detector_data(data, args)
    chirp_power = base.integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )

    t = base.sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t), dtype=float)
    null_stats, mu0, sigma0 = null_statistics(
        detector_strain,
        frequency_track,
        analysis_samples,
        args,
        noise,
    )
    if sigma0 <= 0.0:
        raise ValueError("numerical sigma0 must be positive")
    dof, scale = scaled_chi2_from_moments(mu0, sigma0)
    thresholds = base.detection_thresholds(
        args.false_alarm_probability,
        args.detection_probability,
        dof,
        scale,
    )
    required_signal_statistic = thresholds["required_signal_statistic"]
    threshold_z = required_signal_statistic / sigma0
    false_alarm_z = (
        thresholds["false_alarm_statistic"] - mu0
    ) / sigma0

    injection_start = INJECTION_WINDOW_INDEX * analysis_samples
    injection_stop = injection_start + analysis_samples
    injection_gps_start = section_start + INJECTION_WINDOW_INDEX * duration
    detector_noise = detector_strain[injection_start:injection_stop]

    reference_distance_m = base.semicoherent_distance_sensitivity(
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
    _, reference_signal, _, response, eta, psi = make_detector_injection(
        args,
        reference_distance_m,
        duration,
        injection_gps_start,
    )
    reference_expected_power = expected_power_for_detector_track(
        args,
        reference_distance_m,
        chirp_power,
        frequency_track,
        response,
        noise,
    )
    window = np.hanning(int(round(args.chunk_duration * args.sample_rate)))
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)
    reference_expected_signal_statistic = (
        hann_power_factor * reference_expected_power
    )
    if reference_expected_signal_statistic <= 0.0:
        raise ValueError("expected signal statistic must be positive")
    distance_m = reference_distance_m * np.sqrt(
        reference_expected_signal_statistic / required_signal_statistic
    )

    _, signal, _, response, _, _ = make_detector_injection(
        args,
        distance_m,
        duration,
        injection_gps_start,
    )
    strain = detector_noise + signal
    times, frequencies, powers, recovered_statistic = recover_statistic(
        strain,
        frequency_track,
        args,
        noise,
    )
    plot_chunks(times, frequencies, powers, args.output)

    expected_power = expected_power_for_detector_track(
        args,
        distance_m,
        chirp_power,
        frequency_track,
        response,
        noise,
    )
    expected_signal_statistic = hann_power_factor * expected_power
    expected_statistic = mu0 + expected_signal_statistic
    recovered_excess = recovered_statistic - mu0
    recovered_z_at_reference = recovered_excess / sigma0
    expected_z_at_reference = expected_signal_statistic / sigma0

    distances_m, expected_z, recovered_z = distance_scan(
        distance_m,
        signal,
        detector_noise,
        frequency_track,
        expected_signal_statistic,
        mu0,
        sigma0,
        args,
        noise,
    )
    plot_z_vs_distance(
        {
            "distance_m": distance_m,
            "distances_m": distances_m,
            "expected_z": expected_z,
            "recovered_z": recovered_z,
            "threshold_z": threshold_z,
        },
        args.distance_output,
    )

    expected_p_value, expected_sigma = base.null_significance(
        expected_statistic,
        dof,
        scale,
    )
    recovered_p_value, recovered_sigma = base.null_significance(
        recovered_statistic,
        dof,
        scale,
    )
    expected_detection_probability = base.ncx2.sf(
        thresholds["false_alarm_statistic"] / scale,
        dof,
        expected_signal_statistic / scale,
    )

    print("-" * 9 + "Detector data configuration" + "-" * 9)
    print(f"detector: {DETECTOR}")
    print(f"O4a search range: {O4A_START} to {O4A_END}")
    print(f"data-quality flag: {DATA_QUALITY_FLAG}")
    print(f"continuous section GPS: {section_start:.0f} - {section_end:.0f}")
    print(f"injection GPS start: {injection_gps_start:.0f}")
    print(f"analysis duration per window: {duration:.2f} s")
    print(f"null windows: {NULL_TRIALS}")
    print(f"injection window index: {INJECTION_WINDOW_INDEX}")
    print(f"sample rate: {args.sample_rate:g} Hz")
    print(
        "ASD estimate: GWpy TimeSeries.asd("
        f"method='welch', fftlength={ASD_FFTLENGTH:g}, "
        f"overlap={ASD_OVERLAP:g}, window='hann')"
    )
    print("-" * 9 + "Injection parameters" + "-" * 9)
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(
        "sky location: Galactic Center "
        f"(ra={np.rad2deg(base.GALACTIC_CENTER_RA):.6f} deg, "
        f"dec={np.rad2deg(base.GALACTIC_CENTER_DEC):.6f} deg)"
    )
    print(f"injection polarization: eta={eta:.6g}, psi={psi:.6g} rad")
    print(f"final semicoherent-path frequency: {f_end:.6g} Hz")
    print(
        "zero-mismatch semicoherent distance estimate: "
        f"{reference_distance_m / base.PARSEC_M:.2e} pc"
    )
    print(f"distance sensitivity: {distance_m / base.PARSEC_M:.2e} pc")
    print(f"legacy single-bin lambda threshold: {args.lambda_threshold:.12g}")
    print(f"false alarm probability: {args.false_alarm_probability:.2e}")
    print(f"detection probability target: {args.detection_probability:.3g}")
    print(f"chunks in sensitivity formula: {n_chunks}")
    print(f"analysis chunk duration: {args.chunk_duration:g} s")
    print(f"analysis chunk overlap: {args.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {powers.size}")
    print("-" * 9 + "Numerical null estimate" + "-" * 9)
    print(
        "null statistics min/median/max: "
        f"{null_stats.min():.2e} / {np.median(null_stats):.2e} / "
        f"{null_stats.max():.2e}"
    )
    print(f"numerical mu0: {mu0:.2e}")
    print(f"numerical sigma0: {sigma0:.2e}")
    print(f"moment-matched chi2 dof: {dof:.2e}")
    print(f"moment-matched chi2 scale: {scale:.2e}")
    print("-" * 9 + "Analysis results" + "-" * 9)
    print(f"realized signal+detector-noise statistic: {recovered_statistic:.2e}")
    print(f"expected signal+noise statistic: {expected_statistic:.2e}")
    print(f"expected signal power: {expected_power:.2e}")
    print(
        "unwindowed power required for threshold z: "
        f"{required_signal_statistic / hann_power_factor:.2e}"
    )
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
    print(f"required signal z: {threshold_z:.2e}")
    print(f"recovered z at distance sensitivity: {recovered_z_at_reference:.2e}")
    print(f"expected z at distance sensitivity: {expected_z_at_reference:.2e}")
    print(
        "detection probability at expected statistic: "
        f"{expected_detection_probability:.3g}"
    )
    print(f"null p-value for realized statistic: {recovered_p_value:.2e}")
    print(f"null p-value for expected statistic: {expected_p_value:.2e}")
    print(f"realized Gaussian-equivalent significance: {recovered_sigma:.2e} sigma")
    print(f"expected Gaussian-equivalent significance: {expected_sigma:.2e} sigma")
    print(f"chunk plot: {args.output}")
    print(f"z(d) plot: {args.distance_output}")


if __name__ == "__main__":
    main()
