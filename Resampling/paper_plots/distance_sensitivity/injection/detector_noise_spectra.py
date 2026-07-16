"""Plot local-beta NUFFT and direct-FFT spectra for a detector-noise injection."""

import argparse
import sys
from pathlib import Path

from matplotlib.axes import Axes
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
PAPER_STYLE_PATH = PAPER_PLOTS_DIR / "paper.mplstyle"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import detector_noise_injection as detector  # noqa: E402
from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


base.plt.style.use(PAPER_STYLE_PATH)

OUTPUT_PATH = SCRIPT_DIR / "detector_noise_spectra.png"
DEFAULT_FREQUENCY_BINS = 700
DEFAULT_COLOR_PERCENTILE_MIN = 5.0
DEFAULT_COLOR_PERCENTILE_MAX = 99.9
PSD_BLOCK_SIZE = 128


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject a 3.5PN chirp at the 95% detector-noise distance "
            "threshold and compare local-beta NUFFT spectra with direct FFT "
            "spectra."
        )
    )
    for name, default in base.DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=base.default_arg_type(name, default),
            default=default,
        )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
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
    parser.add_argument(
        "--frequency-bins",
        type=int,
        default=DEFAULT_FREQUENCY_BINS,
        help="Number of common frequency bins used for the plotted spectra.",
    )
    parser.add_argument(
        "--color-percentile-min",
        type=float,
        default=DEFAULT_COLOR_PERCENTILE_MIN,
        help="Lower percentile for the spectrogram color scale.",
    )
    parser.add_argument(
        "--color-percentile-max",
        type=float,
        default=DEFAULT_COLOR_PERCENTILE_MAX,
        help="Upper percentile for the spectrogram color scale.",
    )
    return parser.parse_args()


def validate_args(args):
    base.validate_args(args)
    if not 0.0 < args.false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be in (0, 1)")
    if not 0.0 < args.detection_probability < 1.0:
        raise ValueError("detection_probability must be in (0, 1)")
    if args.frequency_bins < 2:
        raise ValueError("frequency_bins must be at least 2")
    if not 0.0 <= args.color_percentile_min < args.color_percentile_max <= 100.0:
        raise ValueError("color percentiles must satisfy 0 <= min < max <= 100")


def cell_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size == 1:
        return np.array([centers[0] - 0.5, centers[0] + 0.5], dtype=float)

    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - (edges[1] - centers[0])
    edges[-1] = centers[-1] + (centers[-1] - edges[-2])
    return edges


def effective_nufft_psd_for_frequencies(frequencies, t_rel, beta, window, noise):
    frequencies = np.asarray(frequencies, dtype=float)
    psd = np.empty_like(frequencies)
    weights = window**2 / np.sum(window**2)
    tau_dot = (1.0 - 8.0 / 3.0 * beta * t_rel) ** (-3.0 / 8.0)

    for start in range(0, frequencies.size, PSD_BLOCK_SIZE):
        stop = min(start + PSD_BLOCK_SIZE, frequencies.size)
        tracked_frequencies = frequencies[start:stop, None] * tau_dot[None, :]
        psd[start:stop] = noise.psd_at(tracked_frequencies) @ weights

    return psd


def local_beta_spectrogram(strain, frequency_track, args, noise):
    chunk_samples, hop_samples = base.chunk_config(args)
    starts = base.chunk_starts(strain.size, chunk_samples, hop_samples)
    dt = 1.0 / args.sample_rate
    t_rel = dt * np.arange(chunk_samples)
    window = np.hanning(chunk_samples)
    frequency_grid = np.linspace(args.f_min, args.f_max, args.frequency_bins)
    resampler = base.Resampler(nthreads=4, eps=1e-2)

    times = []
    carrier_frequencies = []
    columns = []

    for start in starts:
        stop = start + chunk_samples
        f_start = frequency_track[start]
        if not args.f_min <= f_start <= args.f_max:
            continue

        beta = base.beta_0pn(f_start, args.mchirp)
        tau = base.tau_0pn(t_rel, beta)
        resampler.timeseries = strain[start:stop] * window
        resampler.resampled_time = tau
        resampler.nufft()

        freq_hz = resampler.freq_in_hz
        in_band = (
            np.isfinite(freq_hz)
            & (freq_hz >= args.f_min)
            & (freq_hz <= args.f_max)
        )
        if np.count_nonzero(in_band) < 2:
            continue

        band_frequencies = freq_hz[in_band]
        band_power = resampler.power_normalized[in_band] / np.mean(window**2)
        band_psd = effective_nufft_psd_for_frequencies(
            band_frequencies,
            t_rel,
            beta,
            window,
            noise,
        )
        normalized_power = 2.0 * chunk_samples * dt * band_power / band_psd
        snr = np.sqrt(np.maximum(normalized_power, 0.0))

        columns.append(
            np.interp(
                frequency_grid,
                band_frequencies,
                snr,
                left=np.nan,
                right=np.nan,
            )
        )
        times.append(dt * (start + chunk_samples // 2))
        carrier_frequencies.append(f_start)

    if not columns:
        raise ValueError("No in-band chunks were available for the spectrogram.")

    return (
        np.asarray(times),
        frequency_grid,
        np.asarray(columns, dtype=float).T,
        np.asarray(carrier_frequencies, dtype=float),
    )


def direct_fft_spectrogram(strain, frequency_track, args, noise):
    chunk_samples, hop_samples = base.chunk_config(args)
    starts = base.chunk_starts(strain.size, chunk_samples, hop_samples)
    dt = 1.0 / args.sample_rate
    window = np.hanning(chunk_samples)
    window_power = np.mean(window**2)
    frequency_grid = np.linspace(args.f_min, args.f_max, args.frequency_bins)
    fft_frequencies = np.fft.fftfreq(chunk_samples, dt)

    times = []
    track_frequencies = []
    columns = []

    for start in starts:
        stop = start + chunk_samples
        f_start = frequency_track[start]
        if not args.f_min <= f_start <= args.f_max:
            continue

        spectrum = np.fft.fft(strain[start:stop] * window)
        power = np.abs(spectrum / chunk_samples) ** 2 / window_power
        in_band = (fft_frequencies >= args.f_min) & (fft_frequencies <= args.f_max)
        band_frequencies = fft_frequencies[in_band]
        band_power = power[in_band]
        normalized_power = (
            2.0
            * chunk_samples
            * dt
            * band_power
            / noise.psd_at(band_frequencies)
        )
        snr = np.sqrt(np.maximum(normalized_power, 0.0))

        columns.append(
            np.interp(
                frequency_grid,
                band_frequencies,
                snr,
                left=np.nan,
                right=np.nan,
            )
        )
        center = min(start + chunk_samples // 2, frequency_track.size - 1)
        times.append(dt * center)
        track_frequencies.append(frequency_track[center])

    if not columns:
        raise ValueError("No in-band chunks were available for the FFT spectrogram.")

    return (
        np.asarray(times),
        frequency_grid,
        np.asarray(columns, dtype=float).T,
        np.asarray(track_frequencies, dtype=float),
    )


def spectrogram_values(*spectra):
    values = []
    for spectrum in spectra:
        values.append(np.log10(np.maximum(spectrum, np.finfo(float).tiny)))
    return values


def plot_spectrograms(
    times,
    frequency_grid,
    nufft_spectra,
    nufft_track,
    fft_spectra,
    fft_track,
    result,
    output,
):
    nufft_values, fft_values = spectrogram_values(nufft_spectra, fft_spectra)
    combined = np.concatenate((nufft_values.ravel(), fft_values.ravel()))
    finite = np.isfinite(combined)
    if not np.any(finite):
        raise ValueError("Spectrogram has no finite values to plot.")

    vmin, vmax = np.nanpercentile(
        combined[finite],
        [result["color_percentile_min"], result["color_percentile_max"]],
    )
    time_edges = cell_edges(times)
    frequency_edges = cell_edges(frequency_grid)

    fig = base.plt.figure(figsize=(11.0, 4.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 3, width_ratios=(1.0, 1.0, 0.04), wspace=0.08)
    ax = fig.add_subplot(grid[0, 0], axes_class=Axes)
    fft_ax = fig.add_subplot(grid[0, 1], sharey=ax, axes_class=Axes)
    cax = fig.add_subplot(grid[0, 2], axes_class=Axes)

    mesh = None
    for panel_ax, values, track, title in (
        (ax, nufft_values, nufft_track, "NUFFT"),
        (fft_ax, fft_values, fft_track, "FFT"),
    ):
        mesh = panel_ax.pcolormesh(
            time_edges,
            frequency_edges,
            values,
            shading="auto",
            cmap="inferno",
            vmin=vmin,
            vmax=vmax,
        )
        panel_ax.plot(
            times,
            track,
            color="cyan",
            lw=1.3,
            # label="Injected chirp",
        )
        panel_ax.set_xlabel("Time [s]")
        panel_ax.set_ylim(frequency_grid[0], frequency_grid[-1])
        panel_ax.set_title(title)

    ax.plot(np.nan, np.nan, c='cyan', lw=1.3, label="Injected chirp")
    ax.legend(loc="upper left", frameon=True)
    ax.set_ylabel("Frequency [Hz]")
    fft_ax.tick_params(labelleft=False)
    # fig.suptitle(
    #     f"{result['detector']} detector-noise injection, "
    #     f"d = {result['distance_pc']:.2e} pc"
    # )
    cbar = fig.colorbar(mesh, cax=cax)
    cbar.set_label(r"$\log_{10}$ per-bin SNR")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    base.plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)
    DataQualityFlag, TimeSeries, to_gps = detector.import_gwpy()

    frequency_model = base.semicoherent_frequency_model(args)
    duration, f_end, n_chunks = base.analysis_span(args, frequency_model)
    analysis_samples = int(np.ceil(duration * args.sample_rate))
    section_duration = duration * (detector.NULL_TRIALS + 1)
    section_start, section_end = detector.find_continuous_segment(
        section_duration,
        DataQualityFlag,
        to_gps,
    )
    data = detector.fetch_open_strain(args, section_start, section_end, TimeSeries)
    detector_strain = np.asarray(data.value, dtype=float)
    required_samples = analysis_samples * (detector.NULL_TRIALS + 1)
    if detector_strain.size < required_samples:
        raise RuntimeError(
            f"Fetched {detector_strain.size} samples, need {required_samples}."
        )
    detector_strain = detector_strain[:required_samples]

    noise = detector.noise_curve_from_detector_data(data, args)
    chirp_power = base.integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )

    t = base.sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t), dtype=float)
    null_stats, mu0, sigma0 = detector.null_statistics(
        detector_strain,
        frequency_track,
        analysis_samples,
        args,
        noise,
    )
    if sigma0 <= 0.0:
        raise ValueError("numerical sigma0 must be positive")

    dof, scale = detector.scaled_chi2_from_moments(mu0, sigma0)
    thresholds = base.detection_thresholds(
        args.false_alarm_probability,
        args.detection_probability,
        dof,
        scale,
    )
    required_signal_statistic = thresholds["required_signal_statistic"]

    injection_start = detector.INJECTION_WINDOW_INDEX * analysis_samples
    injection_stop = injection_start + analysis_samples
    injection_gps_start = (
        section_start + detector.INJECTION_WINDOW_INDEX * duration
    )
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
    _, _, _, response, eta, psi = detector.make_detector_injection(
        args,
        reference_distance_m,
        duration,
        injection_gps_start,
    )
    reference_expected_power = detector.expected_power_for_detector_track(
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

    distance_m = 7 * reference_distance_m * np.sqrt(
        reference_expected_signal_statistic / required_signal_statistic
    )
    _, signal, _, response, _, _ = detector.make_detector_injection(
        args,
        distance_m,
        duration,
        injection_gps_start,
    )
    strain = detector_noise + signal
    recovered_times, recovered_frequencies, recovered_powers, recovered_statistic = (
        detector.recover_statistic(strain, frequency_track, args, noise)
    )

    times, frequency_grid, nufft_spectra, nufft_track = local_beta_spectrogram(
        strain,
        frequency_track,
        args,
        noise,
    )
    fft_times, fft_frequency_grid, fft_spectra, fft_track = direct_fft_spectrogram(
        strain,
        frequency_track,
        args,
        noise,
    )
    if not (
        np.array_equal(times, fft_times)
        and np.array_equal(frequency_grid, fft_frequency_grid)
    ):
        raise RuntimeError("NUFFT and FFT spectrogram grids do not match.")

    expected_power = detector.expected_power_for_detector_track(
        args,
        distance_m,
        chirp_power,
        frequency_track,
        response,
        noise,
    )
    expected_signal_statistic = hann_power_factor * expected_power
    expected_detection_probability = base.ncx2.sf(
        thresholds["false_alarm_statistic"] / scale,
        dof,
        expected_signal_statistic / scale,
    )

    plot_spectrograms(
        times,
        frequency_grid,
        nufft_spectra,
        nufft_track,
        fft_spectra,
        fft_track,
        {
            "detector": detector.DETECTOR,
            "distance_pc": distance_m / base.PARSEC_M,
            "color_percentile_min": args.color_percentile_min,
            "color_percentile_max": args.color_percentile_max,
        },
        args.output,
    )

    print("-" * 9 + "Detector data configuration" + "-" * 9)
    print(f"detector: {detector.DETECTOR}")
    print(f"O4a search range: {detector.O4A_START} to {detector.O4A_END}")
    print(f"data-quality flag: {detector.DATA_QUALITY_FLAG}")
    print(f"continuous section GPS: {section_start:.0f} - {section_end:.0f}")
    print(f"injection GPS start: {injection_gps_start:.0f}")
    print(f"analysis duration per window: {duration:.2f} s")
    print(f"sample rate: {args.sample_rate:g} Hz")
    print("-" * 9 + "Injection parameters" + "-" * 9)
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(f"final semicoherent-path frequency: {f_end:.6g} Hz")
    print(
        "zero-mismatch semicoherent distance estimate: "
        f"{reference_distance_m / base.PARSEC_M:.2e} pc"
    )
    print(f"95% distance threshold: {distance_m / base.PARSEC_M:.2e} pc")
    print(f"injection polarization: eta={eta:.6g}, psi={psi:.6g} rad")
    print(f"chunks in sensitivity formula: {n_chunks}")
    print(f"spectrogram columns: {times.size}")
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
    print(f"expected signal statistic: {expected_signal_statistic:.2e}")
    print(f"required signal statistic: {required_signal_statistic:.2e}")
    print(
        "detection probability at expected statistic: "
        f"{expected_detection_probability:.3g}"
    )
    print(f"analysis chunks summed: {recovered_powers.size}")
    print(
        "track frequency range in summed chunks: "
        f"{recovered_frequencies.min():.6g}-{recovered_frequencies.max():.6g} Hz"
    )
    print(
        "track time range in summed chunks: "
        f"{recovered_times.min():.6g}-{recovered_times.max():.6g} s"
    )
    print(f"spectrogram plot: {args.output}")


if __name__ == "__main__":
    main()
