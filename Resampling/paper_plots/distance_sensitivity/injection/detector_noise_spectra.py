"""Plot NUFFT and FFT spectra for an Andromeda detector-noise injection."""

import argparse
import sys
from pathlib import Path

from matplotlib.axes import Axes
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
FIGS_DIR = SCRIPT_DIR / "figs"
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
PAPER_STYLE_PATH = PAPER_PLOTS_DIR / "paper.mplstyle"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import detector_noise_injection as detector  # noqa: E402
from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


base.plt.style.use(PAPER_STYLE_PATH)

OUTPUT_PATH = FIGS_DIR / "detector_noise_spectra.png"
ANDROMEDA_DISTANCE_PC = 7.65e5
# Sky position of the plotted injection, in radians.
INJECTION_RA = base.ANDROMEDA_RA
INJECTION_DEC = base.ANDROMEDA_DEC
# Polarization parameters for the plotted injection.  eta is in [-1, 1] and
# psi is in radians.
INJECTION_ETA = 0
INJECTION_PSI = 0.507
DEFAULT_FREQUENCY_BINS = 700
DEFAULT_COLOR_PERCENTILE_MIN = 5.0
DEFAULT_COLOR_PERCENTILE_MAX = 99.9
PSD_AVERAGES = 16
PSD_OVERLAP_FRACTION = 0.5
NUFFT_RESPONSE_OVERSAMPLE = 4
NUFFT_RESPONSE_BLOCK_SIZE = 32


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject an Andromeda source into cached O4a strain, "
            "comparing local-beta NUFFT spectra with direct FFT spectra."
        )
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument(
        "--injection-index",
        type=int,
        default=detector.INJ_NUM,
        help="Zero-based index of the GPS-ordered cached strain file.",
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
    if args.injection_index < 0:
        raise ValueError("injection_index must be non-negative")
    if args.f0 <= 0.0 or args.mchirp <= 0.0:
        raise ValueError("f0 and chirp mass must be positive.")
    if args.f_min > args.f0 or args.f_max <= args.f0:
        raise ValueError("The saved frequency range must contain f0.")
    if args.sample_rate <= 2.0 * args.f_max:
        raise ValueError("The sample rate must exceed twice the saved upper frequency.")
    if args.chunk_duration <= 0.0 or args.chunk_overlap != 0.0:
        raise ValueError("Use positive, contiguous coherent segments.")
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


def effective_nufft_psd_for_frequencies(
    frequencies,
    tau,
    window,
    noise,
    dt,
):
    """Propagate an ordinary PSD through the finite-duration NUFFT response.

    For each output frequency, the FFT below evaluates the response to every
    uniformly sampled input frequency.  Integrating the input PSD against the
    squared response includes the Hann spectral window and the complete
    nonlinear ``t -> tau`` phase map.
    """
    frequencies = np.asarray(frequencies, dtype=float)
    tau = np.asarray(tau, dtype=float)
    window = np.asarray(window, dtype=float)
    if tau.shape != window.shape:
        raise ValueError("tau and window must have the same shape.")
    if dt <= 0.0:
        raise ValueError("dt must be positive.")

    response_samples = NUFFT_RESPONSE_OVERSAMPLE * tau.size
    input_frequencies = np.fft.fftfreq(response_samples, dt)
    # NoiseCurve stores a positive-frequency one-sided PSD.  Clipping only
    # affects DC; all other FFT frequencies lie in the measured PSD range.
    psd_frequencies = np.clip(
        np.abs(input_frequencies),
        noise.frequency[0],
        noise.frequency[-1],
    )
    input_psd = noise.psd_at(psd_frequencies)
    normalization = response_samples * np.sum(window**2)
    psd = np.empty_like(frequencies)

    for start in range(0, frequencies.size, NUFFT_RESPONSE_BLOCK_SIZE):
        stop = min(start + NUFFT_RESPONSE_BLOCK_SIZE, frequencies.size)
        phase = np.exp(-2j * np.pi * frequencies[start:stop, None] * tau)
        response = np.fft.fft(
            phase * window,
            n=response_samples,
            axis=1,
        )
        psd[start:stop] = np.abs(response) ** 2 @ input_psd / normalization

    return psd


def robust_noise_curve(data, args):
    """Estimate a one-sided PSD from 16 preceding 30-second periodograms."""
    psd = data.psd(
        fftlength=args.chunk_duration,
        overlap=PSD_OVERLAP_FRACTION * args.chunk_duration,
        method="median",
        window="hann",
    )
    frequency = np.asarray(psd.frequencies.value, dtype=float)
    psd_values = np.asarray(psd.value, dtype=float)
    valid = (
        np.isfinite(frequency)
        & np.isfinite(psd_values)
        & (frequency > 0.0)
        & (psd_values > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise ValueError("GWpy median PSD did not produce a usable frequency band.")
    return base.NoiseCurve(frequency[valid], psd_values[valid])


def psd_history_samples(args):
    chunk_samples, _ = base.chunk_config(args)
    psd_hop_samples = int(
        round(chunk_samples * (1.0 - PSD_OVERLAP_FRACTION))
    )
    return chunk_samples + (PSD_AVERAGES - 1) * psd_hop_samples


def robust_segment_noise_curves(data, args, analysis_samples, history_samples):
    """Estimate segment PSDs from local noise-only windows in cached strain.

    Segments with sufficient history use the immediately preceding samples.
    At the start of the cached file, where no preceding data exist, use the
    earliest complete history window instead.
    """
    if history_samples != psd_history_samples(args):
        raise ValueError("PSD history length does not match the configured averages.")
    if len(data) < max(history_samples, analysis_samples):
        raise ValueError("Detector data do not include the required analysis span.")
    starts, _ = detector.analysis_chunk_starts(analysis_samples, args)
    noise_curves = []
    for start in starts:
        history_start = max(0, start - history_samples)
        history = data[history_start : history_start + history_samples]
        noise_curves.append(robust_noise_curve(history, args))
    return noise_curves


def local_beta_spectrogram(strain, frequency_track, args, noise_curves):
    chunk_samples, hop_samples = base.chunk_config(args)
    starts = base.chunk_starts(strain.size, chunk_samples, hop_samples)
    if len(noise_curves) != starts.size:
        raise ValueError("Need one noise curve per coherent segment.")
    dt = 1.0 / args.sample_rate
    t_rel = dt * np.arange(chunk_samples)
    window = np.hanning(chunk_samples)
    frequency_grid = np.linspace(args.f_min, args.f_max, args.frequency_bins)
    resampler = base.Resampler(nthreads=4, eps=1e-2)

    times = []
    carrier_frequencies = []
    columns = []

    for segment_index, start in enumerate(starts):
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
            tau,
            window,
            noise_curves[segment_index],
            dt,
        )
        normalized_power = 4.0 * chunk_samples * dt * band_power / band_psd
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


def direct_fft_spectrogram(strain, frequency_track, args, noise_curves):
    chunk_samples, hop_samples = base.chunk_config(args)
    starts = base.chunk_starts(strain.size, chunk_samples, hop_samples)
    if len(noise_curves) != starts.size:
        raise ValueError("Need one noise curve per coherent segment.")
    dt = 1.0 / args.sample_rate
    window = np.hanning(chunk_samples)
    window_power = np.mean(window**2)
    frequency_grid = np.linspace(args.f_min, args.f_max, args.frequency_bins)
    fft_frequencies = np.fft.fftfreq(chunk_samples, dt)

    times = []
    track_frequencies = []
    columns = []

    for segment_index, start in enumerate(starts):
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
            4.0
            * chunk_samples
            * dt
            * band_power
            / noise_curves[segment_index].psd_at(band_frequencies)
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
        (ax, nufft_values, nufft_track, "Resampled"),
        (fft_ax, fft_values, fft_track, "Non-resampled"),
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
            ls = '--'
            # label="Injected chirp",
        )
        panel_ax.set_xlabel("Time [s]")
        panel_ax.set_ylim(frequency_grid[0], frequency_grid[-1])
        panel_ax.set_title(title)

    ax.plot(np.nan, np.nan, c='cyan', ls='--', label="Injected chirp")
    ax.legend(loc="upper left", frameon=True, handlelength=3)
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
    plot_args = parse_args()
    args = detector.analysis_config()
    vars(args).update(vars(plot_args))
    validate_args(args)
    _, TimeSeries, _ = detector.import_gwpy()

    frequency_model = base.semicoherent_frequency_model(args)
    duration, f_end, n_chunks = base.analysis_span(args, frequency_model)
    analysis_samples = int(round(duration * args.sample_rate))
    history_samples = psd_history_samples(args)
    chunk_samples, _ = base.chunk_config(args)
    background_samples = detector.BACKGROUND_SPECTRA * chunk_samples
    if history_samples > background_samples:
        raise ValueError(
            "The cached background is shorter than the requested PSD history"
        )
    background_duration = detector.BACKGROUND_SPECTRA * args.chunk_duration
    block_duration = background_duration + duration
    data, strain_path = detector.read_cached_strain(
        args,
        block_duration,
        TimeSeries,
        inj_num=args.injection_index,
    )
    required_samples = background_samples + analysis_samples
    if len(data) < required_samples:
        raise RuntimeError(
            "Cached detector data are shorter than the background and "
            "analysis intervals"
        )
    data = data[:required_samples]
    data_start = float(data.t0.value)
    injection_start = data_start
    injection_end = injection_start + duration
    injection_data = data[:analysis_samples]
    detector_noise = np.asarray(injection_data.value, dtype=float)

    t = base.sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t), dtype=float)
    injection_noise_curves = robust_segment_noise_curves(
        data,
        args,
        analysis_samples,
        history_samples,
    )
    distance_m = ANDROMEDA_DISTANCE_PC * base.PARSEC_M
    _, signal, _, _, eta, psi = detector.make_detector_injection(
        args,
        distance_m,
        duration,
        injection_start,
        ra=INJECTION_RA,
        dec=INJECTION_DEC,
        eta=INJECTION_ETA,
        psi=INJECTION_PSI,
    )
    strain = detector_noise + signal

    times, frequency_grid, nufft_spectra, nufft_track = local_beta_spectrogram(
        strain,
        frequency_track,
        args,
        injection_noise_curves,
    )
    fft_times, fft_frequency_grid, fft_spectra, fft_track = direct_fft_spectrogram(
        strain,
        frequency_track,
        args,
        injection_noise_curves,
    )
    if not (
        np.array_equal(times, fft_times)
        and np.array_equal(frequency_grid, fft_frequency_grid)
    ):
        raise RuntimeError("NUFFT and FFT spectrogram grids do not match.")

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
    print(f"data-quality flag: {detector.DATA_QUALITY_FLAG}")
    print(f"cached strain index: {args.injection_index}")
    print(f"cached strain file: {strain_path}")
    print(f"injection GPS: {injection_start:.0f} - {injection_end:.0f}")
    print(f"cached strain GPS start: {data_start:.0f}")
    print(f"analysis duration: {duration:.2f} s")
    print(f"sample rate: {args.sample_rate:g} Hz")
    print(
        f"PSD estimator: median of {PSD_AVERAGES} local noise-only periodograms "
        f"({args.chunk_duration:g} s FFT, "
        f"{PSD_OVERLAP_FRACTION * args.chunk_duration:g} s overlap)"
    )
    print(
        "NUFFT PSD propagation: finite-duration response "
        f"({NUFFT_RESPONSE_OVERSAMPLE}x frequency oversampling)"
    )
    print("-" * 9 + "Injection parameters" + "-" * 9)
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(
        "sky location: Andromeda (M31) "
        f"(ra={np.rad2deg(INJECTION_RA):.6f} deg, "
        f"dec={np.rad2deg(INJECTION_DEC):.6f} deg)"
    )
    print(f"final semicoherent-path frequency: {f_end:.6g} Hz")
    print(f"injection distance: {distance_m / base.PARSEC_M:.2e} pc")
    print(f"injection polarization: eta={eta:.6g}, psi={psi:.6g} rad")
    print(f"chunks in sensitivity formula: {n_chunks}")
    print(f"spectrogram columns: {times.size}")
    print(f"spectrogram plot: {args.output}")


if __name__ == "__main__":
    main()
