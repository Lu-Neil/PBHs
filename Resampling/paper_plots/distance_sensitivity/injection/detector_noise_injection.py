"""Inject a 3.5PN chirp into H1 O4a data and recover it with a weighted stack-slide.

For every coherent segment, the preceding 15 noise-only segments and the
analysis segment are resampled together with a batched NUFFT.  Their carrier
powers determine the effective PSD and semicoherent normalization.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import lal
import numpy as np
from matplotlib.axes import Axes
from scipy.integrate import quad
from scipy.optimize import brentq

PAPER_PLOTS_DIR = Path(__file__).resolve().parents[2]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


# Analysis choices
DETECTOR = "H1"
DATA_QUALITY_FLAG = "H1_DATA"
O4A_START = "2023-05-24 15:00:00 UTC"
O4A_END = "2024-01-16 16:00:00 UTC"
INJ_NUM = 0  # Zero-based index into the GPS-ordered cached strain files.
INJECTION_RA = base.ANDROMEDA_RA
INJECTION_DEC = base.ANDROMEDA_DEC
INJECTION_ETA = 0 #ase.INJECTION_ETA
INJECTION_PSI = 0.5 #base.INJECTION_PSI

BACKGROUND_SPECTRA = 15
RESPONSE_SPACING = 1.0
FALSE_ALARM_PROBABILITY = 1e-6
DETECTION_PROBABILITY = 0.95
COHERENT_PN_MISMATCH = 0.05
SEMICOHERENT_BANK_MISMATCH = 0.05
SIGNAL_POWER_RETENTION = (
    (1 - COHERENT_PN_MISMATCH) * (1 - SEMICOHERENT_BANK_MISMATCH)
)
DISTANCE_RATIOS = np.logspace(-1, 2, 20)
OUTPUT_PATH = base.FIGS_DIR / "z_vs_distance_detector_noise.png"
STRAIN_DATA_DIR = Path(__file__).resolve().parent / "strain_data"
HDF5_DATASET = "strain"

DETECTOR_INDEX = {
    "H1": lal.LALDetectorIndexLHODIFF,
    "L1": lal.LALDetectorIndexLLODIFF,
    "V1": lal.LALDetectorIndexVIRGODIFF,
    "K1": lal.LALDetectorIndexKAGRADIFF,
}


def analysis_config():
    """Parameters shared with ``detector_noise_spectra.py``."""
    return SimpleNamespace(
        f0=40.0,
        mchirp=0.1,
        f_min=40.0,
        f_max=60.0,
        chunk_duration=30.0,
        chunk_overlap=0.0,
        sample_rate=512.0,
    )


def import_gwpy():
    from gwpy.segments import DataQualityFlag
    from gwpy.time import to_gps
    from gwpy.timeseries import TimeSeries

    return DataQualityFlag, TimeSeries, to_gps


def find_continuous_segment(duration, DataQualityFlag, to_gps, excluded_intervals=()):
    """Return the first O4a science segment of the requested duration."""
    flag = DataQualityFlag.fetch_open_data(
        DATA_QUALITY_FLAG, float(to_gps(O4A_START)), float(to_gps(O4A_END))
    )
    excluded_intervals = sorted(excluded_intervals)

    for segment in flag.active:
        start, end = map(float, segment)
        for excluded_start, excluded_end in excluded_intervals:
            if excluded_end <= start or excluded_start >= end:
                continue
            if excluded_start - start >= duration:
                return start, start + duration
            start = max(start, excluded_end)
        if end - start >= duration:
            return start, start + duration

    raise RuntimeError(f"No {duration:g} s O4a {DATA_QUALITY_FLAG} segment found")


def fetch_open_strain(args, start, end, TimeSeries):
    """Download strain for use by the separate cache-building script."""
    data = TimeSeries.fetch_open_data(DETECTOR, start, end, cache=True)
    if not np.isclose(float(data.sample_rate.value), args.sample_rate):
        data = data.resample(args.sample_rate)
    return data.detrend("constant")


def read_cached_strain(args, duration, TimeSeries, inj_num=INJ_NUM):
    """Read and trim one sufficiently long GPS-ordered cached strain file."""
    duration = int(round(duration))
    sample_rate = int(round(args.sample_rate))
    prefix = f"{DETECTOR}-"
    suffix = f"-{sample_rate}Hz.hdf5"
    cached_files = []
    for path in STRAIN_DATA_DIR.glob(f"{prefix}*-*{suffix}"):
        start_text, cached_duration_text = (
            path.name.removeprefix(prefix).removesuffix(suffix).rsplit("-", 1)
        )
        start = int(start_text)
        cached_duration = int(cached_duration_text)
        if cached_duration >= duration:
            cached_files.append((start, path))
    cached_files.sort()

    if not cached_files:
        raise FileNotFoundError(
            f"No cached {DETECTOR} {sample_rate} Hz strain files at least "
            f"{duration} s long in {STRAIN_DATA_DIR}. Run "
            "caching_analysis_segments.py first."
        )
    if not 0 <= inj_num < len(cached_files):
        raise IndexError(
            f"INJ_NUM={inj_num} is outside the available zero-based range "
            f"0-{len(cached_files) - 1}"
        )

    start, path = cached_files[inj_num]
    data = TimeSeries.read(path, format="hdf5", path=HDF5_DATASET)
    required_samples = round(duration * args.sample_rate)
    if len(data) < required_samples:
        raise RuntimeError(
            f"Cached {path.name} contains {len(data)} samples; "
            f"at least {required_samples} are required"
        )
    if not np.isclose(float(data.t0.value), start):
        raise RuntimeError(
            f"Cached {path.name} starts at GPS {float(data.t0.value):g}; "
            f"expected {start:g}"
        )
    if not np.isclose(float(data.sample_rate.value), args.sample_rate):
        raise RuntimeError(
            f"Cached {path.name} has sample rate "
            f"{float(data.sample_rate.value):g} Hz; "
            f"expected {args.sample_rate:g} Hz"
        )
    return data[:required_samples], path


def analysis_chunk_starts(n_samples, args):
    chunk_samples, hop_samples = base.chunk_config(args)
    return base.chunk_starts(n_samples, chunk_samples, hop_samples), chunk_samples


def detector_response(
    detector,
    gps_start,
    t,
    eta,
    psi,
    spacing=RESPONSE_SPACING,
    *,
    ra=INJECTION_RA,
    dec=INJECTION_DEC,
):
    """Complex detector response ``(F+ + i eta Fx) / sqrt(1 + eta^2)``."""
    tensor = lal.CachedDetectors[DETECTOR_INDEX[detector]].response
    response_times = np.arange(0, t[-1] + spacing, spacing)
    response = np.empty(response_times.size, dtype=complex)

    for i, offset in enumerate(response_times):
        gmst = lal.GreenwichMeanSiderealTime(lal.LIGOTimeGPS(gps_start + offset))
        f_plus, f_cross = lal.ComputeDetAMResponse(tensor, ra, dec, psi, gmst)
        response[i] = (f_plus + 1j * eta * f_cross) / np.sqrt(1 + eta**2)

    return np.interp(t, response_times, response.real) + 1j * np.interp(
        t, response_times, response.imag
    )


def make_detector_injection(
    args,
    distance_m,
    duration,
    gps_start,
    *,
    ra=INJECTION_RA,
    dec=INJECTION_DEC,
    eta=INJECTION_ETA,
    psi=INJECTION_PSI,
):
    t = base.sample_times(args.sample_rate, duration)
    track = base.make_35pn_track(t, args.f0, args.f_max, args.mchirp)
    response = detector_response(
        DETECTOR, gps_start, t, eta, psi, ra=ra, dec=dec
    )
    amplitude = base.h0_amplitude(distance_m, track.frequency, args.mchirp)
    strain = amplitude * np.real(response * track.signal)
    return t, strain, track.frequency, response, eta, psi


def resample_carriers(noise_history, analyses, frequency_track, args):
    """Resample each analysis segment with its preceding noise segments."""
    analyses = np.atleast_2d(analyses)
    starts, chunk_samples = analysis_chunk_starts(analyses.shape[1], args)
    history = np.asarray(noise_history).reshape(-1, chunk_samples)
    dt = 1 / args.sample_rate
    t = dt * np.arange(chunk_samples)
    window = np.hanning(chunk_samples)
    resampler = base.Resampler(
        nthreads=4,
        eps=1e-2,
        allow_two_transforms=True,
    )
    carriers = []

    for segment_index, start in enumerate(starts):
        frequency = float(frequency_track[start])
        if not args.f_min <= frequency <= args.f_max:
            continue

        tau = base.tau_0pn(t, base.beta_0pn(frequency, args.mchirp))
        background = history[
            segment_index : segment_index + BACKGROUND_SPECTRA
        ]
        analysis = analyses[:, start : start + chunk_samples]
        resampler.timeseries = np.ascontiguousarray(
            np.vstack((background, analysis)) * window
        )
        resampler.resampled_time = tau
        resampler.nufft()

        bin_index = np.argmin(np.abs(resampler.freq_in_hz - frequency))
        carriers.append(
            resampler.weights_normalized[:, bin_index] / np.sqrt(np.mean(window**2))
        )

    return np.asarray(carriers).T


def normalized_powers(carriers, effective_psds, segment_duration):
    """Return ``chi2_2`` powers using a one-sided PSD for real strain."""
    return 4 * segment_duration * np.abs(carriers) ** 2 / effective_psds


def background_statistics(carriers, segment_duration):
    """Estimate the PSD and null distribution without self-normalization."""
    psds = 2 * segment_duration * np.abs(carriers) ** 2
    effective_psds = psds.mean(axis=0)
    leave_one_out_psds = (psds.sum(axis=0) - psds) / (len(psds) - 1)
    powers = 2 * psds / leave_one_out_psds
    return (
        effective_psds,
        powers,
        powers.mean(axis=0),
        powers.std(axis=0, ddof=1),
    )


def signal_model(args, frequency_track, effective_psds, gps_start):
    """Calculate optimal segment weights and one-parsec noncentralities."""
    starts, chunk_samples = analysis_chunk_starts(frequency_track.size, args)
    dt = 1 / args.sample_rate
    t = dt * np.arange(frequency_track.size)
    window = np.hanning(chunk_samples)
    window_power = np.mean(window**2)
    response = detector_response(
        DETECTOR,
        gps_start,
        t,
        INJECTION_ETA,
        INJECTION_PSI,
        ra=INJECTION_RA,
        dec=INJECTION_DEC,
    )
    raw_weights, antenna_factors = [], []

    for start, psd in zip(starts, effective_psds):
        stop = start + chunk_samples
        frequency = float(frequency_track[start])
        if not args.f_min <= frequency <= args.f_max:
            continue

        amplitude = base.h0_amplitude(
            base.PARSEC_M, frequency_track[start:stop], args.mchirp
        )
        segment_response = response[start:stop]
        coherent_amplitude = np.mean(window * amplitude * segment_response)
        raw_weights.append(np.abs(coherent_amplitude) ** 2 / (window_power * psd))
        antenna = np.mean(window * segment_response)
        antenna_factors.append(np.abs(antenna) ** 2 / window_power)

    raw_weights = np.asarray(raw_weights)
    weights = raw_weights / np.linalg.norm(raw_weights)
    noncentralities = args.chunk_duration * raw_weights
    return weights, noncentralities, np.asarray(antenna_factors)


def statistic(powers, weights, means, standard_deviations):
    return float(weights @ ((powers - means) / standard_deviations))


def weighted_chi2_survival(value, coefficients, noncentralities=None):
    """Survival function of ``sum(a_i chi2_2(lambda_i))`` by Fourier inversion."""
    if noncentralities is None:
        noncentralities = np.zeros_like(coefficients)
    mean = np.sum(coefficients * (2 + noncentralities))

    def integrand(t):
        if t == 0:
            return mean - value
        denominator = 1 - 2j * t * coefficients
        log_cf = -np.sum(np.log(denominator)) + np.sum(
            1j * t * coefficients * noncentralities / denominator
        )
        return np.imag(np.exp(log_cf - 1j * t * value)) / t

    integral = quad(integrand, 0, np.inf, epsabs=1e-10, epsrel=1e-8, limit=500)[0]
    return float(np.clip(0.5 + integral / np.pi, 0, 1))


def weighted_chi2_threshold(coefficients):
    mean = 2 * np.sum(coefficients)
    std = np.sqrt(4 * np.sum(coefficients**2))
    upper = mean + 8 * std
    while weighted_chi2_survival(upper, coefficients) > FALSE_ALARM_PROBABILITY:
        upper += 4 * std
    return brentq(
        lambda x: weighted_chi2_survival(x, coefficients)
        - FALSE_ALARM_PROBABILITY,
        mean,
        upper,
    )


def detection_distance(
    weights,
    reference_noncentralities,
    background_means,
    background_std,
):
    """Find the distance giving the requested weighted-chi2 detection probability."""
    coefficients = weights / background_std
    threshold = weighted_chi2_threshold(coefficients)

    def probability_error(distance_scale):
        return weighted_chi2_survival(
            threshold,
            coefficients,
            distance_scale * reference_noncentralities,
        ) - DETECTION_PROBABILITY

    scale_max = min(1.0, 1 / np.max(reference_noncentralities))
    while probability_error(scale_max) < 0:
        scale_max *= 2
    distance_scale = brentq(
        probability_error,
        0,
        scale_max,
        xtol=np.nextafter(0.0, 1.0),
        rtol=1e-10,
    )
    distance = base.PARSEC_M / np.sqrt(distance_scale)
    statistic_threshold = threshold - np.sum(
        weights * background_means / background_std
    )
    return distance, distance_scale, threshold, statistic_threshold, coefficients


def plot_distance_scan(distances, expected, recovered, sensitivity, threshold):
    base.plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig = base.plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, axes_class=Axes)
    distance_pc = distances / base.PARSEC_M

    ax.plot(distance_pc, expected, label="Expected signal")
    ax.plot(
        distance_pc,
        recovered,
        "x",
        ms=5,
        label="Recovered (one noise realization)",
    )
    ax.axvline(
        sensitivity / base.PARSEC_M,
        color="k",
        ls=":",
        label="95% detection distance",
    )
    ax.axhline(
        threshold,
        color="0.35",
        ls=":",
        label="Weighted-$\\chi^2$ false-alarm threshold",
    )
    ax.set(xlabel="Injection distance [pc]", ylabel=r"weighted $n_\sigma$")
    ax.set_xscale("log")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_xlim(distance_pc[0], distance_pc[-1])
    ax.set_ylim(
        min(0, 1.1 * np.min(recovered)),
        1.1 * max(np.max(expected), np.max(recovered), threshold),
    )
    ax.grid(alpha=0.25)
    ax.legend(handlelength=2)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    base.plt.close(fig)


def main():
    # Build the 3.5PN frequency track and truncate it to complete 30 s chunks.
    args = analysis_config()
    _, TimeSeries, _ = import_gwpy()
    frequency_model = base.semicoherent_frequency_model(args)
    duration, final_frequency, n_chunks = base.analysis_span(args, frequency_model)

    # Select one cached science block.  Its first 15 chunks seed the rolling
    # background; the remaining 35 chunks form the analysis interval.
    chunk_samples, _ = base.chunk_config(args)
    background_samples = BACKGROUND_SPECTRA * chunk_samples
    background_duration = BACKGROUND_SPECTRA * args.chunk_duration
    strain_data, strain_path = read_cached_strain(
        args,
        background_duration + duration,
        TimeSeries,
        INJ_NUM,
    )
    background_start = float(strain_data.t0.value)
    injection_start = background_start + background_duration
    background_end = injection_start
    injection_end = injection_start + duration

    # Keep a noise-only copy of the entire block.  As the analysis advances,
    # previous analysis chunks enter the rolling background without a signal.
    t = base.sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t))
    strain = np.asarray(strain_data.value)
    background = strain[:background_samples]
    noise = strain[background_samples:]
    noise_history = np.concatenate((background, noise))

    # Each NUFFT batch contains the preceding 15 noise chunks followed by the
    # current analysis chunk.  The returned rows retain this ordering.
    carriers = resample_carriers(
        noise_history, noise, frequency_track, args
    )
    background_carriers = carriers[:-1]
    noise_carrier = carriers[-1]

    # Estimate the effective one-sided PSD, mu, and sigma independently for
    # every coherent chunk.  Leave-one-out PSDs avoid forcing mu to equal two.
    effective_psds, null_powers, means, standard_deviations = (
        background_statistics(background_carriers, args.chunk_duration)
    )

    # Predict the one-parsec signal power and construct the Eq. (19) weights.
    # The two bank mismatches reduce the expected noncentrality, not the weights.
    weights, reference_noncentralities, antenna_factors = signal_model(
        args,
        frequency_track,
        effective_psds,
        injection_start,
    )
    expected_noncentralities = SIGNAL_POWER_RETENTION * reference_noncentralities
    null_statistics = ((null_powers - means) / standard_deviations) @ weights

    # Solve for the distance at which 95% of signals exceed the target
    # false-alarm threshold under the weighted-chi-square model.
    distance, distance_scale, raw_threshold, threshold, coefficients = (
        detection_distance(
            weights,
            expected_noncentralities,
            means,
            standard_deviations,
        )
    )

    # Inject at that distance and recover the empirical statistic.  Only the
    # final row contains the signal; all rolling-background rows remain noise.
    _, signal, *_ = make_detector_injection(
        args, distance, duration, injection_start
    )
    injected_carrier = resample_carriers(
        noise_history, noise + signal, frequency_track, args
    )[-1]
    signal_carrier = injected_carrier - noise_carrier
    powers = normalized_powers(
        injected_carrier, effective_psds, args.chunk_duration
    )
    recovered = statistic(powers, weights, means, standard_deviations)
    expected = np.sum(
        weights * distance_scale * expected_noncentralities / standard_deviations
    )

    # Use amplitude linearity to scan other distances without more NUFFTs.
    # Signal power, and hence its expectation, scales as distance^-2.
    distances = distance * DISTANCE_RATIOS
    expected_scan = expected / DISTANCE_RATIOS**2
    scan_carriers = noise_carrier + signal_carrier / DISTANCE_RATIOS[:, None]
    scan_powers = normalized_powers(
        scan_carriers, effective_psds, args.chunk_duration
    )
    recovered_scan = ((scan_powers - means) / standard_deviations) @ weights
    plot_distance_scan(distances, expected_scan, recovered_scan, distance, threshold)

    # Convert the recovered statistic to its null survival probability and
    # check the detection probability achieved at the solved distance.
    statistic_offset = np.sum(weights * means / standard_deviations)
    p_value = weighted_chi2_survival(recovered + statistic_offset, coefficients)
    significance = base.norm.isf(p_value) if p_value else np.inf
    achieved_probability = weighted_chi2_survival(
        raw_threshold,
        coefficients,
        distance_scale * expected_noncentralities,
    )

    print("Detector-noise injection")
    print(f"  detector / data flag:       {DETECTOR} / {DATA_QUALITY_FLAG}")
    print(f"  cached strain index:        {INJ_NUM}")
    print(f"  cached strain file:         {strain_path}")
    print(f"  injection GPS:              {injection_start:.0f}-{injection_end:.0f}")
    print(f"  background GPS:             {background_start:.0f}-{background_end:.0f}")
    print(f"  frequency range:            {args.f0:g}-{final_frequency:.3f} Hz")
    print(f"  chirp mass:                 {args.mchirp:g} Msun")
    print(f"  coherent segments:          {n_chunks} x {args.chunk_duration:g} s")
    print(f"  background spectra:         {BACKGROUND_SPECTRA}")
    print(f"  mismatch power retention:   {SIGNAL_POWER_RETENTION:.4f}")
    print(f"  sensitivity:                {distance / base.PARSEC_M:.3e} pc")
    print(f"  weighted-chi2 threshold:    {threshold:.3f}")
    print(f"  expected / recovered nσ:    {expected:.3f} / {recovered:.3f}")
    print(f"  achieved detection prob.:   {achieved_probability:.3f}")
    print(f"  recovered significance:     {significance:.3f} sigma")
    print(
        f"  background nσ range:        "
        f"{null_statistics.min():.2f} to {null_statistics.max():.2f}"
    )
    print(
        f"  antenna-power range:        "
        f"{antenna_factors.min():.2e} to {antenna_factors.max():.2e}"
    )
    print(
        f"  effective PSD range:        "
        f"{effective_psds.min():.2e} to {effective_psds.max():.2e} Hz^-1"
    )
    print(f"  plot:                       {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
