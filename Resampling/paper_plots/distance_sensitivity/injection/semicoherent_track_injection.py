"""Semicoherent recovery check for a 0PN chirp at the predicted distance."""

import argparse
import sys
from pathlib import Path

import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import integrate
from scipy.stats import chi2, norm


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = SCRIPT_DIR / "semicoherent_track_injection.png"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from signal_generators import (  # noqa: E402
    C,
    G,
    MSUN,
    NoiseCurve,
    beta_0pn,
    make_0pn_track,
    tau_0pn,
)
from resampler import Resampler  # noqa: E402


PARSEC_M = lal.PC_SI
LAMBDA_THRESHOLD = 34.0
MAX_OBS_TIME = 3.0e7


class InjectionConfig:
    f0 = 40.0
    mchirp = 1.0e-2
    f_min = 40.0
    f_max = 60.0
    chunk_duration = 30.0
    chunk_overlap = 0.5
    sample_rate = 512.0
    lambda_threshold = LAMBDA_THRESHOLD

    def __init__(self, **kwargs):
        for name in CONFIG_FIELDS:
            setattr(self, name, kwargs.pop(name, getattr(type(self), name)))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown config fields: {unknown}")


CONFIG_FIELDS = (
    "f0",
    "mchirp",
    "f_min",
    "f_max",
    "chunk_duration",
    "chunk_overlap",
    "sample_rate",
    "lambda_threshold",
)


def time_to_frequency(f0, f_end, beta):
    if f0 > f_end:
        raise ValueError("f0 must not exceed f_end")
    return 3.0 / (8.0 * beta) * (1.0 - (f0 / f_end) ** (8.0 / 3.0))


def semicoherent_chunk_count(signal_duration, chunk_duration):
    return max(1, int(np.ceil(signal_duration / chunk_duration)))


class SemicoherentSensitivity:
    """Same 0PN distance formula used by semicoherent_sensitivity.py."""

    def __init__(self, noise):
        self.noise = noise
        frequency = noise.frequency
        asd = np.sqrt(noise.psd)
        self.base_integrand = 1.0 / (frequency ** (7.0 / 3.0) * asd**2)
        self.base_integral = integrate.cumulative_trapezoid(
            self.base_integrand,
            frequency,
            initial=0.0,
        )
        self.base_integrand_slope = np.diff(self.base_integrand) / np.diff(frequency)

    def base_integral_antiderivative(self, frequency):
        frequency = np.asarray(frequency)
        if np.any(
            (frequency < self.noise.frequency[0])
            | (frequency > self.noise.frequency[-1])
        ):
            raise ValueError("frequency is outside the ASD interpolation range")

        idx = np.searchsorted(self.noise.frequency, frequency, side="right") - 1
        idx = np.clip(idx, 0, len(self.noise.frequency) - 2)
        dx = frequency - self.noise.frequency[idx]
        return (
            self.base_integral[idx]
            + self.base_integrand[idx] * dx
            + 0.5 * self.base_integrand_slope[idx] * dx**2
        )

    def integrated_chirp_power(self, duration, f0, beta):
        chirp_factor = 1.0 - 8.0 / 3.0 * beta * duration
        if np.any(chirp_factor <= 0.0):
            raise ValueError("integration endpoint is past the chirp singularity")

        final_frequency = f0 * chirp_factor ** (-3.0 / 8.0)
        return (
            f0 ** (4.0 / 3.0)
            / beta
            * (
                self.base_integral_antiderivative(final_frequency)
                - self.base_integral_antiderivative(f0)
            )
        )

    def coherent_distance(self, duration, f0, mchirp, lambda_threshold):
        beta = beta_0pn(f0, mchirp)
        integration = self.integrated_chirp_power(duration, f0, beta)
        temp0 = 0.00757 / np.sqrt(lambda_threshold)
        temp1 = 3.0 * C * beta / f0**2
        return temp0 * temp1 * np.sqrt(integration)

    def semicoherent_distance(self, duration, f0, mchirp, lambda_threshold, chunk_duration):
        chunk_count = semicoherent_chunk_count(duration, chunk_duration)
        coherent_distance = self.coherent_distance(
            duration,
            f0,
            mchirp,
            lambda_threshold,
        )
        return coherent_distance / chunk_count ** 0.25


def h0_amplitude(distance_m, frequency, mchirp_msun):
    mchirp_kg = mchirp_msun * MSUN
    return (
        4.0
        / distance_m
        * (G * mchirp_kg / C**2) ** (5.0 / 3.0)
        * (np.pi * frequency / C) ** (2.0 / 3.0)
    )


def make_injection(config, distance_m, duration):
    n_samples = int(np.ceil(duration * config.sample_rate))
    dt = 1.0 / config.sample_rate
    t = dt * np.arange(n_samples)
    track = make_0pn_track(t, config.f0, config.mchirp)
    amplitude = h0_amplitude(distance_m, track.frequency, config.mchirp)
    strain = 0.5 * amplitude * track.signal
    return t, strain, track.frequency


def chunk_starts(n_samples, chunk_samples, hop_samples):
    starts = np.arange(0, n_samples - chunk_samples + 1, hop_samples)
    if starts.size == 0:
        return np.array([0], dtype=int)
    return starts


def resampled_track_power_for_chunk(
    signal_chunk,
    dt,
    window,
    f_start,
    beta_chunk,
    noise,
    resampler,
):
    t_rel = dt * np.arange(signal_chunk.size)
    tau = tau_0pn(t_rel, beta_chunk)

    resampler.timeseries = signal_chunk * window
    resampler.resampled_time = tau
    resampler.nufft()

    carrier_index = int(np.argmin(np.abs(resampler.freq_in_hz - f_start)))
    carrier_power = resampler.power_normalized[carrier_index]
    carrier_power /= np.mean(window**2)

    chunk_duration = signal_chunk.size * dt
    psd = noise.psd_at(f_start)
    return float(4.0 * chunk_duration * carrier_power / psd)


def overlap_scale(config):
    return 1.0 - config.chunk_overlap


def effective_chi_squared_params(n_chunks, window, hop_samples, statistic_scale):
    if n_chunks <= 0:
        raise ValueError("n_chunks must be positive")

    window_norm = float(np.sum(window**2))
    inflation = 1.0
    max_lag = int(np.ceil(window.size / hop_samples))
    for lag in range(1, max_lag):
        shift = lag * hop_samples
        if shift >= window.size:
            break
        correlation = float(np.sum(window[:-shift] * window[shift:]) / window_norm)
        inflation += 2.0 * (1.0 - lag / n_chunks) * correlation**2

    mean = 2.0 * statistic_scale * n_chunks
    variance = 4.0 * statistic_scale**2 * n_chunks * inflation
    dof = 2.0 * mean**2 / variance
    scale = variance / (2.0 * mean)
    return mean, variance, dof, scale, inflation


def null_significance(observed_statistic, dof, scale):
    p_value = chi2.sf(observed_statistic / scale, dof)
    sigma = norm.isf(p_value) if p_value > 0.0 else np.inf
    return p_value, sigma


def semicoherent_track_power(t, strain, frequency_track, config, noise):
    chunk_samples = int(round(config.chunk_duration * config.sample_rate))
    hop_samples = int(round(chunk_samples * (1.0 - config.chunk_overlap)))
    if chunk_samples < 2 or hop_samples < 1:
        raise ValueError("invalid chunking configuration")

    dt = 1.0 / config.sample_rate
    window = np.hanning(chunk_samples)
    starts = chunk_starts(strain.size, chunk_samples, hop_samples)
    resampler = Resampler(nthreads=4, eps=1.0e-9, upsampfac=2.0)

    chunk_times = []
    chunk_frequencies = []
    chunk_powers = []
    chunk_betas = []
    for start in starts:
        stop = start + chunk_samples
        if stop > strain.size:
            break
        center = start + chunk_samples // 2
        chunk_time = t[center]
        track_frequency = frequency_track[start]
        if not config.f_min <= track_frequency <= config.f_max:
            continue
        beta_chunk = beta_0pn(track_frequency, config.mchirp)
        chunk_times.append(chunk_time)
        chunk_frequencies.append(track_frequency)
        chunk_betas.append(beta_chunk)
        chunk_powers.append(
            resampled_track_power_for_chunk(
                strain[start:stop],
                dt,
                window,
                track_frequency,
                beta_chunk,
                noise,
                resampler,
            )
        )

    return (
        np.asarray(chunk_times),
        np.asarray(chunk_frequencies),
        np.asarray(chunk_powers),
        np.asarray(chunk_betas),
    )


def plot_chunks(chunk_times, chunk_frequencies, chunk_powers, output):
    fig, (ax_freq, ax_power) = plt.subplots(
        2,
        1,
        figsize=(7.2, 5.6),
        sharex=True,
        constrained_layout=True,
    )
    ax_freq.plot(chunk_times, chunk_frequencies, marker="o", ms=2.8, lw=1.0)
    ax_freq.set_ylabel("Track frequency [Hz]")
    ax_freq.grid(True, alpha=0.25)

    ax_power.plot(chunk_times, chunk_powers, marker="o", ms=2.8, lw=1.0)
    ax_power.set_xlabel("Chunk center time [s]")
    ax_power.set_ylabel("Resampled ASD-weighted power")
    ax_power.grid(True, alpha=0.25)

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)


def add_config_argument(parser, name, arg_type):
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        type=arg_type,
        default=getattr(InjectionConfig, name),
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject a 0PN chirp at the semicoherent distance sensitivity and "
            "sum Hann-windowed chunk power along the frequency track."
        )
    )
    for name in CONFIG_FIELDS:
        add_config_argument(parser, name, type(getattr(InjectionConfig, name)))
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def config_from_args(args):
    return InjectionConfig(**{field: getattr(args, field) for field in CONFIG_FIELDS})


def validate_config(config):
    if config.f0 <= 0.0:
        raise ValueError("f0 must be positive")
    if config.mchirp <= 0.0:
        raise ValueError("mchirp must be positive")
    if config.f_min > config.f0:
        raise ValueError("f_min must not exceed f0")
    if config.f_max <= config.f0:
        raise ValueError("f_max must be larger than f0")
    if config.sample_rate <= 2.0 * config.f_max:
        raise ValueError("sample_rate must exceed twice f_max")
    if config.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be positive")
    if not 0.0 <= config.chunk_overlap < 1.0:
        raise ValueError("chunk_overlap must be in [0, 1)")
    if config.lambda_threshold <= 0.0:
        raise ValueError("lambda_threshold must be positive")


def main():
    args = parse_args()
    config = config_from_args(args)
    validate_config(config)
    noise = NoiseCurve.from_asd_file(args.asd)

    beta = beta_0pn(config.f0, config.mchirp)
    duration = min(
        time_to_frequency(config.f0, config.f_max, beta),
        MAX_OBS_TIME,
    )
    sensitivity = SemicoherentSensitivity(noise)
    distance_m = sensitivity.semicoherent_distance(
        duration,
        config.f0,
        config.mchirp,
        config.lambda_threshold,
        config.chunk_duration,
    )

    t, strain, frequency_track = make_injection(config, distance_m, duration)
    chunk_times, chunk_frequencies, chunk_powers, chunk_betas = semicoherent_track_power(
        t,
        strain,
        frequency_track,
        config,
        noise,
    )
    raw_summed_power = float(np.sum(chunk_powers))
    statistic_scale = overlap_scale(config)
    signal_statistic = statistic_scale * raw_summed_power

    chunk_samples = int(round(config.chunk_duration * config.sample_rate))
    hop_samples = int(round(chunk_samples * statistic_scale))
    window = np.hanning(chunk_samples)
    null_mean, null_variance, eff_dof, eff_scale, variance_inflation = (
        effective_chi_squared_params(
            chunk_powers.size,
            window,
            hop_samples,
            statistic_scale,
        )
    )
    expected_detection_statistic = null_mean + signal_statistic
    p_value, sigma = null_significance(
        expected_detection_statistic,
        eff_dof,
        eff_scale,
    )

    plot_chunks(chunk_times, chunk_frequencies, chunk_powers, args.output)

    nonoverlap_count = semicoherent_chunk_count(duration, config.chunk_duration)
    nonoverlap_expected_power = config.lambda_threshold * np.sqrt(nonoverlap_count)
    hann_signal_power_factor = np.mean(window) ** 2 / np.mean(window**2)
    hann_expected_power = nonoverlap_expected_power * hann_signal_power_factor
    print("Semicoherent track injection")
    print(f"f0: {config.f0:g} Hz")
    print(f"frequency band: {config.f_min:g}-{config.f_max:g} Hz")
    print(f"Mc: {config.mchirp:.6e} Msun")
    print(f"beta: {beta:.12e}")
    print(f"signal duration to f_max: {duration:.6f} s")
    print(f"distance sensitivity: {distance_m / PARSEC_M:.12e} pc")
    print(f"distance sensitivity: {distance_m:.12e} m")
    print(f"lambda threshold: {config.lambda_threshold:.12g}")
    print(f"non-overlap chunks used in sensitivity formula: {nonoverlap_count}")
    print(f"analysis chunk duration: {config.chunk_duration:g} s")
    print(f"analysis chunk overlap: {config.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {chunk_powers.size}")
    print(f"raw resampled signal power sum: {raw_summed_power:.12e}")
    print(f"overlap correction factor: {statistic_scale:.12e}")
    print(f"semicoherent signal noncentrality: {signal_statistic:.12e}")
    print(
        "non-overlap sensitivity-scale power: "
        f"{nonoverlap_expected_power:.12e}"
    )
    print(f"Hann signal-power factor: {hann_signal_power_factor:.12e}")
    print(f"Hann-windowed expected signal power: {hann_expected_power:.12e}")
    print(
        "recovered / non-overlap scale: "
        f"{signal_statistic / nonoverlap_expected_power:.12e}"
    )
    print(
        "recovered / Hann-windowed scale: "
        f"{signal_statistic / hann_expected_power:.12e}"
    )
    print(f"effective chi2 dof: {eff_dof:.12e}")
    print(f"effective chi2 scale: {eff_scale:.12e}")
    print(f"overlap variance inflation: {variance_inflation:.12e}")
    print(f"null mean statistic: {null_mean:.12e}")
    print(f"null std statistic: {np.sqrt(null_variance):.12e}")
    print(f"expected statistic with signal: {expected_detection_statistic:.12e}")
    print(f"null p-value for expected statistic: {p_value:.12e}")
    print(f"Gaussian-equivalent significance: {sigma:.12e} sigma")
    if chunk_powers.size:
        print(
            "chunk power range: "
            f"{chunk_powers.min():.12e} to {chunk_powers.max():.12e}"
        )
        print(
            "chunk beta range: "
            f"{chunk_betas.min():.12e} to {chunk_betas.max():.12e}"
        )
    print(f"asd: {args.asd}")
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
