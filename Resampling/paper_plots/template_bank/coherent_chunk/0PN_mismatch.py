"""Find the longest chunk time allowed by 0PN-vs-3.5PN GW mismatch."""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = SCRIPT_DIR / "figs" / "0PN_mismatch.png"
MIN_SAMPLES = 8

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from signal_generators import (  # noqa: E402
    NoiseCurve,
    SignalTrack,
    TaylorT4PhaseModel,
    beta_0pn,
    complex_signal_from_phase,
    make_0pn_track,
)


class ZeroPNMismatchConfig:
    f_min = 40.0
    f_max = 60.0
    mchirp_min = 5.0e-4
    mchirp_max = 1.0e-1
    eta = 0.25
    max_mismatch = 0.05
    search_time_max = 90.0
    sample_rate = 512.0
    zero_pad_factor = 4
    amplitude_frequency_power = 2.0 / 3.0
    n_time_plot = 120
    n_f0_validate = 5
    n_mchirp_validate = 5

    def __init__(self, **kwargs):
        for name in CONFIG_FIELDS:
            setattr(self, name, kwargs.pop(name, getattr(type(self), name)))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown config fields: {unknown}")


CONFIG_FIELDS = (
    "f_min",
    "f_max",
    "mchirp_min",
    "mchirp_max",
    "eta",
    "max_mismatch",
    "search_time_max",
    "sample_rate",
    "zero_pad_factor",
    "amplitude_frequency_power",
    "n_time_plot",
    "n_f0_validate",
    "n_mchirp_validate",
)


class PNComparisonWaveformModel:
    """Reusable 0PN and TaylorT4 3.5PN waveform generator for one corner."""

    def __init__(
        self,
        f0_hz,
        mchirp_msun,
        eta,
        t_max,
    ):
        self.f0_hz = float(f0_hz)
        self.mchirp_msun = float(mchirp_msun)
        self.eta = float(eta)
        self.beta = beta_0pn(self.f0_hz, self.mchirp_msun)
        self.phase_model = TaylorT4PhaseModel(
            f0_hz=self.f0_hz,
            mchirp_msun=self.mchirp_msun,
            eta=self.eta,
            t_max=float(t_max),
        )

    def tracks(self, t):
        track_0pn = make_0pn_track(
            t,
            f0_hz=self.f0_hz,
            mchirp_msun=self.mchirp_msun,
            beta=self.beta,
        )
        phase_35pn = np.asarray(self.phase_model.phase(t), dtype=float)
        frequency_35pn = np.asarray(self.phase_model.frequency(t), dtype=float)
        track_35pn = SignalTrack(
            t=t,
            frequency=frequency_35pn,
            phase=phase_35pn,
            signal=complex_signal_from_phase(phase_35pn),
        )
        return track_0pn, track_35pn


class ConstantFrequencyComparisonWaveformModel(PNComparisonWaveformModel):
    """Reusable constant-frequency and TaylorT4 3.5PN waveform generator."""

    def tracks(self, t):
        phase_constant = 2.0 * np.pi * self.f0_hz * t
        track_constant = SignalTrack(
            t=t,
            frequency=np.full_like(t, self.f0_hz, dtype=float),
            phase=phase_constant,
            signal=complex_signal_from_phase(phase_constant),
        )
        phase_35pn = np.asarray(self.phase_model.phase(t), dtype=float)
        frequency_35pn = np.asarray(self.phase_model.frequency(t), dtype=float)
        track_35pn = SignalTrack(
            t=t,
            frequency=frequency_35pn,
            phase=phase_35pn,
            signal=complex_signal_from_phase(phase_35pn),
        )
        return track_constant, track_35pn


def time_samples(duration, sample_rate):
    if duration <= 0.0:
        raise ValueError("duration must be positive.")
    n_samples = max(8, int(np.ceil(duration * sample_rate)))
    dt = duration / n_samples
    return dt * np.arange(n_samples), dt


def track_to_real_strain(
    track,
    amplitude_frequency_power,
):
    """Return a restricted real chirp with optional Newtonian amplitude shape."""

    if amplitude_frequency_power == 0.0:
        amplitude = 1.0
    else:
        amplitude = (track.frequency / track.frequency[0]) ** amplitude_frequency_power
    return np.asarray(amplitude * np.real(track.signal), dtype=float)


def frequency_domain_inner_product(
    h1,
    h2,
    dt,
    noise,
    zero_pad_factor,
):
    """Return 4 int h1(f)^* h2(f) / S_n(f) df for finite sampled strains."""

    if h1.shape != h2.shape:
        raise ValueError("inner-product inputs must have the same shape.")

    n_samples = h1.size
    n_fft = max(n_samples, int(zero_pad_factor) * n_samples)
    frequency = np.fft.rfftfreq(n_fft, dt)
    h1_tilde = dt * np.fft.rfft(h1, n=n_fft)
    h2_tilde = dt * np.fft.rfft(h2, n=n_fft)

    band = (
        (frequency > 0.0)
        & (frequency >= noise.frequency[0])
        & (frequency <= noise.frequency[-1])
    )
    if not np.any(band):
        raise ValueError(
            "No FFT frequency bins overlap the ASD band. Increase sample_rate "
            "or chunk duration."
        )

    df = frequency[1] - frequency[0]
    psd = noise.psd_at(frequency[band])
    return 4.0 * df * np.sum(np.conj(h1_tilde[band]) * h2_tilde[band] / psd)


def waveform_mismatch_at_duration(
    duration,
    waveform_model,
    noise,
    config,
):
    t, dt = time_samples(duration, config.sample_rate)
    track_template, track_35pn = waveform_model.tracks(t)
    h_template = track_to_real_strain(
        track_template,
        config.amplitude_frequency_power,
    )
    h_35pn = track_to_real_strain(track_35pn, config.amplitude_frequency_power)

    template_norm = frequency_domain_inner_product(
        h_template, h_template, dt, noise, config.zero_pad_factor
    ).real
    target_norm = frequency_domain_inner_product(
        h_35pn, h_35pn, dt, noise, config.zero_pad_factor
    ).real
    cross_term = frequency_domain_inner_product(
        h_template, h_35pn, dt, noise, config.zero_pad_factor
    )
    if template_norm <= 0.0 or target_norm <= 0.0:
        raise ValueError("Waveform norm is not positive.")

    match = np.abs(cross_term) / np.sqrt(template_norm * target_norm)
    match = float(np.clip(match, 0.0, 1.0))
    return 1.0 - match


def find_allowed_duration(
    config,
    noise,
    model_class=PNComparisonWaveformModel,
):
    waveform_model = model_class(
        config.f_max,
        config.mchirp_max,
        config.eta,
        config.search_time_max,
    )

    def residual(duration):
        return (
            waveform_mismatch_at_duration(duration, waveform_model, noise, config)
            - config.max_mismatch
        )

    lower = max(1.0e-3, 8.0 / config.sample_rate)
    if residual(lower) >= 0.0:
        return lower, waveform_model

    upper = config.search_time_max
    if residual(upper) < 0.0:
        raise RuntimeError(
            f"search_time_max does not reach the requested mismatch threshold "
            f"for {model_class.__name__}"
        )

    return (
        float(brentq(residual, lower, upper, rtol=1.0e-8, xtol=1.0e-8)),
        waveform_model,
    )


def validate_grid(
    duration,
    config,
    noise,
):
    f0_values = np.linspace(config.f_min, config.f_max, config.n_f0_validate)
    mchirp_values = np.geomspace(
        config.mchirp_min, config.mchirp_max, config.n_mchirp_validate
    )
    mismatches = np.array(
        [
            [
                waveform_mismatch_at_duration(
                    duration,
                    PNComparisonWaveformModel(f0, mchirp, config.eta, duration),
                    noise,
                    config,
                )
                for f0 in f0_values
            ]
            for mchirp in mchirp_values
        ],
        dtype=float,
    )
    return f0_values, mchirp_values, mismatches


def duration_floor(config):
    return max(1.0e-3, MIN_SAMPLES / config.sample_rate)


def mismatch_curve(
    waveform_model,
    config,
    noise,
):
    durations = np.linspace(
        duration_floor(config),
        config.search_time_max,
        config.n_time_plot,
    )
    mismatches = np.array(
        [
            waveform_mismatch_at_duration(
                duration,
                waveform_model,
                noise,
                config,
            )
            for duration in durations
        ]
    )
    return durations, mismatches


def grid_maximum(
    f0_values,
    mchirp_values,
    mismatches,
):
    max_index = np.unravel_index(np.argmax(mismatches), mismatches.shape)
    return (
        float(mismatches[max_index]),
        float(f0_values[max_index[1]]),
        float(mchirp_values[max_index[0]]),
    )


def plot_mismatch(
    durations,
    mismatches,
    allowed_duration,
    config,
):
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(durations, mismatches, lw=1.5)
    ax.axhline(config.max_mismatch, color="black", ls="--", lw=1.0)
    ax.axvline(allowed_duration, color="tab:red", ls=":", lw=1.3)
    ax.set(
        xlabel="Chunk duration [s]",
        ylabel="GW waveform mismatch",
        title=(
            rf"0PN vs 3.5PN, $f_0={config.f_max:g}$ Hz, "
            rf"$M_c={config.mchirp_max:g}M_\odot$"
        ),
    )
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def add_config_argument(
    parser,
    name,
    arg_type,
    **kwargs,
):
    parser.add_argument(
        f"--{name.replace('_', '-')}",
        type=arg_type,
        default=getattr(ZeroPNMismatchConfig, name),
        **kwargs,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Find the longest coherent chunk such that the frequency-domain "
            "0PN-vs-3.5PN GW waveform mismatch stays below the requested "
            "threshold."
        )
    )
    for name in (
        "f_min",
        "f_max",
        "mchirp_min",
        "mchirp_max",
        "eta",
        "max_mismatch",
        "search_time_max",
        "sample_rate",
        "n_time_plot",
        "n_f0_validate",
        "n_mchirp_validate",
    ):
        add_config_argument(parser, name, type(getattr(ZeroPNMismatchConfig, name)))

    add_config_argument(parser, "zero_pad_factor", int)
    add_config_argument(
        parser,
        "amplitude_frequency_power",
        float,
        help=(
            "Power of instantaneous frequency used in the restricted "
            "time-domain amplitude. Use 0 for unit-amplitude phase-only "
            "waveforms."
        ),
    )
    parser.add_argument(
        "--max-power-loss",
        dest="max_mismatch",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def config_from_args(args):
    return ZeroPNMismatchConfig(
        **{field: getattr(args, field) for field in CONFIG_FIELDS}
    )


def print_summary(
    config,
    asd_path,
    output_path,
    worst_waveform_model,
    allowed_duration,
    constant_frequency_allowed_duration,
    max_grid_mismatch,
    max_f0,
    max_mchirp,
):
    lines = [
        "0PN-vs-3.5PN frequency-domain mismatch limit",
        f"f0 range: {config.f_min:g} Hz to {config.f_max:g} Hz",
        (
            f"Mc range: {config.mchirp_min:.6e} Msun to "
            f"{config.mchirp_max:.6e} Msun"
        ),
        f"eta: {config.eta:g}",
        f"max_mismatch: {config.max_mismatch:.12g}",
        f"sample_rate: {config.sample_rate:.12g} Hz",
        f"zero_pad_factor: {config.zero_pad_factor:d}",
        f"amplitude_frequency_power: {config.amplitude_frequency_power:.12g}",
        f"asd: {asd_path}",
        f"worst-corner beta: {worst_waveform_model.beta:.12e}",
        f"0PN allowed T_chunk: {allowed_duration:.12f} s",
        (
            "constant-frequency allowed T_chunk: "
            f"{constant_frequency_allowed_duration:.12f} s"
        ),
        (
            "validation-grid maximum: "
            f"mismatch={max_grid_mismatch:.12e} at f0={max_f0:g} Hz, "
            f"Mc={max_mchirp:.6e} Msun"
        ),
        f"plot: {output_path}",
    ]
    print("\n".join(lines))


def main():
    args = parse_args()
    config = config_from_args(args)
    noise = NoiseCurve.from_asd_file(args.asd)

    allowed_duration, worst_waveform_model = find_allowed_duration(config, noise)
    constant_frequency_allowed_duration, _ = find_allowed_duration(
        config,
        noise,
        ConstantFrequencyComparisonWaveformModel,
    )
    durations, mismatches = mismatch_curve(worst_waveform_model, config, noise)
    f0_values, mchirp_values, grid_mismatches = validate_grid(
        allowed_duration,
        config,
        noise,
    )
    max_grid_mismatch, max_f0, max_mchirp = grid_maximum(
        f0_values,
        mchirp_values,
        grid_mismatches,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_mismatch(durations, mismatches, allowed_duration, config)
    fig.savefig(args.output, dpi=220)

    print_summary(
        config,
        args.asd,
        args.output,
        worst_waveform_model,
        allowed_duration,
        constant_frequency_allowed_duration,
        max_grid_mismatch,
        max_f0,
        max_mchirp,
    )


if __name__ == "__main__":
    main()
