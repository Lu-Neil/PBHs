"""Build a beta bank from a coherent frequency-domain waveform mismatch.

Each adjacent beta interval is chosen so that two finite-duration 0PN waveforms
have no more than the requested ASD-weighted mismatch under the standard
frequency-domain inner product.
"""

import argparse
import csv
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
OUTPUT_PATH = SCRIPT_DIR / "figs" / "beta_mismatch.png"
BANK_OUTPUT_PATH = SCRIPT_DIR / "figs" / "beta_mismatch_bank.csv"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from signal_generators import NoiseCurve, beta_0pn, make_0pn_track  # noqa: E402


class BetaMismatchConfig:
    f_min = 40.0
    f_max = 60.0
    mchirp_min = 5.0e-4
    mchirp_max = 1.0e-1
    t_chunk = 30.0
    max_mismatch = 0.05
    sample_rate = 512.0
    zero_pad_factor = 4
    amplitude_frequency_power = 0.0

    def __init__(self, **kwargs):
        for name in CONFIG_FIELDS:
            setattr(self, name, kwargs.pop(name, getattr(type(self), name)))
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unknown config fields: {unknown}")

    @property
    def beta_min(self):
        return float(np.min(self.beta_corners()))

    @property
    def beta_max(self):
        return float(np.max(self.beta_corners()))

    @property
    def f_ref(self):
        return self.f_max

    def beta_corners(self):
        return np.array(
            [
                beta_0pn(f, mchirp)
                for f in (self.f_min, self.f_max)
                for mchirp in (self.mchirp_min, self.mchirp_max)
            ],
            dtype=float,
        )


CONFIG_FIELDS = (
    "f_min",
    "f_max",
    "mchirp_min",
    "mchirp_max",
    "t_chunk",
    "max_mismatch",
    "sample_rate",
    "zero_pad_factor",
    "amplitude_frequency_power",
)


def time_samples(config):
    if config.t_chunk <= 0.0:
        raise ValueError("t_chunk must be positive.")
    if config.sample_rate <= 0.0:
        raise ValueError("sample_rate must be positive.")
    n_samples = max(8, int(np.ceil(config.t_chunk * config.sample_rate)))
    dt = config.t_chunk / n_samples
    return dt * np.arange(n_samples), dt


def beta_waveform(beta, config, t):
    track = make_0pn_track(
        t,
        f0_hz=config.f_ref,
        mchirp_msun=None,
        beta=beta,
    )
    if config.amplitude_frequency_power == 0.0:
        amplitude = 1.0
    else:
        amplitude = (track.frequency / track.frequency[0]) ** (
            config.amplitude_frequency_power
        )
    return np.asarray(amplitude * np.real(track.signal), dtype=float)


def frequency_domain_inner_product(h1, h2, dt, noise, zero_pad_factor):
    """Return 4 int h1(f)^* h2(f) / S_n(f) df for sampled real strains."""

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


def waveform_mismatch(beta_signal, beta_template, config, noise, t, dt):
    h_signal = beta_waveform(beta_signal, config, t)
    h_template = beta_waveform(beta_template, config, t)

    hss = frequency_domain_inner_product(
        h_signal, h_signal, dt, noise, config.zero_pad_factor
    ).real
    htt = frequency_domain_inner_product(
        h_template, h_template, dt, noise, config.zero_pad_factor
    ).real
    hst = frequency_domain_inner_product(
        h_signal, h_template, dt, noise, config.zero_pad_factor
    )
    if hss <= 0.0 or htt <= 0.0:
        raise ValueError("Waveform norm is not positive.")

    match = np.abs(hst) / np.sqrt(hss * htt)
    match = float(np.clip(match, 0.0, 1.0))
    return 1.0 - match


def interval_mismatch(beta_low, beta_high, config, noise, t, dt):
    return waveform_mismatch(beta_low, beta_high, config, noise, t, dt)


def max_delta_beta(beta, config, noise, t, dt):
    singular_beta = 3.0 / (8.0 * config.t_chunk)
    upper = min(config.beta_max - beta, 0.999999 * singular_beta - beta)
    if upper <= 0.0:
        raise ValueError(f"no positive beta spacing is possible at beta={beta:.6e}")

    def residual(delta_beta):
        return (
            interval_mismatch(beta, beta + delta_beta, config, noise, t, dt)
            - config.max_mismatch
        )

    if residual(upper) <= 0.0:
        return upper

    lo = 0.0
    hi = upper
    while residual(hi) > 0.0:
        hi *= 0.5
        if hi <= np.finfo(float).eps * max(1.0, beta):
            raise RuntimeError(f"could not bracket beta spacing at beta={beta:.6e}")

    return float(brentq(residual, hi, upper, rtol=1.0e-11, xtol=1.0e-16))


def build_beta_bank(config, noise):
    if config.f_min <= 0.0:
        raise ValueError("f_min must be positive")
    if config.f_max <= config.f_min:
        raise ValueError("f_max must be larger than f_min")
    if config.mchirp_min <= 0.0:
        raise ValueError("mchirp_min must be positive")
    if config.mchirp_max <= config.mchirp_min:
        raise ValueError("mchirp_max must be larger than mchirp_min")
    if config.beta_min <= 0.0:
        raise ValueError("computed beta_min must be positive")
    if config.beta_max <= config.beta_min:
        raise ValueError("computed beta_max must be larger than beta_min")
    if not 0.0 < config.max_mismatch < 1.0:
        raise ValueError("max_mismatch must be in the range (0, 1)")
    if config.zero_pad_factor < 1:
        raise ValueError("zero_pad_factor must be at least 1")

    t, dt = time_samples(config)
    betas = [float(config.beta_min)]
    beta = float(config.beta_min)

    while beta < config.beta_max:
        step = max_delta_beta(beta, config, noise, t, dt)
        if not np.isfinite(step) or step <= 0.0:
            raise RuntimeError(f"invalid beta spacing {step} at beta={beta:.6e}")
        beta = min(beta + step, config.beta_max)
        betas.append(beta)

    return np.asarray(betas)


def validate_bank(betas, config, noise):
    t, dt = time_samples(config)
    mismatches = np.empty(max(0, betas.size - 1), dtype=float)
    for i in range(mismatches.size):
        mismatches[i] = interval_mismatch(
            betas[i], betas[i + 1], config, noise, t, dt
        )
    return mismatches


def write_bank_csv(path, betas, mismatches):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["template_number", "beta", "next_delta_beta", "next_mismatch"]
        )
        for i, beta in enumerate(betas):
            if i < mismatches.size:
                writer.writerow([i, beta, betas[i + 1] - beta, mismatches[i]])
            else:
                writer.writerow([i, beta, "", ""])


def plot_beta_bank(betas, mismatches, config):
    fig, (ax_beta, ax_loss) = plt.subplots(
        2, 1, figsize=(7.4, 6.2), sharex=True, constrained_layout=True
    )
    template_number = np.arange(betas.size)
    ax_beta.plot(template_number, betas, marker="o", ms=3.2, lw=1.2)
    ax_beta.set_yscale("log")
    ax_beta.set_ylabel(r"$\beta$")
    ax_beta.set_title(
        rf"$T_{{chunk}}={config.t_chunk:g}$ s, "
        f"f={config.f_min:g}-{config.f_max:g} Hz, "
        f"Mc={config.mchirp_min:g}-{config.mchirp_max:g} Msun, "
        rf"max mismatch = {config.max_mismatch:.3g}"
    )
    ax_beta.grid(True, which="both", alpha=0.25)

    if mismatches.size:
        ax_loss.plot(template_number[:-1], mismatches, marker="o", ms=3.2, lw=1.2)
        ax_loss.axhline(config.max_mismatch, color="black", ls="--", lw=1.0)
    ax_loss.set_xlabel("Template number")
    ax_loss.set_ylabel("Adjacent mismatch")
    ax_loss.grid(True, alpha=0.25)
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Build a beta bank with an explicit ASD-weighted 0PN waveform "
            "mismatch bound."
        )
    )
    parser.add_argument("--f-min", type=float, default=BetaMismatchConfig.f_min)
    parser.add_argument("--f-max", type=float, default=BetaMismatchConfig.f_max)
    parser.add_argument(
        "--mchirp-min",
        "--mc-min",
        dest="mchirp_min",
        type=float,
        default=BetaMismatchConfig.mchirp_min,
    )
    parser.add_argument(
        "--mchirp-max",
        "--mc-max",
        dest="mchirp_max",
        type=float,
        default=BetaMismatchConfig.mchirp_max,
    )
    parser.add_argument("--t-chunk", type=float, default=BetaMismatchConfig.t_chunk)
    parser.add_argument(
        "--max-mismatch",
        type=float,
        default=BetaMismatchConfig.max_mismatch,
    )
    parser.add_argument(
        "--max-power-loss",
        dest="max_mismatch",
        type=float,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--sample-rate",
        type=float,
        default=BetaMismatchConfig.sample_rate,
    )
    parser.add_argument(
        "--zero-pad-factor",
        type=int,
        default=BetaMismatchConfig.zero_pad_factor,
    )
    parser.add_argument(
        "--amplitude-frequency-power",
        type=float,
        default=BetaMismatchConfig.amplitude_frequency_power,
        help=(
            "Power of instantaneous frequency used in the restricted "
            "time-domain amplitude before FFT. Use 0 for unit-amplitude "
            "phase-only waveforms."
        ),
    )
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--bank-output", type=Path, default=BANK_OUTPUT_PATH)
    return parser.parse_args()


def main():
    args = parse_args()
    config = BetaMismatchConfig(
        f_min=args.f_min,
        f_max=args.f_max,
        mchirp_min=args.mchirp_min,
        mchirp_max=args.mchirp_max,
        t_chunk=args.t_chunk,
        max_mismatch=args.max_mismatch,
        sample_rate=args.sample_rate,
        zero_pad_factor=args.zero_pad_factor,
        amplitude_frequency_power=args.amplitude_frequency_power,
    )
    noise = NoiseCurve.from_asd_file(args.asd)
    betas = build_beta_bank(config, noise)
    mismatches = validate_bank(betas, config, noise)
    if mismatches.size and mismatches.max() > config.max_mismatch * (1.0 + 1.0e-8):
        raise RuntimeError(
            f"bank exceeds mismatch bound: {mismatches.max():.12e} "
            f"> {config.max_mismatch:.12e}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_beta_bank(betas, mismatches, config)
    fig.savefig(args.output, dpi=220)
    write_bank_csv(args.bank_output, betas, mismatches)

    spacings = np.diff(betas)
    print("Beta mismatch bank")
    print(f"f range: {config.f_min:g} Hz to {config.f_max:g} Hz")
    print(
        f"Mc range: {config.mchirp_min:.6e} Msun to "
        f"{config.mchirp_max:.6e} Msun"
    )
    print(f"beta range: {config.beta_min:.6e} to {config.beta_max:.6e}")
    print(f"T_chunk: {config.t_chunk:g} s")
    print(f"mismatch reference frequency: {config.f_ref:g} Hz")
    print(f"max_mismatch: {config.max_mismatch:.12g}")
    print(f"sample_rate: {config.sample_rate:.12g} Hz")
    print(f"zero_pad_factor: {config.zero_pad_factor:d}")
    print(f"amplitude_frequency_power: {config.amplitude_frequency_power:.12g}")
    print(f"asd: {args.asd}")
    print(f"templates required: {betas.size}")
    print(f"intervals: {betas.size - 1}")
    if spacings.size:
        print(f"first spacing: {spacings[0]:.12e}")
        print(f"last spacing: {spacings[-1]:.12e}")
        print(f"spacing range: {spacings.min():.12e} to {spacings.max():.12e}")
    if mismatches.size:
        print(
            "validated adjacent mismatch range: "
            f"{mismatches.min():.12e} to {mismatches.max():.12e}"
        )
    print(f"bank: {args.bank_output}")
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
