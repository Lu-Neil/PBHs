"""Compare compact convolution kernels in time and frequency.

The kernels use the normalized coordinate ``x = t / h``, where ``h`` is the
half-width of their compact support.  Their peak values are normalized to one:

* boxcar: ``1``
* Kaiser--Bessel: ``I_0(beta sqrt(1 - x**2)) / I_0(beta)``
* exponential of a semicircle (ES):
  ``exp(beta (sqrt(1 - x**2) - 1))``

Run with, for example,

    conda run -n PBH python resampling/convolution_kernels.py
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "figs" / "convolution_kernels.png"


def compact_kernels(x, beta):
    """Return peak-normalized boxcar, Kaiser--Bessel, and ES kernels."""

    x = np.asarray(x, dtype=float)
    inside = np.abs(x) <= 1.0
    semicircle = np.sqrt(np.clip(1.0 - x**2, 0.0, None))

    boxcar = inside.astype(float)
    kaiser_bessel = np.where(
        inside,
        np.i0(beta * semicircle) / np.i0(beta),
        0.0,
    )
    exponential_semicircle = np.where(
        inside,
        np.exp(beta * (semicircle - 1.0)),
        0.0,
    )
    return {
        "Boxcar": boxcar,
        "Kaiser–Bessel": kaiser_bessel,
        "Exponential of \na semicircle": exponential_semicircle,
    }


def continuous_fourier_magnitudes(x, kernels, frequencies, chunk_size=128):
    r"""Numerically evaluate ``|integral phi(x)e^{-2 pi i nu x} dx|``.

    The kernels are real and even, so a chunked cosine transform avoids a
    large complex work array.  Each result is normalized to unity at zero
    frequency.
    """

    transforms = {name: np.empty_like(frequencies) for name in kernels}
    for start in range(0, frequencies.size, chunk_size):
        stop = min(start + chunk_size, frequencies.size)
        phase = 2.0 * np.pi * np.outer(x, frequencies[start:stop])
        cosine = np.cos(phase)
        for name, values in kernels.items():
            transforms[name][start:stop] = np.abs(
                np.trapezoid(values[:, None] * cosine, x=x, axis=0)
            )

    for name, values in transforms.items():
        values /= values[0]
    return transforms


def make_figure(beta=8.0, max_frequency=8.0):
    """Build the time- and frequency-domain kernel comparison figure."""

    # Include samples just outside the support so compact-support boundaries
    # and the boxcar discontinuity are visible.
    x_support = np.linspace(-1.0, 1.0, 2**14 + 1)
    x_plot = np.concatenate(([-1.2, -1.0 - 1e-8], x_support, [1.0 + 1e-8, 1.2]))
    kernels_plot = compact_kernels(x_plot, beta)
    kernels_transform = compact_kernels(x_support, beta)

    frequencies = np.linspace(0.0, max_frequency, 1601)
    transforms = continuous_fourier_magnitudes(
        x_support,
        kernels_transform,
        frequencies,
    )

    fig, (ax_time, ax_frequency) = plt.subplots(
        2,
        1,
        figsize=(7.0, 8.0),
        constrained_layout=True,
    )

    for name, values in kernels_plot.items():
        ax_time.plot(x_plot, values, label=name)

    magnitude_floor = 1e-6
    for name, values in transforms.items():
        ax_frequency.plot(
            frequencies,
            np.maximum(values, magnitude_floor),
            label=name,
        )

    ax_time.set(
        xlabel=r"t",
        ylabel=r"$\phi(t)$",
        title="Time domain",
        xlim=(-1.2, 1.2),
        ylim=(-0.04, 1.05),
    )
    ax_time.axvline(-1.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax_time.axvline(1.0, color="0.75", lw=0.8, ls=":", zorder=0)
    ax_time.grid(alpha=0.25)
    ax_time.legend(frameon=True, loc='upper right')

    ax_frequency.set(
        xlabel=r"$f$",
        ylabel=r"$\tilde{\phi}(f)$",
        title="Frequency domain",
        xlim=(0.0, max_frequency),
        ylim=(magnitude_floor, 1.2),
    )
    ax_frequency.set_yscale("log")
    ax_frequency.grid(alpha=0.25)
    plt.tight_layout()

    # fig.suptitle(rf"Compact convolution kernels ($\beta={beta:g}$)")
    return fig


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot compact convolution kernels and their Fourier transforms."
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=8.0,
        help="Shape parameter for the Kaiser--Bessel and ES kernels (default: 8).",
    )
    parser.add_argument(
        "--max-frequency",
        type=float,
        default=8.0,
        help="Maximum normalized frequency shown (default: 8).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output image path (default: {DEFAULT_OUTPUT}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive.")
    if args.max_frequency <= 0.0:
        raise ValueError("--max-frequency must be positive.")

    plt.style.use(SCRIPT_DIR.parent / "paper.mplstyle")
    fig = make_figure(beta=args.beta, max_frequency=args.max_frequency)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)
    plt.close(fig)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
