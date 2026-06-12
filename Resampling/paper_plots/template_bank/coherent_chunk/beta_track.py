"""Count beta templates from a dephasing-limited beta spacing.

Default parameters:

* beta range: 1e-8 to 1e-3
* coherent chunk duration: 30 s
* f_min: 20 Hz
* phase threshold: pi/2 rad

The spacing formula is signed for the default range, so the bank uses its
magnitude as the adjacent-template spacing.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "beta_track.png"


@dataclass(frozen=True)
class BetaTrackConfig:
    beta_min: float = 1.0e-8
    beta_max: float = 1.0e-3
    t_chunk: float = 30.0
    f_min: float = 20.0
    delta_phi_thresh: float = np.pi / 2.0


def signed_delta_beta(beta: float, config: BetaTrackConfig) -> float:
    """Return the signed dephasing-limited delta beta from the requested formula."""
    beta_ld = np.longdouble(beta)
    t_chunk = np.longdouble(config.t_chunk)
    f_min = np.longdouble(config.f_min)
    delta_phi_thresh = np.longdouble(config.delta_phi_thresh)

    bracket = np.longdouble(1.0) - np.longdouble(8.0 / 3.0) * beta_ld * t_chunk
    if bracket <= 0.0:
        raise ValueError(
            f"beta={beta:.6e} reaches the 0PN singularity inside T_chunk={config.t_chunk:g} s"
        )

    denominator = (
        (np.longdouble(-1.0) + beta_ld * t_chunk)
        / bracket ** np.longdouble(3.0 / 8.0)
        + np.longdouble(1.0)
    )
    if denominator == 0.0:
        raise ZeroDivisionError(f"delta-beta denominator vanished at beta={beta:.6e}")

    numerator = (
        np.longdouble(5.0)
        * beta_ld**2
        * delta_phi_thresh
        / (np.longdouble(6.0) * np.longdouble(np.pi) * f_min)
    )
    return float(numerator / denominator)


def delta_beta_spacing(beta: float, config: BetaTrackConfig) -> float:
    """Positive adjacent-template spacing."""
    return abs(signed_delta_beta(beta, config))


def build_beta_track(config: BetaTrackConfig) -> np.ndarray:
    if config.beta_min <= 0.0:
        raise ValueError("beta_min must be positive")
    if config.beta_max <= config.beta_min:
        raise ValueError("beta_max must be larger than beta_min")

    betas = [float(config.beta_min)]
    beta = float(config.beta_min)

    while beta < config.beta_max:
        step = delta_beta_spacing(beta, config)
        if not np.isfinite(step) or step <= 0.0:
            raise RuntimeError(f"invalid beta spacing {step} at beta={beta:.6e}")

        beta = min(beta + step, config.beta_max)
        betas.append(beta)

    return np.asarray(betas)


def plot_beta_track(betas: np.ndarray, config: BetaTrackConfig):
    template_numbers = np.arange(betas.size)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot(template_numbers, betas, marker="o", ms=4.0, lw=1.4)
    ax.set_yscale("log")
    ax.set_xlabel("Template number")
    ax.set_ylabel(r"$\beta$")
    ax.set_title(
        rf"$T_{{chunk}}={config.t_chunk:g}$ s, "
        rf"$f_{{min}}={config.f_min:g}$ Hz, "
        rf"$\Delta\phi_{{thresh}}={config.delta_phi_thresh:.3g}$ rad"
    )
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count beta templates using the pi/2 endpoint-dephasing spacing."
    )
    parser.add_argument("--beta-min", type=float, default=BetaTrackConfig.beta_min)
    parser.add_argument("--beta-max", type=float, default=BetaTrackConfig.beta_max)
    parser.add_argument("--t-chunk", type=float, default=BetaTrackConfig.t_chunk)
    parser.add_argument("--f-min", type=float, default=BetaTrackConfig.f_min)
    parser.add_argument(
        "--delta-phi-thresh",
        type=float,
        default=BetaTrackConfig.delta_phi_thresh,
        help="phase threshold in radians",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BetaTrackConfig(
        beta_min=args.beta_min,
        beta_max=args.beta_max,
        t_chunk=args.t_chunk,
        f_min=args.f_min,
        delta_phi_thresh=args.delta_phi_thresh,
    )
    betas = build_beta_track(config)
    signed_steps = np.array([signed_delta_beta(beta, config) for beta in betas[:-1]])
    spacings = np.abs(signed_steps)
    fig = plot_beta_track(betas, config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=220)

    print("Beta-template track")
    print(f"beta range: {config.beta_min:.6e} to {config.beta_max:.6e}")
    print(f"T_chunk: {config.t_chunk:g} s")
    print(f"f_min: {config.f_min:g} Hz")
    print(f"delta_phi_thresh: {config.delta_phi_thresh:.12g} rad")
    print(f"templates required: {betas.size}")
    print(f"intervals: {betas.size - 1}")
    print(f"first spacing: {spacings[0]:.12e}")
    print(f"last full spacing: {spacings[-1]:.12e}")
    print(f"signed spacing range: {signed_steps.min():.12e} to {signed_steps.max():.12e}")
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
