"""Plot expected Gaussian-equivalent significance versus injection distance."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.stats import chi2


SCRIPT_DIR = Path(__file__).resolve().parent
DISTANCE_SENSITIVITY_DIR = SCRIPT_DIR.parent
PAPER_PLOTS_DIR = DISTANCE_SENSITIVITY_DIR.parent
OUTPUT_PATH = SCRIPT_DIR / "significance_vs_distance.png"

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity import semicoherent_sensitivity as sensitivity  # noqa: E402

plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")


DEFAULTS = {
    "f0": sensitivity.F_START,
    "mchirp": 1.0e-2,
    "f_end": sensitivity.F_END,
    "chunk_duration": sensitivity.CHUNK_DURATION,
    "lambda_threshold": sensitivity.LAMBDA_THRESHOLD,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan 20 distances around the semicoherent sensitivity distance and "
            "plot the expected Gaussian-equivalent significance."
        )
    )
    for name, default in DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default,
        )
    parser.add_argument("--n-distances", type=int, default=20)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def validate_args(args):
    if args.f0 <= 0.0:
        raise ValueError("f0 must be positive")
    if args.mchirp <= 0.0:
        raise ValueError("mchirp must be positive")
    if args.f_end <= args.f0:
        raise ValueError("f_end must be larger than f0")
    if args.chunk_duration <= 0.0:
        raise ValueError("chunk_duration must be positive")
    if args.lambda_threshold <= 0.0:
        raise ValueError("lambda_threshold must be positive")
    if args.n_distances < 2:
        raise ValueError("n_distances must be at least 2")


def semicoherent_distance(f0, mchirp, f_end, lambda_threshold, chunk_duration):
    beta = sensitivity.beta_calc(f0, mchirp)
    duration = min(
        sensitivity.time_to_frequency(f0, f_end, beta),
        sensitivity.MAX_OBS_TIME,
    )
    chunk_count = sensitivity.semicoherent_chunk_count_grid(duration, chunk_duration)
    coherent_distance_pc = (
        sensitivity.distance_sensitivity(duration, f0, mchirp, l=lambda_threshold)
        / sensitivity.PARSEC_M
    )
    distance_pc = sensitivity.apply_semicoherent_chunk_penalty(
        coherent_distance_pc,
        chunk_count,
    )
    return float(distance_pc), float(duration), int(chunk_count)


def gaussian_equivalent_significance(statistic, dof):
    log_p_value = chi2.logsf(statistic, dof)
    if np.isneginf(log_p_value):
        log_p_value = chi2_logsf_saddlepoint(statistic, dof)
    return -special.ndtri_exp(log_p_value)


def chi2_logsf_saddlepoint(statistic, dof):
    """Lugannani-Rice right-tail approximation for extreme chi-square values."""
    if statistic <= dof:
        return chi2.logsf(statistic, dof)

    saddlepoint = (statistic - dof) / (2.0 * statistic)
    cumulant = -0.5 * dof * math.log1p(-2.0 * saddlepoint)
    variance = 2.0 * dof / (1.0 - 2.0 * saddlepoint) ** 2
    signed_root = math.sqrt(2.0 * (saddlepoint * statistic - cumulant))
    scaled_root = saddlepoint * math.sqrt(variance)

    normal_logsf = special.log_ndtr(-signed_root)
    correction = 1.0 / signed_root - 1.0 / scaled_root
    correction_log = (
        -0.5 * signed_root**2
        - 0.5 * math.log(2.0 * math.pi)
        + math.log(correction)
    )
    correction_ratio = math.exp(correction_log - normal_logsf)
    if correction_ratio >= 1.0:
        return (
            -0.5 * signed_root**2
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(scaled_root)
        )
    return normal_logsf + math.log1p(-correction_ratio)


def expected_significance(distances_pc, semicoherent_distance_pc, lambda_threshold, n_chunks):
    null_mean = 2.0 * n_chunks
    dof = 2.0 * n_chunks
    signal_power = (
        lambda_threshold
        * np.sqrt(n_chunks)
        * (semicoherent_distance_pc / distances_pc) ** 2
    )
    statistic = null_mean + signal_power
    sigma = np.array(
        [gaussian_equivalent_significance(value, dof) for value in statistic]
    )
    return signal_power, statistic, sigma


def plot_significance(distances_pc, sigma, semicoherent_distance_pc, output):
    fig, ax = plt.subplots()
    ax.plot(distances_pc / semicoherent_distance_pc, sigma, marker="o", lw=1.5)
    ax.axvline(1.0, color="k", ls="--", lw=1.0, label="Semicoherent distance")
    ax.set_xlabel(r"Distance / $D_\mathrm{semi}$")
    ax.set_ylabel(r"Gaussian-equivalent significance [$\sigma$]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)

    distance_pc, duration, n_chunks = semicoherent_distance(
        args.f0,
        args.mchirp,
        args.f_end,
        args.lambda_threshold,
        args.chunk_duration,
    )
    distances_pc = np.linspace(0.5 * distance_pc, 2.0 * distance_pc, args.n_distances)
    signal_power, statistic, sigma = expected_significance(
        distances_pc,
        distance_pc,
        args.lambda_threshold,
        n_chunks,
    )

    plot_significance(distances_pc, sigma, distance_pc, args.output)

    print("-" * 9 + "Scan parameters" + "-" * 9)
    print(f"f0: {args.f0:g} Hz")
    print(f"f_end: {args.f_end:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(f"signal duration: {duration:.2f} s")
    print(f"chunk duration: {args.chunk_duration:g} s")
    print(f"chunks: {n_chunks}")
    print(f"semicoherent distance: {distance_pc:.6e} pc")
    print("-" * 9 + "Distance scan" + "-" * 9)
    for distance, power, value, significance in zip(
        distances_pc, signal_power, statistic, sigma
    ):
        print(
            f"{distance / distance_pc:5.2f} D_semi  "
            f"{distance:.6e} pc  "
            f"signal_power={power:.6e}  "
            f"statistic={value:.6e}  "
            f"sigma={significance:.6e}"
        )
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
