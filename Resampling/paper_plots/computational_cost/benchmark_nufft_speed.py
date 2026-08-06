#!/usr/bin/env python3
"""Benchmark the prepared NUFFT as a function of radix-2 array size.

Signal generation and FINUFFT planning are deliberately kept outside the
timed region.  The reported throughput counts all input samples in the batch.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parent
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from resampler import Resampler, clear_plan_cache


G = 6.67430e-11
C = 299792458.0
MSUN = 2.0e30

MC_MSUN = 1e-2
F0_HZ = 20.0
SAMPLE_RATE_HZ = 512.0
EPS = 1e-3
THREADS = 4
N_TRANSFORMS = 2
MIN_POWER = 18
MAX_POWER = 22
REPEATS = 7
ITERATIONS = 10


def pbh_beta(f0_hz, mc_msun):
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * (G / C**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
        * (mc_msun * MSUN) ** (5.0 / 3.0)
    )


def make_data(n_samples, n_transforms):
    """Make the same PBH-like chirp and resampled time used in the comparison."""
    beta = pbh_beta(F0_HZ, MC_MSUN)
    t = np.arange(n_samples, dtype=np.float64) / SAMPLE_RATE_HZ
    chirp_factor = 1.0 - (8.0 / 3.0) * beta * t
    if chirp_factor[-1] <= 0.0:
        raise ValueError("The largest array extends beyond the coalescence time.")

    phase = -6.0 * np.pi / 5.0 * F0_HZ * chirp_factor ** (5.0 / 8.0) / beta
    signal = np.exp(1j * (phase - phase[0])).astype(np.complex64)
    tau = -(3.0 / (5.0 * beta)) * chirp_factor ** (5.0 / 8.0)
    tau -= tau[0]

    if n_transforms == 1:
        return signal, tau

    offsets = np.exp(0.73j * np.arange(n_transforms)).astype(np.complex64)
    signals = np.ascontiguousarray(offsets[:, None] * signal[None, :])
    return signals, tau


def benchmark(min_power, max_power, repeats, iterations, threads, n_transforms):
    rows = []
    for power in range(min_power, max_power + 1):
        n_samples = 2**power
        signals, tau = make_data(n_samples, n_transforms)

        clear_plan_cache()
        resampler = Resampler(
            nthreads=threads,
            eps=EPS,
            precision="single",
            upsampfac=1.25,
            fftw_measure=False,
            n_modes=n_samples,
            allow_two_transforms=n_transforms > 1,
        )
        resampler.timeseries = signals
        resampler.resampled_time = tau
        resampler.prepare_nufft()

        # Warm up the plan before measuring repeated executions.
        resampler.nufft_prepared()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(iterations):
                resampler.nufft_prepared()
            times.append((time.perf_counter() - start) / iterations)

        seconds = float(np.median(times))
        total_samples = n_transforms * n_samples
        throughput = total_samples / seconds
        rows.append(
            {
                "power": power,
                "n_samples": n_samples,
                "n_transforms": n_transforms,
                "total_input_samples": total_samples,
                "iterations_per_repeat": iterations,
                "median_seconds": seconds,
                "input_samples_per_second": throughput,
            }
        )
        print(
            f"2**{power:02d} = {n_samples:>8,d} samples: "
            f"{seconds:.4g} s, {throughput:.4g} input samples/s"
        )
    return rows


def save_results(rows):
    output_dir = SCRIPT_DIR / "figs"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "nufft_speed.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(
        [row["n_samples"] for row in rows],
        [row["input_samples_per_second"] for row in rows],
        marker="o",
    )
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Input samples per transform")
    ax.set_ylabel("NUFFT throughput [input samples / s]")
    ax.grid(True, alpha=0.25)

    plot_path = output_dir / "nufft_speed.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--min-power", type=int, default=MIN_POWER)
    parser.add_argument("--max-power", type=int, default=MAX_POWER)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--iterations", type=int, default=ITERATIONS)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument("--n-transforms", type=int, default=N_TRANSFORMS)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.min_power < 1 or args.max_power < args.min_power:
        raise ValueError("Require 1 <= min-power <= max-power.")
    if min(args.repeats, args.iterations, args.threads, args.n_transforms) < 1:
        raise ValueError(
            "repeats, iterations, threads, and n-transforms must be positive."
        )

    rows = benchmark(
        args.min_power,
        args.max_power,
        args.repeats,
        args.iterations,
        args.threads,
        args.n_transforms,
    )
    save_results(rows)


if __name__ == "__main__":
    main()
