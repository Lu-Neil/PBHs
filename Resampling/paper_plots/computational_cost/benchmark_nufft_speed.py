#!/usr/bin/env python3
"""Benchmark a prepared NUFFT on a 2D batch of 32-second segments.

Signal generation and FINUFFT planning are deliberately kept outside the
timed region.  Each row of the input batch is one NUFFT iteration, and the
reported throughput counts all input samples in the batch.
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path


CPU_AFFINITY = (0, 1, 2, 3)
os.sched_setaffinity(0, CPU_AFFINITY)

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
F0_HZ = 40.0
SAMPLE_RATE_HZ = 256.
DURATION_S = 32.0
EPS = 1e-3
THREADS = 4
REPEATS = 15
ITERATIONS = 1000


def pbh_beta(f0_hz, mc_msun):
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * (G / C**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
        * (mc_msun * MSUN) ** (5.0 / 3.0)
    )


def require_power_of_two(n_samples):
    if n_samples <= 0 or n_samples & (n_samples - 1):
        raise ValueError(
            "duration * sample_rate must be a power of two; "
            f"got {n_samples} samples."
        )


def make_data(n_samples, iterations, sample_rate_hz):
    """Make a 2D batch of PBH-like chirps and their resampled time."""
    beta = pbh_beta(F0_HZ, MC_MSUN)
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    chirp_factor = 1.0 - (8.0 / 3.0) * beta * t
    if chirp_factor[-1] <= 0.0:
        raise ValueError("The segment extends beyond the coalescence time.")

    phase = -6.0 * np.pi / 5.0 * F0_HZ * chirp_factor ** (5.0 / 8.0) / beta
    signal = np.exp(1j * (phase - phase[0])).astype(np.complex64)
    tau = -(3.0 / (5.0 * beta)) * chirp_factor ** (5.0 / 8.0)
    tau -= tau[0]

    offsets = np.exp(0.73j * np.arange(iterations)).astype(np.complex64)
    signals = np.ascontiguousarray(offsets[:, None] * signal[None, :])
    return signals, tau


def benchmark(duration_s, sample_rate_hz, repeats, iterations, threads):
    exact_n_samples = duration_s * sample_rate_hz
    n_samples = int(np.rint(exact_n_samples))
    if not np.isclose(n_samples, exact_n_samples, rtol=0.0, atol=1e-9):
        raise ValueError(
            "duration * sample_rate must be an integer; "
            f"got {exact_n_samples}."
        )
    require_power_of_two(n_samples)
    signals, tau = make_data(n_samples, iterations, sample_rate_hz)

    clear_plan_cache()
    resampler = Resampler(
        nthreads=threads,
        eps=EPS,
        precision="single",
        upsampfac=1.25,
        fftw_measure=False,
        n_modes=n_samples,
        allow_two_transforms=True,
    )
    resampler.timeseries = signals
    resampler.resampled_time = tau
    resampler.prepare_nufft()

    # Warm up the plan before measuring repeated batched executions.
    resampler.nufft_prepared()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        resampler.nufft_prepared()
        times.append(time.perf_counter() - start)

    batch_seconds = float(np.median(times))
    total_samples = iterations * n_samples
    throughput = total_samples / batch_seconds
    segments_per_second = iterations / batch_seconds
    row = {
        "duration_s": duration_s,
        "sample_rate_hz": sample_rate_hz,
        "power": n_samples.bit_length() - 1,
        "samples_per_segment": n_samples,
        "iterations_per_batch": iterations,
        "total_input_samples": total_samples,
        "repeats": repeats,
        "median_batch_seconds": batch_seconds,
        "median_seconds_per_iteration": batch_seconds / iterations,
        "segments_per_second": segments_per_second,
        "input_samples_per_second": throughput,
    }
    print(
        f"{duration_s:g} s segments: 2**{row['power']} = {n_samples:,d} "
        f"samples/segment, {iterations} iterations/batch"
    )
    print(
        f"Median batch time: {batch_seconds:.4g} s; "
        f"{batch_seconds / iterations:.4g} s/iteration; "
        f"{throughput:.4g} input samples/s"
    )
    print(f"Segments/s: {segments_per_second:.4g}")
    return row


def save_results(row):
    output_dir = SCRIPT_DIR / "figs"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "nufft_speed.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)

    plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig, ax = plt.subplots(constrained_layout=True)
    ax.bar(
        [f"{row['duration_s']:g} s\n$2^{{{row['power']}}}$ samples"],
        [row["input_samples_per_second"]],
    )
    ax.set_ylabel("NUFFT throughput [input samples / s]")
    ax.grid(True, axis="y", alpha=0.25)

    plot_path = output_dir / "nufft_speed.png"
    fig.savefig(plot_path, dpi=220)
    plt.close(fig)
    print(f"Saved {csv_path}")
    print(f"Saved {plot_path}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=DURATION_S)
    parser.add_argument("--sample-rate", type=float, default=SAMPLE_RATE_HZ)
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--iterations",
        type=int,
        default=ITERATIONS,
        help="Number of simultaneous transforms in the 2D input batch.",
    )
    parser.add_argument("--threads", type=int, default=THREADS)
    return parser.parse_args()


def main():
    args = parse_args()
    if min(args.duration, args.sample_rate) <= 0:
        raise ValueError("duration and sample-rate must be positive.")
    if min(args.repeats, args.iterations, args.threads) < 1:
        raise ValueError("repeats, iterations, and threads must be positive.")

    cpu_affinity = ",".join(map(str, CPU_AFFINITY))
    print(f"CPU affinity: {cpu_affinity}")
    row = benchmark(
        args.duration,
        args.sample_rate,
        args.repeats,
        args.iterations,
        args.threads,
    )
    row["cpu_affinity"] = cpu_affinity
    save_results(row)


if __name__ == "__main__":
    main()
