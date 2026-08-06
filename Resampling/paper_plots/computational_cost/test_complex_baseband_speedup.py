#!/usr/bin/env python3
"""Test the measured, end-to-end speedup from complex basebanding.

The baseband path heterodynes an analytic 0PN signal by
``exp(-2j*pi*f_center*tau_beta)``, low-pass filters and decimates it, then runs
the smaller NUFFT.  Timings distinguish the prepared NUFFT from the complete
mix/filter/decimate/NUFFT path.  Timing is deliberately reported rather than
asserted because the result is hardware and thread-count dependent.

Run with:

    conda run -n PBH python computational_cost/test_complex_baseband_speedup.py
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from fractions import Fraction
from pathlib import Path

import numpy as np
from scipy.signal import firwin, resample_poly


PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PAPER_PLOTS_DIR))

from resampler import Resampler, clear_plan_cache  # noqa: E402
from signal_generators import beta_0pn, tau_0pn  # noqa: E402


def filter_taps(up: int, down: int) -> np.ndarray:
    """Return reusable taps matching scipy's default polyphase filter."""

    scale = max(up, down)
    return firwin(20 * scale + 1, 1.0 / scale, window=("kaiser", 5.0))


def build_inputs(args):
    if not 0 < args.f_start < args.carrier < args.f_end:
        raise ValueError("Require 0 < f_start < carrier < f_end.")
    if args.raw_rate < 2 * args.f_end:
        raise ValueError("raw-rate must be at least 2*f_end.")
    if args.baseband_guard < 1:
        raise ValueError("baseband-guard must be at least one.")

    n_raw = round(args.duration * args.raw_rate)
    if not np.isclose(n_raw, args.duration * args.raw_rate):
        raise ValueError("duration*raw-rate must be an integer.")

    requested_rate = args.baseband_guard * (args.f_end - args.f_start)
    ratio = Fraction(requested_rate / args.raw_rate).limit_denominator(4096)
    up, down = ratio.numerator, ratio.denominator
    baseband_rate = args.raw_rate * up / down
    if not np.isclose(baseband_rate, requested_rate, rtol=1e-10):
        raise ValueError("Choose rates with a simpler rational ratio.")

    t = np.arange(n_raw) / args.raw_rate
    beta = beta_0pn(args.carrier, args.mchirp)
    tau = np.asarray(tau_0pn(t, beta))
    signal = np.exp(2j * np.pi * args.carrier * tau)
    center = (args.f_start + args.f_end) / 2
    phasor = np.exp(-2j * np.pi * center * tau)
    taps = filter_taps(up, down)

    n_base = (n_raw * up + down - 1) // down
    t_base = np.arange(n_base) / baseband_rate
    tau_base = np.asarray(tau_0pn(t_base, beta))
    return signal, tau, phasor, taps, up, down, baseband_rate, tau_base, center


def make_baseband(signal, phasor, taps, up, down):
    """Mix, low-pass filter, and decimate one coherent chunk."""

    return resample_poly(
        signal * phasor,
        up,
        down,
        window=taps,
        padtype="constant",
    )


def median_time(function, iterations, repeats):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter() - start) / iterations)
    return float(np.median(samples))


def exact_amplitude(signal, tau, frequency):
    return float(abs(np.mean(signal * np.exp(-2j * np.pi * frequency * tau))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark full-rate and complex-baseband NUFFT paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--f-start", type=float, default=40.0)
    parser.add_argument("--f-end", type=float, default=64.0)
    parser.add_argument("--carrier", type=float, default=50.0)
    parser.add_argument("--mchirp", type=float, default=1e-2)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--raw-rate", type=float, default=128.0)
    parser.add_argument(
        "--baseband-guard",
        type=float,
        default=1.25,
        help="Complex sample rate divided by searched bandwidth.",
    )
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--precision", choices=("single", "double"), default="double")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--upsampfac", type=float, default=1.25)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iterations < 1 or args.repeats < 1 or args.threads < 1:
        raise ValueError("iterations, repeats, and threads must be positive.")

    (
        signal,
        tau,
        phasor,
        taps,
        up,
        down,
        base_rate,
        tau_base,
        center,
    ) = build_inputs(args)
    base_signal = make_baseband(signal, phasor, taps, up, down)

    raw = Resampler(
        nthreads=args.threads,
        eps=args.eps,
        precision=args.precision,
        upsampfac=args.upsampfac,
    )
    base = Resampler(
        nthreads=args.threads,
        eps=args.eps,
        precision=args.precision,
        upsampfac=args.upsampfac,
    )
    raw.timeseries, raw.resampled_time = signal, tau
    base.timeseries, base.resampled_time = base_signal, tau_base

    # Warm both cached FINUFFT plans before measuring steady-state work.
    clear_plan_cache()
    raw.nufft()
    base.nufft()

    raw_peak = float(raw.freq_in_hz[np.argmax(raw.power)])
    base_peak = float(base.freq_in_hz[np.argmax(base.power)] + center)
    bin_tolerance = 1.5 * max(
        abs(raw.freq_in_hz[1] - raw.freq_in_hz[0]),
        abs(base.freq_in_hz[1] - base.freq_in_hz[0]),
    )
    assert abs(raw_peak - args.carrier) < bin_tolerance
    assert abs(base_peak - args.carrier) < bin_tolerance

    amplitude_ratio = exact_amplitude(
        base_signal, tau_base, args.carrier - center
    ) / exact_amplitude(signal, tau, args.carrier)
    assert abs(amplitude_ratio - 1) < 0.05

    def preprocess():
        return make_baseband(signal, phasor, taps, up, down)

    def end_to_end():
        base.timeseries = preprocess()
        base.nufft()

    raw_time = median_time(raw.nufft, args.iterations, args.repeats)
    base_time = median_time(base.nufft, args.iterations, args.repeats)
    prep_time = median_time(preprocess, args.iterations, args.repeats)
    total_time = median_time(end_to_end, args.iterations, args.repeats)

    transform_speedup = raw_time / base_time
    total_speedup = raw_time / total_time
    expected = (
        args.raw_rate * math.log(args.duration * args.raw_rate)
        / (base_rate * math.log(args.duration * base_rate))
    )

    print("Complex-baseband NUFFT speed test")
    print(f"  band and carrier:              {args.f_start:g}--{args.f_end:g} Hz, {args.carrier:g} Hz")
    print(f"  full/baseband rates:           {args.raw_rate:g}/{base_rate:g} Hz")
    print(f"  full/baseband samples:         {signal.size}/{base_signal.size}")
    print(f"  threads:                       {args.threads}")
    print(f"  recovered peaks:               {raw_peak:.6g}/{base_peak:.6g} Hz")
    print(f"  exact-frequency amplitude ratio: {amplitude_ratio:.6g}")
    print(f"  asymptotic predicted speedup:  {expected:.3f}x")
    print(f"  full-rate NUFFT:               {1e3 * raw_time:.6g} ms")
    print(f"  prepared baseband NUFFT:       {1e3 * base_time:.6g} ms")
    print(f"  baseband preprocessing:        {1e3 * prep_time:.6g} ms")
    print(f"  baseband end-to-end:           {1e3 * total_time:.6g} ms")
    print(f"  NUFFT-only measured speedup:   {transform_speedup:.3f}x")
    print(f"  end-to-end measured speedup:   {total_speedup:.3f}x")

    if transform_speedup <= 1:
        conclusion = "fixed FINUFFT/thread overhead dominates the smaller transform"
    elif total_speedup <= 1:
        conclusion = "baseband preprocessing removes the transform speedup"
    else:
        conclusion = "complex baseband is faster end-to-end"
    print(f"  result: {conclusion}")


if __name__ == "__main__":
    main()
