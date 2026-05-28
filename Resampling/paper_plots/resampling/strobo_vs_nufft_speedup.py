from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyfftw


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Nov2025.resampler import Resampler, clear_plan_cache


G = 6.67430e-11
C = 299792458.0
MSUN = 2.0e30

MC_MSUN = 1e-2
F0_HZ = 20.0
F_NEW_HZ = 500.0
NUFFT_SAMPLES = 2**20
THREADS = 4
RATIO_COUNT = 17
RATIOS = tuple(float(ratio) for ratio in np.geomspace(2.0, 128.0, RATIO_COUNT))
REPEATS = 3


@dataclass(frozen=True)
class BenchmarkRow:
    f_ratio: float
    strobo_input_hz: float
    eps: float
    strobo_samples: int
    strobo_output_samples: int
    nufft_samples: int
    strobo_seconds: float
    nufft_seconds: float

    @property
    def nufft_speedup(self) -> float:
        return self.strobo_seconds / self.nufft_seconds


def pbh_beta(f0_hz: float, mc_msun: float) -> float:
    mc_si = mc_msun * MSUN
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * (G / C**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
        * mc_si ** (5.0 / 3.0)
    )


def build_signal(
    beta: float,
    *,
    f0_hz: float,
    n_samples: int,
    sample_rate_hz: float,
    dtype: np.dtype = np.complex64,
) -> tuple[np.ndarray, np.ndarray]:
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz

    chirp_factor = 1.0 - (8.0 / 3.0) * beta * t
    if np.any(chirp_factor <= 0.0):
        raise ValueError("Requested signal reaches coalescence during the observation.")

    phase = (
        -6.0
        * np.pi
        / 5.0
        * f0_hz
        * chirp_factor ** (5.0 / 8.0)
        / beta
    )
    signal = np.exp(1j * (phase - phase[0])).astype(dtype, copy=False)

    tau = -(3.0 / (5.0 * beta)) * chirp_factor ** (5.0 / 8.0)
    tau -= tau[0]
    return signal, tau


def strobo_resample(
    tau: np.ndarray,
    data: np.ndarray,
    *,
    output_rate_hz: float,
    target_samples: int,
) -> np.ndarray:
    scaled_tau = tau * output_rate_hz
    floor_tau = np.floor(scaled_tau)
    indices = np.nonzero(np.diff(floor_tau))[0]
    if indices.size < target_samples:
        raise ValueError(
            f"Strobo produced {indices.size} samples, fewer than requested {target_samples}."
        )
    return np.ascontiguousarray(data[indices[:target_samples]], dtype=np.complex64)


class FFTWPlanCache:
    def __init__(self, threads: int) -> None:
        self.threads = threads
        self._plans = {}

    def plan(self, n_samples: int):
        plan = self._plans.get(n_samples)
        if plan is None:
            in_array = pyfftw.empty_aligned(n_samples, dtype="complex64")
            out_array = pyfftw.empty_aligned(n_samples, dtype="complex64")
            fft_obj = pyfftw.FFTW(
                in_array,
                out_array,
                direction="FFTW_FORWARD",
                threads=self.threads,
                flags=("FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"),
            )
            plan = (in_array, out_array, fft_obj)
            self._plans[n_samples] = plan
        return plan

    def execute(self, data: np.ndarray) -> np.ndarray:
        in_array, out_array, fft_obj = self.plan(data.size)
        in_array[:] = data
        fft_obj()
        return out_array


def time_strobo_fftw(
    tau: np.ndarray,
    data: np.ndarray,
    *,
    output_rate_hz: float,
    target_samples: int,
    fftw_plans: FFTWPlanCache,
) -> tuple[float, int]:
    tic = time.perf_counter()
    strobo_data = strobo_resample(
        tau,
        data,
        output_rate_hz=output_rate_hz,
        target_samples=target_samples,
    )
    fft_out = fftw_plans.execute(strobo_data)
    _ = np.abs(fft_out).max()
    return time.perf_counter() - tic, strobo_data.size


def time_nufft(
    signal: np.ndarray,
    tau: np.ndarray,
    *,
    eps: float,
    threads: int,
) -> float:
    resampler = Resampler(
        nthreads=threads,
        eps=eps,
        precision="single",
        upsampfac=1.25,
        fftw_measure=False,
    )
    resampler.timeseries = signal
    resampler.resampled_time = tau
    tic = time.perf_counter()
    resampler.nufft()
    _ = np.abs(resampler.weights).max()
    return time.perf_counter() - tic


def median_time(values: list[float]) -> float:
    return float(np.median(np.asarray(values, dtype=float)))


def str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected a boolean value, got {value!r}.")


def is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1)) == 0


def run_benchmark(
    *,
    ratios: tuple[float, ...],
    repeats: int,
    nufft_samples: int,
    f_new_hz: float,
    threads: int,
) -> list[BenchmarkRow]:
    pyfftw.interfaces.cache.enable()
    if not is_power_of_two(nufft_samples):
        raise ValueError(f"nufft_samples must be a power of two, got {nufft_samples}.")

    beta = pbh_beta(F0_HZ, MC_MSUN)
    nufft_signal, nufft_tau = build_signal(
        beta,
        f0_hz=F0_HZ,
        n_samples=nufft_samples,
        sample_rate_hz=f_new_hz,
    )
    nufft_samples = nufft_signal.size
    fftw_plans = FFTWPlanCache(threads=threads)
    rows = []

    clear_plan_cache()
    for f_ratio in ratios:
        eps = 1.0 / f_ratio
        strobo_input_hz = f_new_hz * f_ratio
        strobo_input_samples = int(np.ceil(nufft_samples * f_ratio)) + 128
        strobo_signal, strobo_tau = build_signal(
            beta,
            f0_hz=F0_HZ,
            n_samples=strobo_input_samples,
            sample_rate_hz=strobo_input_hz,
        )

        # Warm both paths once. This builds FFTW and FINUFFT plans before timing.
        _, strobo_output_samples = time_strobo_fftw(
            strobo_tau,
            strobo_signal,
            output_rate_hz=f_new_hz,
            target_samples=nufft_samples,
            fftw_plans=fftw_plans,
        )
        time_nufft(nufft_signal, nufft_tau, eps=eps, threads=threads)

        strobo_times = []
        nufft_times = []
        for _ in range(repeats):
            strobo_time, strobo_output_samples = time_strobo_fftw(
                strobo_tau,
                strobo_signal,
                output_rate_hz=f_new_hz,
                target_samples=nufft_samples,
                fftw_plans=fftw_plans,
            )
            nufft_time = time_nufft(
                nufft_signal,
                nufft_tau,
                eps=eps,
                threads=threads,
            )
            strobo_times.append(strobo_time)
            nufft_times.append(nufft_time)

        rows.append(
            BenchmarkRow(
                f_ratio=f_ratio,
                strobo_input_hz=strobo_input_hz,
                eps=eps,
                strobo_samples=strobo_signal.size,
                strobo_output_samples=strobo_output_samples,
                nufft_samples=nufft_samples,
                strobo_seconds=median_time(strobo_times),
                nufft_seconds=median_time(nufft_times),
            )
        )

        print(
            f"f_ratio={f_ratio:>8.3g}: "
            f"strobo={rows[-1].strobo_seconds:.4f}s, "
            f"nufft={rows[-1].nufft_seconds:.4f}s, "
            f"speedup={rows[-1].nufft_speedup:.2f}x"
        )

    return rows


def save_csv(rows: list[BenchmarkRow], output_path: Path) -> None:
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "f_ratio",
                "strobo_input_hz",
                "eps",
                "strobo_samples",
                "strobo_output_samples",
                "nufft_samples",
                "strobo_seconds",
                "nufft_seconds",
                "nufft_speedup",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "f_ratio": row.f_ratio,
                    "strobo_input_hz": row.strobo_input_hz,
                    "eps": row.eps,
                    "strobo_samples": row.strobo_samples,
                    "strobo_output_samples": row.strobo_output_samples,
                    "nufft_samples": row.nufft_samples,
                    "strobo_seconds": row.strobo_seconds,
                    "nufft_seconds": row.nufft_seconds,
                    "nufft_speedup": row.nufft_speedup,
                }
            )


def load_csv(input_path: Path) -> list[BenchmarkRow]:
    rows = []
    with input_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(
                BenchmarkRow(
                    f_ratio=float(row["f_ratio"]),
                    strobo_input_hz=float(row["strobo_input_hz"]),
                    eps=float(row["eps"]),
                    strobo_samples=int(row["strobo_samples"]),
                    strobo_output_samples=int(row["strobo_output_samples"]),
                    nufft_samples=int(row["nufft_samples"]),
                    strobo_seconds=float(row["strobo_seconds"]),
                    nufft_seconds=float(row["nufft_seconds"]),
                )
            )
    if not rows:
        raise ValueError(f"No benchmark rows found in {input_path}.")
    return rows


def infer_f_new_hz(rows: list[BenchmarkRow]) -> float:
    f_new_values = np.asarray(
        [row.strobo_input_hz / row.f_ratio for row in rows], dtype=float
    )
    return float(np.median(f_new_values))


def save_plot(
    rows: list[BenchmarkRow],
    output_path: Path,
    *,
    nufft_samples: int,
    f_new_hz: float,
) -> None:
    ratios = np.asarray([row.f_ratio for row in rows], dtype=float)
    speedups = np.asarray([row.nufft_speedup for row in rows], dtype=float)

    fig, ax_speed = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    ax_speed.semilogx(ratios, speedups, "o-", color="tab:blue", lw=1.8)
    ax_speed.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax_speed.set_xlabel("Upsampling ratio")
    ax_speed.set_ylabel("NUFFT / strobo speedup")
    ax_speed.grid(True, alpha=0.25)
    # ax_speed.set_title(
    #     rf"$M_c={MC_MSUN:.0e}M_\odot$, $f_0={F0_HZ:g}$ Hz, "
    #     rf"$T_\mathrm{{obs}}={nufft_samples / f_new_hz:g}$ s, "
    #     rf"$N_\mathrm{{NUFFT}}=2^{{{int(np.log2(nufft_samples))}}}$"
    # )
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark stroboscopic pyFFTW resampling against the Nov2025 FINUFFT resampler."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=RATIOS,
        help="Strobo input/output sampling ratios. FINUFFT uses eps=1/ratio.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--nufft-samples", type=int, default=NUFFT_SAMPLES)
    parser.add_argument("--f-new", type=float, default=F_NEW_HZ)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--load",
        type=str_to_bool,
        nargs="?",
        const=True,
        default=False,
        help=(
            "Load benchmark rows from figs/strobo_vs_nufft_speedup.csv and only "
            "regenerate the plot. Accepts true/false; passing --load alone means true."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parent / "figs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "strobo_vs_nufft_speedup.csv"
    plot_path = output_dir / "strobo_vs_nufft_speedup.png"

    if args.load:
        rows = load_csv(csv_path)
        plot_nufft_samples = rows[0].nufft_samples
        plot_f_new_hz = infer_f_new_hz(rows)
        print(f"Loaded {csv_path}")
    else:
        rows = run_benchmark(
            ratios=tuple(args.ratios),
            repeats=args.repeats,
            nufft_samples=args.nufft_samples,
            f_new_hz=args.f_new,
            threads=args.threads,
        )
        save_csv(rows, csv_path)
        plot_nufft_samples = args.nufft_samples
        plot_f_new_hz = args.f_new
        print(f"Saved {csv_path}")

    save_plot(
        rows,
        plot_path,
        nufft_samples=plot_nufft_samples,
        f_new_hz=plot_f_new_hz,
    )

    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
