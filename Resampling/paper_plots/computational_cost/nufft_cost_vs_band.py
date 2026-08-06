#!/usr/bin/env python3
"""Estimate NUFFT search cost as the semicoherent band edges vary.

At fixed FINUFFT tolerance, the leading cost model is

    C_NUFFT ~ N_beta T_obs f_samp log(T_coh f_samp).

There are two useful conventions for mapping the searched band to ``f_samp``:

``real-nyquist``
    A raw real-valued strain channel is downsampled only as far as Nyquist
    permits, so ``f_samp = oversampling * 2 f_end``.  The lower band edge does
    not affect the NUFFT input size under this convention.

``complex-baseband``
    The band is heterodyned to a complex analytic/baseband stream before the
    NUFFT, so ``f_samp = oversampling * (f_end - f_start)``.  Both edges then
    affect the input size.  This ignores the modest filtering overhead.

The quoted four-thread CPU throughput of 1e5 NUFFT input samples/s calibrates
the model at the fiducial 40--64 Hz band.  Away from that point, the runtime is
scaled by the logarithmic factor above.  The script is an analytical sweep,
not a FINUFFT benchmark.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "figs"
STYLE_PATH = SCRIPT_DIR.parent / "paper.mplstyle"

FIDUCIAL_F_START_HZ = 40.0
FIDUCIAL_F_END_HZ = 64.0
FIDUCIAL_T_COH_S = 30.0
FIDUCIAL_N_BETA = 144
FIDUCIAL_CPU_THREADS = 4
FIDUCIAL_THROUGHPUT_SAMPLES_PER_S = 1.0e5
DEFAULT_GPU_SPEEDUPS = (10.0, 100.0)
SAMPLING_CONVENTIONS = ("real-nyquist", "complex-baseband")


@dataclass(frozen=True)
class CostConfiguration:
    """Physical and calibration parameters for the NUFFT cost model."""

    observation_time_s: float = SECONDS_PER_YEAR
    coherent_time_s: float = FIDUCIAL_T_COH_S
    n_beta: int = FIDUCIAL_N_BETA
    oversampling: float = 1.0
    fiducial_f_start_hz: float = FIDUCIAL_F_START_HZ
    fiducial_f_end_hz: float = FIDUCIAL_F_END_HZ
    cpu_throughput_samples_per_s: float = FIDUCIAL_THROUGHPUT_SAMPLES_PER_S
    cpu_threads: int = FIDUCIAL_CPU_THREADS

    def validate(self) -> None:
        positive = {
            "observation_time_s": self.observation_time_s,
            "coherent_time_s": self.coherent_time_s,
            "n_beta": self.n_beta,
            "oversampling": self.oversampling,
            "fiducial_f_start_hz": self.fiducial_f_start_hz,
            "fiducial_f_end_hz": self.fiducial_f_end_hz,
            "cpu_throughput_samples_per_s": self.cpu_throughput_samples_per_s,
            "cpu_threads": self.cpu_threads,
        }
        for name, value in positive.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive; got {value!r}.")
        if self.fiducial_f_start_hz >= self.fiducial_f_end_hz:
            raise ValueError("The fiducial band must satisfy f_start < f_end.")


def sampling_rate_hz(
    f_start_hz: np.ndarray | float,
    f_end_hz: np.ndarray | float,
    convention: str,
    oversampling: float = 1.0,
) -> np.ndarray:
    """Map band edges to the rate of samples passed to each NUFFT."""
    f_start = np.asarray(f_start_hz, dtype=float)
    f_end = np.asarray(f_end_hz, dtype=float)
    if convention == "real-nyquist":
        rate = 2.0 * f_end
    elif convention == "complex-baseband":
        rate = f_end - f_start
    else:
        raise ValueError(
            f"Unknown sampling convention {convention!r}; "
            f"choose from {SAMPLING_CONVENTIONS}."
        )
    return oversampling * rate


def estimate_cost_grid(
    f_start_hz: np.ndarray,
    f_end_hz: np.ndarray,
    convention: str,
    config: CostConfiguration,
) -> dict[str, np.ndarray]:
    """Evaluate work and calibrated CPU runtime on a broadcastable grid."""
    config.validate()
    f_start, f_end = np.broadcast_arrays(
        np.asarray(f_start_hz, dtype=float), np.asarray(f_end_hz, dtype=float)
    )
    valid = (
        np.isfinite(f_start)
        & np.isfinite(f_end)
        & (f_start > 0.0)
        & (f_end > f_start)
    )
    f_samp = sampling_rate_hz(f_start, f_end, convention, config.oversampling)
    samples_per_transform = config.coherent_time_s * f_samp
    valid &= samples_per_transform > 1.0

    fiducial_f_samp = float(
        sampling_rate_hz(
            config.fiducial_f_start_hz,
            config.fiducial_f_end_hz,
            convention,
            config.oversampling,
        )
    )
    fiducial_samples_per_transform = config.coherent_time_s * fiducial_f_samp
    if fiducial_samples_per_transform <= 1.0:
        raise ValueError("The fiducial coherent transform must contain more than one sample.")

    total_input_samples = config.n_beta * config.observation_time_s * f_samp
    log_factor = np.full(f_samp.shape, np.nan, dtype=float)
    np.log(samples_per_transform, out=log_factor, where=valid)
    cost_proxy = total_input_samples * log_factor

    # The benchmark states a sample throughput at the fiducial configuration.
    # The logarithm ratio extends that calibration to other sample rates while
    # exactly recovering samples/throughput at the fiducial sample rate.
    cpu_runtime_s = (
        total_input_samples
        / config.cpu_throughput_samples_per_s
        * log_factor
        / np.log(fiducial_samples_per_transform)
    )

    def masked(values: np.ndarray) -> np.ndarray:
        return np.where(valid, values, np.nan)

    return {
        "valid": valid,
        "bandwidth_hz": masked(f_end - f_start),
        "sampling_rate_hz": masked(f_samp),
        "samples_per_transform": masked(samples_per_transform),
        "total_input_samples": masked(total_input_samples),
        "log_factor": masked(log_factor),
        "cost_proxy": masked(cost_proxy),
        "cpu_runtime_s": masked(cpu_runtime_s),
    }


def selected_conventions(selection: str) -> tuple[str, ...]:
    if selection == "both":
        return SAMPLING_CONVENTIONS
    if selection not in SAMPLING_CONVENTIONS:
        raise ValueError(f"Unknown sampling convention {selection!r}.")
    return (selection,)


def make_sweep(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray]:
    if args.n_f_start < 2 or args.n_f_end < 2:
        raise ValueError("n-f-start and n-f-end must each be at least 2.")
    bounds = {
        "f-start-min": args.f_start_min,
        "f-start-max": args.f_start_max,
        "f-end-min": args.f_end_min,
        "f-end-max": args.f_end_max,
    }
    for name, value in bounds.items():
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive; got {value!r}.")
    if args.f_start_min >= args.f_start_max:
        raise ValueError("f-start-min must be smaller than f-start-max.")
    if args.f_end_min >= args.f_end_max:
        raise ValueError("f-end-min must be smaller than f-end-max.")
    if args.f_start_min >= args.f_end_max:
        raise ValueError("The requested ranges contain no points with f_start < f_end.")

    starts = np.linspace(args.f_start_min, args.f_start_max, args.n_f_start)
    ends = np.linspace(args.f_end_min, args.f_end_max, args.n_f_end)
    return np.meshgrid(starts, ends, indexing="xy")


def scenario_labels(speedups: Iterable[float], cpu_threads: int) -> list[tuple[str, float]]:
    scenarios = [(f"Current CPU ({cpu_threads} threads)", 1.0)]
    scenarios.extend((f"GPU projection ({speedup:g}x)", speedup) for speedup in speedups)
    return scenarios


def save_csv(
    path: Path,
    f_start: np.ndarray,
    f_end: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    config: CostConfiguration,
    speedups: tuple[float, ...],
) -> None:
    runtime_fields = [f"runtime_{speedup:g}x_s" for speedup in speedups]
    fieldnames = [
        "sampling_convention",
        "f_start_hz",
        "f_end_hz",
        "bandwidth_hz",
        "oversampling",
        "f_samp_hz",
        "t_obs_s",
        "t_coh_s",
        "n_beta",
        "cpu_threads",
        "calibration_throughput_samples_per_s",
        "samples_per_coherent_transform",
        "total_nufft_input_samples",
        "log_factor_natural",
        "cost_proxy_sample_log_samples",
        "runtime_current_cpu_s",
        *runtime_fields,
    ]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for convention, result in results.items():
            for index in zip(*np.nonzero(result["valid"])):
                cpu_runtime = float(result["cpu_runtime_s"][index])
                row = {
                    "sampling_convention": convention,
                    "f_start_hz": float(f_start[index]),
                    "f_end_hz": float(f_end[index]),
                    "bandwidth_hz": float(result["bandwidth_hz"][index]),
                    "oversampling": config.oversampling,
                    "f_samp_hz": float(result["sampling_rate_hz"][index]),
                    "t_obs_s": config.observation_time_s,
                    "t_coh_s": config.coherent_time_s,
                    "n_beta": config.n_beta,
                    "cpu_threads": config.cpu_threads,
                    "calibration_throughput_samples_per_s": (
                        config.cpu_throughput_samples_per_s
                    ),
                    "samples_per_coherent_transform": float(
                        result["samples_per_transform"][index]
                    ),
                    "total_nufft_input_samples": float(
                        result["total_input_samples"][index]
                    ),
                    "log_factor_natural": float(result["log_factor"][index]),
                    "cost_proxy_sample_log_samples": float(result["cost_proxy"][index]),
                    "runtime_current_cpu_s": cpu_runtime,
                }
                row.update(
                    {
                        field: cpu_runtime / speedup
                        for field, speedup in zip(runtime_fields, speedups)
                    }
                )
                writer.writerow(row)


def make_figure(
    f_start: np.ndarray,
    f_end: np.ndarray,
    results: dict[str, dict[str, np.ndarray]],
    config: CostConfiguration,
    speedups: tuple[float, ...],
) -> plt.Figure:
    conventions = tuple(results)
    scenarios = scenario_labels(speedups, config.cpu_threads)
    all_runtimes_days = [
        result["cpu_runtime_s"] / speedup / SECONDS_PER_DAY
        for result in results.values()
        for _, speedup in scenarios
    ]
    finite_values = np.concatenate(
        [values[np.isfinite(values)] for values in all_runtimes_days]
    )
    norm = LogNorm(vmin=float(finite_values.min()), vmax=float(finite_values.max()))

    fig, axes = plt.subplots(
        len(conventions),
        len(scenarios),
        figsize=(4.3 * len(scenarios), 3.8 * len(conventions)),
        sharex=True,
        sharey=True,
        squeeze=False,
        constrained_layout=True,
    )
    mesh = None
    for row, convention in enumerate(conventions):
        result = results[convention]
        for column, (label, speedup) in enumerate(scenarios):
            ax = axes[row, column]
            runtime_days = result["cpu_runtime_s"] / speedup / SECONDS_PER_DAY
            mesh = ax.pcolormesh(
                f_start,
                f_end,
                np.ma.masked_invalid(runtime_days),
                shading="auto",
                cmap="viridis",
                norm=norm,
                rasterized=True,
            )
            ax.plot(
                config.fiducial_f_start_hz,
                config.fiducial_f_end_hz,
                marker="*",
                color="white",
                markeredgecolor="black",
                markersize=10,
                linestyle="none",
                zorder=3,
            )
            ax.set_title(label if row == 0 else "")
            if column == 0:
                convention_label = convention.replace("-", " ").title()
                ax.set_ylabel(rf"$f_{{\rm end}}$ [Hz]" + "\n" + convention_label)
            if row == len(conventions) - 1:
                ax.set_xlabel(r"$f_{\rm start}$ [Hz]")
            ax.grid(alpha=0.18)

    if mesh is None:  # Defensive: conventions and scenarios are validated non-empty.
        raise RuntimeError("No NUFFT cost panels were generated.")
    colorbar = fig.colorbar(mesh, ax=axes.ravel().tolist(), pad=0.02)
    colorbar.set_label("Projected wall time [days]")
    fig.suptitle(
        rf"NUFFT cost: $T_{{\rm obs}}={config.observation_time_s / SECONDS_PER_YEAR:g}$ yr, "
        rf"$T_{{\rm coh}}={config.coherent_time_s:g}$ s, "
        rf"$N_\beta={config.n_beta}$, oversampling $={config.oversampling:g}$"
    )
    return fig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep the calibrated NUFFT cost over semicoherent f_start and f_end."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--f-start-min", type=float, default=10.0, help="Lowest f_start [Hz].")
    parser.add_argument("--f-start-max", type=float, default=63.0, help="Highest f_start [Hz].")
    parser.add_argument("--f-end-min", type=float, default=41.0, help="Lowest f_end [Hz].")
    parser.add_argument("--f-end-max", type=float, default=128.0, help="Highest f_end [Hz].")
    parser.add_argument("--n-f-start", type=int, default=120, help="Number of f_start samples.")
    parser.add_argument("--n-f-end", type=int, default=120, help="Number of f_end samples.")
    parser.add_argument(
        "--sampling-convention",
        choices=(*SAMPLING_CONVENTIONS, "both"),
        default="both",
        help="How the searched band determines f_samp.",
    )
    parser.add_argument("--oversampling", type=float, default=1.0, help="Multiplier on f_samp.")
    parser.add_argument("--t-obs-days", type=float, default=365.25, help="Observation time [days].")
    parser.add_argument("--t-coh", type=float, default=FIDUCIAL_T_COH_S, help="Coherent chunk time [s].")
    parser.add_argument("--n-beta", type=int, default=FIDUCIAL_N_BETA, help="Number of beta templates.")
    parser.add_argument(
        "--fiducial-f-start", type=float, default=FIDUCIAL_F_START_HZ, help="Benchmark f_start [Hz]."
    )
    parser.add_argument(
        "--fiducial-f-end", type=float, default=FIDUCIAL_F_END_HZ, help="Benchmark f_end [Hz]."
    )
    parser.add_argument(
        "--throughput",
        type=float,
        default=FIDUCIAL_THROUGHPUT_SAMPLES_PER_S,
        help="Measured four-thread CPU throughput [NUFFT input samples/s].",
    )
    parser.add_argument("--cpu-threads", type=int, default=FIDUCIAL_CPU_THREADS)
    parser.add_argument(
        "--gpu-speedups",
        type=float,
        nargs="+",
        default=DEFAULT_GPU_SPEEDUPS,
        help="Projected GPU speedup factors relative to the current CPU.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--stem", default="nufft_cost_vs_band", help="Output filename stem.")
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution.")
    # parser.add_argument(
    #     "--no-pdf", action="store_true", help="Do not also save a PDF figure."
    # )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    speedups = tuple(float(value) for value in args.gpu_speedups)
    if not speedups or any(not np.isfinite(value) or value <= 0 for value in speedups):
        raise ValueError("Every GPU speedup must be finite and positive.")
    if len(set(speedups)) != len(speedups):
        raise ValueError("GPU speedup factors must be unique.")
    if args.dpi <= 0:
        raise ValueError("dpi must be positive.")

    config = CostConfiguration(
        observation_time_s=args.t_obs_days * SECONDS_PER_DAY,
        coherent_time_s=args.t_coh,
        n_beta=args.n_beta,
        oversampling=args.oversampling,
        fiducial_f_start_hz=args.fiducial_f_start,
        fiducial_f_end_hz=args.fiducial_f_end,
        cpu_throughput_samples_per_s=args.throughput,
        cpu_threads=args.cpu_threads,
    )
    config.validate()
    conventions = selected_conventions(args.sampling_convention)
    f_start, f_end = make_sweep(args)
    results = {
        convention: estimate_cost_grid(f_start, f_end, convention, config)
        for convention in conventions
    }
    if not all(np.any(result["valid"]) for result in results.values()):
        raise ValueError("The requested sweep has no valid f_start < f_end grid points.")

    if STYLE_PATH.exists():
        plt.style.use(STYLE_PATH)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    png_path = args.output_dir / f"{args.stem}.png"
    # csv_path = args.output_dir / f"{args.stem}.csv"
    # save_csv(csv_path, f_start, f_end, results, config, speedups)
    figure = make_figure(f_start, f_end, results, config, speedups)
    figure.savefig(png_path, dpi=args.dpi)
    output_paths = [png_path]
    # if not args.no_pdf:
    #     pdf_path = args.output_dir / f"{args.stem}.pdf"
    #     figure.savefig(pdf_path)
    #     output_paths.append(pdf_path)
    plt.close(figure)

    for convention, result in results.items():
        fiducial = estimate_cost_grid(
            np.array(config.fiducial_f_start_hz),
            np.array(config.fiducial_f_end_hz),
            convention,
            config,
        )
        runtime_days = float(fiducial["cpu_runtime_s"]) / SECONDS_PER_DAY
        f_samp = float(fiducial["sampling_rate_hz"])
        projections = ", ".join(
            f"{speedup:g}x GPU: {runtime_days / speedup:.3g} d" for speedup in speedups
        )
        print(
            f"{convention}: fiducial f_samp={f_samp:g} Hz, "
            f"CPU={runtime_days:.3g} d; {projections}"
        )
    print("Wrote " + ", ".join(str(path) for path in output_paths))


if __name__ == "__main__":
    main()
