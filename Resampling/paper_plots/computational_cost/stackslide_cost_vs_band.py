#!/usr/bin/env python3
"""Estimate StackSlide cost and spectrum storage across a frequency band.

The sweep treats ``f_start`` and ``f_end`` as independent coordinates, retaining
only points for which ``f_end > f_start``.  With the fiducial model used in the
paper, the StackSlide traffic is set by the number of templates and the mean
number of coherent chunks crossed by each template.  It is consequently
independent of bandwidth when those two inputs are held fixed.  Spectrum memory
and the maximum in-memory time-block duration do depend on bandwidth:

    M = N_beta T_obs (f_end - f_start) b_storage
    T_block = M_GPU / [N_beta (f_end - f_start) b_storage].

Sizes and bandwidths use decimal SI units throughout: 1 GB = 10**9 bytes and
1 TB = 10**12 bytes.  The default read size is four bytes (single precision),
whereas the default stored spectrum uses two bytes (half precision).

Examples
--------
Run the fiducial two-dimensional sweep::

    conda run -n PBH python computational_cost/stackslide_cost_vs_band.py

Use a smaller custom frequency range::

    conda run -n PBH python computational_cost/stackslide_cost_vs_band.py \
        --f-start-min 10 --f-start-max 20 --f-end-min 20 --f-end-max 32
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
BYTES_PER_GB = 1.0e9
BYTES_PER_TB = 1.0e12


@dataclass(frozen=True)
class CostParameters:
    """Parameters that are fixed across the frequency-band sweep."""

    n_templates: float
    reads_per_template: float
    read_bytes: float
    cpu_bandwidth_gb_s: float
    gpu_bandwidth_gb_s: float
    n_beta: int
    observation_time_s: float
    storage_bytes: float
    gpu_memory_gb: float


def positive_float(value: str) -> float:
    """Argparse converter requiring a finite, strictly positive float."""

    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def positive_int(value: str) -> int:
    """Argparse converter requiring a strictly positive integer."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def at_least_two(value: str) -> int:
    """Argparse converter for a grid size that can define an interval."""

    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be an integer of at least two")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep valid (f_start, f_end) bands and estimate StackSlide memory "
            "traffic, bandwidth-limited runtime, and GPU block duration."
        )
    )
    band = parser.add_argument_group("frequency sweep")
    band.add_argument("--f-start-min", type=float, default=20.0, help="minimum f_start [Hz]")
    band.add_argument("--f-start-max", type=float, default=60.0, help="maximum f_start [Hz]")
    band.add_argument("--f-end-min", type=float, default=40.0, help="minimum f_end [Hz]")
    band.add_argument("--f-end-max", type=float, default=80.0, help="maximum f_end [Hz]")
    band.add_argument("--n-f-start", type=at_least_two, default=101)
    band.add_argument("--n-f-end", type=at_least_two, default=121)
    band.add_argument(
        "--fiducial-f-start", type=float, default=40.0, help="fiducial marker start [Hz]"
    )
    band.add_argument(
        "--fiducial-f-end", type=float, default=64.0, help="fiducial marker end [Hz]"
    )

    cost = parser.add_argument_group("StackSlide cost model")
    cost.add_argument(
        "--n-templates",
        type=positive_float,
        default=1.0e10,
        help="fixed template count at every valid point in the band sweep",
    )
    cost.add_argument(
        "--reads-per-template",
        type=positive_float,
        default=1.0e3,
        help="fixed mean coherent chunks (power reads) per template",
    )
    cost.add_argument(
        "--read-bytes",
        type=positive_float,
        default=4.0,
        help="bytes transferred for each power read (default: float32)",
    )
    cost.add_argument(
        "--cpu-bandwidth-gb-s",
        type=positive_float,
        default=50.0,
        help="sustained CPU memory bandwidth in decimal GB/s",
    )
    cost.add_argument(
        "--gpu-bandwidth-gb-s",
        type=positive_float,
        default=500.0,
        help="sustained GPU memory bandwidth in decimal GB/s",
    )

    memory = parser.add_argument_group("resampled-spectrum storage")
    memory.add_argument("--n-beta", type=positive_int, default=144)
    memory.add_argument(
        "--observation-time-s", type=positive_float, default=SECONDS_PER_YEAR
    )
    memory.add_argument(
        "--storage-bytes",
        type=positive_float,
        default=2.0,
        help="bytes per stored frequency bin (default: float16)",
    )
    memory.add_argument(
        "--gpu-memory-gb",
        type=positive_float,
        default=80.0,
        help="available GPU memory in decimal GB",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figs",
    )
    output.add_argument(
        "--formats",
        nargs="+",
        choices=("pdf", "png", "svg"),
        default=("png",),
        help="figure formats to write (PDF saving is currently disabled)",
    )
    output.add_argument("--dpi", type=positive_int, default=300)
    output.add_argument("--show", action="store_true", help="display the figure interactively")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    bounds = (
        args.f_start_min,
        args.f_start_max,
        args.f_end_min,
        args.f_end_max,
        args.fiducial_f_start,
        args.fiducial_f_end,
    )
    if not all(math.isfinite(value) for value in bounds):
        parser.error("all frequency values must be finite")
    if args.f_start_min < 0.0 or args.f_end_min < 0.0:
        parser.error("frequency bounds must be non-negative")
    if args.f_start_max <= args.f_start_min:
        parser.error("--f-start-max must exceed --f-start-min")
    if args.f_end_max <= args.f_end_min:
        parser.error("--f-end-max must exceed --f-end-min")
    if args.f_end_max <= args.f_start_min:
        parser.error("the requested grid contains no valid f_end > f_start bands")
    if args.fiducial_f_end <= args.fiducial_f_start:
        parser.error("--fiducial-f-end must exceed --fiducial-f-start")


def evaluate_sweep(
    f_starts_hz: np.ndarray, f_ends_hz: np.ndarray, parameters: CostParameters
) -> dict[str, np.ndarray]:
    """Evaluate all derived quantities on the rectangular frequency grid."""

    f_start, f_end = np.meshgrid(f_starts_hz, f_ends_hz, indexing="ij")
    f_band = f_end - f_start
    valid = f_band > 0.0

    # Invalid triangle entries remain NaN so they are excluded from plots/CSV.
    bandwidth = np.where(valid, f_band, np.nan)
    spectrum_bytes = (
        parameters.n_beta
        * parameters.observation_time_s
        * bandwidth
        * parameters.storage_bytes
    )
    gpu_memory_bytes = parameters.gpu_memory_gb * BYTES_PER_GB
    block_duration_s = gpu_memory_bytes / (
        parameters.n_beta * bandwidth * parameters.storage_bytes
    )

    total_reads = parameters.n_templates * parameters.reads_per_template
    traffic_bytes = total_reads * parameters.read_bytes
    cpu_time_s = traffic_bytes / (parameters.cpu_bandwidth_gb_s * BYTES_PER_GB)
    gpu_time_s = traffic_bytes / (parameters.gpu_bandwidth_gb_s * BYTES_PER_GB)

    return {
        "f_start_hz": f_start,
        "f_end_hz": f_end,
        "f_band_hz": bandwidth,
        "valid": valid,
        "spectrum_memory_gb": spectrum_bytes / BYTES_PER_GB,
        "block_duration_s": block_duration_s,
        "block_duration_days": block_duration_s / SECONDS_PER_DAY,
        "blocks_required": np.maximum(1.0, np.ceil(spectrum_bytes / gpu_memory_bytes)),
        "total_power_reads": np.full_like(bandwidth, total_reads),
        "traffic_gb": np.full_like(bandwidth, traffic_bytes / BYTES_PER_GB),
        "traffic_tb": np.full_like(bandwidth, traffic_bytes / BYTES_PER_TB),
        "cpu_time_s": np.full_like(bandwidth, cpu_time_s),
        "gpu_time_s": np.full_like(bandwidth, gpu_time_s),
    }


def write_csv(
    path: Path, results: dict[str, np.ndarray], parameters: CostParameters
) -> None:
    """Write one row for every valid frequency band."""

    fields = (
        "f_start_hz",
        "f_end_hz",
        "f_band_hz",
        "n_templates",
        "reads_per_template",
        "total_power_reads",
        "read_bytes",
        "traffic_gb_decimal",
        "traffic_tb_decimal",
        "cpu_bandwidth_gb_s_decimal",
        "cpu_time_s",
        "gpu_bandwidth_gb_s_decimal",
        "gpu_time_s",
        "n_beta",
        "observation_time_s",
        "storage_bytes_per_bin",
        "spectrum_memory_gb_decimal",
        "gpu_memory_gb_decimal",
        "block_duration_s",
        "block_duration_days",
        "blocks_required",
    )
    valid = results["valid"]
    valid_indices = np.argwhere(valid)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for i, j in valid_indices:
            writer.writerow(
                {
                    "f_start_hz": f"{results['f_start_hz'][i, j]:.9g}",
                    "f_end_hz": f"{results['f_end_hz'][i, j]:.9g}",
                    "f_band_hz": f"{results['f_band_hz'][i, j]:.9g}",
                    "n_templates": f"{parameters.n_templates:.9g}",
                    "reads_per_template": f"{parameters.reads_per_template:.9g}",
                    "total_power_reads": f"{results['total_power_reads'][i, j]:.9g}",
                    "read_bytes": f"{parameters.read_bytes:.9g}",
                    "traffic_gb_decimal": f"{results['traffic_gb'][i, j]:.9g}",
                    "traffic_tb_decimal": f"{results['traffic_tb'][i, j]:.9g}",
                    "cpu_bandwidth_gb_s_decimal": f"{parameters.cpu_bandwidth_gb_s:.9g}",
                    "cpu_time_s": f"{results['cpu_time_s'][i, j]:.9g}",
                    "gpu_bandwidth_gb_s_decimal": f"{parameters.gpu_bandwidth_gb_s:.9g}",
                    "gpu_time_s": f"{results['gpu_time_s'][i, j]:.9g}",
                    "n_beta": parameters.n_beta,
                    "observation_time_s": f"{parameters.observation_time_s:.9g}",
                    "storage_bytes_per_bin": f"{parameters.storage_bytes:.9g}",
                    "spectrum_memory_gb_decimal": (
                        f"{results['spectrum_memory_gb'][i, j]:.9g}"
                    ),
                    "gpu_memory_gb_decimal": f"{parameters.gpu_memory_gb:.9g}",
                    "block_duration_s": f"{results['block_duration_s'][i, j]:.9g}",
                    "block_duration_days": f"{results['block_duration_days'][i, j]:.9g}",
                    "blocks_required": int(results["blocks_required"][i, j]),
                }
            )


def make_figure(
    results: dict[str, np.ndarray],
    parameters: CostParameters,
    fiducial_f_start: float,
    fiducial_f_end: float,
):
    """Construct the two-panel publication figure."""

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    f_start = results["f_start_hz"]
    f_end = results["f_end_hz"]
    memory_gb = np.ma.masked_invalid(results["spectrum_memory_gb"])
    block_days = np.ma.masked_invalid(results["block_duration_days"])

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)
    panels = (
        (memory_gb, r"Spectrum memory [decimal GB]", "viridis"),
        (block_days, r"Maximum GPU block [days]", "magma_r"),
    )
    for ax, (values, colorbar_label, cmap) in zip(axes, panels):
        positive = values.compressed()
        image = ax.pcolormesh(
            f_start,
            f_end,
            values,
            shading="auto",
            cmap=cmap,
            norm=LogNorm(vmin=positive.min(), vmax=positive.max()),
        )
        colorbar = fig.colorbar(image, ax=ax, pad=0.02)
        colorbar.set_label(colorbar_label)
        ax.set_xlabel(r"$f_{\rm start}$ [Hz]")
        ax.set_ylabel(r"$f_{\rm end}$ [Hz]")
        ax.set_facecolor("0.92")

        # The 24-Hz contour reproduces the frequency width quoted in the paper.
        bandwidth = results["f_band_hz"]
        if np.nanmin(bandwidth) <= 24.0 <= np.nanmax(bandwidth):
            contour = ax.contour(
                f_start, f_end, bandwidth, levels=[24.0], colors="white", linewidths=1.4
            )
            ax.clabel(contour, fmt={24.0: r"$\Delta f=24\,$Hz"}, fontsize=10)
        if (
            f_end.min() <= fiducial_f_end <= f_end.max()
            and f_start.min() <= fiducial_f_start <= f_start.max()
        ):
            ax.plot(
                fiducial_f_start,
                fiducial_f_end,
                marker="*",
                markersize=10,
                markeredgecolor="black",
                markerfacecolor="white",
                linestyle="none",
                label="fiducial band",
            )
            ax.legend(loc="upper right")

    total_reads = parameters.n_templates * parameters.reads_per_template
    traffic_tb = total_reads * parameters.read_bytes / BYTES_PER_TB
    cpu_s = traffic_tb * BYTES_PER_TB / (parameters.cpu_bandwidth_gb_s * BYTES_PER_GB)
    gpu_s = traffic_tb * BYTES_PER_TB / (parameters.gpu_bandwidth_gb_s * BYTES_PER_GB)
    fig.suptitle(
        "StackSlide storage and blocking versus search band\n"
        rf"Fixed workload: {parameters.n_templates:.1e} templates $\times$ "
        rf"{parameters.reads_per_template:.1e} reads; traffic = {traffic_tb:.1f} decimal TB; "
        rf"CPU/GPU = {cpu_s:.0f}/{gpu_s:.0f} s",
        fontsize=14,
    )
    return fig


def fiducial_summary(parameters: CostParameters, f_start: float, f_end: float) -> str:
    """Return a compact terminal summary of the selected fiducial band."""

    one = evaluate_sweep(np.array([f_start]), np.array([f_end]), parameters)
    return "\n".join(
        (
            f"Fiducial band: {f_start:g}--{f_end:g} Hz "
            f"(width {f_end - f_start:g} Hz)",
            f"Total power reads: {one['total_power_reads'][0, 0]:.3g}",
            f"Memory traffic: {one['traffic_tb'][0, 0]:.3g} TB (decimal)",
            f"Ideal CPU time: {one['cpu_time_s'][0, 0]:.3g} s",
            f"Ideal GPU time: {one['gpu_time_s'][0, 0]:.3g} s",
            f"Stored spectra: {one['spectrum_memory_gb'][0, 0]:.3g} GB (decimal)",
            f"Maximum GPU block: {one['block_duration_days'][0, 0]:.3g} days",
            f"Required time blocks: {int(one['blocks_required'][0, 0])}",
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)

    parameters = CostParameters(
        n_templates=args.n_templates,
        reads_per_template=args.reads_per_template,
        read_bytes=args.read_bytes,
        cpu_bandwidth_gb_s=args.cpu_bandwidth_gb_s,
        gpu_bandwidth_gb_s=args.gpu_bandwidth_gb_s,
        n_beta=args.n_beta,
        observation_time_s=args.observation_time_s,
        storage_bytes=args.storage_bytes,
        gpu_memory_gb=args.gpu_memory_gb,
    )
    f_starts = np.linspace(args.f_start_min, args.f_start_max, args.n_f_start)
    f_ends = np.linspace(args.f_end_min, args.f_end_max, args.n_f_end)
    results = evaluate_sweep(f_starts, f_ends, parameters)
    if not np.any(results["valid"]):
        parser.error("the requested grid contains no valid f_end > f_start bands")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # csv_path = args.output_dir / "stackslide_cost_vs_band.csv"
    # write_csv(csv_path, results, parameters)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path(__file__).resolve().parents[1] / "paper.mplstyle"
    if style_path.is_file():
        plt.style.use(style_path)
    figure = make_figure(
        results, parameters, args.fiducial_f_start, args.fiducial_f_end
    )
    output_paths = []
    for extension in dict.fromkeys(args.formats):
        if extension == "pdf":
            # PDF saving is intentionally disabled.
            continue
        figure_path = args.output_dir / f"stackslide_cost_vs_band.{extension}"
        figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
        output_paths.append(figure_path)
    if args.show:
        plt.show()
    else:
        plt.close(figure)

    print(fiducial_summary(parameters, args.fiducial_f_start, args.fiducial_f_end))
    # print(f"Wrote {csv_path}")
    for figure_path in output_paths:
        print(f"Wrote {figure_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
