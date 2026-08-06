#!/usr/bin/env python3
"""Combined NUFFT, StackSlide, and storage cost versus search-band edges.

This script sweeps ``(f_start, f_end)`` and applies the leading cost models
quoted in the paper.  The valid half-plane is ``f_end > f_start``.

NUFFT timing is calibrated to a measured effective throughput rather than an
operation count::

    t_NUFFT = N_beta T_obs f_samp / R_ref
              * log(T_coh f_samp) / log(T_coh,ref f_samp,ref).

The default ``R_ref = 1e5 sample evaluations/s`` is the four-thread laptop
measurement.  Here one sample evaluation means processing one input sample for
one beta template.  A configurable acceleration factor projects that measured
time to a GPU.  Two sampling conventions are available: real time series use
``f_samp = 2 f_end`` by default, while heterodyned complex baseband data use
``f_samp = f_end - f_start`` by default.

StackSlide is treated as memory-bandwidth limited.  Its number of reads is
``N_templates * N_chunks``.  By default the quoted 1e10-template bank is held
fixed across the sweep and ``N_chunks`` is the 0PN chirp time through the band,
capped at ``T_obs``, divided by ``T_coh``.  The optional bandwidth exponent can
instead rescale the bank from the fiducial 40--64 Hz band.  A constant 1e3-read
model is also available for reproducing the paper's order-of-magnitude quote.

Storage uses decimal SI units and the paper equations::

    M = N_beta T_obs (f_end - f_start) b_storage,
    T_block = M_GPU / [N_beta (f_end - f_start) b_storage].

The defaults deliberately distinguish half-precision stored spectra
(``b_storage=2`` bytes/bin) from single-precision StackSlide power reads
(``b_read=4`` bytes/read).

Examples
--------
Run the fiducial sweep and write a multipanel plot to ``figs/``::

    conda run -n PBH python computational_cost/combined_cost_vs_band.py

Model a complex-baseband implementation and a fixed 1000 reads/template::

    conda run -n PBH python computational_cost/combined_cost_vs_band.py \
        --sampling-model complex-baseband --chunk-model constant
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


G_SI = 6.67430e-11
C_SI = 299_792_458.0
M_SUN_SI = 1.98847e30
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
BYTES_PER_GB = 1.0e9
BYTES_PER_TB = 1.0e12


@dataclass(frozen=True)
class CostParameters:
    """Parameters held fixed across one frequency-band sweep."""

    observation_time_s: float
    coherent_time_s: float
    n_beta: int
    sampling_model: str
    sample_rate_factor: float
    nufft_rate_samples_s: float
    nufft_reference_coherent_s: float
    nufft_reference_sample_rate_hz: float
    nufft_acceleration: float
    reference_templates: float
    reference_bandwidth_hz: float
    template_band_exponent: float
    chunk_model: str
    constant_chunks: float
    mchirp_msun: float
    read_bytes: float
    cpu_bandwidth_gb_s: float
    gpu_bandwidth_gb_s: float
    storage_bytes: float
    gpu_memory_gb: float


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return parsed


def nonnegative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError("must be a finite non-negative number")
    return parsed


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def at_least_two(value: str) -> int:
    parsed = int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError("must be an integer of at least two")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate combined NUFFT, semicoherent StackSlide, and spectrum-storage "
            "costs over a grid of f_start and f_end."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    sweep = parser.add_argument_group("frequency sweep")
    sweep.add_argument("--f-start-min", type=float, default=20.0, help="minimum f_start [Hz]")
    sweep.add_argument("--f-start-max", type=float, default=60.0, help="maximum f_start [Hz]")
    sweep.add_argument("--f-end-min", type=float, default=40.0, help="minimum f_end [Hz]")
    sweep.add_argument("--f-end-max", type=float, default=100.0, help="maximum f_end [Hz]")
    sweep.add_argument("--n-f-start", type=at_least_two, default=101)
    sweep.add_argument("--n-f-end", type=at_least_two, default=121)
    sweep.add_argument("--fiducial-f-start", type=positive_float, default=40.0)
    sweep.add_argument("--fiducial-f-end", type=positive_float, default=64.0)

    nufft = parser.add_argument_group("NUFFT timing model")
    nufft.add_argument("--observation-time-s", type=positive_float, default=SECONDS_PER_YEAR)
    nufft.add_argument("--coherent-time-s", type=positive_float, default=30.0)
    nufft.add_argument("--n-beta", type=positive_int, default=144)
    nufft.add_argument(
        "--sampling-model",
        choices=("real-nyquist", "complex-baseband"),
        default="real-nyquist",
        help=(
            "real-nyquist sets f_samp=factor*f_end; complex-baseband sets "
            "f_samp=factor*(f_end-f_start)"
        ),
    )
    nufft.add_argument(
        "--sample-rate-factor",
        type=positive_float,
        default=None,
        help="override sampling factor (defaults: 2 real, 1 complex)",
    )
    nufft.add_argument(
        "--nufft-rate-samples-s",
        type=positive_float,
        default=1.0e5,
        help="measured effective beta-sample evaluations per wall second",
    )
    nufft.add_argument("--nufft-reference-coherent-s", type=positive_float, default=30.0)
    nufft.add_argument("--nufft-reference-sample-rate-hz", type=positive_float, default=128.0)
    nufft.add_argument(
        "--nufft-acceleration",
        type=positive_float,
        default=30.0,
        help="projected acceleration relative to the measured laptop runtime",
    )

    stack = parser.add_argument_group("StackSlide model")
    stack.add_argument("--reference-templates", type=positive_float, default=1.0e10)
    stack.add_argument(
        "--template-band-exponent",
        type=nonnegative_float,
        default=0.0,
        help=(
            "N_templates scaling with bandwidth relative to the fiducial band; "
            "zero holds the quoted bank size constant"
        ),
    )
    stack.add_argument(
        "--chunk-model",
        choices=("chirp", "constant"),
        default="chirp",
        help="derive chunks from a 0PN chirp duration or hold them constant",
    )
    stack.add_argument("--constant-chunks", type=positive_float, default=1.0e3)
    stack.add_argument(
        "--mchirp-msun",
        type=positive_float,
        default=1.0e-2,
        help="chirp mass used only by the chirp chunk model [solar masses]",
    )
    stack.add_argument(
        "--read-bytes",
        type=positive_float,
        default=4.0,
        help="bytes transferred per StackSlide power lookup",
    )
    stack.add_argument("--cpu-bandwidth-gb-s", type=positive_float, default=50.0)
    stack.add_argument("--gpu-bandwidth-gb-s", type=positive_float, default=500.0)

    memory = parser.add_argument_group("spectrum storage")
    memory.add_argument(
        "--storage-bytes",
        type=positive_float,
        default=2.0,
        help="bytes per stored resampled-spectrum frequency bin",
    )
    memory.add_argument("--gpu-memory-gb", type=positive_float, default=80.0)

    output = parser.add_argument_group("output")
    output.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "figs",
    )
    output.add_argument("--output-stem", default="combined_cost_vs_band")
    output.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=("png",),
        help="figure formats to write (PDF saving is currently disabled)",
    )
    output.add_argument("--dpi", type=positive_int, default=300)
    output.add_argument("--show", action="store_true")
    return parser


def validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    frequency_values = (
        args.f_start_min,
        args.f_start_max,
        args.f_end_min,
        args.f_end_max,
        args.fiducial_f_start,
        args.fiducial_f_end,
    )
    if not all(math.isfinite(value) for value in frequency_values):
        parser.error("all frequencies must be finite")
    if args.f_start_min <= 0.0 or args.f_end_min <= 0.0:
        parser.error("frequency bounds must be positive")
    if args.f_start_max <= args.f_start_min:
        parser.error("--f-start-max must exceed --f-start-min")
    if args.f_end_max <= args.f_end_min:
        parser.error("--f-end-max must exceed --f-end-min")
    if args.f_end_max <= args.f_start_min:
        parser.error("the sweep contains no valid f_end > f_start points")
    if args.fiducial_f_end <= args.fiducial_f_start:
        parser.error("--fiducial-f-end must exceed --fiducial-f-start")
    if not args.output_stem or Path(args.output_stem).name != args.output_stem:
        parser.error("--output-stem must be a non-empty filename stem")


def parameters_from_args(args: argparse.Namespace) -> CostParameters:
    sample_rate_factor = args.sample_rate_factor
    if sample_rate_factor is None:
        sample_rate_factor = 2.0 if args.sampling_model == "real-nyquist" else 1.0
    return CostParameters(
        observation_time_s=args.observation_time_s,
        coherent_time_s=args.coherent_time_s,
        n_beta=args.n_beta,
        sampling_model=args.sampling_model,
        sample_rate_factor=sample_rate_factor,
        nufft_rate_samples_s=args.nufft_rate_samples_s,
        nufft_reference_coherent_s=args.nufft_reference_coherent_s,
        nufft_reference_sample_rate_hz=args.nufft_reference_sample_rate_hz,
        nufft_acceleration=args.nufft_acceleration,
        reference_templates=args.reference_templates,
        reference_bandwidth_hz=args.fiducial_f_end - args.fiducial_f_start,
        template_band_exponent=args.template_band_exponent,
        chunk_model=args.chunk_model,
        constant_chunks=args.constant_chunks,
        mchirp_msun=args.mchirp_msun,
        read_bytes=args.read_bytes,
        cpu_bandwidth_gb_s=args.cpu_bandwidth_gb_s,
        gpu_bandwidth_gb_s=args.gpu_bandwidth_gb_s,
        storage_bytes=args.storage_bytes,
        gpu_memory_gb=args.gpu_memory_gb,
    )


def inspiral_time_s(
    f_start_hz: np.ndarray, f_end_hz: np.ndarray, mchirp_msun: float
) -> np.ndarray:
    """Return the leading-order time for an inspiral to cross a band."""

    chirp_mass_seconds = G_SI * (mchirp_msun * M_SUN_SI) / C_SI**3
    prefactor = (5.0 / 256.0) * chirp_mass_seconds ** (-5.0 / 3.0)
    prefactor *= np.pi ** (-8.0 / 3.0)
    return prefactor * (f_start_hz ** (-8.0 / 3.0) - f_end_hz ** (-8.0 / 3.0))


def evaluate_sweep(
    f_starts_hz: np.ndarray,
    f_ends_hz: np.ndarray,
    parameters: CostParameters,
) -> dict[str, np.ndarray]:
    """Evaluate all costs on a rectangular grid; invalid bands become NaN."""

    f_start, f_end = np.meshgrid(f_starts_hz, f_ends_hz, indexing="xy")
    bandwidth_hz = f_end - f_start
    valid = bandwidth_hz > 0.0
    band = np.where(valid, bandwidth_hz, np.nan)

    if parameters.sampling_model == "real-nyquist":
        sample_rate_hz = parameters.sample_rate_factor * f_end
    else:
        sample_rate_hz = parameters.sample_rate_factor * band
    sample_rate_hz = np.where(valid, sample_rate_hz, np.nan)

    log_argument = parameters.coherent_time_s * sample_rate_hz
    reference_log_argument = (
        parameters.nufft_reference_coherent_s
        * parameters.nufft_reference_sample_rate_hz
    )
    if np.any(log_argument[valid] <= 1.0) or reference_log_argument <= 1.0:
        raise ValueError("NUFFT logarithm requires T_coh * f_samp > 1")
    nufft_work_samples = (
        parameters.n_beta * parameters.observation_time_s * sample_rate_hz
    )
    nufft_laptop_s = nufft_work_samples / parameters.nufft_rate_samples_s
    nufft_laptop_s *= np.log(log_argument) / np.log(reference_log_argument)
    nufft_projected_s = nufft_laptop_s / parameters.nufft_acceleration

    templates = parameters.reference_templates * (
        band / parameters.reference_bandwidth_hz
    ) ** parameters.template_band_exponent
    chirp_duration_s = inspiral_time_s(f_start, f_end, parameters.mchirp_msun)
    chirp_duration_s = np.where(valid, chirp_duration_s, np.nan)
    if parameters.chunk_model == "chirp":
        chunks = np.maximum(
            1.0,
            np.minimum(chirp_duration_s, parameters.observation_time_s)
            / parameters.coherent_time_s,
        )
    else:
        chunks = np.full_like(band, parameters.constant_chunks)
    chunks = np.where(valid, chunks, np.nan)

    power_reads = templates * chunks
    stack_traffic_bytes = power_reads * parameters.read_bytes
    stack_cpu_s = stack_traffic_bytes / (
        parameters.cpu_bandwidth_gb_s * BYTES_PER_GB
    )
    stack_gpu_s = stack_traffic_bytes / (
        parameters.gpu_bandwidth_gb_s * BYTES_PER_GB
    )

    spectra_bytes = (
        parameters.n_beta
        * parameters.observation_time_s
        * band
        * parameters.storage_bytes
    )
    gpu_memory_bytes = parameters.gpu_memory_gb * BYTES_PER_GB
    block_duration_s = gpu_memory_bytes / (
        parameters.n_beta * band * parameters.storage_bytes
    )
    blocks_required = np.maximum(1.0, np.ceil(spectra_bytes / gpu_memory_bytes))

    return {
        "f_start_hz": f_start,
        "f_end_hz": f_end,
        "bandwidth_hz": band,
        "valid": valid,
        "sample_rate_hz": sample_rate_hz,
        "nufft_work_samples": nufft_work_samples,
        "nufft_laptop_s": nufft_laptop_s,
        "nufft_projected_s": nufft_projected_s,
        "n_templates": templates,
        "chirp_duration_s": chirp_duration_s,
        "chunks_per_template": chunks,
        "power_reads": power_reads,
        "stack_traffic_tb": stack_traffic_bytes / BYTES_PER_TB,
        "stack_cpu_s": stack_cpu_s,
        "stack_gpu_s": stack_gpu_s,
        "total_cpu_s": nufft_laptop_s + stack_cpu_s,
        "total_projected_gpu_s": nufft_projected_s + stack_gpu_s,
        "spectra_memory_gb": spectra_bytes / BYTES_PER_GB,
        "block_duration_s": block_duration_s,
        "block_duration_days": block_duration_s / SECONDS_PER_DAY,
        "blocks_required": blocks_required,
    }


CSV_FIELDS = (
    "f_start_hz",
    "f_end_hz",
    "bandwidth_hz",
    "sample_rate_hz",
    "sampling_model",
    "sample_rate_factor",
    "n_beta",
    "observation_time_s",
    "coherent_time_s",
    "nufft_work_samples",
    "nufft_rate_samples_s",
    "nufft_laptop_s",
    "nufft_acceleration",
    "nufft_projected_s",
    "n_templates",
    "template_band_exponent",
    "chunk_model",
    "mchirp_msun",
    "chirp_duration_s",
    "chunks_per_template",
    "power_reads",
    "read_bytes",
    "stack_traffic_tb_decimal",
    "cpu_bandwidth_gb_s_decimal",
    "stack_cpu_s",
    "gpu_bandwidth_gb_s_decimal",
    "stack_gpu_s",
    "total_cpu_s",
    "total_projected_gpu_s",
    "storage_bytes_per_bin",
    "spectra_memory_gb_decimal",
    "gpu_memory_gb_decimal",
    "block_duration_s",
    "block_duration_days",
    "blocks_required",
)


def write_csv(
    path: Path,
    results: dict[str, np.ndarray],
    parameters: CostParameters,
) -> None:
    """Write one row per valid band, including assumptions for reproducibility."""

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for j, i in np.argwhere(results["valid"]):
            writer.writerow(
                {
                    "f_start_hz": f"{results['f_start_hz'][j, i]:.10g}",
                    "f_end_hz": f"{results['f_end_hz'][j, i]:.10g}",
                    "bandwidth_hz": f"{results['bandwidth_hz'][j, i]:.10g}",
                    "sample_rate_hz": f"{results['sample_rate_hz'][j, i]:.10g}",
                    "sampling_model": parameters.sampling_model,
                    "sample_rate_factor": f"{parameters.sample_rate_factor:.10g}",
                    "n_beta": parameters.n_beta,
                    "observation_time_s": f"{parameters.observation_time_s:.10g}",
                    "coherent_time_s": f"{parameters.coherent_time_s:.10g}",
                    "nufft_work_samples": f"{results['nufft_work_samples'][j, i]:.10g}",
                    "nufft_rate_samples_s": f"{parameters.nufft_rate_samples_s:.10g}",
                    "nufft_laptop_s": f"{results['nufft_laptop_s'][j, i]:.10g}",
                    "nufft_acceleration": f"{parameters.nufft_acceleration:.10g}",
                    "nufft_projected_s": f"{results['nufft_projected_s'][j, i]:.10g}",
                    "n_templates": f"{results['n_templates'][j, i]:.10g}",
                    "template_band_exponent": f"{parameters.template_band_exponent:.10g}",
                    "chunk_model": parameters.chunk_model,
                    "mchirp_msun": f"{parameters.mchirp_msun:.10g}",
                    "chirp_duration_s": f"{results['chirp_duration_s'][j, i]:.10g}",
                    "chunks_per_template": f"{results['chunks_per_template'][j, i]:.10g}",
                    "power_reads": f"{results['power_reads'][j, i]:.10g}",
                    "read_bytes": f"{parameters.read_bytes:.10g}",
                    "stack_traffic_tb_decimal": f"{results['stack_traffic_tb'][j, i]:.10g}",
                    "cpu_bandwidth_gb_s_decimal": f"{parameters.cpu_bandwidth_gb_s:.10g}",
                    "stack_cpu_s": f"{results['stack_cpu_s'][j, i]:.10g}",
                    "gpu_bandwidth_gb_s_decimal": f"{parameters.gpu_bandwidth_gb_s:.10g}",
                    "stack_gpu_s": f"{results['stack_gpu_s'][j, i]:.10g}",
                    "total_cpu_s": f"{results['total_cpu_s'][j, i]:.10g}",
                    "total_projected_gpu_s": f"{results['total_projected_gpu_s'][j, i]:.10g}",
                    "storage_bytes_per_bin": f"{parameters.storage_bytes:.10g}",
                    "spectra_memory_gb_decimal": f"{results['spectra_memory_gb'][j, i]:.10g}",
                    "gpu_memory_gb_decimal": f"{parameters.gpu_memory_gb:.10g}",
                    "block_duration_s": f"{results['block_duration_s'][j, i]:.10g}",
                    "block_duration_days": f"{results['block_duration_days'][j, i]:.10g}",
                    "blocks_required": int(results["blocks_required"][j, i]),
                }
            )


def _log_norm(values: np.ndarray):
    from matplotlib.colors import LogNorm

    finite = values[np.isfinite(values) & (values > 0.0)]
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if math.isclose(vmin, vmax):
        vmin *= 0.9
        vmax *= 1.1
    return LogNorm(vmin=vmin, vmax=vmax)


def make_figure(
    results: dict[str, np.ndarray],
    parameters: CostParameters,
    fiducial_f_start: float,
    fiducial_f_end: float,
):
    """Create a publication-style six-panel summary."""

    import matplotlib.pyplot as plt

    panel_specs = (
        ("nufft_laptop_s", 1.0 / SECONDS_PER_DAY, "NUFFT: measured CPU model", "time [days]"),
        ("stack_gpu_s", 1.0, "StackSlide: GPU bandwidth model", "time [s]"),
        (
            "total_projected_gpu_s",
            1.0 / 3600.0,
            f"Total: NUFFT/{parameters.nufft_acceleration:g} + StackSlide GPU",
            "time [h]",
        ),
        ("spectra_memory_gb", 1.0, "Resampled spectra", "memory [GB]"),
        ("block_duration_days", 1.0, "Maximum in-memory block", "duration [days]"),
        ("blocks_required", 1.0, "GPU blocks required", "number of blocks"),
    )

    f_start = results["f_start_hz"]
    f_end = results["f_end_hz"]
    figure, axes = plt.subplots(2, 3, figsize=(14.0, 8.5), constrained_layout=True)
    labels = "abcdef"
    for label, axis, (key, scale, title, colorbar_label) in zip(
        labels, axes.flat, panel_specs
    ):
        values = results[key] * scale
        image = axis.pcolormesh(
            f_start,
            f_end,
            np.ma.masked_invalid(values),
            shading="auto",
            cmap="viridis",
            norm=_log_norm(values),
        )
        axis.plot(
            fiducial_f_start,
            fiducial_f_end,
            marker="*",
            markersize=11,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
            linestyle="none",
            zorder=5,
            label=f"{fiducial_f_start:g}--{fiducial_f_end:g} Hz fiducial",
        )
        axis.set_title(f"({label}) {title}")
        axis.set_xlabel(r"$f_{\rm start}$ [Hz]")
        axis.set_ylabel(r"$f_{\rm end}$ [Hz]")
        axis.set_xlim(float(np.nanmin(f_start)), float(np.nanmax(f_start)))
        axis.set_ylim(float(np.nanmin(f_end)), float(np.nanmax(f_end)))
        colorbar = figure.colorbar(image, ax=axis, pad=0.02)
        colorbar.set_label(colorbar_label)

    sample_description = (
        r"$f_{\rm samp}="
        + f"{parameters.sample_rate_factor:g}"
        + (r"f_{\rm end}$" if parameters.sampling_model == "real-nyquist" else r"\Delta f$")
    )
    template_description = (
        rf"$N_{{\rm temp}}=10^{{{np.log10(parameters.reference_templates):.1f}}}"
        rf"(\Delta f/{parameters.reference_bandwidth_hz:g}\,{{\rm Hz}})"
        rf"^{{{parameters.template_band_exponent:g}}}$"
    )
    figure.suptitle(
        "Combined NUFFT and semicoherent resource estimates\n"
        + sample_description
        + "; "
        + template_description
        + rf"; $N_\beta={parameters.n_beta}$; $T_{{\rm coh}}={parameters.coherent_time_s:g}$ s",
        fontsize=14,
    )
    return figure


def _fiducial_results(
    f_start: float, f_end: float, parameters: CostParameters
) -> dict[str, float]:
    arrays = evaluate_sweep(np.array([f_start]), np.array([f_end]), parameters)
    return {
        key: float(value[0, 0])
        for key, value in arrays.items()
        if key != "valid"
    }


def print_summary(
    parameters: CostParameters,
    f_start: float,
    f_end: float,
    values: dict[str, float],
) -> None:
    template_note = (
        "held fixed across bandwidth"
        if parameters.template_band_exponent == 0.0
        else f"scaled as bandwidth^{parameters.template_band_exponent:g}"
    )
    print(f"Fiducial band: {f_start:g}--{f_end:g} Hz")
    print(
        f"Sampling: {parameters.sampling_model}, factor={parameters.sample_rate_factor:g}, "
        f"f_samp={values['sample_rate_hz']:.3g} Hz"
    )
    print(
        f"Templates: {values['n_templates']:.3g} ({template_note}); "
        f"chunks/template={values['chunks_per_template']:.3g} ({parameters.chunk_model})"
    )
    print(
        f"NUFFT: {values['nufft_laptop_s'] / SECONDS_PER_DAY:.3g} laptop days; "
        f"{values['nufft_projected_s'] / 3600.0:.3g} projected accelerator hours"
    )
    print(
        f"StackSlide: {values['power_reads']:.3g} reads, "
        f"{values['stack_traffic_tb']:.3g} TB traffic, "
        f"{values['stack_cpu_s']:.3g} CPU s / {values['stack_gpu_s']:.3g} GPU s"
    )
    print(
        f"Storage: {values['spectra_memory_gb']:.3g} GB; "
        f"T_block={values['block_duration_days']:.3g} days; "
        f"blocks={int(values['blocks_required'])}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    validate_args(args, parser)
    parameters = parameters_from_args(args)

    f_starts = np.linspace(args.f_start_min, args.f_start_max, args.n_f_start)
    f_ends = np.linspace(args.f_end_min, args.f_end_max, args.n_f_end)
    try:
        results = evaluate_sweep(f_starts, f_ends, parameters)
    except ValueError as error:
        parser.error(str(error))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # csv_path = args.output_dir / f"{args.output_stem}.csv"
    # write_csv(csv_path, results, parameters)

    import matplotlib

    if not args.show:
        matplotlib.use("Agg")
    from matplotlib import style

    style_path = Path(__file__).resolve().parents[1] / "paper.mplstyle"
    if style_path.exists():
        style.use(style_path)
    figure = make_figure(
        results,
        parameters,
        args.fiducial_f_start,
        args.fiducial_f_end,
    )
    figure_paths = []
    for figure_format in args.formats:
        if figure_format == "pdf":
            # PDF saving is intentionally disabled.
            continue
        figure_path = args.output_dir / f"{args.output_stem}.{figure_format}"
        figure.savefig(figure_path, dpi=args.dpi, bbox_inches="tight")
        figure_paths.append(figure_path)

    if args.show:
        import matplotlib.pyplot as plt

        plt.show()
    else:
        import matplotlib.pyplot as plt

        plt.close(figure)

    fiducial = _fiducial_results(
        args.fiducial_f_start,
        args.fiducial_f_end,
        parameters,
    )
    print_summary(
        parameters,
        args.fiducial_f_start,
        args.fiducial_f_end,
        fiducial,
    )
    # print(f"Wrote {csv_path}")
    for path in figure_paths:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
