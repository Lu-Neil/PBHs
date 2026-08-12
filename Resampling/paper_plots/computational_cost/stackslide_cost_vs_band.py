#!/usr/bin/env python3
"""Plot StackSlide cost and spectrum storage across a frequency band.

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

Edit the settings below, then run::

    conda run -n PBH python computational_cost/stackslide_cost_vs_band.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY
BYTES_PER_GB = 1.0e9
BYTES_PER_TB = 1.0e12

# -----------------------------------------------------------------------------
# Settings: edit these values to change the calculation or output.
# -----------------------------------------------------------------------------

# Frequency-band sweep [Hz]
F_START_MIN = 20.0
F_START_MAX = 60.0
F_END_MIN = 40.0
F_END_MAX = 80.0
N_F_START = 101
N_F_END = 121
FIDUCIAL_F_START = 40.0
FIDUCIAL_F_END = 60.0

# StackSlide cost model
N_TEMPLATES = 1.0e10
READS_PER_TEMPLATE = 1.0e3
READ_BYTES = 4.0  # float32
CPU_BANDWIDTH_GB_S = 50.0
GPU_BANDWIDTH_GB_S = 500.0

# Resampled-spectrum storage
N_BETA = 69
OBSERVATION_TIME_S = SECONDS_PER_YEAR
STORAGE_BYTES = 2.0  # float16, half precision
GPU_MEMORY_GB = 80.0

# Output
OUTPUT_DIR = Path(__file__).resolve().parent / "figs"
OUTPUT_FORMAT = "png"
DPI = 300
SHOW_FIGURE = False


def evaluate_sweep(
    f_starts_hz: np.ndarray, f_ends_hz: np.ndarray
) -> dict[str, np.ndarray]:
    """Evaluate all derived quantities on the rectangular frequency grid."""

    f_start, f_end = np.meshgrid(f_starts_hz, f_ends_hz, indexing="ij")
    f_band = f_end - f_start
    valid = f_band > 0.0

    # Invalid triangle entries remain NaN so they are excluded from the plots.
    bandwidth = np.where(valid, f_band, np.nan)
    spectrum_bytes = N_BETA * OBSERVATION_TIME_S * bandwidth * STORAGE_BYTES
    gpu_memory_bytes = GPU_MEMORY_GB * BYTES_PER_GB
    block_duration_s = gpu_memory_bytes / (N_BETA * bandwidth * STORAGE_BYTES)

    return {
        "f_start_hz": f_start,
        "f_end_hz": f_end,
        "f_band_hz": bandwidth,
        "valid": valid,
        "spectrum_memory_gb": spectrum_bytes / BYTES_PER_GB,
        "block_duration_s": block_duration_s,
        "block_duration_days": block_duration_s / SECONDS_PER_DAY,
        "blocks_required": np.maximum(1.0, np.ceil(spectrum_bytes / gpu_memory_bytes)),
    }


def make_figure(
    results: dict[str, np.ndarray],
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

    total_reads = N_TEMPLATES * READS_PER_TEMPLATE
    traffic_tb = total_reads * READ_BYTES / BYTES_PER_TB
    cpu_s = traffic_tb * BYTES_PER_TB / (CPU_BANDWIDTH_GB_S * BYTES_PER_GB)
    gpu_s = traffic_tb * BYTES_PER_TB / (GPU_BANDWIDTH_GB_S * BYTES_PER_GB)
    fig.suptitle(
        "StackSlide storage and blocking versus search band\n"
        rf"Fixed workload: {N_TEMPLATES:.1e} templates $\times$ "
        rf"{READS_PER_TEMPLATE:.1e} reads; traffic = {traffic_tb:.1f} decimal TB; "
        rf"CPU/GPU = {cpu_s:.0f}/{gpu_s:.0f} s",
        fontsize=14,
    )
    return fig


def fiducial_summary(f_start: float, f_end: float) -> str:
    """Return a compact terminal summary of the selected fiducial band."""

    one = evaluate_sweep(np.array([f_start]), np.array([f_end]))
    total_reads = N_TEMPLATES * READS_PER_TEMPLATE
    traffic_bytes = total_reads * READ_BYTES
    return "\n".join(
        (
            f"Fiducial band: {f_start:g}--{f_end:g} Hz "
            f"(width {f_end - f_start:g} Hz)",
            f"Total power reads: {total_reads:.3g}",
            f"Memory traffic: {traffic_bytes / BYTES_PER_TB:.3g} TB (decimal)",
            f"Ideal CPU time: {traffic_bytes / (CPU_BANDWIDTH_GB_S * BYTES_PER_GB):.3g} s",
            f"Ideal GPU time: {traffic_bytes / (GPU_BANDWIDTH_GB_S * BYTES_PER_GB):.3g} s",
            f"Stored spectra: {one['spectrum_memory_gb'][0, 0]:.3g} GB (decimal)",
            f"Maximum GPU block: {one['block_duration_days'][0, 0]:.3g} days",
            f"Required time blocks: {int(one['blocks_required'][0, 0])}",
        )
    )


def main() -> None:
    f_starts = np.linspace(F_START_MIN, F_START_MAX, N_F_START)
    f_ends = np.linspace(F_END_MIN, F_END_MAX, N_F_END)
    results = evaluate_sweep(f_starts, f_ends)
    if not np.any(results["valid"]):
        raise ValueError("The frequency grid contains no bands with f_end > f_start")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    import matplotlib

    if not SHOW_FIGURE:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style_path = Path(__file__).resolve().parents[1] / "paper.mplstyle"
    if style_path.is_file():
        plt.style.use(style_path)
    figure = make_figure(results, FIDUCIAL_F_START, FIDUCIAL_F_END)
    figure_path = OUTPUT_DIR / f"stackslide_cost_vs_band.{OUTPUT_FORMAT}"
    figure.savefig(figure_path, dpi=DPI, bbox_inches="tight")
    if SHOW_FIGURE:
        plt.show()
    else:
        plt.close(figure)

    print(fiducial_summary(FIDUCIAL_F_START, FIDUCIAL_F_END))
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
