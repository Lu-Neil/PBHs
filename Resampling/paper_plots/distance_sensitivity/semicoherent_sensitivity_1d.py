# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PAPER_PLOTS_DIR = SCRIPT_DIR.parent

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity import semicoherent_sensitivity as sc

mpl.use("Agg")
pl.style.use(SCRIPT_DIR.parent / "paper.mplstyle")

# %%
FREQUENCY_CONFIGS = [
    (20.0, 200.0),
    (20.0, 60.0),
    (40.0, 60.0),
    (80.0, 100.0),
    # (200.0, 220.0),
]
FREQUENCY_GRID_STEP = 1.0
MASS_GRID_POINTS = 121
OUTPUT_PATH = sc.FIG_DIR / "semicoherent_sensitivity_1d.png"


def sensitivity_envelope(f_start, f_stop):
    """Return max-over-f0 optimal and semicoherent sensitivities versus Mc."""
    fspace = np.arange(f_start, f_stop + 0.5 * FREQUENCY_GRID_STEP, FREQUENCY_GRID_STEP)
    log_mspace = np.linspace(sc.LOG_M_LOWER, sc.LOG_M_UPPER, MASS_GRID_POINTS)
    mspace = 10**log_mspace
    fgrid, mgrid = np.meshgrid(fspace, mspace)

    beta_grid = sc.beta_calc(fgrid, mgrid)
    duration_grid = np.minimum(
        sc.time_to_frequency(fgrid, sc.F_END, beta_grid),
        sc.MAX_OBS_TIME,
    )
    chirp_power_grid = sc.integrated_chirp_power(duration_grid, fgrid, beta_grid)
    coherent_lambda_thresh = sc.semicoherent_noncentrality_threshold(
        1,
        false_alarm_probability=sc.DEFAULT_FALSE_ALARM_PROBABILITY,
        detection_probability=sc.DEFAULT_DETECTION_PROBABILITY,
    )

    optimal_grid = (
        sc.coherent_distance_sensitivity(
            fgrid,
            mgrid,
            chirp_power_grid,
            lambda_thresh=coherent_lambda_thresh,
        )
        / sc.PARSEC_M
    )
    semicoherent_grid = (
        sc.semicoherent_distance_sensitivity(
            fgrid,
            mgrid,
            chunk_duration=sc.CHUNK_DURATION,
            f_end=f_stop,
        )
        / sc.PARSEC_M
    )

    return {
        "mspace": mspace,
        "optimal": np.max(optimal_grid, axis=1),
        "semicoherent": np.max(semicoherent_grid, axis=1),
    }


def main():
    fig, ax = pl.subplots(figsize=(7.0, 4.8), constrained_layout=True)

    for idx, (f_start, f_stop) in enumerate(FREQUENCY_CONFIGS):
        result = sensitivity_envelope(f_start, f_stop)
        label_prefix = f"{f_start:g}-{f_stop:g} Hz"
        if (f_start, f_stop) == FREQUENCY_CONFIGS[0]:
            ax.plot(
                result["mspace"],
                result["optimal"],
                ls="--",
                label=f"Optimal",
            )
        ax.plot(
            result["mspace"],
            result["semicoherent"],
            label=label_prefix,
        )

    ax.axhline(
        sc.GALACTIC_CENTER_PC,
        color=f"C{len(FREQUENCY_CONFIGS) + 1}",
        ls=":",
        label="Galactic center",
    )
    ax.axhline(
        sc.ANDROMEDA_PC,
        color=f"C{len(FREQUENCY_CONFIGS) + 2}",
        ls=":",
        label="Andromeda",
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(r"$M_c [M_\odot$]")
    ax.set_ylabel("Distance sensitivity [pc]")
    # ax.set_title("Distance sensitivity maximized over initial frequency")
    ax.legend()

    fig.savefig(OUTPUT_PATH, bbox_inches="tight")


if __name__ == "__main__":
    main()
