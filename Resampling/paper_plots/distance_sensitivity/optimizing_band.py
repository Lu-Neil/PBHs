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
import argparse
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as pl
import numpy as np
from joblib import Parallel, delayed

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PAPER_PLOTS_DIR = SCRIPT_DIR.parent

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity import semicoherent_sensitivity as sc
from signal_generators import NoiseCurve
from template_bank.coherent_chunk.pn_mismatch import (
    ZeroPNMismatchConfig,
    find_allowed_duration,
)

mpl.use("Agg")
pl.style.use(SCRIPT_DIR.parent / "paper.mplstyle")

# %%
FREQUENCY_GRID_STEP = 1.0
MASS_GRID_POINTS = 121
OUTPUT_PATH = sc.FIG_DIR / "semicoherent_sensitivity_1d.png"

# These are intentionally different criteria: the first bounds the
# 0PN-vs-3.5PN waveform overlap, while the second is the coherent power loss
# used in the distance-sensitivity model.
COHERENCE_WAVEFORM_MISMATCH = 0.05
SENSITIVITY_COHERENT_POWER_MISMATCH = 0.1
COHERENCE_SEARCH_TIME_MAX = 2000.0
COHERENCE_SAMPLE_RATE = 512.0
COHERENCE_ETA = 0.25


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Plot optimal and semicoherent sensitivity for search bands."
    )
    parser.add_argument(
        "--frequency-config",
        dest="frequency_configs",
        action="append",
        nargs=2,
        type=float,
        required=True,
        metavar=("F_START", "F_STOP"),
        help=(
            "Search-band endpoints in Hz. Repeat this option to plot multiple "
            "bands."
        ),
    )
    args = parser.parse_args(argv)

    for f_start, f_stop in args.frequency_configs:
        if f_start <= 0.0 or f_stop <= f_start:
            parser.error(
                "each frequency config must satisfy 0 < F_START < F_STOP"
            )

    args.frequency_configs = [tuple(config) for config in args.frequency_configs]
    return args


def maximum_coherence_duration(f_start, f_stop):
    """Return the 0PN-vs-3.5PN coherence limit for one search band."""
    if f_start <= 0.0 or f_stop <= f_start:
        raise ValueError("The frequency band must satisfy 0 < f_start < f_stop.")

    search_time_max = COHERENCE_SEARCH_TIME_MAX
    sample_rate = max(COHERENCE_SAMPLE_RATE, 2.5 * f_stop)
    noise = NoiseCurve.from_asd_file(sc.ASD_PATH)

    while True:
        config = ZeroPNMismatchConfig(
            f_min=f_start,
            f_max=f_stop,
            mchirp_min=10**sc.LOG_M_LOWER,
            mchirp_max=10**sc.LOG_M_UPPER,
            eta=COHERENCE_ETA,
            max_mismatch=COHERENCE_WAVEFORM_MISMATCH,
            search_time_max=search_time_max,
            sample_rate=sample_rate,
        )
        try:
            duration, _ = find_allowed_duration(config, noise)
            return duration
        except ValueError as error:
            if "reaches ISCO before t_max" not in str(error):
                raise
            search_time_max *= 0.5
            if search_time_max <= 8.0 / sample_rate:
                raise RuntimeError(
                    "Could not bracket the coherence limit before the "
                    "TaylorT4 track reaches ISCO."
                ) from error


def sensitivity_envelope(f_start, f_stop):
    """Return max-over-f0 optimal and semicoherent sensitivities versus Mc."""
    coherence_duration = maximum_coherence_duration(f_start, f_stop)
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
            chunk_duration=coherence_duration,
            f_end=f_stop,
            mismatch_coh=SENSITIVITY_COHERENT_POWER_MISMATCH,
        )
        / sc.PARSEC_M
    )

    return {
        "mspace": mspace,
        "optimal": np.max(optimal_grid, axis=1),
        "semicoherent": np.max(semicoherent_grid, axis=1),
        "coherence_duration": coherence_duration,
    }


def format_coherence_duration(duration):
    """Format a coherence duration with one significant figure."""
    return np.format_float_positional(
        duration,
        precision=1,
        unique=False,
        fractional=False,
        trim="-",
    )


def main(frequency_configs):
    fig, ax = pl.subplots(figsize=(7.0, 4.8), constrained_layout=True)

    results = Parallel(n_jobs=4, prefer="processes")(
        delayed(sensitivity_envelope)(f_start, f_stop)
        for f_start, f_stop in frequency_configs
    )

    for idx, ((f_start, f_stop), result) in enumerate(
        zip(frequency_configs, results)
    ):
        label_prefix = f"{f_start:g}-{f_stop:g} Hz"
        if idx == 0:
            ax.plot(
                result["mspace"],
                result["optimal"],
                color=f"C{idx + 1}",
                ls="-",
                label="Optimal",
            )
        ax.plot(
            result["mspace"],
            result["semicoherent"],
            color=f"C{idx + 2}",
            ls="-",
            label=("Semicoherent, "
                f"{label_prefix}, "
                rf"$T_{{\rm coh}}="
                rf"{format_coherence_duration(result['coherence_duration'])}"
                rf"\,\rm s$"
            ),
        )

    ax.axhline(
        sc.GALACTIC_CENTER_PC,
        color="red",
        ls="--",
        label="Galactic center",
    )
    ax.axhline(
        sc.ANDROMEDA_PC,
        color="darkorange",
        ls="--",
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
    main(parse_args().frequency_configs)
