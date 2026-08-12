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
from functools import lru_cache

import matplotlib as mpl
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate, integrate
from scipy.optimize import brentq
from scipy.stats import chi2, ncx2

SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PAPER_PLOTS_DIR = SCRIPT_DIR.parent

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.maximum_sensitivity_35PN import (
    beta_calc,
    distance_sensitivity as coherent_distance_sensitivity,
)

mpl.use("Agg")
pl.style.use(SCRIPT_DIR.parent / "paper.mplstyle")

# %%
PARSEC_M = 3e16
GALACTIC_CENTER_PC = 8000
ANDROMEDA_PC = 7.65e5

F_END = 2000
PLOT_F_START = 40
PLOT_F_END = 120
MAX_OBS_TIME = 3e7
CHUNK_DURATION = 30.0
LOG_M_LOWER = -5
LOG_M_UPPER = -1
DEFAULT_FALSE_ALARM_PROBABILITY = 1.0e-6
DEFAULT_DETECTION_PROBABILITY = 0.95

# %%
ASD_PATH = SCRIPT_DIR.parents[1] / "asd.txt"
FIG_DIR = SCRIPT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)

asd = np.loadtxt(ASD_PATH)
asd_freq = asd[:, 0]
asd_values = asd[:, 1]
interp_asd = interpolate.interp1d(asd_freq, asd_values)
base_integrand = 1 / (asd_freq**(7 / 3) * asd_values**2)
base_integral = integrate.cumulative_trapezoid(base_integrand, asd_freq, initial=0)
asd_freq_diff = np.diff(asd_freq)
base_integrand_slope = np.diff(base_integrand) / asd_freq_diff


def semicoherent_chunk_count_grid(signal_duration_grid, chunk_duration=CHUNK_DURATION):
    """Count fixed-duration semicoherent chunks from the 0PN signal duration."""
    signal_duration_grid = np.asarray(signal_duration_grid)
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if np.any(signal_duration_grid < 0):
        raise ValueError("signal durations must be non-negative")
    return np.maximum(
        1,
        np.ceil(signal_duration_grid / chunk_duration).astype(np.int64),
    )


@lru_cache(maxsize=None)
def _semicoherent_noncentrality_threshold_cached(
    n_coh,
    false_alarm_probability,
    detection_probability,
):
    dof = 2 * n_coh
    statistic_threshold = chi2.isf(false_alarm_probability, dof)

    def detection_probability_error(noncentrality):
        return (
            ncx2.sf(statistic_threshold, dof, noncentrality)
            - detection_probability
        )

    high = max(1.0, statistic_threshold)
    while detection_probability_error(high) < 0.0:
        high *= 2.0

    return float(brentq(detection_probability_error, 0.0, high))


def semicoherent_noncentrality_threshold(
    n_coh,
    false_alarm_probability=DEFAULT_FALSE_ALARM_PROBABILITY,
    detection_probability=DEFAULT_DETECTION_PROBABILITY,
):
    """Return Lambda required for p_det at fixed p_FA for N coherent chunks."""
    n_coh_int = int(n_coh)
    if n_coh_int != n_coh:
        raise ValueError("n_coh must be an integer")
    if n_coh_int < 1:
        raise ValueError("n_coh must be positive")
    if not 0.0 < false_alarm_probability < 1.0:
        raise ValueError("false_alarm_probability must be in (0, 1)")
    if not 0.0 < detection_probability < 1.0:
        raise ValueError("detection_probability must be in (0, 1)")
    if detection_probability <= false_alarm_probability:
        raise ValueError("detection_probability must exceed false_alarm_probability")

    return _semicoherent_noncentrality_threshold_cached(
        n_coh_int,
        float(false_alarm_probability),
        float(detection_probability),
    )


def semicoherent_penalty_factor(
    chunk_count_grid,
    false_alarm_probability=DEFAULT_FALSE_ALARM_PROBABILITY,
    detection_probability=DEFAULT_DETECTION_PROBABILITY,
):
    """Return sqrt(Lambda_thresh(1) / Lambda_thresh(N_coh))."""
    chunk_count_grid = np.asarray(chunk_count_grid)
    if np.any(~np.isfinite(chunk_count_grid)):
        raise ValueError("chunk counts must be finite")
    if np.any(chunk_count_grid != np.floor(chunk_count_grid)):
        raise ValueError("chunk counts must be integers")
    if np.any(chunk_count_grid < 1):
        raise ValueError("chunk counts must be positive")

    unique_counts, inverse = np.unique(
        chunk_count_grid.astype(np.int64),
        return_inverse=True,
    )
    coherent_lambda_thresh = semicoherent_noncentrality_threshold(
        1,
        false_alarm_probability=false_alarm_probability,
        detection_probability=detection_probability,
    )
    # Solve all independent monotonic equations together.  Calling brentq once
    # per grid value spends most of its time crossing the Python/SciPy boundary.
    dof = 2.0 * unique_counts
    statistic_thresholds = chi2.isf(false_alarm_probability, dof)
    lower = np.zeros_like(statistic_thresholds)
    upper = np.maximum(1.0, statistic_thresholds)

    below_target = (
        ncx2.sf(statistic_thresholds, dof, upper) < detection_probability
    )
    while np.any(below_target):
        upper[below_target] *= 2.0
        below_target = (
            ncx2.sf(statistic_thresholds, dof, upper) < detection_probability
        )

    # Bisection is robust here because the survival function increases
    # monotonically with noncentrality.  Sixty iterations reach double-
    # precision accuracy at the scales used by this model.
    for _ in range(60):
        midpoint = 0.5 * (lower + upper)
        below_target = (
            ncx2.sf(statistic_thresholds, dof, midpoint) < detection_probability
        )
        lower[below_target] = midpoint[below_target]
        upper[~below_target] = midpoint[~below_target]

    thresholds = 0.5 * (lower + upper)
    penalty = np.sqrt(coherent_lambda_thresh / thresholds)
    return penalty[inverse].reshape(chunk_count_grid.shape)


def apply_semicoherent_chunk_penalty(
    sensitivity_grid,
    chunk_count_grid,
    false_alarm_probability=DEFAULT_FALSE_ALARM_PROBABILITY,
    detection_probability=DEFAULT_DETECTION_PROBABILITY,
):
    """Reduce sensitivity by the noncentral-chi-square semicoherent penalty."""
    return sensitivity_grid * semicoherent_penalty_factor(
        chunk_count_grid,
        false_alarm_probability=false_alarm_probability,
        detection_probability=detection_probability,
    )


def semicoherent_distance_sensitivity(
    f0,
    Mc,
    integration=None,
    duration=None,
    chunk_duration=CHUNK_DURATION,
    f_end=None,
    false_alarm_probability=DEFAULT_FALSE_ALARM_PROBABILITY,
    detection_probability=DEFAULT_DETECTION_PROBABILITY,
    mismatch_bank=0.05,
    mismatch_coh=0.1,
):
    """Return d_sc = P(|C|) sqrt((1-mu_max)(1-M_coh)) d_opt."""
    if not 0.0 <= mismatch_bank < 1.0:
        raise ValueError("mismatch_bank must be in [0, 1)")
    if not 0.0 <= mismatch_coh < 1.0:
        raise ValueError("mismatch_coh must be in [0, 1)")

    # f_end is the semicoherent band endpoint. If it is provided, only the
    # power accumulated from f0 to f_end enters this semicoherent sensitivity.
    # If omitted, callers may provide precomputed integration/duration values;
    # otherwise the hard-coded physical endpoint F_END is used.
    if f_end is not None or integration is None or duration is None:
        endpoint = F_END if f_end is None else f_end
        beta = beta_calc(f0, Mc)
        duration = np.minimum(time_to_frequency(f0, endpoint, beta), MAX_OBS_TIME)
        integration = integrated_chirp_power(duration, f0, beta)

    lambda_thresh = semicoherent_noncentrality_threshold(
        1,
        false_alarm_probability=false_alarm_probability,
        detection_probability=detection_probability,
    )

    coherent_sensitivity = coherent_distance_sensitivity(
        f0,
        Mc,
        integration,
        lambda_thresh=lambda_thresh,
    )
    chunk_count = semicoherent_chunk_count_grid(duration, chunk_duration)
    mismatch_factor = np.sqrt((1.0 - mismatch_bank) * (1.0 - mismatch_coh))
    return mismatch_factor * apply_semicoherent_chunk_penalty(
        coherent_sensitivity,
        chunk_count,
        false_alarm_probability=false_alarm_probability,
        detection_probability=detection_probability,
    )


def f_calc(t, f0, Mc, beta=None):
    if beta is None:
        beta = beta_calc(f0, Mc)
    return f0 * (1 - 8 / 3 * beta * t)**(-3 / 8)


def time_to_frequency(f0, f_end, beta):
    """0PN chirp time from f0 to f_end."""
    f0 = np.asarray(f0)
    if np.any(f0 > f_end):
        raise ValueError("f0 must not exceed f_end")
    return 3 / (8 * beta) * (1 - (f0 / f_end)**(8 / 3))


# %%
fspace = np.linspace(PLOT_F_START, PLOT_F_END, 51)
Mspace = np.logspace(LOG_M_LOWER, LOG_M_UPPER, 49)
fgrid, Mgrid = np.meshgrid(fspace, Mspace)

betaGrid = beta_calc(fgrid, Mgrid)
tMax_grid = np.minimum(time_to_frequency(fgrid, F_END, betaGrid), MAX_OBS_TIME)

final_f = f_calc(tMax_grid, fgrid, Mgrid, betaGrid)


# %% [markdown]
# ## Sensitivity

# %%
def integrated_func(t, f0, beta):
    chirp_factor = 1 - 8 / 3 * beta * t
    temp0 = chirp_factor**(-0.5)
    temp1 = interp_asd(f0 * chirp_factor**(-3 / 8))**2
    return temp0 / temp1


def base_integral_antiderivative(frequency):
    frequency = np.asarray(frequency)
    if np.any((frequency < asd_freq[0]) | (frequency > asd_freq[-1])):
        raise ValueError("frequency is outside the ASD interpolation range")

    idx = np.searchsorted(asd_freq, frequency, side="right") - 1
    idx = np.clip(idx, 0, len(asd_freq) - 2)
    dx = frequency - asd_freq[idx]
    return (
        base_integral[idx]
        + base_integrand[idx] * dx
        + 0.5 * base_integrand_slope[idx] * dx**2
    )


def integrated_chirp_power(T, f0, beta):
    chirp_factor = 1 - 8 / 3 * beta * T
    if np.any(chirp_factor <= 0):
        raise ValueError("integration endpoint is past the chirp singularity")

    final_frequency = f0 * chirp_factor**(-3 / 8)
    return (
        f0**(4 / 3)
        / beta
        * (
            base_integral_antiderivative(final_frequency)
            - base_integral_antiderivative(f0)
        )
    )


def main():
    chirp_power_grid = integrated_chirp_power(tMax_grid, fgrid, betaGrid)
    max_sens_grid = (
        coherent_distance_sensitivity(
            fgrid,
            Mgrid,
            chirp_power_grid,
        )
        / PARSEC_M
    )
    semicoherent_sens_grid = (
        semicoherent_distance_sensitivity(
            fgrid,
            Mgrid,
            chirp_power_grid,
            tMax_grid,
            chunk_duration=CHUNK_DURATION,
        )
        / PARSEC_M
    )
    galactic_center_reachable = np.any(
        semicoherent_sens_grid >= GALACTIC_CENTER_PC,
        axis=1,
    )
    if np.any(galactic_center_reachable):
        reachable_masses = Mspace[galactic_center_reachable]
        print(
            "Mc range reaching Galactic center for f0 between:"
            f"{reachable_masses[0]:.1e} - {reachable_masses[-1]:.1e} Msun\n \n"
        )
    else:
        print("No Mc values reach the Galactic center for any f0.")

    fig, axes = pl.subplots(
        1,
        2,
        figsize=(12, 4.8),
        sharey=True,
        constrained_layout=True,
    )
    log_Mspace = np.log10(Mspace)
    plot_grids = [
        (max_sens_grid, "Coherent distance sensitivity"),
        (semicoherent_sens_grid, "Semicoherent distance sensitivity"),
    ]

    contour = None
    for ax, (sens_grid, title) in zip(axes, plot_grids):
        log_sens_grid = np.log10(sens_grid)
        contour = ax.contourf(fspace, log_Mspace, log_sens_grid, levels=range(1, 9))
        ax.contour(
            fspace,
            log_Mspace,
            log_sens_grid,
            [np.log10(GALACTIC_CENTER_PC), np.log10(ANDROMEDA_PC)],
            colors=["red", "darkorange"],
            linestyles="--",
        )
        ax.set_title(title)
        ax.set_xlabel("Initial frequency (Hz)")
        ax.set_xlim(PLOT_F_START, PLOT_F_END)

    axes[0].set_ylabel(r"$log(M_c/M_\odot$)")
    fig.colorbar(contour, ax=axes, label=r"log(Distance Sensitivity / pc)")

    line0 = Line2D([0], [0], label="Galactic center", color="r", ls="--")
    line1 = Line2D([0], [0], label="Andromeda", color="darkorange", ls="--")
    axes[0].legend(handles=[line0, line1])
    fig.savefig(
        FIG_DIR / f"distance_sensitivity_f={PLOT_F_START}-{PLOT_F_END}.png",
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
