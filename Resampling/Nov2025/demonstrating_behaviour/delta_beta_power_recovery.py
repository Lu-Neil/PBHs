"""Show recovered PBH power versus analysis delta_beta for one fixed injection."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as pl
import numpy as np

NOV2025_DIR = Path(__file__).resolve().parents[1]
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import _create_PBH_signal, _estimator, _resample_and_extract_5vec, _time_domain_5vec


SEED = 1
N_DELTA_BETA = 80
DELTA_BETA_MIN = 1e-12
DELTA_BETA_MAX = 1e-8
OUTPUT_PATH = Path(__file__).resolve().parent / "delta_beta_power_recovery.png"


def _infer_beta_from_amplitude(t_offset, h0):
    t_last = t_offset[-1]
    return (3.0 / (8.0 * t_last)) * (1.0 - (h0[0] / h0[-1]) ** 4)


def _analysis_tau(t_offset, beta_analysis):
    tau = -(3.0 / (5.0 * beta_analysis)) * (1.0 - (8.0 / 3.0) * beta_analysis * t_offset) ** (5.0 / 8.0)
    tau -= tau[0]
    return tau


def _delta_f(t_offset, f0, beta, delta_beta):
    temp0 = (3.0 * f0 * delta_beta) / (5.0 * beta * (1.0 - (8.0 / 3.0) * beta * t_offset) ** (3.0 / 8.0))
    temp1 = 1.0 + (-1.0 + beta * t_offset) / (1.0 - (8.0 / 3.0) * beta * t_offset)
    return np.abs(temp0 * temp1)


def _crossover_time(t_offset, f0, beta, delta_beta):
    T_obs = t_offset[-1]
    crossed = np.flatnonzero(_delta_f(t_offset, f0, beta, delta_beta) >= 1.0 / T_obs)
    if crossed.size == 0:
        return T_obs
    return t_offset[crossed[0]]


def _retained_power_from_tcross(beta, tcross, T_obs):
    if tcross >= T_obs:
        return 1.0

    edge_tcross = np.sqrt(1.0 - (8.0 / 3.0) * beta * tcross)
    edge_T = np.sqrt(1.0 - (8.0 / 3.0) * beta * T_obs)
    loss_fraction = (edge_tcross - edge_T) / (1.0 - edge_T)
    return 1.0 - loss_fraction


def _cutoff_delta_beta(t_offset, f0, beta):
    T_obs = t_offset[-1]
    slope = _delta_f(t_offset, f0, beta, 1.0)[-1]
    return (1.0 / T_obs) / slope


def _fixed_signal_case(seed=SEED):
    np.random.seed(seed)
    signal, _, omega0, sidereal, h0, _, t = _create_PBH_signal(f0_setting="midpoint", delta_beta=0.0)
    t_offset = t.gps - t.gps[0]
    beta = _infer_beta_from_amplitude(t_offset, h0)
    f0 = omega0 / (2.0 * np.pi)
    return signal, omega0, sidereal, h0, t, t_offset, beta, f0


def _recovered_power(signal, sidereal, t, tau, omega0):
    data_X, _ = _resample_and_extract_5vec(signal, tau, omega0)
    template_X, _, _ = _time_domain_5vec(sidereal, t, tau)
    h_est = _estimator(data_X, template_X)
    return float(np.abs(h_est) ** 2)


def run_scan(delta_beta_values):
    signal, omega0, sidereal, _, t, t_offset, beta, f0 = _fixed_signal_case()

    base_tau = _analysis_tau(t_offset, beta)
    base_power = _recovered_power(signal, sidereal, t, base_tau, omega0)

    recovered = []
    predicted = []
    tcross_over_T = []
    for delta_beta in delta_beta_values:
        tau = _analysis_tau(t_offset, beta + delta_beta)
        recovered.append(_recovered_power(signal, sidereal, t, tau, omega0) / base_power)

        tcross = _crossover_time(t_offset, f0, beta, delta_beta)
        predicted.append(_retained_power_from_tcross(beta, tcross, t_offset[-1]))
        tcross_over_T.append(tcross / t_offset[-1])

    return {
        "recovered": np.array(recovered),
        "predicted": np.array(predicted),
        "tcross_over_T": np.array(tcross_over_T),
        "beta_cutoff": _cutoff_delta_beta(t_offset, f0, beta),
        "T_obs": t_offset[-1],
        "f0": f0,
        "beta": beta,
    }


def make_plot(delta_beta_values, results):
    fig, ax = pl.subplots(figsize=(8.5, 5.5), constrained_layout=True)

    ax.plot(delta_beta_values, results["recovered"], "o-", ms=4, lw=1.5, label="Recovered 5-vector power")
    ax.plot(delta_beta_values, results["predicted"], lw=2.0, label=r"Tail-power estimate from $t_{\rm crossover}$")
    ax.axvline(
        results["beta_cutoff"],
        color="0.35",
        lw=1.5,
        ls="--",
        label=r"$t_{\rm crossover}=T_{\rm obs}$ cutoff",
    )
    ax.axhline(1.0, color="0.7", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\Delta\beta$")
    ax.set_ylabel("Recovered power fraction")
    ax.set_title(
        "Fixed midpoint PBH signal: recovered power vs analysis "
        + r"$\Delta\beta$"
        + f"\nseed={SEED}, f0={results['f0']:.6f} Hz, Tobs={results['T_obs'] / 86164.090:.2f} d"
    )
    ax.legend()

    ax.text(
        0.98,
        0.04,
        r"$\beta_{\rm cutoff}$" + f" = {results['beta_cutoff']:.2e}\n" + r"$\beta$" + f" = {results['beta']:.2e}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=10,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="0.8"),
    )

    return fig


def main():
    delta_beta_values = np.logspace(np.log10(DELTA_BETA_MIN), np.log10(DELTA_BETA_MAX), N_DELTA_BETA)
    results = run_scan(delta_beta_values)
    fig = make_plot(delta_beta_values, results)
    fig.savefig(OUTPUT_PATH, dpi=200)

    print(f"Saved plot to {OUTPUT_PATH}")
    print(f"beta_cutoff = {results['beta_cutoff']:.6e}")
    print(f"beta = {results['beta']:.6e}")
    print(f"f0 = {results['f0']:.6f} Hz")
    print(f"T_obs = {results['T_obs'] / 86400:.6f} days")


if __name__ == "__main__":
    main()
