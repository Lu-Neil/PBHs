"""Show recovered power and detection statistic versus analysis delta_beta for one fixed injection."""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as pl
import numpy as np

NOV2025_DIR = Path(__file__).resolve().parents[1]
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

from fiveVec_resampler_utils import (
    _build_time_domain_templates,
    _create_PBH_signal,
    _detection_stat,
    _estimator,
    _resample_and_extract_5vec,
    _time_domain_5vec,
)


SEED = 1
N_DELTA_BETA = 100
DELTA_BETA_MIN = 1e-12
DELTA_BETA_MAX = 1e-8
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "delta_beta_behaviour.png"


def _infer_beta_from_amplitude(t_offset, h0):
    t_last = t_offset[-1]
    return (3.0 / (8.0 * t_last)) * (1.0 - (h0[0] / h0[-1]) ** 4)


def _analysis_tau(t_offset, beta_analysis):
    tau = -(3.0 / (5.0 * beta_analysis)) * (1.0 - (8.0 / 3.0) * beta_analysis * t_offset) ** (5.0 / 8.0)
    tau -= tau[0]
    return tau


def _delta_f(t_offset, f0, beta, delta_beta):
    # tau-space frequency mismatch from linearising
    # f_tau(t) = f0 * [(1 - 8(beta+delta_beta)t/3) / (1 - 8 beta t/3)]^(3/8)
    # in delta_beta. The NUFFT bin width lives in tau-space, so this is what
    # should be compared against 1/T_obs.
    return np.abs(f0 * delta_beta * t_offset / (1.0 - (8.0 / 3.0) * beta * t_offset))


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
    signal, _, omega0, sidereal, h0, _, t = _create_PBH_signal(f0_setting="midpoint", Mc=1e1, delta_beta=0.0)
    t_offset = t.gps - t.gps[0]
    beta = _infer_beta_from_amplitude(t_offset, h0)
    f0 = omega0 / (2.0 * np.pi)
    return signal, omega0, sidereal, h0, t, t_offset, beta, f0


def _recovered_power_and_stat(signal, sidereal, t, tau, omega0):
    data_X, _ = _resample_and_extract_5vec(signal, tau, omega0)
    template_X, template_Xp, template_Xc = _time_domain_5vec(sidereal, t, tau)
    h_est = _estimator(data_X, template_X)
    hp_est = _estimator(data_X, template_Xp)
    hc_est = _estimator(data_X, template_Xc)
    power = float(np.abs(h_est) ** 2)
    det_stat = float(_detection_stat(template_Xp, template_Xc, hp_est, hc_est))
    return power, det_stat


def _recovered_power_and_stat_scan(signal, template_comb, template_p, template_c, tau, omega0):
    """Same as _recovered_power_and_stat but scans the carrier across every
    bin in the resampled spectrum and returns the maximum power and the
    maximum detection statistic over the scan.

    The time-domain templates (template_comb/p/c) are τ-independent and are
    expected to be precomputed once by the caller.
    """
    _, resampler = _resample_and_extract_5vec(signal, tau, omega0)
    template_X, _ = _resample_and_extract_5vec(template_comb, tau, 0)
    template_Xp, _ = _resample_and_extract_5vec(template_p, tau, 0)
    template_Xc, _ = _resample_and_extract_5vec(template_c, tau, 0)

    data_matrix = resampler.extract_5vec(resampler.freqs)  # (N, 5)

    h_est = data_matrix @ np.conj(template_X) / np.sum(np.abs(template_X) ** 2)
    hp_est = data_matrix @ np.conj(template_Xp) / np.sum(np.abs(template_Xp) ** 2)
    hc_est = data_matrix @ np.conj(template_Xc) / np.sum(np.abs(template_Xc) ** 2)

    power_arr = np.abs(h_est) ** 2
    det_stat_arr = (
        np.sum(np.abs(template_Xp) ** 4) * np.abs(hp_est) ** 2 + np.sum(np.abs(template_Xc) ** 4) * np.abs(hc_est) ** 2
    )

    return float(power_arr.max()), float(det_stat_arr.max())


def run_scan(delta_beta_values):
    signal, omega0, sidereal, _, t, t_offset, beta, f0 = _fixed_signal_case()

    template_comb, template_p, template_c = _build_time_domain_templates(sidereal, t)

    base_tau = _analysis_tau(t_offset, beta)
    base_power, base_det_stat = _recovered_power_and_stat_scan(
        signal, template_comb, template_p, template_c, base_tau, omega0
    )

    recovered = []
    det_stats = []
    predicted = []
    tcross_over_T = []
    for delta_beta in delta_beta_values:
        tau = _analysis_tau(t_offset, beta + delta_beta)
        power, det_stat = _recovered_power_and_stat_scan(signal, template_comb, template_p, template_c, tau, omega0)
        recovered.append(power / base_power)
        det_stats.append(det_stat / base_det_stat)

        tcross = _crossover_time(t_offset, f0, beta, delta_beta)
        predicted.append(_retained_power_from_tcross(beta, tcross, t_offset[-1]))
        tcross_over_T.append(tcross / t_offset[-1])

    c, G = 3e8, 6.67e-11
    const = 96 / 5 * np.pi ** (8 / 3) * (G / c**3) ** (5 / 3)
    Mc = (beta / (const * f0 ** (8 / 3))) ** (3 / 5)

    return {
        "recovered": np.array(recovered),
        "det_stats": np.array(det_stats),
        "predicted": np.array(predicted),
        "tcross_over_T": np.array(tcross_over_T),
        "beta_cutoff": _cutoff_delta_beta(t_offset, f0, beta),
        "T_obs": t_offset[-1],
        "f0": f0,
        "beta": beta,
        "Mc": Mc,
    }


def make_plot(delta_beta_values, results):
    fig, axes = pl.subplots(2, 1, figsize=(8.5, 9.0), sharex=True, constrained_layout=True)

    beta_mantissa, beta_exp = f"{results['beta']:.1e}".split("e")
    beta_str = rf"{beta_mantissa} \times 10^{{{int(beta_exp)}}}"
    mc_exp = np.log10(results["Mc"] / 2e30)
    title = (
        "Fixed midpoint PBH signal vs analysis "
        + r"$\Delta\beta$"
        + f"\nseed={SEED}, f0={results['f0']:.6f} Hz, Tobs={results['T_obs'] / 86164.090:.2f} d"
        + "\n"
        + rf"$\beta = {beta_str}$, $M_c = 10^{{{mc_exp:.2f}}}\,M_\odot$"
    )
    fig.suptitle(title)

    cutoff_mantissa, cutoff_exp = f"{results['beta_cutoff']:.2e}".split("e")
    cutoff_label = rf"$\beta_{{\rm cutoff}} = {cutoff_mantissa} \times 10^{{{int(cutoff_exp)}}}$"

    # --- top panel: power ---
    ax = axes[0]
    ax.plot(delta_beta_values, results["recovered"], "o-", ms=4, lw=1.5, label="Recovered 5-vector power")
    ax.plot(delta_beta_values, results["predicted"], lw=2.0, label=r"Tail-power estimate from $t_{\rm crossover}$")
    ax.axvline(
        results["beta_cutoff"],
        color="0.35",
        lw=1.5,
        ls="--",
        label=r"$t_{\rm crossover}=T_{\rm obs}$ cutoff",
    )
    ax.plot(np.nan, np.nan, ls="none", label=cutoff_label)
    ax.axhline(1.0, color="0.7", lw=1.0, ls=":")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("Recovered power fraction")
    ax.legend()

    # --- bottom panel: detection statistic ---
    ax = axes[1]
    ax.plot(delta_beta_values, results["det_stats"], "s-", ms=4, lw=1.5, color="tab:blue", label="Detection statistic")
    ax.axvline(
        results["beta_cutoff"],
        color="0.35",
        lw=1.5,
        ls="--",
        label=r"$t_{\rm crossover}=T_{\rm obs}$ cutoff",
    )
    ax.plot(np.nan, np.nan, ls="none", label=cutoff_label)
    ax.axhline(1.0, color="0.7", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"$\Delta\beta$")
    ax.set_ylabel("Detection statistic (normalised)")
    ax.legend()

    return fig


def main():
    delta_beta_values = np.logspace(np.log10(DELTA_BETA_MIN), np.log10(DELTA_BETA_MAX), N_DELTA_BETA)
    results = run_scan(delta_beta_values)
    fig = make_plot(delta_beta_values, results)
    fig.savefig(OUTPUT_PATH, dpi=200)

    print(f"Saved plot to {OUTPUT_PATH}")
    print(f"beta_cutoff = {results['beta_cutoff']:.6e}")
    print(f"beta        = {results['beta']:.6e}")
    print(f"f0          = {results['f0']:.6f} Hz")
    print(f"T_obs       = {results['T_obs'] / 86400:.6f} days")


if __name__ == "__main__":
    main()
