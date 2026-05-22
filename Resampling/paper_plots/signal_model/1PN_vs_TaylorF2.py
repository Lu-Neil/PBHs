import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

import lal
import lalsimulation as lalsim

from plot_limits import apply_standard_plot_limits

# --- Physical Constants (SI Units) ---
G = lal.G_SI
c = lal.C_SI

# --- Binary System Parameters ---
Mc_values_msun = [1e-1, 1e-2, 1e-3]
q = 1
eta = q / (1 + q) ** 2

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 200.0        # Stop GW frequency (Hz)

# TaylorF2 phase truncation. PNORDER_THREE_POINT_FIVE is the standard 3.5PN
# nonspinning TaylorF2 phase order used for the default comparison.
TAYLORF2_PN_PHASE_ORDER = lalsim.PNORDER_THREE_POINT_FIVE
TAYLORF2_PN_LABEL = "3.5PN"

# --- 1. Define 1PN ODE and TaylorF2 chirp helpers ---
omega0 = np.pi * f0
omega_stop = np.pi * f_stop


def domega_dt_1pn(t, omega, M_sec):
    factor0pn = (24.0 / 5.0) * (M_sec ** (5 / 3)) * (omega ** (11 / 3))
    correction1pn = 1.0 - (487.0 / 168.0) * ((M_sec * omega) ** (2 / 3))
    return factor0pn * correction1pn


def domega_dt_0pn(t, omega, M_sec):
    return (24.0 / 5.0) * (M_sec ** (5 / 3)) * (omega ** (11 / 3))


def reach_f_stop(t, omega, M_sec):
    return omega[0] - omega_stop


reach_f_stop.terminal = True
reach_f_stop.direction = 1


def sample_times(t_end, n=2000):
    t_start = 1e-1
    return np.geomspace(t_start, t_end, n)


def component_masses_from_mchirp_eta(Mc_msun, eta):
    Mc = Mc_msun * lal.MSUN_SI
    M = Mc / (eta ** (3 / 5))
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return Mc, M, m1, m2


def taylorf2_frequency_interpolator(m1, m2, pn_phase_order=TAYLORF2_PN_PHASE_ORDER, n_grid=20000):
    f_grid = np.geomspace(f0, f_stop, n_grid)
    mtot_sec = G * (m1 + m2) / c ** 3

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, pn_phase_order)
    phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)

    t_of_f = np.array(
        [lalsim.PNPhaseDerivative(f, 2, phasing, mtot_sec) / (2.0 * np.pi) for f in f_grid]
    )

    t_elapsed = t_of_f[0] - t_of_f
    valid = np.isfinite(t_elapsed) & np.isfinite(f_grid)
    t_elapsed = t_elapsed[valid]
    f_grid = f_grid[valid]

    if np.any(np.diff(t_elapsed) <= 0.0):
        raise ValueError("TaylorF2 chirp time must increase monotonically over the requested band.")

    return interp1d(t_elapsed, f_grid, kind="cubic", bounds_error=True), t_elapsed[-1]


def signal_amplitude(Mc_msun, f):
    return 1.6e-24 * (Mc_msun / 1e-3) ** (5.0 / 3.0) * (f / 50.0) ** (2.0 / 3.0)


def weighted_rms_residual(t, residual, amplitude):
    weighted_residual_power = np.trapezoid((amplitude * residual) ** 2, t)
    weight_power = np.trapezoid(amplitude ** 2, t)
    return np.sqrt(weighted_residual_power / weight_power)


# --- 2. Integrate and sample frequency trajectories ---
trajectory_data = []
t_max_estimate = 1e12
for Mc_msun in Mc_values_msun:
    Mc, M, m1, m2 = component_masses_from_mchirp_eta(Mc_msun, eta)
    M_sec = (G * M) / (c ** 3)

    sol_1pn = solve_ivp(
        domega_dt_1pn,
        (0, t_max_estimate),
        [omega0],
        args=(M_sec,),
        events=reach_f_stop,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if len(sol_1pn.t_events[0]) == 0:
        raise RuntimeError(f"1PN integration did not reach {f_stop} Hz for Mc = {Mc_msun:.0e} Msun.")

    t_end_1pn = sol_1pn.t_events[0][0]
    taylorf2_f_of_t, t_end_taylorf2 = taylorf2_frequency_interpolator(m1, m2)

    t_1pn = sample_times(t_end_1pn)
    t_taylorf2 = sample_times(t_end_taylorf2)
    f_1pn = sol_1pn.sol(t_1pn)[0] / np.pi
    f_taylorf2 = taylorf2_f_of_t(t_taylorf2)

    t_res = sample_times(min(t_end_1pn, t_end_taylorf2))
    f_1pn_res = sol_1pn.sol(t_res)[0] / np.pi
    f_taylorf2_res = taylorf2_f_of_t(t_res)
    residual = f_taylorf2_res - f_1pn_res
    amplitude_res = signal_amplitude(Mc_msun, f_taylorf2_res)
    weighted_rms = weighted_rms_residual(t_res, residual, amplitude_res)

    trajectory_data.append(
        {
            "Mc_msun": Mc_msun,
            "t_1pn": t_1pn,
            "f_1pn": f_1pn,
            "t_taylorf2": t_taylorf2,
            "f_taylorf2": f_taylorf2,
            "t_res": t_res,
            "f_1pn_res": f_1pn_res,
            "residual": residual,
            "weighted_rms": weighted_rms,
            "t_end_1pn": t_end_1pn,
            "t_end_taylorf2": t_end_taylorf2,
        }
    )


residual_comparison_Mc_msun = 1e-2
_, M_comp, m1_comp, m2_comp = component_masses_from_mchirp_eta(residual_comparison_Mc_msun, eta)
M_sec_comp = (G * M_comp) / (c ** 3)

sol_0pn_comp = solve_ivp(
    domega_dt_0pn,
    (0, t_max_estimate),
    [omega0],
    args=(M_sec_comp,),
    events=reach_f_stop,
    dense_output=True,
    rtol=1e-10,
    atol=1e-12,
)
sol_1pn_comp = solve_ivp(
    domega_dt_1pn,
    (0, t_max_estimate),
    [omega0],
    args=(M_sec_comp,),
    events=reach_f_stop,
    dense_output=True,
    rtol=1e-10,
    atol=1e-12,
)

if len(sol_0pn_comp.t_events[0]) == 0:
    raise RuntimeError(
        f"0PN integration did not reach {f_stop} Hz for Mc = {residual_comparison_Mc_msun:.0e} Msun."
    )
if len(sol_1pn_comp.t_events[0]) == 0:
    raise RuntimeError(
        f"1PN integration did not reach {f_stop} Hz for Mc = {residual_comparison_Mc_msun:.0e} Msun."
    )

taylorf2_f_of_t_comp, t_end_taylorf2_comp = taylorf2_frequency_interpolator(m1_comp, m2_comp)
t_end_0pn_comp = sol_0pn_comp.t_events[0][0]
t_end_1pn_comp = sol_1pn_comp.t_events[0][0]
t_res_comp = sample_times(min(t_end_0pn_comp, t_end_1pn_comp, t_end_taylorf2_comp), n=4000)
f_taylorf2_comp = taylorf2_f_of_t_comp(t_res_comp)
f_0pn_comp = sol_0pn_comp.sol(t_res_comp)[0] / np.pi
f_1pn_comp = sol_1pn_comp.sol(t_res_comp)[0] / np.pi
frequency_residual_comparison = {
    "t": t_res_comp,
    "f_taylorf2": f_taylorf2_comp,
    "taylorf2_minus_0pn": f_taylorf2_comp - f_0pn_comp,
    "taylorf2_minus_1pn": f_taylorf2_comp - f_1pn_comp,
    "1pn_minus_0pn": f_1pn_comp - f_0pn_comp,
}


# --- 3. Plotting ---
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(3, 1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])
colors = ["crimson", "darkorange", "dodgerblue"]
for color, data in zip(colors, trajectory_data):
    Mc_msun = data["Mc_msun"]
    ax1.plot(data["t_1pn"], data["f_1pn"], color=color, linestyle="--", label=rf"1PN, $M_c={Mc_msun:.0e}M_\odot$")
    ax1.plot(
        data["t_taylorf2"],
        data["f_taylorf2"],
        color=color,
        linestyle="-",
        label=rf"TaylorF2 {TAYLORF2_PN_LABEL}, $M_c={Mc_msun:.0e}M_\odot$",
    )
    residual_abs = np.maximum(np.abs(data["residual"]), np.finfo(float).tiny)
    ax2.plot(
        data["t_res"],
        residual_abs,
        color=color,
        label=rf"$M_c={Mc_msun:.0e}M_\odot$",
    )
    ax3.plot(
        data["f_1pn_res"],
        residual_abs,
        color=color,
        label=rf"$M_c={Mc_msun:.0e}M_\odot$",
    )

ax1.set_xscale("log")
ax1.set_ylabel("GW Frequency [Hz]", fontsize=11)
ax1.set_title(f"1PN and TaylorF2 {TAYLORF2_PN_LABEL} frequency trajectories", fontsize=13, fontweight="bold")
ax1.legend(loc="best", ncols=2)
ax1.grid(True, which="both", alpha=0.3)

ax2.set_yscale("log")
ax2.set_xlabel("Time [s]", fontsize=11)
ax2.set_ylabel(r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{1PN}}|$ [Hz]", fontsize=11)
ax2.legend(loc="best")
ax2.grid(True, which="both", alpha=0.3)

ax3.set_yscale("log")
ax3.set_xlabel(r"$f_{\mathrm{1PN}}$ [Hz]", fontsize=11)
ax3.set_ylabel(r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{1PN}}|$ [Hz]", fontsize=11)
ax3.legend(loc="best")
ax3.grid(True, which="both", alpha=0.3)

apply_standard_plot_limits(ax1, ax2, ax3)
plt.tight_layout()

fig_residual, (ax_residual_taylorf2, ax_residual_pn, ax6) = plt.subplots(
    3,
    1,
    figsize=(8.5, 7.0),
)
ax_residual_taylorf2.plot(
    frequency_residual_comparison["t"],
    np.maximum(np.abs(frequency_residual_comparison["taylorf2_minus_1pn"]), np.finfo(float).tiny),
    color="darkorange",
    linestyle="--",
    label=r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{1PN}}|$",
)
ax_residual_taylorf2.plot(
    frequency_residual_comparison["t"],
    np.maximum(np.abs(frequency_residual_comparison["taylorf2_minus_0pn"]), np.finfo(float).tiny),
    color="darkorange",
    linestyle="-",
    label=r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{0PN}}|$",
)
ax_residual_taylorf2.plot(
    frequency_residual_comparison["t"],
    1/frequency_residual_comparison["t"],
    color="grey",
    linestyle=":",
    label=r"Frequency bin size",
)
ax_residual_taylorf2.set_xscale("log")
ax_residual_taylorf2.set_yscale("log")
ax_residual_taylorf2.set_ylabel(r"$\Delta f$ [Hz]", fontsize=11)
ax_residual_taylorf2.set_title(
    rf"TaylorF2 {TAYLORF2_PN_LABEL} residual comparison, $M_c={residual_comparison_Mc_msun:.0e}M_\odot$",
    fontsize=13,
    fontweight="bold",
)
ax_residual_taylorf2.legend(loc="best")
ax_residual_taylorf2.grid(True, which="both", alpha=0.3)

ax_residual_pn.plot(
    frequency_residual_comparison["t"],
    np.maximum(np.abs(frequency_residual_comparison["1pn_minus_0pn"]), np.finfo(float).tiny),
    color="darkorange",
    label=r"$|f_{\mathrm{1PN}} - f_{\mathrm{0PN}}|$",
)

ax_residual_pn.set_xscale("log")
ax_residual_pn.set_yscale("log")
ax_residual_pn.set_xlabel("Time [s]", fontsize=11)
ax_residual_pn.set_ylabel(r"$\Delta f$ [Hz]", fontsize=11)
ax_residual_pn.legend(loc="best")
ax_residual_pn.grid(True, which="both", alpha=0.3)

ax6.plot(
    frequency_residual_comparison["f_taylorf2"],
    np.maximum(np.abs(frequency_residual_comparison["taylorf2_minus_1pn"]), np.finfo(float).tiny),
    color="darkorange",
    linestyle="--",
    label=r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{1PN}}|$",
)
ax6.plot(
    frequency_residual_comparison["f_taylorf2"],
    np.maximum(np.abs(frequency_residual_comparison["taylorf2_minus_0pn"]), np.finfo(float).tiny),
    color="darkorange",
    linestyle="-",
    label=r"$|f_{\mathrm{TaylorF2}} - f_{\mathrm{0PN}}|$",
)
ax6.set_xscale("log")
ax6.set_yscale("log")
ax6.set_xlabel(r"$f_{\mathrm{TaylorF2}}$ [Hz]", fontsize=11)
ax6.set_ylabel(r"$\Delta f$ [Hz]", fontsize=11)
ax6.legend(loc="best")
ax6.grid(True, which="both", alpha=0.3)
fig_residual.tight_layout()

plt.savefig("figs/1PN_vs_3.5PN.png")

for data in trajectory_data:
    print(
        f"Mc = {data['Mc_msun']:.0e} Msun: "
        f"1PN reaches {f_stop} Hz in {data['t_end_1pn']:.4f} s; "
        f"TaylorF2 {TAYLORF2_PN_LABEL} reaches {f_stop} Hz in {data['t_end_taylorf2']:.4f} s; "
        f"weighted RMS residual = {data['weighted_rms']:.6e} Hz"
    )
