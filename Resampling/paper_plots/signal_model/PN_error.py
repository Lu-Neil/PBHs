import numpy as np
import matplotlib
from pathlib import Path
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
import lal
import lalsimulation as lalsim
from plot_limits import apply_standard_plot_limits

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
plt.style.use(SCRIPT_DIR.parent / "paper.mplstyle")

# --- Physical Constants (SI Units) ---
G = lal.G_SI
c = lal.C_SI

# --- Binary System Parameters ---
Mc_values_msun = [1e-1, 1e-3]
q = 1
eta = q / (1 + q)**2

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 200.0        # Stop GW frequency (Hz)

# --- 1. Define 0PN ODE and TaylorF2 chirp helpers ---
TAYLORF2_35PN_ORDER = lalsim.PNORDER_THREE_POINT_FIVE
TAYLORF2_1PN_ORDER = lalsim.PNORDER_ONE

omega0 = np.pi * f0
omega_stop = np.pi * f_stop


def domega_dt_0pn(t, omega, M_sec):
    return (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))


def reach_f_stop(t, omega, M_sec):
    return omega[0] - omega_stop


reach_f_stop.terminal = True
reach_f_stop.direction = 1


def sample_times(t_end, n=2000):
    t_start = 1e-1
    return np.geomspace(t_start, t_end, n)


def component_masses_from_mchirp_eta(Mc_msun, eta):
    Mc = Mc_msun * lal.MSUN_SI
    M = Mc / (eta**(3/5))
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return Mc, M, m1, m2


def mchirp_latex(Mc_msun):
    exponent = int(np.round(np.log10(Mc_msun)))
    if np.isclose(Mc_msun, 10.0**exponent):
        return rf"10^{{{exponent}}}"
    return f"{Mc_msun:g}"


def truncate_taylorf2_phasing(phasing, pn_phase_order):
    """Zero TaylorF2 phasing coefficients above the requested PN order."""
    max_order = lalsim.PNORDER_THREE_POINT_FIVE
    for order in range(pn_phase_order + 1, max_order + 1):
        phasing.v[order] = 0.0
        phasing.vlogv[order] = 0.0
        phasing.vlogvsq[order] = 0.0
    return phasing


def taylorf2_frequency_interpolator(m1, m2, pn_phase_order=TAYLORF2_35PN_ORDER, n_grid=20000):
    f_grid = np.geomspace(f0, f_stop, n_grid)
    mtot_sec = G * (m1 + m2) / c**3
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, pn_phase_order)
    phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)
    phasing = truncate_taylorf2_phasing(phasing, pn_phase_order)
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


# --- 2. Integrate and sample frequency trajectories ---
trajectory_data = []
t_max_estimate = 1e12
for Mc_msun in Mc_values_msun:
    Mc, M, m1, m2 = component_masses_from_mchirp_eta(Mc_msun, eta)
    M_sec = (G * M) / (c**3)

    sol_0pn = solve_ivp(
        domega_dt_0pn,
        (0, t_max_estimate),
        [omega0],
        args=(M_sec,),
        events=reach_f_stop,
        dense_output=True,
        rtol=1e-10,
        atol=1e-12,
    )

    if len(sol_0pn.t_events[0]) == 0:
        raise RuntimeError(f"0PN integration did not reach {f_stop} Hz for Mc = {Mc_msun:.0e} Msun.")

    t_end_0pn = sol_0pn.t_events[0][0]
    taylorf2_f_of_t, t_end_taylorf2 = taylorf2_frequency_interpolator(
        m1, m2, TAYLORF2_35PN_ORDER
    )
    taylorf2_1pn_f_of_t, t_end_taylorf2_1pn = taylorf2_frequency_interpolator(
        m1, m2, TAYLORF2_1PN_ORDER
    )

    t_0pn = sample_times(t_end_0pn)
    t_taylorf2 = sample_times(t_end_taylorf2)
    t_taylorf2_1pn = sample_times(t_end_taylorf2_1pn)
    f_0pn = sol_0pn.sol(t_0pn)[0] / np.pi
    f_taylorf2 = taylorf2_f_of_t(t_taylorf2)
    f_taylorf2_1pn = taylorf2_1pn_f_of_t(t_taylorf2_1pn)

    t_res = sample_times(min(t_end_0pn, t_end_taylorf2))
    f_0pn_res = sol_0pn.sol(t_res)[0] / np.pi
    f_taylorf2_res = taylorf2_f_of_t(t_res)
    residual = f_taylorf2_res - f_0pn_res

    t_res_pn = sample_times(min(t_end_taylorf2, t_end_taylorf2_1pn))
    f_taylorf2_res_pn = taylorf2_f_of_t(t_res_pn)
    f_taylorf2_1pn_res = taylorf2_1pn_f_of_t(t_res_pn)
    residual_35pn_minus_1pn = f_taylorf2_res_pn - f_taylorf2_1pn_res

    trajectory_data.append(
        {
            "Mc_msun": Mc_msun,
            "t_0pn": t_0pn,
            "f_0pn": f_0pn,
            "t_taylorf2": t_taylorf2,
            "f_taylorf2": f_taylorf2,
            "t_taylorf2_1pn": t_taylorf2_1pn,
            "f_taylorf2_1pn": f_taylorf2_1pn,
            "t_res": t_res,
            "f_0pn_res": f_0pn_res,
            "residual": residual,
            "t_res_pn": t_res_pn,
            "f_taylorf2_1pn_res": f_taylorf2_1pn_res,
            "residual_35pn_minus_1pn": residual_35pn_minus_1pn,
            "t_end_0pn": t_end_0pn,
            "t_end_taylorf2": t_end_taylorf2,
            "t_end_taylorf2_1pn": t_end_taylorf2_1pn,
        }
    )

# --- 3. Plotting ---
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(2, 1)
ax2 = fig.add_subplot(gs[0])
ax3 = fig.add_subplot(gs[1])
colors = ["crimson", "dodgerblue"]
for color, data in zip(colors, trajectory_data):
    Mc_msun = data["Mc_msun"]
    mc_label = mchirp_latex(Mc_msun)
    residual_abs = np.maximum(np.abs(data["residual"]), np.finfo(float).tiny)
    residual_35pn_minus_1pn_abs = np.maximum(
        np.abs(data["residual_35pn_minus_1pn"]), np.finfo(float).tiny
    )
    ax2.plot(
        data["t_res"],
        residual_abs,
        color=color,
        alpha=0.65,
        label=rf"$M_c={mc_label}\,M_\odot$",
    )
    # ax2.plot(
    #     data["t_res_pn"],
    #     residual_35pn_minus_1pn_abs,
    #     color=color,
    #     linestyle="--",
    #     label=rf"$f_{{\mathrm{{model}}}}=\mathrm{{1PN}}$, $M_c={mc_label}\,M_\odot$",
    # )
    
    ax3.plot(
        data["f_taylorf2"],
        residual_abs,
        color=color,
        alpha=0.65,
        label=rf"$M_c={mc_label}\,M_\odot$",
    )
    # ax3.plot(
    #     data["f_taylorf2"],
    #     residual_35pn_minus_1pn_abs,
    #     color=color,
    #     linestyle="--",
    #     label=rf"$f_{{\mathrm{{model}}}}=\mathrm{{1PN}}$, $M_c={mc_label}\,M_\odot$",
    # )
ax2.plot(
    data["t_res_pn"],
    1/data["t_res_pn"],
    color='grey',
    linestyle="-",
    label="Fourier bin width",
)
ax2.set_yscale("log")
ax2.set_xscale("log")
ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"$|f_{\rm 3.5PN} - f_{\rm 0PN}|$ [Hz]")
ax2.legend(loc="best")
ax2.grid(True, which="both", alpha=0.3)

ax3.set_yscale("log")
ax3.set_xscale("log")
ax3.set_ylim(1e-6)
ax3_ticks = np.sort(
    np.unique(np.append(ax3.get_xticks(), [20, 50]))
)
ax3_ticks = ax3_ticks[(ax3_ticks >= f0) & (ax3_ticks <= f_stop)]
ax3.set_xticks(ax3_ticks)
ax3.set_xticklabels([f"{tick:g}" for tick in ax3_ticks])
ax3.set_xlabel(r"$f_{\rm 3.5PN}$ [Hz]")
ax3.set_ylabel(r"$|f_{\rm 3.5PN} - f_{\rm 0PN}|$ [Hz]")
ax3.legend(loc="best")
ax3.grid(True, which="both", alpha=0.3)

# apply_standard_plot_limits(ax2, ax3)
plt.tight_layout()
plt.savefig(SCRIPT_DIR / "figs" / "PN_error.png")
