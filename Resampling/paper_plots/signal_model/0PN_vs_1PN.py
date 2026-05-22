import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import minimize_scalar

from plot_limits import apply_standard_plot_limits

# --- Physical Constants (SI Units) ---
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m s^-1
Msun = 2e30

# --- Binary System Parameters ---
Mc_values_msun = [1e-1, 1e-2, 1e-3]
q = 1
eta = q / (1 + q)**2

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 200.0        # Stop GW frequency (Hz)

# --- 1. Define ODEs with Event Detection ---
omega0 = np.pi * f0
omega_stop = np.pi * f_stop

def domega_dt_1pn(t, omega, M_sec):
    factor0pn = (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))
    correction1pn = 1.0 - (487.0 / 168.0) * ((M_sec * omega)**(2/3))
    return factor0pn * correction1pn

def domega_dt_0pn(t, omega, M_sec):
    return (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))

def reach_f_stop(t, omega, M_sec):
    return omega[0] - omega_stop
reach_f_stop.terminal = True
reach_f_stop.direction = 1

def sample_times(t_end, n=1200):
    t_start = max(t_end * 1e-6, 1e-3)
    return np.geomspace(t_start, t_end, n)

def f_0pn_analytic(t, f_start, M_sec):
    k_0pn = (24.0 / 5.0) * (np.pi ** (8.0 / 3.0)) * (M_sec ** (5.0 / 3.0))
    denominator = f_start ** (-8.0 / 3.0) - (8.0 / 3.0) * k_0pn * t
    if np.any(denominator <= 0.0):
        return np.full_like(t, np.nan, dtype=float)
    return denominator ** (-3.0 / 8.0)

def f_0pn_minus_1pn_theory(t, f_start, M_sec, eta):
    f_0pn = f_0pn_analytic(t, f_start, M_sec)
    psi2 = 3715.0 / 756.0 + 55.0 * eta / 9.0
    correction = (9.0 / 20.0) * psi2 * (np.pi * M_sec) ** (2.0 / 3.0)
    return 0.5 * correction * f_0pn ** (11.0 / 3.0) * (f_start ** -2.0 - f_0pn ** -2.0)

def signal_amplitude(Mc_msun, f):
    return 1.6e-24 * (Mc_msun / 1e-3) ** (5.0 / 3.0) * (f / 50.0) ** (2.0 / 3.0)

def weighted_rms_residual(t, residual, amplitude):
    weighted_residual_power = np.trapezoid((amplitude * residual) ** 2, t)
    weight_power = np.trapezoid(amplitude ** 2, t)
    return np.sqrt(weighted_residual_power / weight_power)

# --- 2. Integrate Frequency Trajectories ---
trajectory_data = []
t_max_estimate = 1e9
for Mc_msun in Mc_values_msun:
    Mc = Mc_msun * Msun
    M = Mc / (eta**(3/5))
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

    if len(sol_0pn.t_events[0]) == 0 or len(sol_1pn.t_events[0]) == 0:
        raise RuntimeError(f"Integration did not reach {f_stop} Hz for Mc = {Mc_msun:.0e} Msun.")

    t_end_0pn = sol_0pn.t_events[0][0]
    t_end_1pn = sol_1pn.t_events[0][0]

    t_0pn = sample_times(t_end_0pn)
    t_1pn = sample_times(t_end_1pn)
    f_0pn = sol_0pn.sol(t_0pn)[0] / np.pi
    f_1pn = sol_1pn.sol(t_1pn)[0] / np.pi

    t_res = sample_times(min(t_end_0pn, t_end_1pn))
    f_0pn_res = sol_0pn.sol(t_res)[0] / np.pi
    f_1pn_res = sol_1pn.sol(t_res)[0] / np.pi
    residual = f_1pn_res - f_0pn_res
    amplitude_res = signal_amplitude(Mc_msun, f_1pn_res)
    same_parameter_weighted_rms = weighted_rms_residual(t_res, residual, amplitude_res)

    trajectory_data.append(
        {
            "Mc_msun": Mc_msun,
            "t_0pn": t_0pn,
            "f_0pn": f_0pn,
            "t_1pn": t_1pn,
            "f_1pn": f_1pn,
            "t_res": t_res,
            "f_0pn_res": f_0pn_res,
            "residual": residual,
            "same_parameter_weighted_rms": same_parameter_weighted_rms,
            "t_end_0pn": t_end_0pn,
            "t_end_1pn": t_end_1pn,
        }
    )

# --- 3. Plotting ---
colors = ["crimson", "darkorange", "dodgerblue"]

fig_same = plt.figure(figsize=(9, 10))
gs_same = fig_same.add_gridspec(3, 1)
ax1 = fig_same.add_subplot(gs_same[0])
ax2 = fig_same.add_subplot(gs_same[1], sharex=ax1)
ax3 = fig_same.add_subplot(gs_same[2])
for color, data in zip(colors, trajectory_data):
    Mc_msun = data["Mc_msun"]
    ax1.plot(data["t_0pn"], data["f_0pn"], color=color, linestyle="--", label=rf"0PN, $M_c={Mc_msun:.0e}M_\odot$")
    ax1.plot(data["t_1pn"], data["f_1pn"], color=color, linestyle="-", label=rf"1PN, $M_c={Mc_msun:.0e}M_\odot$")
    residual_abs = np.maximum(np.abs(data["residual"]), np.finfo(float).tiny)
    ax2.plot(
        data["t_res"],
        residual_abs,
        color=color,
        label=rf"$M_c={Mc_msun:.0e}M_\odot$",
    )
    if np.isclose(Mc_msun, 1e-3):
        Mc = Mc_msun * Msun
        M = Mc / (eta**(3/5))
        M_sec = (G * M) / (c**3)
        theory_residual = f_0pn_minus_1pn_theory(data["t_res"], f0, M_sec, eta)
        ax2.plot(
            data["t_res"],
            np.maximum(theory_residual, np.finfo(float).tiny),
            color="blue",
            linestyle="--",
            linewidth=2.2,
            label=r"$M_c=1e-03M_\odot$, theoretical $f_\mathrm{1PN}-f_\mathrm{0PN}$",
        )
    ax3.plot(
        data["f_0pn_res"],
        residual_abs,
        color=color,
        label=rf"$M_c={Mc_msun:.0e}M_\odot$",
    )

ax1.set_xscale("log")
ax1.set_ylabel('GW Frequency [Hz]', fontsize=11)
ax1.set_title('Same-parameter 0PN and 1PN frequency trajectories', fontsize=13, fontweight='bold')
ax1.legend(loc='best', ncols=2)
ax1.grid(True, which='both', alpha=0.3)

ax2.set_yscale("log")
ax2.set_xlabel('Time [s]', fontsize=11)
ax2.set_ylabel(r'$|f_{\mathrm{1PN}} - f_{\mathrm{0PN}}|$ [Hz]', fontsize=11)
ax2.legend(loc='best')
ax2.grid(True, which='both', alpha=0.3)

ax3.set_yscale("log")
ax3.set_xlabel(r'$f_{\mathrm{0PN}}$ [Hz]', fontsize=11)
ax3.set_ylabel(r'$|f_{\mathrm{1PN}} - f_{\mathrm{0PN}}|$ [Hz]', fontsize=11)
ax3.legend(loc='best')
ax3.grid(True, which='both', alpha=0.3)

apply_standard_plot_limits(ax1, ax2, ax3)
fig_same.tight_layout()
plt.savefig("figs/0PN_vs_1PN.png")

for data in trajectory_data:
    print(
        f"Mc = {data['Mc_msun']:.0e} Msun: "
        f"0PN reaches {f_stop} Hz in {data['t_end_0pn']:.4f} s; "
        f"1PN reaches {f_stop} Hz in {data['t_end_1pn']:.4f} s; "
        f"weighted RMS residual = {data['same_parameter_weighted_rms']:.6e} Hz"
    )
