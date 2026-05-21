import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

# --- Physical Constants (SI Units) ---
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m s^-1

# --- Binary System Parameters ---
Mc = 1e-1 * 2e30
q=1
    
# Calculate masses:
eta = q / (1 + q)**2
M = Mc / (eta**(3/5))
m2 = M / (1 + q)
m1 = q * m2
M_sec = (G * M) / (c**3) 

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 200.0        # Stop GW frequency (Hz)

omega0 = np.pi * f0
omega_stop = np.pi * f_stop

# --- 1. Define ODEs with Event Detection ---
def domega_dt_1pn(t, omega):
    factor0pn = (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))
    correction1pn = 1.0 - (487.0 / 168.0) * ((M_sec * omega)**(2/3))
    return factor0pn * correction1pn

def domega_dt_0pn(t, omega):
    return (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))

# Termination event when omega reaches omega_stop
def reach_f_stop(t, omega):
    return omega[0] - omega_stop
reach_f_stop.terminal = True

# --- 2. Integrate Both Systems ---
# An upper limit estimate for time duration to ensure the solver completes
t_max_estimate = 1e9

# Evolve 1PN
sol_1pn = solve_ivp(domega_dt_1pn, (0, t_max_estimate), [omega0], 
                    events=reach_f_stop, rtol=1e-10, atol=1e-12)

# Evolve 0PN
sol_0pn = solve_ivp(domega_dt_0pn, (0, t_max_estimate), [omega0], 
                    events=reach_f_stop, rtol=1e-10, atol=1e-12)

# Extract time and convert orbital frequency to GW frequency (f = omega / pi)
t_1pn = sol_1pn.t
f_1pn = sol_1pn.y[0] / np.pi

t_0pn = sol_0pn.t
f_0pn = sol_0pn.y[0] / np.pi

# --- 3. Compute Residuals via Interpolation ---
# Since 0PN reaches f_stop faster, we evaluate the residual over the 0PN lifespan
t_common = t_1pn[t_1pn <= t_0pn[-1]]
f_1pn_interp = f_1pn[t_1pn <= t_0pn[-1]]

interp_0pn = interp1d(t_0pn, f_0pn, kind='cubic')
f_0pn_interp = interp_0pn(t_common)

residual = f_1pn_interp - f_0pn_interp

# --- 4. Noise-weighted stationary-phase power ---
asd_data = np.loadtxt("../asd.txt")
asd_freq = asd_data[:, 0]
asd = asd_data[:, 1]
valid_asd = np.isfinite(asd_freq) & np.isfinite(asd) & (asd_freq > 0.0) & (asd > 0.0)
asd_freq = asd_freq[valid_asd]
asd = asd[valid_asd]

sort_idx = np.argsort(asd_freq)
asd_freq = asd_freq[sort_idx]
asd = asd[sort_idx]

f_power_min = max(f0, asd_freq[0])
f_power_max = min(f_stop, asd_freq[-1])
if f_power_min >= f_power_max:
    raise ValueError(
        f"No overlap between requested band [{f0}, {f_stop}] Hz and ASD "
        f"band [{asd_freq[0]}, {asd_freq[-1]}] Hz."
    )

f_power = np.logspace(np.log10(f_power_min), np.log10(f_power_max), 2000)
log_psd_interp = interp1d(np.log(asd_freq), np.log(asd**2), kind="linear", bounds_error=True)
Sn_power = np.exp(log_psd_interp(np.log(f_power)))

dfdt_0pn = np.asarray(domega_dt_0pn(0, np.pi * f_power)) / np.pi
dfdt_1pn = np.asarray(domega_dt_1pn(0, np.pi * f_power)) / np.pi
if np.any(dfdt_0pn <= 0.0) or np.any(dfdt_1pn <= 0.0):
    raise ValueError("df/dt must remain positive over the plotted frequency band.")

# Common distance factors cancel in the normalized curves, so d is omitted.
h0_shape = 4.0 * (G * Mc / c**2)**(5.0 / 3.0) * (np.pi * f_power / c)**(2.0 / 3.0)
p0 = h0_shape**2 / (dfdt_0pn * Sn_power)
p1 = h0_shape**2 / (dfdt_1pn * Sn_power)
p_norm = max(p0.max(), p1.max())
p0_norm = p0 / p_norm
p1_norm = p1 / p_norm

# --- 5. Plotting ---
fig = plt.figure(figsize=(9, 10))
gs = fig.add_gridspec(3, 1, height_ratios=[1.0, 1.0, 1.1])
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2])

# Top Subplot: Frequency Evolutions
ax1.plot(t_0pn, f_0pn, label='0PN', color='crimson', linestyle='--')
ax1.plot(t_1pn, f_1pn, label='1PN', color='dodgerblue')
ax1.axhline(f_stop, color='gray', linestyle=':', alpha=0.7, label=f'f_stop ({f_stop} Hz)')
ax1.set_ylabel('GW Frequency [Hz]', fontsize=11)
ax1.set_title(rf'Mc = {Mc/2e30:.1e}$M_\odot$', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# Bottom Subplot: Residuals
ax2.semilogy(t_common, abs(residual), color='purple', label=r'$\Delta f$ ($f_{\mathrm{1PN}} - f_{\mathrm{0PN}}$)')
ax2.set_xlabel('Time [s]', fontsize=11)
ax2.set_ylabel('Residual [Hz]', fontsize=11)
ax2.legend(loc='lower left')
ax2.grid(True, alpha=0.3)

# Third Subplot: Noise-weighted power
ax3.loglog(f_power, p0_norm, label='0PN', color='crimson', linestyle='--')
ax3.loglog(f_power, p1_norm, label='1PN', color='dodgerblue')
ax3.set_xlabel('GW Frequency [Hz]', fontsize=11)
ax3.set_ylabel('Normalized noise-weighted power', fontsize=11)
ax3.legend(loc='best')
ax3.grid(True, which='both', alpha=0.3)

plt.tight_layout()
plt.show()

# Print termination diagnostics
print(f"0PN reached {f_stop} Hz in: {t_0pn[-1]:.4f} seconds")
print(f"1PN reached {f_stop} Hz in: {t_1pn[-1]:.4f} seconds")
print(
    f"Noise-weighted power evaluated over {f_power[0]:.2f}-{f_power[-1]:.2f} Hz; "
    f"min df/dt 1PN = {dfdt_1pn.min():.6e} Hz/s"
)
