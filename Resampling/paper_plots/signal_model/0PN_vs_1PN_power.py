import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# --- Physical Constants (SI Units) ---
G = 6.67430e-11  # m^3 kg^-1 s^-2
c = 299792458.0  # m s^-1

# --- Binary System Parameters ---
Mc_values_msun = [1e-1, 1e-2, 1e-3]
q = 1
eta = q / (1 + q)**2

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 500.0        # Stop GW frequency (Hz)

# --- 1. Define Frequency Evolution ---
def domega_dt_1pn(omega, M_sec):
    factor0pn = (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))
    correction1pn = 1.0 - (487.0 / 168.0) * ((M_sec * omega)**(2/3))
    return factor0pn * correction1pn

def domega_dt_0pn(omega, M_sec):
    return (24.0 / 5.0) * (M_sec**(5/3)) * (omega**(11/3))

# --- 2. Noise PSD ---
asd_data = np.loadtxt("../../asd.txt")
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

# --- 3. Noise-weighted stationary-phase power ---
power_curves = []
for Mc_msun in Mc_values_msun:
    Mc = Mc_msun * 2e30
    M = Mc / (eta**(3/5))
    M_sec = (G * M) / (c**3)

    dfdt_0pn = np.asarray(domega_dt_0pn(np.pi * f_power, M_sec)) / np.pi
    dfdt_1pn = np.asarray(domega_dt_1pn(np.pi * f_power, M_sec)) / np.pi
    if np.any(dfdt_0pn <= 0.0) or np.any(dfdt_1pn <= 0.0):
        raise ValueError(f"df/dt must remain positive for Mc = {Mc_msun:.0e} Msun.")

    # Common distance factors cancel in the normalized curves, so d is omitted.
    h0_shape = 4.0 * (G * Mc / c**2)**(5.0 / 3.0) * (np.pi * f_power / c)**(2.0 / 3.0)
    p0 = h0_shape**2 / (dfdt_0pn * Sn_power)
    p1 = h0_shape**2 / (dfdt_1pn * Sn_power)
    power_curves.append((Mc_msun, p0, p1, dfdt_1pn.min()))

# --- 4. Plotting ---
fig, ax3 = plt.subplots(figsize=(9, 5.5))
colors = ["crimson", "darkorange", "dodgerblue"]
for color, (Mc_msun, p0, p1, _) in zip(colors, power_curves):
    ax3.loglog(f_power, p0 / np.sum(p0), color=color, linestyle="--", label=rf"0PN, $M_c={Mc_msun:.0e}M_\odot$")
    ax3.loglog(f_power, p1 / np.sum(p1), color=color, linestyle="-", label=rf"1PN, $M_c={Mc_msun:.0e}M_\odot$")

ax3.set_xlabel('GW Frequency [Hz]', fontsize=11)
ax3.set_ylabel('Sum-normalized noise-weighted power', fontsize=11)
ax3.set_title('Noise-weighted stationary-phase power', fontsize=13, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, which='both', alpha=0.3)

plt.tight_layout()

print(f"Noise-weighted power evaluated over {f_power[0]:.2f}-{f_power[-1]:.2f} Hz")
for Mc_msun, _, _, min_dfdt_1pn in power_curves:
    print(f"Mc = {Mc_msun:.0e} Msun: min df/dt 1PN = {min_dfdt_1pn:.6e} Hz/s")
