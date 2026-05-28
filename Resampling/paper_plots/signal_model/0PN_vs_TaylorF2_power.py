from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d

import lal
import lalsimulation as lalsim


# --- Physical Constants (SI Units) ---
G = lal.G_SI
c = lal.C_SI

# --- Binary System Parameters ---
Mc_values_msun = [1e-1, 1e-2, 1e-3]
q = 1
eta = q / (1 + q) ** 2

# --- Simulation Limits ---
f0 = 20.0             # Start GW frequency (Hz)
f_stop = 500.0        # Stop GW frequency (Hz)

# TaylorF2 phase truncation. PNORDER_THREE_POINT_FIVE is the standard 3.5PN
# nonspinning TaylorF2 phase order used for the default comparison.
TAYLORF2_PN_PHASE_ORDER = lalsim.PNORDER_THREE_POINT_FIVE
TAYLORF2_PN_LABEL = "3.5PN"


# --- 1. Define Frequency Evolution ---
def domega_dt_0pn(omega, M_sec):
    return (24.0 / 5.0) * (M_sec ** (5 / 3)) * (omega ** (11 / 3))


def component_masses_from_mchirp_eta(Mc_msun, eta):
    Mc = Mc_msun * lal.MSUN_SI
    M = Mc / (eta ** (3 / 5))
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return Mc, M, m1, m2


def taylorf2_dfdt(f, m1, m2, pn_phase_order=TAYLORF2_PN_PHASE_ORDER):
    """Return df/dt from the TaylorF2 stationary-phase time-frequency map."""
    mtot_sec = G * (m1 + m2) / c**3
    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(params, pn_phase_order)
    phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)

    # PNPhaseDerivative = dPsi/df = 2 pi t(f).  The elapsed chirp time has
    # dt_elapsed/df = -d t(f)/df, so df/dt_elapsed = -2 pi / d2Psi/df2.
    phase_second_derivative = np.array(
        [
            lalsim.PNPhaseSecondDerivative(float(freq), 2, phasing, mtot_sec)
            for freq in np.atleast_1d(f)
        ]
    )
    return -2.0 * np.pi / phase_second_derivative


def cumulative_power_fraction(f, power):
    cumulative_power = cumulative_trapezoid(power, f, initial=0.0)
    total_power = cumulative_power[-1]
    if total_power <= 0.0:
        raise ValueError("Total recovered power must be positive.")
    return cumulative_power / total_power


# --- 2. Noise PSD ---
asd_path = Path(__file__).resolve().parents[2] / "asd.txt"
asd_data = np.loadtxt(asd_path)
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
    Mc, M, m1, m2 = component_masses_from_mchirp_eta(Mc_msun, eta)
    M_sec = G * M / c**3

    dfdt_0pn = np.asarray(domega_dt_0pn(np.pi * f_power, M_sec)) / np.pi
    dfdt_taylorf2 = np.asarray(taylorf2_dfdt(f_power, m1, m2))
    if np.any(dfdt_0pn <= 0.0) or np.any(dfdt_taylorf2 <= 0.0):
        raise ValueError(
            f"df/dt must remain positive for Mc = {Mc_msun:.0e} Msun."
        )

    # Common distance factors cancel in the normalized curves, so d is omitted.
    h0_shape = 4.0 * (G * Mc / c**2) ** (5.0 / 3.0) * (np.pi * f_power / c) ** (2.0 / 3.0)
    p0 = h0_shape**2 / (dfdt_0pn * Sn_power)
    p_taylorf2 = h0_shape**2 / (dfdt_taylorf2 * Sn_power)
    power_curves.append((Mc_msun, p0, p_taylorf2, dfdt_taylorf2.min()))

# --- 4. Plotting ---
fig, ax3 = plt.subplots(figsize=(9, 5.5))
fig_cdf, ax_cdf = plt.subplots(figsize=(9, 5.5))
colors = ["crimson", "darkorange", "dodgerblue"]
for color, (Mc_msun, p0, p_taylorf2, _) in zip(colors, power_curves):
    ax3.loglog(f_power, p0 / np.sum(p0), color=color, linestyle="--", label=rf"0PN, $M_c={Mc_msun:.0e}M_\odot$")
    ax3.loglog(
        f_power,
        p_taylorf2 / np.sum(p_taylorf2),
        color=color,
        linestyle="-",
        label=rf"TaylorF2 {TAYLORF2_PN_LABEL}, $M_c={Mc_msun:.0e}M_\odot$",
    )
    ax_cdf.plot(
        f_power,
        cumulative_power_fraction(f_power, p0),
        color=color,
        linestyle="--",
        label=rf"0PN, $M_c={Mc_msun:.0e}M_\odot$",
    )
    ax_cdf.plot(
        f_power,
        cumulative_power_fraction(f_power, p_taylorf2),
        color=color,
        linestyle="-",
        label=rf"TaylorF2 {TAYLORF2_PN_LABEL}, $M_c={Mc_msun:.0e}M_\odot$",
    )

ax3.set_xlabel("GW Frequency [Hz]", fontsize=11)
ax3.set_ylabel("Sum-normalized noise-weighted power", fontsize=11)
ax3.set_title(f"Noise-weighted stationary-phase power: 0PN vs TaylorF2 {TAYLORF2_PN_LABEL}", fontsize=13, fontweight="bold")
ax3.legend(loc="best")
ax3.grid(True, which="both", alpha=0.3)

ax_cdf.set_xscale("log")
ax_cdf.set_xlabel(r"$f_{\mathrm{end}}$ [Hz]", fontsize=11)
ax_cdf.set_ylabel(r"Recovered power fraction from 20 Hz to $f_{\mathrm{end}}$", fontsize=11)
ax_cdf.set_title(
    f"Cumulative recovered power: 0PN vs TaylorF2 {TAYLORF2_PN_LABEL}",
    fontsize=13,
    fontweight="bold",
)
ax_cdf.set_ylim(-0.02, 1.02)
ax_cdf.legend(loc="best")
ax_cdf.grid(True, which="both", alpha=0.3)
fig_cdf.tight_layout()

plt.tight_layout()
plt.show()

print(f"Noise-weighted power evaluated over {f_power[0]:.2f}-{f_power[-1]:.2f} Hz")
for Mc_msun, _, _, min_dfdt_taylorf2 in power_curves:
    print(
        f"Mc = {Mc_msun:.0e} Msun: "
        f"min df/dt TaylorF2 {TAYLORF2_PN_LABEL} = {min_dfdt_taylorf2:.6e} Hz/s"
    )
