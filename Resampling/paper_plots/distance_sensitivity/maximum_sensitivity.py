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

import matplotlib as mpl
mpl.use("Agg")
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate, integrate

import lal

# %%
C = lal.C_SI
G = lal.G_SI
PI = np.pi
SOLAR_MASS_KG = lal.MSUN_SI
PARSEC_M = lal.PC_SI
MAX_OBS_TIME = 3e7
LAMBDA_THRESHOLD = 34
GALACTIC_CENTER_PC = 8000
ANDROMEDA_PC = 7.65e5
CHIRP_CONST = 96 / 5 * PI**(8 / 3) * (G / C**3)**(5 / 3)
F_MAX = 2000

# %%
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ASD_PATH = SCRIPT_DIR.parents[1] / "asd.txt"
FIG_DIR = SCRIPT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)

asd = np.loadtxt(ASD_PATH)
asd_freq = asd[:, 0]
asd_values = asd[:, 1]
interp_asd = interpolate.interp1d(asd_freq, asd_values) # , bounds_error = False, fill_value="extrapolate"
base_integrand = 1 / (asd_freq**(7 / 3) * asd_values**2)
base_integral = integrate.cumulative_trapezoid(base_integrand, asd_freq, initial=0)


# %%
def beta_calc(f0, Mc):
    M = np.multiply(Mc, SOLAR_MASS_KG)
    return CHIRP_CONST * f0**(8 / 3) * M**(5 / 3)


def f_calc(t, f0, Mc, beta=None):
    if beta is None:
        beta = beta_calc(f0, Mc)
    return f0 * (1 - 8 / 3 * beta * t)**(-3 / 8)


# %%
fspace = np.linspace(20, 200, 51)
Mspace = np.logspace(-5, 0, 49)
fgrid, Mgrid = np.meshgrid(fspace, Mspace)

betaGrid = beta_calc(fgrid, Mgrid)
tMax_grid = np.minimum(0.375 / betaGrid * (1 - (fgrid/F_MAX)**(8/3)), MAX_OBS_TIME)
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
    slopes = np.diff(base_integrand)[idx] / np.diff(asd_freq)[idx]
    return base_integral[idx] + base_integrand[idx] * dx + 0.5 * slopes * dx**2


def integrated_chirp_power(T, f0, beta):
    chirp_factor = 1 - 8 / 3 * beta * T
    if np.any(chirp_factor <= 0):
        raise ValueError("integration endpoint is past the chirp singularity")

    final_frequency = f0 * chirp_factor**(-3 / 8)
    return (
        f0**(4 / 3)
        / beta
        * (base_integral_antiderivative(final_frequency) - base_integral_antiderivative(f0))
    )


def distance_sensitivity(T, f0, Mc, l=47):
    #l := lambda. l=47 is the threshold for FAP=1e-6, detection probability = 0.95
    beta = beta_calc(f0, Mc)
    integration = integrated_chirp_power(T, f0, beta)
    temp0 = 0.00757 / np.sqrt(l)
    temp1 = 3 * C * beta / f0**2
    return temp0 * temp1 * np.sqrt(integration)

# %%
sens_grid = distance_sensitivity(tMax_grid, fgrid, Mgrid, l=LAMBDA_THRESHOLD) / PARSEC_M

# %%
fig, ax = pl.subplots()
log_Mspace = np.log10(Mspace)
log_sens_grid = np.log10(sens_grid)
contour = ax.contourf(fspace, log_Mspace, log_sens_grid, levels=range(1, 10))
fig.colorbar(contour, ax=ax, label='log(Distance Sensitivity / pc)')
ax.contour(
    fspace,
    log_Mspace,
    log_sens_grid,
    [np.log10(GALACTIC_CENTER_PC), np.log10(ANDROMEDA_PC)],
    colors=['red', 'darkorange'],
    linestyles='--',
)
ax.set_title(r'Maximum distance sensitivity')
ax.set_xlabel("Initial frequency (Hz)")
ax.set_ylabel(r'$log(M_c/M_\odot$)')

line0 = Line2D([0], [0], label='Galactic center', color='r', ls='--')
line1 = Line2D([0], [0], label='Andromeda', color='darkorange', ls='--')
ax.legend(handles=[line0, line1])
fig.savefig(FIG_DIR / "maximum_distance_sensitivity.png", bbox_inches="tight")
