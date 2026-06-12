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

# %%
C = 3e8
G = 6.67e-11
PI = np.pi
SOLAR_MASS_KG = 2e30
PARSEC_M = 3e16
LAMBDA_THRESHOLD = 34
GALACTIC_CENTER_PC = 8000
ANDROMEDA_PC = 7.65e5
CHIRP_CONST = 96 / 5 * PI**(8 / 3) * (G / C**3)**(5 / 3)

F_START = 40
F_END = 60
MAX_OBS_TIME = 3e7
CHUNK_DURATION = 30.0
LOG_M_LOWER = -5
LOG_M_UPPER = -1

# %%
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
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


# %%
def beta_calc(f0, Mc):
    M = np.multiply(Mc, SOLAR_MASS_KG)
    return CHIRP_CONST * f0**(8 / 3) * M**(5 / 3)


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


def apply_semicoherent_chunk_penalty(sensitivity_grid, chunk_count_grid):
    """Reduce sensitivity by the semicoherent combination factor M^(-1/4)."""
    return sensitivity_grid / np.asarray(chunk_count_grid) ** (1 / 4)


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
fspace = np.linspace(F_START, F_END, 51, endpoint=False)
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


def distance_sensitivity(T, f0, Mc, l=47):
    #l := lambda. l=47 is the threshold for FAP=1e-6, detection probability = 0.95
    beta = beta_calc(f0, Mc)
    integration = integrated_chirp_power(T, f0, beta)
    temp0 = 0.00757 / np.sqrt(l)
    temp1 = 3 * C * beta / f0**2
    return temp0 * temp1 * np.sqrt(integration)

# %%
chunk_count_grid = semicoherent_chunk_count_grid(tMax_grid)
sens_grid = distance_sensitivity(tMax_grid, fgrid, Mgrid, l=LAMBDA_THRESHOLD) / PARSEC_M
sens_grid = apply_semicoherent_chunk_penalty(sens_grid, chunk_count_grid)
galactic_center_reachable = np.any(sens_grid >= GALACTIC_CENTER_PC, axis=1)
if np.any(galactic_center_reachable):
    reachable_masses = Mspace[galactic_center_reachable]
    print(
        "Mc range reaching Galactic center for f0 between:"
        f"{reachable_masses[0]:.1e} - {reachable_masses[-1]:.1e} Msun\n \n"
    )
else:
    print("No Mc values reach the Galactic center for any f0.")

# %%
fig, ax = pl.subplots()
log_Mspace = np.log10(Mspace)
log_sens_grid = np.log10(sens_grid)
contour = ax.contourf(fspace, log_Mspace, log_sens_grid, levels=range(1, 10))
fig.colorbar(contour, ax=ax, label=r'log(Distance Sensitivity / pc)')

ax.contour(
    fspace,
    log_Mspace,
    log_sens_grid,
    [np.log10(GALACTIC_CENTER_PC), np.log10(ANDROMEDA_PC)],
    colors=['red', 'darkorange'],
    linestyles='--',
)
ax.set_title(f'Semicoherent sensitivity, f={F_START}-{F_END} Hz')
ax.set_xlabel("Initial frequency (Hz)")
ax.set_ylabel(r'$log(M_c/M_\odot$)')

line0 = Line2D([0], [0], label='Galactic center', color='r', ls='--')
line1 = Line2D([0], [0], label='Andromeda', color='darkorange', ls='--')
ax.legend(handles=[line0, line1])
fig.savefig(FIG_DIR / f"semicoherent_sensitivity_f={F_START}-{F_END}.png", bbox_inches="tight")


# # %%
# def h0_calc(d, f, M):
#     m_corr = M*2e30
#     temp0 = 4/d
#     temp1 = (G*m_corr/(c**2))**(5/3)
#     temp2 = (pi*f/c)**(2/3)
#     return temp0 * temp1 * temp2


# # %%
# def d_calc(h0, f, M):
#     m_corr = M*2e30
#     temp0 = 4/h0
#     temp1 = (G*m_corr/(c**2))**(5/3)
#     temp2 = (pi*f/c)**(2/3)
#     return temp0 * temp1 * temp2


# # %%
# temp_dist = sens_grid[0,0] * 3e16
# h0_calc(temp_dist, fspace[0], Mspace[0])

# %%
