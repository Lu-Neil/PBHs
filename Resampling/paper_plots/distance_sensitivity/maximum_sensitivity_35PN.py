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
from scipy import integrate, interpolate

import lal
import lalsimulation as lalsim

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
ETA = 0.25
TAYLORF2_PN_PHASE_ORDER = lalsim.PNORDER_THREE_POINT_FIVE
TAYLORF2_PN_LABEL = "3.5PN"

# %%
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
ASD_PATH = SCRIPT_DIR.parents[1] / "asd.txt"
FIG_DIR = SCRIPT_DIR / "figs"
FIG_DIR.mkdir(exist_ok=True)

asd = np.loadtxt(ASD_PATH)
asd_freq = asd[:, 0]
asd_values = asd[:, 1]
interp_asd = interpolate.interp1d(asd_freq, asd_values)


# %%
def beta_calc(f0, Mc):
    M = np.multiply(Mc, SOLAR_MASS_KG)
    return CHIRP_CONST * f0**(8 / 3) * M**(5 / 3)


def component_masses_from_mchirp_eta(Mc_msun, eta=ETA):
    Mc = Mc_msun * SOLAR_MASS_KG
    M = Mc / eta**(3 / 5)
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * M * (1.0 + sqrt_term)
    m2 = 0.5 * M * (1.0 - sqrt_term)
    return m1, m2


class TaylorF2FrequencyEvolution:
    """3.5PN TaylorF2 time-frequency map and sensitivity-integral cache."""

    def __init__(self, Mc_msun, frequency_grid, eta=ETA):
        self.Mc_msun = Mc_msun
        self.frequency_grid = np.asarray(frequency_grid)
        m1, m2 = component_masses_from_mchirp_eta(Mc_msun, eta=eta)
        mtot_sec = G * (m1 + m2) / C**3

        params = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
            params, TAYLORF2_PN_PHASE_ORDER
        )
        phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(
            m1, m2, 0.0, 0.0, params
        )

        self.t_of_f_grid = np.array(
            [
                lalsim.PNPhaseDerivative(f, 2, phasing, mtot_sec) / (2.0 * PI)
                for f in self.frequency_grid
            ]
        )

        valid = np.isfinite(self.frequency_grid) & np.isfinite(self.t_of_f_grid)
        self.frequency_grid = self.frequency_grid[valid]
        self.t_of_f_grid = self.t_of_f_grid[valid]

        if np.any(np.diff(self.t_of_f_grid) >= 0.0):
            raise ValueError(
                f"TaylorF2 chirp time must decrease monotonically for "
                f"Mc = {Mc_msun:.3e} Msun."
            )

        self.t_of_f = interpolate.interp1d(
            self.frequency_grid,
            self.t_of_f_grid,
            kind="linear",
            bounds_error=True,
        )
        self.f_of_t = interpolate.interp1d(
            self.t_of_f_grid[::-1],
            self.frequency_grid[::-1],
            kind="linear",
            bounds_error=True,
        )

        self.elapsed_from_low_frequency_grid = self.t_of_f_grid[0] - self.t_of_f_grid
        self.power_integrand_grid = self.frequency_grid**(4 / 3) / interp_asd(
            self.frequency_grid
        )**2
        self.power_antiderivative_grid = integrate.cumulative_trapezoid(
            self.power_integrand_grid,
            self.elapsed_from_low_frequency_grid,
            initial=0.0,
        )

    def elapsed_time_to(self, f_start, f_end):
        return float(self.t_of_f(f_start) - self.t_of_f(f_end))

    def frequency_after(self, f_start, elapsed_time):
        target_t_of_f = float(self.t_of_f(f_start)) - elapsed_time
        return float(self.f_of_t(target_t_of_f))

    def power_antiderivative(self, frequency):
        frequency = np.asarray(frequency)
        if np.any(
            (frequency < self.frequency_grid[0])
            | (frequency > self.frequency_grid[-1])
        ):
            raise ValueError("frequency is outside the TaylorF2 interpolation range")

        idx = np.searchsorted(self.frequency_grid, frequency, side="right") - 1
        idx = np.clip(idx, 0, len(self.frequency_grid) - 2)
        elapsed = self.t_of_f_grid[0] - self.t_of_f(frequency)
        power_integrand = frequency**(4 / 3) / interp_asd(frequency)**2
        d_elapsed = elapsed - self.elapsed_from_low_frequency_grid[idx]
        return (
            self.power_antiderivative_grid[idx]
            + 0.5
            * (self.power_integrand_grid[idx] + power_integrand)
            * d_elapsed
        )

    def integrated_chirp_power(self, f_start, f_end):
        # Eq. 49's 0PN factor (1 - 8 beta t / 3)^(-1/2) becomes
        # (f_TaylorF2(t) / f_start)^(4/3).
        return f_start**(-4 / 3) * float(
            self.power_antiderivative(f_end) - self.power_antiderivative(f_start)
        )


def taylorf2_frequency_grid(f_min, f_max):
    grid = asd_freq[(asd_freq >= f_min) & (asd_freq <= f_max)]
    return np.unique(np.concatenate(([f_min, f_max], grid)))


# %%
fspace = np.linspace(20, 200, 51)
Mspace = np.logspace(-5, 1, 49)
fgrid, Mgrid = np.meshgrid(fspace, Mspace)

betaGrid = beta_calc(fgrid, Mgrid)
tMax_grid = np.empty_like(fgrid)
final_f = np.empty_like(fgrid)
chirp_power_grid = np.empty_like(fgrid)

for i, Mc in enumerate(Mspace):
    evolution = TaylorF2FrequencyEvolution(
        Mc,
        taylorf2_frequency_grid(fspace.min(), F_MAX),
    )
    for j, f0 in enumerate(fspace):
        time_to_fmax = evolution.elapsed_time_to(f0, F_MAX)
        t_obs = min(time_to_fmax, MAX_OBS_TIME)
        if time_to_fmax <= MAX_OBS_TIME:
            f_end = F_MAX
        else:
            f_end = evolution.frequency_after(f0, t_obs)

        tMax_grid[i, j] = t_obs
        final_f[i, j] = f_end
        chirp_power_grid[i, j] = evolution.integrated_chirp_power(f0, f_end)


# %% [markdown]
# ## Sensitivity

# %%
def distance_sensitivity(f0, Mc, integration, l=47):
    # l := lambda. l=47 is the threshold for FAP=1e-6, detection probability = 0.95
    # The 3.5PN TaylorF2 evolution changes the integration endpoint and the
    # f(t)-dependent amplitude/noise term; the original Newtonian amplitude
    # prefactor is retained for direct comparison with maximum_sensitivity.py.
    beta = beta_calc(f0, Mc)
    temp0 = 0.00757 / np.sqrt(l)
    temp1 = 3 * C * beta / f0**2
    return temp0 * temp1 * np.sqrt(integration)


# %%
sens_grid = (
    distance_sensitivity(fgrid, Mgrid, chirp_power_grid, l=LAMBDA_THRESHOLD)
    / PARSEC_M
)

# %%
fig, ax = pl.subplots()
log_Mspace = np.log10(Mspace)
log_sens_grid = np.log10(sens_grid)
contour = ax.contourf(fspace, log_Mspace, log_sens_grid)
fig.colorbar(contour, ax=ax, label="log(Distance Sensitivity / pc)")
ax.contour(
    fspace,
    log_Mspace,
    log_sens_grid,
    [np.log10(GALACTIC_CENTER_PC), np.log10(ANDROMEDA_PC)],
    colors=["red", "darkorange"],
    linestyles="--",
)
ax.set_title(rf"Maximum distance sensitivity ({TAYLORF2_PN_LABEL} TaylorF2)")
ax.set_xlabel("Initial frequency (Hz)")
ax.set_ylabel(r"$log(M_c/M_\odot$)")

line0 = Line2D([0], [0], label="Galactic center", color="r", ls="--")
line1 = Line2D([0], [0], label="Andromeda", color="darkorange", ls="--")
ax.legend(handles=[line0, line1])
fig.savefig(FIG_DIR / "maximum_distance_sensitivity_35PN.png", bbox_inches="tight")
