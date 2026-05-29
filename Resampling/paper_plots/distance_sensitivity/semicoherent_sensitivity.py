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
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as pl
from scipy import interpolate, integrate, optimize

# %%
C = 3e8
G = 6.67e-11
PI = np.pi
SOLAR_MASS_KG = 2e30
PARSEC_M = 3e16
MAX_OBS_TIME = 3e7
LAMBDA_THRESHOLD = 34
GALACTIC_CENTER_PC = 8000
CHIRP_CONST = 96 / 5 * PI**(8 / 3) * (G / C**3)**(5 / 3)

F_START = 20
F_END = 256
LOG_M_LOWER = -2
LOG_M_UPPER = -1
CHUNK_COUNT_N_GRID = 10_000

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


def _total_mass_seconds_from_mchirp_eta(Mc, eta):
    chirp_mass_kg = Mc * SOLAR_MASS_KG
    total_mass_kg = chirp_mass_kg / eta ** (3 / 5)
    return G * total_mass_kg / C**3


def _taylor_t4_factor_35pn(v, eta):
    """Nonspinning point-particle TaylorT4 3.5PN velocity factor."""
    a2 = -(743 + 924 * eta) / 336
    a3 = 4 * PI
    a4 = (34103 + 122949 * eta + 59472 * eta**2) / 18144
    a5 = -PI * (4159 + 15876 * eta) / 672
    a6 = (
        16447322263 / 139708800
        - 1712 * np.euler_gamma / 105
        - 856 * np.log(16) / 105
        - 56198689 * eta / 217728
        + PI**2 * (16 / 3 + 451 * eta / 48)
        + 541 * eta**2 / 896
        - 5605 * eta**3 / 2592
    )
    a6_log = -1712 / 105
    a7 = PI * (-13245 + 717350 * eta + 731960 * eta**2) / 12096

    return (
        1
        + a2 * v**2
        + a3 * v**3
        + a4 * v**4
        + a5 * v**5
        + (a6 + a6_log * np.log(v)) * v**6
        + a7 * v**7
    )


def _v_0pn_at_time(t, v_start, total_mass_seconds, eta):
    denominator = v_start ** -8 - (256 * eta / (5 * total_mass_seconds)) * t
    if denominator <= 0:
        return np.inf
    return denominator ** (-1 / 8)


class _TaylorT4PhaseMap:
    def __init__(self, Mc, f_min, f_max, eta=0.25, n_grid=10_000):
        self.eta = eta
        self.total_mass_seconds = _total_mass_seconds_from_mchirp_eta(Mc, eta)
        self.v_grid = np.geomspace(
            self.velocity_from_frequency(f_min),
            self.velocity_from_frequency(f_max),
            n_grid,
        )

        factor = _taylor_t4_factor_35pn(self.v_grid, eta)
        if np.any(factor <= 0):
            raise ValueError("TaylorT4 factor became non-positive over the band")

        dt_dv = (
            5
            * self.total_mass_seconds
            / (32 * eta)
            * self.v_grid ** -9
            / factor
        )
        dphi_dv = 2 / self.total_mass_seconds * self.v_grid**3 * dt_dv
        elapsed_time = integrate.cumulative_trapezoid(dt_dv, self.v_grid, initial=0)
        phase = integrate.cumulative_trapezoid(dphi_dv, self.v_grid, initial=0)

        self.elapsed_time_of_v = interpolate.interp1d(
            self.v_grid, elapsed_time, kind="cubic", bounds_error=True
        )
        self.phase_of_v = interpolate.interp1d(
            self.v_grid, phase, kind="cubic", bounds_error=True
        )

    def velocity_from_frequency(self, frequency):
        return (PI * self.total_mass_seconds * frequency) ** (1 / 3)

    def frequency_from_velocity(self, velocity):
        return velocity**3 / (PI * self.total_mass_seconds)

    def elapsed_time_between(self, v_start, v_end):
        return float(self.elapsed_time_of_v(v_end) - self.elapsed_time_of_v(v_start))

    def taylor_t4_phase_between(self, v_start, v_end):
        return float(self.phase_of_v(v_end) - self.phase_of_v(v_start))

    def dephasing(self, v_start, v_end):
        elapsed_time = self.elapsed_time_between(v_start, v_end)
        v_0pn = _v_0pn_at_time(
            elapsed_time, v_start, self.total_mass_seconds, self.eta
        )
        phi_0pn = (v_start ** -5 - v_0pn ** -5) / (16 * self.eta)
        return self.taylor_t4_phase_between(v_start, v_end) - phi_0pn

    def chunk_endpoint(self, f_start, f_end, dephasing_threshold=PI):
        v_start = self.velocity_from_frequency(f_start)
        v_stop = self.velocity_from_frequency(f_end)
        elapsed_start = float(self.elapsed_time_of_v(v_start))
        phase_start = float(self.phase_of_v(v_start))

        def dephasing_to(v_end):
            elapsed_time = float(self.elapsed_time_of_v(v_end)) - elapsed_start
            v_0pn = _v_0pn_at_time(
                elapsed_time, v_start, self.total_mass_seconds, self.eta
            )
            phi_0pn = (v_start ** -5 - v_0pn ** -5) / (16 * self.eta)
            return float(self.phase_of_v(v_end)) - phase_start - phi_0pn

        def threshold_function(v):
            return abs(dephasing_to(v)) - dephasing_threshold

        if threshold_function(v_stop) < 0:
            return {
                "f_end": f_end,
                "duration": float(self.elapsed_time_of_v(v_stop)) - elapsed_start,
                "dephasing": dephasing_to(v_stop),
                "threshold_reached": False,
            }

        v_end = optimize.brentq(threshold_function, v_start, v_stop, rtol=1e-12)
        return {
            "f_end": self.frequency_from_velocity(v_end),
            "duration": float(self.elapsed_time_of_v(v_end)) - elapsed_start,
            "dephasing": dephasing_to(v_end),
            "threshold_reached": True,
        }


def _semicoherent_taylor_t4_chunks(
    f_start,
    f_end,
    taylor_t4_map,
    dephasing_threshold=PI,
):
    chunks = []
    current_f = f_start
    while current_f < f_end:
        endpoint = taylor_t4_map.chunk_endpoint(current_f, f_end, dephasing_threshold)
        next_f = min(endpoint["f_end"], f_end)
        if next_f <= current_f:
            raise RuntimeError("TaylorT4 chunk endpoint did not advance in frequency")

        chunks.append(
            {
                "f_start": current_f,
                "f_end": next_f,
                "duration": endpoint["duration"],
                "dephasing": endpoint["dephasing"],
                "threshold_reached": endpoint["threshold_reached"] and next_f < f_end,
            }
        )
        current_f = next_f

    return chunks


def _dephasing_to_fixed_endpoint(taylor_t4_map, v_start, v_end, elapsed_end, phase_end):
    elapsed_time = elapsed_end - float(taylor_t4_map.elapsed_time_of_v(v_start))
    v_0pn = _v_0pn_at_time(
        elapsed_time,
        v_start,
        taylor_t4_map.total_mass_seconds,
        taylor_t4_map.eta,
    )
    phi_0pn = (v_start ** -5 - v_0pn ** -5) / (16 * taylor_t4_map.eta)
    return phase_end - float(taylor_t4_map.phase_of_v(v_start)) - phi_0pn


def _semicoherent_taylor_t4_chunk_boundaries(
    f_min,
    f_end,
    taylor_t4_map,
    dephasing_threshold=PI,
):
    """Return descending start-frequency boundaries for 1, 2, ... chunks.

    The forward chunk endpoint is monotonic in start frequency. For a fixed
    final frequency, the boundary between N and N+1 chunks is therefore the
    start frequency whose one-chunk endpoint lands on the previous boundary.
    Computing these boundaries once per chirp mass avoids re-rooting the same
    chunk walk for every f_start grid point.
    """
    boundaries = []
    v_min = taylor_t4_map.velocity_from_frequency(f_min)
    current_f_end = f_end

    while current_f_end > f_min:
        v_end = taylor_t4_map.velocity_from_frequency(current_f_end)
        elapsed_end = float(taylor_t4_map.elapsed_time_of_v(v_end))
        phase_end = float(taylor_t4_map.phase_of_v(v_end))

        def threshold_function(v_start):
            dephasing = _dephasing_to_fixed_endpoint(
                taylor_t4_map, v_start, v_end, elapsed_end, phase_end
            )
            return abs(dephasing) - dephasing_threshold

        if threshold_function(v_min) <= 0:
            break

        v_boundary = optimize.brentq(
            threshold_function, v_min, v_end, rtol=1e-12
        )
        boundary = taylor_t4_map.frequency_from_velocity(v_boundary)
        if boundary >= current_f_end:
            raise RuntimeError("TaylorT4 chunk boundary did not move backward")

        boundaries.append(boundary)
        current_f_end = boundary

    return np.asarray(boundaries)


def semicoherent_chunk_count(
    Mc,
    f_start,
    f_end,
    eta=0.25,
    dephasing_threshold=PI,
    return_chunks=False,
    n_grid=10_000,
):
    """Count chunks whose 0PN-vs-3.5PN TaylorT4 dephasing stays below threshold."""
    if not 0.0 < eta <= 0.25:
        raise ValueError("eta must be in the interval (0, 0.25]")
    if f_start <= 0.0 or f_end <= 0.0:
        raise ValueError("frequencies must be positive")
    if f_end <= f_start:
        raise ValueError("f_end must be greater than f_start")
    if dephasing_threshold <= 0:
        raise ValueError("dephasing_threshold must be positive")

    taylor_t4_map = _TaylorT4PhaseMap(Mc, f_start, f_end, eta=eta, n_grid=n_grid)
    chunks = _semicoherent_taylor_t4_chunks(
        f_start,
        f_end,
        taylor_t4_map,
        dephasing_threshold=dephasing_threshold,
    )

    if return_chunks:
        return len(chunks), chunks
    return len(chunks)


def semicoherent_chunk_count_grid(
    f_start_grid,
    Mc_grid,
    f_end,
    eta=0.25,
    dephasing_threshold=PI,
    n_grid=10_000,
):
    """Compute 3.5PN TaylorT4 dephasing chunk counts over an f_start/Mc grid."""
    f_start_grid = np.asarray(f_start_grid)
    Mc_grid = np.asarray(Mc_grid)
    if f_start_grid.shape != Mc_grid.shape:
        raise ValueError("f_start_grid and Mc_grid must have the same shape")
    if f_end <= 0:
        raise ValueError("f_end must be positive")
    if dephasing_threshold <= 0:
        raise ValueError("dephasing_threshold must be positive")

    chunk_counts = np.ones(f_start_grid.shape, dtype=int)
    for Mc in np.unique(Mc_grid):
        mass_mask = Mc_grid == Mc
        active_mask = mass_mask & (f_start_grid < f_end)
        if not np.any(active_mask):
            continue

        f_min = float(np.min(f_start_grid[active_mask]))
        taylor_t4_map = _TaylorT4PhaseMap(
            float(Mc), f_min, f_end, eta=eta, n_grid=n_grid
        )
        boundaries = _semicoherent_taylor_t4_chunk_boundaries(
            f_min,
            f_end,
            taylor_t4_map,
            dephasing_threshold=dephasing_threshold,
        )
        if boundaries.size == 0:
            chunk_counts[active_mask] = 1
            continue

        ascending_boundaries = boundaries[::-1]
        start_values = f_start_grid[active_mask]
        chunk_counts[active_mask] = (
            1
            + ascending_boundaries.size
            - np.searchsorted(ascending_boundaries, start_values, side="right")
        )

    return chunk_counts


def apply_semicoherent_chunk_penalty(sensitivity_grid, chunk_count_grid):
    """Reduce sensitivity by the semicoherent combination factor M^(-1/4)."""
    return sensitivity_grid / np.asarray(chunk_count_grid) ** (1 / 4)


def f_calc(t, f0, Mc, beta=None):
    if beta is None:
        beta = beta_calc(f0, Mc)
    return f0 * (1 - 8 / 3 * beta * t)**(-3 / 8)


# %%
fspace = np.linspace(F_START, F_END, 51)
Mspace = np.logspace(LOG_M_LOWER, LOG_M_UPPER, 49)
fgrid, Mgrid = np.meshgrid(fspace, Mspace)

betaGrid = beta_calc(fgrid, Mgrid)
tMax_grid = np.minimum(0.37 / betaGrid, MAX_OBS_TIME)

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
chunk_count_grid = semicoherent_chunk_count_grid(
    fgrid, Mgrid, F_END, n_grid=CHUNK_COUNT_N_GRID
)
sens_grid = distance_sensitivity(tMax_grid, fgrid, Mgrid, l=LAMBDA_THRESHOLD) / PARSEC_M
sens_grid = apply_semicoherent_chunk_penalty(sens_grid, chunk_count_grid)

# %%
fig, ax = pl.subplots()
log_Mspace = np.log10(Mspace)
log_sens_grid = np.log10(sens_grid)
contour = ax.contourf(fspace, log_Mspace, log_sens_grid)
fig.colorbar(contour, ax=ax)

ax.contour(
    fspace,
    log_Mspace,
    log_sens_grid,
    [np.log10(GALACTIC_CENTER_PC)],
    colors='red',
    linestyles='--',
)
ax.set_title(r'log(Distance Sensitivity / pc)')
ax.set_xlabel("Initial frequency (Hz)")
ax.set_ylabel(r'$log(M_c/M_\odot$)')

line = Line2D([0], [0], label='Galactic center', color='r', ls='--')
ax.legend(handles=[line])
pl.show()
fig.savefig(FIG_DIR / "distance_sensitivity.png", bbox_inches="tight")


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
