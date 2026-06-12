"""Build a TaylorF2 semicoherent bank for 30 s chunks.

The stack-slide track coordinates are chirp mass and the reference crossing
time t20, where the TaylorF2 track crosses 20 Hz.  The carrier frequency is
handled by the per-chunk frequency search, with chunk frequencies in the band
20--64 Hz.  Template spacing is chosen so that a signal half-way between
adjacent grid points has a bounded instantaneous-frequency error inside any
in-band part of a 30 s coherent chunk.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

import lal
import lalsimulation as lalsim


DEFAULT_MCHIRP_MIN = 5.0e-4
DEFAULT_MCHIRP_MAX = 1.0e-1
DEFAULT_F_MIN = 20.0
DEFAULT_F_MAX = 64.0
DEFAULT_T_COH = 30.0
DEFAULT_OBSERVATION_DAYS = 365.25
DEFAULT_ETA = 0.25
DEFAULT_FREQ_ERROR_HZ = 1.0 / DEFAULT_T_COH
DEFAULT_N_MCHIRP_SCAN = 48
DEFAULT_N_F0_SCAN = 96
DEFAULT_N_TIME_SCAN = 64
DEFAULT_N_METRIC_TIME_SCAN = 256
DEFAULT_N_TAYLORF2_GRID = 2048
DEFAULT_FINITE_DIFF_DLOGM = 1.0e-4
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "figs"


@dataclass(frozen=True)
class BankConfig:
    mchirp_min: float
    mchirp_max: float
    f_min: float
    f_max: float
    t_coh: float
    observation_days: float
    eta: float
    freq_error_hz: float
    n_mchirp_scan: int
    n_f0_scan: int
    n_time_scan: int
    n_metric_time_scan: int
    n_taylorf2_grid: int
    finite_diff_dlogm: float
    output_dir: Path

    @property
    def observation_seconds(self) -> float:
        return self.observation_days * 86400.0


def component_masses_from_mchirp_eta(
    mchirp_msun: float, eta: float
) -> tuple[float, float]:
    """Return component masses in SI units for a chirp mass and symmetric eta."""
    if not 0.0 < eta <= 0.25:
        raise ValueError("eta must be in the range (0, 0.25]")

    mchirp = mchirp_msun * lal.MSUN_SI
    total_mass = mchirp / eta ** (3.0 / 5.0)
    sqrt_term = np.sqrt(1.0 - 4.0 * eta)
    m1 = 0.5 * total_mass * (1.0 + sqrt_term)
    m2 = 0.5 * total_mass * (1.0 - sqrt_term)
    return float(m1), float(m2)


def taylorf2_phasing(
    mchirp_msun: float, eta: float
) -> tuple[object, float]:
    m1, m2 = component_masses_from_mchirp_eta(mchirp_msun, eta)
    mtot_sec = lal.G_SI * (m1 + m2) / lal.C_SI**3

    params = lal.CreateDict()
    lalsim.SimInspiralWaveformParamsInsertPNPhaseOrder(
        params, lalsim.PNORDER_THREE_POINT_FIVE
    )
    phasing = lalsim.SimInspiralTaylorF2AlignedPhasing(m1, m2, 0.0, 0.0, params)
    return phasing, mtot_sec


def taylorf2_time_of_frequency(
    frequency_hz: np.ndarray,
    mchirp_msun: float,
    eta: float,
) -> np.ndarray:
    """Return TaylorF2 stationary-phase time t(f) = dPsi/df / (2 pi)."""
    t_of_f, _ = taylorf2_time_and_dfdt_of_frequency(
        frequency_hz, mchirp_msun, eta
    )
    return t_of_f


def taylorf2_dfdt_of_frequency(
    frequency_hz: np.ndarray,
    mchirp_msun: float,
    eta: float,
) -> np.ndarray:
    """Return df/dt_elapsed from the TaylorF2 stationary-phase map."""
    _, dfdt = taylorf2_time_and_dfdt_of_frequency(
        frequency_hz, mchirp_msun, eta
    )
    return dfdt


def taylorf2_time_and_dfdt_of_frequency(
    frequency_hz: np.ndarray,
    mchirp_msun: float,
    eta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return TaylorF2 t(f) and df/dt using one LAL phasing construction."""
    frequency_hz = np.asarray(frequency_hz, dtype=float)
    phasing, mtot_sec = taylorf2_phasing(mchirp_msun, eta)

    flat_frequency = frequency_hz.ravel()
    t_of_f = np.empty_like(flat_frequency)
    dfdt = np.empty_like(flat_frequency)
    for i, freq in enumerate(flat_frequency):
        phase_derivative = lalsim.PNPhaseDerivative(
            float(freq), 2, phasing, mtot_sec
        )
        phase_second_derivative = lalsim.PNPhaseSecondDerivative(
            float(freq), 2, phasing, mtot_sec
        )
        t_of_f[i] = phase_derivative / (2.0 * np.pi)
        dfdt[i] = -2.0 * np.pi / phase_second_derivative

    return t_of_f.reshape(frequency_hz.shape), dfdt.reshape(frequency_hz.shape)


class TaylorF2FrequencyTrack:
    """Interpolate f(t) for chunks starting at arbitrary in-band frequencies."""

    def __init__(self, mchirp_msun: float, config: BankConfig):
        self.mchirp_msun = float(mchirp_msun)
        self.frequency_grid = np.geomspace(
            config.f_min, config.f_max, config.n_taylorf2_grid
        )
        self.t_of_f_grid, self.dfdt_grid = taylorf2_time_and_dfdt_of_frequency(
            self.frequency_grid, self.mchirp_msun, config.eta
        )

        valid = (
            np.isfinite(self.frequency_grid)
            & np.isfinite(self.t_of_f_grid)
            & np.isfinite(self.dfdt_grid)
            & (self.dfdt_grid > 0.0)
        )
        self.frequency_grid = self.frequency_grid[valid]
        self.t_of_f_grid = self.t_of_f_grid[valid]
        self.dfdt_grid = self.dfdt_grid[valid]

        if self.frequency_grid.size < 4:
            raise ValueError("TaylorF2 time grid has fewer than four valid points")
        if np.any(np.diff(self.t_of_f_grid) >= 0.0):
            raise ValueError("TaylorF2 t(f) must decrease with frequency")

        self._t_of_f = PchipInterpolator(
            self.frequency_grid, self.t_of_f_grid, extrapolate=False
        )
        self._f_of_t = PchipInterpolator(
            self.t_of_f_grid[::-1], self.frequency_grid[::-1], extrapolate=False
        )
        self._dfdt_of_f = PchipInterpolator(
            self.frequency_grid, self.dfdt_grid, extrapolate=False
        )
        self.duration = float(self.t_of_f_grid[0] - self.t_of_f_grid[-1])
        self.t_at_f_max = float(self.t_of_f_grid[-1])

    def frequency_after(
        self,
        f0_values: np.ndarray,
        elapsed_times: np.ndarray,
    ) -> np.ndarray:
        """Return f(t) after a chunk starts at each f0.

        Values after the track has chirped beyond f_max are returned as NaN
        because they are outside the requested in-band part of the chunk.
        """
        f0_values = np.asarray(f0_values, dtype=float)
        elapsed_times = np.asarray(elapsed_times, dtype=float)

        t0 = np.asarray(self._t_of_f(f0_values), dtype=float)
        target_t = t0[:, None] - elapsed_times[None, :]
        valid = target_t >= self.t_at_f_max

        frequency = np.full(target_t.shape, np.nan, dtype=float)
        frequency[valid] = self._f_of_t(target_t[valid])
        return frequency

    def frequency_since_band_start(self, elapsed_times: np.ndarray) -> np.ndarray:
        return self.frequency_after(np.asarray([self.frequency_grid[0]]), elapsed_times)[0]

    def dfdt_since_band_start(self, elapsed_times: np.ndarray) -> np.ndarray:
        frequency = self.frequency_since_band_start(elapsed_times)
        dfdt = np.full_like(frequency, np.nan, dtype=float)
        finite = np.isfinite(frequency)
        dfdt[finite] = self._dfdt_of_f(frequency[finite])
        return dfdt

    def max_dfdt(self) -> float:
        return float(np.max(self.dfdt_grid))


def dfdlogm_grid(
    mchirp_msun: float,
    config: BankConfig,
    f0_values: np.ndarray,
    elapsed_times: np.ndarray,
) -> np.ndarray:
    """Centered finite-difference derivative of the chunk track wrt log Mc."""
    h = config.finite_diff_dlogm
    low_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(-h), config)
    high_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(h), config)
    return dfdlogm_grid_from_tracks(
        low_track, high_track, h, f0_values, elapsed_times
    )


def dfdlogm_grid_from_tracks(
    low_track: TaylorF2FrequencyTrack,
    high_track: TaylorF2FrequencyTrack,
    finite_diff_dlogm: float,
    f0_values: np.ndarray,
    elapsed_times: np.ndarray,
) -> np.ndarray:
    """Centered finite-difference derivative using precomputed track objects."""
    f_low = low_track.frequency_after(f0_values, elapsed_times)
    f_high = high_track.frequency_after(f0_values, elapsed_times)
    derivative = (f_high - f_low) / (2.0 * finite_diff_dlogm)

    finite = np.isfinite(f_low) & np.isfinite(f_high)
    derivative[~finite] = np.nan
    return derivative


def scan_max_dfdlogm(config: BankConfig) -> dict[str, np.ndarray | float]:
    """Scan the requested parameter space for the worst Mc frequency derivative."""
    mchirp_scan = np.geomspace(
        config.mchirp_min, config.mchirp_max, config.n_mchirp_scan
    )
    f0_values = np.linspace(config.f_min, config.f_max, config.n_f0_scan)
    elapsed_times = np.linspace(0.0, config.t_coh, config.n_time_scan)

    max_by_mchirp = np.empty_like(mchirp_scan)
    for i, mchirp in enumerate(mchirp_scan):
        derivative = dfdlogm_grid(mchirp, config, f0_values, elapsed_times)
        finite = np.isfinite(derivative)
        if not np.any(finite):
            raise RuntimeError(f"no in-band TaylorF2 track points for Mc={mchirp:g}")
        max_by_mchirp[i] = float(np.nanmax(np.abs(derivative)))

    global_max = float(np.max(max_by_mchirp))
    if not np.isfinite(global_max) or global_max <= 0.0:
        raise RuntimeError("invalid TaylorF2 chirp-mass derivative scan")

    return {
        "mchirp_scan": mchirp_scan,
        "max_dfdlogm_by_mchirp": max_by_mchirp,
        "global_max_dfdlogm": global_max,
    }


def rectangular_axis_freq_error_hz(config: BankConfig) -> float:
    return 0.5 * config.freq_error_hz


def build_mchirp_bank(
    config: BankConfig,
    mchirp_scan: np.ndarray,
    max_dfdlogm_by_mchirp: np.ndarray,
) -> np.ndarray:
    """Return an adaptive log-Mc bank using the local template density."""
    log_min = np.log(config.mchirp_min)
    log_max = np.log(config.mchirp_max)
    derivative_of_logm = mchirp_derivative_interpolator(
        mchirp_scan, max_dfdlogm_by_mchirp
    )

    log_grid = np.linspace(
        log_min, log_max, max(4096, 32 * np.asarray(mchirp_scan).size)
    )
    density = np.asarray(derivative_of_logm(log_grid)) / (
        2.0 * rectangular_axis_freq_error_hz(config)
    )
    if np.any(~np.isfinite(density)) or np.any(density <= 0.0):
        raise RuntimeError("invalid local chirp-mass template density")

    cumulative = np.empty_like(log_grid)
    cumulative[0] = 0.0
    cumulative[1:] = np.cumsum(
        0.5 * (density[1:] + density[:-1]) * np.diff(log_grid)
    )
    template_density_volume = float(cumulative[-1])
    if not np.isfinite(template_density_volume) or template_density_volume <= 0.0:
        raise RuntimeError("invalid chirp-mass template density integral")

    n_intervals = int(np.ceil(template_density_volume))
    target_cumulative = np.linspace(0.0, template_density_volume, n_intervals + 1)
    return np.exp(np.interp(target_cumulative, cumulative, log_grid))


def mchirp_derivative_interpolator(
    mchirp_scan: np.ndarray,
    max_dfdlogm_by_mchirp: np.ndarray,
) -> PchipInterpolator:
    """Return an interpolator for max |df/dlogMc| as a function of log Mc."""
    log_scan = np.log(np.asarray(mchirp_scan, dtype=float))
    derivative_scan = np.asarray(max_dfdlogm_by_mchirp, dtype=float)
    valid = (
        np.isfinite(log_scan)
        & np.isfinite(derivative_scan)
        & (derivative_scan > 0.0)
    )
    if np.count_nonzero(valid) < 2:
        raise RuntimeError("need at least two valid chirp-mass derivative samples")

    sort_order = np.argsort(log_scan[valid])
    return PchipInterpolator(
        log_scan[valid][sort_order],
        derivative_scan[valid][sort_order],
        extrapolate=False,
    )


def mchirp_bank_edge_errors_hz(
    config: BankConfig,
    mchirp_bank: np.ndarray,
    mchirp_scan: np.ndarray,
    max_dfdlogm_by_mchirp: np.ndarray,
) -> np.ndarray:
    """Return local midpoint edge errors for adjacent chirp-mass templates."""
    if mchirp_bank.size < 2:
        return np.asarray([], dtype=float)

    log_scan = np.log(np.asarray(mchirp_scan, dtype=float))
    derivative_scan = np.asarray(max_dfdlogm_by_mchirp, dtype=float)
    valid = (
        np.isfinite(log_scan)
        & np.isfinite(derivative_scan)
        & (derivative_scan > 0.0)
    )
    derivative_of_logm = PchipInterpolator(
        log_scan[valid], derivative_scan[valid], extrapolate=False
    )

    log_bank = np.log(mchirp_bank)
    cell_width = np.diff(log_bank)
    cell_midpoint = 0.5 * (log_bank[:-1] + log_bank[1:])
    return 0.5 * cell_width * np.asarray(derivative_of_logm(cell_midpoint))


def t20_summary(mchirp_bank: np.ndarray, config: BankConfig) -> dict[str, np.ndarray]:
    """Return crossing-time spacing and counts for each chirp-mass template."""
    durations = np.empty_like(mchirp_bank)
    max_dfdt = np.empty_like(mchirp_bank)
    mean_dfdt = np.empty_like(mchirp_bank)
    spacing_limit = np.empty_like(mchirp_bank)
    max_spacing_limit = np.empty_like(mchirp_bank)
    actual_spacing = np.empty_like(mchirp_bank)
    t20_min = np.empty_like(mchirp_bank)
    t20_max = np.full_like(mchirp_bank, config.observation_seconds)
    counts = np.empty(mchirp_bank.size, dtype=np.int64)
    axis_error_hz = rectangular_axis_freq_error_hz(config)

    for i, mchirp in enumerate(mchirp_bank):
        track = TaylorF2FrequencyTrack(float(mchirp), config)
        durations[i] = track.duration
        max_dfdt[i] = track.max_dfdt()
        mean_dfdt[i] = (config.f_max - config.f_min) / durations[i]
        spacing_limit[i] = 2.0 * axis_error_hz / max_dfdt[i]
        max_spacing_limit[i] = 2.0 * axis_error_hz / track.dfdt_grid[0]
        t20_min[i] = -durations[i]

        t20_range = t20_max[i] - t20_min[i]
        interval_density_integral = (
            config.observation_seconds / spacing_limit[i]
            + (config.f_max - config.f_min) / (2.0 * axis_error_hz)
        )
        n_intervals = int(np.ceil(interval_density_integral))
        counts[i] = n_intervals + 1
        actual_spacing[i] = t20_range / n_intervals

    worst_edge_error = np.full_like(mchirp_bank, axis_error_hz)
    return {
        "duration_20_to_fmax_s": durations,
        "max_dfdt_hz_per_s": max_dfdt,
        "mean_dfdt_hz_per_s": mean_dfdt,
        "t20_spacing_min_s": spacing_limit,
        "t20_spacing_max_s": max_spacing_limit,
        "t20_spacing_s": actual_spacing,
        "t20_min_s": t20_min,
        "t20_max_s": t20_max,
        "n_t20_templates": counts,
        "t20_worst_edge_error_hz": worst_edge_error,
    }


def lookup_count_summary(
    t20: dict[str, np.ndarray],
    config: BankConfig,
) -> dict[str, np.ndarray | float | int]:
    """Count time-frequency grid entries touched by each Mc/t20 sub-bank.

    The time-frequency grid is assumed to have one power entry per coherent
    chunk, located at the chunk center.  A template looks up one frequency-bin
    entry for each chunk center that lies inside its in-band track interval
    [t20, t20 + duration_20_to_fmax].
    """
    durations = np.asarray(t20["duration_20_to_fmax_s"], dtype=float)
    t20_min = np.asarray(t20["t20_min_s"], dtype=float)
    t20_spacing = np.asarray(t20["t20_spacing_s"], dtype=float)
    n_t20_templates = np.asarray(t20["n_t20_templates"], dtype=np.int64)

    total_lookups = np.empty(n_t20_templates.size, dtype=np.int64)
    total_additions = np.empty(n_t20_templates.size, dtype=np.int64)
    nonzero_templates = np.empty(n_t20_templates.size, dtype=np.int64)
    min_lookups = np.empty(n_t20_templates.size, dtype=np.int64)
    mean_lookups = np.empty(n_t20_templates.size, dtype=float)
    max_lookups = np.empty(n_t20_templates.size, dtype=np.int64)

    for i in range(n_t20_templates.size):
        stats = lookup_count_stats_for_t20_axis(
            float(t20_min[i]),
            float(t20_spacing[i]),
            int(n_t20_templates[i]),
            float(durations[i]),
            config,
        )
        total_lookups[i] = stats["total_lookups"]
        total_additions[i] = stats["total_additions"]
        nonzero_templates[i] = stats["nonzero_templates"]
        min_lookups[i] = stats["min_lookups_per_template"]
        mean_lookups[i] = stats["mean_lookups_per_template"]
        max_lookups[i] = stats["max_lookups_per_template"]

    return {
        "time_grid_entries": time_grid_entry_count(config),
        "lookup_total_by_mchirp": total_lookups,
        "addition_total_by_mchirp": total_additions,
        "nonzero_lookup_templates_by_mchirp": nonzero_templates,
        "lookup_min_by_template_by_mchirp": min_lookups,
        "lookup_mean_by_template_by_mchirp": mean_lookups,
        "lookup_max_by_template_by_mchirp": max_lookups,
        "total_lookup_entries": int(np.sum(total_lookups, dtype=np.int64)),
        "total_sum_additions": int(np.sum(total_additions, dtype=np.int64)),
        "total_nonzero_lookup_templates": int(
            np.sum(nonzero_templates, dtype=np.int64)
        ),
    }


def time_grid_entry_count(config: BankConfig) -> int:
    """Return the number of full coherent chunks in the observation."""
    return int(np.floor(config.observation_seconds / config.t_coh))


def lookup_count_stats_for_t20_axis(
    t20_min: float,
    t20_spacing: float,
    n_t20_templates: int,
    duration: float,
    config: BankConfig,
    chunk_size: int = 1_000_000,
) -> dict[str, int | float]:
    """Return lookup-count stats for one uniformly spaced t20 axis."""
    n_time = time_grid_entry_count(config)
    if n_t20_templates <= 0 or n_time <= 0:
        return {
            "total_lookups": 0,
            "total_additions": 0,
            "nonzero_templates": 0,
            "min_lookups_per_template": 0,
            "mean_lookups_per_template": 0.0,
            "max_lookups_per_template": 0,
        }

    total_lookups = 0

    for start in range(0, n_time, chunk_size):
        stop = min(start + chunk_size, n_time)
        chunk_center = (np.arange(start, stop, dtype=float) + 0.5) * config.t_coh

        first = np.ceil((chunk_center - duration - t20_min) / t20_spacing).astype(
            np.int64
        )
        last = np.floor((chunk_center - t20_min) / t20_spacing).astype(np.int64)
        first = np.maximum(first, 0)
        last = np.minimum(last, n_t20_templates - 1)

        counts = last - first + 1
        counts[counts < 0] = 0

        total_lookups += int(np.sum(counts, dtype=np.int64))

    first_center = 0.5 * config.t_coh
    last_center = (n_time - 0.5) * config.t_coh
    first_nonzero = int(np.ceil((first_center - duration - t20_min) / t20_spacing))
    last_nonzero = int(np.floor((last_center - t20_min) / t20_spacing))
    first_nonzero = max(first_nonzero, 0)
    last_nonzero = min(last_nonzero, n_t20_templates - 1)
    nonzero_templates = max(0, last_nonzero - first_nonzero + 1)

    total_additions = total_lookups - nonzero_templates
    return {
        "total_lookups": total_lookups,
        "total_additions": total_additions,
        "nonzero_templates": nonzero_templates,
        "min_lookups_per_template": 0,
        "mean_lookups_per_template": total_lookups / n_t20_templates,
        "max_lookups_per_template": min(
            n_time, int(np.floor(duration / config.t_coh)) + 1
        ),
    }


def metric_2d_at_mchirp(mchirp_msun: float, config: BankConfig) -> dict[str, float]:
    """Return the local 2D metric for theta=(log Mc, t20).

    The metric is dimensionless: g_ab = <partial_a f partial_b f> / Delta f_max^2,
    where the average is over the in-band TaylorF2 track sampled uniformly in
    elapsed time.  The crossing-time derivative is partial_t20 f = -df/dt.
    """
    track = TaylorF2FrequencyTrack(float(mchirp_msun), config)
    h = config.finite_diff_dlogm
    low_track = TaylorF2FrequencyTrack(float(mchirp_msun) * np.exp(-h), config)
    high_track = TaylorF2FrequencyTrack(float(mchirp_msun) * np.exp(h), config)
    return metric_2d_from_tracks(
        float(mchirp_msun), config, track, low_track, high_track
    )


def metric_2d_from_tracks(
    mchirp_msun: float,
    config: BankConfig,
    track: TaylorF2FrequencyTrack,
    low_track: TaylorF2FrequencyTrack,
    high_track: TaylorF2FrequencyTrack,
) -> dict[str, float]:
    """Return the local 2D metric using precomputed TaylorF2 tracks."""
    elapsed = np.linspace(0.0, track.duration, config.n_metric_time_scan)
    dfdlogm = dfdlogm_grid_from_tracks(
        low_track,
        high_track,
        config.finite_diff_dlogm,
        np.asarray([config.f_min]),
        elapsed,
    )[0]
    dfdt20 = -track.dfdt_since_band_start(elapsed)

    finite = np.isfinite(dfdlogm) & np.isfinite(dfdt20)
    if np.count_nonzero(finite) < 2:
        raise RuntimeError(f"too few finite metric samples for Mc={mchirp_msun:g}")

    dfdlogm = dfdlogm[finite]
    dfdt20 = dfdt20[finite]
    inv_df2 = 1.0 / config.freq_error_hz**2

    g_logm_logm = float(np.mean(dfdlogm * dfdlogm) * inv_df2)
    g_logm_t20 = float(np.mean(dfdlogm * dfdt20) * inv_df2)
    g_t20_t20 = float(np.mean(dfdt20 * dfdt20) * inv_df2)
    determinant = max(0.0, float(g_logm_logm * g_t20_t20 - g_logm_t20**2))

    correlation = np.nan
    if g_logm_logm > 0.0 and g_t20_t20 > 0.0:
        correlation = g_logm_t20 / np.sqrt(g_logm_logm * g_t20_t20)

    return {
        "duration_20_to_fmax_s": track.duration,
        "g_logm_logm": g_logm_logm,
        "g_logm_t20": g_logm_t20,
        "g_t20_t20": g_t20_t20,
        "metric_det": determinant,
        "metric_sqrt_det": float(np.sqrt(determinant)),
        "metric_correlation": float(correlation),
    }


def metric_2d_summary(config: BankConfig) -> dict[str, np.ndarray | float]:
    """Compute a continuous 2D metric-bank count estimate.

    A hexagonal/A2 covering in locally Euclidean metric coordinates has
    fundamental area 3 sqrt(3) / 2 for covering radius one.  Since this metric
    uses Delta f_max as the allowed rms frequency error, the count estimate is
    integral sqrt(det(g)) dlogMc dt20 divided by that area.
    """
    mchirp = np.geomspace(config.mchirp_min, config.mchirp_max, config.n_mchirp_scan)
    log_mchirp = np.log(mchirp)

    duration = np.empty_like(mchirp)
    g_logm_logm = np.empty_like(mchirp)
    g_logm_t20 = np.empty_like(mchirp)
    g_t20_t20 = np.empty_like(mchirp)
    metric_det = np.empty_like(mchirp)
    metric_sqrt_det = np.empty_like(mchirp)
    metric_correlation = np.empty_like(mchirp)

    for i, mc in enumerate(mchirp):
        metric = metric_2d_at_mchirp(float(mc), config)
        duration[i] = metric["duration_20_to_fmax_s"]
        g_logm_logm[i] = metric["g_logm_logm"]
        g_logm_t20[i] = metric["g_logm_t20"]
        g_t20_t20[i] = metric["g_t20_t20"]
        metric_det[i] = metric["metric_det"]
        metric_sqrt_det[i] = metric["metric_sqrt_det"]
        metric_correlation[i] = metric["metric_correlation"]

    return finish_metric_2d_summary(
        config,
        mchirp,
        log_mchirp,
        duration,
        g_logm_logm,
        g_logm_t20,
        g_t20_t20,
        metric_det,
        metric_sqrt_det,
        metric_correlation,
    )


def scan_max_dfdlogm_and_metric_summary(
    config: BankConfig,
) -> tuple[dict[str, np.ndarray | float], dict[str, np.ndarray | float]]:
    """Compute the worst dfdlogMc scan and 2D metric in one mass-scan pass."""
    mchirp = np.geomspace(config.mchirp_min, config.mchirp_max, config.n_mchirp_scan)
    log_mchirp = np.log(mchirp)
    f0_values = np.linspace(config.f_min, config.f_max, config.n_f0_scan)
    elapsed_times = np.linspace(0.0, config.t_coh, config.n_time_scan)
    h = config.finite_diff_dlogm

    max_by_mchirp = np.empty_like(mchirp)
    duration = np.empty_like(mchirp)
    g_logm_logm = np.empty_like(mchirp)
    g_logm_t20 = np.empty_like(mchirp)
    g_t20_t20 = np.empty_like(mchirp)
    metric_det = np.empty_like(mchirp)
    metric_sqrt_det = np.empty_like(mchirp)
    metric_correlation = np.empty_like(mchirp)

    for i, mc in enumerate(mchirp):
        low_track = TaylorF2FrequencyTrack(float(mc) * np.exp(-h), config)
        high_track = TaylorF2FrequencyTrack(float(mc) * np.exp(h), config)

        derivative = dfdlogm_grid_from_tracks(
            low_track, high_track, h, f0_values, elapsed_times
        )
        finite = np.isfinite(derivative)
        if not np.any(finite):
            raise RuntimeError(f"no in-band TaylorF2 track points for Mc={mc:g}")
        max_by_mchirp[i] = float(np.nanmax(np.abs(derivative)))

        track = TaylorF2FrequencyTrack(float(mc), config)
        metric = metric_2d_from_tracks(
            float(mc), config, track, low_track, high_track
        )
        duration[i] = metric["duration_20_to_fmax_s"]
        g_logm_logm[i] = metric["g_logm_logm"]
        g_logm_t20[i] = metric["g_logm_t20"]
        g_t20_t20[i] = metric["g_t20_t20"]
        metric_det[i] = metric["metric_det"]
        metric_sqrt_det[i] = metric["metric_sqrt_det"]
        metric_correlation[i] = metric["metric_correlation"]

    global_max = float(np.max(max_by_mchirp))
    if not np.isfinite(global_max) or global_max <= 0.0:
        raise RuntimeError("invalid TaylorF2 chirp-mass derivative scan")

    scan = {
        "mchirp_scan": mchirp,
        "max_dfdlogm_by_mchirp": max_by_mchirp,
        "global_max_dfdlogm": global_max,
    }
    metric = finish_metric_2d_summary(
        config,
        mchirp,
        log_mchirp,
        duration,
        g_logm_logm,
        g_logm_t20,
        g_t20_t20,
        metric_det,
        metric_sqrt_det,
        metric_correlation,
    )
    return scan, metric


def finish_metric_2d_summary(
    config: BankConfig,
    mchirp: np.ndarray,
    log_mchirp: np.ndarray,
    duration: np.ndarray,
    g_logm_logm: np.ndarray,
    g_logm_t20: np.ndarray,
    g_t20_t20: np.ndarray,
    metric_det: np.ndarray,
    metric_sqrt_det: np.ndarray,
    metric_correlation: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Assemble metric arrays and integrated count estimates."""
    t20_range = config.observation_seconds + duration
    metric_volume_density = metric_sqrt_det * t20_range
    metric_volume = float(np.trapezoid(metric_volume_density, log_mchirp))
    hex_cell_area = 3.0 * np.sqrt(3.0) / 2.0
    square_cell_area = 2.0

    return {
        "metric_mchirp_scan": mchirp,
        "metric_log_mchirp_scan": log_mchirp,
        "metric_duration_20_to_fmax_s": duration,
        "metric_t20_range_s": t20_range,
        "g_logm_logm": g_logm_logm,
        "g_logm_t20": g_logm_t20,
        "g_t20_t20": g_t20_t20,
        "metric_det": metric_det,
        "metric_sqrt_det": metric_sqrt_det,
        "metric_correlation": metric_correlation,
        "metric_volume_density": metric_volume_density,
        "metric_volume": metric_volume,
        "metric_hex_count_estimate": metric_volume / hex_cell_area,
        "metric_square_count_estimate": metric_volume / square_cell_area,
    }


def write_mchirp_bank_csv(path: Path, mchirp_bank: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["index", "mchirp_msun", "delta_log_mchirp_to_next"])
        deltas = np.diff(np.log(mchirp_bank))
        for i, mchirp in enumerate(mchirp_bank):
            delta = deltas[i] if i < deltas.size else ""
            writer.writerow([i, f"{mchirp:.16e}", delta])


def write_t20_summary_csv(
    path: Path,
    mchirp_bank: np.ndarray,
    summary: dict[str, np.ndarray],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mchirp_index",
                "mchirp_msun",
                "duration_20_to_fmax_s",
                "max_dfdt_hz_per_s",
                "mean_dfdt_hz_per_s",
                "t20_min_s",
                "t20_max_s",
                "t20_spacing_min_s",
                "t20_spacing_max_s",
                "t20_spacing_s",
                "n_t20_templates",
                "t20_worst_edge_error_hz",
            ]
        )
        for i, mchirp in enumerate(mchirp_bank):
            writer.writerow(
                [
                    i,
                    f"{mchirp:.16e}",
                    f"{summary['duration_20_to_fmax_s'][i]:.16e}",
                    f"{summary['max_dfdt_hz_per_s'][i]:.16e}",
                    f"{summary['mean_dfdt_hz_per_s'][i]:.16e}",
                    f"{summary['t20_min_s'][i]:.16e}",
                    f"{summary['t20_max_s'][i]:.16e}",
                    f"{summary['t20_spacing_min_s'][i]:.16e}",
                    f"{summary['t20_spacing_max_s'][i]:.16e}",
                    f"{summary['t20_spacing_s'][i]:.16e}",
                    int(summary["n_t20_templates"][i]),
                    f"{summary['t20_worst_edge_error_hz'][i]:.16e}",
                ]
            )




def write_2d_metric_csv(path: Path, metric: dict[str, np.ndarray | float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mchirp = np.asarray(metric["metric_mchirp_scan"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "mchirp_msun",
                "duration_20_to_fmax_s",
                "t20_range_s",
                "g_logm_logm",
                "g_logm_t20",
                "g_t20_t20",
                "metric_det",
                "metric_sqrt_det",
                "metric_correlation",
                "metric_volume_density",
            ]
        )
        for i, mc in enumerate(mchirp):
            writer.writerow(
                [
                    i,
                    f"{mc:.16e}",
                    f"{metric['metric_duration_20_to_fmax_s'][i]:.16e}",
                    f"{metric['metric_t20_range_s'][i]:.16e}",
                    f"{metric['g_logm_logm'][i]:.16e}",
                    f"{metric['g_logm_t20'][i]:.16e}",
                    f"{metric['g_t20_t20'][i]:.16e}",
                    f"{metric['metric_det'][i]:.16e}",
                    f"{metric['metric_sqrt_det'][i]:.16e}",
                    f"{metric['metric_correlation'][i]:.16e}",
                    f"{metric['metric_volume_density'][i]:.16e}",
                ]
            )


def write_lookup_summary_csv(
    path: Path,
    mchirp_bank: np.ndarray,
    lookup: dict[str, np.ndarray | float | int],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "mchirp_index",
                "mchirp_msun",
                "lookup_min_per_template",
                "lookup_mean_per_template",
                "lookup_max_per_template",
                "nonzero_lookup_templates",
                "total_lookup_entries",
                "total_sum_additions",
            ]
        )
        for i, mchirp in enumerate(mchirp_bank):
            writer.writerow(
                [
                    i,
                    f"{mchirp:.16e}",
                    int(lookup["lookup_min_by_template_by_mchirp"][i]),
                    f"{lookup['lookup_mean_by_template_by_mchirp'][i]:.6f}",
                    int(lookup["lookup_max_by_template_by_mchirp"][i]),
                    int(lookup["nonzero_lookup_templates_by_mchirp"][i]),
                    int(lookup["lookup_total_by_mchirp"][i]),
                    int(lookup["addition_total_by_mchirp"][i]),
                ]
            )


def format_seconds(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.1f} s"
    if seconds < 3600.0:
        return f"{seconds / 60.0:.1f} min"
    if seconds < 86400.0:
        return f"{seconds / 3600.0:.1f} hr"
    return f"{seconds / 86400.0:.1f} days"


def make_plot(
    path: Path,
    mchirp_bank: np.ndarray,
    scan: dict[str, np.ndarray | float],
    t20: dict[str, np.ndarray],
    metric: dict[str, np.ndarray | float],
    config: BankConfig,
) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(8.0, 11.0), sharex=True)

    mchirp_scan = np.asarray(scan["mchirp_scan"])
    max_by_mchirp = np.asarray(scan["max_dfdlogm_by_mchirp"])
    global_max = float(scan["global_max_dfdlogm"])

    axes[0].loglog(mchirp_scan, max_by_mchirp, color="tab:blue", lw=1.8)
    axes[0].axhline(global_max, color="tab:red", ls="--", lw=1.2)
    axes[0].set_ylabel(r"$\max |\partial f / \partial \log M_c|$ [Hz]")
    axes[0].grid(True, which="both", alpha=0.25)

    axes[1].vlines(mchirp_bank, 0.0, 1.0, color="black", lw=0.5)
    axes[1].set_xscale("log")
    axes[1].set_yticks([])
    axes[1].set_xlabel(r"$M_c$ [$M_\odot$]")
    axes[1].set_ylabel("templates")
    axes[1].grid(True, which="both", axis="x", alpha=0.25)

    axes[2].loglog(
        mchirp_bank,
        t20["t20_spacing_s"],
        color="tab:green",
        lw=1.8,
    )
    axes[2].set_xlabel(r"$M_c$ [$M_\odot$]")
    axes[2].set_ylabel(r"$\Delta t_{20}$ [s]")
    axes[2].grid(True, which="both", alpha=0.25)

    axes[3].loglog(
        metric["metric_mchirp_scan"],
        metric["metric_sqrt_det"],
        color="tab:purple",
        lw=1.8,
    )
    axes[3].set_xlabel(r"$M_c$ [$M_\odot$]")
    axes[3].set_ylabel(r"$\sqrt{\det g}$")
    axes[3].grid(True, which="both", alpha=0.25)

    fig.suptitle(
        (
            "TaylorF2 semicoherent bank, "
            rf"$f_0 \in [{config.f_min:g}, {config.f_max:g}]$ Hz, "
            rf"$T_{{coh}}={config.t_coh:g}$ s"
        ),
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TaylorF2 semicoherent Mc and t20 bank for the requested "
            "coherent chunk duration."
        )
    )
    parser.add_argument("--mchirp-min", type=float, default=DEFAULT_MCHIRP_MIN)
    parser.add_argument("--mchirp-max", type=float, default=DEFAULT_MCHIRP_MAX)
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument(
        "--t-coh",
        type=float,
        default=DEFAULT_T_COH,
        help="Coherent chunk duration in seconds.",
    )
    parser.add_argument(
        "--observation-days", type=float, default=DEFAULT_OBSERVATION_DAYS
    )
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument(
        "--freq-error-hz",
        type=float,
        default=DEFAULT_FREQ_ERROR_HZ,
        help="Maximum nearest-template instantaneous frequency error in Hz.",
    )
    parser.add_argument("--n-mchirp-scan", type=int, default=DEFAULT_N_MCHIRP_SCAN)
    parser.add_argument("--n-f0-scan", type=int, default=DEFAULT_N_F0_SCAN)
    parser.add_argument("--n-time-scan", type=int, default=DEFAULT_N_TIME_SCAN)
    parser.add_argument(
        "--n-metric-time-scan", type=int, default=DEFAULT_N_METRIC_TIME_SCAN
    )
    parser.add_argument(
        "--n-taylorf2-grid", type=int, default=DEFAULT_N_TAYLORF2_GRID
    )
    parser.add_argument(
        "--finite-diff-dlogm", type=float, default=DEFAULT_FINITE_DIFF_DLOGM
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def config_from_args(args: argparse.Namespace) -> BankConfig:
    if args.mchirp_min <= 0.0 or args.mchirp_max <= args.mchirp_min:
        raise ValueError("require 0 < mchirp-min < mchirp-max")
    if args.f_min <= 0.0 or args.f_max <= args.f_min:
        raise ValueError("require 0 < f-min < f-max")
    if args.t_coh <= 0.0:
        raise ValueError("t-coh must be positive")
    if args.observation_days <= 0.0:
        raise ValueError("observation-days must be positive")
    if args.freq_error_hz <= 0.0:
        raise ValueError("freq-error-hz must be positive")
    if args.n_mchirp_scan < 2 or args.n_f0_scan < 2 or args.n_time_scan < 2:
        raise ValueError("scan sizes must be at least 2")
    if args.n_metric_time_scan < 2:
        raise ValueError("n-metric-time-scan must be at least 2")
    if args.n_taylorf2_grid < 16:
        raise ValueError("n-taylorf2-grid must be at least 16")
    if args.finite_diff_dlogm <= 0.0:
        raise ValueError("finite-diff-dlogm must be positive")

    return BankConfig(
        mchirp_min=args.mchirp_min,
        mchirp_max=args.mchirp_max,
        f_min=args.f_min,
        f_max=args.f_max,
        t_coh=args.t_coh,
        observation_days=args.observation_days,
        eta=args.eta,
        freq_error_hz=args.freq_error_hz,
        n_mchirp_scan=args.n_mchirp_scan,
        n_f0_scan=args.n_f0_scan,
        n_time_scan=args.n_time_scan,
        n_metric_time_scan=args.n_metric_time_scan,
        n_taylorf2_grid=args.n_taylorf2_grid,
        finite_diff_dlogm=args.finite_diff_dlogm,
        output_dir=args.output_dir,
    )


def main() -> None:
    config = config_from_args(parse_args())
    config.output_dir.mkdir(parents=True, exist_ok=True)

    scan, metric = scan_max_dfdlogm_and_metric_summary(config)
    bank = build_mchirp_bank(
        config,
        np.asarray(scan["mchirp_scan"]),
        np.asarray(scan["max_dfdlogm_by_mchirp"]),
    )
    t20 = t20_summary(bank, config)
    dlogm = float(np.max(np.diff(np.log(bank)))) if bank.size > 1 else 0.0
    mchirp_edge_errors = mchirp_bank_edge_errors_hz(
        config,
        bank,
        np.asarray(scan["mchirp_scan"]),
        np.asarray(scan["max_dfdlogm_by_mchirp"]),
    )
    mchirp_worst_edge_error = (
        float(np.max(mchirp_edge_errors)) if mchirp_edge_errors.size else 0.0
    )
    t20_worst_edge_error = float(np.max(t20["t20_worst_edge_error_hz"]))
    total_t20_templates = int(np.sum(t20["n_t20_templates"]))
    lookup = lookup_count_summary(t20, config)

    mchirp_csv_path = config.output_dir / "semicoherent_pbh_mchirp_bank.csv"
    t20_csv_path = config.output_dir / "semicoherent_pbh_t20_summary.csv"
    metric_csv_path = config.output_dir / "semicoherent_pbh_2d_metric.csv"
    lookup_csv_path = config.output_dir / "semicoherent_pbh_lookup_cost.csv"
    npz_path = config.output_dir / "semicoherent_pbh_mchirp_bank.npz"
    plot_path = config.output_dir / "semicoherent_pbh_mchirp_bank.png"

    write_mchirp_bank_csv(mchirp_csv_path, bank)
    write_t20_summary_csv(t20_csv_path, bank, t20)
    write_2d_metric_csv(metric_csv_path, metric)
    write_lookup_summary_csv(lookup_csv_path, bank, lookup)
    np.savez(
        npz_path,
        mchirp_bank=bank,
        rectangular_axis_freq_error_hz=rectangular_axis_freq_error_hz(config),
        **t20,
        **metric,
        **lookup,
        mchirp_scan=scan["mchirp_scan"],
        max_dfdlogm_by_mchirp=scan["max_dfdlogm_by_mchirp"],
        global_max_dfdlogm=scan["global_max_dfdlogm"],
        mchirp_edge_error_hz_by_interval=mchirp_edge_errors,
        mchirp_worst_edge_error_hz=mchirp_worst_edge_error,
        max_t20_worst_edge_error_hz=t20_worst_edge_error,
        total_t20_templates=total_t20_templates,
        mchirp_min=config.mchirp_min,
        mchirp_max=config.mchirp_max,
        f_min=config.f_min,
        f_max=config.f_max,
        t_coh=config.t_coh,
        observation_days=config.observation_days,
        eta=config.eta,
        freq_error_hz=config.freq_error_hz,
    )
    make_plot(plot_path, bank, scan, t20, metric, config)

    print("TaylorF2 semicoherent Mc/t20 bank")
    print(f"  Mc range: {config.mchirp_min:.1e}--{config.mchirp_max:.1e} Msun")
    print(f"  f0 range: {config.f_min:.6g}--{config.f_max:.6g} Hz")
    print(f"  coherent chunk: {config.t_coh:.6g} s")
    print(f"  observation: {config.observation_days:.6g} days")
    # print(f"  rectangular per-axis error: {rectangular_axis_freq_error_hz(config):.6e} Hz")
    print(f"  Mc templates: {bank.size}")
    # print(f"  delta log Mc: {dlogm:.6e}")
    # print(f"  global max |df/dlogMc|: {float(scan['global_max_dfdlogm']):.6e} Hz")
    # print(f"  Mc worst nearest-template error: {mchirp_worst_edge_error:.6e} Hz")
    print(
        "  t20 spacing range: "
        f"{float(np.min(t20['t20_spacing_s'])):.1e}--"
        f"{float(np.max(t20['t20_spacing_s'])):.1e} s"
    )
    print(f"  total templates: {total_t20_templates:.1e}")
    print(f"  time grid chunks: {lookup['time_grid_entries']}")
    print(
        "  lookups per template: "
        f"{int(np.min(lookup['lookup_min_by_template_by_mchirp']))}--"
        f"{int(np.max(lookup['lookup_max_by_template_by_mchirp']))}, "
        f"mean {lookup['total_lookup_entries'] / total_t20_templates:.1f}"
    )
    print(f"  total grid-entry lookups: {lookup['total_lookup_entries']:.3e}")
    print(f"  total summation additions: {lookup['total_sum_additions']:.3e}")
    for rate in (1.0e6, 1.0e7, 1.0e8, 1.0e9):
        seconds = lookup["total_lookup_entries"] / rate
        print(
            f"  lookup+sum time at {rate:.0e} entries/s: {format_seconds(seconds)}"
        )
    # print(f"  t20 worst nearest-template error: {t20_worst_edge_error:.3e} Hz")
    print(
        "  2D metric hex count estimate: "
        f"{float(metric['metric_hex_count_estimate']):.1e}"
    )
    print(
        "  2D metric square count estimate: "
        f"{float(metric['metric_square_count_estimate']):.1e}"
    )
    print(f"  wrote {mchirp_csv_path}")
    print(f"  wrote {t20_csv_path}")
    print(f"  wrote {metric_csv_path}")
    print(f"  wrote {lookup_csv_path}")
    print(f"  wrote {npz_path}")
    print(f"  wrote {plot_path}")


if __name__ == "__main__":
    main()
