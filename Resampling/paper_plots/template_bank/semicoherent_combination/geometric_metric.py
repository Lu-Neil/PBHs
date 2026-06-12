"""Geometric semicoherent metric for TaylorF2 frequency-time tracks.

This script estimates a template count for the stack-slide track bank using a
local mismatch metric, analogous to waveform-metric bank placement.  The
parameters are theta = (log Mc, t20).  The mismatch is the small-offset loss of
the coherent frequency-bin power, averaged over the observed in-band part of a
TaylorF2 track.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import lal
import numpy as np

try:
    from .semicoherent_pbh_bank import (
        DEFAULT_ETA,
        DEFAULT_F_MAX,
        DEFAULT_F_MIN,
        DEFAULT_FINITE_DIFF_DLOGM,
        DEFAULT_FREQ_ERROR_HZ,
        DEFAULT_MCHIRP_MAX,
        DEFAULT_MCHIRP_MIN,
        DEFAULT_N_TAYLORF2_GRID,
        DEFAULT_OBSERVATION_DAYS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_T_COH,
        BankConfig,
        TaylorF2FrequencyTrack,
        dfdlogm_grid_from_tracks,
    )
except ImportError:
    from semicoherent_pbh_bank import (
        DEFAULT_ETA,
        DEFAULT_F_MAX,
        DEFAULT_F_MIN,
        DEFAULT_FINITE_DIFF_DLOGM,
        DEFAULT_FREQ_ERROR_HZ,
        DEFAULT_MCHIRP_MAX,
        DEFAULT_MCHIRP_MIN,
        DEFAULT_N_TAYLORF2_GRID,
        DEFAULT_OBSERVATION_DAYS,
        DEFAULT_OUTPUT_DIR,
        DEFAULT_T_COH,
        BankConfig,
        TaylorF2FrequencyTrack,
        dfdlogm_grid_from_tracks,
    )


DEFAULT_N_MCHIRP_SCAN = 40
DEFAULT_N_T20_SCAN = 60
DEFAULT_MAX_MISMATCH = 0.1
DEFAULT_RESPONSE = "bin-averaged-sinc2"
DEFAULT_WEIGHT_MODEL = "uniform"
BIN_AVERAGE_MAX_ELEMENTS = 4_000_000

G = lal.G_SI
C = lal.C_SI
MSUN = lal.MSUN_SI
PI = np.pi


@dataclass(frozen=True)
class StackSamples:
    """Frequency-track derivatives evaluated at valid stack-slide samples."""

    elapsed: np.ndarray
    dfdlogm_bins: np.ndarray
    dfdt20_bins: np.ndarray
    bin_width_hz: np.ndarray
    weights: np.ndarray


def sinc2(offset_bins: np.ndarray | float) -> np.ndarray:
    """Return rectangular-window Fourier-bin power response."""
    return np.sinc(offset_bins) ** 2


def bin_averaged_sinc2(
    offset_bins: np.ndarray | float, n_bin_offset_scan: int = 4096
) -> np.ndarray | float:
    """Average nearest-bin power over unknown true offset in [-1/2, 1/2]."""
    offset_bins = np.asarray(offset_bins, dtype=float)
    scalar_input = offset_bins.ndim == 0
    flat_offsets = offset_bins.ravel()
    values = np.empty_like(flat_offsets)

    bin_offset = np.linspace(-0.5, 0.5, n_bin_offset_scan)
    chunk_size = max(1, BIN_AVERAGE_MAX_ELEMENTS // n_bin_offset_scan)
    for start in range(0, flat_offsets.size, chunk_size):
        stop = min(start + chunk_size, flat_offsets.size)
        response = sinc2(bin_offset[:, None] + flat_offsets[None, start:stop])
        values[start:stop] = np.trapezoid(response, bin_offset, axis=0)

    values = values.reshape(offset_bins.shape)
    if scalar_input:
        return float(values)
    return values


def response_power(offset_bins: np.ndarray, response: str) -> np.ndarray:
    """Return coherent-bin power retained at the requested bin offsets."""
    if response == "sinc2":
        return sinc2(offset_bins)
    if response == "bin-averaged-sinc2":
        return np.asarray(bin_averaged_sinc2(offset_bins), dtype=float)
    raise ValueError(f"unknown response model: {response}")


def response_curvature(response: str) -> float:
    """Return kappa for mismatch ~= kappa * offset_bins**2."""
    if response == "sinc2":
        return float(np.pi**2 / 3.0)
    if response == "bin-averaged-sinc2":
        eps = 1.0e-4
        response_at_zero = bin_averaged_sinc2(0.0)
        response_at_eps = bin_averaged_sinc2(eps)
        return float((1.0 - response_at_eps / response_at_zero) / eps**2)
    raise ValueError(f"unknown response model: {response}")


def beta_0pn(f_start_hz: np.ndarray, mchirp_msun: float) -> np.ndarray:
    """Return the leading-order chirp beta at the requested frequency."""
    mchirp_sec = G * (mchirp_msun * MSUN) / C**3
    return (
        (96.0 / 5.0)
        * PI ** (8.0 / 3.0)
        * mchirp_sec ** (5.0 / 3.0)
        * np.asarray(f_start_hz, dtype=float) ** (8.0 / 3.0)
    )


def tau_0pn_duration(duration_s: float, beta: np.ndarray) -> np.ndarray:
    """Return the 0PN resampled duration for one coherent chunk."""
    beta = np.asarray(beta, dtype=float)
    bracket = 1.0 - (8.0 / 3.0) * beta * duration_s
    tau = np.full_like(beta, np.nan, dtype=float)
    valid = bracket > 0.0
    tau[valid] = (
        3.0
        / (5.0 * beta[valid])
        * (1.0 - bracket[valid] ** (5.0 / 8.0))
    )
    return tau


def chunk_center_elapsed_samples(
    t20: float,
    track_duration: float,
    config: BankConfig,
) -> np.ndarray:
    """Return elapsed track times sampled by actual coherent chunk centers."""
    n_time = int(np.floor(config.observation_seconds / config.t_coh))
    if n_time <= 0:
        return np.asarray([], dtype=float)

    first_chunk = max(0, int(np.ceil((t20 / config.t_coh) - 0.5)))
    last_chunk = min(
        n_time - 1,
        int(np.floor(((t20 + track_duration) / config.t_coh) - 0.5)),
    )
    if last_chunk < first_chunk:
        return np.asarray([], dtype=float)

    chunk_center = np.arange(first_chunk, last_chunk + 1, dtype=float) + 0.5
    chunk_center *= config.t_coh
    elapsed = chunk_center - t20
    return elapsed[(elapsed >= 0.0) & (elapsed <= track_duration)]


def coherent_bin_width_hz(
    mchirp_msun: float,
    track: TaylorF2FrequencyTrack,
    elapsed: np.ndarray,
    config: BankConfig,
) -> np.ndarray:
    """Return the per-chunk NUFFT bin width in physical Hz."""
    frequency = track.frequency_since_band_start(elapsed)
    beta = beta_0pn(frequency, mchirp_msun)
    tau_duration = tau_0pn_duration(config.t_coh, beta)
    bin_width = np.full_like(tau_duration, np.nan, dtype=float)
    valid = tau_duration > 0.0
    bin_width[valid] = 1.0 / tau_duration[valid]
    return bin_width


def stack_slide_weights(elapsed: np.ndarray, weight_model: str) -> np.ndarray:
    """Return statistic weights for the visible stack-slide samples."""
    if weight_model == "uniform":
        return np.ones_like(elapsed, dtype=float)
    raise ValueError(f"unknown weight model: {weight_model}")


def finite_mean(values: np.ndarray, default: float = np.nan) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return default
    return float(np.mean(values[finite]))


def finite_median(values: np.ndarray, default: float = np.nan) -> float:
    finite = np.isfinite(values)
    if not np.any(finite):
        return default
    return float(np.median(values[finite]))


def collect_stack_samples(
    mchirp_msun: float,
    t20: float,
    config: BankConfig,
    track: TaylorF2FrequencyTrack,
    low_track: TaylorF2FrequencyTrack,
    high_track: TaylorF2FrequencyTrack,
    weight_model: str,
) -> StackSamples | None:
    """Evaluate valid frequency-offset derivatives at stack-slide samples."""
    elapsed = chunk_center_elapsed_samples(t20, track.duration, config)
    if elapsed.size == 0:
        return None

    dfdlogm_hz = dfdlogm_grid_from_tracks(
        low_track,
        high_track,
        config.finite_diff_dlogm,
        np.asarray([config.f_min]),
        elapsed,
    )[0]
    dfdt20_hz = -track.dfdt_since_band_start(elapsed)
    bin_width_hz = coherent_bin_width_hz(mchirp_msun, track, elapsed, config)
    weights = stack_slide_weights(elapsed, weight_model)

    finite = (
        np.isfinite(dfdlogm_hz)
        & np.isfinite(dfdt20_hz)
        & np.isfinite(bin_width_hz)
        & (bin_width_hz > 0.0)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(finite):
        return None

    bin_width_hz = bin_width_hz[finite]
    return StackSamples(
        elapsed=elapsed[finite],
        dfdlogm_bins=dfdlogm_hz[finite] / bin_width_hz,
        dfdt20_bins=dfdt20_hz[finite] / bin_width_hz,
        bin_width_hz=bin_width_hz,
        weights=weights[finite],
    )


def empty_metric_terms() -> dict[str, float]:
    return {
        "observed_duration_s": 0.0,
        "n_stack_samples": 0,
        "mean_bin_width_hz": np.nan,
        "g_logm_logm": 0.0,
        "g_logm_t20": 0.0,
        "g_t20_t20": 0.0,
        "metric_det": 0.0,
        "metric_sqrt_det": 0.0,
        "metric_correlation": np.nan,
    }


def stack_slide_metric_terms(
    mchirp_msun: float,
    t20: float,
    config: BankConfig,
    track: TaylorF2FrequencyTrack,
    low_track: TaylorF2FrequencyTrack,
    high_track: TaylorF2FrequencyTrack,
    kappa: float,
    weight_model: str,
) -> dict[str, float]:
    """Return local metric entries for the actual stack-slide samples."""
    samples = collect_stack_samples(
        mchirp_msun,
        t20,
        config,
        track,
        low_track,
        high_track,
        weight_model,
    )
    if samples is None:
        return empty_metric_terms()

    weight_sum = float(np.sum(samples.weights))
    g_logm_logm = float(
        kappa * np.sum(samples.weights * samples.dfdlogm_bins**2) / weight_sum
    )
    g_logm_t20 = float(
        kappa
        * np.sum(samples.weights * samples.dfdlogm_bins * samples.dfdt20_bins)
        / weight_sum
    )
    g_t20_t20 = float(
        kappa * np.sum(samples.weights * samples.dfdt20_bins**2) / weight_sum
    )
    determinant = max(0.0, g_logm_logm * g_t20_t20 - g_logm_t20**2)

    correlation = np.nan
    if g_logm_logm > 0.0 and g_t20_t20 > 0.0:
        correlation = g_logm_t20 / np.sqrt(g_logm_logm * g_t20_t20)

    return {
        "observed_duration_s": float(
            samples.elapsed[-1] - samples.elapsed[0] + config.t_coh
        ),
        "n_stack_samples": int(samples.elapsed.size),
        "mean_bin_width_hz": float(
            np.sum(samples.weights * samples.bin_width_hz) / weight_sum
        ),
        "g_logm_logm": g_logm_logm,
        "g_logm_t20": g_logm_t20,
        "g_t20_t20": g_t20_t20,
        "metric_det": determinant,
        "metric_sqrt_det": float(np.sqrt(determinant)),
        "metric_correlation": float(correlation),
    }


def stack_slide_retained_power(
    mchirp_msun: float,
    t20: float,
    delta_logm: float,
    delta_t20: float,
    config: BankConfig,
    response: str = DEFAULT_RESPONSE,
    weight_model: str = DEFAULT_WEIGHT_MODEL,
) -> float:
    """Return finite-offset retained stack-slide power for one nearby track."""
    track = TaylorF2FrequencyTrack(mchirp_msun, config)
    h = config.finite_diff_dlogm
    low_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(-h), config)
    high_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(h), config)
    samples = collect_stack_samples(
        mchirp_msun,
        t20,
        config,
        track,
        low_track,
        high_track,
        weight_model,
    )
    if samples is None:
        return np.nan

    offset_bins = (
        samples.dfdlogm_bins * delta_logm + samples.dfdt20_bins * delta_t20
    )
    retained = response_power(offset_bins, response)
    reference = response_power(np.zeros_like(offset_bins), response)
    return float(
        np.sum(samples.weights * retained) / np.sum(samples.weights * reference)
    )


def metric_at_mchirp_t20(
    mchirp_msun: float,
    t20: float,
    config: BankConfig,
    kappa: float,
    weight_model: str = DEFAULT_WEIGHT_MODEL,
) -> dict[str, float]:
    """Return local metric entries at one (Mc, t20)."""
    h = config.finite_diff_dlogm
    track = TaylorF2FrequencyTrack(mchirp_msun, config)
    low_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(-h), config)
    high_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(h), config)
    return stack_slide_metric_terms(
        mchirp_msun,
        t20,
        config,
        track,
        low_track,
        high_track,
        kappa,
        weight_model,
    )


def metric_volume_by_mchirp(
    mchirp_msun: float,
    config: BankConfig,
    kappa: float,
    n_t20_scan: int,
    weight_model: str,
) -> dict[str, float]:
    """Integrate sqrt(det g) over t20 for one chirp mass."""
    track = TaylorF2FrequencyTrack(mchirp_msun, config)
    t20_values = np.linspace(
        -track.duration, config.observation_seconds, n_t20_scan
    )

    sqrt_det = np.empty_like(t20_values)
    observed_duration = np.empty_like(t20_values)
    n_stack_samples = np.empty_like(t20_values)
    mean_bin_width = np.empty_like(t20_values)
    correlation = np.empty_like(t20_values)
    g_logm_logm = np.empty_like(t20_values)
    g_logm_t20 = np.empty_like(t20_values)
    g_t20_t20 = np.empty_like(t20_values)

    h = config.finite_diff_dlogm
    low_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(-h), config)
    high_track = TaylorF2FrequencyTrack(mchirp_msun * np.exp(h), config)

    for i, t20 in enumerate(t20_values):
        terms = stack_slide_metric_terms(
            mchirp_msun,
            float(t20),
            config,
            track,
            low_track,
            high_track,
            kappa,
            weight_model,
        )
        observed_duration[i] = terms["observed_duration_s"]
        n_stack_samples[i] = terms["n_stack_samples"]
        mean_bin_width[i] = terms["mean_bin_width_hz"]
        sqrt_det[i] = terms["metric_sqrt_det"]
        correlation[i] = terms["metric_correlation"]
        g_logm_logm[i] = terms["g_logm_logm"]
        g_logm_t20[i] = terms["g_logm_t20"]
        g_t20_t20[i] = terms["g_t20_t20"]

    return {
        "mchirp_msun": float(mchirp_msun),
        "duration_20_to_fmax_s": track.duration,
        "t20_min_s": -track.duration,
        "t20_max_s": config.observation_seconds,
        "t20_range_s": config.observation_seconds + track.duration,
        "mean_observed_duration_s": float(np.mean(observed_duration)),
        "mean_stack_samples": float(np.mean(n_stack_samples)),
        "mean_bin_width_hz": finite_mean(mean_bin_width),
        "max_metric_sqrt_det": float(np.max(sqrt_det)),
        "median_metric_correlation": finite_median(correlation),
        "t20_metric_volume": float(np.trapezoid(sqrt_det, t20_values)),
        "mean_g_logm_logm": float(np.mean(g_logm_logm)),
        "mean_g_logm_t20": float(np.mean(g_logm_t20)),
        "mean_g_t20_t20": float(np.mean(g_t20_t20)),
    }


def geometric_metric_summary(
    config: BankConfig,
    max_mismatch: float,
    response: str,
    n_mchirp_scan: int,
    n_t20_scan: int,
    weight_model: str = DEFAULT_WEIGHT_MODEL,
) -> dict[str, np.ndarray | float | str]:
    """Return metric-volume and count estimates over the requested parameter space."""
    if max_mismatch <= 0.0:
        raise ValueError("max_mismatch must be positive")
    if weight_model != "uniform":
        raise ValueError(f"unknown weight model: {weight_model}")

    kappa = response_curvature(response)
    mchirp_values = np.geomspace(
        config.mchirp_min, config.mchirp_max, n_mchirp_scan
    )
    log_mchirp = np.log(mchirp_values)

    rows = [
        metric_volume_by_mchirp(
            float(mchirp),
            config,
            kappa,
            n_t20_scan,
            weight_model,
        )
        for mchirp in mchirp_values
    ]

    t20_metric_volume = np.asarray([row["t20_metric_volume"] for row in rows])
    metric_volume = float(np.trapezoid(t20_metric_volume, log_mchirp))

    square_cell_area = 2.0 * max_mismatch
    hex_cell_area = 3.0 * np.sqrt(3.0) * max_mismatch / 2.0

    return {
        "mchirp_msun": mchirp_values,
        "log_mchirp": log_mchirp,
        "duration_20_to_fmax_s": np.asarray(
            [row["duration_20_to_fmax_s"] for row in rows]
        ),
        "t20_min_s": np.asarray([row["t20_min_s"] for row in rows]),
        "t20_max_s": np.asarray([row["t20_max_s"] for row in rows]),
        "t20_range_s": np.asarray([row["t20_range_s"] for row in rows]),
        "mean_observed_duration_s": np.asarray(
            [row["mean_observed_duration_s"] for row in rows]
        ),
        "mean_stack_samples": np.asarray([row["mean_stack_samples"] for row in rows]),
        "mean_bin_width_hz": np.asarray([row["mean_bin_width_hz"] for row in rows]),
        "max_metric_sqrt_det": np.asarray(
            [row["max_metric_sqrt_det"] for row in rows]
        ),
        "median_metric_correlation": np.asarray(
            [row["median_metric_correlation"] for row in rows]
        ),
        "t20_metric_volume": t20_metric_volume,
        "mean_g_logm_logm": np.asarray([row["mean_g_logm_logm"] for row in rows]),
        "mean_g_logm_t20": np.asarray([row["mean_g_logm_t20"] for row in rows]),
        "mean_g_t20_t20": np.asarray([row["mean_g_t20_t20"] for row in rows]),
        "metric_volume": metric_volume,
        "square_count_estimate": metric_volume / square_cell_area,
        "hex_count_estimate": metric_volume / hex_cell_area,
        "max_mismatch": max_mismatch,
        "response": response,
        "response_curvature": kappa,
        "weight_model": weight_model,
        "n_mchirp_scan": n_mchirp_scan,
        "n_t20_scan": n_t20_scan,
    }


def write_summary_csv(path: Path, summary: dict[str, np.ndarray | float | str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mchirp = np.asarray(summary["mchirp_msun"])
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "index",
                "mchirp_msun",
                "duration_20_to_fmax_s",
                "t20_range_s",
                "mean_observed_duration_s",
                "mean_stack_samples",
                "mean_bin_width_hz",
                "max_metric_sqrt_det",
                "median_metric_correlation",
                "t20_metric_volume",
                "mean_g_logm_logm",
                "mean_g_logm_t20",
                "mean_g_t20_t20",
            ]
        )
        for i, mc in enumerate(mchirp):
            writer.writerow(
                [
                    i,
                    f"{mc:.16e}",
                    f"{summary['duration_20_to_fmax_s'][i]:.16e}",
                    f"{summary['t20_range_s'][i]:.16e}",
                    f"{summary['mean_observed_duration_s'][i]:.16e}",
                    f"{summary['mean_stack_samples'][i]:.16e}",
                    f"{summary['mean_bin_width_hz'][i]:.16e}",
                    f"{summary['max_metric_sqrt_det'][i]:.16e}",
                    f"{summary['median_metric_correlation'][i]:.16e}",
                    f"{summary['t20_metric_volume'][i]:.16e}",
                    f"{summary['mean_g_logm_logm'][i]:.16e}",
                    f"{summary['mean_g_logm_t20'][i]:.16e}",
                    f"{summary['mean_g_t20_t20'][i]:.16e}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate a geometric semicoherent template-bank count for "
            "TaylorF2 frequency-time tracks."
        )
    )
    parser.add_argument("--mchirp-min", type=float, default=DEFAULT_MCHIRP_MIN)
    parser.add_argument("--mchirp-max", type=float, default=DEFAULT_MCHIRP_MAX)
    parser.add_argument("--f-min", type=float, default=DEFAULT_F_MIN)
    parser.add_argument("--f-max", type=float, default=DEFAULT_F_MAX)
    parser.add_argument("--t-coh", type=float, default=DEFAULT_T_COH)
    parser.add_argument(
        "--observation-days", type=float, default=DEFAULT_OBSERVATION_DAYS
    )
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--max-mismatch", type=float, default=DEFAULT_MAX_MISMATCH)
    parser.add_argument(
        "--response",
        choices=["sinc2", "bin-averaged-sinc2"],
        default=DEFAULT_RESPONSE,
    )
    parser.add_argument(
        "--weight-model",
        choices=["uniform"],
        default=DEFAULT_WEIGHT_MODEL,
        help="Statistic weights used in the stack-slide average.",
    )
    parser.add_argument("--n-mchirp-scan", type=int, default=DEFAULT_N_MCHIRP_SCAN)
    parser.add_argument("--n-t20-scan", type=int, default=DEFAULT_N_T20_SCAN)
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
    if args.max_mismatch <= 0.0:
        raise ValueError("max-mismatch must be positive")
    if args.n_mchirp_scan < 2:
        raise ValueError("n-mchirp-scan must be at least 2")
    if args.n_t20_scan < 3:
        raise ValueError("n-t20-scan must be at least 3")
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
        freq_error_hz=DEFAULT_FREQ_ERROR_HZ,
        n_mchirp_scan=args.n_mchirp_scan,
        n_f0_scan=2,
        n_time_scan=2,
        n_metric_time_scan=2,
        n_taylorf2_grid=args.n_taylorf2_grid,
        finite_diff_dlogm=args.finite_diff_dlogm,
        output_dir=args.output_dir,
    )


def main() -> None:
    args = parse_args()
    config = config_from_args(args)

    summary = geometric_metric_summary(
        config=config,
        max_mismatch=args.max_mismatch,
        response=args.response,
        n_mchirp_scan=args.n_mchirp_scan,
        n_t20_scan=args.n_t20_scan,
        weight_model=args.weight_model,
    )

    csv_path = config.output_dir / "geometric_metric_summary.csv"
    npz_path = config.output_dir / "geometric_metric_summary.npz"
    write_summary_csv(csv_path, summary)
    np.savez(npz_path, **summary)

    print("TaylorF2 semicoherent geometric metric")
    print(f"  Mc range: {config.mchirp_min:.1e}--{config.mchirp_max:.1e} Msun")
    print(f"  f range: {config.f_min:.6g}--{config.f_max:.6g} Hz")
    print(f"  observation: {config.observation_days:.6g} days")
    print(f"  max mismatch: {args.max_mismatch:.6g}")
    print(f"  response: {args.response}")
    print(f"  weight model: {args.weight_model}")
    print(f"  response curvature: {float(summary['response_curvature']):.6e}")
    print("  metric bin widths: per-chunk 1/tau_0PN(T_coh, beta)")
    print(f"  metric volume: {float(summary['metric_volume']):.6e}")
    print(f"  square count estimate: {float(summary['square_count_estimate']):.6e}")
    print(f"  hex/A2 count estimate: {float(summary['hex_count_estimate']):.6e}")
    print(f"  wrote {csv_path}")
    print(f"  wrote {npz_path}")


if __name__ == "__main__":
    main()
