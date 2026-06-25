"""TaylorF2 semicoherent template-count estimate.

This is a deliberately small script for estimating the number of templates in
the two-dimensional bank (Mc, t20), where t20 is the time at which the TaylorF2
track crosses the reference frequency f0.  The name t20 is kept for consistency
with the notes; f0 is 20 Hz by default but can be changed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    from Resampling.paper_plots.signal_generators import (
        DEFAULT_ETA,
        NoiseCurve,
        TaylorF2FrequencyModel,
        beta_0pn,
    )
except ModuleNotFoundError:
    REPO_ROOT = Path(__file__).resolve().parents[4]
    sys.path.insert(0, str(REPO_ROOT))
    from Resampling.paper_plots.signal_generators import (
        DEFAULT_ETA,
        NoiseCurve,
        TaylorF2FrequencyModel,
        beta_0pn,
    )


DEFAULT_MCHIRP_MIN = 5.0e-4
DEFAULT_MCHIRP_MAX = 1.0e-1
DEFAULT_F0 = 40.0
DEFAULT_F_STOP = 64.0
DEFAULT_T_OBS = 365.25 * 86400.0
DEFAULT_T_COH = 30.0
DEFAULT_MAX_MISMATCH = 0.1
DEFAULT_N_MCHIRP = 32
DEFAULT_N_T20 = 48
DEFAULT_N_TAYLORF2_GRID = 2048
DEFAULT_FINITE_DIFF_FRACTION = 1.0e-4
N_CHUNK_WEIGHT_SAMPLES = 8
DEFAULT_ASD_PATH = Path(__file__).resolve().parents[2] / "asd.txt"

HEXAGONAL_COVERING_THETA = 2.0 / (3.0 * np.sqrt(3.0))


def _track(mchirp_msun, f0, f_stop, eta, n_taylorf2_grid):
    return TaylorF2FrequencyModel(
        f0_hz=f0,
        f_stop_hz=f_stop,
        mchirp_msun=mchirp_msun,
        eta=eta,
        n_grid=n_taylorf2_grid,
    )


def _frequency_derivative(track):
    return np.gradient(track.frequency_grid, track.elapsed_time_grid)


def _chunk_elapsed_times(t20, track_duration, T_obs, t_coh):
    n_chunk = int(np.floor(T_obs / t_coh))
    if n_chunk <= 0:
        return np.asarray([], dtype=float)

    centers = (np.arange(n_chunk, dtype=float) + 0.5) * t_coh
    elapsed = centers - t20
    return elapsed[(elapsed >= 0.0) & (elapsed <= track_duration)]


def _chunk_sample_frequencies(track, elapsed, t_coh):
    half_chunk = 0.5 * t_coh
    offsets = np.linspace(-half_chunk, half_chunk, N_CHUNK_WEIGHT_SAMPLES)
    sample_elapsed = elapsed[:, None] + offsets[None, :]
    sample_elapsed = np.clip(sample_elapsed, 0.0, track.t_end)
    return track.frequency(sample_elapsed)


def _tau_bin_width_hz(mchirp_msun, frequency, t_coh):
    beta = beta_0pn(frequency, mchirp_msun)
    bracket = 1.0 - (8.0 / 3.0) * beta * t_coh
    tau_span = np.full_like(frequency, np.nan, dtype=float)
    valid = bracket > 0.0
    tau_span[valid] = (
        3.0
        / (5.0 * beta[valid])
        * (1.0 - bracket[valid] ** (5.0 / 8.0))
    )
    return 1.0 / tau_span


def _metric_weights(mchirp_msun, sample_frequency, noise_curve):
    """Return fixed local weights for the fractional loss of sum w_i P_i."""
    h0 = (mchirp_msun / 1.0e-3) ** (5.0 / 3.0)
    h0 *= (sample_frequency / 50.0) ** (2.0 / 3.0)
    amplitude_squared = np.mean(h0**2, axis=1)
    effective_psd = np.mean(noise_curve.psd_at(sample_frequency), axis=1)
    # The statistic weight is A_i^2 / S_eff_i^2, and the on-template raw
    # signal power contributes another A_i^2 to the fractional-loss metric.
    return amplitude_squared**2 / effective_psd**2


def _metric_sqrt_det(
    mchirp_msun,
    t20,
    *,
    f0,
    f_stop,
    T_obs,
    t_coh,
    eta,
    n_taylorf2_grid,
    finite_diff_fraction,
    noise_curve,
):
    track = _track(mchirp_msun, f0, f_stop, eta, n_taylorf2_grid)
    elapsed = _chunk_elapsed_times(t20, track.t_end, T_obs, t_coh)
    if elapsed.size == 0:
        return 0.0

    dm = finite_diff_fraction * mchirp_msun
    low_track = _track(mchirp_msun - dm, f0, f_stop, eta, n_taylorf2_grid)
    high_track = _track(mchirp_msun + dm, f0, f_stop, eta, n_taylorf2_grid)
    max_elapsed = min(track.t_end, low_track.t_end, high_track.t_end)
    elapsed = elapsed[elapsed <= max_elapsed]
    if elapsed.size == 0:
        return 0.0

    df_dmc = (
        high_track.frequency(elapsed) - low_track.frequency(elapsed)
    ) / (2.0 * dm)
    df_dt = np.interp(
        elapsed,
        track.elapsed_time_grid,
        _frequency_derivative(track),
    )

    bin_width_hz = _tau_bin_width_hz(mchirp_msun, track.frequency(elapsed), t_coh)
    valid_bin_width = np.isfinite(bin_width_hz) & (bin_width_hz > 0.0)
    if not np.any(valid_bin_width):
        return 0.0

    elapsed = elapsed[valid_bin_width]
    df_dmc = df_dmc[valid_bin_width]
    df_dt = df_dt[valid_bin_width]
    bin_width_hz = bin_width_hz[valid_bin_width]

    dmc_bins = np.asarray(df_dmc, dtype=float) / bin_width_hz
    dt20_bins = -np.asarray(df_dt, dtype=float) / bin_width_hz
    sample_frequency = _chunk_sample_frequencies(track, elapsed, t_coh)
    weights = _metric_weights(mchirp_msun, sample_frequency, noise_curve)

    finite = (
        np.isfinite(dmc_bins)
        & np.isfinite(dt20_bins)
        & np.isfinite(weights)
        & (weights > 0.0)
    )
    if not np.any(finite):
        return 0.0

    dmc_bins = dmc_bins[finite]
    dt20_bins = dt20_bins[finite]
    weights = weights[finite]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        return 0.0

    # Kappa is intentionally omitted: mismatch ~= <delta_bin**2>.  The weights
    # are held fixed locally, so derivatives of w_i(theta) are not included.
    g_mcmc = float(np.sum(weights * dmc_bins**2) / weight_sum)
    g_mct20 = float(np.sum(weights * dmc_bins * dt20_bins) / weight_sum)
    g_t20t20 = float(np.sum(weights * dt20_bins**2) / weight_sum)
    det_g = max(0.0, g_mcmc * g_t20t20 - g_mct20**2)
    return float(np.sqrt(det_g))


def _metric_volume_at_mchirp(
    mchirp_msun,
    *,
    f0,
    f_stop,
    T_obs,
    t_coh,
    eta,
    n_t20,
    n_taylorf2_grid,
    finite_diff_fraction,
    noise_curve,
):
    track = _track(mchirp_msun, f0, f_stop, eta, n_taylorf2_grid)
    t20_values = np.linspace(-track.t_end, T_obs, n_t20)
    sqrt_det = np.array(
        [
            _metric_sqrt_det(
                mchirp_msun,
                t20,
                f0=f0,
                f_stop=f_stop,
                T_obs=T_obs,
                t_coh=t_coh,
                eta=eta,
                n_taylorf2_grid=n_taylorf2_grid,
                finite_diff_fraction=finite_diff_fraction,
                noise_curve=noise_curve,
            )
            for t20 in t20_values
        ],
        dtype=float,
    )
    return float(np.trapezoid(sqrt_det, t20_values)), float(track.t_end)


def template_number(
    *,
    mchirp_min=DEFAULT_MCHIRP_MIN,
    mchirp_max=DEFAULT_MCHIRP_MAX,
    f0=DEFAULT_F0,
    T_obs=DEFAULT_T_OBS,
    max_mismatch=DEFAULT_MAX_MISMATCH,
    f_stop=DEFAULT_F_STOP,
    t_coh=DEFAULT_T_COH,
    eta=DEFAULT_ETA,
    n_mchirp=DEFAULT_N_MCHIRP,
    n_t20=DEFAULT_N_T20,
    n_taylorf2_grid=DEFAULT_N_TAYLORF2_GRID,
    finite_diff_fraction=DEFAULT_FINITE_DIFF_FRACTION,
):
    """Return the hexagonal-covering template-count estimate.

    The estimate is

        N ~= theta * mu_max**(-n/2) * integral sqrt(det g) dMc dt20

    with n=2 and theta set to the hexagonal-covering value.
    """

    if not 0.0 < mchirp_min < mchirp_max:
        raise ValueError("Require 0 < mchirp_min < mchirp_max.")
    if not 0.0 < f0 < f_stop:
        raise ValueError("Require 0 < f0 < f_stop.")
    if T_obs <= 0.0 or t_coh <= 0.0:
        raise ValueError("T_obs and t_coh must be positive.")
    if max_mismatch <= 0.0:
        raise ValueError("max_mismatch must be positive.")
    if n_mchirp < 2 or n_t20 < 2:
        raise ValueError("n_mchirp and n_t20 must be at least 2.")

    noise_curve = NoiseCurve.from_asd_file(DEFAULT_ASD_PATH)
    if f0 < noise_curve.frequency[0] or f_stop > noise_curve.frequency[-1]:
        raise ValueError(
            "The requested frequency band is outside "
            f"{DEFAULT_ASD_PATH} "
            f"[{noise_curve.frequency[0]:.6g}, {noise_curve.frequency[-1]:.6g}] Hz."
        )

    mchirp_values = np.geomspace(mchirp_min, mchirp_max, n_mchirp)
    volume_by_mchirp = np.empty_like(mchirp_values)
    track_duration = np.empty_like(mchirp_values)

    for i, mchirp in enumerate(mchirp_values):
        volume_by_mchirp[i], track_duration[i] = _metric_volume_at_mchirp(
            float(mchirp),
            f0=f0,
            f_stop=f_stop,
            T_obs=T_obs,
            t_coh=t_coh,
            eta=eta,
            n_t20=n_t20,
            n_taylorf2_grid=n_taylorf2_grid,
            finite_diff_fraction=finite_diff_fraction,
            noise_curve=noise_curve,
        )

    metric_volume = float(np.trapezoid(volume_by_mchirp, mchirp_values))
    n_dim = 2
    count = HEXAGONAL_COVERING_THETA * max_mismatch ** (-n_dim / 2.0)
    count *= metric_volume

    return {
        "template_count": float(count),
        "metric_volume": metric_volume,
        "covering": "hexagonal",
        "covering_theta": float(HEXAGONAL_COVERING_THETA),
        "max_mismatch": float(max_mismatch),
        "mchirp_msun": mchirp_values,
        "t20_metric_volume": volume_by_mchirp,
        "track_duration_s": track_duration,
        "f0_hz": float(f0),
        "f_stop_hz": float(f_stop),
        "T_obs_s": float(T_obs),
        "t_coh_s": float(t_coh),
        "asd_path": str(DEFAULT_ASD_PATH),
        "weight_model": "fractional-loss weights A_i^4 / S_eff^2, with Gamma_i = 1",
    }


def main():
    parser = argparse.ArgumentParser(
        description="Estimate a TaylorF2 semicoherent (Mc, t20) template count."
    )
    parser.add_argument("--mchirp-min", type=float, default=DEFAULT_MCHIRP_MIN)
    parser.add_argument("--mchirp-max", type=float, default=DEFAULT_MCHIRP_MAX)
    parser.add_argument("--f0", type=float, default=DEFAULT_F0)
    parser.add_argument("--f-stop", type=float, default=DEFAULT_F_STOP)
    parser.add_argument(
        "--T-obs", "--t-obs", dest="T_obs", type=float, default=DEFAULT_T_OBS
    )
    parser.add_argument("--t-coh", type=float, default=DEFAULT_T_COH)
    parser.add_argument("--max-mismatch", type=float, default=DEFAULT_MAX_MISMATCH)
    parser.add_argument("--eta", type=float, default=DEFAULT_ETA)
    parser.add_argument("--n-mchirp", type=int, default=DEFAULT_N_MCHIRP)
    parser.add_argument("--n-t20", type=int, default=DEFAULT_N_T20)
    parser.add_argument("--n-taylorf2-grid", type=int, default=DEFAULT_N_TAYLORF2_GRID)
    parser.add_argument(
        "--finite-diff-fraction",
        type=float,
        default=DEFAULT_FINITE_DIFF_FRACTION,
    )
    args = parser.parse_args()

    result = template_number(**vars(args))
    print("TaylorF2 semicoherent template-count estimate")
    print(f"  covering: {result['covering']}")
    print(f"  theta: {result['covering_theta']:.6e}")
    print(f"  weights: {result['weight_model']}")
    print(f"  ASD: {result['asd_path']}")
    print(f"  metric volume: {result['metric_volume']:.6e}")
    print(f"  template count: {result['template_count']:.6e}")


if __name__ == "__main__":
    main()
