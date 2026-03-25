"""Plot carrier amplitude and phase errors versus bin offset for uniform PBH injections."""

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as pl
import numpy as np

from fiveVec_resampler_utils import (
    _build_gap_mask,
    _create_PBH_signal,
    _estimator,
    _expected_carrier_response,
    _resample_and_extract_5vec,
    _time_domain_5vec,
    _wrapped_phase_diff,
)


N_TRIALS = 200
GAP_FRACTION = 0.15
OUTPUT_PATH = Path(__file__).resolve().parent / "plots" / "bin_dephasing_for_resampler_5vec.png"


def run_uniform_pbh_trials(n_trials=N_TRIALS, gap_fraction=GAP_FRACTION):
    delta_omegas = []
    amp_errors = []
    phase_errors = []

    for _ in range(n_trials):
        signal, tau, omega0, sidereal, h0, gamma, t = _create_PBH_signal(
            f0_setting="uniform",
            delta_beta=0,
        )
        gap_mask, _ = _build_gap_mask(signal.size, gap_fraction=gap_fraction)
        signal = np.where(gap_mask, signal, 0.0)

        data_X, resampler = _resample_and_extract_5vec(signal, tau, omega0)
        template_X, _, _ = _time_domain_5vec(sidereal, t, tau, gap_mask=gap_mask)
        h_est = _estimator(data_X, template_X)

        expected_h, _, delta_omega = _expected_carrier_response(
            resampler,
            omega0,
            tau,
            h0,
            gamma,
            gap_mask=gap_mask,
        )

        delta_omegas.append(delta_omega / np.diff(resampler.freqs)[0])
        amp_errors.append(np.abs(h_est) / np.abs(expected_h) - 1.0)
        phase_errors.append(_wrapped_phase_diff(np.angle(h_est), np.angle(expected_h)))

    return np.array(delta_omegas), np.array(amp_errors), np.array(phase_errors)


def make_plot(delta_omegas, amp_errors, phase_errors):
    order = np.argsort(delta_omegas)
    delta_omegas = delta_omegas[order]
    amp_errors = amp_errors[order]
    phase_errors = phase_errors[order]

    fig, axes = pl.subplots(2, 1, figsize=(8, 7), sharex=True, constrained_layout=True)

    axes[0].plot(delta_omegas, amp_errors, "o", ms=4)
    axes[0].axhline(0.0, color="0.5", lw=1, ls="--")
    axes[0].set_ylabel(r"Amplitude Error: $|h_{\rm est}|/|h_{\rm exp}| - 1$")
    axes[0].set_title(
        f"Uniform PBH carrier errors vs bin offset ({len(delta_omegas)} trials, gap fraction = {GAP_FRACTION:.2f})"
    )

    axes[1].plot(delta_omegas, phase_errors, "o", ms=4)
    axes[1].axhline(0.0, color="0.5", lw=1, ls="--")
    axes[1].set_xlabel(r"Bin offset")
    axes[1].set_ylabel(r"Phase Error [rad]")

    return fig


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    delta_omegas, amp_errors, phase_errors = run_uniform_pbh_trials()
    fig = make_plot(delta_omegas, amp_errors, phase_errors)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"Saved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
