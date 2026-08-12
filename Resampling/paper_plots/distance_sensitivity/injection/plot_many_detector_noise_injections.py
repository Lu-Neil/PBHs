"""Compare repeated detector-noise injections with the expected statistic."""

import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from scipy.stats import chi2

PAPER_PLOTS_DIR = Path(__file__).resolve().parents[2]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402
from distance_sensitivity.maximum_sensitivity_35PN import (  # noqa: E402
    FOURIER_BIN_POWER_AVERAGE,
)


INPUT_PATH = Path(__file__).resolve().parent / "data" / "detector_noise_injections.h5"
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "many_detector_noise_injections.png"
SKY_AVERAGED_RESPONSE_POWER = 1 / 5
FALSE_ALARM_PROBABILITY = 1e-6


def expected_statistic(distances, effective_psds, background_stds, attrs):
    """Expected weighted statistic, averaged over sky position and polarization."""
    sample_rate = attrs["sample_rate_hz"]
    segment_duration = attrs["coherent_duration_s"]
    n_segments = attrs["coherent_segments"]
    f0 = attrs["initial_frequency_hz"]
    f_max = attrs["maximum_frequency_hz"]
    mchirp = attrs["chirp_mass_msun"]

    duration = n_segments * segment_duration
    t = base.sample_times(sample_rate, duration)
    frequency_model = base.TaylorF2FrequencyModel(
        f0_hz=f0,
        f_stop_hz=f_max,
        mchirp_msun=mchirp,
    )
    frequency_track = np.asarray(frequency_model.frequency(t))
    chunk_samples = round(sample_rate * segment_duration)
    window = np.hanning(chunk_samples)
    window_power = np.mean(window**2)

    coherent_amplitudes = []
    for start in np.arange(n_segments) * chunk_samples:
        amplitude = base.h0_amplitude(
            base.PARSEC_M,
            frequency_track[start : start + chunk_samples],
            mchirp,
        )
        coherent_amplitudes.append(np.mean(window * amplitude))

    signal_power = (
        SKY_AVERAGED_RESPONSE_POWER
        * FOURIER_BIN_POWER_AVERAGE
        * np.square(coherent_amplitudes)
        / window_power
    )
    raw_weights = signal_power / effective_psds
    weights = raw_weights / np.linalg.norm(raw_weights, axis=1)[:, None]
    noncentralities = segment_duration * raw_weights
    retention = (1 - attrs["coherent_pn_mismatch"]) * (
        1 - attrs["semicoherent_bank_mismatch"]
    )
    one_parsec = retention * np.sum(
        weights * noncentralities / background_stds, axis=1
    )
    return np.mean(one_parsec) / distances**2


def main():
    with h5py.File(INPUT_PATH) as data:
        completed = int(data.attrs["completed_injections"])
        distances = data["distances_pc"][:]
        n_segments = int(data.attrs["coherent_segments"])
        recovered = data["recovered_statistics"][:completed]
        expected = expected_statistic(
            distances,
            data["effective_psds"][:completed],
            data["background_power_standard_deviations"][:completed],
            data.attrs,
        )

    threshold = (
        chi2.isf(FALSE_ALARM_PROBABILITY, 2 * n_segments) - 2 * n_segments
    ) / (2 * np.sqrt(n_segments))
    median = np.median(recovered, axis=0)
    low, high = np.percentile(recovered, [5, 95], axis=0)

    plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig = plt.figure(figsize=(7.2, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1, axes_class=Axes)
    recovered_handle = ax.errorbar(
        distances,
        median,
        yerr=(median - low, high - median),
        fmt="o",
        ms=4,
        capsize=2,
        label="Recovered",
    )
    theoretical_handle, = ax.plot(distances, expected, label="Theoretical")
    # threshold_handle = ax.axhline(
    #     threshold,
    #     color="0.5",
    #     ls="--",
    #     label=r"$n_\sigma^{\rm thresh}$ (${\rm FAP}=10^{-6}$)",
    # )
    y_min = expected.min()
    y_max = max(high.max(), expected.max(), threshold)
    ax.set(
        xlabel="Injection distance [pc]",
        ylabel=r"$n_\sigma$",
        xscale="log",
        yscale="log",
        xlim=(distances.min() / 1.2, 3e7),
    )
    ax.set_ylim(1e-2, y_max*1.2)
    ax.grid(alpha=0.25)
    ax.legend(
        handles=[theoretical_handle, recovered_handle]
    )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)
    print(f"Plotted {completed} injections from {INPUT_PATH}")
    print(f"Saved {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
