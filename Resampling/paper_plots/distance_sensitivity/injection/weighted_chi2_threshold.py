"""Compare weighted and equal-weight stack-slide false-alarm thresholds."""

from pathlib import Path
import sys

import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import chi2


SCRIPT_DIR = Path(__file__).resolve().parent
FIGS_DIR = SCRIPT_DIR / "figs"
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(PAPER_PLOTS_DIR))

from five_vec import five_vec  # noqa: E402
from signal_generators import (  # noqa: E402
    NoiseCurve,
    TaylorF2FrequencyModel,
    beta_0pn,
)


F_START = 40.0
F_STOP = 64.0
MCHIRPS = (1.0e-1, 1.0e-2)
T_COH = 30.0
FALSE_ALARM_PROBABILITY = 1.0e-6
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = FIGS_DIR / "weighted_chi2_threshold.png"
BACKGROUND_MEAN = 2.0
BACKGROUND_STANDARD_DEVIATION = 2.0  # chi2_2 has variance 4

# Use the same source, detector, polarization, and epoch as the injection scripts.
RA = np.deg2rad(266.416833)
DEC = np.deg2rad(-29.007806)
LAT = np.deg2rad(30.562894333574896)
LNG = np.deg2rad(269.2257596112789)
AZ = np.deg2rad(72.28350084422942)
ETA = 0.0
PSI = 1.0
EPOCH = (2023, 5, 24, 0, 0, 0, 0, 0, 0)


def antenna_power(times):
    """Detector antenna power at seconds after EPOCH."""
    julian_date = lal.ConvertCivilTimeToJD(EPOCH) + times / 86400.0
    response = five_vec(
        ra=RA, dec=DEC, eta=ETA, psi=PSI, lat=LAT, lng=LNG, az=AZ
    )
    response.compute_H()
    response.compute_A(response.gmst(julian_date))
    return np.abs(response.amp_modulation) ** 2


def stack_slide_weights(mchirp):
    """Return A_i^2/S_i_eff for every complete 30 s segment in band."""
    track = TaylorF2FrequencyModel(F_START, F_STOP, mchirp)
    noise = NoiseCurve.from_asd_file(ASD_PATH)

    n_segments = int(track.t_end // T_COH)
    starts = T_COH * np.arange(n_segments)
    offsets = np.linspace(0.0, T_COH, 32)
    times = starts[:, None] + offsets
    frequencies = track.frequency(times)

    # A^2 includes the leading inspiral amplitude f^(2/3) and antenna response.
    amplitude_squared = np.mean(
        frequencies ** (4.0 / 3.0) * antenna_power(times.ravel()).reshape(times.shape),
        axis=1,
    )

    # Window-free NUFFT PSD: average the detector PSD along each local 0PN track.
    f_segment = frequencies[:, 0]
    beta = beta_0pn(f_segment, mchirp)
    tau_dot = (1.0 - (8.0 / 3.0) * beta[:, None] * offsets) ** (-3.0 / 8.0)
    effective_psd = np.mean(noise.psd_at(f_segment[:, None] * tau_dot), axis=1)

    weights = amplitude_squared / effective_psd
    return weights / np.sqrt(np.sum(weights**2))


def weighted_chi2_survival(value, weights):
    """Survival function of sum_i weights[i] * chi2_2 via Fourier inversion."""
    mean = 2.0 * np.sum(weights)

    def integrand(t):
        if t == 0.0:
            return mean - value
        log_characteristic_function = -np.sum(np.log(1.0 - 2.0j * t * weights))
        return np.imag(np.exp(log_characteristic_function - 1.0j * t * value)) / t

    integral, _ = quad(
        integrand,
        0.0,
        np.inf,
        epsabs=1.0e-12,
        epsrel=1.0e-10,
        limit=500,
    )
    return 0.5 + integral / np.pi


def weighted_threshold(weights, survival_probability):
    mean = 2.0 * np.sum(weights)
    upper = mean + 20.0
    while weighted_chi2_survival(upper, weights) > survival_probability:
        upper += 20.0
    return brentq(
        lambda value: weighted_chi2_survival(value, weights)
        - survival_probability,
        mean,
        upper,
        xtol=1.0e-9,
    )


def n_sigma_thresholds(weights, survival_probabilities):
    """Weighted and equal-weight thresholds on the standardized statistic."""
    n_segments = weights.size
    weighted_raw = np.array(
        [weighted_threshold(weights, probability) for probability in survival_probabilities]
    )
    weighted = (
        weighted_raw - BACKGROUND_MEAN * np.sum(weights)
    ) / BACKGROUND_STANDARD_DEVIATION

    equal_weight_raw = chi2.isf(
        survival_probabilities, 2 * n_segments
    ) / np.sqrt(n_segments)
    equal_weight = (
        equal_weight_raw - BACKGROUND_MEAN * np.sqrt(n_segments)
    ) / BACKGROUND_STANDARD_DEVIATION
    return weighted, equal_weight


def main():
    false_alarm_probabilities = np.logspace(-2.0, -8.0, 31)
    plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)

    print("Weighted versus equal-weight stack-slide thresholds")
    print(f"  TaylorF2 band: {F_START:g}-{F_STOP:g} Hz")
    print(f"  false-alarm probability: {FALSE_ALARM_PROBABILITY:g}")

    reference_index = np.argmin(
        np.abs(false_alarm_probabilities - FALSE_ALARM_PROBABILITY)
    )
    for color_index, mchirp in enumerate(MCHIRPS):
        weights = stack_slide_weights(mchirp)
        weighted_thresholds, equal_weight_thresholds = n_sigma_thresholds(
            weights, false_alarm_probabilities
        )
        weighted_n_sigma = weighted_thresholds[reference_index]
        equal_weight_n_sigma = equal_weight_thresholds[reference_index]
        difference = weighted_n_sigma - equal_weight_n_sigma
        relative_difference = difference / equal_weight_n_sigma

        color = f"C{color_index}"
        mass_label = rf"$M_c=10^{{{np.log10(mchirp):.0f}}}\,M_\odot$"
        ax.plot(
            false_alarm_probabilities,
            weighted_thresholds,
            color=color,
            label=mass_label + ", weighted",
        )
        ax.plot(
            false_alarm_probabilities,
            equal_weight_thresholds,
            color=color,
            linestyle="--",
            label=mass_label + r", equal weights",
        )

        print(f"  Mc = {mchirp:g} Msun")
        print(f"    segments: {weights.size} x {T_COH:g} s")
        print(f"    effective segments, (sum w)^2: {np.sum(weights) ** 2:.2f}")
        print(f"    weighted threshold: {weighted_n_sigma:.6f}")
        print(f"    equal-weight threshold: {equal_weight_n_sigma:.6f}")
        print(f"    change: {difference:+.6f} ({relative_difference:+.2%})")

    ax.set_xscale("log")
    ax.set_xlabel("False-alarm probability")
    ax.set_ylabel(r"$n_\sigma^{\rm thresh}$")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=220)
    plt.close(fig)
    print(f"  plot: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
