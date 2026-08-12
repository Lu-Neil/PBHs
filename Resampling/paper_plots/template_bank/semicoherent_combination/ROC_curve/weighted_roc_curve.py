"""Compare unweighted, known-sky, and sky-averaged semicoherent ROC curves.

The injections are made directly in PSD-normalized coherent power.  For each
30 s segment, the recovered power is distributed as ``chi2_2(lambda_i)``,
where ``lambda_i`` is calculated from the physical inspiral amplitude, the
LLO antenna response, and the effective NUFFT PSD.  This is the statistical
equivalent of injecting the corresponding signal into Gaussian detector
noise and reading the perfectly matched track bin.

The unweighted true-positive probability is evaluated exactly with the
noncentral-chi-square survival function.  Two weighted curves are shown.  The
known-sky curve uses the locally optimal weights ``w_i proportional to
lambda_i``.  The sky-averaged curve averages the antenna power over an
isotropic sky before constructing the weights; that average is independent of
sidereal time, so these weights are proportional to the intrinsic
``A_i^2/S_i_eff``.  For the very large number of segments at the configured
chirp mass, both weighted distributions are evaluated with central-limit
approximations from their exact means and variances.  The sidereal response is
compressed into a small Fourier basis, avoiding an ``n_sky x n_segments``
array.
"""

import argparse
from pathlib import Path
import sys

import lal
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import chi2, ncx2, norm


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[2]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from signal_generators import (  # noqa: E402
    C,
    NoiseCurve,
    TaylorF2FrequencyModel,
    beta_0pn,
)


F_START = 40.0
F_STOP = 60.0
MCHIRP = 1.0e-3
T_COH = 30.0
ANDROMEDA_PC = 7.65e5
DISTANCE_PC = 5.0e4
DISTANCE_IN_ANDROMEDA = DISTANCE_PC / ANDROMEDA_PC
DISTANCE_M = DISTANCE_PC * lal.PC_SI

# Match the injection geometry used elsewhere in paper_plots.
LLO_LAT = np.deg2rad(30.562894333574896)
LLO_LNG = np.deg2rad(269.2257596112789)
LLO_AZ = np.deg2rad(72.28350084422942)
POLARIZATION_RATIO = 0.0
POLARIZATION_ANGLE = 1.0
INJECTION_EPOCH_TM_UTC = (2023, 5, 24, 0, 0, 0, 0, 0, 0)
INJECTION_EPOCH_JD = lal.ConvertCivilTimeToJD(INJECTION_EPOCH_TM_UTC)

FOURIER_BIN_POWER_AVERAGE = 2.4308 / np.pi
ASD_PATH = PAPER_PLOTS_DIR / "asd.txt"
OUTPUT_PATH = SCRIPT_DIR / "weighted_vs_equal_roc.png"
DATA_PATH = SCRIPT_DIR / "data" / "weighted_vs_equal_roc.npz"
DEFAULT_N_INJECTIONS = 200_000
DEFAULT_BATCH_SIZE = 25_000
DEFAULT_SEED = 190521
SIDEREAL_GRID_SIZE = 32
NONCENTRALITY_QUANTILE_BINS = 2048


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Make equal- and unequal-weight ROC curves for random-sky "
            "Andromeda-scaled injections."
        )
    )
    parser.add_argument("--n-injections", type=int, default=DEFAULT_N_INJECTIONS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--asd", type=Path, default=ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--data-output", type=Path, default=DATA_PATH)
    return parser.parse_args()


def validate_args(args):
    if args.n_injections < 10_000:
        raise ValueError("n-injections must be at least 10000")
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    if args.batch_size > args.n_injections:
        raise ValueError("batch-size must not exceed n-injections")


def injection_gmst(times):
    """Greenwich mean sidereal time at each elapsed injection time."""
    jd = INJECTION_EPOCH_JD + np.asarray(times, dtype=float) / 86400.0
    jd0 = np.floor(jd - 0.5) + 0.5
    hours = (jd - jd0) * 24.0
    days = jd - 2451545.0
    days0 = jd0 - 2451545.0
    centuries = days / 36525.0
    sidereal_hours = np.mod(
        6.697374558
        + 0.06570982441908 * days0
        + 1.00273790935 * hours
        + 0.000026 * centuries**2,
        24.0,
    )
    return sidereal_hours * np.pi / 12.0


def antenna_amplitude_squared(ra, dec, gmst):
    """Return the squared complex LLO response for a batch of sky positions."""
    ra = np.asarray(ra, dtype=float)[:, None]
    dec = np.asarray(dec, dtype=float)[:, None]
    gmst = np.asarray(gmst, dtype=float)[None, :]

    c_dec = np.cos(dec)
    s_dec = np.sin(dec)
    c_2dec = np.cos(2.0 * dec)
    s_2dec = np.sin(2.0 * dec)
    c_lat = np.cos(LLO_LAT)
    s_lat = np.sin(LLO_LAT)
    c_2lat = np.cos(2.0 * LLO_LAT)
    s_2lat = np.sin(2.0 * LLO_LAT)
    c_2az = np.cos(2.0 * LLO_AZ)
    s_2az = np.sin(2.0 * LLO_AZ)

    a0 = -(3.0 / 16.0) * (1.0 + c_2dec) * (1.0 + c_2lat) * c_2az
    a1c = -(1.0 / 4.0) * s_2dec * s_2lat * c_2az
    a1s = -(1.0 / 2.0) * s_2dec * c_lat * s_2az
    a2c = -(1.0 / 16.0) * (3.0 - c_2dec) * (3.0 - c_2lat) * c_2az
    a2s = -(1.0 / 4.0) * (3.0 - c_2dec) * s_lat * s_2az

    b1c = -c_dec * c_lat * s_2az
    b1s = 0.5 * c_dec * s_2lat * c_2az
    b2c = -s_dec * s_lat * s_2az
    b2s = 0.25 * s_dec * (3.0 - c_2lat) * c_2az

    hour_angle = gmst + LLO_LNG - ra
    a_plus = (
        a0
        + a1c * np.cos(hour_angle)
        + a1s * np.sin(hour_angle)
        + a2c * np.cos(2.0 * hour_angle)
        + a2s * np.sin(2.0 * hour_angle)
    )
    a_cross = (
        b1c * np.cos(hour_angle)
        + b1s * np.sin(hour_angle)
        + b2c * np.cos(2.0 * hour_angle)
        + b2s * np.sin(2.0 * hour_angle)
    )

    eta = POLARIZATION_RATIO
    psi = POLARIZATION_ANGLE
    h_plus = (np.cos(2.0 * psi) - 1.0j * eta * np.sin(2.0 * psi)) / np.sqrt(
        1.0 + eta**2
    )
    h_cross = (np.sin(2.0 * psi) + 1.0j * eta * np.cos(2.0 * psi)) / np.sqrt(
        1.0 + eta**2
    )
    return np.abs(a_plus * h_plus + a_cross * h_cross) ** 2


def segment_model(asd_path):
    """Return chunk times and the sky-independent part of each lambda_i."""
    track = TaylorF2FrequencyModel(F_START, F_STOP, MCHIRP)
    noise = NoiseCurve.from_asd_file(asd_path)
    n_segments = int(track.t_end // T_COH)
    starts = T_COH * np.arange(n_segments)
    offsets = np.linspace(0.0, T_COH, 32)
    times = starts[:, None] + offsets
    frequencies = np.asarray(track.frequency(times), dtype=float)

    amplitude_squared = np.mean((frequencies / F_START) ** (4.0 / 3.0), axis=1)
    f_segment = frequencies[:, 0]
    beta = beta_0pn(f_segment, MCHIRP)
    tau_dot = (1.0 - (8.0 / 3.0) * beta[:, None] * offsets) ** (-3.0 / 8.0)
    effective_psd = np.mean(
        noise.psd_at(f_segment[:, None] * tau_dot),
        axis=1,
    )

    beta_start = beta_0pn(F_START, MCHIRP)
    amplitude_prefactor = (
        FOURIER_BIN_POWER_AVERAGE
        * 16.0
        / np.pi**4
        * (5.0 / 96.0) ** 2
        * (C * beta_start / F_START**2) ** 2
        / DISTANCE_M**2
    )
    intrinsic_noncentrality = (
        amplitude_prefactor * T_COH * amplitude_squared / effective_psd
    )
    mid_times = starts + 0.5 * T_COH
    return track.t_end, mid_times, intrinsic_noncentrality


def sidereal_time_harmonics(mid_times, intrinsic):
    """Compress intrinsic-weighted segment times into sidereal harmonics."""
    gmst = injection_gmst(mid_times)
    harmonics = {}
    for power in (1, 2, 3):
        intrinsic_power = intrinsic**power
        max_harmonic = 4 * power
        harmonics[power] = np.array(
            [
                np.sum(intrinsic_power * np.exp(1.0j * order * gmst))
                for order in range(max_harmonic + 1)
            ]
        )
    return harmonics


def moment_from_response_fft(
    response_squared,
    time_harmonics,
    intrinsic_power,
    response_power,
):
    """Return ``sum_i intrinsic_i**p * response_squared_i**r``."""
    max_harmonic = 4 * response_power
    coefficients = (
        np.fft.fft(response_squared**response_power, axis=1)
        / response_squared.shape[1]
    )
    moments = (
        coefficients[:, 0].real
        * time_harmonics[intrinsic_power][0].real
    )
    moments += 2.0 * np.real(
        np.sum(
            coefficients[:, 1 : max_harmonic + 1]
            * time_harmonics[intrinsic_power][1 : max_harmonic + 1],
            axis=1,
        )
    )
    return moments


def compressed_signal_moments(
    n_injections,
    batch_size,
    seed,
    mid_times,
    intrinsic,
):
    """Return moments needed by the known- and unknown-sky statistics."""
    rng = np.random.default_rng(seed)
    time_harmonics = sidereal_time_harmonics(mid_times, intrinsic)
    sidereal_grid = 2.0 * np.pi * np.arange(SIDEREAL_GRID_SIZE) / SIDEREAL_GRID_SIZE
    known_sky_moments = np.empty((3, n_injections))
    sky_averaged_weight_moments = np.empty((2, n_injections))

    for start in range(0, n_injections, batch_size):
        stop = min(start + batch_size, n_injections)
        size = stop - start
        ra = rng.uniform(0.0, 2.0 * np.pi, size)
        dec = np.arcsin(rng.uniform(-1.0, 1.0, size))
        response_squared = antenna_amplitude_squared(ra, dec, sidereal_grid)
        for power in (1, 2, 3):
            known_sky_moments[power - 1, start:stop] = moment_from_response_fft(
                response_squared,
                time_harmonics,
                power,
                power,
            )
        for power in (2, 3):
            sky_averaged_weight_moments[
                power - 2, start:stop
            ] = moment_from_response_fft(
                response_squared,
                time_harmonics,
                power,
                1,
            )

    return (*known_sky_moments, *sky_averaged_weight_moments)


def sky_averaged_equal_weight_survival(
    thresholds,
    n_segments,
    total_noncentrality,
    n_bins=NONCENTRALITY_QUANTILE_BINS,
):
    """Average the exact ncx2 survival function over compressed sky samples."""
    sorted_noncentrality = np.sort(total_noncentrality)
    n_bins = min(n_bins, sorted_noncentrality.size)
    edges = np.linspace(0, sorted_noncentrality.size, n_bins + 1, dtype=int)
    counts = np.diff(edges)
    representative_noncentrality = np.array(
        [
            np.mean(sorted_noncentrality[edges[index] : edges[index + 1]])
            for index in range(n_bins)
        ]
    )
    probabilities = counts / sorted_noncentrality.size
    dof = 2 * n_segments
    return np.array(
        [
            np.sum(
                probabilities
                * ncx2.sf(threshold, dof, representative_noncentrality)
            )
            for threshold in thresholds
        ]
    )


def make_roc(
    total_noncentrality,
    known_sky_squared_sum,
    known_sky_cubed_sum,
    sky_averaged_mean_numerator,
    sky_averaged_variance_numerator,
    intrinsic_squared_sum,
    n_segments,
):
    """Return unweighted and two weighted random-sky ROCs."""
    interior_false_positive_probability = np.geomspace(1.0e-4, 0.9, 61)
    false_positive_probability = np.concatenate(
        ([0.0], interior_false_positive_probability, [1.0])
    )

    weighted_threshold = norm.isf(false_positive_probability)
    known_sky_signal_mean = 0.5 * np.sqrt(known_sky_squared_sum)
    known_sky_signal_sigma = np.sqrt(
        1.0 + known_sky_cubed_sum / known_sky_squared_sum
    )
    known_sky_true_positive = np.array(
        [
            np.mean(
                norm.sf(
                    (threshold - known_sky_signal_mean)
                    / known_sky_signal_sigma
                )
            )
            for threshold in weighted_threshold
        ]
    )

    sky_averaged_signal_mean = (
        0.5
        * sky_averaged_mean_numerator
        / np.sqrt(intrinsic_squared_sum)
    )
    sky_averaged_signal_sigma = np.sqrt(
        1.0
        + sky_averaged_variance_numerator / intrinsic_squared_sum
    )
    sky_averaged_true_positive = np.array(
        [
            np.mean(
                norm.sf(
                    (threshold - sky_averaged_signal_mean)
                    / sky_averaged_signal_sigma
                )
            )
            for threshold in weighted_threshold
        ]
    )

    equal_raw_threshold = chi2.isf(
        false_positive_probability,
        2 * n_segments,
    )
    equal_true_positive = sky_averaged_equal_weight_survival(
        equal_raw_threshold,
        n_segments,
        total_noncentrality,
    )
    return (
        false_positive_probability,
        equal_true_positive,
        known_sky_true_positive,
        sky_averaged_true_positive,
    )


def plot_roc(
    false_positive,
    equal_true_positive,
    known_sky_true_positive,
    sky_averaged_true_positive,
    output,
):
    plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")
    fig, ax = plt.subplots(figsize=(6.4, 4.6), constrained_layout=True)
    ax.plot(
        false_positive,
        equal_true_positive,
        lw=2.0,
        marker="o",
        ms=3.0,
        label=r"Unweighted",
    )
    ax.plot(
        false_positive,
        sky_averaged_true_positive,
        lw=2.0,
        marker="o",
        ms=3.0,
        label=r"Sky-averaged weights",
    )
    ax.plot(
        false_positive,
        known_sky_true_positive,
        lw=2.0,
        marker="o",
        ms=3.0,
        label=r"Targeted sky-location weights",
    )
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xticks(np.linspace(0.0, 1.0, 6))
    ax.set_yticks(np.linspace(0.0, 1.0, 6))
    ax.set_xlabel("FAP")
    ax.set_ylabel("TPP")
    ax.grid(True, alpha=0.4)
    ax.legend(loc="lower right")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)
    duration, mid_times, intrinsic = segment_model(args.asd)
    (
        total_noncentrality,
        known_sky_squared_sum,
        known_sky_cubed_sum,
        sky_averaged_mean_numerator,
        sky_averaged_variance_numerator,
    ) = compressed_signal_moments(
        args.n_injections,
        args.batch_size,
        args.seed,
        mid_times,
        intrinsic,
    )
    (
        false_positive,
        equal_true_positive,
        known_sky_true_positive,
        sky_averaged_true_positive,
    ) = make_roc(
        total_noncentrality,
        known_sky_squared_sum,
        known_sky_cubed_sum,
        sky_averaged_mean_numerator,
        sky_averaged_variance_numerator,
        np.sum(intrinsic**2),
        mid_times.size,
    )
    plot_roc(
        false_positive,
        equal_true_positive,
        known_sky_true_positive,
        sky_averaged_true_positive,
        args.output,
    )

    args.data_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.data_output,
        false_positive_probability=false_positive,
        equal_weight_true_positive_probability=equal_true_positive,
        unequal_weight_true_positive_probability=known_sky_true_positive,
        known_sky_weight_true_positive_probability=known_sky_true_positive,
        sky_averaged_weight_true_positive_probability=sky_averaged_true_positive,
        total_noncentrality_quantiles=np.quantile(
            total_noncentrality,
            [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0],
        ),
        total_noncentrality_quantile_probabilities=np.array(
            [0.0, 0.01, 0.1, 0.5, 0.9, 0.99, 1.0]
        ),
        f_start=F_START,
        f_stop=F_STOP,
        mchirp=MCHIRP,
        distance_pc=DISTANCE_PC,
        distance_in_andromeda=DISTANCE_IN_ANDROMEDA,
        coherent_duration=T_COH,
        n_segments=mid_times.size,
        n_injections=args.n_injections,
        seed=args.seed,
        weighted_distribution="normal CLT from exact lambda moments",
        known_sky_weighting_convention="w_i proportional to lambda_i",
        sky_averaged_weighting_convention=(
            "w_i proportional to intrinsic A_i^2 / S_i_eff"
        ),
        equal_distribution="ncx2 survival over quantile-compressed sky samples",
    )

    known_sky_gain = known_sky_true_positive - equal_true_positive
    sky_averaged_gain = sky_averaged_true_positive - equal_true_positive
    print("Random-sky semicoherent ROC")
    print(
        "  source distance: "
        f"{DISTANCE_IN_ANDROMEDA:g} x Andromeda ({DISTANCE_PC:.3e} pc)"
    )
    print(f"  Mc: {MCHIRP:.1e} Msun")
    print(f"  f0 / analysis band: {F_START:g} / {F_START:g}-{F_STOP:g} Hz")
    print(f"  signal duration: {duration:.3f} s")
    print(f"  segments: {mid_times.size} x {T_COH:g} s")
    print(f"  random-sky injections: {args.n_injections}")
    print(
        "  median total noncentrality: "
        f"{np.median(total_noncentrality):.3e}"
    )
    print(
        "  known-sky-weight TPR gain at FAP=1e-4: "
        f"{known_sky_gain[1]:+.4f}"
    )
    print(
        "  sky-averaged-weight TPR gain at FAP=1e-4: "
        f"{sky_averaged_gain[1]:+.4f}"
    )
    print(
        "  maximum plotted known-sky TPR gain: "
        f"{np.max(known_sky_gain[:-1]):+.4f}"
    )
    print(
        "  maximum plotted sky-averaged TPR gain: "
        f"{np.max(sky_averaged_gain[:-1]):+.4f}"
    )
    print(f"  plot: {args.output}")
    print(f"  data: {args.data_output}")


if __name__ == "__main__":
    main()
