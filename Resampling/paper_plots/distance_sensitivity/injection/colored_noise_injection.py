"""Inject a 3.5PN chirp into colored noise and recover track power."""

import argparse
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]

if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from distance_sensitivity.injection import nonoise_injection as base  # noqa: E402


OUTPUT_PATH = base.SCRIPT_DIR / "semicoherent_track_injection_colored_noise.png"
Z_DISTANCE_OUTPUT_PATH = base.SCRIPT_DIR / "z_vs_distance_colored_noise.png"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inject a 3.5PN chirp at the semicoherent distance sensitivity "
            "into Gaussian noise colored by the imported ASD."
        )
    )
    for name, default in base.DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=base.default_arg_type(name, default),
            default=default,
        )
    parser.add_argument("--asd", type=Path, default=base.ASD_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--distance-output", type=Path, default=Z_DISTANCE_OUTPUT_PATH)
    parser.add_argument(
        "--false-alarm-probability",
        type=float,
        default=base.DEFAULT_FALSE_ALARM_PROBABILITY,
    )
    parser.add_argument(
        "--detection-probability",
        type=float,
        default=base.DEFAULT_DETECTION_PROBABILITY,
    )
    parser.add_argument("--n-distances", type=int, default=base.DEFAULT_N_DISTANCES)
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=base.DEFAULT_MIN_DISTANCE_RATIO,
        help="Smallest plotted distance as a multiple of the semicoherent distance.",
    )
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=base.DEFAULT_MAX_DISTANCE_RATIO,
        help="Largest plotted distance as a multiple of the semicoherent distance.",
    )
    parser.add_argument("--seed", type=int, default=12345)
    return parser.parse_args()


def make_colored_noise(noise, sample_rate, n_samples, seed):
    rng = np.random.default_rng(seed)
    freqs = np.fft.rfftfreq(n_samples, d=1.0 / sample_rate)
    spectrum = np.zeros(freqs.size, dtype=complex)

    band = (
        (freqs >= noise.frequency[0])
        & (freqs <= noise.frequency[-1])
        & (freqs > 0.0)
    )
    psd = noise.psd_at(freqs[band])
    sigma = np.sqrt(n_samples * sample_rate * psd / 4.0)
    spectrum[band] = sigma * (
        rng.normal(size=psd.size) + 1j * rng.normal(size=psd.size)
    )

    if n_samples % 2 == 0 and band[-1]:
        spectrum[-1] = np.sqrt(n_samples * sample_rate * noise.psd_at(freqs[-1]) / 2.0)
        spectrum[-1] *= rng.normal()

    return np.fft.irfft(spectrum, n=n_samples)


def recover_statistic(t, strain, frequency_track, args, noise):
    times, frequencies, powers = base.recover_track_power(
        t, strain, frequency_track, args, noise
    )
    statistic = (1.0 - args.chunk_overlap) * float(np.sum(powers))
    return times, frequencies, powers, statistic


def colored_noise_distance_scan(
    reference_distance_m,
    reference_signal,
    colored_noise,
    t,
    frequency_track,
    expected_signal_statistic,
    null_mean,
    sigma0,
    args,
    noise,
):
    ratios = np.linspace(
        args.min_distance_ratio,
        args.max_distance_ratio,
        args.n_distances,
    )
    distances_m = reference_distance_m * ratios
    expected_z = expected_signal_statistic / sigma0 / ratios**2
    recovered_z = np.empty_like(expected_z)

    for idx, ratio in enumerate(ratios):
        strain = reference_signal / ratio + colored_noise
        _, _, _, recovered_statistic = recover_statistic(
            t,
            strain,
            frequency_track,
            args,
            noise,
        )
        recovered_z[idx] = (recovered_statistic - null_mean) / sigma0

    return distances_m, expected_z, recovered_z


def plot_z_vs_distance(result, output):
    fig, ax = base.plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    distance_pc = result["distances_m"] / base.PARSEC_M

    ax.plot(
        distance_pc,
        result["expected_z"],
        marker="o",
        ms=3.5,
        lw=1.5,
        label="Expected signal",
    )
    ax.plot(
        distance_pc,
        result["recovered_z"],
        marker="s",
        ms=3.5,
        lw=1.2,
        ls="--",
        label="Recovered signal+noise excess",
    )
    ax.axvline(
        result["distance_m"] / base.PARSEC_M,
        color="k",
        ls=":",
        lw=1.0,
        label="Semicoherent distance",
    )
    ax.axhline(
        result["threshold_z"],
        color="0.35",
        ls=":",
        lw=1.0,
        label="95% detection threshold",
    )
    ax.set_xlabel("Injection distance [pc]")
    ax.set_ylabel(r"$z = (\Lambda - \mu_0) / \sigma_0$")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    base.plt.close(fig)


def main():
    args = parse_args()
    base.validate_args(args)
    base.validate_distance_scan_args(args)

    noise = base.NoiseCurve.from_asd_file(args.asd)
    frequency_model = base.semicoherent_frequency_model(args)
    duration, f_end, n_chunks = base.analysis_span(args, frequency_model)
    chirp_power = base.integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )
    reference_distance_m = base.semicoherent_distance_sensitivity(
        args.f0,
        args.mchirp,
        chirp_power,
        duration,
        chunk_duration=args.chunk_duration,
        false_alarm_probability=args.false_alarm_probability,
        detection_probability=args.detection_probability,
        mismatch_bank=0.0,
        mismatch_coh=0.0,
    )

    overlap_scale = 1.0 - args.chunk_overlap
    chunk_samples, hop_samples = base.chunk_config(args)
    window = np.hanning(chunk_samples)
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)

    t = base.sample_times(args.sample_rate, duration)
    frequency_track = np.asarray(frequency_model.frequency(t), dtype=float)
    eta = base.INJECTION_ETA
    psi = base.INJECTION_PSI
    n_summed_chunks = base.count_summed_chunks(frequency_track, args)
    null_mean, null_variance, dof, scale, variance_inflation = (
        base.effective_chi_squared_params(
            n_summed_chunks,
            window,
            hop_samples,
            overlap_scale,
        )
    )
    sigma0 = float(np.sqrt(null_variance))
    thresholds = base.detection_thresholds(
        args.false_alarm_probability,
        args.detection_probability,
        dof,
        scale,
    )
    required_signal_statistic = thresholds["required_signal_statistic"]
    threshold_z = required_signal_statistic / sigma0
    false_alarm_z = (
        thresholds["false_alarm_statistic"] - null_mean
    ) / sigma0
    reference_expected_power = base.expected_power_for_track(
        args,
        reference_distance_m,
        chirp_power,
        t,
        frequency_track,
        eta,
        psi,
        noise,
    )
    reference_expected_signal_statistic = (
        hann_power_factor * reference_expected_power
    )
    if reference_expected_signal_statistic <= 0.0:
        raise ValueError("expected signal statistic must be positive")
    distance_m = reference_distance_m * np.sqrt(
        reference_expected_signal_statistic / required_signal_statistic
    )

    _, signal, _, _, _ = base.make_injection(
        args,
        distance_m,
        duration,
    )
    colored_noise = make_colored_noise(noise, args.sample_rate, signal.size, args.seed)
    strain = signal + colored_noise

    times, frequencies, powers, recovered_statistic = recover_statistic(
        t,
        strain,
        frequency_track,
        args,
        noise,
    )
    base.plot_chunks(times, frequencies, powers, args.output)

    expected_signal_power = base.expected_power_for_track(
        args,
        distance_m,
        chirp_power,
        t,
        frequency_track,
        eta,
        psi,
        noise,
    )
    recovered_excess_power = recovered_statistic - null_mean
    window_normalized_excess_power = recovered_excess_power / hann_power_factor
    expected_signal_statistic = hann_power_factor * expected_signal_power
    expected_statistic = null_mean + expected_signal_statistic
    recovered_z_at_reference = recovered_excess_power / sigma0
    expected_z_at_reference = expected_signal_statistic / sigma0
    distances_m, expected_z, recovered_z = colored_noise_distance_scan(
        distance_m,
        signal,
        colored_noise,
        t,
        frequency_track,
        expected_signal_statistic,
        null_mean,
        sigma0,
        args,
        noise,
    )
    plot_z_vs_distance(
        {
            "distance_m": distance_m,
            "distances_m": distances_m,
            "expected_z": expected_z,
            "recovered_z": recovered_z,
            "threshold_z": threshold_z,
        },
        args.distance_output,
    )
    expected_p_value, expected_sigma = base.null_significance(
        expected_statistic,
        dof,
        scale,
    )
    recovered_p_value, recovered_sigma = base.null_significance(
        recovered_statistic,
        dof,
        scale,
    )
    expected_detection_probability = base.ncx2.sf(
        thresholds["false_alarm_statistic"] / scale,
        dof,
        expected_signal_statistic / scale,
    )

    print("-" * 9 + "Injection parameters" + "-" * 9)
    print("Signal plus colored Gaussian noise")
    print(f"noise seed: {args.seed}")
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(
        "sky location: Galactic Center "
        f"(ra={np.rad2deg(base.GALACTIC_CENTER_RA):.6f} deg, "
        f"dec={np.rad2deg(base.GALACTIC_CENTER_DEC):.6f} deg)"
    )
    print(
        "detector: LIGO Livingston "
        f"(lat={np.rad2deg(base.LLO_LAT):.6f} deg, "
        f"lng={np.rad2deg(base.LLO_LNG):.6f} deg, "
        f"az={np.rad2deg(base.LLO_AZ):.6f} deg)"
    )
    print("t0: 2023-05-24 00:00:00 UTC")
    print(f"injection polarization: eta={eta:.6g}, psi={psi:.6g} rad")
    print(f"signal duration: {duration:.2f} s")
    print(f"final semicoherent-path frequency: {f_end:.6g} Hz")
    print(
        "zero-mismatch semicoherent distance estimate: "
        f"{reference_distance_m / base.PARSEC_M:.2e} pc"
    )
    print(f"distance sensitivity: {distance_m / base.PARSEC_M:.2e} pc")
    print(f"legacy single-bin lambda threshold: {args.lambda_threshold:.12g}")
    print(f"false alarm probability: {args.false_alarm_probability:.2e}")
    print(f"detection probability target: {args.detection_probability:.3g}")
    print(
        "chunks in sensitivity formula: "
        f"{n_chunks}"
    )
    print(f"analysis chunk duration: {args.chunk_duration:g} s")
    print(f"analysis chunk overlap: {args.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {powers.size}")
    print("-" * 9 + "Analysis results" + "-" * 9)
    print(f"realized signal+noise statistic: {recovered_statistic:.2e}")
    print(f"expected signal+noise statistic: {expected_statistic:.2e}")
    print(
        "overlap/window-normalized recovered excess power: "
        f"{window_normalized_excess_power:.2e}"
    )
    print(f"expected signal power: {expected_signal_power:.2e}")
    print(
        "unwindowed power required for threshold z: "
        f"{required_signal_statistic / hann_power_factor:.2e}"
    )
    print(f"effective chi2 dof: {dof:.2e}")
    print(f"effective chi2 scale: {scale:.2e}")
    print(f"overlap variance inflation: {variance_inflation:.2e}")
    print(f"null mean statistic: {null_mean:.2e}")
    print(f"null std statistic: {sigma0:.2e}")
    print(
        "false-alarm statistic threshold: "
        f"{thresholds['false_alarm_statistic']:.2e}"
    )
    print(f"false-alarm Gaussian z: {thresholds['false_alarm_gaussian_z']:.2e}")
    print(f"false-alarm threshold in script z: {false_alarm_z:.2e}")
    print(
        "required noncentrality for detection probability: "
        f"{thresholds['required_noncentrality']:.2e}"
    )
    print(f"required signal statistic: {required_signal_statistic:.2e}")
    print(f"expected statistic if in white noise: {expected_statistic:.2e}")
    print(f"required signal z: {threshold_z:.2e}")
    print(f"recovered z at distance sensitivity: {recovered_z_at_reference:.2e}")
    print(f"expected z at distance sensitivity: {expected_z_at_reference:.2e}")
    print(f"detection probability at expected statistic: {expected_detection_probability:.3g}")
    print(f"null p-value for realized statistic: {recovered_p_value:.2e}")
    print(f"null p-value for expected statistic: {expected_p_value:.2e}")
    print(f"realized Gaussian-equivalent significance: {recovered_sigma:.2e} sigma")
    print(f"expected Gaussian-equivalent significance: {expected_sigma:.2e} sigma")
    print(f"z(d) plot: {args.distance_output}")


if __name__ == "__main__":
    main()
