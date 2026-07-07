"""Inject a 3.5PN chirp into colored noise and recover track power."""

import argparse
from pathlib import Path

import numpy as np

import semicoherent_injection_nonoise as base


OUTPUT_PATH = base.SCRIPT_DIR / "semicoherent_track_injection_colored_noise.png"


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


def main():
    args = parse_args()
    base.validate_args(args)

    noise = base.NoiseCurve.from_asd_file(args.asd)
    frequency_model = base.semicoherent_frequency_model(args)
    duration, f_end = base.observation_span(frequency_model)
    chirp_power = base.integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )
    distance_m = base.semicoherent_distance_sensitivity(
        args.f0,
        args.mchirp,
        chirp_power,
        duration,
        chunk_duration=args.chunk_duration,
        lambda_thresh=args.lambda_threshold,
    )

    t, signal, frequency_track, eta, psi = base.make_injection(
        args,
        distance_m,
        duration,
    )
    colored_noise = make_colored_noise(noise, args.sample_rate, signal.size, args.seed)
    strain = signal + colored_noise

    times, frequencies, powers = base.recover_track_power(
        t, strain, frequency_track, args, noise
    )
    base.plot_chunks(times, frequencies, powers, args.output)

    overlap_scale = 1.0 - args.chunk_overlap
    recovered_statistic = overlap_scale * float(np.sum(powers))
    n_chunks = base.chunk_count(duration, args.chunk_duration)
    expected_sky_averaged_power = args.lambda_threshold * np.sqrt(n_chunks)
    response_weights = frequency_track ** (4.0 / 3.0) / noise.psd_at(frequency_track)
    expected_signal_power = base.power_at_distance(
        distance_m,
        args.f0,
        args.mchirp,
        chirp_power,
        base.GALACTIC_CENTER_RA,
        base.GALACTIC_CENTER_DEC,
        eta,
        psi,
        base.injection_gmst(t),
        weights=response_weights,
    )
    window = np.hanning(int(round(args.chunk_duration * args.sample_rate)))
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)
    null_mean, null_variance, dof, scale, variance_inflation = (
        base.effective_chi_squared_params(
            powers.size,
            window,
            int(round(window.size * overlap_scale)),
            overlap_scale,
        )
    )
    recovered_excess_power = recovered_statistic - null_mean
    window_normalized_excess_power = recovered_excess_power / hann_power_factor
    expected_signal_statistic = hann_power_factor * expected_signal_power
    expected_statistic = null_mean + expected_signal_statistic
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
    print(f"distance sensitivity: {distance_m / base.PARSEC_M:.2e} pc")
    print(f"lambda threshold: {args.lambda_threshold:.12g}")
    print(
        "chunks in sensitivity formula: "
        f"{base.chunk_count(duration, args.chunk_duration)}"
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
    print(f"expected sky-averaged signal power: {expected_sky_averaged_power:.2e}")
    print(f"effective chi2 dof: {dof:.2e}")
    print(f"effective chi2 scale: {scale:.2e}")
    print(f"overlap variance inflation: {variance_inflation:.2e}")
    print(f"null mean statistic: {null_mean:.2e}")
    print(f"null std statistic: {np.sqrt(null_variance):.2e}")
    print(f"null p-value for realized statistic: {recovered_p_value:.2e}")
    print(f"null p-value for expected statistic: {expected_p_value:.2e}")
    print(f"realized Gaussian-equivalent significance: {recovered_sigma:.2e} sigma")
    print(f"expected Gaussian-equivalent significance: {expected_sigma:.2e} sigma")


if __name__ == "__main__":
    main()
