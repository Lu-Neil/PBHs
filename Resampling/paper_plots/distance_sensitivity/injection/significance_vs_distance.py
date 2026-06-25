"""Plot expected no-noise injection significance versus injection distance."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import special
from scipy.stats import chi2


SCRIPT_DIR = Path(__file__).resolve().parent
PAPER_PLOTS_DIR = SCRIPT_DIR.parents[1]
OUTPUT_PATH = SCRIPT_DIR / "significance_vs_distance.png"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

import semicoherent_injection_nonoise as injection  # noqa: E402

plt.style.use(PAPER_PLOTS_DIR / "paper.mplstyle")


DEFAULT_RESPONSE_SAMPLES = 20_000
DEFAULT_MIN_DISTANCE_RATIO = 0.5
DEFAULT_MAX_DISTANCE_RATIO = 2.0


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Scan distances around the no-noise semicoherent injection "
            "sensitivity and plot the expected Gaussian-equivalent significance."
        )
    )
    for name, default in injection.DEFAULTS.items():
        parser.add_argument(
            f"--{name.replace('_', '-')}",
            type=type(default),
            default=default,
        )
    parser.add_argument(
        "--f-end",
        dest="f_max",
        type=float,
        help="Backward-compatible alias for --f-max.",
    )
    parser.add_argument("--asd", type=Path, default=injection.ASD_PATH)
    parser.add_argument("--n-distances", type=int, default=20)
    parser.add_argument(
        "--min-distance-ratio",
        type=float,
        default=DEFAULT_MIN_DISTANCE_RATIO,
        help="Smallest plotted distance as a multiple of the semicoherent distance.",
    )
    parser.add_argument(
        "--max-distance-ratio",
        type=float,
        default=DEFAULT_MAX_DISTANCE_RATIO,
        help="Largest plotted distance as a multiple of the semicoherent distance.",
    )
    parser.add_argument(
        "--response-samples",
        type=int,
        default=DEFAULT_RESPONSE_SAMPLES,
        help=(
            "Number of time samples used to average the detector response in "
            "the expected signal power calculation."
        ),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    return parser.parse_args()


def validate_args(args):
    injection.validate_args(args)
    if args.n_distances < 2:
        raise ValueError("n_distances must be at least 2")
    if args.min_distance_ratio <= 0.0:
        raise ValueError("min_distance_ratio must be positive")
    if args.max_distance_ratio <= args.min_distance_ratio:
        raise ValueError("max_distance_ratio must exceed min_distance_ratio")
    if args.response_samples < 2:
        raise ValueError("response_samples must be at least 2")


def trapezoid_weights(x):
    if x.size < 2:
        raise ValueError("at least two quadrature samples are required")

    weights = np.empty_like(x, dtype=float)
    weights[0] = 0.5 * (x[1] - x[0])
    weights[-1] = 0.5 * (x[-1] - x[-2])
    if x.size > 2:
        weights[1:-1] = 0.5 * (x[2:] - x[:-2])
    return weights


def response_quadrature(frequency_model, duration, noise, n_samples):
    times = np.linspace(0.0, duration, n_samples)
    frequencies = frequency_model.frequency(times)
    weights = (
        trapezoid_weights(times)
        * frequencies ** (4.0 / 3.0)
        / noise.psd_at(frequencies)
    )
    gmst = injection.injection_gmst(times)
    return gmst, weights


def semicoherent_setup(args):
    noise = injection.NoiseCurve.from_asd_file(args.asd)
    frequency_model = injection.semicoherent_frequency_model(args)
    duration, f_end = injection.observation_span(frequency_model)
    chirp_power = injection.integrated_chirp_power_35pn(
        args.f0,
        f_end,
        frequency_model,
        noise,
    )
    distance_m = injection.semicoherent_distance_sensitivity(
        args.f0,
        args.mchirp,
        chirp_power,
        duration,
        chunk_duration=args.chunk_duration,
        lambda_thresh=args.lambda_threshold,
    )
    return noise, frequency_model, duration, f_end, chirp_power, distance_m


def statistic_distribution(args, duration):
    chunk_samples = int(round(args.chunk_duration * args.sample_rate))
    hop_samples = int(round(chunk_samples * (1.0 - args.chunk_overlap)))
    if chunk_samples < 2 or hop_samples < 1:
        raise ValueError("invalid chunking configuration")

    n_samples = int(np.ceil(duration * args.sample_rate))
    n_chunks = injection.chunk_starts(n_samples, chunk_samples, hop_samples).size
    if n_chunks <= 0:
        raise ValueError("observation is shorter than one analysis chunk")

    window = np.hanning(chunk_samples)
    overlap_scale = 1.0 - args.chunk_overlap
    hann_power_factor = np.mean(window) ** 2 / np.mean(window**2)
    null_mean, null_variance, dof, scale, variance_inflation = (
        injection.effective_chi_squared_params(
            n_chunks,
            window,
            hop_samples,
            overlap_scale,
        )
    )
    return {
        "n_chunks": int(n_chunks),
        "overlap_scale": float(overlap_scale),
        "hann_power_factor": float(hann_power_factor),
        "null_mean": float(null_mean),
        "null_variance": float(null_variance),
        "dof": float(dof),
        "scale": float(scale),
        "variance_inflation": float(variance_inflation),
    }


def expected_signal_power(args, distance_m, chirp_power, gmst, weights):
    return injection.power_at_distance(
        distance_m,
        args.f0,
        args.mchirp,
        chirp_power,
        injection.GALACTIC_CENTER_RA,
        injection.GALACTIC_CENTER_DEC,
        injection.INJECTION_ETA,
        injection.INJECTION_PSI,
        gmst,
        weights=weights,
    )


def chi2_logsf_saddlepoint(statistic, dof):
    """Lugannani-Rice right-tail approximation for extreme chi-square values."""
    if statistic <= dof:
        return chi2.logsf(statistic, dof)

    saddlepoint = (statistic - dof) / (2.0 * statistic)
    cumulant = -0.5 * dof * math.log1p(-2.0 * saddlepoint)
    variance = 2.0 * dof / (1.0 - 2.0 * saddlepoint) ** 2
    signed_root = math.sqrt(2.0 * (saddlepoint * statistic - cumulant))
    scaled_root = saddlepoint * math.sqrt(variance)

    normal_logsf = special.log_ndtr(-signed_root)
    correction = 1.0 / signed_root - 1.0 / scaled_root
    correction_log = (
        -0.5 * signed_root**2
        - 0.5 * math.log(2.0 * math.pi)
        + math.log(correction)
    )
    correction_ratio = math.exp(correction_log - normal_logsf)
    if correction_ratio >= 1.0:
        return (
            -0.5 * signed_root**2
            - 0.5 * math.log(2.0 * math.pi)
            - math.log(scaled_root)
        )
    return normal_logsf + math.log1p(-correction_ratio)


def null_significance(statistic, dof, scale):
    p_value, sigma = injection.null_significance(statistic, dof, scale)
    if np.isfinite(sigma):
        return p_value, sigma

    scaled_statistic = statistic / scale
    log_p_value = chi2.logsf(scaled_statistic, dof)
    if np.isneginf(log_p_value):
        log_p_value = chi2_logsf_saddlepoint(scaled_statistic, dof)
    return math.exp(log_p_value), -special.ndtri_exp(log_p_value)


def significance_scan(args):
    noise, frequency_model, duration, f_end, chirp_power, distance_m = (
        semicoherent_setup(args)
    )
    gmst, response_weights = response_quadrature(
        frequency_model,
        duration,
        noise,
        args.response_samples,
    )
    reference_signal_power = expected_signal_power(
        args,
        distance_m,
        chirp_power,
        gmst,
        response_weights,
    )
    statistic_params = statistic_distribution(args, duration)

    ratios = np.linspace(
        args.min_distance_ratio,
        args.max_distance_ratio,
        args.n_distances,
    )
    distances_m = distance_m * ratios
    windowed_signal_power = (
        statistic_params["hann_power_factor"]
        * reference_signal_power
        / ratios**2
    )
    statistic = statistic_params["null_mean"] + windowed_signal_power

    p_values = np.empty_like(statistic)
    sigma = np.empty_like(statistic)
    for idx, value in enumerate(statistic):
        p_values[idx], sigma[idx] = null_significance(
            value,
            statistic_params["dof"],
            statistic_params["scale"],
        )

    return {
        "noise": noise,
        "frequency_model": frequency_model,
        "duration": float(duration),
        "f_end": float(f_end),
        "chirp_power": float(chirp_power),
        "distance_m": float(distance_m),
        "ratios": ratios,
        "distances_m": distances_m,
        "reference_signal_power": float(reference_signal_power),
        "windowed_signal_power": windowed_signal_power,
        "statistic": statistic,
        "p_values": p_values,
        "sigma": sigma,
        "statistic_params": statistic_params,
    }


def plot_significance(result, output):
    fig, ax = plt.subplots()
    ax.plot(result["ratios"], result["sigma"], marker="o", lw=1.5)
    ax.axvline(1.0, color="k", ls="--", lw=1.0, label="Semicoherent distance")
    ax.set_xlabel(r"Distance / $D_\mathrm{semi}$")
    ax.set_ylabel(r"Gaussian-equivalent significance [$\sigma$]")
    ax.grid(True, alpha=0.25)
    ax.legend()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220)
    plt.close(fig)


def main():
    args = parse_args()
    validate_args(args)

    result = significance_scan(args)
    plot_significance(result, args.output)

    distance_pc = result["distance_m"] / injection.PARSEC_M
    params = result["statistic_params"]

    print("-" * 9 + "Scan parameters" + "-" * 9)
    print("No-noise semicoherent injection expectation")
    print(f"f0: {args.f0:g} Hz")
    print(f"frequency band: {args.f_min:g}-{args.f_max:g} Hz")
    print(f"Mc: {args.mchirp:.2e} Msun")
    print(f"signal duration: {result['duration']:.2f} s")
    print(f"final semicoherent-path frequency: {result['f_end']:.6g} Hz")
    print(f"distance sensitivity: {distance_pc:.6e} pc")
    print(f"lambda threshold: {args.lambda_threshold:.12g}")
    print(f"analysis chunk duration: {args.chunk_duration:g} s")
    print(f"analysis chunk overlap: {args.chunk_overlap:.3g}")
    print(f"analysis chunks summed: {params['n_chunks']}")
    print(f"response quadrature samples: {args.response_samples}")
    print("-" * 9 + "Null model" + "-" * 9)
    print(f"effective chi2 dof: {params['dof']:.6e}")
    print(f"effective chi2 scale: {params['scale']:.6e}")
    print(f"overlap variance inflation: {params['variance_inflation']:.6e}")
    print(f"null mean statistic: {params['null_mean']:.6e}")
    print(f"null std statistic: {np.sqrt(params['null_variance']):.6e}")
    print("-" * 9 + "Distance scan" + "-" * 9)
    for ratio, distance, power, value, p_value, significance in zip(
        result["ratios"],
        result["distances_m"] / injection.PARSEC_M,
        result["windowed_signal_power"],
        result["statistic"],
        result["p_values"],
        result["sigma"],
    ):
        print(
            f"{ratio:5.2f} D_semi  "
            f"{distance:.6e} pc  "
            f"signal_power={power:.6e}  "
            f"statistic={value:.6e}  "
            f"p={p_value:.6e}  "
            f"sigma={significance:.6e}"
        )
    print(f"plot: {args.output}")


if __name__ == "__main__":
    main()
