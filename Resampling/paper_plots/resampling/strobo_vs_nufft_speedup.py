import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pyfftw


PAPER_PLOTS_ROOT = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_ROOT) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_ROOT))

from resampler import Resampler, clear_plan_cache


G = 6.67430e-11
C = 299792458.0
MSUN = 2.0e30

MC_MSUN = 1e-2
F0_HZ = 20.0
F_NEW_HZ = 512.0
T_OBS_S = (30.0, 300.0, 3000.0)
THREADS = 4
REPEATS = 7
RATIOS = tuple(np.geomspace(2.0, 64.0, 17))

CSV_FIELDS = [
    "t_obs_s",
    "f_ratio",
    "strobo_input_hz",
    "eps",
    "strobo_samples",
    "strobo_output_samples",
    "nufft_samples",
    "nufft_modes",
    "strobo_seconds",
    "nufft_seconds",
    "nufft_speedup",
]


def pbh_beta(f0_hz, mc_msun):
    mc_si = mc_msun * MSUN
    return (
        96.0
        / 5.0
        * np.pi ** (8.0 / 3.0)
        * (G / C**3) ** (5.0 / 3.0)
        * f0_hz ** (8.0 / 3.0)
        * mc_si ** (5.0 / 3.0)
    )


def build_signal(beta, f0_hz, n_samples, sample_rate_hz):
    t = np.arange(n_samples, dtype=np.float64) / sample_rate_hz
    chirp_factor = 1.0 - (8.0 / 3.0) * beta * t

    if np.any(chirp_factor <= 0.0):
        raise ValueError("Requested signal reaches coalescence during the observation.")

    phase = (
        -6.0
        * np.pi
        / 5.0
        * f0_hz
        * chirp_factor ** (5.0 / 8.0)
        / beta
    )
    signal = np.exp(1j * (phase - phase[0])).astype(np.complex64, copy=False)

    tau = -(3.0 / (5.0 * beta)) * chirp_factor ** (5.0 / 8.0)
    tau -= tau[0]
    return signal, tau


def strobo_resample(tau, data, output_rate_hz, target_samples):
    scaled_tau = tau * output_rate_hz
    indices = np.nonzero(np.diff(np.floor(scaled_tau)))[0]

    if indices.size < target_samples:
        raise ValueError(
            f"Strobo produced {indices.size} samples, fewer than requested {target_samples}."
        )

    return np.ascontiguousarray(data[indices[:target_samples]], dtype=np.complex64)


def get_fftw_plan(plans, n_samples, threads):
    if n_samples not in plans:
        in_array = pyfftw.empty_aligned(n_samples, dtype="complex64")
        out_array = pyfftw.empty_aligned(n_samples, dtype="complex64")
        fft = pyfftw.FFTW(
            in_array,
            out_array,
            direction="FFTW_FORWARD",
            threads=threads,
            flags=("FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"),
        )
        plans[n_samples] = in_array, out_array, fft

    return plans[n_samples]


def run_strobo_once(tau, signal, output_rate_hz, target_samples, fftw_plans, threads):
    start = time.perf_counter()
    strobo_data = strobo_resample(tau, signal, output_rate_hz, target_samples)

    in_array, out_array, fft = get_fftw_plan(fftw_plans, strobo_data.size, threads)
    in_array[:] = strobo_data
    fft()

    return time.perf_counter() - start, strobo_data.size


def make_nufft_resampler(signal, tau, n_modes, eps, threads):
    resampler = Resampler(
        nthreads=threads,
        eps=eps,
        precision="single",
        upsampfac=1.25,
        fftw_measure=False,
        n_modes=n_modes,
    )
    resampler.timeseries = signal
    resampler.resampled_time = tau
    resampler.prepare_nufft()
    return resampler


def run_nufft_once(resampler):
    start = time.perf_counter()
    resampler.nufft_prepared()

    return time.perf_counter() - start


def median(values):
    return float(np.median(np.asarray(values, dtype=float)))


def run_benchmark(ratios, repeats, t_obs_values_s, f_new_hz, threads):
    clear_plan_cache()

    beta = pbh_beta(F0_HZ, MC_MSUN)
    rows = []
    fftw_plans = {}

    for t_obs_s in t_obs_values_s:
        nufft_samples = int(np.rint(t_obs_s * f_new_hz))
        if nufft_samples <= 0:
            raise ValueError(f"t_obs_s must give at least one sample, got {t_obs_s}.")

        nufft_signal, nufft_tau = build_signal(
            beta,
            f0_hz=F0_HZ,
            n_samples=nufft_samples,
            sample_rate_hz=f_new_hz,
        )
        # Match the stroboscopic FFT period to the NUFFT's full resampled-time
        # span. The input sample count and output mode count need not coincide.
        nufft_modes = int(np.rint(nufft_tau[-1] * f_new_hz))
        if nufft_modes <= 0:
            raise ValueError("The resampled-time span must contain at least one sample.")

        for f_ratio in ratios:
            eps = 1.0 / f_ratio
            strobo_input_hz = f_new_hz * f_ratio
            strobo_input_samples = int(np.ceil(nufft_samples * f_ratio)) + 128
            strobo_signal, strobo_tau = build_signal(
                beta,
                f0_hz=F0_HZ,
                n_samples=strobo_input_samples,
                sample_rate_hz=strobo_input_hz,
            )

            # Build the Resampler and call FINUFFT setpts outside the timing loop.
            nufft_resampler = make_nufft_resampler(
                nufft_signal,
                nufft_tau,
                nufft_modes,
                eps,
                threads,
            )

            # Warm the FFT and NUFFT execution paths once outside timings.
            _, strobo_output_samples = run_strobo_once(
                strobo_tau,
                strobo_signal,
                f_new_hz,
                nufft_modes,
                fftw_plans,
                threads,
            )
            run_nufft_once(nufft_resampler)

            strobo_times = []
            nufft_times = []
            for _ in range(repeats):
                strobo_time, strobo_output_samples = run_strobo_once(
                    strobo_tau,
                    strobo_signal,
                    f_new_hz,
                    nufft_modes,
                    fftw_plans,
                    threads,
                )
                strobo_times.append(strobo_time)
                nufft_times.append(
                    run_nufft_once(nufft_resampler)
                )

            strobo_seconds = median(strobo_times)
            nufft_seconds = median(nufft_times)
            speedup = strobo_seconds / nufft_seconds

            row = {
                "t_obs_s": t_obs_s,
                "f_ratio": f_ratio,
                "strobo_input_hz": strobo_input_hz,
                "eps": eps,
                "strobo_samples": strobo_signal.size,
                "strobo_output_samples": strobo_output_samples,
                "nufft_samples": nufft_signal.size,
                "nufft_modes": nufft_modes,
                "strobo_seconds": strobo_seconds,
                "nufft_seconds": nufft_seconds,
                "nufft_speedup": speedup,
            }
            rows.append(row)

            print(
                f"t_obs={t_obs_s:>7.3g}s, f_ratio={f_ratio:>8.3g}: "
                f"strobo={strobo_seconds:.4f}s, "
                f"nufft={nufft_seconds:.4f}s, "
                f"speedup={speedup:.2f}x"
            )

    return rows


def save_csv(rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(input_path):
    rows = []

    with input_path.open(newline="") as f:
        for row in csv.DictReader(f):
            eps = float(row["eps"])
            rows.append(
                {
                    "t_obs_s": float(row["t_obs_s"]),
                    "f_ratio": float(row["f_ratio"]),
                    "strobo_input_hz": float(row["strobo_input_hz"]),
                    "eps": eps,
                    "strobo_samples": int(row["strobo_samples"]),
                    "strobo_output_samples": int(row["strobo_output_samples"]),
                    "nufft_samples": int(row["nufft_samples"]),
                    "nufft_modes": int(row.get("nufft_modes", row["nufft_samples"])),
                    "strobo_seconds": float(row["strobo_seconds"]),
                    "nufft_seconds": float(row["nufft_seconds"]),
                    "nufft_speedup": float(row["nufft_speedup"]),
                }
            )

    if not rows:
        raise ValueError(f"No benchmark rows found in {input_path}.")

    return rows


def save_plot(rows, output_path):
    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)
    t_obs_values_s = sorted({row["t_obs_s"] for row in rows})
    for t_obs_s, color in zip(t_obs_values_s, plt.rcParams["axes.prop_cycle"].by_key()["color"]):
        t_obs_rows = sorted(
            (row for row in rows if row["t_obs_s"] == t_obs_s),
            key=lambda row: row["f_ratio"],
        )
        if t_obs_rows:
            ratios = np.asarray([row["f_ratio"] for row in t_obs_rows], dtype=float)
            speedups = np.asarray(
                [row["nufft_speedup"] for row in t_obs_rows], dtype=float
            )
            ax.semilogx(
                ratios,
                speedups,
                "o-",
                color=color,
                lw=1.8,
                label=rf"$t_{{\rm obs}} = {t_obs_s:g}\,\mathrm{{s}}$",
            )
    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.7)
    ax.set_xlabel("Upsampling ratio")
    ax.set_ylabel("NUFFT / stroboscopic speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark stroboscopic pyFFTW resampling against the paper-plots FINUFFT resampler."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=RATIOS,
        help="Strobo input/output sampling ratios; the NUFFT uses epsilon = 1 / ratio.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--t-obs",
        type=float,
        nargs="+",
        default=T_OBS_S,
        help="Observation durations in seconds, all sampled at --f-new.",
    )
    parser.add_argument("--f-new", type=float, default=F_NEW_HZ)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load figs/strobo_vs_nufft_speedup.csv and only regenerate the plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_dir = Path(__file__).resolve().parent / "figs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "strobo_vs_nufft_speedup.csv"
    plot_path = output_dir / "strobo_vs_nufft_speedup.png"

    if args.load:
        rows = load_csv(csv_path)
        print(f"Loaded {csv_path}")
    else:
        rows = run_benchmark(
            ratios=args.ratios,
            repeats=args.repeats,
            t_obs_values_s=args.t_obs,
            f_new_hz=args.f_new,
            threads=args.threads,
        )
        save_csv(rows, csv_path)
        print(f"Saved {csv_path}")

    save_plot(rows, plot_path)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
