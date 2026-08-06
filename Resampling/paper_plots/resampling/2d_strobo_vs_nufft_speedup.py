"""Benchmark batched Stroboscopic FFTs against a batched NUFFT."""

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


PAPER_PLOTS_DIR = Path(__file__).resolve().parents[1]
if str(PAPER_PLOTS_DIR) not in sys.path:
    sys.path.insert(0, str(PAPER_PLOTS_DIR))

from resampler import Resampler, clear_plan_cache

SCRIPT_DIR = Path(__file__).resolve().parent
plt.style.use(SCRIPT_DIR.parent / "paper.mplstyle")



G = 6.67430e-11
C = 299792458.0
MSUN = 2.0e30

MC_MSUN = 1e-2
F0_HZ = 20.0
F_NEW_HZ = 512.0
T_OBS_S = (32.0, 256.0, 2048.0)
THREADS = 4
REPEATS = 15
RATIO_START = 8.0
RATIO_STOP = 256.0
RATIOS = tuple(
    2.0 ** np.arange(np.log2(RATIO_START), np.log2(RATIO_STOP) + 1.0)
)
EPS_MODE = "scaling"
EPS_LABEL = r"Scaling $\epsilon = 1 / r$"

CSV_FIELDS = [
    "t_obs_s",
    "f_ratio",
    "strobo_input_hz",
    "eps_mode",
    "eps",
    "strobo_samples",
    "strobo_output_samples",
    "nufft_samples",
    "nufft_modes",
    "nufft_transforms",
    "nufft_total_input_samples",
    "strobo_seconds",
    "nufft_seconds",
    "nufft_input_samples_per_second",
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


def require_power_of_two(n_samples, name):
    if n_samples <= 0 or n_samples & (n_samples - 1):
        raise ValueError(f"{name} must be a power of two, got {n_samples}.")


def build_signal(beta, f0_hz, n_samples, sample_rate_hz):
    require_power_of_two(n_samples, "n_samples")
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


def build_batched_signals(beta, f0_hz, n_samples, sample_rate_hz, n_transforms):
    if n_transforms <= 0:
        raise ValueError("n_transforms must be positive.")
    signal, tau = build_signal(beta, f0_hz, n_samples, sample_rate_hz)
    phase_offsets = np.complex64(np.exp(0.73j * np.arange(n_transforms)))
    return np.ascontiguousarray(phase_offsets[:, None] * signal[None, :]), tau


def strobo_resample(tau, data, output_rate_hz, target_samples):
    scaled_tau = tau * output_rate_hz
    indices = np.nonzero(np.diff(np.floor(scaled_tau)))[0]
    if indices.size < target_samples:
        raise ValueError(
            f"Strobo produced {indices.size} samples, fewer than requested {target_samples}."
        )

    indices = indices[:target_samples]
    if data.ndim == 1:
        strobo_data = data[indices]
    elif data.ndim == 2 and data.shape[0] > 0:
        strobo_data = data[:, indices]
    else:
        raise ValueError("data must have shape (N,) or (n_transforms, N).")
    return np.ascontiguousarray(strobo_data, dtype=np.complex64)


def get_fftw_plan(plans, n_samples, n_trans, threads):
    require_power_of_two(n_samples, "Stroboscopic FFT length")
    key = (n_samples, n_trans)
    if key not in plans:
        shape = (n_samples,) if n_trans == 1 else (n_trans, n_samples)
        in_array = pyfftw.empty_aligned(shape, dtype="complex64")
        out_array = pyfftw.empty_aligned(shape, dtype="complex64")
        fft = pyfftw.FFTW(
            in_array,
            out_array,
            axes=(-1,),
            direction="FFTW_FORWARD",
            threads=threads,
            flags=("FFTW_ESTIMATE", "FFTW_DESTROY_INPUT"),
        )
        plans[key] = in_array, out_array, fft
    return plans[key]


def run_strobo_once(tau, signal, output_rate_hz, target_samples, fftw_plans, threads):
    start = time.perf_counter()
    strobo_data = strobo_resample(tau, signal, output_rate_hz, target_samples)
    n_trans = 1 if strobo_data.ndim == 1 else strobo_data.shape[0]
    in_array, out_array, fft = get_fftw_plan(
        fftw_plans, strobo_data.shape[-1], n_trans, threads
    )
    in_array[:] = strobo_data
    fft()
    return time.perf_counter() - start, strobo_data.shape[-1]


def make_nufft_resampler(signal, tau, n_modes, eps, threads):
    require_power_of_two(signal.shape[-1], "NUFFT input length")
    require_power_of_two(n_modes, "NUFFT output length")
    resampler = Resampler(
        nthreads=threads,
        eps=eps,
        precision="single",
        upsampfac=1.25,
        fftw_measure=False,
        n_modes=n_modes,
        allow_two_transforms=True,
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


def run_benchmark(ratios, repeats, t_obs_values_s, f_new_hz, threads, n_transforms):
    clear_plan_cache()
    beta = pbh_beta(F0_HZ, MC_MSUN)
    rows = []
    fftw_plans = {}
    for t_obs_s in t_obs_values_s:
        nufft_samples = int(np.rint(t_obs_s * f_new_hz))
        require_power_of_two(
            nufft_samples,
            f"NUFFT sample count for t_obs_s={t_obs_s:g}",
        )

        nufft_signal, nufft_tau = build_batched_signals(
            beta, F0_HZ, nufft_samples, f_new_hz, n_transforms
        )
        # Use the same radix-2 output length for the stroboscopic FFT and NUFFT.
        nufft_modes = nufft_samples

        for f_ratio in ratios:
            eps = 1.0 / f_ratio
            strobo_input_hz = f_new_hz * f_ratio
            exact_strobo_samples = nufft_samples * f_ratio
            strobo_input_samples = int(np.rint(exact_strobo_samples))
            if not np.isclose(
                strobo_input_samples,
                exact_strobo_samples,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(
                    "nufft_samples * f_ratio must be an integer, got "
                    f"{exact_strobo_samples}."
                )
            require_power_of_two(
                strobo_input_samples,
                f"Stroboscopic input sample count for f_ratio={f_ratio:g}",
            )
            strobo_signal, strobo_tau = build_batched_signals(
                beta,
                F0_HZ,
                strobo_input_samples,
                strobo_input_hz,
                n_transforms,
            )

            _, strobo_output_samples = run_strobo_once(
                strobo_tau,
                strobo_signal,
                f_new_hz,
                nufft_modes,
                fftw_plans,
                threads,
            )
            nufft_resampler = make_nufft_resampler(
                nufft_signal,
                nufft_tau,
                nufft_modes,
                eps,
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
                nufft_times.append(run_nufft_once(nufft_resampler))

            strobo_seconds = median(strobo_times)
            nufft_seconds = median(nufft_times)
            nufft_transforms = nufft_signal.shape[0]
            nufft_total_input_samples = nufft_transforms * nufft_signal.shape[-1]
            nufft_input_samples_per_second = (
                nufft_total_input_samples / nufft_seconds
            )
            speedup = strobo_seconds / nufft_seconds
            rows.append(
                {
                    "t_obs_s": t_obs_s,
                    "f_ratio": f_ratio,
                    "strobo_input_hz": strobo_input_hz,
                    "eps_mode": EPS_MODE,
                    "eps": eps,
                    "strobo_samples": strobo_signal.shape[-1],
                    "strobo_output_samples": strobo_output_samples,
                    "nufft_samples": nufft_signal.shape[-1],
                    "nufft_modes": nufft_modes,
                    "nufft_transforms": nufft_transforms,
                    "nufft_total_input_samples": nufft_total_input_samples,
                    "strobo_seconds": strobo_seconds,
                    "nufft_seconds": nufft_seconds,
                    "nufft_input_samples_per_second": nufft_input_samples_per_second,
                    "nufft_speedup": speedup,
                }
            )
            print(
                f"t_obs={t_obs_s:>7.3g}s, f_ratio={f_ratio:>8.3g}: "
                f"strobo={strobo_seconds:.4f}s, "
                f"nufft={nufft_seconds:.4f}s, "
                f"nufft throughput={nufft_input_samples_per_second:.3g} samples/s, "
                f"speedup={speedup:.2f}x"
            )
    return rows


def save_csv(rows, output_path):
    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(input_path):
    with input_path.open(newline="") as f:
        rows = [
            {
                "t_obs_s": float(row["t_obs_s"]),
                "f_ratio": float(row["f_ratio"]),
                "strobo_input_hz": float(row["strobo_input_hz"]),
                "eps_mode": row["eps_mode"],
                "eps": float(row["eps"]),
                "strobo_samples": int(row["strobo_samples"]),
                "strobo_output_samples": int(row["strobo_output_samples"]),
                "nufft_samples": int(row["nufft_samples"]),
                "nufft_modes": int(row["nufft_modes"]),
                "nufft_transforms": int(row.get("nufft_transforms", 2)),
                "nufft_total_input_samples": int(
                    row.get("nufft_total_input_samples", 2 * int(row["nufft_samples"]))
                ),
                "strobo_seconds": float(row["strobo_seconds"]),
                "nufft_seconds": float(row["nufft_seconds"]),
                "nufft_input_samples_per_second": float(
                    row.get(
                        "nufft_input_samples_per_second",
                        2 * int(row["nufft_samples"]) / float(row["nufft_seconds"]),
                    )
                ),
                "nufft_speedup": float(row["nufft_speedup"]),
            }
            for row in csv.DictReader(f)
        ]
    if not rows:
        raise ValueError(f"No benchmark rows found in {input_path}.")
    return rows


def save_plot(rows, output_path):
    fig, ax = plt.subplots(constrained_layout=True)
    t_obs_values_s = sorted({row["t_obs_s"] for row in rows})
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for t_obs_s, color in zip(t_obs_values_s, colors):
        duration_rows = sorted(
            (
                row
                for row in rows
                if row["t_obs_s"] == t_obs_s and row["eps_mode"] == EPS_MODE
            ),
            key=lambda row: row["f_ratio"],
        )
        if duration_rows:
            ax.semilogx(
                [1 / row["f_ratio"] for row in duration_rows],
                [row["nufft_speedup"] for row in duration_rows],
                marker="o",
                color=color,
                label=(
                    rf"$T_{{\rm coh}} = {t_obs_s:g}\,\mathrm{{s}}$"
                ),
            )
    # ax.axhline(1.0, color="black", ls="--", alpha=0.7)
    ax.set_xlabel("Resampling error")
    ax.set_ylabel("NUFFT / Stroboscopic speedup")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(output_path, dpi=220)
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark batched Stroboscopic FFTs against a batched FINUFFT."
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=RATIOS,
        help="Rate ratios that keep T_obs * f_new * ratio a power of two.",
    )
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument(
        "--t-obs",
        type=float,
        nargs="+",
        default=T_OBS_S,
        help=(
            "Observation durations in seconds; each T_obs * f_new must be a "
            "power of two."
        ),
    )
    parser.add_argument("--f-new", type=float, default=F_NEW_HZ)
    parser.add_argument("--threads", type=int, default=THREADS)
    parser.add_argument(
        "--n-transforms",
        type=int,
        default=2,
        help="Number of simultaneous transforms in each batch.",
    )
    parser.add_argument(
        "--load",
        action="store_true",
        help="Load figs/2d_strobo_vs_nufft_speedup.csv and only regenerate the plot.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(__file__).resolve().parent / "figs"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "2d_strobo_vs_nufft_speedup.csv"
    plot_path = output_dir / "2d_strobo_vs_nufft_speedup.png"
    if args.load:
        rows = load_csv(csv_path)
        print(f"Loaded {csv_path}")
    else:
        rows = run_benchmark(
            args.ratios,
            args.repeats,
            args.t_obs,
            args.f_new,
            args.threads,
            args.n_transforms,
        )
        save_csv(rows, csv_path)
        print(f"Saved {csv_path}")
    save_plot(rows, plot_path)
    print(f"Saved {plot_path}")


if __name__ == "__main__":
    main()
