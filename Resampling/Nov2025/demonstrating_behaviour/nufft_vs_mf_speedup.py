"""Wall-time comparison of NUFFT resampling vs scalar matched filtering as a function of N.

This benchmark neglects Doppler/sidereal detector modulation. The synthetic data
are a pure demodulated carrier,

    h(t) = H exp(i omega0 tau(t)),

so there are no 5-vector sidebands and no plus/cross detector-response
templates. The NUFFT path runs one FINUFFT transform on the data and reads the
carrier bin. The matched-filter path uses the corresponding scalar phase
template and runs two pyCBC FFTs (data + template) with plans cached by N. FFTW
is configured to run on NUM_FFT_THREADS OpenMP threads.

Key points illustrated:
  - Both methods scale as O(N log N); reference lines show this.
  - FFT has a smaller constant factor than NUFFT (NUFFT carries FINUFFT overhead).
  - The bottom panel shows the raw speedup ratio (MF / NUFFT); values > 1 mean NUFFT is faster.
  - In a real multi-template search the data NUFFT is done once, then each
    additional intrinsic template costs only a bin lookup, whereas MF requires
    another scalar template projection.
"""

import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import finufft
import matplotlib.pyplot as pl
import numpy as np
import pycbc.scheme
from pycbc.fft import FFT
from pycbc.fft import fftw as _pycbc_fftw
from pycbc.types import FrequencySeries, TimeSeries

PBH_CONST = 96 / 5 * np.pi ** (8 / 3) * (6.67e-11 / 3e8**3) ** (5 / 3)
SIDE_DAY = 86164.09053083288
NUM_FFT_THREADS = 4
NUM_NUFFT_THREADS = 4
FINUFFT_EPS = 1e-6
FINUFFT_UPSAMPFAC = 1.25
FFTW_ESTIMATE = 64

# FFTW threads backend is locked on first use, so set it before any plan exists.
# Tolerate already-set state (e.g. on module re-import in the same process).
try:
    _pycbc_fftw.set_threads_backend("openmp")
except RuntimeError:
    pass


SEED = 42
SAMPLE_RATE_HZ = 4.0
PHASE_MODEL = "pbh"
F0_SETTING = "midpoint"
# Keep the default benchmark small enough for laptops while still showing the
# large-N trend.
N_REPEATS = 7
DAYS_LIST = [0.25, 0.5, 1, 2, 4, 8]
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "nufft_vs_mf_speedup.png"


_PYCBC_FFT_PLANS: dict[int, tuple[TimeSeries, FrequencySeries, FFT]] = {}
_FINUFFT_PLANS: dict[int, finufft.Plan] = {}


@dataclass
class ScalarSimulation:
    signal: np.ndarray
    tau: np.ndarray
    omega0: float
    target_h: complex
    phase_template: np.ndarray
    n_samples: int
    number_of_days: float


@dataclass
class ScalarOutputs:
    h_est: complex


@dataclass
class TimingRow:
    number_of_days: float
    n_samples: int
    h_rel_err: float
    h_phase_err: float
    resampling_time_s: float
    matched_filter_time_s: float


def _get_pycbc_fft_plan(n):
    plan = _PYCBC_FFT_PLANS.get(n)
    if plan is None:
        _PYCBC_FFT_PLANS.clear()
        in_vec = TimeSeries(np.zeros(n, dtype=np.complex128), delta_t=1.0)
        out_vec = FrequencySeries(np.zeros(n, dtype=np.complex128), delta_f=1.0 / n)
        fft_obj = FFT(in_vec, out_vec)
        plan = (in_vec, out_vec, fft_obj)
        _PYCBC_FFT_PLANS[n] = plan
    return plan


def _pycbc_fft(arr):
    in_vec, out_vec, fft_obj = _get_pycbc_fft_plan(arr.shape[0])
    in_vec.data[:] = arr
    fft_obj.execute()
    return np.array(out_vec.data, copy=True)


def _get_finufft_plan(n):
    plan = _FINUFFT_PLANS.get(n)
    if plan is None:
        _FINUFFT_PLANS.clear()
        plan = finufft.Plan(
            nufft_type=1,
            n_modes_or_dim=(n,),
            isign=-1,
            eps=FINUFFT_EPS,
            nthreads=NUM_NUFFT_THREADS,
            dtype="complex128",
            upsampfac=FINUFFT_UPSAMPFAC,
            fftw=FFTW_ESTIMATE,
        )
        _FINUFFT_PLANS[n] = plan
    return plan


def _estimator(data_vec: np.ndarray, template_vec: np.ndarray) -> complex:
    return np.vdot(template_vec, data_vec) / np.vdot(template_vec, template_vec)


def _wrapped_phase_diff(phi_a: float, phi_b: float) -> float:
    return float(np.angle(np.exp(1j * (phi_a - phi_b))))


def _relative_error(estimate: complex, target: complex) -> float:
    return float(abs(estimate - target) / abs(target))


def _nearest_power_of_two(n: float) -> int:
    if n < 1:
        return 1

    lower = 2 ** int(np.floor(np.log2(n)))
    upper = 2 * lower
    if n - lower <= upper - n:
        return lower
    return upper


def _pbh_tau(t_offset: np.ndarray, f0: float, chirp_mass: float) -> tuple[np.ndarray, float]:
    beta = PBH_CONST * f0 ** (8 / 3) * chirp_mass ** (5 / 3)
    tau = -(3 / (5 * beta)) * (1 - 8 / 3 * beta * t_offset) ** (5 / 8)
    tau -= tau[0]
    return tau, beta


def _midpoint_pbh_frequency(t_last: float, chirp_mass: float, carrier_bin: int) -> float:
    def _tau_span(f0_local: float) -> float:
        beta_local = PBH_CONST * f0_local ** (8 / 3) * chirp_mass ** (5 / 3)
        return (3 / (5 * beta_local)) * (1 - (1 - 8 / 3 * beta_local * t_last) ** (5 / 8))

    f0 = carrier_bin / t_last
    for _ in range(20):
        next_f0 = carrier_bin / _tau_span(f0)
        if np.isclose(next_f0, f0, rtol=0.0, atol=1e-14):
            break
        f0 = next_f0
    return f0


def _create_scalar_simulation(
    *,
    number_of_days: float,
    sample_rate_hz: float,
    phase_model: str,
    f0_setting: str,
    rng: np.random.Generator,
) -> ScalarSimulation:
    t_obs = number_of_days * SIDE_DAY
    n_samples = _nearest_power_of_two(sample_rate_hz * t_obs)
    t_offset = np.arange(n_samples, dtype=float) / sample_rate_hz
    t_last = t_offset[-1]
    carrier_bin = 20000

    if phase_model == "pbh":
        chirp_mass = 10 ** rng.uniform(-3, -1) * 2e30
        if f0_setting == "midpoint":
            f0 = _midpoint_pbh_frequency(t_last, chirp_mass, carrier_bin)
        elif f0_setting == "uniform":
            f0 = rng.uniform(0.1, 0.2)
        else:
            raise ValueError(f"Unknown f0_setting: {f0_setting}")
        tau, _ = _pbh_tau(t_offset, f0, chirp_mass)
    elif phase_model == "fdot":
        fdot = 1e-9
        if f0_setting == "midpoint":
            f0 = (carrier_bin - 0.5 * fdot * t_last**2) / t_last
        elif f0_setting == "uniform":
            f0 = rng.uniform(0.1, 0.2)
        else:
            raise ValueError(f"Unknown f0_setting: {f0_setting}")
        tau = t_offset + 0.5 * (fdot / f0) * t_offset**2
    else:
        raise ValueError(f"Unknown phase_model: {phase_model}")

    if not (0 < f0 < sample_rate_hz / 2):
        raise ValueError(
            f"Injected carrier f0={f0:.6f} Hz is outside the Nyquist range for "
            f"sample_rate_hz={sample_rate_hz:.6f} Hz."
        )

    omega0 = 2 * np.pi * f0
    target_h = rng.uniform(1, 5) * np.exp(1j * rng.uniform(0, 2 * np.pi))
    phase_template = np.exp(1j * omega0 * tau)
    signal = target_h * phase_template

    return ScalarSimulation(
        signal=signal,
        tau=tau,
        omega0=omega0,
        target_h=target_h,
        phase_template=phase_template,
        n_samples=n_samples,
        number_of_days=number_of_days,
    )


def _extract_carrier_from_weights(weights, scale, omega0):
    index = int(np.round(omega0 / scale + weights.size // 2))
    index = np.clip(index, 0, weights.size - 1)
    return weights[index] / weights.size


def resampling_pipeline_outputs(sim):
    n = sim.tau.size
    scale = (2 * np.pi) / (sim.tau[-1] - sim.tau[0])
    tau_scaled = np.ascontiguousarray(scale * (sim.tau - sim.tau[0]), dtype=np.float64)

    input_data = np.ascontiguousarray(sim.signal, dtype=np.complex128)
    plan = _get_finufft_plan(n)
    plan.setpts(tau_scaled)
    weights = plan.execute(input_data)

    h_est = _extract_carrier_from_weights(weights, scale, sim.omega0)
    return ScalarOutputs(h_est=h_est)


def matched_filter_outputs(sim):
    # CPUScheme pushes NUM_FFT_THREADS into FFTW's plan-with-nthreads slot, so
    # plans are built (and re-executed) with that thread count.
    with pycbc.scheme.CPUScheme(num_threads=NUM_FFT_THREADS):
        signal_fft = _pycbc_fft(sim.signal)
        phase_template_fft = _pycbc_fft(sim.phase_template)

    h_est = _estimator(signal_fft, phase_template_fft)
    return ScalarOutputs(h_est=h_est)


def _time_call(fn, *args, **kwargs):
    start = time.perf_counter()
    out = fn(*args, **kwargs)
    elapsed = time.perf_counter() - start
    return out, elapsed


def _benchmark_method(fn, sim: ScalarSimulation, repeats: int) -> tuple[ScalarOutputs, float]:
    timings = []
    last_out = None
    for _ in range(repeats):
        last_out, elapsed = _time_call(fn, sim)
        timings.append(elapsed)
    return last_out, float(np.median(timings))


def benchmark_lengths(
    *,
    days_list: list[float],
    sample_rate_hz: float,
    phase_model: str,
    f0_setting: str,
    repeats: int,
    seed: int,
) -> list[TimingRow]:
    rows = []
    for idx, number_of_days in enumerate(days_list):
        rng = np.random.default_rng(seed + idx)
        sim = _create_scalar_simulation(
            number_of_days=number_of_days,
            sample_rate_hz=sample_rate_hz,
            phase_model=phase_model,
            f0_setting=f0_setting,
            rng=rng,
        )

        _ = resampling_pipeline_outputs(sim)
        _ = matched_filter_outputs(sim)

        resampling_out, resampling_time = _benchmark_method(resampling_pipeline_outputs, sim, repeats)
        matched_out, matched_time = _benchmark_method(matched_filter_outputs, sim, repeats)

        rows.append(
            TimingRow(
                number_of_days=number_of_days,
                n_samples=sim.n_samples,
                h_rel_err=_relative_error(resampling_out.h_est, matched_out.h_est),
                h_phase_err=abs(
                    _wrapped_phase_diff(np.angle(resampling_out.h_est), np.angle(matched_out.h_est))
                ),
                resampling_time_s=resampling_time,
                matched_filter_time_s=matched_time,
            )
        )
    return rows


def run_benchmark():
    print(
        f"Benchmarking: phase_model={PHASE_MODEL!r}, f0_setting={F0_SETTING!r}, "
        f"sample_rate={SAMPLE_RATE_HZ} Hz, {N_REPEATS} repeats (median); "
        f"NUFFT backend = FINUFFT scalar ({NUM_NUFFT_THREADS} threads); "
        f"MF backend = pyCBC FFT scalar ({NUM_FFT_THREADS} threads); "
        "Doppler/sidereal modulation neglected"
    )
    rows = benchmark_lengths(
        days_list=DAYS_LIST,
        sample_rate_hz=SAMPLE_RATE_HZ,
        phase_model=PHASE_MODEL,
        f0_setting=F0_SETTING,
        repeats=N_REPEATS,
        seed=SEED,
    )
    for row in rows:
        speedup = row.matched_filter_time_s / row.resampling_time_s
        print(
            f"  days={row.number_of_days:5.2f}  N={row.n_samples:>9d}  "
            f"NUFFT={1e3 * row.resampling_time_s:8.2f} ms  "
            f"MF={1e3 * row.matched_filter_time_s:8.2f} ms  "
            f"speedup={speedup:.2f}x"
        )
    return rows


def make_plot(rows):
    ns = np.array([r.n_samples for r in rows])
    t_nufft = np.array([r.resampling_time_s for r in rows])
    t_mf = np.array([r.matched_filter_time_s for r in rows])
    speedup = t_mf / t_nufft

    fig, axes = pl.subplots(2, 1, figsize=(8, 9), constrained_layout=True)
    fig.suptitle(
        "NUFFT carrier extraction vs scalar matched filtering: wall time\n"
        "Doppler/sidereal modulation neglected; "
        f"phase_model={PHASE_MODEL!r}, f0_setting={F0_SETTING!r}, "
        f"$F_S$={SAMPLE_RATE_HZ} Hz, {N_REPEATS} repeats (median)"
    )

    # --- top panel: absolute wall time ---
    ax = axes[0]
    ax.loglog(ns, t_nufft * 1e3, "o-", ms=6, lw=1.5, color="C0", label="NUFFT carrier bin")
    ax.loglog(ns, t_mf * 1e3, "s-", ms=6, lw=1.5, color="C1", label="Scalar MF (2 pyCBC FFTs)")
    ax.plot(np.nan, np.nan, label=r"Real analysis will use $\approx 10^9$ samples")

    ax.set_ylabel("Wall time (ms)")
    ax.set_xlabel("N (samples)")
    ax.legend(fontsize=9)

    # --- bottom panel: speedup ratio ---
    ax = axes[1]
    ax.semilogx(ns, speedup, "D-", ms=6, lw=1.5, color="C2", label="Speedup = MF time / NUFFT time")
    ax.axhline(1.0, color="0.5", lw=1.0, ls="--", label="Equal time")
    ax.set_xlabel("N (samples)")
    ax.set_ylabel("Speedup (MF time / NUFFT time)")
    ax.legend(fontsize=9)

    return fig


def main():
    rows = run_benchmark()
    fig = make_plot(rows)
    fig.savefig(OUTPUT_PATH, dpi=200)
    print(f"\nSaved plot to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
