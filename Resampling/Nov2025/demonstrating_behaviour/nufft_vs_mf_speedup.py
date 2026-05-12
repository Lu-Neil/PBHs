"""Wall-time comparison of NUFFT resampling vs frequency-domain matched filtering as a function of N.

Both methods evaluate the same 3-template pipeline (h, h_p, h_c) on a synthetic PBH signal.
The NUFFT pipeline runs 4 NUFFTs (data + 3 templates) on non-uniform tau samples.
The frequency-domain MF runs 4 FFTs on the uniform time-domain samples, using the
pyCBC class-based FFT API (FFTW-backed) with plans cached by N. FFTW is configured
to run on NUM_FFT_THREADS OpenMP threads (pyCBC defaults to unthreaded).

Key points illustrated:
  - Both methods scale as O(N log N); reference lines show this.
  - FFT has a smaller constant factor than NUFFT (NUFFT carries FINUFFT overhead).
  - The bottom panel shows the raw speedup ratio (MF / NUFFT); values > 1 mean NUFFT is faster.
  - In a real multi-template search the NUFFT advantage grows: the data NUFFT is done once,
    then each additional template costs only a 5-bin lookup, whereas MF requires one FFT per
    template. This O(1) vs O(N log N) per-template cost is noted as an annotation.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as pl
import numpy as np
import pycbc.scheme
from pycbc.fft import FFT
from pycbc.fft import fftw as _pycbc_fftw
from pycbc.types import FrequencySeries, TimeSeries

NUM_FFT_THREADS = 4

# FFTW threads backend is locked on first use, so set it before any plan exists.
# Tolerate already-set state (e.g. on module re-import in the same process).
try:
    _pycbc_fftw.set_threads_backend("openmp")
except RuntimeError:
    pass

NOV2025_DIR = Path(__file__).resolve().parents[1]
if str(NOV2025_DIR) not in sys.path:
    sys.path.insert(0, str(NOV2025_DIR))

import matched_filter_comparison as mfc
from matched_filter_comparison import (
    FilterOutputs,
    _estimator,
    _joint_estimator,
    benchmark_lengths,
)


SEED = 42
SAMPLE_RATE_HZ = 4.0
PHASE_MODEL = "pbh"
F0_SETTING = "midpoint"
N_REPEATS = 5
DAYS_LIST = [0.25, 0.5, 1, 2, 4, 8, 16, 32, 64]
OUTPUT_PATH = Path(__file__).resolve().parent / "figs" / "nufft_vs_mf_speedup.png"


_PYCBC_FFT_PLANS: dict[int, tuple[TimeSeries, FrequencySeries, FFT]] = {}


def _get_pycbc_fft_plan(n):
    plan = _PYCBC_FFT_PLANS.get(n)
    if plan is None:
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


def pycbc_frequency_domain_outputs(sim):
    # CPUScheme pushes NUM_FFT_THREADS into FFTW's plan-with-nthreads slot, so
    # plans are built (and re-executed) with that thread count.
    with pycbc.scheme.CPUScheme(num_threads=NUM_FFT_THREADS):
        signal_fft = _pycbc_fft(sim.signal)
        template_fft = _pycbc_fft(sim.template)
        template_p_fft = _pycbc_fft(sim.template_p)
        template_c_fft = _pycbc_fft(sim.template_c)

    h_est = _estimator(signal_fft, template_fft)
    hp_est, hc_est = _joint_estimator(signal_fft, template_p_fft, template_c_fft)
    return FilterOutputs(h_est=h_est, hp_est=hp_est, hc_est=hc_est)


# Swap the np.fft-based MF in matched_filter_comparison for the pyCBC version
# so that benchmark_lengths times the pyCBC class-based FFT pipeline.
mfc.frequency_domain_outputs = pycbc_frequency_domain_outputs


def run_benchmark():
    print(
        f"Benchmarking: phase_model={PHASE_MODEL!r}, f0_setting={F0_SETTING!r}, "
        f"sample_rate={SAMPLE_RATE_HZ} Hz, {N_REPEATS} repeats (median); "
        f"MF backend = pyCBC FFT ({NUM_FFT_THREADS} threads)"
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
        "NUFFT resampling vs frequency-domain matched filtering: wall time\n"
        f"phase_model={PHASE_MODEL!r}, f0_setting={F0_SETTING!r}, "
        f"$F_S$={SAMPLE_RATE_HZ} Hz, {N_REPEATS} repeats (median)"
    )

    # --- top panel: absolute wall time ---
    ax = axes[0]
    ax.loglog(ns, t_nufft * 1e3, "o-", ms=6, lw=1.5, color="C0", label="NUFFT pipeline (4 NUFFTs)")
    ax.loglog(ns, t_mf * 1e3, "s-", ms=6, lw=1.5, color="C1", label="Freq-domain MF (4 pyCBC FFTs)")
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
