# Computational-cost sweeps

These scripts estimate the resource cost of the semicoherent search over a
two-dimensional grid of lower and upper frequency edges.  Grid points with
`f_end <= f_start` are excluded.

The fiducial configuration is the 40--64 Hz band used in the paper, with one
year of data and `N_beta = 144`.

## Scripts

- `nufft_cost_vs_band.py` evaluates
  `N_beta T_obs f_samp log(T_coh f_samp)`, calibrated to the quoted four-thread
  CPU throughput of `1e5` NUFFT input samples/s.  It shows both raw real-data
  Nyquist sampling (`f_samp = 2 f_end`) and complex baseband sampling
  (`f_samp = f_end - f_start`) and includes 10x and 100x GPU projections.
- `stackslide_cost_vs_band.py` evaluates power-read traffic and ideal
  bandwidth-limited CPU/GPU time, as well as resampled-spectrum memory, GPU
  block duration, and the required number of blocks.  The default StackSlide
  workload (`1e10` templates times `1e3` reads/template) is explicitly held
  fixed across the band sweep; only storage and blocking vary with bandwidth.
- `combined_cost_vs_band.py` puts the NUFFT timing, StackSlide timing, storage,
  and blocking estimates in one sweep for comparing end-to-end scenarios.
- `test_complex_baseband_speedup.py` directly benchmarks the current full-rate
  NUFFT against tau-domain heterodyning, polyphase decimation, and a smaller
  complex-baseband NUFFT. It reports transform-only and end-to-end timings so
  fixed FINUFFT and preprocessing overheads are visible.

Storage sizes and memory bandwidths use decimal units (`1 GB = 1e9 bytes`).
Power reads default to four bytes (single precision), while stored spectra
default to two bytes per bin (half precision).

Run the default sweeps from `paper_plots` with:

```bash
conda run -n PBH python computational_cost/nufft_cost_vs_band.py
conda run -n PBH python computational_cost/stackslide_cost_vs_band.py
conda run -n PBH python computational_cost/combined_cost_vs_band.py
conda run -n PBH python computational_cost/test_complex_baseband_speedup.py
```

Each script exposes its assumptions through command-line options (`--help`)
and writes publication-style raster figures to `computational_cost/figs/` by
default. CSV and PDF saving are currently commented out. The scripts are
analytical projections, not direct hardware benchmarks.
