# CODEX.md

## Project Overview

This repository is a research codebase for gravitational-wave data analysis. The main active workflow is currently the paper-plot and supporting-calculation work in `Resampling/paper_plots/`. Those scripts draw on the older `Resampling/Nov2025/` resampling and 5-vector implementation where needed.

The underlying search idea targets sources with a prescribed phase evolution, remaps the data into a nonuniform time coordinate `tau`, and uses a non-uniform FFT (via `finufft`) to recover signals that become approximately monochromatic in that resampled coordinate.

The current science focus in the active code is PBH-inspired/chirping signals plus sidereal detector modulation:

- model a source phase evolution in the time domain
- choose `tau` so that `phi(t) ~= omega0 * tau`
- run a NUFFT on samples at nonuniform `tau`
- extract the carrier and the four sidereal sidebands as a 5-vector
- match against detector-response templates to estimate signal amplitude and polarization content

In other words, this is not a generic data-processing repo: it is a gravitational-wave search prototype built around phase-demodulation, nonuniform Fourier methods, and 5-vector signal reconstruction, with the current active work focused on paper-ready demonstrations and sensitivity/modeling calculations.

## Codex Operating Notes

- At the start of each session, read the last two days of notes in `DIARY.json` if the file exists.
- When making changes, add a brief dated note to `DIARY.json` describing what changed. Keep entries compact.
- Prefer changes in `Resampling/paper_plots/`, which is now the primary active working area, unless the user explicitly targets older pipeline code.
- Treat `Resampling/Nov2025/` as an implementation and regression-test reference for the current paper scripts, not as the default editing target.
- Read the relevant tests before changing core resampling or 5-vector behavior in `Resampling/Nov2025/`.
- Use `rg`/`rg --files` for repository search.
- Keep scratch scripts outside the repository, preferably under `/tmp/`.
- Preserve user or local working-tree changes. Do not clean generated caches, plots, notebooks, or unrelated files unless asked.
- Many analysis files are scripts that generate figures. Avoid changing saved figures in `figs/` unless that is part of the requested work.
- For edits to Jupytext-backed analysis files, preserve the paired `py:percent` structure and notebook metadata.

## Where To Start

- `Resampling/paper_plots/`
  Primary active working area for current paper figures, demonstrations, and supporting calculations.
- `Resampling/paper_plots/resampling/`
  Current resampling demonstrations and NUFFT/stroboscopic comparison scripts.
- `Resampling/paper_plots/signal_model/`
  Current signal-model comparison and post-Newtonian validity scripts.
- `Resampling/paper_plots/distance_sensitivity/`
  Current sensitivity and distance-reach calculations.
- `Resampling/Nov2025/`
  Useful implementation and regression-test reference for the underlying resampler and 5-vector pipeline, but no longer the primary working folder.

## Repo Layout

- `Resampling/paper_plots/` is the main active implementation and analysis area.
  - `resampling/`: resampling examples, NUFFT/stroboscopic comparisons, and related generated figures.
  - `signal_model/`: 0PN/1PN/TaylorF2/3.5PN comparisons, dephasing plots, and paper text snippets.
  - `distance_sensitivity/`: sensitivity, distance-reach, and time-frequency integral calculations.
- `Resampling/Nov2025/` remains a valuable implementation and test reference for the resampler and 5-vector pipeline.
- `Resampling/NUFFT/`, `Resampling/5-vector/`, `Resampling/5_vec+doppler/`, `Resampling/Template_grid/`, and related notebooks are valuable historical/exploratory references, but many are prototypes rather than hardened library code.
- `Resampling/Defunct/` is archival.
- Top-level notebooks and plotting artifacts are mostly exploratory analysis outputs.

Prefer making changes in `Resampling/paper_plots/` unless the task explicitly targets another branch of the work.

## Scientific And Coding Conventions

- Internal frequencies in the active resampler code are usually angular frequencies in radians per second, not Hz. Check carefully before changing formulas.
- Paper-plot scripts often present frequencies in Hz while calling into older internals. Verify unit conversions at boundaries between `paper_plots` scripts and `Nov2025` code.
- The resampled coordinate `tau` is constructed from the assumed phase model. Small sign or normalization mistakes can silently break recovery.
- The 5-vector logic assumes sidereal sidebands at offsets of `0, +/- 1/day, +/- 2/day` around the carrier.
- Several validation checks use Dirichlet-kernel/bin-mismatch corrections when the recovered carrier does not land exactly on a Fourier bin.
- Tests are stochastic and use random source/geometry parameters, so keep tolerances and statistical intent intact unless you are deliberately redesigning them.

## Notebooks And Paired Files

Some analysis scripts, including files under `Resampling/paper_plots/`, use Jupytext `py:percent` structure. `Resampling/Nov2025/jupytext.toml` also pairs notebooks with `py:percent` files. If you edit notebook-backed analysis, preserve the Jupytext cell markers and metadata instead of converting it into plain script format.

## Environment

The repository includes a Conda environment named `PBH` (`environment.yml`). Run all commands inside it via `conda run -n PBH ...` (e.g. `conda run -n PBH python script.py`, `conda run -n PBH pytest ...`). Do **not** use `conda activate PBH` or `conda init` -- the non-interactive shell used by agents is not configured for activation, and `conda run` is the reliable way to dispatch into the environment.

For ad-hoc scripts and scratch experiments, write the file to `/tmp/` and execute it from there (e.g. `conda run -n PBH python /tmp/scratch.py`) rather than creating files inside the repo. This keeps the working tree clean of throwaway artifacts.

## Practical Guidance For Codex

- Run paper-plot scripts from the repository root with `conda run -n PBH python Resampling/paper_plots/.../script.py` unless the script itself documents a different working directory.
- For current paper workflow changes, verify the specific edited script where practical and check that expected outputs under the local `figs/` directory are produced.
- When you need current paper behavior, read the relevant script in `Resampling/paper_plots/` first, then trace imports into `Resampling/Nov2025/` only as needed.
- When changing shared resampling or 5-vector internals, use `conda run -n PBH pytest Resampling/Nov2025/test_suite -q` for the most relevant verification pass.
- Be careful with imports: some analysis files are written to run as local scripts, while tests import them as package modules.
- Avoid spending time cleaning `__pycache__`, plot outputs, or old exploratory notebooks unless the user asks.
- Do not treat sparse top-level docs as authoritative; `Resampling/paper_plots/` is the best source of truth for current work, while `Resampling/Nov2025/` remains the best reference for older core pipeline internals.

## One-Line Summary

This project develops a gravitational-wave search pipeline that targets signals with a chosen phase evolution, demodulates them through nonuniform resampling, and uses a NUFFT plus 5-vector sidereal template matching to identify and reconstruct those signals.

---

## Pipeline Development Roadmap

The items below represent the major open tasks needed to turn the prototype into a deployable search pipeline. **This roadmap is planning context, not a statement that `Resampling/Nov2025/` is the active working area. Current implementation and paper-facing work should still start in `Resampling/paper_plots/` unless the user says otherwise.**

### 1. NUFFT noise transfer function (analytical + numerical)

The NUFFT on non-uniform `tau` does not preserve the input noise PSD. By a stationary-phase argument, the expected NUFFT PSD at output frequency `f_out` is:

```text
<S_NUFFT(f_out)> = (1/T) integral_0^T S_n( f_out * tau'(t) ) dt
```

where `tau'(t) = d tau / dt = (1 - 8 beta t / 3)^(-3/8)` -- the time-average of `S_n` along the instantaneous frequency track. This formula is derived in `Resampling/Nov2025/demonstrating_behaviour/nufft_noise_psd.py`, which also compares FFT and NUFFT PSDs numerically using bilby H1 design noise.

Key practical consequence: you cannot use `S_n(f0)` alone to normalise the detection statistic -- the effective noise depends on the whole chirp track.

### 2. Noise-weighted matched filter

The current `_estimator` and `_joint_estimator` in `fiveVec_resampler_utils.py` assume equal, uncorrelated noise across all 5 sidereal bins. The optimal estimator is `X dot (C^-1 A) / (A^dagger C^-1 A)` where `C` is the 5x5 noise covariance matrix of the extracted 5-vector. Measuring `C` from signal-free data, or predicting it from the transfer function above, is needed before the detection statistic can be properly calibrated.

### 3. Coloured Gaussian noise injections

Inject synthetic signals into coloured noise drawn from a known PSD, for example bilby H1 design. Verify that the recovered SNR matches the theoretical prediction from the noise transfer function formula.

### 4. Detection statistic: absolute SNR normalisation and threshold setting

The current `_detection_stat` in `fiveVec_resampler_utils.py` is a power-based quantity, not yet normalised against the noise floor. Needed:

- SNR definition: `rho^2 = Lambda / sigma^2_noise`
- Distribution of the statistic under `H0` (noise only) for a given template bank size
- FAP threshold for a target false alarm probability

The behaviour of the detection statistic, as opposed to recovered power, versus `Delta beta` is already plotted numerically in `demonstrating_behaviour/delta_beta_behaviour.py`, but a theoretical model for its degradation analogous to the tail-power formula is still missing.

### 5. Template bank construction

The `Delta beta` crossover condition gives the beta spacing; the NUFFT bin width gives the `f0` spacing. Neither has been turned into an explicit covering algorithm. Needed: parameter-space bounds (`f0` range, `Mc` range to beta range), maximum mismatch tolerance, and a total template count estimate.

### 6. Real GW data infrastructure

- GWpy-based data reading and segment queries
- Bandpass and downsample to the target frequency band
- Science-mode / data-quality segment handling
- Calibration line awareness

### 7. Signal injections into real detector noise

End-to-end test: inject a synthetic PBH chirp into real H1/L1 noise and verify detection.

### 8. Multi-detector combination

Each detector contributes its own `A_p`, `A_c` with its own `(lat, lng, az)`. The 5-vectors can be combined with noise weighting. Helps with glitch rejection and sky localisation.

### 9. Glitch rejection

Real noise is non-stationary. A chi-squared consistency test across the 5 sidereal bins, or a time-frequency excess-power veto before resampling, is needed to suppress glitch candidates.

### 10. Post-detection parameter estimation

After a candidate: estimate `(Mc, f0, beta, ra, dec, eta, psi)`. The 5-vector amplitude ratios encode polarisation state; sidereal phases encode sky position; `beta = const * f0^(8/3) * Mc^(5/3)` gives chirp mass once `f0` is measured.

---

## bilby Usage Notes

`bilby` is available in the `PBH` environment and is used for generating simulated detector noise:

```python
import bilby
bilby.core.utils.logger.setLevel("WARNING")

ifo = bilby.gw.detector.InterferometerList(["H1"])[0]
ifo.set_strain_data_from_power_spectral_density(
    sampling_frequency=F_S, duration=T_OBS, start_time=-T_OBS / 2
)
strain = ifo.strain_data.time_domain_strain   # real numpy array
```

Constraints:

- `sampling_frequency * duration` must be an integer; bilby enforces this strictly.
- The H1 design PSD is finite only above roughly 10 Hz; below that bilby sets frequency bins to zero, so `F_S = 1 Hz` (Nyquist = 0.5 Hz) produces all-zero strain.
- For demonstrations targeting the H1 sensitive band, use `F_S >= 64 Hz` so the band 10-32 Hz is accessible.
