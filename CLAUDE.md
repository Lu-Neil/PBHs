# AGENTS.md

## Project overview

This repository is a research codebase for gravitational-wave data analysis. The main active workflow is a signal-search pipeline that targets sources with a prescribed phase evolution, remaps the data into a nonuniform time coordinate `tau`, and uses a non-uniform FFT (via `finufft`) to recover signals that become approximately monochromatic in that resampled coordinate.

The current science focus in the active code is PBH-inspired/chirping signals plus sidereal detector modulation:

- model a source phase evolution in the time domain
- choose `tau` so that `phi(t) ~= omega0 * tau`
- run a NUFFT on samples at nonuniform `tau`
- extract the carrier and the four sidereal sidebands as a 5-vector
- match against detector-response templates to estimate signal amplitude and polarization content

In other words, this is not a generic data-processing repo: it is a gravitational-wave search prototype built around phase-demodulation, nonuniform Fourier methods, and 5-vector signal reconstruction.

## Where to start

- `Resampling/Nov2025/resampler.py`
  Core `Resampler` class. Wraps `finufft` and computes spectra on a resampled time axis.
- `Resampling/Nov2025/five_vec.py`
  Implements the 5-vector detector-response model and sidereal modulation quantities.
- `Resampling/Nov2025/fiveVec_resampler_utils.py`
  Best end-to-end reference for the current pipeline: synthetic signal generation, PBH chirps, gaps, template construction, and estimators.
- `Resampling/Nov2025/test_suite/`
  Most useful regression tests for understanding intended behavior.
- `Resampling/Nov2025/PBH-5vec.py` and `Resampling/Nov2025/PBH-5vec_gaps.py`
  Jupytext-backed analysis scripts that show the intended scientific workflow.

## Repo layout

- `Resampling/Nov2025/` is the main active implementation area.
- `Resampling/NUFFT/`, `Resampling/5-vector/`, `Resampling/5_vec+doppler/`, `Resampling/Template_grid/`, and related notebooks are valuable historical/exploratory references, but many are prototypes rather than hardened library code.
- `Resampling/Defunct/` is archival.
- Top-level notebooks and plotting artifacts are mostly exploratory analysis outputs.

Prefer making changes in `Resampling/Nov2025/` unless the task explicitly targets an older branch of the work.

## Scientific and coding conventions

- Internal frequencies in the active resampler code are usually angular frequencies in radians per second, not Hz. Check carefully before changing formulas.
- The resampled coordinate `tau` is constructed from the assumed phase model. Small sign or normalization mistakes can silently break recovery.
- The 5-vector logic assumes sidereal sidebands at offsets of `0, +/- 1/day, +/- 2/day` around the carrier.
- Several validation checks use Dirichlet-kernel/bin-mismatch corrections when the recovered carrier does not land exactly on a Fourier bin.
- Tests are stochastic and use random source/geometry parameters, so keep tolerances and statistical intent intact unless you are deliberately redesigning them.

## Notebooks and paired files

`Resampling/Nov2025/jupytext.toml` pairs notebooks with `py:percent` files. If you edit a notebook-backed analysis in that directory, preserve the Jupytext structure instead of converting it into plain script format.

## Environment

The repository includes a Conda environment named `PBH` (`environment.yml`). When working on this project, activate it with `conda activate PBH`.

## Practical guidance for agents

- When you need the current pipeline behavior, read the tests before changing the implementation.
- Use `pytest Resampling/Nov2025/test_suite -q` for the most relevant verification pass.
- Be careful with imports: some analysis files are written to run as local scripts, while tests import them as package modules.
- Avoid spending time cleaning `__pycache__`, plot outputs, or old exploratory notebooks unless the user asks.
- Do not treat sparse top-level docs as authoritative; the code in `Resampling/Nov2025/` is the best source of truth.

## One-line summary

This project develops a gravitational-wave search pipeline that targets signals with a chosen phase evolution, demodulates them through nonuniform resampling, and uses a NUFFT plus 5-vector sidereal template matching to identify and reconstruct those signals.
