# CODEX.md

## Project Overview

This repository is a research codebase for gravitational-wave data analysis. The active work is in `Resampling/paper_plots/`.

The underlying search idea targets sources with a prescribed phase evolution, remaps the data into a nonuniform time coordinate `tau`, and uses a non-uniform FFT (via `finufft`) to recover signals that become approximately monochromatic in that resampled coordinate.

The current science focus in the active code is PBH-inspired/chirping signals:

- model a source phase evolution in the time domain
- choose `tau` so that `phi(t) ~= omega0 * tau`
- run a NUFFT on samples at nonuniform `tau`

In other words, this is not a generic data-processing repo: it is a gravitational-wave search prototype built around phase-demodulation, nonuniform Fourier methods, and semicoherent combination, with the current active work focused on paper-ready demonstrations and sensitivity/modeling calculations.


## One-Line Summary

This project develops a gravitational-wave search pipeline that targets long-duration (longer than 10s of seconds) signals with a known phase evolution by analyzing it semicoherently, demodulating individual chunks with a NUFFT then semicoherently combining different chunks. 

## Codex Operating Notes

- At the start of each session, read the last 5 entries in the repository-root `DIARY.json` if the file exists.
- After substantive file changes, add a brief dated note to the repository-root `DIARY.json` describing what changed. Skip diary updates for read-only inspections or purely conversational work. Keep entries compact.
- Choose the default editing target from the task: use `Resampling/Nov2025/` for core resampling, 5-vector, and search-pipeline behavior; use `Resampling/paper_plots/` for paper figures, demonstrations, and supporting calculations.
- Treat `Resampling/paper_plots/` as paper-facing code built on top of the core implementation, not as the default location for shared pipeline changes.
- Read the relevant tests before changing core resampling or 5-vector behavior in `Resampling/Nov2025/`.
- Use `rg`/`rg --files` for repository search.
- Keep scratch scripts outside the repository, preferably under `/tmp/`.
- Preserve user or local working-tree changes. Do not clean generated caches, plots, notebooks, or unrelated files unless asked.
- Many analysis files are scripts that generate figures. Avoid changing saved figures in `figs/` unless that is part of the requested work.
- For edits to Jupytext-backed analysis files, preserve the paired `py:percent` structure and notebook metadata.


## Where To Start

- `Resampling/Nov2025/`
  Core implementation and regression-test reference for the resampler, 5-vector model, synthetic-signal utilities, and current search-pipeline behavior.
- `Resampling/paper_plots/`
  Active paper-facing working area for current figures, demonstrations, and supporting calculations.
- `Resampling/paper_plots/resampling/`
  Current resampling demonstrations and NUFFT/stroboscopic comparison scripts.
- `Resampling/paper_plots/signal_model/`
  Current signal-model comparison and post-Newtonian validity scripts.
- `Resampling/paper_plots/distance_sensitivity/`
  Current sensitivity and distance-reach calculations.


## Environment

The repository includes a Conda environment named `PBH` (`environment.yml`). Run all commands inside it via `conda run -n PBH ...` (e.g. `conda run -n PBH python script.py`, `conda run -n PBH pytest ...`). Do **not** use `conda activate PBH` or `conda init` -- the non-interactive shell used by agents is not configured for activation, and `conda run` is the reliable way to dispatch into the environment.

For ad-hoc scripts and scratch experiments, write the file to `/tmp/` and execute it from there (e.g. `conda run -n PBH python /tmp/scratch.py`) rather than creating files inside the repo. This keeps the working tree clean of throwaway artifacts.


## Finding Zotero Papers

Many papers are stored under `/home/neil-lu/Zotero/storage`, which is Zotero's `storage/` folder. Subdirectory names such as `HGY2X65J` are Zotero attachment item keys, not hashes or title-derived names.

When you know all or part of a Zotero item title, query `/home/neil-lu/Zotero/zotero.sqlite` to resolve the attachment path:

```bash
python3 - 'search terms here' <<'PY'
import os
import sqlite3
import sys

term = sys.argv[1].lower()
db = '/home/neil-lu/Zotero/zotero.sqlite'
storage = '/home/neil-lu/Zotero/storage'

conn = sqlite3.connect(f'file:{db}?mode=ro', uri=True)
cur = conn.cursor()

query = """
SELECT title.value, att.key, ia.path
FROM items parent
JOIN itemData d ON d.itemID = parent.itemID
JOIN fields f ON f.fieldID = d.fieldID AND f.fieldName = 'title'
JOIN itemDataValues title ON title.valueID = d.valueID
JOIN itemAttachments ia ON ia.parentItemID = parent.itemID
JOIN items att ON att.itemID = ia.itemID
WHERE lower(title.value) LIKE ?
  AND ia.path LIKE 'storage:%'
ORDER BY title.value
"""

for title, key, path in cur.execute(query, (f'%{term}%',)):
    filename = path.removeprefix('storage:')
    print(title)
    print(os.path.join(storage, key, filename))
    print()
PY
```

If the live database is locked because Zotero is open, use `/home/neil-lu/Zotero/zotero.sqlite.bak` or close Zotero first. Do not edit Zotero's SQLite database directly.


## Practical Guidance For Codex

- Run paper-plot scripts from the repository root with `conda run -n PBH python Resampling/paper_plots/.../script.py` unless the script itself documents a different working directory.
- Some Python plotting scripts may include `pl.show()`/`plt.show()`, which can block termination when run non-interactively. When running scripts from the CLI for verification, disable or guard interactive `show()` calls so the process exits after saving/checking figures.
- For current paper workflow changes, verify the specific edited script where practical and check that expected outputs under the local `figs/` directory are produced.
- When you need current paper behavior, read the relevant script in `Resampling/paper_plots/` first, then trace imports into `Resampling/Nov2025/` only as needed.
- When changing shared resampling or 5-vector internals, use `conda run -n PBH pytest Resampling/Nov2025/test_suite -q` for the most relevant verification pass.
- Be careful with imports: some analysis files are written to run as local scripts, while tests import them as package modules.
- Avoid spending time cleaning `__pycache__`, plot outputs, or old exploratory notebooks unless the user asks.
- Do not treat sparse top-level docs as authoritative; `Resampling/Nov2025/` is the best source of truth for core pipeline behavior, while `Resampling/paper_plots/` is the best source of truth for current paper-facing scripts and calculations.


## Repo Layout

- `Resampling/Nov2025/` is the main active area for core resampling, 5-vector, and search-pipeline implementation.
- `Resampling/paper_plots/` is the main active paper-facing implementation and analysis area.
  - `resampling/`: resampling examples, NUFFT/stroboscopic comparisons, and related generated figures.
  - `signal_model/`: 0PN/1PN/TaylorF2/3.5PN comparisons, dephasing plots, and paper text snippets.
  - `distance_sensitivity/`: sensitivity, distance-reach, and time-frequency integral calculations.
- `Resampling/NUFFT/`, `Resampling/5-vector/`, `Resampling/5_vec+doppler/`, `Resampling/Template_grid/`, and related notebooks are valuable historical/exploratory references, but many are prototypes rather than hardened library code.
- `Resampling/Defunct/` is archival.
- Top-level notebooks and plotting artifacts are mostly exploratory analysis outputs.

Prefer making changes in the directory that owns the requested behavior: `Resampling/Nov2025/` for shared pipeline internals, and `Resampling/paper_plots/` for paper-specific scripts and calculations.


## Scientific And Coding Conventions

- Internal frequencies in the active resampler code are usually angular frequencies in radians per second, not Hz. Check carefully before changing formulas.
- Paper-plot scripts often present frequencies in Hz while calling into older internals. Verify unit conversions at boundaries between `paper_plots` scripts and `Nov2025` code.
- The resampled coordinate `tau` is constructed from the assumed phase model. Small sign or normalization mistakes can silently break recovery.
- The 5-vector logic assumes sidereal sidebands at offsets of `0, +/- 1/day, +/- 2/day` around the carrier.
- Several validation checks use Dirichlet-kernel/bin-mismatch corrections when the recovered carrier does not land exactly on a Fourier bin.
- Tests are stochastic and use random source/geometry parameters, so keep tolerances and statistical intent intact unless you are deliberately redesigning them.


## Notebooks And Paired Files

Some analysis scripts, including files under `Resampling/paper_plots/`, use Jupytext `py:percent` structure. `Resampling/Nov2025/jupytext.toml` also pairs notebooks with `py:percent` files. If you edit notebook-backed analysis, preserve the Jupytext cell markers and metadata instead of converting it into plain script format.
