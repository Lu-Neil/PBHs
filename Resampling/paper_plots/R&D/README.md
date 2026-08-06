# Scientific research harness

This directory contains the PBH semicoherent-search research workflow. The
harness runs a strict task graph, preserves every attempt and model transcript,
validates declared outputs, stops on failed required work, and can resume or
selectively rerun work.

The workflow is:

```text
planner
  → literature
  → theory_and_optimization
  → numerical_experiment
  → results_analysis
  → referee
  → synthesized report
```

`idea.md` contains the research question. `agents.yml` is the versioned task
contract. `run_graph.py` is the runner.

`--smoke-test` uses `gpt-5.4-mini` by default; set `SMOKE_MODEL` to override it.

## Before spending model calls

Run these commands from this `R&D` directory:

```bash
conda run -n PBH python run_graph.py \
    --out runs/preview \
    --dry-run
```

The output directory must be new and empty. A dry run creates only planning
metadata and an event record; it makes no Codex calls.

Validate the Python implementation:

```bash
conda run -n PBH pytest test_run_graph.py -q
```

## Start a complete run

```bash
conda run -n PBH python run_graph.py --out runs/iter_002
```

The configured validation mode is `hybrid`: deterministic checks run first and
an independent Codex rubric checks scientific content. To run only
deterministic checks:

```bash
conda run -n PBH python run_graph.py \
    --validation-mode deterministic \
    --out runs/iter_002_deterministic
```

## Resume interrupted work

```bash
conda run -n PBH python run_graph.py --resume runs/iter_002
```

Accepted tasks are retained. Pending, interrupted, rejected, or blocked tasks
are reconsidered, together with their downstream dependants. The effective
configuration must match the configuration that started the run.

## Run only part of the workflow

`--select` runs the requested task and all prerequisites:

```bash
conda run -n PBH python run_graph.py \
    --select numerical_experiment \
    --out runs/dimension_experiment
```

## Start a new iteration from accepted work

`--rerun` requires a completed or partial v2 run supplied by `--reuse-from`.
The requested task and all downstream dependants are rerun. Unaffected accepted
artifacts are referenced directly from the earlier run.

```bash
conda run -n PBH python run_graph.py \
    --reuse-from runs/iter_002 \
    --rerun theory_and_optimization \
    --previous-report runs/iter_002/report.md \
    --feedback feedback/iter_003_comments.md \
    --out runs/iter_003
```

Runs produced by the older harness have metadata schema v1 and cannot be
resumed or reused automatically. They remain usable as `--previous-report`
context.

## Run artifacts

Each run contains:

```text
metadata.json       continuously updated task and run state
events.jsonl        timestamped state transitions
report.md           synthesis that labels failed and incomplete work
_prompts/           exact prompts
_logs/              Codex JSONL transcripts and stderr
_plans/             per-attempt planning artifacts
_attempts/          preserved primary artifacts for every attempt
_validation/        structured and human-readable validator reports
experiment/         script, parameters, results, commands, logs, and plot
```

Task states are `pending`, `running`, `accepted`, `reused`, `warning`,
`rejected`, `blocked`, or `skipped`. Downstream work may consume only
`accepted`, `reused`, or explicitly warning-level tasks.

## Failure behaviour

Required tasks use `on_failure: stop`. When they exhaust their retry budget,
the run is marked failed and downstream tasks are blocked. The referee is
configured as `continue_with_warning`, because a weak referee report should be
visible but should not erase otherwise accepted research artifacts.

Retries occur only after a failed validator, runtime failure, or an explicitly
configured minimum-attempt policy. The project default is one minimum attempt.

## Numerical experiment safety and provenance

The experiment task may inspect repository code and execute commands. It is
instructed not to modify repository source files and to write generated
artifacts only inside the run directory or `/tmp`. Python commands must use:

```bash
conda run -n PBH ...
```

The exact Codex event stream is preserved even if the agent fails to write its
own command log. Required scripts, JSON parameters, CSV results, logs, and plot
are independently checked before the task can pass.
