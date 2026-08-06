"""Run an auditable, resumable scientific-research workflow.

The workflow is configured by ``agents.yml`` but is task-oriented: every task
declares dependencies, output files, validators, retry behaviour, and failure
semantics.  A run preserves prompts, Codex JSONL transcripts, stderr, attempts,
validation reports, events, and a continuously updated metadata file.

Typical commands
----------------

Start a new run::

    conda run -n PBH python run_graph.py --out runs/iter_002

Inspect the work without making model calls::

    conda run -n PBH python run_graph.py --out runs/preview --dry-run

Resume an interrupted run::

    conda run -n PBH python run_graph.py --resume runs/iter_002

Reuse accepted artifacts from an earlier v2 run and rerun one task plus all of
its downstream dependants::

    conda run -n PBH python run_graph.py \
        --reuse-from runs/iter_002 \
        --rerun theory_and_optimization \
        --feedback feedback/iter_003_comments.md \
        --out runs/iter_003

Run a selected task and its prerequisites::

    conda run -n PBH python run_graph.py \
        --select numerical_experiment \
        --out runs/experiment_only

``--smoke-test`` checks the complete orchestration with short artifacts and
deterministic validation.  It still makes one Codex call per selected task.
Unit tests do not make Codex calls.
"""

from __future__ import annotations

import argparse
import ast
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import platform
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "agents.yml"
SCHEMA_VERSION = 2
TERMINAL_SUCCESS = {"accepted", "reused", "warning"}
TERMINAL_STATES = TERMINAL_SUCCESS | {"rejected", "blocked", "skipped"}


# ---------------------------------------------------------------------------
# Strict configuration models
# ---------------------------------------------------------------------------


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class OutputSpec(StrictModel):
    path: str
    description: str = ""
    required: bool = True
    primary: bool = False
    kind: Literal["markdown", "json", "csv", "python", "log", "plot", "other"] = (
        "other"
    )


class ValidatorSpec(StrictModel):
    name: str
    type: Literal[
        "artifact_completeness",
        "citation_sanity",
        "mathematical_sanity",
        "research_value",
        "experiment_artifacts",
        "referee_quality",
        "codex_rubric",
    ]
    options: dict[str, Any] = Field(default_factory=dict)


class RetrySpec(StrictModel):
    enabled: bool = True
    min_attempts: int = Field(default=1, ge=1)
    max_attempts: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def valid_bounds(self) -> "RetrySpec":
        if self.min_attempts > self.max_attempts:
            raise ValueError("min_attempts cannot exceed max_attempts")
        return self


class PlanningSpec(StrictModel):
    enabled: bool = True
    instructions: list[str] = Field(default_factory=list)
    model: str | None = None


class TaskSpec(StrictModel):
    purpose: str
    kind: Literal[
        "planning",
        "literature",
        "theory",
        "experiment",
        "analysis",
        "referee",
    ]
    depends_on: list[str] = Field(default_factory=list)
    outputs: list[OutputSpec]
    validators: list[ValidatorSpec]
    must_include: list[str] = Field(default_factory=list)
    rules: list[str] = Field(default_factory=list)
    cautions: list[str] = Field(default_factory=list)
    planning: PlanningSpec | None = None
    retry: RetrySpec | None = None
    model: str | None = None
    on_failure: Literal["stop", "continue_with_warning"] = "stop"

    @model_validator(mode="after")
    def exactly_one_primary(self) -> "TaskSpec":
        primaries = [output for output in self.outputs if output.primary]
        if len(primaries) != 1:
            raise ValueError("each task must declare exactly one primary output")
        if not primaries[0].path.endswith(".md"):
            raise ValueError("the primary output must be a Markdown file")
        return self


class RuntimeSpec(StrictModel):
    command_prefix: list[str] = Field(
        default_factory=lambda: ["conda", "run", "-n", "PBH"]
    )
    scratch_root: str = "/tmp"
    codex_timeout_seconds: int = Field(default=1800, ge=1)
    codex_sandbox: Literal["read-only", "workspace-write", "danger-full-access"] = (
        "workspace-write"
    )


class ValidationSpec(StrictModel):
    mode: Literal["deterministic", "hybrid", "codex"] = "hybrid"
    model: str | None = None


class DefaultsSpec(StrictModel):
    retry: RetrySpec = Field(default_factory=RetrySpec)
    planning: PlanningSpec = Field(default_factory=PlanningSpec)


class PipelineSpec(StrictModel):
    order: list[str]


class HarnessConfig(StrictModel):
    schema_version: Literal[2]
    global_rules: list[str] = Field(default_factory=list)
    runtime: RuntimeSpec = Field(default_factory=RuntimeSpec)
    validation: ValidationSpec = Field(default_factory=ValidationSpec)
    defaults: DefaultsSpec = Field(default_factory=DefaultsSpec)
    pipeline: PipelineSpec
    tasks: dict[str, TaskSpec]

    @model_validator(mode="after")
    def valid_graph(self) -> "HarnessConfig":
        order = self.pipeline.order
        if not order:
            raise ValueError("pipeline.order cannot be empty")
        if len(order) != len(set(order)):
            raise ValueError("pipeline.order contains duplicate tasks")
        if set(order) != set(self.tasks):
            missing = sorted(set(self.tasks) - set(order))
            unknown = sorted(set(order) - set(self.tasks))
            raise ValueError(
                f"pipeline.order/tasks mismatch; missing={missing}, unknown={unknown}"
            )

        position = {name: index for index, name in enumerate(order)}
        output_paths: set[str] = set()
        for name, task in self.tasks.items():
            for dependency in task.depends_on:
                if dependency not in self.tasks:
                    raise ValueError(f"{name} has unknown dependency {dependency}")
                if position[dependency] >= position[name]:
                    raise ValueError(
                        f"{name} dependency {dependency} must occur earlier in pipeline.order"
                    )
            for output in task.outputs:
                if output.path in output_paths:
                    raise ValueError(f"duplicate output path: {output.path}")
                output_paths.add(output.path)
        return self


# ---------------------------------------------------------------------------
# Data and filesystem helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_from_root(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def read_text(path: str | Path) -> str:
    candidate = resolve_from_root(path)
    if not candidate.exists():
        return f"[Missing file: {candidate}]"
    return candidate.read_text(encoding="utf-8")


def read_optional(path: str | Path | None) -> str | None:
    return None if path is None else read_text(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def load_config(path: str | Path) -> HarnessConfig:
    config_path = resolve_from_root(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing harness config: {config_path}")
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    return HarnessConfig.model_validate(raw)


def config_sha256(config: HarnessConfig) -> str:
    canonical = json.dumps(config.model_dump(mode="json"), sort_keys=True)
    return sha256_text(canonical)


def primary_output(task: TaskSpec) -> OutputSpec:
    return next(output for output in task.outputs if output.primary)


def task_output_paths(task: TaskSpec, out_dir: Path) -> dict[str, Path]:
    return {output.path: out_dir / output.path for output in task.outputs}


def effective_retry(config: HarnessConfig, task: TaskSpec) -> RetrySpec:
    return task.retry or config.defaults.retry


def effective_planning(config: HarnessConfig, task: TaskSpec) -> PlanningSpec:
    return task.planning or config.defaults.planning


class EventRecorder:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, event: str, **payload: Any) -> None:
        record = {"timestamp": utc_now(), "event": event, **payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")


@dataclass
class InvocationResult:
    text: str
    model: str | None
    duration_seconds: float
    prompt_sha256: str
    transcript_path: str
    stderr_path: str
    usage: dict[str, Any] | None
    session_id: str | None
    command: list[str]


def find_codex_binary() -> str:
    configured = os.environ.get("CODEX_BIN", "codex")
    resolved = shutil.which(configured)
    if resolved is None:
        raise RuntimeError(f"Could not find Codex CLI binary: {configured!r}")
    return resolved


def walk_json(value: Any):
    yield value
    if isinstance(value, dict):
        for child in value.values():
            yield from walk_json(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_json(child)


def parse_transcript_metadata(transcript: str) -> tuple[dict[str, Any] | None, str | None]:
    usage: dict[str, Any] | None = None
    session_id: str | None = None
    for line in transcript.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        for value in walk_json(event):
            if not isinstance(value, dict):
                continue
            if {"input_tokens", "output_tokens"} <= set(value):
                usage = {
                    key: value[key]
                    for key in (
                        "input_tokens",
                        "cached_input_tokens",
                        "output_tokens",
                        "reasoning_output_tokens",
                        "total_tokens",
                    )
                    if key in value
                }
            for key in ("thread_id", "session_id"):
                if isinstance(value.get(key), str):
                    session_id = value[key]
    return usage, session_id


def run_codex(
    prompt: str,
    output_path: Path,
    *,
    config: HarnessConfig,
    model: str | None,
    transcript_path: Path,
    stderr_path: Path,
    output_schema_path: Path | None = None,
    smoke_test: bool = False,
) -> InvocationResult:
    codex = find_codex_binary()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path.parent.mkdir(parents=True, exist_ok=True)

    sandbox = os.environ.get("CODEX_SANDBOX", config.runtime.codex_sandbox)
    timeout = int(
        os.environ.get(
            "CODEX_TIMEOUT_SECONDS",
            "180" if smoke_test else str(config.runtime.codex_timeout_seconds),
        )
    )
    selected_model = (
        model
        or (os.environ.get("SMOKE_MODEL", "gpt-5.4-mini") if smoke_test else None)
        or os.environ.get("RESEARCH_MODEL")
    )

    command = [
        codex,
        "exec",
        "--cd",
        str(ROOT),
        "--sandbox",
        sandbox,
        "--json",
        "--output-last-message",
        str(output_path),
        "--color",
        "never",
    ]
    command += ["--add-dir", str(output_path.parent)]
    profile = os.environ.get("CODEX_PROFILE")
    if selected_model:
        command += ["--model", selected_model]
    if profile:
        command += ["--profile", profile]
    if output_schema_path:
        command += ["--output-schema", str(output_schema_path)]
    command += ["-"]

    started = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            input=prompt,
            text=True,
            capture_output=True,
            cwd=ROOT,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else ""
        write_text(transcript_path, stdout)
        write_text(stderr_path, stderr)
        raise RuntimeError(
            f"Codex timed out after {timeout}s; transcript={transcript_path}"
        ) from exc

    duration = time.monotonic() - started
    write_text(transcript_path, completed.stdout)
    write_text(stderr_path, completed.stderr)

    if completed.returncode != 0:
        raise RuntimeError(
            f"Codex failed with exit code {completed.returncode}; "
            f"stderr={stderr_path}, transcript={transcript_path}"
        )
    if not output_path.exists():
        raise RuntimeError(f"Codex did not write its final message to {output_path}")

    usage, session_id = parse_transcript_metadata(completed.stdout)
    return InvocationResult(
        text=output_path.read_text(encoding="utf-8"),
        model=selected_model,
        duration_seconds=duration,
        prompt_sha256=sha256_text(prompt),
        transcript_path=str(transcript_path),
        stderr_path=str(stderr_path),
        usage=usage,
        session_id=session_id,
        command=command,
    )


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def dependency_context(
    config: HarnessConfig,
    task: TaskSpec,
    state: dict[str, Any],
) -> str:
    chunks: list[str] = []
    for dependency in task.depends_on:
        dependency_record = state["tasks"][dependency]
        dependency_spec = config.tasks[dependency]
        primary = primary_output(dependency_spec)
        path = dependency_record["outputs"].get(primary.path)
        if not path:
            chunks.append(f"# Dependency {dependency}\n\n[No accepted artifact]")
            continue
        chunks.append(
            f"# Dependency: {dependency}\n\n"
            f"Artifact: {path}\n\n"
            f"{read_text(path)}"
        )
    return "\n\n".join(chunks) or "[No task dependencies]"


def output_contract(task: TaskSpec, out_dir: Path) -> str:
    rows = []
    for output in task.outputs:
        path = out_dir / output.path
        rows.append(
            {
                "path": str(path),
                "required": output.required,
                "primary": output.primary,
                "kind": output.kind,
                "description": output.description,
            }
        )
    return yaml.safe_dump(rows, sort_keys=False).strip()


def iteration_context(
    previous_report: str | None,
    human_feedback: str | None,
) -> str:
    chunks: list[str] = []
    if previous_report:
        chunks.append(f"# Previous report\n\n{previous_report}")
    if human_feedback:
        chunks.append(
            "# Human feedback / new directions\n\n"
            f"{human_feedback}\n\n"
            "Address the feedback that is relevant to this task. Preserve supported "
            "results, but do not merely polish the previous wording."
        )
    return "\n\n".join(chunks) or "[No previous-run context]"


def build_planning_prompt(
    *,
    task_name: str,
    task: TaskSpec,
    config: HarnessConfig,
    idea: str,
    dependencies: str,
    previous_report: str | None,
    human_feedback: str | None,
    previous_output: str,
    previous_validation: str,
    attempt: int,
) -> str:
    planning = effective_planning(config, task)
    instructions = planning.instructions or [
        "Identify claims, calculations, or evidence the artifact must establish.",
        "List likely failure modes and the checks that would detect them.",
        "Describe concrete changes required from the previous attempt.",
    ]
    return f"""
You are planning task `{task_name}` in a scientific research workflow.

Purpose:
{task.purpose}

Attempt: {attempt}

Global rules:
{yaml.safe_dump(config.global_rules, sort_keys=False).strip()}

Required content:
{yaml.safe_dump(task.must_include, sort_keys=False).strip()}

Core research idea:
{idea}

Declared dependency artifacts:
{dependencies}

Previous-run context:
{iteration_context(previous_report, human_feedback)}

Previous artifact:
{previous_output if attempt > 1 else "[No previous attempt]"}

Previous validation feedback:
{previous_validation or "[No previous validation feedback]"}

Planning instructions:
{yaml.safe_dump(instructions, sort_keys=False).strip()}

Return a concise Markdown plan only. Do not perform the task yet. Do not invent
citations or results.
""".strip()


def build_task_prompt(
    *,
    task_name: str,
    task: TaskSpec,
    config: HarnessConfig,
    out_dir: Path,
    idea: str,
    dependencies: str,
    previous_report: str | None,
    human_feedback: str | None,
    previous_output: str,
    previous_validation: str,
    plan: str | None,
    attempt: int,
    smoke_test: bool,
) -> str:
    experiment_rules = ""
    if task.kind == "experiment":
        prefix = " ".join(config.runtime.command_prefix)
        experiment_rules = f"""
# Experiment execution rules

This is an execution task, not a proposal. You may inspect repository files, use
shell commands, and create the declared output artifacts. Do not edit the
repository's source code. Write generated scripts, data, logs, and plots only to
the declared paths or to `{config.runtime.scratch_root}`.

Run Python and tests through this configured prefix:

    {prefix}

Before finishing:
- execute the experiment rather than merely describing it;
- record commands in the declared command log;
- preserve parameters and random seeds;
- preserve stdout/stderr, including failures and warnings;
- make the primary report state what ran, what failed, and what was measured;
- create every required output in the output contract.

Return the primary Markdown report as the final response. The wrapper also saves
that response at the primary output path.
"""
    else:
        experiment_rules = """
# File/tool rules

This is a document task. You may inspect the repository and run read-only checks
when useful, but do not edit repository source files or create undeclared files.
Return the primary Markdown artifact as your final response; the wrapper saves it.
"""

    smoke_instruction = ""
    if smoke_test:
        smoke_instruction = (
            "\nSMOKE TEST: produce a short artifact under 250 words. For an "
            "experiment task, create tiny syntactically valid required artifacts "
            "with one data row and a non-empty placeholder plot file.\n"
        )

    return f"""
You are executing scientific workflow task `{task_name}`.

# Purpose

{task.purpose}

# Task kind and attempt

Kind: {task.kind}
Attempt: {attempt}

# Global rules

{yaml.safe_dump(config.global_rules, sort_keys=False).strip()}

# Task-specific rules

{yaml.safe_dump(task.rules, sort_keys=False).strip()}

# Cautions

{yaml.safe_dump(task.cautions, sort_keys=False).strip()}

# Output contract

{output_contract(task, out_dir)}

# Required content

{yaml.safe_dump(task.must_include, sort_keys=False).strip()}

# Core research idea

{idea}

# Declared dependency artifacts

{dependencies}

# Previous-run context

{iteration_context(previous_report, human_feedback)}

# Previous artifact

{previous_output if attempt > 1 else "[No previous attempt]"}

# Previous validation feedback

{previous_validation or "[No previous validation feedback]"}

# Plan for this attempt

{plan or "[Planning pass disabled]"}

{experiment_rules}

# Scientific reporting rules

Distinguish established evidence, derivations, assumptions, conjectures, and
open questions. Never invent citations, commands, numerical results, or tests.
If a required action fails, preserve the failure and explain it rather than
claiming success.
{smoke_instruction}
""".strip()


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------


def normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))


def contains_concept(text: str, concept: str) -> bool:
    words = [
        word
        for word in normalize_words(concept)
        if len(word) >= 4 and word not in {"with", "from", "that", "this"}
    ]
    artifact_words = normalize_words(text)
    needed = min(2, len(words))
    def present(word: str) -> bool:
        variants = {word, f"{word}s", f"{word}es"}
        if word.endswith("s"):
            variants.add(word[:-1])
        return bool(variants & artifact_words)

    return not words or sum(present(word) for word in words) >= needed


def result(
    spec: ValidatorSpec,
    passed: bool,
    feedback: str,
    **details: Any,
) -> dict[str, Any]:
    return {
        "name": spec.name,
        "type": spec.type,
        "passed": passed,
        "feedback": feedback,
        "details": details,
    }


def validate_artifact_completeness(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    missing_files: list[str] = []
    empty_files: list[str] = []
    hashes: dict[str, str | None] = {}
    for output in task.outputs:
        path = out_dir / output.path
        hashes[output.path] = sha256_file(path)
        if output.required and not path.exists():
            missing_files.append(output.path)
        elif output.required and path.is_file() and path.stat().st_size == 0:
            empty_files.append(output.path)

    primary_path = out_dir / primary_output(task).path
    text = read_text(primary_path) if primary_path.exists() else ""
    minimum_chars = int(spec.options.get("minimum_chars", 1200))
    too_short = len(text.strip()) < minimum_chars
    missing_concepts = [
        concept for concept in task.must_include if not contains_concept(text, concept)
    ]
    placeholders = [
        marker
        for marker in ("lorem ipsum", "[insert", "to be written")
        if marker in text.lower()
    ]
    passed = not (
        missing_files
        or empty_files
        or too_short
        or missing_concepts
        or placeholders
    )
    feedback_parts = []
    if missing_files:
        feedback_parts.append(f"Missing required files: {missing_files}")
    if empty_files:
        feedback_parts.append(f"Empty required files: {empty_files}")
    if too_short:
        feedback_parts.append(
            f"Primary artifact has {len(text.strip())} characters; "
            f"minimum is {minimum_chars}"
        )
    if missing_concepts:
        feedback_parts.append(f"Required concepts not evidenced: {missing_concepts}")
    if placeholders:
        feedback_parts.append(f"Placeholder text detected: {placeholders}")
    return result(
        spec,
        passed,
        "Passed completeness checks." if passed else "; ".join(feedback_parts),
        missing_files=missing_files,
        empty_files=empty_files,
        missing_concepts=missing_concepts,
        hashes=hashes,
    )


def validate_citation_sanity(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    text = read_text(out_dir / primary_output(task).path)
    patterns = {
        "doi": r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b",
        "arxiv": r"\barXiv:\s*\d{4}\.\d{4,5}\b",
        "url": r"https?://[^\s)>]+",
        "markdown_link": r"\[[^\]]+\]\([^)]+\)",
    }
    counts = {
        name: len(re.findall(pattern, text, flags=re.IGNORECASE))
        for name, pattern in patterns.items()
    }
    total = sum(counts.values())
    minimum = int(spec.options.get("minimum_reference_markers", 1))
    novelty_unhedged = bool(
        re.search(r"\b(first|novel|no prior work|has never been)\b", text, re.I)
        and not re.search(
            r"\b(possible|appears|may|uncertain|not established|requires verification)\b",
            text,
            re.I,
        )
    )
    passed = total >= minimum and not novelty_unhedged
    feedback = []
    if total < minimum:
        feedback.append(
            f"Found {total} resolvable citation markers; require at least {minimum}"
        )
    if novelty_unhedged:
        feedback.append("Potential novelty claim is not explicitly hedged")
    return result(
        spec,
        passed,
        "Passed citation-structure checks." if passed else "; ".join(feedback),
        reference_marker_counts=counts,
        novelty_unhedged=novelty_unhedged,
    )


def validate_mathematical_sanity(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    text = read_text(out_dir / primary_output(task).path)
    required_sections = spec.options.get(
        "required_concepts",
        ["assumptions", "units", "limiting case", "open questions"],
    )
    missing = [
        concept for concept in required_sections if not contains_concept(text, concept)
    ]
    requires_frequency_warning = bool(spec.options.get("frequency_units", False))
    frequency_ok = True
    if requires_frequency_warning:
        frequency_ok = bool(
            re.search(r"angular frequency|rad(?:ian)?s?\s*(?:per|/)\s*s", text, re.I)
            and re.search(r"\bHz\b|ordinary frequency", text, re.I)
        )
    passed = not missing and frequency_ok
    feedback = []
    if missing:
        feedback.append(f"Missing mathematical checks: {missing}")
    if not frequency_ok:
        feedback.append("Does not explicitly distinguish angular frequency from Hz")
    return result(
        spec,
        passed,
        "Passed deterministic mathematical checks." if passed else "; ".join(feedback),
        missing_concepts=missing,
        frequency_units_explicit=frequency_ok,
    )


def validate_research_value(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    text = read_text(out_dir / primary_output(task).path)
    needed = spec.options.get(
        "required_concepts",
        ["conclusion", "next steps", "unresolved", "test"],
    )
    missing = [concept for concept in needed if not contains_concept(text, concept)]
    repeated_idea_only = len(normalize_words(text) - normalize_words(read_text("idea.md"))) < int(
        spec.options.get("minimum_novel_vocabulary", 40)
    )
    passed = not missing and not repeated_idea_only
    feedback = []
    if missing:
        feedback.append(f"Missing research-value elements: {missing}")
    if repeated_idea_only:
        feedback.append("Artifact appears to add little beyond idea.md")
    return result(
        spec,
        passed,
        "Passed research-value checks." if passed else "; ".join(feedback),
        missing_concepts=missing,
        repeated_idea_only=repeated_idea_only,
    )


def validate_experiment_artifacts(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    checked: dict[str, Any] = {}
    outputs_by_kind: dict[str, list[Path]] = {}
    for output in task.outputs:
        outputs_by_kind.setdefault(output.kind, []).append(out_dir / output.path)

    for path in outputs_by_kind.get("python", []):
        if path.exists():
            try:
                ast.parse(path.read_text(encoding="utf-8"))
                checked[str(path)] = "valid Python syntax"
            except (SyntaxError, UnicodeDecodeError) as exc:
                errors.append(f"{path}: invalid Python: {exc}")

    for path in outputs_by_kind.get("json", []):
        if path.exists():
            try:
                json.loads(path.read_text(encoding="utf-8"))
                checked[str(path)] = "valid JSON"
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                errors.append(f"{path}: invalid JSON: {exc}")

    for path in outputs_by_kind.get("csv", []):
        if path.exists():
            try:
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.reader(handle))
                if len(rows) < 2 or not rows[0]:
                    errors.append(f"{path}: CSV needs a header and at least one data row")
                else:
                    checked[str(path)] = f"{len(rows) - 1} data rows"
            except (csv.Error, UnicodeDecodeError) as exc:
                errors.append(f"{path}: invalid CSV: {exc}")

    for kind in ("log", "plot"):
        for path in outputs_by_kind.get(kind, []):
            if path.exists() and path.stat().st_size == 0:
                errors.append(f"{path}: {kind} artifact is empty")
            elif path.exists():
                checked[str(path)] = f"non-empty {kind}"

    report = read_text(out_dir / primary_output(task).path)
    required_report_concepts = spec.options.get(
        "report_concepts", ["command", "parameters", "result", "failure"]
    )
    missing_report = [
        concept
        for concept in required_report_concepts
        if not contains_concept(report, concept)
    ]
    if missing_report:
        errors.append(f"Experiment report missing: {missing_report}")

    return result(
        spec,
        not errors,
        "Passed experiment artifact checks." if not errors else "; ".join(errors),
        checked=checked,
        errors=errors,
    )


def validate_referee_quality(
    spec: ValidatorSpec,
    task: TaskSpec,
    out_dir: Path,
) -> dict[str, Any]:
    text = read_text(out_dir / primary_output(task).path)
    required = spec.options.get(
        "required_concepts",
        ["objection", "decisive test", "severity", "artifact", "recommendation"],
    )
    missing = [concept for concept in required if not contains_concept(text, concept)]
    numbered_items = len(re.findall(r"(?m)^\s*(?:\d+\.|-)\s+", text))
    minimum_items = int(spec.options.get("minimum_concrete_items", 5))
    passed = not missing and numbered_items >= minimum_items
    feedback = []
    if missing:
        feedback.append(f"Missing referee elements: {missing}")
    if numbered_items < minimum_items:
        feedback.append(
            f"Found {numbered_items} concrete list items; require {minimum_items}"
        )
    return result(
        spec,
        passed,
        "Passed referee-structure checks." if passed else "; ".join(feedback),
        missing_concepts=missing,
        concrete_items=numbered_items,
    )


DETERMINISTIC_VALIDATORS = {
    "artifact_completeness": validate_artifact_completeness,
    "citation_sanity": validate_citation_sanity,
    "mathematical_sanity": validate_mathematical_sanity,
    "research_value": validate_research_value,
    "experiment_artifacts": validate_experiment_artifacts,
    "referee_quality": validate_referee_quality,
}


def codex_validation_schema(path: Path) -> Path:
    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "passed": {"type": "boolean"},
            "feedback": {"type": "string"},
            "failed_criteria": {"type": "array", "items": {"type": "string"}},
            "evidence": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["passed", "feedback", "failed_criteria", "evidence"],
    }
    write_json(path, schema)
    return path


def run_codex_rubric(
    *,
    validator: ValidatorSpec,
    task_name: str,
    task: TaskSpec,
    config: HarnessConfig,
    out_dir: Path,
    attempt: int,
    logs_dir: Path,
    smoke_test: bool,
) -> tuple[dict[str, Any], InvocationResult]:
    primary_path = out_dir / primary_output(task).path
    criteria = validator.options.get("criteria", [])
    prompt = f"""
You are an independent scientific validator. Assess task `{task_name}`.

Task purpose:
{task.purpose}

Required content:
{yaml.safe_dump(task.must_include, sort_keys=False).strip()}

Rubric criteria:
{yaml.safe_dump(criteria, sort_keys=False).strip()}

Artifact:
{read_text(primary_path)}

Pass only if every criterion is substantively satisfied. Keyword mentions are
not evidence. Identify exact failed criteria and cite artifact sections or
specific omissions. Return only JSON matching the supplied schema.
""".strip()
    prompt_path = logs_dir / f"{task_name}_attempt_{attempt}_validator_prompt.md"
    output_path = logs_dir / f"{task_name}_attempt_{attempt}_codex_validation.json"
    transcript_path = logs_dir / f"{task_name}_attempt_{attempt}_validator.jsonl"
    stderr_path = logs_dir / f"{task_name}_attempt_{attempt}_validator.stderr.log"
    schema_path = codex_validation_schema(logs_dir / "codex_validation_schema.json")
    write_text(prompt_path, prompt)
    invocation = run_codex(
        prompt,
        output_path,
        config=config,
        model=config.validation.model or os.environ.get("VALIDATOR_MODEL"),
        transcript_path=transcript_path,
        stderr_path=stderr_path,
        output_schema_path=schema_path,
        smoke_test=smoke_test,
    )
    try:
        parsed = json.loads(invocation.text)
    except json.JSONDecodeError:
        parsed = {
            "passed": False,
            "feedback": "Validator returned invalid JSON.",
            "failed_criteria": ["valid structured response"],
            "evidence": [],
        }
    return (
        result(
            validator,
            bool(parsed["passed"]),
            str(parsed["feedback"]),
            failed_criteria=parsed["failed_criteria"],
            evidence=parsed["evidence"],
        ),
        invocation,
    )


def validate_task(
    *,
    task_name: str,
    task: TaskSpec,
    config: HarnessConfig,
    out_dir: Path,
    attempt: int,
    validation_dir: Path,
    smoke_test: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mode = "deterministic" if smoke_test else config.validation.mode
    results: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []

    for spec in task.validators:
        if smoke_test and spec.type not in {
            "artifact_completeness",
            "experiment_artifacts",
        }:
            results.append(
                {
                    "name": spec.name,
                    "type": spec.type,
                    "passed": None,
                    "feedback": "Not run in smoke-test validation mode.",
                    "details": {"status": "not_run"},
                }
            )
            continue
        if smoke_test and spec.type == "artifact_completeness":
            spec = spec.model_copy(
                update={
                    "options": {
                        **spec.options,
                        "minimum_chars": min(
                            int(spec.options.get("minimum_chars", 1200)), 50
                        ),
                    }
                }
            )
            task = task.model_copy(update={"must_include": []})
        if spec.type == "codex_rubric":
            if mode == "deterministic":
                results.append(
                    {
                        "name": spec.name,
                        "type": spec.type,
                        "passed": None,
                        "feedback": "Not run in deterministic validation mode.",
                        "details": {"status": "not_run"},
                    }
                )
                continue
            failed_deterministic = [
                item["name"] for item in results if item.get("passed") is False
            ]
            if failed_deterministic:
                results.append(
                    {
                        "name": spec.name,
                        "type": spec.type,
                        "passed": None,
                        "feedback": (
                            "Not run because deterministic validators failed: "
                            f"{failed_deterministic}"
                        ),
                        "details": {
                            "status": "not_run",
                            "failed_dependencies": failed_deterministic,
                        },
                    }
                )
                continue
            rubric_result, invocation = run_codex_rubric(
                validator=spec,
                task_name=task_name,
                task=task,
                config=config,
                out_dir=out_dir,
                attempt=attempt,
                logs_dir=validation_dir,
                smoke_test=smoke_test,
            )
            results.append(rubric_result)
            invocations.append(invocation.__dict__)
            continue

        if mode == "codex":
            # Codex-only mode still runs file/schema checks because model judgment
            # cannot substitute for required files or parseable data.
            if spec.type not in {"artifact_completeness", "experiment_artifacts"}:
                continue
        validator = DETERMINISTIC_VALIDATORS[spec.type]
        results.append(validator(spec, task, out_dir))

    counted = [item for item in results if item["passed"] is not None]
    passed = bool(counted) and all(bool(item["passed"]) for item in counted)
    report = {
        "task": task_name,
        "attempt": attempt,
        "mode": mode,
        "passed": passed,
        "results": results,
        "feedback": [
            item["feedback"] for item in counted if not bool(item["passed"])
        ],
    }
    write_json(validation_dir / f"{task_name}_attempt_{attempt}.json", report)
    lines = [
        f"# Validation report: {task_name}",
        "",
        f"- Attempt: {attempt}",
        f"- Mode: {mode}",
        f"- Passed: {passed}",
        "",
    ]
    for item in results:
        lines.extend(
            [
                f"## {item['name']}",
                "",
                f"- Type: {item['type']}",
                f"- Passed: {item['passed']}",
                "",
                item["feedback"],
                "",
            ]
        )
    write_text(validation_dir / f"{task_name}_attempt_{attempt}.md", "\n".join(lines))
    return results, invocations


# ---------------------------------------------------------------------------
# Selection, reuse, resume, metadata, and orchestration
# ---------------------------------------------------------------------------


def descendants(config: HarnessConfig, seeds: set[str]) -> set[str]:
    selected = set(seeds)
    changed = True
    while changed:
        changed = False
        for name, task in config.tasks.items():
            if name not in selected and any(dep in selected for dep in task.depends_on):
                selected.add(name)
                changed = True
    return selected


def ancestors(config: HarnessConfig, seeds: set[str]) -> set[str]:
    selected = set(seeds)
    changed = True
    while changed:
        changed = False
        for name in list(selected):
            for dependency in config.tasks[name].depends_on:
                if dependency not in selected:
                    selected.add(dependency)
                    changed = True
    return selected


def validate_task_names(config: HarnessConfig, names: list[str] | None) -> set[str]:
    selected = set(names or [])
    unknown = sorted(selected - set(config.tasks))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    return selected


def initial_task_record(task: TaskSpec, out_dir: Path) -> dict[str, Any]:
    return {
        "status": "pending",
        "attempts": 0,
        "started_at": None,
        "finished_at": None,
        "outputs": {
            output.path: str(out_dir / output.path) for output in task.outputs
        },
        "output_hashes": {},
        "validations": [],
        "invocations": [],
        "failure_reason": None,
        "reused_from": None,
    }


def command_output(command: list[str], cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def environment_snapshot(config: HarnessConfig) -> dict[str, Any]:
    git_root_text = command_output(["git", "rev-parse", "--show-toplevel"], ROOT)
    git_root = Path(git_root_text) if git_root_text else ROOT
    git_status = command_output(["git", "status", "--porcelain"], git_root)
    codex = shutil.which(os.environ.get("CODEX_BIN", "codex"))
    codex_version = (
        command_output([codex, "--version"], ROOT) if codex is not None else None
    )
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "working_directory": str(ROOT),
        "runtime_command_prefix": config.runtime.command_prefix,
        "git_root": str(git_root) if git_root_text else None,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], git_root),
        "git_dirty": bool(git_status),
        "git_status_sha256": sha256_text(git_status) if git_status else None,
        "codex_version": codex_version,
    }


def new_state(
    config: HarnessConfig,
    out_dir: Path,
    *,
    args: argparse.Namespace,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": out_dir.name,
        "status": "pending",
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "finished_at": None,
        "config": str(resolve_from_root(args.config)),
        "config_sha256": config_sha256(config),
        "idea": str(resolve_from_root(args.idea)),
        "out": str(out_dir),
        "previous_report": (
            str(resolve_from_root(args.previous_report))
            if args.previous_report
            else None
        ),
        "feedback": (
            str(resolve_from_root(args.feedback)) if args.feedback else None
        ),
        "reuse_from": (
            str(resolve_from_root(args.reuse_from))
            if getattr(args, "reuse_from", None)
            else None
        ),
        "input_hashes": {
            "idea": sha256_file(resolve_from_root(args.idea)),
            "previous_report": (
                sha256_file(resolve_from_root(args.previous_report))
                if args.previous_report
                else None
            ),
            "feedback": (
                sha256_file(resolve_from_root(args.feedback)) if args.feedback else None
            ),
        },
        "environment": environment_snapshot(config),
        "resume_count": 0,
        "selection": {"mode": "all", "requested": [], "effective": []},
        "tasks": {
            name: initial_task_record(task, out_dir)
            for name, task in config.tasks.items()
        },
        "synthesis": None,
    }


def load_v2_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing metadata file: {path}")
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{path} uses metadata schema {state.get('schema_version')}; "
            f"v2 resume/reuse requires schema {SCHEMA_VERSION}"
        )
    return state


def apply_reuse(
    config: HarnessConfig,
    state: dict[str, Any],
    source_dir: Path,
) -> None:
    source = load_v2_state(source_dir / "metadata.json")
    for name, task in config.tasks.items():
        old = source["tasks"].get(name)
        if not old or old.get("status") not in TERMINAL_SUCCESS:
            continue
        missing = [
            path
            for path in old.get("outputs", {}).values()
            if not Path(path).exists()
        ]
        if missing:
            continue
        record = state["tasks"][name]
        record.update(
            {
                "status": "reused",
                "outputs": old["outputs"],
                "output_hashes": old.get("output_hashes", {}),
                "validations": old.get("validations", []),
                "reused_from": str(source_dir),
                "finished_at": utc_now(),
            }
        )


def configure_selection(
    config: HarnessConfig,
    state: dict[str, Any],
    *,
    select_names: list[str] | None,
    rerun_names: list[str] | None,
    has_base_state: bool,
) -> set[str]:
    if select_names and rerun_names:
        raise ValueError("--select and --rerun cannot be combined")

    if select_names:
        requested = validate_task_names(config, select_names)
        effective = ancestors(config, requested)
        mode = "select"
    elif rerun_names:
        if not has_base_state:
            raise ValueError("--rerun requires --resume or --reuse-from")
        requested = validate_task_names(config, rerun_names)
        effective = descendants(config, requested)
        mode = "rerun"
    else:
        requested = set()
        if has_base_state:
            effective = {
                name
                for name, record in state["tasks"].items()
                if record.get("status") in {"pending", "running"}
            }
            # Any task downstream of interrupted work must be reconsidered.
            effective = descendants(config, effective)
        else:
            effective = set(config.tasks)
        mode = "resume" if has_base_state else "all"

    for name, record in state["tasks"].items():
        if name in effective:
            record["status"] = "pending"
            record["failure_reason"] = None
            if mode == "rerun":
                record["outputs"] = {
                    output.path: str(Path(state["out"]) / output.path)
                    for output in config.tasks[name].outputs
                }
                record["attempts"] = 0
                record["started_at"] = None
                record["finished_at"] = None
                record["output_hashes"] = {}
                record["validations"] = []
                record["invocations"] = []
                record["reused_from"] = None
        elif record["status"] not in TERMINAL_SUCCESS:
            record["status"] = "skipped"
            record["finished_at"] = utc_now()

    state["selection"] = {
        "mode": mode,
        "requested": sorted(requested),
        "effective": [
            name for name in config.pipeline.order if name in effective
        ],
    }
    return effective


def summarize_outputs(task: TaskSpec, record: dict[str, Any]) -> dict[str, str | None]:
    return {
        output.path: sha256_file(Path(record["outputs"][output.path]))
        for output in task.outputs
    }


def save_state(state: dict[str, Any], metadata_path: Path) -> None:
    state["updated_at"] = utc_now()
    write_json(metadata_path, state)


def validation_feedback(results: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {item['name']}: {item['feedback']}"
        for item in results
        if item.get("passed") is False
    )


def run_task(
    *,
    task_name: str,
    config: HarnessConfig,
    state: dict[str, Any],
    out_dir: Path,
    idea: str,
    previous_report: str | None,
    human_feedback: str | None,
    recorder: EventRecorder,
    metadata_path: Path,
    smoke_test: bool,
) -> bool:
    task = config.tasks[task_name]
    record = state["tasks"][task_name]
    retry = effective_retry(config, task)
    planning = effective_planning(config, task)
    attempts_dir = out_dir / "_attempts"
    plans_dir = out_dir / "_plans"
    prompts_dir = out_dir / "_prompts"
    logs_dir = out_dir / "_logs"
    validation_dir = out_dir / "_validation"
    for directory in (
        attempts_dir,
        plans_dir,
        prompts_dir,
        logs_dir,
        validation_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    dependency_failures = [
        dependency
        for dependency in task.depends_on
        if state["tasks"][dependency]["status"] not in TERMINAL_SUCCESS
    ]
    if dependency_failures:
        record["status"] = "blocked"
        record["failure_reason"] = (
            f"Dependencies not accepted: {dependency_failures}"
        )
        record["finished_at"] = utc_now()
        recorder.emit(
            "task_blocked",
            task=task_name,
            dependencies=dependency_failures,
        )
        save_state(state, metadata_path)
        return False

    dependencies = dependency_context(config, task, state)
    max_attempts = 1 if smoke_test else retry.max_attempts
    min_attempts = 1 if smoke_test else retry.min_attempts
    last_results: list[dict[str, Any]] = []
    record["status"] = "running"
    record["started_at"] = record["started_at"] or utc_now()
    save_state(state, metadata_path)

    while record["attempts"] < max_attempts:
        attempt = int(record["attempts"]) + 1
        record["attempts"] = attempt
        recorder.emit("task_attempt_started", task=task_name, attempt=attempt)
        previous_primary_path = Path(
            record["outputs"][primary_output(task).path]
        )
        previous_output = (
            read_text(previous_primary_path)
            if attempt > 1 and previous_primary_path.exists()
            else "[No previous attempt]"
        )
        previous_validation = validation_feedback(last_results)
        plan_text: str | None = None

        try:
            if planning.enabled and not smoke_test:
                plan_prompt = build_planning_prompt(
                    task_name=task_name,
                    task=task,
                    config=config,
                    idea=idea,
                    dependencies=dependencies,
                    previous_report=previous_report,
                    human_feedback=human_feedback,
                    previous_output=previous_output,
                    previous_validation=previous_validation,
                    attempt=attempt,
                )
                plan_prompt_path = (
                    prompts_dir / f"{task_name}_attempt_{attempt}_plan.md"
                )
                plan_path = plans_dir / f"{task_name}_attempt_{attempt}.md"
                write_text(plan_prompt_path, plan_prompt)
                plan_invocation = run_codex(
                    plan_prompt,
                    plan_path,
                    config=config,
                    model=planning.model or task.model,
                    transcript_path=logs_dir
                    / f"{task_name}_attempt_{attempt}_plan.jsonl",
                    stderr_path=logs_dir
                    / f"{task_name}_attempt_{attempt}_plan.stderr.log",
                    smoke_test=smoke_test,
                )
                record["invocations"].append(
                    {"kind": "planning", "attempt": attempt, **plan_invocation.__dict__}
                )
                plan_text = plan_invocation.text

            prompt = build_task_prompt(
                task_name=task_name,
                task=task,
                config=config,
                out_dir=out_dir,
                idea=idea,
                dependencies=dependencies,
                previous_report=previous_report,
                human_feedback=human_feedback,
                previous_output=previous_output,
                previous_validation=previous_validation,
                plan=plan_text,
                attempt=attempt,
                smoke_test=smoke_test,
            )
            prompt_path = prompts_dir / f"{task_name}_attempt_{attempt}.md"
            write_text(prompt_path, prompt)
            configured_primary = out_dir / primary_output(task).path
            invocation = run_codex(
                prompt,
                configured_primary,
                config=config,
                model=task.model,
                transcript_path=logs_dir / f"{task_name}_attempt_{attempt}.jsonl",
                stderr_path=logs_dir
                / f"{task_name}_attempt_{attempt}.stderr.log",
                smoke_test=smoke_test,
            )
            record["invocations"].append(
                {"kind": "execution", "attempt": attempt, **invocation.__dict__}
            )
            write_text(
                attempts_dir / f"{task_name}_attempt_{attempt}.md",
                invocation.text,
            )
            record["outputs"] = {
                output.path: str(out_dir / output.path) for output in task.outputs
            }

            last_results, validator_invocations = validate_task(
                task_name=task_name,
                task=task,
                config=config,
                out_dir=out_dir,
                attempt=attempt,
                validation_dir=validation_dir,
                smoke_test=smoke_test,
            )
            record["invocations"].extend(
                {
                    "kind": "validation",
                    "attempt": attempt,
                    **validator_invocation,
                }
                for validator_invocation in validator_invocations
            )
            passed_results = [
                item for item in last_results if item.get("passed") is not None
            ]
            passed = bool(passed_results) and all(
                bool(item["passed"]) for item in passed_results
            )
            record["validations"].append(
                {
                    "attempt": attempt,
                    "passed": passed,
                    "results": last_results,
                }
            )
            record["output_hashes"] = summarize_outputs(task, record)
            recorder.emit(
                "task_attempt_validated",
                task=task_name,
                attempt=attempt,
                passed=passed,
                failed_validators=[
                    item["name"]
                    for item in last_results
                    if item.get("passed") is False
                ],
            )
            save_state(state, metadata_path)

            if passed and attempt >= min_attempts:
                record["status"] = "accepted"
                record["finished_at"] = utc_now()
                record["failure_reason"] = None
                recorder.emit("task_accepted", task=task_name, attempt=attempt)
                save_state(state, metadata_path)
                return True

            if passed and attempt < min_attempts:
                last_results = [
                    {
                        "name": "minimum_attempts",
                        "type": "policy",
                        "passed": False,
                        "feedback": (
                            f"Configured minimum attempts is {min_attempts}; make "
                            "a materially stronger next artifact."
                        ),
                        "details": {},
                    }
                ]
            elif not retry.enabled:
                break
        except Exception as exc:  # preserve failure, then retry if allowed
            message = f"{type(exc).__name__}: {exc}"
            last_results = [
                {
                    "name": "execution",
                    "type": "runtime",
                    "passed": False,
                    "feedback": message,
                    "details": {},
                }
            ]
            record["validations"].append(
                {"attempt": attempt, "passed": False, "results": last_results}
            )
            record["failure_reason"] = message
            recorder.emit(
                "task_attempt_failed",
                task=task_name,
                attempt=attempt,
                error=message,
            )
            save_state(state, metadata_path)
            if not retry.enabled:
                break

    record["finished_at"] = utc_now()
    record["failure_reason"] = validation_feedback(last_results) or record.get(
        "failure_reason"
    )
    if task.on_failure == "continue_with_warning":
        record["status"] = "warning"
        recorder.emit(
            "task_warning",
            task=task_name,
            reason=record["failure_reason"],
        )
        save_state(state, metadata_path)
        return True

    record["status"] = "rejected"
    recorder.emit(
        "task_rejected",
        task=task_name,
        reason=record["failure_reason"],
    )
    save_state(state, metadata_path)
    return False


def build_synthesis_prompt(
    config: HarnessConfig,
    state: dict[str, Any],
    idea: str,
    human_feedback: str | None,
) -> str:
    sources: list[str] = []
    status_lines: list[str] = []
    for name in config.pipeline.order:
        task = config.tasks[name]
        record = state["tasks"][name]
        status_lines.append(
            f"- {name}: {record['status']}"
            + (
                f" — {record['failure_reason']}"
                if record.get("failure_reason")
                else ""
            )
        )
        primary = primary_output(task)
        artifact_path = record["outputs"].get(primary.path)
        if artifact_path and Path(artifact_path).exists():
            sources.append(
                f"# Source task: {name}\n"
                f"Status: {record['status']}\n"
                f"Artifact: {artifact_path}\n\n"
                f"{read_text(artifact_path)}"
            )
    return f"""
Write the final synthesized report for this scientific research run.

Lead with the main conclusion. Integrate rather than concatenate artifacts.
Clearly distinguish:
- accepted or reused results;
- warning-level or inconclusive results;
- failed, blocked, and skipped tasks;
- established evidence, derivations, assumptions, conjectures, and open questions.

Never present a failed or unvalidated artifact as an accepted result. Give
concrete next steps and relative links where possible. Do not invent citations,
commands, experiments, or results.

Core idea:
{idea}

Human feedback:
{human_feedback or "[None]"}

Task statuses:
{chr(10).join(status_lines)}

Task artifacts:
{chr(10).join(sources) or "[No artifacts were produced]"}

Return only the Markdown report.
""".strip()


def write_fallback_report(
    config: HarnessConfig,
    state: dict[str, Any],
    report_path: Path,
    error: str,
) -> None:
    lines = [
        "# Research run status",
        "",
        "Automated synthesis could not be completed.",
        "",
        f"Error: {error}",
        "",
        "## Task statuses",
        "",
    ]
    for name in config.pipeline.order:
        record = state["tasks"][name]
        lines.append(
            f"- `{name}`: {record['status']}"
            + (
                f" — {record['failure_reason']}"
                if record.get("failure_reason")
                else ""
            )
        )
    write_text(report_path, "\n".join(lines) + "\n")


def synthesize(
    *,
    config: HarnessConfig,
    state: dict[str, Any],
    out_dir: Path,
    idea: str,
    human_feedback: str | None,
    recorder: EventRecorder,
    smoke_test: bool,
) -> None:
    report_path = out_dir / "report.md"
    prompt = build_synthesis_prompt(config, state, idea, human_feedback)
    prompt_path = out_dir / "_prompts" / "synthesis.md"
    write_text(prompt_path, prompt)
    recorder.emit("synthesis_started")
    try:
        invocation = run_codex(
            prompt,
            report_path,
            config=config,
            model=os.environ.get("REPORT_MODEL"),
            transcript_path=out_dir / "_logs" / "synthesis.jsonl",
            stderr_path=out_dir / "_logs" / "synthesis.stderr.log",
            smoke_test=smoke_test,
        )
        state["synthesis"] = {
            "status": "accepted",
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "invocation": invocation.__dict__,
        }
        recorder.emit("synthesis_finished", status="accepted")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        write_fallback_report(config, state, report_path, error)
        state["synthesis"] = {
            "status": "failed",
            "path": str(report_path),
            "sha256": sha256_file(report_path),
            "failure_reason": error,
        }
        recorder.emit("synthesis_finished", status="failed", error=error)


def dry_run_summary(
    config: HarnessConfig,
    state: dict[str, Any],
    effective: set[str],
) -> str:
    lines = [
        f"Run directory: {state['out']}",
        f"Selection mode: {state['selection']['mode']}",
        "Tasks:",
    ]
    for name in config.pipeline.order:
        task = config.tasks[name]
        action = "RUN" if name in effective else state["tasks"][name]["status"].upper()
        lines.append(
            f"  {action:8} {name} "
            f"(depends_on={task.depends_on}, on_failure={task.on_failure})"
        )
    return "\n".join(lines)


def preflight(
    config: HarnessConfig,
    *,
    idea_path: str | Path,
    previous_report_path: str | Path | None,
    feedback_path: str | Path | None,
    require_codex: bool,
) -> list[str]:
    issues: list[str] = []
    for label, path in (
        ("idea", idea_path),
        ("previous report", previous_report_path),
        ("feedback", feedback_path),
    ):
        if path is not None and not resolve_from_root(path).exists():
            issues.append(f"Missing {label} file: {resolve_from_root(path)}")
    if require_codex:
        try:
            find_codex_binary()
        except RuntimeError as exc:
            issues.append(str(exc))
    if config.runtime.command_prefix:
        executable = config.runtime.command_prefix[0]
        if shutil.which(executable) is None:
            issues.append(f"Runtime executable not found: {executable}")
    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an auditable, resumable scientific task graph."
    )
    parser.add_argument("--idea", default="idea.md")
    parser.add_argument("--config", "--agents", dest="config", default="agents.yml")
    parser.add_argument("--previous-report")
    parser.add_argument("--feedback")
    output = parser.add_mutually_exclusive_group(required=False)
    output.add_argument("--out")
    output.add_argument("--resume")
    parser.add_argument("--reuse-from")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--select", nargs="+")
    selection.add_argument("--rerun", nargs="+")
    parser.add_argument(
        "--validation-mode",
        choices=["deterministic", "hybrid", "codex"],
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument(
        "--no-synthesis",
        action="store_true",
        help="Do not make the final synthesis model call.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if not args.out and not args.resume:
        print("Either --out or --resume is required.", file=sys.stderr)
        return 2

    try:
        config = load_config(args.config)
        if args.validation_mode:
            config.validation.mode = args.validation_mode

        if args.resume:
            if args.rerun:
                raise ValueError(
                    "--rerun starts a new iteration and cannot be combined with "
                    "--resume. Use --reuse-from OLD_RUN --rerun TASK --out NEW_RUN."
                )
            out_dir = resolve_from_root(args.resume)
            metadata_path = out_dir / "metadata.json"
            state = load_v2_state(metadata_path)
            recorded_config_sha = state.get("config_sha256")
            current_config_sha = config_sha256(config)
            if recorded_config_sha and recorded_config_sha != current_config_sha:
                raise ValueError(
                    "The harness configuration changed since this run started. "
                    "Resume with the original config, or start a new run with "
                    "--reuse-from and --rerun."
                )
            state["resume_count"] = int(state.get("resume_count", 0)) + 1
            args.idea = state.get("idea") or args.idea
            args.previous_report = args.previous_report or state.get("previous_report")
            args.feedback = args.feedback or state.get("feedback")
            # A crashed "running" task is eligible for resumption.
            for record in state["tasks"].values():
                if record.get("status") == "running":
                    record["status"] = "pending"
        else:
            out_dir = resolve_from_root(args.out)
            if out_dir.exists() and any(out_dir.iterdir()):
                raise ValueError(
                    f"Output directory is not empty: {out_dir}. "
                    "Use --resume or choose a new directory."
                )
            out_dir.mkdir(parents=True, exist_ok=True)
            state = new_state(config, out_dir, args=args)
            metadata_path = out_dir / "metadata.json"

        recorder = EventRecorder(out_dir / "events.jsonl")
        if args.reuse_from:
            if args.resume:
                raise ValueError("--reuse-from cannot be combined with --resume")
            apply_reuse(config, state, resolve_from_root(args.reuse_from))

        has_base_state = bool(args.resume or args.reuse_from)
        effective = configure_selection(
            config,
            state,
            select_names=args.select,
            rerun_names=args.rerun,
            has_base_state=has_base_state,
        )
        state["status"] = "planned"
        if not (args.dry_run and args.resume):
            save_state(state, metadata_path)

        issues = preflight(
            config,
            idea_path=args.idea,
            previous_report_path=args.previous_report,
            feedback_path=args.feedback,
            require_codex=not args.dry_run,
        )
        if issues:
            raise ValueError("Preflight failed:\n- " + "\n- ".join(issues))

        print(dry_run_summary(config, state, effective))
        if args.dry_run:
            if not args.resume:
                recorder.emit("dry_run_completed", selection=state["selection"])
            return 0

        idea = read_text(args.idea)
        previous_report = read_optional(args.previous_report)
        human_feedback = read_optional(args.feedback)
        state["status"] = "running"
        recorder.emit("run_started", selection=state["selection"])
        save_state(state, metadata_path)

        stopped = False
        for name in config.pipeline.order:
            if name not in effective:
                continue
            accepted = run_task(
                task_name=name,
                config=config,
                state=state,
                out_dir=out_dir,
                idea=idea,
                previous_report=previous_report,
                human_feedback=human_feedback,
                recorder=recorder,
                metadata_path=metadata_path,
                smoke_test=args.smoke_test,
            )
            if not accepted and config.tasks[name].on_failure == "stop":
                stopped = True
                for later in config.pipeline.order[
                    config.pipeline.order.index(name) + 1 :
                ]:
                    if later in effective and state["tasks"][later]["status"] == "pending":
                        state["tasks"][later]["status"] = "blocked"
                        state["tasks"][later]["failure_reason"] = (
                            f"Run stopped after required task {name} failed"
                        )
                        state["tasks"][later]["finished_at"] = utc_now()
                break

        synthesis_already_accepted = (
            isinstance(state.get("synthesis"), dict)
            and state["synthesis"].get("status") == "accepted"
        )
        if not args.no_synthesis and (effective or not synthesis_already_accepted):
            synthesize(
                config=config,
                state=state,
                out_dir=out_dir,
                idea=idea,
                human_feedback=human_feedback,
                recorder=recorder,
                smoke_test=args.smoke_test,
            )

        failed = [
            name
            for name, record in state["tasks"].items()
            if record["status"] in {"rejected", "blocked"}
        ]
        state["status"] = "failed" if stopped or failed else "completed"
        state["finished_at"] = utc_now()
        recorder.emit("run_finished", status=state["status"], failed_tasks=failed)
        save_state(state, metadata_path)

        print(f"\nRun status: {state['status']}")
        print(f"Run directory: {out_dir}")
        print(f"Metadata: {metadata_path}")
        if state.get("synthesis"):
            print(f"Report: {state['synthesis']['path']}")
        return 1 if state["status"] == "failed" else 0
    except (FileNotFoundError, ValueError, ValidationError, RuntimeError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
