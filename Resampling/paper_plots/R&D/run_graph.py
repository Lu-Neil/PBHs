"""Run the research-agent workflow defined by ``agents.yml``.

Intended usage
--------------
This script runs one explicit iteration of the local research graph. It reads
``idea.md`` and ``agents.yml``, runs the configured agents in
``pipeline.order``, validates each agent artifact, and writes a synthesized
report plus metadata to the requested output directory.

Normal runs can require multiple refinement attempts per agent through
``ralph_loop.min_attempts`` and can run a scratchpad planning pass before each
artifact through ``planning_pass_defaults`` or per-agent ``planning_pass``.
Planning artifacts are written to ``_plans/`` under the run directory.

Run commands from this directory through the PBH conda environment:

    conda run -n PBH python run_graph.py --out runs/iter_000

For a fast wiring check that avoids expensive full-agent behavior:

    conda run -n PBH python run_graph.py --out runs/smoke --smoke-test

To run a new iteration from an earlier report while incorporating human
comments:

    conda run -n PBH python run_graph.py \
        --previous-report runs/iter_000/report.md \
        --feedback feedback/iter_001_comments.md \
        --out runs/iter_001

Leaving human comments for a new run
------------------------------------
Write comments in a normal Markdown or text file, then pass that file with
``--feedback`` on the next run. The script reads the file into the agent prompt
under "Human feedback / new directions", sets iteration mode, and tells agents
to decide what should be rerun, challenged, extended, or rewritten.

Useful feedback files are concrete and auditable. For example, include bullets
such as:

    - Literature: check whether this is equivalent to StackSlide with a
      nonlinear frequency coordinate.
    - Theory: clarify the null distribution before making sensitivity claims.
    - Referee: focus on possible novelty conflicts with semicoherent metrics.

Pair ``--feedback`` with ``--previous-report`` when the comments refer to a
prior run. The previous report is provided as context, while the feedback is
treated as the main instruction for the new iteration. Each run writes its own
artifacts under ``--out`` and does not modify earlier run directories.

Relevant environment variables
------------------------------
``RESEARCH_MODEL`` sets the model for agent runs, ``REPORT_MODEL`` sets the
model for the synthesized report, ``VALIDATOR_MODE`` may be ``checklist`` or
``codex``, ``CODEX_BIN`` selects the Codex CLI binary, and
``CODEX_TIMEOUT_SECONDS`` adjusts the per-Codex-call timeout.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Literal, TypedDict

import yaml
from langgraph.graph import END, START, StateGraph


# ---------------------------------------------------------------------
# Optional .env support
# ---------------------------------------------------------------------

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
ATTEMPT_DIR = OUTPUT_DIR / "_attempts"
VALIDATION_DIR = OUTPUT_DIR / "_validation"
PLAN_DIR = OUTPUT_DIR / "_plans"


# ---------------------------------------------------------------------
# State
# ---------------------------------------------------------------------

class ResearchState(TypedDict):
    idea: str
    previous_report: str | None
    human_feedback: str | None
    iteration_mode: bool
    config: dict[str, Any]
    agent_order: list[str]
    current_agent: str
    attempts: dict[str, int]
    outputs: dict[str, str]
    feedback: dict[str, str]
    plans: dict[str, str]
    validation_passed: bool


ValidationMode = Literal["checklist", "codex"]


# ---------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------

def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def read_text(path: str | Path) -> str:
    path = resolve_path(path)
    if not path.exists():
        return f"[Missing file: {path}]"
    return path.read_text(encoding="utf-8")


def read_optional(path: str | Path | None) -> str | None:
    if path is None:
        return None
    return read_text(path)


def write_text(path: str | Path, text: str) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Missing agents config at {config_path}")
    return yaml.safe_load(config_path.read_text(encoding="utf-8"))


def ensure_dirs() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ATTEMPT_DIR.mkdir(parents=True, exist_ok=True)
    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    PLAN_DIR.mkdir(parents=True, exist_ok=True)


def configure_run_paths(out_dir: str | Path) -> Path:
    global OUTPUT_DIR, ATTEMPT_DIR, VALIDATION_DIR, PLAN_DIR

    OUTPUT_DIR = resolve_path(out_dir)
    ATTEMPT_DIR = OUTPUT_DIR / "_attempts"
    VALIDATION_DIR = OUTPUT_DIR / "_validation"
    PLAN_DIR = OUTPUT_DIR / "_plans"
    ensure_dirs()
    return OUTPUT_DIR


def path_under_out(path: str | Path, out_dir: Path) -> str:
    path = Path(path)
    if path.is_absolute():
        return str(path)

    parts = path.parts
    if parts and parts[0] == "outputs":
        return str(out_dir / Path(*parts[1:]))

    return str(out_dir / path)


def configure_agent_outputs(config: dict[str, Any], out_dir: Path) -> dict[str, Any]:
    config = copy.deepcopy(config)

    for agent_cfg in config.get("agents", {}).values():
        for key in ("output", "secondary_output"):
            if key in agent_cfg:
                agent_cfg[key] = path_under_out(agent_cfg[key], out_dir)

    return config


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------

def get_agent_config(state: ResearchState, agent_name: str) -> dict[str, Any]:
    return state["config"]["agents"][agent_name]


def get_max_attempts(state: ResearchState, agent_name: str) -> int:
    defaults = state["config"].get("ralph_loop_defaults", {})
    agent_cfg = get_agent_config(state, agent_name)
    loop_cfg = agent_cfg.get("ralph_loop", {}) or {}
    return int(loop_cfg.get("max_attempts", defaults.get("max_attempts", 1)))


def get_min_attempts(state: ResearchState, agent_name: str) -> int:
    defaults = state["config"].get("ralph_loop_defaults", {})
    agent_cfg = get_agent_config(state, agent_name)
    loop_cfg = agent_cfg.get("ralph_loop", {}) or {}

    max_attempts = get_max_attempts(state, agent_name)
    requested = int(loop_cfg.get("min_attempts", defaults.get("min_attempts", 1)))
    return max(1, min(requested, max_attempts))


def get_planning_pass_config(
    state: ResearchState,
    agent_name: str,
) -> dict[str, Any]:
    defaults = state["config"].get("planning_pass_defaults", {})
    agent_cfg = get_agent_config(state, agent_name)
    agent_planning_cfg = agent_cfg.get("planning_pass", {}) or {}
    return {**defaults, **agent_planning_cfg}


def planning_pass_enabled(state: ResearchState, agent_name: str) -> bool:
    cfg = get_planning_pass_config(state, agent_name)
    return bool(cfg.get("enabled", False))


def get_global_rules(state: ResearchState) -> list[str]:
    return state["config"].get("global_rules", [])


def get_codex_model(agent_cfg: dict[str, Any]) -> str | None:
    # Priority:
    # 1. model explicitly set on the agent
    # 2. RESEARCH_MODEL environment variable
    # 3. Codex CLI default model from ~/.codex/config.toml
    return agent_cfg.get("model") or os.environ.get("RESEARCH_MODEL")


def get_validation_mode() -> ValidationMode:
    mode = os.environ.get("VALIDATOR_MODE", "checklist").strip().lower()
    if mode not in {"checklist", "codex"}:
        raise ValueError("VALIDATOR_MODE must be 'checklist' or 'codex'")
    return mode  # type: ignore[return-value]


# ---------------------------------------------------------------------
# Codex CLI wrapper
# ---------------------------------------------------------------------

def find_codex_binary() -> str:
    codex_bin = os.environ.get("CODEX_BIN", "codex")
    resolved = shutil.which(codex_bin)
    if resolved is None:
        raise RuntimeError(
            f"Could not find Codex CLI binary: {codex_bin!r}\n"
            "Install/login first, e.g.:\n"
            "  curl -fsSL https://chatgpt.com/codex/install.sh | sh\n"
            "  codex login\n"
        )
    return resolved


def run_codex(
    prompt: str,
    output_path: str | Path,
    *,
    model: str | None = None,
    output_schema_path: str | Path | None = None,
    timeout_seconds: int | None = None,
) -> str:
    """
    Run Codex non-interactively.

    Uses:
      codex exec --sandbox workspace-write --ask-for-approval never \
        --output-last-message <output_path> - < prompt

    The final assistant message is read from output_path.
    """

    codex = find_codex_binary()
    output_path = resolve_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sandbox = os.environ.get("CODEX_SANDBOX", "workspace-write")
    profile = os.environ.get("CODEX_PROFILE")

    cmd = [
        codex,
        "exec",
        "--cd",
        str(ROOT),
        "--sandbox",
        sandbox,
        "--output-last-message",
        str(output_path),
        "--color",
        "never",
    ]

    if model:
        cmd += ["--model", model]

    if profile:
        cmd += ["--profile", profile]

    if output_schema_path is not None:
        cmd += ["--output-schema", str(resolve_path(output_schema_path))]

    # Read prompt from stdin.
    cmd += ["-"]

    timeout_seconds = timeout_seconds or int(os.environ.get("CODEX_TIMEOUT_SECONDS", "1800"))

    result = subprocess.run(
        cmd,
        input=prompt,
        text=True,
        capture_output=True,
        cwd=ROOT,
        timeout=timeout_seconds,
        check=False,
    )

    if result.returncode != 0:
        debug_path = VALIDATION_DIR / f"codex_error_{int(time.time())}.log"
        debug_path.write_text(
            "COMMAND:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + result.stdout
            + "\n\nSTDERR:\n"
            + result.stderr,
            encoding="utf-8",
        )

        raise RuntimeError(
            f"Codex failed with exit code {result.returncode}.\n"
            f"Debug log written to: {debug_path}\n\n"
            f"STDERR excerpt:\n{result.stderr[-3000:]}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"Codex exited successfully but did not write output file: {output_path}"
        )

    return output_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------

def collect_prior_outputs(state: ResearchState, agent_name: str) -> str:
    chunks: list[str] = []

    for prior_agent, output_path in state["outputs"].items():
        if prior_agent == agent_name:
            continue

        text = read_text(output_path)
        chunks.append(
            f"\n\n# Prior output from {prior_agent}\n\n"
            f"File: {output_path}\n\n"
            f"{text}"
        )

    return "\n".join(chunks).strip() or "[No prior agent outputs yet]"


def build_agent_planning_prompt_text(
    state: ResearchState,
    agent_name: str,
    attempt: int,
) -> str:
    agent_cfg = get_agent_config(state, agent_name)
    planning_cfg = get_planning_pass_config(state, agent_name)

    global_rules = get_global_rules(state)
    previous_feedback = state["feedback"].get(agent_name, "")
    output_path = agent_cfg.get("output")
    existing_output = read_text(output_path) if output_path else "[No output path set]"
    prior_outputs = collect_prior_outputs(state, agent_name)
    previous_report = state.get("previous_report")
    human_feedback = state.get("human_feedback")

    iteration_context = ""

    if previous_report:
        iteration_context += f"""

# Previous report

{previous_report}
"""

    if human_feedback:
        iteration_context += f"""

# Human feedback / new directions

{human_feedback}

Use this feedback to decide what needs to be rerun, challenged, extended, or rewritten.
Do not merely polish the previous report.
Preserve useful previous findings, but treat the feedback as the main instruction.
"""

    instructions = planning_cfg.get(
        "instructions",
        [
            "Identify the most important claims or derivations the artifact must establish.",
            "List the assumptions, gaps, likely failure modes, and validation checks.",
            "Outline the artifact structure before drafting.",
            "For refinement attempts, identify what should be improved from the existing output.",
        ],
    )

    return f"""
You are preparing a scratchpad plan for the research agent named: {agent_name}

# Purpose

{agent_cfg.get("purpose", "")}

# Global rules

{yaml.dump(global_rules, sort_keys=False).strip()}

# Agent-specific rules

{yaml.dump(agent_cfg.get("rules", []), sort_keys=False).strip()}

# Attempt

This is attempt {attempt}.

# Required contents for the final artifact

{yaml.dump(agent_cfg.get("must_include", []), sort_keys=False).strip()}

# Special focus

{yaml.dump(agent_cfg.get("special_focus", []), sort_keys=False).strip()}

# Cautions

{yaml.dump(agent_cfg.get("cautions", []), sort_keys=False).strip()}

# idea.md

{state["idea"]}

{iteration_context}

# Prior agent outputs

{prior_outputs}

# Previous validator feedback for this agent

{previous_feedback or "[No previous feedback]"}

# Existing output from previous attempt

{existing_output if attempt > 1 else "[No previous attempt]"}

# Planning instructions

Return a concise Markdown scratchpad plan only. This plan will be fed into the
final artifact prompt for this attempt and saved for audit.

Do not write the final artifact yet.
Do not include conversational framing.
Do not invent citations.

{yaml.dump(instructions, sort_keys=False).strip()}
""".strip()


def build_agent_prompt_text(
    state: ResearchState,
    agent_name: str,
    attempt: int,
    plan_text: str | None = None,
) -> str:
    agent_cfg = get_agent_config(state, agent_name)

    global_rules = get_global_rules(state)
    previous_feedback = state["feedback"].get(agent_name, "")
    output_path = agent_cfg.get("output")
    existing_output = read_text(output_path) if output_path else "[No output path set]"
    prior_outputs = collect_prior_outputs(state, agent_name)
    previous_report = state.get("previous_report")
    human_feedback = state.get("human_feedback")

    iteration_context = ""

    if previous_report:
        iteration_context += f"""

# Previous report

{previous_report}
"""

    if human_feedback:
        iteration_context += f"""

# Human feedback / new directions

{human_feedback}

Use this feedback to decide what needs to be rerun, challenged, extended, or rewritten.
Do not merely polish the previous report.
Preserve useful previous findings, but treat the feedback as the main instruction.
"""

    return f"""
You are the research agent named: {agent_name}

# Purpose

{agent_cfg.get("purpose", "")}

# Global rules

{yaml.dump(global_rules, sort_keys=False).strip()}

# Agent-specific rules

{yaml.dump(agent_cfg.get("rules", []), sort_keys=False).strip()}

# Attempt

This is attempt {attempt}.

# Required output file

{agent_cfg.get("output")}

# Required contents

{yaml.dump(agent_cfg.get("must_include", []), sort_keys=False).strip()}

# Special focus

{yaml.dump(agent_cfg.get("special_focus", []), sort_keys=False).strip()}

# Cautions

{yaml.dump(agent_cfg.get("cautions", []), sort_keys=False).strip()}

# idea.md

{state["idea"]}

{iteration_context}

# Prior agent outputs

{prior_outputs}

# Previous validator feedback for this agent

{previous_feedback or "[No previous feedback]"}

# Existing output from previous attempt

{existing_output if attempt > 1 else "[No previous attempt]"}

# Scratchpad plan for this attempt

{plan_text or "[Planning pass disabled]"}

# Instructions

Write the artifact as Markdown.

Use the scratchpad plan to improve coverage and depth, but do not copy it
mechanically. If this is a refinement attempt after a passing validation,
make the artifact materially stronger rather than merely rephrasing it.

Do not use shell commands, apply_patch, or attempt to create/edit files.
Return the Markdown artifact as your final response only.
The Python wrapper will save your final response to the required output file.

Do not include conversational framing like "Sure" or "Here is...".
Do not claim that you read external papers unless the content was actually available to you.
Do not invent citations.
For citation gaps, write "citation needed" or list the search target.
Distinguish:
- established facts
- derived results
- conjectures
- assumptions
- open questions

Make the output useful for the next agent in the workflow.
""".strip()


# ---------------------------------------------------------------------
# Agent nodes
# ---------------------------------------------------------------------

def make_agent_node(agent_name: str):
    def agent_node(state: ResearchState) -> dict[str, Any]:
        agent_cfg = get_agent_config(state, agent_name)

        attempts = dict(state["attempts"])
        attempt = attempts.get(agent_name, 0) + 1
        attempts[agent_name] = attempt

        output_path = agent_cfg["output"]
        model = get_codex_model(agent_cfg)

        print(f"\n=== Running agent: {agent_name} attempt {attempt} ===")
        print(f"Output: {output_path}")

        plan_text: str | None = None
        plans = dict(state["plans"])

        if planning_pass_enabled(state, agent_name):
            planning_cfg = get_planning_pass_config(state, agent_name)
            plan_path = PLAN_DIR / f"{agent_name}_attempt_{attempt}_plan.md"
            plan_model = planning_cfg.get("model") or model
            print(f"Planning pass: {plan_path}")

            plan_prompt = build_agent_planning_prompt_text(
                state,
                agent_name,
                attempt,
            )
            plan_text = run_codex(
                plan_prompt,
                plan_path,
                model=plan_model,
            )
            plans[f"{agent_name}:attempt:{attempt}"] = str(plan_path)

        prompt = build_agent_prompt_text(
            state,
            agent_name,
            attempt,
            plan_text=plan_text,
        )

        output_text = run_codex(
            prompt,
            output_path,
            model=model,
        )

        # Save every attempt separately, even though the main output file is overwritten.
        attempt_path = ATTEMPT_DIR / f"{agent_name}_attempt_{attempt}.md"
        write_text(attempt_path, output_text)

        outputs = dict(state["outputs"])
        outputs[agent_name] = output_path

        return {
            "current_agent": agent_name,
            "attempts": attempts,
            "outputs": outputs,
            "plans": plans,
            "validation_passed": False,
        }

    return agent_node


# ---------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------

def normalize_words(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z0-9_\\{}^$]+", text.lower()))


def checklist_validate(
    state: ResearchState,
    agent_name: str,
    artifact: str,
) -> dict[str, Any]:
    """
    Deterministic-ish validator.

    This is intentionally simple. It checks:
    - artifact is non-trivial
    - required sections are at least gestured at
    - no obvious placeholders
    """

    agent_cfg = get_agent_config(state, agent_name)
    must_include = [str(x) for x in agent_cfg.get("must_include", [])]

    artifact_lower = artifact.lower()
    artifact_words = normalize_words(artifact)

    missing: list[str] = []
    weak: list[str] = []

    for item in must_include:
        item_words = [
            w for w in normalize_words(item)
            if len(w) >= 4 and w not in {"must", "include", "with", "from"}
        ]

        if not item_words:
            continue

        hits = sum(1 for w in item_words if w in artifact_words)

        if hits == 0:
            missing.append(item)
        elif hits < max(1, min(2, len(item_words))):
            weak.append(item)

    placeholder_patterns = [
        "todo",
        "tbd",
        "to be written",
        "lorem ipsum",
        "[insert",
        "placeholder",
    ]

    placeholders = [
        p for p in placeholder_patterns
        if p in artifact_lower
    ]

    too_short = len(artifact.strip()) < int(
        agent_cfg.get("minimum_chars", os.environ.get("MIN_ARTIFACT_CHARS", "1200"))
    )

    passed = not too_short and not missing and len(placeholders) == 0

    feedback_lines: list[str] = []

    if passed:
        feedback_lines.append("Passed checklist validation.")
    else:
        feedback_lines.append("Revise the artifact to address the following issues:")

        if too_short:
            feedback_lines.append("- The artifact is too short or too shallow.")

        if missing:
            feedback_lines.append("- Missing required items:")
            feedback_lines.extend(f"  - {x}" for x in missing)

        if weak:
            feedback_lines.append("- Weakly covered required items:")
            feedback_lines.extend(f"  - {x}" for x in weak)

        if placeholders:
            feedback_lines.append("- Placeholder text detected:")
            feedback_lines.extend(f"  - {x}" for x in placeholders)

    return {
        "passed": passed,
        "feedback": "\n".join(feedback_lines),
        "missing_items": missing,
        "weak_items": weak,
        "validator": "checklist",
    }


def validation_schema_path() -> Path:
    path = VALIDATION_DIR / "validation_schema.json"

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "passed": {"type": "boolean"},
            "feedback": {"type": "string"},
            "missing_items": {
                "type": "array",
                "items": {"type": "string"},
            },
            "weak_items": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["passed", "feedback", "missing_items", "weak_items"],
    }

    path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    return path


def codex_validate(
    state: ResearchState,
    agent_name: str,
    artifact: str,
) -> dict[str, Any]:
    agent_cfg = get_agent_config(state, agent_name)

    prompt = f"""
You are a strict validator for a scientific research artifact.

Validate the output produced by agent: {agent_name}

# Agent purpose

{agent_cfg.get("purpose", "")}

# Required contents

{yaml.dump(agent_cfg.get("must_include", []), sort_keys=False).strip()}

# Validation standard

Pass if the artifact is useful enough for the next agent.
Do not require perfection.
Do not pass vague placeholder text.
Do not reward mere repetition of idea.md.
Require clear assumptions, concrete next steps, and specific technical content.

# Artifact

{artifact}

# Required response

Return only valid JSON matching the provided schema.
""".strip()

    attempt = state["attempts"].get(agent_name, 1)
    output_path = VALIDATION_DIR / f"{agent_name}_attempt_{attempt}_codex_validation.json"

    model = os.environ.get("VALIDATOR_MODEL") or os.environ.get("RESEARCH_MODEL")
    schema_path = validation_schema_path()

    result_text = run_codex(
        prompt,
        output_path,
        model=model,
        output_schema_path=schema_path,
    )

    try:
        result = json.loads(result_text)
    except json.JSONDecodeError:
        # Fallback: fail closed.
        result = {
            "passed": False,
            "feedback": (
                "Codex validator did not return valid JSON. "
                "Revise the artifact and ensure the validator can assess it."
            ),
            "missing_items": [],
            "weak_items": [],
        }

    result["validator"] = "codex"
    return result


def validate_node(state: ResearchState) -> dict[str, Any]:
    agent_name = state["current_agent"]
    agent_cfg = get_agent_config(state, agent_name)
    output_path = agent_cfg["output"]
    artifact = read_text(output_path)

    mode = get_validation_mode()

    if mode == "codex":
        result = codex_validate(state, agent_name, artifact)
    else:
        result = checklist_validate(state, agent_name, artifact)

    attempt = state["attempts"].get(agent_name, 1)
    validation_path = VALIDATION_DIR / f"{agent_name}_attempt_{attempt}.md"

    validation_report = f"""# Validation report

Agent: {agent_name}
Attempt: {attempt}
Validator: {result.get("validator")}
Passed: {result.get("passed")}

## Missing items

{yaml.dump(result.get("missing_items", []), sort_keys=False).strip()}

## Weak items

{yaml.dump(result.get("weak_items", []), sort_keys=False).strip()}

## Feedback

{result.get("feedback", "")}
"""

    write_text(validation_path, validation_report)

    feedback = dict(state["feedback"])
    feedback[agent_name] = str(result.get("feedback", ""))

    print(f"Validation for {agent_name} attempt {attempt}: {result.get('passed')}")

    return {
        "validation_passed": bool(result.get("passed", False)),
        "feedback": feedback,
    }


# ---------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------

def route_after_validation(state: ResearchState) -> str:
    agent_name = state["current_agent"]
    order = state["agent_order"]

    attempt = state["attempts"].get(agent_name, 1)
    max_attempts = get_max_attempts(state, agent_name)
    min_attempts = get_min_attempts(state, agent_name)

    if state["validation_passed"] and attempt < min_attempts:
        print(
            f"Refining {agent_name}: attempt {attempt + 1}/{min_attempts} "
            "minimum attempts"
        )
        return agent_name

    if not state["validation_passed"] and attempt < max_attempts:
        print(f"Retrying {agent_name}: attempt {attempt + 1}/{max_attempts}")
        return agent_name

    idx = order.index(agent_name)

    if not state["validation_passed"]:
        print(
            f"Max attempts reached for {agent_name}; continuing to next agent."
        )

    if idx + 1 >= len(order):
        return END

    next_agent = order[idx + 1]
    print(f"Moving from {agent_name} to {next_agent}")
    return next_agent


def build_graph(agent_order: list[str]):
    graph = StateGraph(ResearchState)

    for agent_name in agent_order:
        graph.add_node(agent_name, make_agent_node(agent_name))
        graph.add_edge(agent_name, "validate")

    graph.add_node("validate", validate_node)

    graph.add_edge(START, agent_order[0])

    path_map = {agent_name: agent_name for agent_name in agent_order}
    path_map[END] = END

    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        path_map=path_map,
    )

    return graph.compile()


# ---------------------------------------------------------------------
# Run outputs
# ---------------------------------------------------------------------

def collect_report_sources(
    final_state: ResearchState,
    out_dir: Path,
) -> list[dict[str, str]]:
    sources: list[dict[str, str]] = []

    for agent_name in final_state["agent_order"]:
        output_path = final_state["outputs"].get(agent_name)
        if not output_path:
            continue

        path = resolve_path(output_path)
        sources.append(
            {
                "agent": agent_name,
                "path": str(path),
                "relative_path": os.path.relpath(path, out_dir),
                "text": read_text(path),
            }
        )

    return sources


def build_report_prompt(
    final_state: ResearchState,
    out_dir: Path,
) -> str:
    sources = collect_report_sources(final_state, out_dir)
    source_index = "\n".join(
        f"- {source['agent']}: [{source['relative_path']}]({source['relative_path']})"
        for source in sources
    )
    source_text = "\n\n".join(
        f"# Source: {source['agent']}\n"
        f"Relative link: [{source['relative_path']}]({source['relative_path']})\n\n"
        f"{source['text']}"
        for source in sources
    )

    iteration_context = ""
    if final_state.get("iteration_mode"):
        iteration_context += (
            "\nThis was an iterative run. The final report should make clear "
            "how the new synthesis responds to the human feedback.\n"
        )

    if final_state.get("human_feedback"):
        iteration_context += f"""

Human feedback for this iteration:
{final_state["human_feedback"]}
"""

    return f"""
You are writing the final synthesized report for a research-agent run.

Do not concatenate the source artifacts. Write a coherent synthesis that:
- states the main conclusion up front;
- integrates the useful findings across agents;
- resolves or highlights tensions between the planner, literature, theory, and referee artifacts;
- distinguishes established facts, derived results, assumptions, conjectures, and open questions;
- gives concrete next steps;
- is substantially shorter than the combined source artifacts;
- uses relative Markdown links to the detailed artifacts when pointing to supporting detail.

Available detailed artifacts:
{source_index}
{iteration_context}

Core idea:
{final_state["idea"]}

Detailed source artifacts:

{source_text}

Return only the final Markdown report.
Do not include conversational framing.
Do not invent citations.
If a citation is needed but not present in the source artifacts, write "citation needed".
""".strip()


def write_synthesized_report(
    final_state: ResearchState,
    out_dir: Path,
    report_path: Path,
) -> None:
    prompt = build_report_prompt(final_state, out_dir)
    model = os.environ.get("REPORT_MODEL") or os.environ.get("RESEARCH_MODEL")
    run_codex(prompt, report_path, model=model)


def write_run_outputs(
    final_state: ResearchState,
    *,
    args: argparse.Namespace,
    out_dir: Path,
) -> None:
    report_path = out_dir / "report.md"
    metadata_path = out_dir / "metadata.json"

    write_synthesized_report(final_state, out_dir, report_path)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "idea": str(resolve_path(args.idea)),
        "agents": str(resolve_path(args.agents)),
        "out": str(out_dir),
        "previous_report": (
            str(resolve_path(args.previous_report))
            if args.previous_report is not None
            else None
        ),
        "feedback": (
            str(resolve_path(args.feedback))
            if args.feedback is not None
            else None
        ),
        "iteration_mode": final_state.get("iteration_mode", False),
        "agent_order": final_state["agent_order"],
        "outputs": final_state["outputs"],
        "plans": final_state.get("plans", {}),
        "attempts": final_state["attempts"],
        "validation_mode": get_validation_mode(),
        "report_mode": "synthesized",
    }

    write_text(metadata_path, json.dumps(metadata, indent=2) + "\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def preflight(
    config: dict[str, Any],
    *,
    idea_path: str | Path,
    previous_report_path: str | Path | None,
    feedback_path: str | Path | None,
) -> list[str]:
    issues: list[str] = []

    if not resolve_path(idea_path).exists():
        issues.append(f"Missing idea file: {resolve_path(idea_path)}")

    if (
        previous_report_path is not None
        and not resolve_path(previous_report_path).exists()
    ):
        issues.append(f"Missing previous report: {resolve_path(previous_report_path)}")

    if feedback_path is not None and not resolve_path(feedback_path).exists():
        issues.append(f"Missing feedback file: {resolve_path(feedback_path)}")

    if "pipeline" not in config or "order" not in config["pipeline"]:
        issues.append("agents.yml must contain pipeline.order")

    if "agents" not in config:
        issues.append("agents.yml must contain agents")

    if "pipeline" in config and "agents" in config:
        for agent_name in config["pipeline"].get("order", []):
            if agent_name not in config["agents"]:
                issues.append(f"pipeline.order contains unknown agent: {agent_name}")
                continue

            if "output" not in config["agents"][agent_name]:
                issues.append(f"Agent {agent_name} is missing output")

    try:
        find_codex_binary()
    except RuntimeError as exc:
        issues.append(str(exc))

    return issues


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the research-agent graph for one explicit iteration."
    )
    parser.add_argument("--idea", default="idea.md")
    parser.add_argument("--agents", default="agents.yml")
    parser.add_argument("--previous-report", default=None)
    parser.add_argument("--feedback", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--smoke-test", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = configure_run_paths(args.out)
    config = configure_agent_outputs(load_config(args.agents), out_dir)

    # -------------------------------------------------------------
    # Smoke test mode
    # Usage:
    #   python run_graph.py --out runs/smoke --smoke-test
    #
    # Runs only the first agent, disables Codex validation,
    # disables retries, and makes checklist validation easy to pass.
    # -------------------------------------------------------------
    if args.smoke_test:
        smoke_agents = list(config["pipeline"]["order"])[:4]
        config["pipeline"]["order"] = smoke_agents

        os.environ["VALIDATOR_MODE"] = "checklist"
        os.environ["MIN_ARTIFACT_CHARS"] = "50"
        os.environ["CODEX_TIMEOUT_SECONDS"] = os.environ.get(
            "CODEX_TIMEOUT_SECONDS", "180"
        )
        os.environ["LANGGRAPH_RECURSION_LIMIT"] = "20"
        os.environ["CODEX_PROFILE"] = os.environ.get("CODEX_PROFILE", "smoke-low")

        config.setdefault("ralph_loop_defaults", {})["max_attempts"] = 1
        config.setdefault("ralph_loop_defaults", {})["min_attempts"] = 1
        config.setdefault("planning_pass_defaults", {})["enabled"] = False

        for agent_name in smoke_agents:
            agent_cfg = config["agents"][agent_name]

            # Force cheap/fast model for smoke testing.
            # This works because get_codex_model() prioritises agent_cfg["model"].
            agent_cfg["model"] = "gpt-5.4-mini"

            # Force no Ralph-loop retries.
            agent_cfg.setdefault("ralph_loop", {})["max_attempts"] = 1
            agent_cfg.setdefault("ralph_loop", {})["min_attempts"] = 1
            agent_cfg.setdefault("planning_pass", {})["enabled"] = False

            # Avoid overwriting real outputs.
            agent_cfg["output"] = str(out_dir / f"_smoke_{agent_name}.md")

            # Make validation check only "did it produce something?"
            agent_cfg["minimum_chars"] = 50
            agent_cfg["must_include"] = []

        config["global_rules"] = [
            *config.get("global_rules", []),
            (
                "SMOKE TEST MODE: produce a very short Markdown artifact, "
                "under 150 words. Prioritise speed over completeness. "
                "Do not use shell commands, apply_patch, or attempt to create/edit files. "
                "Return the Markdown artifact as your final response only; the Python wrapper "
                "will save it to disk."
            ),
        ]

        print(f"Smoke test mode: running agents once: {smoke_agents}")
    # Normal mode --------------------------------------------------------
    issues = preflight(
        config,
        idea_path=args.idea,
        previous_report_path=args.previous_report,
        feedback_path=args.feedback,
    )

    if issues:
        print("Preflight failed:\n", file=sys.stderr)
        for issue in issues:
            print(f"- {issue}", file=sys.stderr)
        sys.exit(1)

    agent_order = list(config["pipeline"]["order"])

    initial_state: ResearchState = {
        "idea": read_text(args.idea),
        "previous_report": read_optional(args.previous_report),
        "human_feedback": read_optional(args.feedback),
        "iteration_mode": args.feedback is not None,
        "config": config,
        "agent_order": agent_order,
        "current_agent": "",
        "attempts": {},
        "outputs": {},
        "feedback": {},
        "plans": {},
        "validation_passed": False,
    }

    graph = build_graph(agent_order)

    recursion_limit = int(os.environ.get("LANGGRAPH_RECURSION_LIMIT", "100"))

    final_state = graph.invoke(
        initial_state,
        {"recursion_limit": recursion_limit},
    )

    write_run_outputs(final_state, args=args, out_dir=out_dir)

    print("\nDone.\n")
    print(f"Run directory: {out_dir}")
    print(f"Combined report: {out_dir / 'report.md'}")
    print(f"Metadata: {out_dir / 'metadata.json'}")

    print("Outputs:")
    for agent_name in agent_order:
        output_path = final_state["outputs"].get(agent_name)
        if output_path:
            print(f"  {agent_name}: {output_path}")

    print("\nAttempts:")
    for agent_name, n in final_state["attempts"].items():
        print(f"  {agent_name}: {n}")


if __name__ == "__main__":
    main()
