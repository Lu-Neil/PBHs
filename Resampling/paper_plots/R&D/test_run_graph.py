from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
import yaml

import run_graph


def minimal_config_dict() -> dict:
    return {
        "schema_version": 2,
        "global_rules": [],
        "runtime": {
            "command_prefix": [],
            "scratch_root": "/tmp",
            "codex_timeout_seconds": 30,
            "codex_sandbox": "workspace-write",
        },
        "validation": {"mode": "deterministic"},
        "defaults": {
            "retry": {"enabled": True, "min_attempts": 1, "max_attempts": 1},
            "planning": {"enabled": False, "instructions": []},
        },
        "pipeline": {"order": ["first", "second"]},
        "tasks": {
            "first": {
                "purpose": "Produce the first artifact.",
                "kind": "planning",
                "depends_on": [],
                "outputs": [
                    {
                        "path": "first.md",
                        "required": True,
                        "primary": True,
                        "kind": "markdown",
                    }
                ],
                "validators": [
                    {
                        "name": "complete",
                        "type": "artifact_completeness",
                        "options": {"minimum_chars": 10},
                    }
                ],
                "must_include": ["required magic"],
                "on_failure": "stop",
            },
            "second": {
                "purpose": "Consume the first artifact.",
                "kind": "analysis",
                "depends_on": ["first"],
                "outputs": [
                    {
                        "path": "second.md",
                        "required": True,
                        "primary": True,
                        "kind": "markdown",
                    }
                ],
                "validators": [
                    {
                        "name": "complete",
                        "type": "artifact_completeness",
                        "options": {"minimum_chars": 10},
                    }
                ],
                "must_include": [],
                "on_failure": "stop",
            },
        },
    }


def namespace(tmp_path: Path, config_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        config=str(config_path),
        idea="idea.md",
        previous_report=None,
        feedback=None,
        out=str(tmp_path / "run"),
    )


def test_project_config_is_strict_and_contains_experiment() -> None:
    config = run_graph.load_config(run_graph.DEFAULT_CONFIG)
    assert config.defaults.retry.min_attempts == 1
    assert "numerical_experiment" in config.tasks
    experiment = config.tasks["numerical_experiment"]
    assert experiment.kind == "experiment"
    assert {output.kind for output in experiment.outputs} >= {
        "python",
        "json",
        "csv",
        "log",
        "plot",
    }


def test_unknown_configuration_fields_are_rejected() -> None:
    raw = minimal_config_dict()
    raw["tasks"]["first"]["decorative_unimplemented_field"] = True
    with pytest.raises(ValidationError, match="decorative_unimplemented_field"):
        run_graph.HarnessConfig.model_validate(raw)


def test_dependencies_must_precede_consumers() -> None:
    raw = minimal_config_dict()
    raw["pipeline"]["order"] = ["second", "first"]
    with pytest.raises(ValidationError, match="must occur earlier"):
        run_graph.HarnessConfig.model_validate(raw)


def test_selection_uses_ancestors_and_rerun_uses_descendants(tmp_path: Path) -> None:
    config = run_graph.HarnessConfig.model_validate(minimal_config_dict())
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(minimal_config_dict()), encoding="utf-8")
    state = run_graph.new_state(
        config,
        tmp_path / "run",
        args=namespace(tmp_path, config_path),
    )

    selected = run_graph.configure_selection(
        config,
        state,
        select_names=["second"],
        rerun_names=None,
        has_base_state=False,
    )
    assert selected == {"first", "second"}

    for record in state["tasks"].values():
        record["status"] = "reused"
    rerun = run_graph.configure_selection(
        config,
        state,
        select_names=None,
        rerun_names=["first"],
        has_base_state=True,
    )
    assert rerun == {"first", "second"}
    assert all(state["tasks"][name]["status"] == "pending" for name in rerun)


def test_resume_does_not_run_intentionally_skipped_tasks(tmp_path: Path) -> None:
    config = run_graph.HarnessConfig.model_validate(minimal_config_dict())
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(minimal_config_dict()), encoding="utf-8")
    state = run_graph.new_state(
        config,
        tmp_path / "run",
        args=namespace(tmp_path, config_path),
    )
    state["tasks"]["first"]["status"] = "accepted"
    state["tasks"]["second"]["status"] = "skipped"

    effective = run_graph.configure_selection(
        config,
        state,
        select_names=None,
        rerun_names=None,
        has_base_state=True,
    )
    assert effective == set()
    assert state["tasks"]["second"]["status"] == "skipped"


def test_experiment_validator_checks_machine_readable_artifacts(
    tmp_path: Path,
) -> None:
    task = run_graph.TaskSpec.model_validate(
        {
            "purpose": "Run an experiment.",
            "kind": "experiment",
            "outputs": [
                {
                    "path": "report.md",
                    "primary": True,
                    "kind": "markdown",
                },
                {"path": "experiment.py", "kind": "python"},
                {"path": "parameters.json", "kind": "json"},
                {"path": "results.csv", "kind": "csv"},
                {"path": "commands.log", "kind": "log"},
                {"path": "plot.png", "kind": "plot"},
            ],
            "validators": [
                {
                    "name": "experiment",
                    "type": "experiment_artifacts",
                    "options": {},
                }
            ],
        }
    )
    (tmp_path / "report.md").write_text(
        "Commands, parameters, results, and failures.", encoding="utf-8"
    )
    (tmp_path / "experiment.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "parameters.json").write_text('{"seed": 1}\n', encoding="utf-8")
    (tmp_path / "results.csv").write_text("name,value\nbaseline,1\n", encoding="utf-8")
    (tmp_path / "commands.log").write_text("python experiment.py\n", encoding="utf-8")
    (tmp_path / "plot.png").write_bytes(b"not-empty")

    spec = task.validators[0]
    valid = run_graph.validate_experiment_artifacts(spec, task, tmp_path)
    assert valid["passed"] is True

    (tmp_path / "results.csv").write_text("name,value\n", encoding="utf-8")
    invalid = run_graph.validate_experiment_artifacts(spec, task, tmp_path)
    assert invalid["passed"] is False
    assert "at least one data row" in invalid["feedback"]


def test_required_failure_stops_and_blocks_downstream(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(minimal_config_dict()), encoding="utf-8")
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("A small test idea.", encoding="utf-8")
    out_dir = tmp_path / "run"

    def fake_run_codex(
        prompt: str,
        output_path: Path,
        *,
        transcript_path: Path,
        stderr_path: Path,
        **kwargs,
    ) -> run_graph.InvocationResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # Long enough, but intentionally missing "required magic".
        output_path.write_text("This artifact is deliberately incorrect.", encoding="utf-8")
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("{}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return run_graph.InvocationResult(
            text=output_path.read_text(encoding="utf-8"),
            model=None,
            duration_seconds=0.0,
            prompt_sha256="test",
            transcript_path=str(transcript_path),
            stderr_path=str(stderr_path),
            usage=None,
            session_id=None,
            command=["codex", "exec"],
        )

    monkeypatch.setattr(run_graph, "run_codex", fake_run_codex)
    exit_code = run_graph.main(
        [
            "--config",
            str(config_path),
            "--idea",
            str(idea_path),
            "--out",
            str(out_dir),
            "--no-synthesis",
        ]
    )
    assert exit_code == 1
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["tasks"]["first"]["status"] == "rejected"
    assert metadata["tasks"]["second"]["status"] == "blocked"
    assert metadata["status"] == "failed"


def test_passing_tasks_are_not_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(minimal_config_dict()), encoding="utf-8")
    idea_path = tmp_path / "idea.md"
    idea_path.write_text("A small test idea.", encoding="utf-8")
    out_dir = tmp_path / "run"

    def fake_run_codex(
        prompt: str,
        output_path: Path,
        *,
        transcript_path: Path,
        stderr_path: Path,
        **kwargs,
    ) -> run_graph.InvocationResult:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            "This artifact contains the required magic and useful detail.",
            encoding="utf-8",
        )
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text("{}\n", encoding="utf-8")
        stderr_path.write_text("", encoding="utf-8")
        return run_graph.InvocationResult(
            text=output_path.read_text(encoding="utf-8"),
            model="test",
            duration_seconds=0.0,
            prompt_sha256="test",
            transcript_path=str(transcript_path),
            stderr_path=str(stderr_path),
            usage=None,
            session_id=None,
            command=["codex", "exec"],
        )

    monkeypatch.setattr(run_graph, "run_codex", fake_run_codex)
    exit_code = run_graph.main(
        [
            "--config",
            str(config_path),
            "--idea",
            str(idea_path),
            "--out",
            str(out_dir),
            "--no-synthesis",
        ]
    )
    assert exit_code == 0
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["status"] == "completed"
    assert metadata["tasks"]["first"]["attempts"] == 1
    assert metadata["tasks"]["second"]["attempts"] == 1


def test_resume_rejects_changed_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = minimal_config_dict()
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = run_graph.HarnessConfig.model_validate(raw)
    args = namespace(tmp_path, config_path)
    out_dir = Path(args.out)
    out_dir.mkdir()
    state = run_graph.new_state(config, out_dir, args=args)
    run_graph.write_json(out_dir / "metadata.json", state)

    raw["global_rules"].append("A changed rule")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    exit_code = run_graph.main(
        [
            "--config",
            str(config_path),
            "--resume",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 2


def test_resume_dry_run_does_not_mutate_existing_metadata(tmp_path: Path) -> None:
    raw = minimal_config_dict()
    config_path = tmp_path / "agents.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    config = run_graph.HarnessConfig.model_validate(raw)
    args = namespace(tmp_path, config_path)
    out_dir = Path(args.out)
    out_dir.mkdir()
    state = run_graph.new_state(config, out_dir, args=args)
    state["status"] = "completed"
    for record in state["tasks"].values():
        record["status"] = "skipped"
    metadata_path = out_dir / "metadata.json"
    run_graph.write_json(metadata_path, state)
    before = metadata_path.read_bytes()

    exit_code = run_graph.main(
        [
            "--config",
            str(config_path),
            "--resume",
            str(out_dir),
            "--dry-run",
        ]
    )
    assert exit_code == 0
    assert metadata_path.read_bytes() == before
