"""Mutable, atomic run manifest for one canonical scene workspace."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .stages import STAGE_BY_NAME, STAGES, descendants, execution_order

VALID_STATUSES = {
    "pending",
    "running",
    "completed",
    "failed",
    "invalidated",
    "skipped",
}


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


class RunState:
    def __init__(self, workspace: Path, payload: dict[str, Any]):
        self.workspace = workspace.resolve()
        self.path = self.workspace / "run.json"
        self.payload = payload

    @classmethod
    def create_or_load(
        cls,
        workspace: Path,
        scene_id: str,
        input_video: Path | None,
    ) -> RunState:
        if not scene_id or "/" in scene_id or "\\" in scene_id:
            raise ValueError("scene_id must be a non-empty path-safe identifier")
        workspace = workspace.resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        path = workspace / "run.json"
        payload: dict[str, Any]
        if path.exists():
            payload = json.loads(path.read_text())
            if payload.get("scene_id") != scene_id:
                raise ValueError(
                    f"Workspace belongs to scene {payload.get('scene_id')!r}, "
                    f"not {scene_id!r}"
                )
            if input_video is not None:
                payload["input_video"] = str(input_video.resolve())
            return cls(workspace, payload)

        payload = {
            "schema": "nht_pipeline_run_v1",
            "scene_id": scene_id,
            "status": "pending",
            "input_video": str(input_video.resolve()) if input_video else None,
            "resolved_config": "resolved-config.yaml",
            "requested_from_stage": None,
            "requested_through_stage": None,
            "created_at_utc": _now(),
            "updated_at_utc": _now(),
            "stages": {
                stage.name: {
                    "status": "pending",
                    "attempts": 0,
                    "outputs": [str(path) for path in stage.fixed_outputs],
                    "summary": None,
                    "error": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                }
                for stage in STAGES
            },
        }
        state = cls(workspace, payload)
        state.save()
        return state

    @classmethod
    def load(cls, workspace: Path) -> RunState:
        workspace = workspace.resolve()
        path = workspace / "run.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing run manifest: {path}")
        return cls(workspace, json.loads(path.read_text()))

    def save(self) -> None:
        self.payload["updated_at_utc"] = _now()
        _atomic_json(self.path, self.payload)

    def request(self, from_stage: str, through_stage: str | None = None) -> None:
        if from_stage not in STAGE_BY_NAME:
            raise KeyError(f"Unknown stage: {from_stage}")
        if through_stage is not None and through_stage not in execution_order(
            from_stage
        ):
            raise ValueError(
                f"through_stage {through_stage} does not follow {from_stage}"
            )
        self.payload["requested_from_stage"] = from_stage
        self.payload["requested_through_stage"] = through_stage
        self.payload["status"] = "running"
        for name in descendants(from_stage, include_self=True):
            stage = self.payload["stages"][name]
            stage.update(
                {
                    "status": "invalidated",
                    "summary": None,
                    "error": None,
                    "started_at_utc": None,
                    "finished_at_utc": None,
                }
            )
        self.save()

    def finish_request(self) -> None:
        """Close a successful CLI request, including a diagnostic partial run."""
        if any(
            record["status"] == "running" for record in self.payload["stages"].values()
        ):
            raise RuntimeError("Cannot finish a request while a stage is running")
        complete = all(
            record["status"] in {"completed", "skipped"}
            for record in self.payload["stages"].values()
        )
        self.payload["status"] = "completed" if complete else "pending"
        self.save()

    def require_dependencies(self, stage_name: str) -> None:
        missing = [
            dependency
            for dependency in STAGE_BY_NAME[stage_name].dependencies
            if self.payload["stages"][dependency]["status"] != "completed"
        ]
        if missing:
            raise RuntimeError(
                f"Stage {stage_name} requires completed dependencies: {missing}"
            )
        absent = [
            str(path)
            for path in STAGE_BY_NAME[stage_name].required_inputs
            if not (self.workspace / path).exists()
        ]
        if absent:
            raise RuntimeError(
                f"Stage {stage_name} is missing required inputs: {absent}"
            )

    def validate_outputs(self, stage_name: str) -> None:
        absent = [
            str(path)
            for path in STAGE_BY_NAME[stage_name].fixed_outputs
            if not (self.workspace / path).exists()
        ]
        if absent:
            raise RuntimeError(
                f"Stage {stage_name} did not produce fixed outputs: {absent}"
            )

    def mark_running(self, stage_name: str) -> None:
        self.require_dependencies(stage_name)
        stage = self.payload["stages"][stage_name]
        stage["status"] = "running"
        stage["attempts"] = int(stage.get("attempts", 0)) + 1
        stage["started_at_utc"] = _now()
        stage["finished_at_utc"] = None
        stage["summary"] = None
        stage["error"] = None
        self.payload["status"] = "running"
        self.save()

    def mark_completed(self, stage_name: str, summary: dict[str, Any]) -> None:
        stage = self.payload["stages"][stage_name]
        stage["status"] = "completed"
        stage["summary"] = summary
        stage["error"] = None
        stage["finished_at_utc"] = _now()
        if all(
            record["status"] in {"completed", "skipped"}
            for record in self.payload["stages"].values()
        ):
            self.payload["status"] = "completed"
        self.save()

    def mark_failed(self, stage_name: str, error: BaseException | str) -> None:
        stage = self.payload["stages"][stage_name]
        stage["status"] = "failed"
        stage["error"] = {
            "type": type(error).__name__ if isinstance(error, BaseException) else None,
            "message": str(error),
        }
        stage["finished_at_utc"] = _now()
        self.payload["status"] = "failed"
        self.save()
