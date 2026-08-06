"""Run video extraction through robust SfM, NHT training, and scene export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    earliest_affected_stage,
    effective_start_stage,
    load_config,
    write_resolved_config,
)
from .pipeline import PipelineContext, run_pipeline
from .run_state import RunState
from .stages import STAGES, stage_names
from .workspace import (
    WorkspaceLock,
    cleanup_staging,
    invalidate_published_outputs,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--input-video", type=Path)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--from-stage", choices=stage_names(), default="frames")
    parser.add_argument(
        "--through-stage",
        choices=stage_names(),
        help="Stop after this stage; primarily useful for research and diagnostics.",
    )
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    workspace = args.workspace.resolve()
    with WorkspaceLock(workspace) as lock:
        manifest_path = workspace / "run.json"
        resolved_path = workspace / "resolved-config.yaml"
        previous_config = load_config(resolved_path) if resolved_path.exists() else None
        if args.config is not None:
            config = load_config(args.config)
        elif previous_config is not None:
            config = previous_config
        else:
            config = load_config(None)

        if manifest_path.exists():
            state = RunState.load(workspace)
            if state.payload["scene_id"] != args.scene_id:
                raise SystemExit(
                    f"Workspace belongs to {state.payload['scene_id']!r}, "
                    f"not {args.scene_id!r}"
                )
            interrupted = state.recover_interrupted()
            if interrupted:
                order = [stage.name for stage in STAGES]
                earliest = min(interrupted, key=order.index)
                invalidate_published_outputs(workspace, earliest)
            if lock.recovered_stale_lock or interrupted:
                cleanup_staging(workspace)
        else:
            state = RunState.create_or_load(workspace, args.scene_id, args.input_video)

        existing_video = state.payload.get("input_video")
        incoming_video = (
            str(args.input_video.resolve()) if args.input_video is not None else None
        )
        input_changed = bool(
            incoming_video and existing_video and incoming_video != existing_video
        )
        if incoming_video is not None:
            state.payload["input_video"] = incoming_video
        affected = earliest_affected_stage(previous_config, config)
        effective = effective_start_stage(
            args.from_stage, affected, input_video_changed=input_changed
        )
        state.request(args.from_stage, args.through_stage, effective)
        write_resolved_config(resolved_path, config)
        context = PipelineContext(
            workspace=workspace,
            state=state,
            config=config,
            repository_root=Path(__file__).resolve().parents[1],
        )
        run_pipeline(context, effective, args.through_stage)


if __name__ == "__main__":
    main()
