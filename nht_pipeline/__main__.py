"""Run video extraction through robust SfM, NHT training, and scene export."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config, write_resolved_config
from .pipeline import PipelineContext, run_pipeline
from .run_state import RunState
from .stages import stage_names


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
    manifest_path = workspace / "run.json"
    if manifest_path.exists():
        state = RunState.load(workspace)
        if state.payload["scene_id"] != args.scene_id:
            raise SystemExit(
                f"Workspace belongs to {state.payload['scene_id']!r}, not {args.scene_id!r}"
            )
        if args.input_video is not None:
            incoming = str(args.input_video.resolve())
            existing = state.payload.get("input_video")
            if existing and incoming != existing and args.from_stage != "frames":
                raise SystemExit("A changed input video requires --from-stage frames")
            state.payload["input_video"] = incoming
    else:
        state = RunState.create_or_load(workspace, args.scene_id, args.input_video)

    resolved_path = workspace / "resolved-config.yaml"
    if args.config is not None:
        config = load_config(args.config)
    elif resolved_path.exists():
        config = load_config(resolved_path)
    else:
        config = load_config(None)
    write_resolved_config(resolved_path, config)
    state.request(args.from_stage, args.through_stage)
    context = PipelineContext(
        workspace=workspace,
        state=state,
        config=config,
        repository_root=Path(__file__).resolve().parents[1],
    )
    run_pipeline(context, args.from_stage, args.through_stage)


if __name__ == "__main__":
    main()
