from __future__ import annotations

import json
import os
from pathlib import Path

import jsonschema
import pytest

from nht_pipeline.config import (
    earliest_affected_stage,
    effective_start_stage,
    load_config,
    validate_config,
)
from nht_pipeline.pipeline import STAGE_EXECUTORS, PipelineContext, run_pipeline
from nht_pipeline.run_state import RunState
from nht_pipeline.stages import descendants, execution_order
from nht_pipeline.workspace import WorkspaceLock


def test_stage_graph_descendants_are_topological() -> None:
    assert execution_order("sfm") == (
        "sfm",
        "sfm_selection",
        "nht_training",
        "scene_export",
        "reconstruction_report",
    )
    assert descendants("scene_export") == ("reconstruction_report",)


def test_run_state_invalidates_stage_and_descendants(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.mark_running("frames")
    state.mark_completed("frames", {"frame_count": 10})
    (tmp_path / "frames/raw").mkdir(parents=True)
    (tmp_path / "frames/extraction.json").write_text("{}")
    state.mark_running("preprocess")
    state.mark_completed("preprocess", {"accepted": 9})
    (tmp_path / "frames/images").mkdir()
    (tmp_path / "frames/frames.json").write_text("{}")
    state.mark_running("sfm")
    state.mark_completed("sfm", {"candidates": 1})

    state.request("preprocess")
    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["stages"]["frames"]["status"] == "completed"
    assert payload["stages"]["preprocess"]["status"] == "invalidated"
    assert payload["stages"]["sfm"]["status"] == "invalidated"
    assert payload["stages"]["scene_export"]["status"] == "invalidated"


def test_run_state_rejects_cross_scene_reuse(tmp_path) -> None:
    RunState.create_or_load(tmp_path, "B00", None)
    try:
        RunState.create_or_load(tmp_path, "B01", None)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected a scene/workspace ownership error")


def test_partial_request_finishes_as_pending(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.request("frames", "frames")
    state.mark_running("frames")
    state.mark_completed("frames", {"frame_count": 10})
    state.finish_request()

    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["status"] == "pending"
    assert payload["requested_through_stage"] == "frames"


def test_missing_required_input_is_rejected_before_stage_runs(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.mark_running("frames")
    state.mark_completed("frames", {"frame_count": 10})

    try:
        state.mark_running("preprocess")
    except RuntimeError as error:
        assert "missing required inputs" in str(error)
    else:
        raise AssertionError("Expected required-input validation")


def test_pipeline_records_missing_upstream_output_as_stage_failure(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.payload["stages"]["frames"]["status"] = "completed"
    state.save()
    state.request("preprocess", "preprocess")
    context = PipelineContext(
        workspace=tmp_path,
        state=state,
        config=load_config(None),
        repository_root=tmp_path,
    )

    try:
        run_pipeline(context, "preprocess", "preprocess")
    except RuntimeError as error:
        assert "missing required inputs" in str(error)
    else:
        raise AssertionError("Expected required-input validation")

    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["status"] == "failed"
    assert payload["stages"]["preprocess"]["status"] == "failed"
    assert "missing required inputs" in payload["stages"]["preprocess"]["error"][
        "message"
    ]


def test_failed_rerun_physically_unpublishes_stale_descendants(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.payload["stages"]["frames"]["status"] = "completed"
    state.payload["stages"]["preprocess"]["status"] = "completed"
    state.save()
    (tmp_path / "export").mkdir()
    (tmp_path / "export/scene.json").write_text("stale")
    (tmp_path / "3dgs").mkdir()
    (tmp_path / "3dgs/training.json").write_text("stale")
    state.request("sfm", "sfm")
    context = PipelineContext(
        workspace=tmp_path,
        state=state,
        config=load_config(None),
        repository_root=tmp_path,
    )

    with pytest.raises(RuntimeError, match="missing required inputs"):
        run_pipeline(context, "sfm", "sfm")

    assert not (tmp_path / "export/scene.json").exists()
    assert not (tmp_path / "3dgs/training.json").exists()


def test_stage_publishes_only_after_staged_output_validation(
    tmp_path, monkeypatch
) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    state = RunState.create_or_load(tmp_path, "B00", video)
    state.request("frames", "frames")
    context = PipelineContext(
        workspace=tmp_path,
        state=state,
        config=load_config(None),
        repository_root=tmp_path,
    )

    def fake_frames(stage_context):
        raw = stage_context.output_root / "frames/raw"
        raw.mkdir(parents=True)
        (raw / "frame.jpg").write_bytes(b"frame")
        output = stage_context.output_root / "frames/extraction.json"
        output.write_text("{}")
        assert not (tmp_path / "frames/extraction.json").exists()
        return {"frame_count": 1}

    monkeypatch.setattr(
        "nht_pipeline.pipeline.preflight_stage", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setitem(STAGE_EXECUTORS, "frames", fake_frames)
    run_pipeline(context, "frames", "frames")

    assert (tmp_path / "frames/extraction.json").is_file()
    assert not (tmp_path / ".staging").exists()
    assert (tmp_path / "logs/frames/attempt-1.log").is_file()
    assert state.payload["stages"]["frames"]["status"] == "completed"


def test_failed_stage_discards_temporary_output(tmp_path, monkeypatch) -> None:
    video = tmp_path / "video.mp4"
    video.write_bytes(b"fixture")
    state = RunState.create_or_load(tmp_path, "B00", video)
    state.request("frames", "frames")
    context = PipelineContext(
        workspace=tmp_path,
        state=state,
        config=load_config(None),
        repository_root=tmp_path,
    )

    def failing_frames(stage_context):
        output = stage_context.output_root / "frames/extraction.json"
        output.parent.mkdir(parents=True)
        output.write_text("partial")
        raise RuntimeError("synthetic process signal")

    monkeypatch.setattr(
        "nht_pipeline.pipeline.preflight_stage", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setitem(STAGE_EXECUTORS, "frames", failing_frames)
    with pytest.raises(RuntimeError, match="synthetic process signal"):
        run_pipeline(context, "frames", "frames")

    assert not (tmp_path / "frames/extraction.json").exists()
    assert not (tmp_path / ".staging/frames").exists()
    error = state.payload["stages"]["frames"]["error"]
    assert error["category"] == "process_signal"

    schema_path = Path(__file__).resolve().parents[1] / "schemas/run.schema.json"
    jsonschema.Draft202012Validator(
        json.loads(schema_path.read_text()),
        format_checker=jsonschema.FormatChecker(),
    ).validate(json.loads((tmp_path / "run.json").read_text()))


def test_workspace_lock_refuses_live_parallel_process(tmp_path) -> None:
    with WorkspaceLock(tmp_path), pytest.raises(RuntimeError, match="already locked"):
        WorkspaceLock(tmp_path).__enter__()


def test_workspace_lock_recovers_dead_process_record(tmp_path) -> None:
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / ".pipeline.lock").write_text(
        json.dumps({"pid": 999_999_999, "workspace": str(tmp_path)})
    )
    with WorkspaceLock(tmp_path) as lock:
        assert lock.recovered_stale_lock is True
        assert json.loads((tmp_path / ".pipeline.lock").read_text())["pid"] == os.getpid()
    assert not (tmp_path / ".pipeline.lock").exists()


def test_config_change_expands_request_to_owning_stage() -> None:
    previous = load_config(None)
    changed = load_config(None)
    changed["preprocess"]["minimum_temporal_difference"] = 2.0
    affected = earliest_affected_stage(previous, changed)

    assert affected == "preprocess"
    assert (
        effective_start_stage(
            "nht_training", affected, input_video_changed=False
        )
        == "preprocess"
    )
    assert (
        effective_start_stage(
            "scene_export", None, input_video_changed=True
        )
        == "frames"
    )


def test_stage_record_stores_actual_config_subset(tmp_path) -> None:
    state = RunState.create_or_load(tmp_path, "B00", tmp_path / "video.mp4")
    state.mark_running("frames", {"frames": {"frames_per_second": 2.0}})
    assert state.payload["stages"]["frames"]["config"] == {
        "frames": {"frames_per_second": 2.0}
    }


def test_config_rejects_backend_camera_sharing_mismatch() -> None:
    config = load_config(None)
    config["sfm"]["candidates"][1]["camera_sharing"] = "segments"

    with pytest.raises(ValueError, match="camera_sharing"):
        validate_config(config)


def test_config_rejects_invalid_segment_size() -> None:
    config = load_config(None)
    candidate = config["sfm"]["candidates"][0]
    candidate["camera_sharing"] = "segments"
    candidate["camera_segment_size"] = 1

    with pytest.raises(ValueError, match="camera_segment_size"):
        validate_config(config)
