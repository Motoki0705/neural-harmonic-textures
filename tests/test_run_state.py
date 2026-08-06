from __future__ import annotations

import json

from nht_pipeline.config import load_config
from nht_pipeline.pipeline import PipelineContext, run_pipeline
from nht_pipeline.run_state import RunState
from nht_pipeline.stages import descendants, execution_order


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
