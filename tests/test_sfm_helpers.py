from __future__ import annotations

import json

import numpy as np

from nht_pipeline.config import load_config
from nht_pipeline.pipeline import PipelineContext, _run_sfm_selection
from nht_pipeline.run_state import RunState
from nht_pipeline.sfm.compare import similarity_alignment
from nht_pipeline.sfm.metrics import trajectory_metrics
from nht_pipeline.sfm.pairs import sequential_pairs
from nht_pipeline.sfm.select import NoValidCandidateError, select_candidate


def test_sequential_pairs_are_unique_and_bounded() -> None:
    names = [f"frame_{index:03}.jpg" for index in range(5)]
    pairs = sequential_pairs(names, overlap=2)
    assert len(pairs) == 7
    assert ("frame_000.jpg", "frame_002.jpg") in pairs
    assert ("frame_000.jpg", "frame_003.jpg") not in pairs


def test_similarity_alignment_recovers_transform() -> None:
    source = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    target = 2.5 * (source @ rotation.T) + np.asarray([4.0, -2.0, 7.0])
    scale, recovered_rotation, translation = similarity_alignment(source, target)
    aligned = scale * (source @ recovered_rotation.T) + translation
    np.testing.assert_allclose(aligned, target, atol=1e-10)


def test_trajectory_metrics_detects_a_pose_spike() -> None:
    centers = np.zeros((8, 3), dtype=np.float64)
    centers[:, 0] = np.arange(8)
    centers[4, 1] = 30.0
    rotations = np.repeat(np.eye(3)[None], len(centers), axis=0)
    metrics = trajectory_metrics(centers, rotations, np.arange(len(centers)))
    assert metrics["maximum_step_to_median"] > 5.0
    assert metrics["step_outlier_count"] >= 1


def test_trajectory_metrics_counts_near_duplicate_pose() -> None:
    centers = np.zeros((5, 3), dtype=np.float64)
    centers[:, 0] = [0.0, 1.0, 1.001, 2.0, 3.0]
    rotations = np.repeat(np.eye(3)[None], len(centers), axis=0)
    metrics = trajectory_metrics(centers, rotations, np.arange(len(centers)))
    assert metrics["near_duplicate_step_count"] == 1
    assert metrics["near_duplicate_fraction"] == 0.25


def _candidate(
    identifier: str,
    *,
    accepted: bool,
    supported: int,
    points: int,
    reprojection: float = 2.0,
) -> dict:
    return {
        "id": identifier,
        "backend": identifier,
        "metrics": {
            "accepted": accepted,
            "supported_registered_images": supported,
            "supported_registration_ratio": supported / 100,
            "sparse_points": points,
            "p95_reprojection_error_px": reprojection,
            "median_track_length": 4.0,
            "trajectory": {"maximum_step_to_median": 1.2},
            "gates": {},
        },
    }


def test_selection_rejects_invalid_and_prefers_supported_coverage() -> None:
    result = select_candidate(
        [
            _candidate("many-points", accepted=True, supported=95, points=500_000),
            _candidate("coverage", accepted=True, supported=100, points=80_000),
            _candidate("invalid", accepted=False, supported=100, points=1_000_000),
        ]
    )
    assert result["selected_candidate"] == "coverage"
    assert result["rejected_candidates"] == ["invalid"]


def test_selection_fails_when_every_candidate_is_invalid() -> None:
    try:
        select_candidate([_candidate("bad", accepted=False, supported=20, points=10)])
    except NoValidCandidateError:
        pass
    else:
        raise AssertionError("Expected selection to reject all invalid candidates")


def test_all_invalid_selection_writes_failure_comparison(tmp_path) -> None:
    candidate = _candidate("bad", accepted=False, supported=20, points=10)
    candidate.update({"status": "completed", "model": "bad/model"})
    candidates_path = tmp_path / "sfm/candidates/candidates.json"
    candidates_path.parent.mkdir(parents=True)
    candidates_path.write_text(
        json.dumps(
            {
                "schema": "nht_sfm_candidates_v1",
                "input_images": 100,
                "completed_candidates": 1,
                "candidates": [candidate],
            }
        )
    )
    state = RunState.create_or_load(tmp_path, "B00", None)
    context = PipelineContext(tmp_path, state, load_config(None), tmp_path)

    try:
        _run_sfm_selection(context)
    except NoValidCandidateError:
        pass
    else:
        raise AssertionError("Expected selection to reject all invalid candidates")

    comparison = json.loads(
        (tmp_path / "sfm/diagnostics/candidate-comparison.json").read_text()
    )
    assert comparison["selection"]["status"] == "failed"
    assert comparison["selection"]["selected_candidate"] is None
    assert comparison["candidates"][0]["id"] == "bad"
