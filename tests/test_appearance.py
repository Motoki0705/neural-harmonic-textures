from __future__ import annotations

import json
import math

import numpy as np
import pytest
from test_export import _valid_export

from nht_pipeline.appearance import (
    AppearanceControls,
    AppearancePolicy,
    blended_log_scales,
    blended_opacities,
    blended_ray_channels,
    gaussian_splat_progress,
    largest_first_ranks,
    optical_axis_ray_channels,
)
from nht_pipeline.render import render_scene

POLICY = {"point_radius": 0.01, "splat_order": "largest_first", "splat_stagger": 0.25}


def test_largest_first_ranks_order_by_largest_axis_and_keep_ties_stable() -> None:
    log_scales = np.asarray(
        [[-3.0, -3.0, -1.0], [0.0, -5.0, -5.0], [-2.0, -2.0, -2.0], [-1.0, -9.0, -9.0]]
    )

    ranks = largest_first_ranks(log_scales)

    # Largest axes: -1, 0, -2, -1 -> order 1, (0, 3 tie in checkpoint order), 2.
    np.testing.assert_allclose(ranks, [1 / 3, 0.0, 1.0, 2 / 3])


def test_largest_first_ranks_rejects_malformed_scales() -> None:
    with pytest.raises(ValueError, match="shape"):
        largest_first_ranks(np.zeros((0, 3)))
    with pytest.raises(ValueError, match="finite"):
        largest_first_ranks(np.asarray([[0.0, np.nan, 0.0]]))


def test_splat_progress_keeps_points_at_zero_and_completes_at_one() -> None:
    ranks = np.linspace(0.0, 1.0, 11)

    np.testing.assert_array_equal(gaussian_splat_progress(ranks, 0.0, 0.25), 0.0)
    np.testing.assert_array_equal(gaussian_splat_progress(ranks, 1.0, 0.25), 1.0)
    middle = gaussian_splat_progress(ranks, 0.5, 0.25)
    assert np.all(np.diff(middle) <= 0.0), "larger Gaussians must grow first"
    assert middle[0] == 1.0 and middle[-1] == 0.0


def test_blends_reproduce_point_and_learned_endpoints() -> None:
    learned = np.asarray([[-1.0, -2.0, -3.0], [0.5, 0.0, -0.5]])
    radius = 0.02

    points = blended_log_scales(learned, np.zeros(2), radius)
    np.testing.assert_allclose(points, math.log(radius))
    np.testing.assert_allclose(blended_log_scales(learned, np.ones(2), radius), learned)

    opacities = np.asarray([0.1, 0.9])
    np.testing.assert_allclose(blended_opacities(opacities, np.zeros(2)), 1.0)
    np.testing.assert_allclose(blended_opacities(opacities, np.ones(2)), opacities)


def test_optical_axis_ray_channels_match_rasterizer_encoding() -> None:
    channels = optical_axis_ray_channels(np.eye(4), 3.0)

    np.testing.assert_allclose(channels, [0.5, 0.5, 2.0])
    actual = np.full((2, 2, 3), 0.25)
    np.testing.assert_allclose(blended_ray_channels(actual, channels, 1.0), actual)
    np.testing.assert_allclose(
        blended_ray_channels(actual, channels, 0.0), np.broadcast_to(channels, actual.shape)
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [("splat_progress", -0.1), ("splat_progress", 1.5), ("view_dependence", math.nan)],
)
def test_controls_reject_values_outside_unit_interval(field, value) -> None:
    payload = {"splat_progress": 0.0, "view_dependence": 0.0}
    payload[field] = value
    with pytest.raises(ValueError, match=field):
        AppearanceControls.from_payload(payload)


def test_policy_rejects_nonpositive_radius_and_unknown_order() -> None:
    with pytest.raises(ValueError, match="point_radius"):
        AppearancePolicy.from_payload({**POLICY, "point_radius": 0.0})
    with pytest.raises(ValueError, match="splat_order"):
        AppearancePolicy.from_payload({**POLICY, "splat_order": "random"})


def _staged_request(cameras, controls, *, policy=POLICY):
    camera = cameras["cameras"][0]
    payload = {"schema": "nht_render_request_v1", "cameras": []}
    for index, control in enumerate(controls):
        item = {
            "camera_id": f"stage-{index}",
            "width": camera["width"],
            "height": camera["height"],
            "intrinsics": camera["intrinsics"],
            "camera_to_scene": camera["camera_to_scene"],
        }
        if control is not None:
            item["appearance"] = control
        payload["cameras"].append(item)
    if policy is not None:
        payload["appearance_policy"] = policy
    return payload


def _patch_recording_renderer(monkeypatch, batches):
    class FakeRenderer:
        def __init__(self, checkpoint, config):
            del checkpoint, config

        def render_batch(self, requests, policy=None):
            batches.append(([item["camera_id"] for item in requests], policy))
            shape = (requests[0]["height"], requests[0]["width"])
            return [
                (
                    np.full((*shape, 3), 0.5, dtype=np.float32),
                    np.ones((*shape, 1), dtype=np.float32),
                    np.ones((*shape, 1), dtype=np.float32),
                )
                for _ in requests
            ]

    monkeypatch.setattr("nht_pipeline.render.ResidentSceneRenderer", FakeRenderer)


def test_render_splits_batches_by_appearance_and_echoes_controls(
    tmp_path, monkeypatch
) -> None:
    cameras = _valid_export(tmp_path)
    first = {"splat_progress": 0.0, "view_dependence": 0.0}
    second = {"splat_progress": 0.5, "view_dependence": 0.0}
    request_path = tmp_path / "request.json"
    request_path.write_text(
        json.dumps(_staged_request(cameras, [first, first, second, None]))
    )
    batches: list[tuple[list[str], AppearancePolicy | None]] = []
    _patch_recording_renderer(monkeypatch, batches)

    result = render_scene(
        tmp_path / "scene.json",
        tmp_path.parent / f"{tmp_path.name}-render",
        request_path=request_path,
    )

    policy = AppearancePolicy.from_payload(POLICY)
    assert batches == [
        (["stage-0", "stage-1"], policy),
        (["stage-2"], policy),
        (["stage-3"], None),
    ]
    records = result["renders"]
    assert records[0]["appearance"] == {"policy": POLICY, "controls": first}
    assert records[2]["appearance"]["controls"] == second
    assert "appearance" not in records[3]


def test_render_rejects_controls_without_policy_and_unused_policy(tmp_path) -> None:
    cameras = _valid_export(tmp_path)
    controls = {"splat_progress": 0.0, "view_dependence": 0.0}
    request_path = tmp_path / "request.json"
    output = tmp_path.parent / f"{tmp_path.name}-render"

    request_path.write_text(
        json.dumps(_staged_request(cameras, [controls], policy=None))
    )
    with pytest.raises(ValueError, match="appearance_policy"):
        render_scene(tmp_path / "scene.json", output, request_path=request_path)

    request_path.write_text(json.dumps(_staged_request(cameras, [None])))
    with pytest.raises(ValueError, match="no camera uses appearance"):
        render_scene(tmp_path / "scene.json", output, request_path=request_path)


def test_render_request_schema_rejects_out_of_range_controls(tmp_path) -> None:
    cameras = _valid_export(tmp_path)
    controls = {"splat_progress": 1.5, "view_dependence": 0.0}
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(_staged_request(cameras, [controls])))

    with pytest.raises(ValueError, match="render-request schema"):
        render_scene(
            tmp_path / "scene.json",
            tmp_path.parent / f"{tmp_path.name}-render",
            request_path=request_path,
        )
