from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from nht_pipeline.export import validate_scene_export
from nht_pipeline.render import render_scene


def _valid_export(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (16, 12)).save(tmp_path / "images/frame_000000.jpg")
    (tmp_path / "model/ckpts").mkdir(parents=True)
    (tmp_path / "model/ckpts/model.pt").write_bytes(b"checkpoint")
    points = np.asarray([[0, 0, 0, 1, 0.5, 0]], dtype=np.float32)
    np.save(tmp_path / "points_scene.npy", points)
    cameras = {
        "schema": "nht_standard_cameras_v1",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "cameras": [
            {
                "camera_id": "frame_000000",
                "source_frame_index": 0,
                "time_seconds": 0.0,
                "width": 16,
                "height": 12,
                "intrinsics": {
                    "model": "PINHOLE",
                    "distortion_model": "NONE",
                    "params": [10.0, 10.0, 8.0, 6.0],
                    "matrix": [[10, 0, 8], [0, 10, 6], [0, 0, 1]],
                },
                "camera_to_scene": np.eye(4).tolist(),
                "image": "images/frame_000000.jpg",
            }
        ],
    }
    (tmp_path / "cameras.json").write_text(json.dumps(cameras))
    scene = {
        "schema": "nht_standard_scene_v1",
        "scene_id": "B00",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "scene_coordinate_convention": "right-handed",
        "pixel_coordinate_convention": "top-left",
        "image_resolution_semantics": "full resolution",
        "camera_count": 1,
        "cameras": "cameras.json",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": [1, 6],
            "dtype": "float32",
        },
        "scene_from_sfm": np.eye(4).tolist(),
        "sfm_from_scene": np.eye(4).tolist(),
        "normalization": {
            "applied": True,
            "camera_similarity": np.eye(4).tolist(),
            "principal_axis_alignment": np.eye(4).tolist(),
            "upside_down_correction": np.eye(4).tolist(),
        },
        "model_root": "model",
        "renderer": {
            "command": "nht-render",
            "checkpoint": "model/ckpts/model.pt",
            "runtime_config": "model/runtime-config.json",
        },
        "capabilities": ["nht_rendering_model"],
    }
    (tmp_path / "model/runtime-config.json").write_text("{}")
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    return cameras


def test_export_validator_accepts_semantically_consistent_scene(tmp_path) -> None:
    _valid_export(tmp_path)
    result = validate_scene_export(tmp_path)
    assert result["schema"] == "nht_standard_scene_v1"
    assert result["camera_count"] == 1
    assert result["point_count"] == 1
    assert result["valid"] is True
    assert "proper_orthonormal_rotations" in result["checks"]


def test_export_validator_rejects_improper_rotation(tmp_path) -> None:
    cameras = _valid_export(tmp_path)
    cameras["cameras"][0]["camera_to_scene"][0][0] = -1
    (tmp_path / "cameras.json").write_text(json.dumps(cameras))
    try:
        validate_scene_export(tmp_path)
    except ValueError as error:
        assert "Improper camera rotation" in str(error)
    else:
        raise AssertionError("Expected an improper-rotation validation error")


def test_export_validator_rejects_disagreeing_camera_convention(tmp_path) -> None:
    cameras = _valid_export(tmp_path)
    cameras["camera_coordinate_convention"] = "different camera frame"
    (tmp_path / "cameras.json").write_text(json.dumps(cameras))

    with pytest.raises(ValueError, match="coordinate conventions disagree"):
        validate_scene_export(tmp_path)


def test_render_boundary_publishes_observed_and_arbitrary_rgb_alpha_depth(
    tmp_path, monkeypatch
) -> None:
    cameras = _valid_export(tmp_path)
    request = {
        "schema": "nht_render_request_v1",
        "cameras": [
            {
                "camera_id": "novel-view",
                "width": cameras["cameras"][0]["width"],
                "height": cameras["cameras"][0]["height"],
                "intrinsics": cameras["cameras"][0]["intrinsics"],
                "camera_to_scene": cameras["cameras"][0]["camera_to_scene"],
            }
        ],
    }
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))

    def fake_render(_checkpoint, _runtime, camera):
        shape = (camera["height"], camera["width"])
        return (
            np.full((*shape, 3), 0.5, dtype=np.float32),
            np.ones((*shape, 1), dtype=np.float32),
            np.full((*shape, 1), 2.0, dtype=np.float32),
        )

    monkeypatch.setattr("nht_pipeline.render._render_one", fake_render)
    output = tmp_path / "render"
    result = render_scene(
        tmp_path / "scene.json",
        output,
        camera_ids=["frame_000000"],
        request_path=request_path,
    )

    assert [record["request_source"] for record in result["renders"]] == [
        "observed",
        "arbitrary",
    ]
    for camera_id in ("frame_000000", "novel-view"):
        assert (output / camera_id / "rgb.npy").is_file()
        assert (output / camera_id / "alpha.npy").is_file()
        assert (output / camera_id / "depth.npy").is_file()
    assert json.loads((output / "render.json").read_text())["schema"] == (
        "nht_render_result_v1"
    )


def test_render_boundary_rejects_nonfinite_output_without_publication(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)

    def fake_render(_checkpoint, _runtime, camera):
        shape = (camera["height"], camera["width"])
        return (
            np.full((*shape, 3), np.nan, dtype=np.float32),
            np.ones((*shape, 1), dtype=np.float32),
            np.ones((*shape, 1), dtype=np.float32),
        )

    monkeypatch.setattr("nht_pipeline.render._render_one", fake_render)
    output = tmp_path / "render"
    with pytest.raises(RuntimeError, match="invalid rgb"):
        render_scene(
            tmp_path / "scene.json", output, camera_ids=["frame_000000"]
        )
    assert not output.exists()
    assert not (tmp_path / ".render.staging").exists()
