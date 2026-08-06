from __future__ import annotations

import json

import numpy as np
from PIL import Image

from nht_pipeline.export import validate_scene_export


def _valid_export(tmp_path):
    (tmp_path / "images").mkdir()
    Image.new("RGB", (16, 12)).save(tmp_path / "images/frame_000000.jpg")
    (tmp_path / "model/ckpts").mkdir(parents=True)
    (tmp_path / "model/ckpts/model.pt").write_bytes(b"checkpoint")
    points = np.asarray([[0, 0, 0, 1, 0.5, 0]], dtype=np.float32)
    np.save(tmp_path / "points_scene.npy", points)
    cameras = {
        "schema": "nht_standard_cameras_v1",
        "cameras": [
            {
                "camera_id": "frame_000000",
                "source_frame_index": 0,
                "time_seconds": 0.0,
                "width": 16,
                "height": 12,
                "intrinsics": {
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
        "model_root": "model",
        "capabilities": ["nht_rendering_model"],
    }
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
