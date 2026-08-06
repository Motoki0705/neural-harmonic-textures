from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nht_pipeline.export import validate_scene_export
from nht_pipeline.render import render_scene
from nht_pipeline.schema import schema_validator, validate_schema_payload


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
        "transform_semantics": (
            "camera_to_scene maps homogeneous camera coordinates to scene coordinates"
        ),
        "cameras": [
            {
                "camera_id": "frame_000000",
                "source_frame_index": 0,
                "time_seconds": 0.0,
                "split": "train",
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
                "source_image_processing": {
                    "source_resolution": [16, 12],
                    "crop_xywh": [0, 0, 16, 12],
                    "undistorted": True,
                    "data_factor": 1,
                },
                "diagnostics": {
                    "sfm_camera_id": 0,
                    "sfm_camera_to_world": np.eye(4).tolist(),
                },
                "group": "default",
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
        "image_root": "images",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": [1, 6],
            "dtype": "float32",
            "columns": ["x", "y", "z", "red", "green", "blue"],
            "color_range": [0.0, 1.0],
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
            "model": "model",
            "checkpoint": "model/ckpts/model.pt",
            "runtime_config": "model/runtime-config.json",
            "outputs": {
                "rgb": "float32 HxWx3",
                "alpha": "float32 HxWx1",
                "depth": "float32 HxWx1",
            },
        },
        "sfm_summary": {},
        "nht_training_summary": {},
        "capabilities": ["nht_rendering_model"],
    }
    (tmp_path / "model/runtime-config.json").write_text(
        json.dumps(
            {
                "schema": "nht_runtime_config_v1",
                "camera_model": "pinhole",
                "pose_opt": False,
                "post_processing": None,
                "near_plane": 0.125,
                "far_plane": 456.0,
            }
        )
    )
    (tmp_path / "scene.json").write_text(json.dumps(scene))
    return cameras


def test_export_validator_accepts_semantically_consistent_scene(tmp_path) -> None:
    _valid_export(tmp_path)
    result = validate_scene_export(tmp_path / "scene.json")
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
        validate_scene_export(tmp_path / "scene.json")
    except ValueError as error:
        assert "Improper camera rotation" in str(error)
    else:
        raise AssertionError("Expected an improper-rotation validation error")


def test_export_validator_rejects_disagreeing_camera_convention(tmp_path) -> None:
    cameras = _valid_export(tmp_path)
    cameras["camera_coordinate_convention"] = "different camera frame"
    (tmp_path / "cameras.json").write_text(json.dumps(cameras))

    with pytest.raises(ValueError, match="coordinate conventions disagree"):
        validate_scene_export(tmp_path / "scene.json")


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

    def fake_render(_checkpoint, runtime, camera):
        assert runtime["near_plane"] == 0.125
        assert runtime["far_plane"] == 456.0
        shape = (camera["height"], camera["width"])
        return (
            np.full((*shape, 3), 0.5, dtype=np.float32),
            np.ones((*shape, 1), dtype=np.float32),
            np.full((*shape, 1), 2.0, dtype=np.float32),
        )

    monkeypatch.setattr("nht_pipeline.render._render_one", fake_render)
    output = tmp_path.parent / f"{tmp_path.name}-render"
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
    output = tmp_path.parent / f"{tmp_path.name}-render"
    with pytest.raises(RuntimeError, match="invalid rgb"):
        render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])
    assert not output.exists()
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_validator_rejects_same_directory_fake_scene(tmp_path) -> None:
    _valid_export(tmp_path)
    fake = tmp_path / "fake-scene.json"
    fake.write_text((tmp_path / "scene.json").read_text())

    with pytest.raises(ValueError, match="ordinary scene.json"):
        validate_scene_export(fake)


@pytest.mark.parametrize(
    "reference",
    ["cameras", "points", "image", "model", "checkpoint", "runtime"],
)
def test_validator_rejects_parent_traversal_for_every_export_reference(
    tmp_path, reference
) -> None:
    cameras = _valid_export(tmp_path)
    scene = json.loads((tmp_path / "scene.json").read_text())
    if reference == "cameras":
        scene["cameras"] = "../cameras.json"
    elif reference == "points":
        scene["point_cloud"]["path"] = "../points_scene.npy"
    elif reference == "image":
        cameras["cameras"][0]["image"] = "../frame.jpg"
        (tmp_path / "cameras.json").write_text(json.dumps(cameras))
    elif reference == "model":
        scene["model_root"] = "../model"
    elif reference == "checkpoint":
        scene["renderer"]["checkpoint"] = "../model.pt"
    else:
        scene["renderer"]["runtime_config"] = "../runtime-config.json"
    (tmp_path / "scene.json").write_text(json.dumps(scene))

    with pytest.raises(
        ValueError, match="canonical.*schema|export-relative|outside image_root"
    ):
        validate_scene_export(tmp_path / "scene.json")


def test_validator_rejects_symlink_escape(tmp_path) -> None:
    _valid_export(tmp_path)
    outside = tmp_path.parent / f"{tmp_path.name}-outside-cameras.json"
    outside.write_text((tmp_path / "cameras.json").read_text())
    (tmp_path / "cameras.json").unlink()
    (tmp_path / "cameras.json").symlink_to(outside)

    with pytest.raises(ValueError, match="escapes the export root"):
        validate_scene_export(tmp_path / "scene.json")


def _arbitrary_request(cameras, camera):
    observed = cameras["cameras"][0]
    return {
        "schema": "nht_render_request_v1",
        "cameras": [
            {
                "camera_id": "invalid-view",
                "width": observed["width"],
                "height": observed["height"],
                "intrinsics": json.loads(json.dumps(observed["intrinsics"])),
                "camera_to_scene": json.loads(json.dumps(observed["camera_to_scene"])),
                **camera,
            }
        ],
    }


@pytest.mark.parametrize(
    "case",
    [
        "nonhomogeneous_pose",
        "reflection",
        "singular_rotation",
        "nonhomogeneous_intrinsics",
        "negative_focal",
        "nonfinite_principal_point",
    ],
)
def test_arbitrary_camera_rejects_non_rigid_or_non_pinhole_matrices(
    tmp_path, case
) -> None:
    cameras = _valid_export(tmp_path)
    request = _arbitrary_request(cameras, {})
    camera = request["cameras"][0]
    if case == "nonhomogeneous_pose":
        camera["camera_to_scene"][3] = [0, 0, 0, 2]
    elif case == "reflection":
        camera["camera_to_scene"][0][0] = -1
    elif case == "singular_rotation":
        camera["camera_to_scene"][0][:3] = [0, 0, 0]
    elif case == "nonhomogeneous_intrinsics":
        camera["intrinsics"]["matrix"][2] = [0, 0, 2]
    elif case == "negative_focal":
        camera["intrinsics"]["matrix"][0][0] = -1
    else:
        camera["intrinsics"]["matrix"][0][2] = float("nan")
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))

    with pytest.raises(ValueError):
        render_scene(
            tmp_path / "scene.json",
            tmp_path.parent / f"{tmp_path.name}-render",
            request_path=request_path,
        )


@pytest.mark.parametrize(
    "case",
    ["nonhomogeneous_pose", "nonhomogeneous_intrinsics", "negative_focal"],
)
def test_render_request_schema_matches_runtime_matrix_envelope(tmp_path, case) -> None:
    cameras = _valid_export(tmp_path)
    request = _arbitrary_request(cameras, {})
    camera = request["cameras"][0]
    if case == "nonhomogeneous_pose":
        camera["camera_to_scene"][3] = [0, 0, 0, 2]
    elif case == "nonhomogeneous_intrinsics":
        camera["intrinsics"]["matrix"][2] = [0, 0, 2]
    else:
        camera["intrinsics"]["matrix"][0][0] = -1
    assert not schema_validator("render-request").is_valid(request)


@pytest.mark.parametrize("target", ["filesystem_root", "export_root", "workspace"])
def test_renderer_rejects_destructive_output_targets(tmp_path, target) -> None:
    _valid_export(tmp_path)
    output = {
        "filesystem_root": Path("/"),
        "export_root": tmp_path,
        "workspace": tmp_path.parent,
    }[target]

    with pytest.raises(ValueError, match="Render output"):
        render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])


def test_renderer_rejects_ordinary_file_output(tmp_path) -> None:
    _valid_export(tmp_path)
    output = tmp_path / "render-file"
    output.write_text("do not delete")

    with pytest.raises(ValueError, match="ordinary file"):
        render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])

    assert output.read_text() == "do not delete"


def _fake_successful_render(_checkpoint, _runtime, camera):
    shape = (camera["height"], camera["width"])
    return (
        np.full((*shape, 3), 0.5, dtype=np.float32),
        np.ones((*shape, 1), dtype=np.float32),
        np.ones((*shape, 1), dtype=np.float32),
    )


def _render_result_marker(scene_id: str) -> dict:
    return {
        "schema": "nht_render_result_v1",
        "scene_schema": "nht_standard_scene_v1",
        "scene_id": scene_id,
        "coordinate_space": "canonical NHT scene space",
        "export_validation": {},
        "renders": [
            {
                "camera_id": "old-view",
                "request_source": "observed",
                "width": 1,
                "height": 1,
                "rgb": "old-view/rgb.npy",
                "rgb_preview": "old-view/rgb.png",
                "alpha": "old-view/alpha.npy",
                "alpha_preview": "old-view/alpha.png",
                "depth": "old-view/depth.npy",
            }
        ],
    }


def test_renderer_preserves_unowned_nonempty_output_before_render(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-unrelated"
    output.mkdir()
    sentinel = output / "important.txt"
    sentinel.write_text("preserve me")

    def unexpected_render(*_args):
        raise AssertionError("renderer must not run for an unowned output")

    monkeypatch.setattr("nht_pipeline.render._render_one", unexpected_render)

    with pytest.raises(ValueError, match="ownership marker"):
        render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])

    assert sentinel.read_text() == "preserve me"


def test_renderer_preserves_output_owned_by_another_scene(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-foreign-render"
    output.mkdir()
    sentinel = output / "important.txt"
    sentinel.write_text("preserve me")
    (output / "render.json").write_text(json.dumps(_render_result_marker("B99")))
    monkeypatch.setattr("nht_pipeline.render._render_one", _fake_successful_render)

    with pytest.raises(ValueError, match="another scene"):
        render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])

    assert sentinel.read_text() == "preserve me"


@pytest.mark.parametrize("existing", ["empty", "owned"])
def test_renderer_replaces_only_empty_or_owned_output(
    tmp_path, monkeypatch, existing
) -> None:
    _valid_export(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-{existing}-render"
    output.mkdir()
    if existing == "owned":
        (output / "obsolete.txt").write_text("replace me")
        (output / "render.json").write_text(
            json.dumps(_render_result_marker("B00"))
        )
    monkeypatch.setattr("nht_pipeline.render._render_one", _fake_successful_render)

    render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])

    assert not (output / "obsolete.txt").exists()
    assert json.loads((output / "render.json").read_text())["scene_id"] == "B00"


def test_renderer_does_not_reclaim_fixed_name_staging_directory(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)
    output = tmp_path.parent / f"{tmp_path.name}-render"
    old_staging = output.parent / f".{output.name}.staging"
    old_staging.mkdir()
    sentinel = old_staging / "important.txt"
    sentinel.write_text("not owned by this process")
    monkeypatch.setattr("nht_pipeline.render._render_one", _fake_successful_render)

    render_scene(tmp_path / "scene.json", output, camera_ids=["frame_000000"])

    assert sentinel.read_text() == "not owned by this process"
    assert not list(output.parent.glob(f".{output.name}.*.staging"))


def test_schema_contract_accepts_all_generated_standard_payloads(
    tmp_path, monkeypatch
) -> None:
    cameras = _valid_export(tmp_path)
    scene = json.loads((tmp_path / "scene.json").read_text())
    request = _arbitrary_request(cameras, {})
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request))
    output = tmp_path.parent / f"{tmp_path.name}-schema-render"
    monkeypatch.setattr("nht_pipeline.render._render_one", _fake_successful_render)

    validate_scene_export(tmp_path / "scene.json")
    render_scene(tmp_path / "scene.json", output, request_path=request_path)
    result = json.loads((output / "render.json").read_text())
    payloads = {
        "scene": scene,
        "cameras": cameras,
        "render-request": request,
        "render-result": result,
    }

    for name, payload in payloads.items():
        assert schema_validator(name).is_valid(payload)
        validate_schema_payload(name, payload, context=f"generated {name}")


@pytest.mark.parametrize(
    "boundary",
    ["scene", "cameras", "render-request", "render-result"],
    ids=[
        "scene-missing-required",
        "cameras-unknown-field",
        "request-wrong-discriminator",
        "result-unknown-field",
    ],
)
def test_schema_contract_runtime_rejects_the_same_structural_payloads(
    tmp_path, monkeypatch, boundary
) -> None:
    cameras = _valid_export(tmp_path)
    scene_path = tmp_path / "scene.json"
    if boundary == "scene":
        payload = json.loads(scene_path.read_text())
        payload.pop("sfm_summary")
        scene_path.write_text(json.dumps(payload))
        runtime_call = lambda: validate_scene_export(scene_path)
    elif boundary == "cameras":
        payload = cameras
        payload["unknown"] = True
        (tmp_path / "cameras.json").write_text(json.dumps(payload))
        runtime_call = lambda: validate_scene_export(scene_path)
    elif boundary == "render-request":
        payload = _arbitrary_request(cameras, {})
        payload["schema"] = "nht_render_request_v0"
        request_path = tmp_path / "invalid-request.json"
        request_path.write_text(json.dumps(payload))
        runtime_call = lambda: render_scene(
            scene_path,
            tmp_path.parent / f"{tmp_path.name}-invalid-request-render",
            request_path=request_path,
        )
    else:
        payload = _render_result_marker("B00")
        payload["unknown"] = True
        output = tmp_path.parent / f"{tmp_path.name}-invalid-result-render"
        output.mkdir()
        (output / "render.json").write_text(json.dumps(payload))
        runtime_call = lambda: render_scene(
            scene_path, output, camera_ids=["frame_000000"]
        )

    monkeypatch.setattr(
        "nht_pipeline.render._render_one",
        lambda *_args: pytest.fail("schema-invalid payload reached the renderer"),
    )
    assert not schema_validator(boundary).is_valid(payload)
    with pytest.raises(ValueError, match="canonical.*schema"):
        runtime_call()
