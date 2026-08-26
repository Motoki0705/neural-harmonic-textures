from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from nht_pipeline.composed_render import (
    _joint_eval3d_channel_layout,
    _pad_joint_eval3d_channels,
    _split_joint_eval3d_output,
)
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


class _FakeComposedRenderer:
    cuda_peak_bytes = 4096

    def render_background(self, request):
        shape = (int(request["height"]), int(request["width"]))
        return (
            np.full((*shape, 3), 0.1, dtype=np.float32),
            np.ones((*shape, 1), dtype=np.float32),
            np.full((*shape, 1), 10.0, dtype=np.float32),
        )

    def render_frame(self, request, frame_index):
        rgb, alpha, depth = self.render_background(request)
        labels = np.zeros(rgb.shape[:2], dtype=np.int32)
        if frame_index == 0:
            labels[2, 3] = 1
            rgb[2, 3] = (0.72, 0.92, 0.08)
            alpha[2, 3, 0] = 0.97
            depth[2, 3, 0] = 2.0
        return rgb, alpha, depth, labels


class _FakeChannelTensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.shape = self.values.shape
        self.dtype = self.values.dtype
        self.device = "cpu"


class _FakeChannelTorch:
    @staticmethod
    def zeros(*shape, dtype, device):
        assert device == "cpu"
        return _FakeChannelTensor(np.zeros(shape, dtype=dtype))

    @staticmethod
    def cat(values, dim):
        return _FakeChannelTensor(
            np.concatenate([value.values for value in values], axis=dim)
        )


def test_joint_eval3d_channel_layout_pads_eight_objects_and_preserves_one() -> None:
    single = _joint_eval3d_channel_layout(1)
    assert single.logical_input_channels == 4
    assert single.physical_input_channels == 4
    assert single.physical_total_channels == 5
    single_channels = _FakeChannelTensor(np.ones((2, 4), dtype=np.float32))
    assert (
        _pad_joint_eval3d_channels(
            _FakeChannelTorch,
            single_channels,
            layout=single,
        )
        is single_channels
    )

    multi = _joint_eval3d_channel_layout(8)
    assert multi.logical_input_channels == 11
    assert multi.physical_input_channels == 15
    assert multi.physical_total_channels == 16
    logical = _FakeChannelTensor(
        np.arange(22, dtype=np.float32).reshape(2, 11)
    )
    padded = _pad_joint_eval3d_channels(
        _FakeChannelTorch,
        logical,
        layout=multi,
    )
    assert padded.shape == (2, 15)
    np.testing.assert_array_equal(padded.values[:, :11], logical.values)
    np.testing.assert_array_equal(padded.values[:, 11:], 0.0)

    physical_output = np.arange(16, dtype=np.float32).reshape(1, 16)
    direct_rgb, semantics, depth = _split_joint_eval3d_output(
        physical_output,
        object_count=8,
        layout=multi,
    )
    np.testing.assert_array_equal(direct_rgb, [[0.0, 1.0, 2.0]])
    np.testing.assert_array_equal(
        semantics,
        [np.arange(3, 11, dtype=np.float32)],
    )
    np.testing.assert_array_equal(depth, [[15.0]])


def test_joint_eval3d_channel_layout_fails_above_compiled_limit() -> None:
    maximum = _joint_eval3d_channel_layout(509)
    assert maximum.physical_total_channels == 513
    with pytest.raises(ValueError, match="compiled CUDA channel limit"):
        _joint_eval3d_channel_layout(510)


def _composition_request(root: Path, *, asset_dtype=np.float32) -> Path:
    root.mkdir()
    np.savez(
        root / "asset.npz",
        means_m=np.asarray([[0.0, 0.0, 0.0335]], dtype=asset_dtype),
        quats_wxyz=np.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=asset_dtype),
        log_scales_m=np.log(
            np.asarray([[0.0048, 0.0048, 0.0018]], dtype=asset_dtype)
        ),
        opacity_logits=np.asarray([2.75], dtype=asset_dtype),
        colors_linear_rgb=np.asarray([[0.72, 0.92, 0.08]], dtype=asset_dtype),
    )
    transforms = np.repeat(np.eye(4, dtype=np.float64)[None, None], 2, axis=0)
    np.savez(
        root / "timeline.npz",
        transforms_nht_from_asset=transforms,
        present=np.asarray([[True], [False]], dtype=np.bool_),
        instance_ids=np.asarray([1], dtype=np.int32),
    )
    request = {
        "schema": "nht_composed_render_request_v1",
        "asset": {
            "asset_id": "regulation-tennis-ball",
            "coordinate_space": "right_handed_asset_local_metres",
            "appearance_model": "direct_linear_rgb",
            "gaussian_count": 1,
            "tensors": "asset.npz",
        },
        "timeline": {
            "coordinate_space": "canonical NHT scene space",
            "frame_count": 2,
            "object_count": 1,
            "object_ids": ["ball-001"],
            "instance_ids": [1],
            "tensors": "timeline.npz",
            "chunks": [{"chunk_index": 0, "frame_indices": [0, 1]}],
        },
        "visibility_threshold": 0.0001,
    }
    path = root / "composition.json"
    path.write_text(json.dumps(request))
    return path


def test_composed_render_boundary_publishes_joint_sparse_frames(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)
    composition = _composition_request(
        tmp_path.parent / f"{tmp_path.name}-composition"
    )
    output = tmp_path.parent / f"{tmp_path.name}-composed-render"

    def fake_build(_checkpoint, _runtime, loaded, _requests):
        assert loaded.frame_count == 2
        assert loaded.object_count == 1
        assert loaded.asset.gaussian_count == 1
        return _FakeComposedRenderer()

    monkeypatch.setattr("nht_pipeline.composed_render._build_renderer", fake_build)
    result = render_scene(
        tmp_path / "scene.json",
        output,
        camera_ids=["frame_000000"],
        composition_path=composition,
    )

    assert schema_validator("composed-render-result").is_valid(result)
    validate_schema_payload(
        "composed-render-result", result, context="generated composed result"
    )
    assert result["composition"]["frame_count"] == 2
    assert result["chunks"][0]["sample_count"] == 2
    with np.load(output / result["chunks"][0]["arrays"], allow_pickle=False) as arrays:
        np.testing.assert_array_equal(arrays["frame_indices"], [0, 1])
        np.testing.assert_array_equal(arrays["offsets"], [0, 1, 1])
        np.testing.assert_array_equal(arrays["pixel_indices"], [2 * 16 + 3])
        np.testing.assert_array_equal(arrays["instance_ids"], [1])
        np.testing.assert_allclose(arrays["depth"], [2.0])
    assert (output / "background/frame_000000/depth.npy").is_file()


def test_composed_render_boundary_rejects_non_float32_asset(
    tmp_path, monkeypatch
) -> None:
    _valid_export(tmp_path)
    composition = _composition_request(
        tmp_path.parent / f"{tmp_path.name}-float64-composition",
        asset_dtype=np.float64,
    )

    monkeypatch.setattr(
        "nht_pipeline.composed_render._build_renderer",
        lambda *_args: pytest.fail("invalid asset reached the renderer"),
    )
    with pytest.raises(ValueError, match="must be float32"):
        render_scene(
            tmp_path / "scene.json",
            tmp_path.parent / f"{tmp_path.name}-float64-composed-render",
            camera_ids=["frame_000000"],
            composition_path=composition,
        )
