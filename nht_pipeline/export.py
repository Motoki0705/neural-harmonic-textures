"""Standard scene export and semantic validation."""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


def _frame_index(name: str, fallback: int) -> int:
    suffix = Path(name).stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def _camera_to_world(image: Any) -> np.ndarray:
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :] = np.asarray(image.cam_from_world().matrix())
    return np.linalg.inv(world_to_camera)


def create_scene_export(
    workspace: Path,
    scene_id: str,
    schema: str,
) -> dict[str, Any]:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - runtime dependency
        raise RuntimeError("pycolmap is required for scene export") from error

    export_root = workspace / "export"
    image_root = export_root / "images"
    model_root = export_root / "model"
    image_root.mkdir(parents=True, exist_ok=True)
    model_root.mkdir(parents=True, exist_ok=True)
    reconstruction_path = workspace / "sfm/reconstruction.json"
    reconstruction_summary = json.loads(reconstruction_path.read_text())
    training_summary = json.loads((workspace / "3dgs/training.json").read_text())
    frames_summary = json.loads((workspace / "frames/frames.json").read_text())
    frame_metadata = {record["filename"]: record for record in frames_summary["frames"]}
    reconstruction = pycolmap.Reconstruction(workspace / "sfm/model")
    source_images = workspace / "frames/images"

    cameras = []
    for fallback, image in enumerate(
        sorted(reconstruction.images.values(), key=lambda item: item.name)
    ):
        source = source_images / image.name
        if not source.is_file():
            raise FileNotFoundError(f"Registered image is absent: {source}")
        destination = image_root / image.name
        destination.hardlink_to(source)
        camera = reconstruction.cameras[image.camera_id]
        metadata = frame_metadata.get(image.name)
        if metadata is None or not metadata["accepted"]:
            raise ValueError(
                f"Registered image has no accepted frame record: {image.name}"
            )
        cameras.append(
            {
                "camera_id": Path(image.name).stem,
                "source_frame_index": int(metadata["source_frame_index"]),
                "time_seconds": float(metadata["source_time_seconds"]),
                "image": f"images/{image.name}",
                "width": int(camera.width),
                "height": int(camera.height),
                "intrinsics": {
                    "model": camera.model_name,
                    "params": [float(value) for value in camera.params],
                    "matrix": np.asarray(camera.calibration_matrix()).tolist(),
                },
                "camera_to_scene": _camera_to_world(image).tolist(),
                "group": "default",
            }
        )

    points = np.asarray(
        [
            [*point.xyz, *(np.asarray(point.color, dtype=np.float64) / 255.0)]
            for point in reconstruction.points3D.values()
        ],
        dtype=np.float32,
    )
    np.save(export_root / "points_scene.npy", points)
    shutil.copytree(workspace / "3dgs/model", model_root, dirs_exist_ok=True)
    cameras_payload = {
        "schema": "nht_standard_cameras_v1",
        "camera_coordinate_convention": "x-right, y-down, z-forward",
        "transform_semantics": "camera_to_scene maps homogeneous camera coordinates to scene coordinates",
        "cameras": cameras,
    }
    (export_root / "cameras.json").write_text(
        json.dumps(cameras_payload, indent=2) + "\n"
    )
    scene = {
        "schema": schema,
        "scene_id": scene_id,
        "camera_coordinate_convention": "COLMAP: x-right, y-down, z-forward",
        "scene_coordinate_convention": "selected COLMAP world coordinates; right-handed",
        "pixel_coordinate_convention": "origin at top-left; x-right, y-down; pixel centers",
        "image_resolution_semantics": "width and height are full-resolution source image pixels",
        "camera_count": len(cameras),
        "cameras": "cameras.json",
        "point_cloud": {
            "path": "points_scene.npy",
            "shape": list(points.shape),
            "dtype": "float32",
            "columns": ["x", "y", "z", "red", "green", "blue"],
            "color_range": [0.0, 1.0],
        },
        "image_root": "images",
        "model_root": "model",
        "scene_from_sfm": np.eye(4).tolist(),
        "normalization": {"applied": False, "transform": np.eye(4).tolist()},
        "sfm_summary": reconstruction_summary,
        "nht_training_summary": training_summary,
        "capabilities": [
            "calibrated_cameras",
            "sparse_point_cloud",
            "nht_rendering_model",
        ],
    }
    (export_root / "scene.json").write_text(json.dumps(scene, indent=2) + "\n")
    validate_scene_export(export_root)
    return scene


def validate_scene_export(export_root: Path) -> dict[str, Any]:
    scene = json.loads((export_root / "scene.json").read_text())
    cameras_payload = json.loads((export_root / scene["cameras"]).read_text())
    cameras = cameras_payload["cameras"]
    if scene["camera_count"] != len(cameras) or not cameras:
        raise ValueError("camera_count does not match a non-empty camera list")
    points = np.load(export_root / scene["point_cloud"]["path"], allow_pickle=False)
    if (
        points.ndim != 2
        or points.shape[1] != 6
        or points.dtype != np.float32
        or not np.isfinite(points).all()
    ):
        raise ValueError("points_scene.npy must be a finite Nx6 float array")
    if list(points.shape) != scene["point_cloud"]["shape"]:
        raise ValueError("Point cloud shape disagrees with scene.json")
    if scene["point_cloud"].get("dtype") != "float32":
        raise ValueError("Point cloud dtype disagrees with scene.json")
    if len(points) and (points[:, 3:].min() < 0 or points[:, 3:].max() > 1):
        raise ValueError("Point colors must be normalized to [0, 1]")
    for field in (
        "camera_coordinate_convention",
        "scene_coordinate_convention",
        "pixel_coordinate_convention",
        "image_resolution_semantics",
    ):
        if not isinstance(scene.get(field), str) or not scene[field]:
            raise ValueError(f"Missing scene convention: {field}")
    for field in ("scene_from_sfm",):
        transform = np.asarray(scene[field], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Invalid scene transform: {field}")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"Non-homogeneous scene transform: {field}")
    identifiers: set[str] = set()
    image_paths: set[str] = set()
    for camera in cameras:
        identifier = camera["camera_id"]
        if identifier in identifiers:
            raise ValueError(f"Duplicate stable camera ID: {identifier}")
        identifiers.add(identifier)
        if (
            not isinstance(camera.get("source_frame_index"), int)
            or camera["source_frame_index"] < 0
        ):
            raise ValueError(f"Invalid source frame index for {identifier}")
        if not math.isfinite(float(camera.get("time_seconds", math.nan))):
            raise ValueError(f"Invalid camera time for {identifier}")
        if camera["width"] <= 0 or camera["height"] <= 0:
            raise ValueError(f"Invalid resolution for {identifier}")
        intrinsics = np.asarray(camera["intrinsics"]["matrix"], dtype=np.float64)
        if intrinsics.shape != (3, 3) or not np.isfinite(intrinsics).all():
            raise ValueError(f"Invalid intrinsics for {identifier}")
        if intrinsics[0, 0] <= 0 or intrinsics[1, 1] <= 0:
            raise ValueError(f"Non-positive focal length for {identifier}")
        parameters = np.asarray(
            camera["intrinsics"].get("params", []), dtype=np.float64
        )
        if not len(parameters) or not np.isfinite(parameters).all():
            raise ValueError(f"Invalid camera parameters for {identifier}")
        transform = np.asarray(camera["camera_to_scene"], dtype=np.float64)
        if transform.shape != (4, 4) or not np.isfinite(transform).all():
            raise ValueError(f"Invalid camera transform for {identifier}")
        if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
            raise ValueError(f"Non-homogeneous camera transform for {identifier}")
        determinant = float(np.linalg.det(transform[:3, :3]))
        if not math.isclose(determinant, 1.0, abs_tol=1e-5):
            raise ValueError(
                f"Improper camera rotation for {identifier}: {determinant}"
            )
        if not np.allclose(
            transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-5
        ):
            raise ValueError(f"Non-orthonormal camera rotation for {identifier}")
        if camera["image"] in image_paths:
            raise ValueError(f"Duplicate camera image path: {camera['image']}")
        image_paths.add(camera["image"])
        image_path = export_root / camera["image"]
        if not image_path.is_file():
            raise FileNotFoundError(f"Export image is absent: {camera['image']}")
        with Image.open(image_path) as exported_image:
            if (exported_image.width, exported_image.height) != (
                camera["width"],
                camera["height"],
            ):
                raise ValueError(f"Image resolution mismatch for {identifier}")
    if not (export_root / scene["model_root"]).is_dir():
        raise FileNotFoundError("Export model root is absent")
    if "nht_rendering_model" in scene.get("capabilities", []) and not list(
        (export_root / scene["model_root"] / "ckpts").glob("*.pt")
    ):
        raise FileNotFoundError("Export NHT model has no checkpoint")
    return {
        "schema": scene["schema"],
        "camera_count": len(cameras),
        "point_count": len(points),
        "checks": [
            "schema_and_conventions",
            "finite_float32_points",
            "normalized_point_colors",
            "unique_camera_and_image_ids",
            "positive_intrinsics",
            "proper_orthonormal_rotations",
            "camera_image_resolution",
            "nht_checkpoint",
        ],
        "valid": True,
    }
