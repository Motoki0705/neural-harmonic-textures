"""Measure SfM geometry and apply semantic candidate quality gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _percentile(values: np.ndarray, percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if len(values) else None


def _ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator <= 0:
        return None
    return numerator / denominator


def rotation_angle_deg(rotation: np.ndarray) -> float:
    cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
    return math.degrees(math.acos(cosine))


def trajectory_metrics(
    centers: np.ndarray, rotations: np.ndarray, frame_indices: np.ndarray
) -> dict[str, Any]:
    if len(centers) < 2:
        return {
            "median_step": None,
            "p95_step": None,
            "maximum_step": None,
            "maximum_step_to_median": None,
            "step_outlier_count": None,
            "median_rotation_step_deg": None,
            "p95_rotation_step_deg": None,
            "maximum_rotation_step_deg": None,
            "planarity_ratio": None,
            "extent": None,
            "p05_step": None,
            "near_duplicate_step_count": None,
            "near_duplicate_fraction": None,
        }

    frame_deltas = np.diff(frame_indices).astype(np.float64)
    if np.any(frame_deltas <= 0):
        raise ValueError("Registered images are not in strictly increasing frame order")
    steps = np.linalg.norm(np.diff(centers, axis=0), axis=1) / frame_deltas
    median_step = float(np.median(steps))
    p95_step = _percentile(steps, 95)
    maximum_step = float(steps.max())
    median_absolute_deviation = float(np.median(np.abs(steps - median_step)))
    step_limit = max(
        5.0 * median_step,
        median_step + 6.0 * median_absolute_deviation,
    )

    rotation_steps = np.asarray(
        [
            rotation_angle_deg(rotations[index + 1] @ rotations[index].T)
            / frame_deltas[index]
            for index in range(len(rotations) - 1)
        ],
        dtype=np.float64,
    )
    centered = centers - centers.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    planarity_ratio = (
        float(singular_values[-1] / singular_values[0])
        if singular_values[0] > 0
        else None
    )
    return {
        "median_step": median_step,
        "p95_step": p95_step,
        "maximum_step": maximum_step,
        "maximum_step_to_median": _ratio(maximum_step, median_step),
        "step_outlier_count": int(np.count_nonzero(steps > step_limit)),
        "median_rotation_step_deg": float(np.median(rotation_steps)),
        "p95_rotation_step_deg": _percentile(rotation_steps, 95),
        "maximum_rotation_step_deg": float(rotation_steps.max()),
        "planarity_ratio": planarity_ratio,
        "extent": np.ptp(centers, axis=0).tolist(),
        "p05_step": _percentile(steps, 5),
        "near_duplicate_step_count": int(
            np.count_nonzero(steps <= max(median_step * 0.05, 1e-7))
        ),
        "near_duplicate_fraction": float(
            np.count_nonzero(steps <= max(median_step * 0.05, 1e-7)) / len(steps)
        ),
    }


def _gate(value: float | None, operator: str, threshold: float) -> dict[str, Any]:
    if value is None:
        passed = False
    elif operator == ">=":
        passed = value >= threshold
    elif operator == "<=":
        passed = value <= threshold
    else:
        raise ValueError(f"Unknown gate operator: {operator}")
    return {
        "value": value,
        "operator": operator,
        "threshold": threshold,
        "passed": bool(passed),
    }


DEFAULT_QUALITY_THRESHOLDS = {
    "minimum_registration_ratio": 0.95,
    "minimum_supported_registration_ratio": 0.95,
    "minimum_sparse_points": 50_000,
    "minimum_median_track_length": 3.0,
    "maximum_p95_reprojection_error_px": 3.0,
    "maximum_trajectory_step_ratio": 5.0,
    "maximum_trajectory_step_outliers": 0,
    "maximum_planarity_ratio": 0.15,
    "maximum_rotation_step_deg": 120.0,
    "maximum_mapping_components": 1,
    "maximum_near_duplicate_fraction": 0.05,
    "minimum_focal_to_width": 0.1,
    "maximum_focal_to_width": 5.0,
}


def quality_gates(
    metrics: dict[str, Any], thresholds: dict[str, float] | None = None
) -> dict[str, dict[str, Any]]:
    thresholds = {**DEFAULT_QUALITY_THRESHOLDS, **(thresholds or {})}
    trajectory = metrics["trajectory"]
    return {
        "registration_ratio": _gate(
            metrics["registration_ratio"],
            ">=",
            thresholds["minimum_registration_ratio"],
        ),
        "supported_registration_ratio": _gate(
            metrics["supported_registration_ratio"],
            ">=",
            thresholds["minimum_supported_registration_ratio"],
        ),
        "sparse_points": _gate(
            metrics["sparse_points"], ">=", thresholds["minimum_sparse_points"]
        ),
        "median_track_length": _gate(
            metrics["median_track_length"],
            ">=",
            thresholds["minimum_median_track_length"],
        ),
        "p95_reprojection_error_px": _gate(
            metrics["p95_reprojection_error_px"],
            "<=",
            thresholds["maximum_p95_reprojection_error_px"],
        ),
        "trajectory_step_ratio": _gate(
            trajectory["maximum_step_to_median"],
            "<=",
            thresholds["maximum_trajectory_step_ratio"],
        ),
        "trajectory_step_outliers": _gate(
            trajectory["step_outlier_count"],
            "<=",
            thresholds["maximum_trajectory_step_outliers"],
        ),
        "trajectory_planarity": _gate(
            trajectory["planarity_ratio"],
            "<=",
            thresholds["maximum_planarity_ratio"],
        ),
        "rotation_continuity_deg": _gate(
            trajectory["maximum_rotation_step_deg"],
            "<=",
            thresholds["maximum_rotation_step_deg"],
        ),
        "mapping_components": _gate(
            metrics["mapping_components"],
            "<=",
            thresholds["maximum_mapping_components"],
        ),
        "near_duplicate_cameras": _gate(
            trajectory["near_duplicate_fraction"],
            "<=",
            thresholds["maximum_near_duplicate_fraction"],
        ),
        "minimum_focal_to_width": _gate(
            metrics["minimum_focal_to_width"],
            ">=",
            thresholds["minimum_focal_to_width"],
        ),
        "maximum_focal_to_width": _gate(
            metrics["maximum_focal_to_width"],
            "<=",
            thresholds["maximum_focal_to_width"],
        ),
    }


def _frame_index(name: str, fallback: int) -> int:
    stem = Path(name).stem
    suffix = stem.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def evaluate_reconstruction(
    model_path: Path,
    image_dir: Path,
    input_image_count: int | None = None,
    minimum_supported_points_per_image: int = 100,
    thresholds: dict[str, float] | None = None,
    mapping_components: int = 1,
) -> dict[str, Any]:
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - depends on runtime
        raise RuntimeError("pycolmap is required to evaluate an SfM model") from error

    reconstruction = pycolmap.Reconstruction(model_path)
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    points = list(reconstruction.points3D.values())
    if input_image_count is None:
        input_image_count = sum(
            1
            for path in image_dir.rglob("*")
            if path.is_file() and path.suffix in {".jpg", ".jpeg", ".png"}
        )
    errors = np.asarray([point.error for point in points], dtype=np.float64)
    track_lengths = np.asarray(
        [point.track.length() for point in points], dtype=np.float64
    )
    registered_points = np.asarray(
        [image.num_points3D for image in images], dtype=np.int64
    )
    weak_images = [
        {"name": image.name, "registered_points": int(image.num_points3D)}
        for image in images
        if image.num_points3D < minimum_supported_points_per_image
    ]
    centers = np.asarray(
        [image.projection_center() for image in images], dtype=np.float64
    )
    rotations = np.asarray(
        [image.cam_from_world().rotation.matrix() for image in images],
        dtype=np.float64,
    )
    frame_indices = np.asarray(
        [_frame_index(image.name, index) for index, image in enumerate(images)],
        dtype=np.int64,
    )

    cameras = {
        str(camera_id): {
            "model": camera.model_name,
            "width": camera.width,
            "height": camera.height,
            "params": [float(value) for value in camera.params],
            "focal_to_width": float(camera.mean_focal_length() / camera.width),
        }
        for camera_id, camera in reconstruction.cameras.items()
    }
    focal_to_width = np.asarray(
        [camera["focal_to_width"] for camera in cameras.values()], dtype=np.float64
    )
    point_positions = np.asarray([point.xyz for point in points], dtype=np.float64)
    if len(point_positions):
        robust_low = np.percentile(point_positions, 1, axis=0)
        robust_high = np.percentile(point_positions, 99, axis=0)
        point_cloud_coverage: dict[str, Any] = {
            "extent": np.ptp(point_positions, axis=0).tolist(),
            "robust_p01_p99_extent": (robust_high - robust_low).tolist(),
            "centroid": point_positions.mean(axis=0).tolist(),
        }
    else:
        point_cloud_coverage = {
            "extent": None,
            "robust_p01_p99_extent": None,
            "centroid": None,
        }
    metrics: dict[str, Any] = {
        "model_path": str(model_path.resolve()),
        "input_images": input_image_count,
        "registered_images": reconstruction.num_reg_images(),
        "registration_ratio": reconstruction.num_reg_images() / input_image_count,
        "supported_registered_images": reconstruction.num_reg_images()
        - len(weak_images),
        "supported_registration_ratio": (
            reconstruction.num_reg_images() - len(weak_images)
        )
        / input_image_count,
        "weak_registered_images": weak_images,
        "minimum_registered_points_per_image": (
            int(registered_points.min()) if len(registered_points) else None
        ),
        "p05_registered_points_per_image": _percentile(registered_points, 5),
        "median_registered_points_per_image": _percentile(registered_points, 50),
        "sparse_points": reconstruction.num_points3D(),
        "point_cloud_coverage": point_cloud_coverage,
        "mean_reprojection_error_px": float(errors.mean()) if len(errors) else None,
        "median_reprojection_error_px": _percentile(errors, 50),
        "p95_reprojection_error_px": _percentile(errors, 95),
        "mean_track_length": float(track_lengths.mean())
        if len(track_lengths)
        else None,
        "median_track_length": _percentile(track_lengths, 50),
        "p95_track_length": _percentile(track_lengths, 95),
        "trajectory": trajectory_metrics(centers, rotations, frame_indices),
        "cameras": cameras,
        "mapping_components": mapping_components,
        "minimum_focal_to_width": (
            float(focal_to_width.min()) if len(focal_to_width) else None
        ),
        "maximum_focal_to_width": (
            float(focal_to_width.max()) if len(focal_to_width) else None
        ),
    }
    metrics["minimum_supported_points_per_image"] = minimum_supported_points_per_image
    metrics["gates"] = quality_gates(metrics, thresholds)
    metrics["accepted"] = all(gate["passed"] for gate in metrics["gates"].values())
    return metrics


def write_trajectory_diagnostics(model_path: Path, output_root: Path) -> None:
    """Write dependency-free camera trajectory CSV and top-down SVG diagnostics."""
    try:
        import pycolmap
    except ImportError as error:  # pragma: no cover - depends on runtime
        raise RuntimeError("pycolmap is required for trajectory diagnostics") from error

    reconstruction = pycolmap.Reconstruction(model_path)
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    centers = np.asarray(
        [image.projection_center() for image in images], dtype=np.float64
    )
    output_root.mkdir(parents=True, exist_ok=True)
    rows = ["image,frame_index,x,y,z"]
    for fallback, (image, center) in enumerate(zip(images, centers, strict=True)):
        rows.append(
            f"{image.name},{_frame_index(image.name, fallback)},"
            f"{center[0]:.9g},{center[1]:.9g},{center[2]:.9g}"
        )
    (output_root / "trajectory.csv").write_text("\n".join(rows) + "\n")

    if not len(centers):
        return
    xy = centers[:, [0, 2]]
    low = xy.min(axis=0)
    extent = np.maximum(np.ptp(xy, axis=0), 1e-9)
    normalized = (xy - low) / extent
    width, height, margin = 1000.0, 700.0, 30.0
    plotted = np.column_stack(
        [
            margin + normalized[:, 0] * (width - 2 * margin),
            height - margin - normalized[:, 1] * (height - 2 * margin),
        ]
    )
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in plotted)
    circles = "\n".join(
        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="2" fill="#2563eb" />' for x, y in plotted
    )
    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width:g} {height:g}">
<rect width="100%" height="100%" fill="white" />
<polyline points="{points}" fill="none" stroke="#111827" stroke-width="2" />
{circles}
<circle cx="{plotted[0, 0]:.2f}" cy="{plotted[0, 1]:.2f}" r="7" fill="#16a34a" />
<circle cx="{plotted[-1, 0]:.2f}" cy="{plotted[-1, 1]:.2f}" r="7" fill="#dc2626" />
</svg>
"""
    (output_root / "trajectory.svg").write_text(svg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--input-image-count", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    metrics = evaluate_reconstruction(
        args.model, args.image_dir, args.input_image_count
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
