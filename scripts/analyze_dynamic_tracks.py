#!/usr/bin/env python3
"""Measure dynamic-box contamination and static support in SfM candidates."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from nht_pipeline.run_state import RunState

DYNAMIC_COCO_CLASSES = (0, 1, 2, 3, 5, 7)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--minimum-static-points", type=int, default=50)
    return parser.parse_args()


def _inside_any_box(xy: np.ndarray, boxes: np.ndarray) -> bool:
    if boxes.size == 0:
        return False
    x, y = float(xy[0]), float(xy[1])
    return bool(
        np.any(
            (boxes[:, 0] <= x)
            & (x <= boxes[:, 2])
            & (boxes[:, 1] <= y)
            & (y <= boxes[:, 3])
        )
    )


def _detect_dynamic_boxes(
    image_root: Path, image_names: list[str], checkpoint: Path, confidence: float
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    from ultralytics import YOLO

    started = time.monotonic()
    model = YOLO(str(checkpoint))
    paths = [str(image_root / name) for name in image_names]
    predictions = model.predict(
        paths,
        classes=list(DYNAMIC_COCO_CLASSES),
        conf=confidence,
        device=0,
        verbose=False,
        stream=False,
    )
    boxes: dict[str, np.ndarray] = {}
    detections = Counter()
    for name, prediction in zip(image_names, predictions, strict=True):
        values = prediction.boxes.xyxy.detach().cpu().numpy().astype(np.float64)
        classes = prediction.boxes.cls.detach().cpu().numpy().astype(np.int64)
        boxes[name] = values
        detections.update(int(value) for value in classes)
    return boxes, {
        "checkpoint": str(checkpoint.resolve()),
        "confidence": confidence,
        "dynamic_coco_classes": list(DYNAMIC_COCO_CLASSES),
        "class_detection_counts": dict(sorted(detections.items())),
        "images_with_detections": sum(len(value) > 0 for value in boxes.values()),
        "elapsed_seconds": time.monotonic() - started,
    }


def _candidate_metrics(
    model_path: Path,
    boxes_by_image: dict[str, np.ndarray],
    minimum_static_points: int,
) -> dict[str, Any]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(model_path)
    dynamic_observations = 0
    static_observations = 0
    static_supported_images = 0
    dynamic_point_ids: set[int] = set()
    static_observations_by_point: Counter[int] = Counter()
    registered_with_detections = 0
    for image in reconstruction.images.values():
        boxes = boxes_by_image.get(image.name, np.empty((0, 4), dtype=np.float64))
        registered_with_detections += bool(len(boxes))
        image_static = 0
        for point in image.points2D:
            if not point.has_point3D():
                continue
            point_id = int(point.point3D_id)
            if _inside_any_box(np.asarray(point.xy), boxes):
                dynamic_observations += 1
                dynamic_point_ids.add(point_id)
            else:
                static_observations += 1
                image_static += 1
                static_observations_by_point[point_id] += 1
        static_supported_images += image_static >= minimum_static_points
    total_observations = dynamic_observations + static_observations
    static_track_ids = {
        identifier
        for identifier, count in static_observations_by_point.items()
        if count >= 2
    }
    point_count = reconstruction.num_points3D()
    image_count = reconstruction.num_reg_images()
    return {
        "registered_images": image_count,
        "registered_images_with_dynamic_detections": registered_with_detections,
        "static_supported_images": static_supported_images,
        "static_supported_registration_ratio": (
            static_supported_images / image_count if image_count else 0.0
        ),
        "minimum_static_points_per_image": minimum_static_points,
        "registered_point_observations": total_observations,
        "dynamic_box_observations": dynamic_observations,
        "dynamic_observation_fraction": (
            dynamic_observations / total_observations if total_observations else 0.0
        ),
        "points_with_any_dynamic_box_observation": len(dynamic_point_ids),
        "dynamic_track_contamination_fraction": (
            len(dynamic_point_ids) / point_count if point_count else 0.0
        ),
        "tracks_with_at_least_two_static_observations": len(static_track_ids),
        "sparse_points": point_count,
    }


def main() -> None:
    import pycolmap

    args = _arguments()
    workspace = args.workspace.resolve()
    candidate_manifest = json.loads(
        (workspace / "sfm/candidates/candidates.json").read_text()
    )
    candidates = [
        record
        for record in candidate_manifest["candidates"]
        if record["status"] == "completed"
    ]
    image_names = sorted(
        {
            image.name
            for record in candidates
            for image in pycolmap.Reconstruction(
                workspace / "sfm/candidates" / record["id"] / "model"
            ).images.values()
        }
    )
    boxes, detector = _detect_dynamic_boxes(
        workspace / "frames/images",
        image_names,
        args.checkpoint.resolve(),
        args.confidence,
    )
    payload = {
        "schema": "nht_dynamic_track_analysis_v1",
        "scene_id": RunState.load(workspace).payload["scene_id"],
        "detector": detector,
        "candidates": {},
    }
    for record in candidates:
        metrics = _candidate_metrics(
            workspace / "sfm/candidates" / record["id"] / "model",
            boxes,
            args.minimum_static_points,
        )
        metrics["pose_continuity"] = record["metrics"]["trajectory"]
        payload["candidates"][record["id"]] = metrics
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
