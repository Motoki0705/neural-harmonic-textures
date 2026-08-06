"""Resolved configuration for the video-to-scene pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

DEFAULT_CONFIG: dict[str, Any] = {
    "schema": "nht_pipeline_config_v1",
    "seed": 42,
    "frames": {
        "frames_per_second": 1.0,
        "jpeg_quality": 2,
    },
    "preprocess": {
        "absolute_minimum_sharpness": 10.0,
        "p05_sharpness_fraction": 0.15,
        "maximum_clipped_fraction": 0.60,
        "minimum_temporal_difference": 1.0,
        "training_image_factor": 2,
    },
    "sfm": {
        "minimum_supported_points_per_camera": 100,
        "quality_gates": {
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
        },
        "candidates": [
            {
                "id": "sift-incremental",
                "backend": "pycolmap_sift_incremental",
                "run_condition": "always",
                "camera_model": "OPENCV",
                "sequential_overlap": 10,
                "quadratic_overlap": True,
                "max_features": 4096,
                "max_image_size": 1920,
                "num_threads": 4,
            },
            {
                "id": "aliked-lightglue",
                "backend": "hloc_aliked_lightglue",
                "run_condition": "primary_rejected",
                "camera_model": "OPENCV",
                "sequential_overlap": 10,
                "retrieval_neighbors": 10,
                "max_image_size": 1024,
                "num_threads": 4,
            },
        ],
        "maximum_candidates": 2,
    },
    "nht_training": {
        "seed": 42,
        "data_factor": 2,
        "max_steps": 30_000,
        "cap_max": 1_000_000,
        "test_every": 8,
        "lpips_net": "alex",
        "python": None,
        "trainer": None,
        "extra_args": [],
        "seed_argument": None,
    },
    "export": {
        "schema": "nht_standard_scene_v1",
    },
}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        config = copy.deepcopy(DEFAULT_CONFIG)
        validate_config(config)
        return config
    text = path.read_text()
    try:
        override = json.loads(text)
    except json.JSONDecodeError:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - optional runtime
            raise ValueError(
                "Non-JSON configuration requires the optional PyYAML package"
            ) from error
        override = yaml.safe_load(text)
    if not isinstance(override, dict):
        raise TypeError("Pipeline configuration must contain a mapping")
    config = _merge(DEFAULT_CONFIG, override)
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    if float(config["frames"]["frames_per_second"]) <= 0:
        raise ValueError("frames.frames_per_second must be positive")
    if int(config["preprocess"]["training_image_factor"]) < 1:
        raise ValueError("preprocess.training_image_factor must be positive")
    if float(config["preprocess"]["minimum_temporal_difference"]) < 0:
        raise ValueError("preprocess.minimum_temporal_difference cannot be negative")
    candidates = config["sfm"]["candidates"]
    identifiers = [candidate["id"] for candidate in candidates]
    if not candidates or len(set(identifiers)) != len(identifiers):
        raise ValueError("sfm candidates must have unique IDs and cannot be empty")
    maximum = int(config["sfm"]["maximum_candidates"])
    if maximum < 1 or maximum > len(candidates):
        raise ValueError("sfm.maximum_candidates must select one or more candidates")
    for candidate in candidates:
        if candidate.get("run_condition", "always") not in {
            "always",
            "primary_rejected",
        }:
            raise ValueError(f"Unsupported candidate run_condition: {candidate['id']}")
    training = config["nht_training"]
    for name in ("data_factor", "max_steps", "cap_max", "test_every"):
        if int(training[name]) < 1:
            raise ValueError(f"nht_training.{name} must be positive")
    if int(training["data_factor"]) != int(
        config["preprocess"]["training_image_factor"]
    ):
        raise ValueError(
            "nht_training.data_factor must equal preprocess.training_image_factor"
        )
    if config["export"]["schema"] != "nht_standard_scene_v1":
        raise ValueError("export.schema must be nht_standard_scene_v1")


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    """Write JSON syntax, which is also valid YAML 1.2, to the fixed YAML path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)
