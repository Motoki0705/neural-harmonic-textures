"""Resolved configuration for the video-to-scene pipeline."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .stages import STAGES

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
            "minimum_points_per_supported_camera": 100.0,
            "minimum_spatial_voxel_coverage_fraction": 0.005,
            "minimum_median_triangulation_angle_deg": 0.5,
            "maximum_focal_length_coefficient_of_variation": 0.15,
        },
        "short_clip_max_images": 90,
        "short_clip_quality_gates": {
            "minimum_sparse_points": 500,
            "minimum_points_per_supported_camera": 20.0,
            "minimum_spatial_voxel_coverage_fraction": 0.002,
            "minimum_median_triangulation_angle_deg": 0.2,
        },
        "candidates": [
            {
                "id": "sift-incremental",
                "backend": "pycolmap_sift_incremental",
                "run_condition": "always",
                "camera_model": "OPENCV",
                "camera_sharing": "single",
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
                "camera_sharing": "single",
                "sequential_overlap": 10,
                "retrieval_neighbors": 10,
                "max_image_size": 1024,
                "num_threads": 4,
            },
        ],
        "maximum_candidates": 2,
    },
    "nht_training": {
        "data_factor": 2,
        "max_steps": 30_000,
        "cap_max": 1_000_000,
        "test_every": 8,
        "lpips_net": "alex",
        "python": None,
        "trainer": None,
        "extra_args": [],
        "cuda_device": 0,
    },
    "export": {
        "schema": "nht_standard_scene_v1",
    },
    "operations": {
        "minimum_free_gb": 5.0,
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
    legacy_training_seed = config["nht_training"].pop("seed", None)
    if legacy_training_seed is not None and int(legacy_training_seed) != int(
        config["seed"]
    ):
        raise ValueError(
            "nht_training.seed is obsolete and must equal the canonical top-level seed"
        )
    config["nht_training"].pop("seed_argument", None)
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
        backend = candidate.get("backend")
        sharing = candidate.get("camera_sharing", "single")
        supported_sharing = {
            "pycolmap_sift_incremental": {
                "single",
                "per_folder",
                "per_image",
                "segments",
            },
            "hloc_aliked_lightglue": {"single", "per_image"},
        }
        if backend not in supported_sharing:
            raise ValueError(
                f"Unsupported SfM backend for candidate {candidate['id']}: {backend}"
            )
        if sharing not in supported_sharing[backend]:
            modes = ", ".join(sorted(supported_sharing[backend]))
            raise ValueError(
                f"candidate {candidate['id']} camera_sharing={sharing!r} is invalid "
                f"for {backend}; choose one of: {modes}"
            )
        if sharing == "segments" and int(candidate.get("camera_segment_size", 30)) < 2:
            raise ValueError(
                f"candidate {candidate['id']} camera_segment_size must be at least 2"
            )
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
    if float(config["operations"]["minimum_free_gb"]) < 0:
        raise ValueError("operations.minimum_free_gb cannot be negative")


def earliest_affected_stage(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
) -> str | None:
    """Return the earliest stage whose owned configuration changed."""
    if previous is None:
        return "frames"
    for stage in STAGES:
        if any(previous.get(key) != current.get(key) for key in stage.config_sections):
            return stage.name
    if previous.get("schema") != current.get("schema"):
        return "frames"
    return None


def effective_start_stage(
    requested: str,
    affected: str | None,
    *,
    input_video_changed: bool,
) -> str:
    """Expand a request upstream when inputs or resolved configuration changed."""
    order = [stage.name for stage in STAGES]
    required = "frames" if input_video_changed else affected
    if required is None:
        return requested
    return order[min(order.index(requested), order.index(required))]


def write_resolved_config(path: Path, config: dict[str, Any]) -> None:
    """Write JSON syntax, which is also valid YAML 1.2, to the fixed YAML path."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(config, indent=2) + "\n")
    temporary.replace(path)
