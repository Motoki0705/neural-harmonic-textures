"""Resolved configuration for the video-to-scene pipeline."""

from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass
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
        "adapter": None,
        "extra_args": [],
        "cuda_device": 0,
        "camera_model": "pinhole",
        "pose_opt": False,
        "post_processing": None,
        "near_plane": 0.01,
        "far_plane": 1.0e10,
    },
    "export": {
        "schema": "nht_standard_scene_v1",
    },
    "operations": {
        "minimum_free_gb": 5.0,
    },
}

_TOP_LEVEL_KEYS = frozenset(DEFAULT_CONFIG)
_SECTION_KEYS = {
    name: frozenset(value)
    for name, value in DEFAULT_CONFIG.items()
    if isinstance(value, dict) and name != "sfm"
}
_SFM_KEYS = frozenset(DEFAULT_CONFIG["sfm"])
_QUALITY_GATE_KEYS = frozenset(DEFAULT_CONFIG["sfm"]["quality_gates"])
_SHORT_QUALITY_GATE_KEYS = frozenset(DEFAULT_CONFIG["sfm"]["short_clip_quality_gates"])
_COMMON_CANDIDATE_KEYS = {
    "id",
    "backend",
    "run_condition",
    "camera_model",
    "camera_sharing",
    "sequential_overlap",
    "max_image_size",
    "num_threads",
}
_CLASSIC_CANDIDATE_KEYS = frozenset(
    _COMMON_CANDIDATE_KEYS | {"quadratic_overlap", "max_features"}
)
_LEARNED_CANDIDATE_KEYS = frozenset(
    _COMMON_CANDIDATE_KEYS
    | {
        "retrieval_neighbors",
        "site_packages",
        "lightglue_root",
        "hloc_root",
    }
)


def _expect_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{context} must be a mapping")
    return value


def _expect_exact_keys(
    value: dict[str, Any], allowed: frozenset[str], context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    missing = sorted(allowed - set(value))
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")
    if missing:
        raise ValueError(f"Missing {context} keys: {missing}")


def _expect_override_keys(
    value: dict[str, Any], allowed: frozenset[str], context: str
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"Unknown {context} keys: {unknown}")


def _expect_int(value: Any, context: str, *, minimum: int | None = None) -> int:
    if type(value) is not int:
        raise TypeError(f"{context} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return value


def _expect_number(value: Any, context: str, *, minimum: float | None = None) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{context} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{context} must be finite")
    if minimum is not None and result < minimum:
        raise ValueError(f"{context} must be at least {minimum}")
    return result


def _expect_bool(value: Any, context: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{context} must be a boolean")
    return value


def _expect_string(value: Any, context: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{context} must be a non-empty string")
    return value


def _expect_optional_path(value: Any, context: str) -> str | None:
    if value is None:
        return None
    return _expect_string(value, context)


def _validate_extra_args(value: Any) -> list[str]:
    if not isinstance(value, list) or any(type(item) is not str for item in value):
        raise TypeError("nht_training.extra_args must be a list of strings")
    result: list[str] = []
    seen: set[str] = set()
    index = 0
    while index < len(value):
        option = value[index]
        if option in seen:
            raise ValueError(f"Duplicate nht_training.extra_args option: {option}")
        seen.add(option)
        if option == "--disable_video":
            result.append(option)
            index += 1
            continue
        if option == "--num_workers":
            if index + 1 >= len(value):
                raise ValueError("--num_workers requires a non-negative integer")
            raw = value[index + 1]
            try:
                workers = int(raw)
            except ValueError as error:
                raise ValueError(
                    "--num_workers requires a non-negative integer"
                ) from error
            if str(workers) != raw or workers < 0:
                raise ValueError("--num_workers requires a non-negative integer")
            result.extend((option, raw))
            index += 2
            continue
        raise ValueError(f"Unsupported nht_training.extra_args option: {option}")
    return result


@dataclass(frozen=True)
class NhtTrainingConfig:
    """Typed, fail-closed production envelope shared with the NHT trainer."""

    data_factor: int
    max_steps: int
    cap_max: int
    test_every: int
    lpips_net: str
    python: str | None
    trainer: str | None
    adapter: str | None
    extra_args: tuple[str, ...]
    cuda_device: int
    camera_model: str
    pose_opt: bool
    post_processing: None
    near_plane: float
    far_plane: float

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> NhtTrainingConfig:
        _expect_exact_keys(value, _SECTION_KEYS["nht_training"], "nht_training")
        camera_model = _expect_string(
            value["camera_model"], "nht_training.camera_model"
        )
        if camera_model != "pinhole":
            raise ValueError("Production NHT supports camera_model=pinhole only")
        pose_opt = _expect_bool(value["pose_opt"], "nht_training.pose_opt")
        if pose_opt:
            raise ValueError("Production NHT requires pose_opt=false")
        if value["post_processing"] is not None:
            raise ValueError("Production NHT requires post_processing=null")
        near_plane = _expect_number(
            value["near_plane"], "nht_training.near_plane", minimum=0.0
        )
        far_plane = _expect_number(
            value["far_plane"], "nht_training.far_plane", minimum=0.0
        )
        if near_plane <= 0 or far_plane <= near_plane:
            raise ValueError("nht_training requires 0 < near_plane < far_plane")
        return cls(
            data_factor=_expect_int(
                value["data_factor"], "nht_training.data_factor", minimum=1
            ),
            max_steps=_expect_int(
                value["max_steps"], "nht_training.max_steps", minimum=1
            ),
            cap_max=_expect_int(value["cap_max"], "nht_training.cap_max", minimum=1),
            test_every=_expect_int(
                value["test_every"], "nht_training.test_every", minimum=1
            ),
            lpips_net=_expect_string(value["lpips_net"], "nht_training.lpips_net"),
            python=_expect_optional_path(value["python"], "nht_training.python"),
            trainer=_expect_optional_path(value["trainer"], "nht_training.trainer"),
            adapter=_expect_optional_path(value["adapter"], "nht_training.adapter"),
            extra_args=tuple(_validate_extra_args(value["extra_args"])),
            cuda_device=_expect_int(
                value["cuda_device"], "nht_training.cuda_device", minimum=0
            ),
            camera_model=camera_model,
            pose_opt=pose_opt,
            post_processing=None,
            near_plane=near_plane,
            far_plane=far_plane,
        )

    def to_mapping(self) -> dict[str, Any]:
        return {
            "data_factor": self.data_factor,
            "max_steps": self.max_steps,
            "cap_max": self.cap_max,
            "test_every": self.test_every,
            "lpips_net": self.lpips_net,
            "python": self.python,
            "trainer": self.trainer,
            "adapter": self.adapter,
            "extra_args": list(self.extra_args),
            "cuda_device": self.cuda_device,
            "camera_model": self.camera_model,
            "pose_opt": self.pose_opt,
            "post_processing": self.post_processing,
            "near_plane": self.near_plane,
            "far_plane": self.far_plane,
        }


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _validate_override(override: dict[str, Any]) -> None:
    _expect_override_keys(override, _TOP_LEVEL_KEYS, "top-level")
    for section, allowed in _SECTION_KEYS.items():
        if section in override:
            _expect_override_keys(
                _expect_mapping(override[section], section), allowed, section
            )
    if "sfm" not in override:
        return
    sfm = _expect_mapping(override["sfm"], "sfm")
    _expect_override_keys(sfm, _SFM_KEYS, "sfm")
    if "quality_gates" in sfm:
        _expect_override_keys(
            _expect_mapping(sfm["quality_gates"], "sfm.quality_gates"),
            _QUALITY_GATE_KEYS,
            "sfm.quality_gates",
        )
    if "short_clip_quality_gates" in sfm:
        _expect_override_keys(
            _expect_mapping(
                sfm["short_clip_quality_gates"],
                "sfm.short_clip_quality_gates",
            ),
            _SHORT_QUALITY_GATE_KEYS,
            "sfm.short_clip_quality_gates",
        )
    if "candidates" in sfm:
        candidates = sfm["candidates"]
        if not isinstance(candidates, list):
            raise TypeError("sfm.candidates must be a list")
        for index, raw_candidate in enumerate(candidates):
            candidate = _expect_mapping(raw_candidate, f"sfm.candidates[{index}]")
            backend = candidate.get("backend")
            if type(backend) is not str:
                raise TypeError(f"candidate {index}.backend must be a string")
            candidate_allowed = {
                "pycolmap_sift_incremental": _CLASSIC_CANDIDATE_KEYS
                | {"camera_segment_size"},
                "hloc_aliked_lightglue": _LEARNED_CANDIDATE_KEYS,
            }.get(backend)
            if candidate_allowed is None:
                raise ValueError(
                    f"Unsupported SfM backend for candidate {index}: {backend}"
                )
            _expect_override_keys(
                candidate, frozenset(candidate_allowed), f"candidate {index}"
            )


def _normalize_config(config: dict[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(config)
    for candidate in normalized["sfm"]["candidates"]:
        if candidate.get("backend") == "hloc_aliked_lightglue":
            for key in ("site_packages", "lightglue_root", "hloc_root"):
                candidate.setdefault(key, None)
    normalized["nht_training"] = NhtTrainingConfig.from_mapping(
        normalized["nht_training"]
    ).to_mapping()
    return normalized


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        config = _normalize_config(DEFAULT_CONFIG)
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
    _validate_override(override)
    config = _normalize_config(_merge(DEFAULT_CONFIG, override))
    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    _expect_exact_keys(config, _TOP_LEVEL_KEYS, "top-level")
    if config["schema"] != "nht_pipeline_config_v1":
        raise ValueError("schema must be nht_pipeline_config_v1")
    _expect_int(config["seed"], "seed", minimum=0)
    for section, section_keys in _SECTION_KEYS.items():
        _expect_exact_keys(
            _expect_mapping(config[section], section), section_keys, section
        )
    sfm = _expect_mapping(config["sfm"], "sfm")
    _expect_exact_keys(sfm, _SFM_KEYS, "sfm")
    _expect_exact_keys(
        _expect_mapping(sfm["quality_gates"], "sfm.quality_gates"),
        _QUALITY_GATE_KEYS,
        "sfm.quality_gates",
    )
    _expect_exact_keys(
        _expect_mapping(
            sfm["short_clip_quality_gates"], "sfm.short_clip_quality_gates"
        ),
        _SHORT_QUALITY_GATE_KEYS,
        "sfm.short_clip_quality_gates",
    )

    frames = config["frames"]
    if _expect_number(frames["frames_per_second"], "frames.frames_per_second") <= 0:
        raise ValueError("frames.frames_per_second must be positive")
    _expect_int(frames["jpeg_quality"], "frames.jpeg_quality", minimum=1)

    preprocess = config["preprocess"]
    for name in (
        "absolute_minimum_sharpness",
        "p05_sharpness_fraction",
        "maximum_clipped_fraction",
        "minimum_temporal_difference",
    ):
        _expect_number(preprocess[name], f"preprocess.{name}", minimum=0.0)
    _expect_int(
        preprocess["training_image_factor"],
        "preprocess.training_image_factor",
        minimum=1,
    )
    if preprocess["minimum_temporal_difference"] < 0:
        raise ValueError("preprocess.minimum_temporal_difference cannot be negative")

    _expect_int(
        sfm["minimum_supported_points_per_camera"],
        "sfm.minimum_supported_points_per_camera",
        minimum=1,
    )
    _expect_int(sfm["short_clip_max_images"], "sfm.short_clip_max_images", minimum=1)
    for section in ("quality_gates", "short_clip_quality_gates"):
        for name, value in sfm[section].items():
            context = f"sfm.{section}.{name}"
            if name in {
                "minimum_sparse_points",
                "maximum_trajectory_step_outliers",
                "maximum_mapping_components",
            }:
                _expect_int(value, context, minimum=0)
            else:
                _expect_number(value, context, minimum=0.0)

    candidates = sfm["candidates"]
    if not isinstance(candidates, list):
        raise TypeError("sfm.candidates must be a list")
    if not candidates:
        raise ValueError("sfm candidates cannot be empty")
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"sfm.candidates[{index}] must be a mapping")
        if "id" not in candidate:
            raise ValueError(f"sfm.candidates[{index}] is missing id")
    identifiers = [
        _expect_string(candidate["id"], f"candidate {index}.id")
        for index, candidate in enumerate(candidates)
    ]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("sfm candidates must have unique IDs")
    maximum = _expect_int(
        sfm["maximum_candidates"], "sfm.maximum_candidates", minimum=1
    )
    if maximum < 1 or maximum > len(candidates):
        raise ValueError("sfm.maximum_candidates must select one or more candidates")
    for index, raw_candidate in enumerate(candidates):
        candidate = _expect_mapping(raw_candidate, f"sfm.candidates[{index}]")
        backend = candidate.get("backend")
        if backend == "pycolmap_sift_incremental":
            candidate_keys = set(_CLASSIC_CANDIDATE_KEYS)
            if candidate.get("camera_sharing") == "segments":
                candidate_keys.add("camera_segment_size")
        elif backend == "hloc_aliked_lightglue":
            candidate_keys = set(_LEARNED_CANDIDATE_KEYS)
        else:
            raise ValueError(
                f"Unsupported SfM backend for candidate {index}: {backend}"
            )
        _expect_exact_keys(candidate, frozenset(candidate_keys), f"candidate {index}")
        identifier = _expect_string(candidate["id"], f"candidate {index}.id")
        _expect_string(
            candidate["camera_model"], f"candidate {identifier}.camera_model"
        )
        run_condition = _expect_string(
            candidate["run_condition"], f"candidate {identifier}.run_condition"
        )
        if run_condition not in {
            "always",
            "primary_rejected",
        }:
            raise ValueError(f"Unsupported candidate run_condition: {identifier}")
        sharing = _expect_string(
            candidate["camera_sharing"], f"candidate {identifier}.camera_sharing"
        )
        supported_sharing = {
            "pycolmap_sift_incremental": {
                "single",
                "per_folder",
                "per_image",
                "segments",
            },
            "hloc_aliked_lightglue": {"single", "per_image"},
        }
        if sharing not in supported_sharing[backend]:
            modes = ", ".join(sorted(supported_sharing[backend]))
            raise ValueError(
                f"candidate {identifier} camera_sharing={sharing!r} is invalid "
                f"for {backend}; choose one of: {modes}"
            )
        for name in ("sequential_overlap", "max_image_size", "num_threads"):
            _expect_int(candidate[name], f"candidate {identifier}.{name}", minimum=1)
        if backend == "pycolmap_sift_incremental":
            _expect_bool(
                candidate["quadratic_overlap"],
                f"candidate {identifier}.quadratic_overlap",
            )
            _expect_int(
                candidate["max_features"],
                f"candidate {identifier}.max_features",
                minimum=1,
            )
            if sharing == "segments":
                _expect_int(
                    candidate["camera_segment_size"],
                    f"candidate {identifier}.camera_segment_size",
                    minimum=2,
                )
        else:
            _expect_int(
                candidate["retrieval_neighbors"],
                f"candidate {identifier}.retrieval_neighbors",
                minimum=1,
            )
            for name in ("site_packages", "lightglue_root", "hloc_root"):
                if name in candidate:
                    _expect_optional_path(
                        candidate[name], f"candidate {identifier}.{name}"
                    )

    training = NhtTrainingConfig.from_mapping(config["nht_training"])
    if training.data_factor != preprocess["training_image_factor"]:
        raise ValueError(
            "nht_training.data_factor must equal preprocess.training_image_factor"
        )
    if config["export"]["schema"] != "nht_standard_scene_v1":
        raise ValueError("export.schema must be nht_standard_scene_v1")
    _expect_number(
        config["operations"]["minimum_free_gb"],
        "operations.minimum_free_gb",
        minimum=0.0,
    )


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
