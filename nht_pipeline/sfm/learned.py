"""HLOC ALIKED/LightGlue reconstruction retry candidate."""

from __future__ import annotations

import copy
import sys
import time
from pathlib import Path
from typing import Any

from .pairs import write_pair_graph


def learned_feature_config(base: dict[str, Any], max_image_size: int) -> dict[str, Any]:
    """Apply the public candidate resize setting to an isolated HLOC config."""
    config = copy.deepcopy(base)
    config.setdefault("preprocessing", {})["resize_max"] = int(max_image_size)
    return config


def _camera_mode(config: dict[str, Any], pycolmap: Any) -> Any:
    sharing = config["camera_sharing"]
    if sharing == "single":
        return pycolmap.CameraMode.SINGLE
    if sharing == "per_image":
        return pycolmap.CameraMode.PER_IMAGE
    raise ValueError(
        "HLOC supports camera_sharing=single or per_image; use the SIFT backend "
        "for per_folder or segments"
    )


def _add_import_path(value: str | None) -> None:
    if value:
        path = str(Path(value).resolve())
        if path not in sys.path:
            sys.path.insert(0, path)


def run_aliked_lightglue(
    image_dir: Path,
    candidate_dir: Path,
    config: dict[str, Any],
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    for key in ("site_packages", "lightglue_root", "hloc_root"):
        _add_import_path(config[key])
    try:
        import pycolmap
        from hloc import (
            extract_features,
            match_features,
            pairs_from_retrieval,
            reconstruction,
        )
    except ImportError as error:  # pragma: no cover - optional retry runtime
        raise RuntimeError(
            "Missing dependency for the ALIKED retry: "
            f"{type(error).__name__}: {error}. Install HLOC and LightGlue or set "
            "sfm candidate hloc_root/lightglue_root/site_packages."
        ) from error

    started = time.monotonic()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features.main(
        learned_feature_config(
            extract_features.confs["aliked-n16"],
            int(config["max_image_size"]),
        ),
        image_dir,
        candidate_dir,
    )
    retrieval = extract_features.main(
        extract_features.confs["netvlad"], image_dir, candidate_dir
    )
    retrieval_pairs = candidate_dir / "pairs-retrieval.txt"
    pairs_from_retrieval.main(
        retrieval,
        retrieval_pairs,
        int(config["retrieval_neighbors"]),
    )
    pairs = candidate_dir / "pairs.txt"
    pair_summary = write_pair_graph(
        image_dir,
        pairs,
        int(config["sequential_overlap"]),
        retrieval_pairs,
        candidate_dir / "pair-graph.json",
    )
    matches_path = candidate_dir / "matches-aliked-lightglue.h5"
    match_features.main(
        match_features.confs["aliked+lightglue"],
        pairs,
        features,
        matches=matches_path,
    )
    model_dir = candidate_dir / "model"
    mapper_options = {
        "multiple_models": False,
        "min_model_size": 10,
        "ba_refine_focal_length": True,
        "ba_refine_principal_point": False,
        "ba_refine_extra_params": True,
        "random_seed": seed,
        "num_threads": int(config["num_threads"]),
    }
    result = reconstruction.main(
        model_dir,
        image_dir,
        pairs,
        features,
        matches_path,
        camera_mode=_camera_mode(config, pycolmap),
        image_options={
            "camera_model": config["camera_model"],
            "default_focal_length_factor": 1.0,
        },
        mapper_options=mapper_options,
    )
    if result is None:
        raise RuntimeError("HLOC/COLMAP produced no reconstruction")
    return result, {
        "backend": "hloc_aliked_lightglue",
        "elapsed_seconds": time.monotonic() - started,
        "pair_graph": pair_summary,
        "selected_registered_images": result.num_reg_images(),
        "selected_sparse_points": result.num_points3D(),
        "camera_sharing": config["camera_sharing"],
        "reconstructed_cameras": result.num_cameras(),
        "effective_seed": seed,
    }
