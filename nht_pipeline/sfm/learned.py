"""HLOC ALIKED/LightGlue reconstruction retry candidate."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

from .pairs import write_pair_graph


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
        _add_import_path(config.get(key))
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
            "The ALIKED retry requires HLOC and LightGlue; install them or set "
            "sfm candidate hloc_root/lightglue_root/site_packages"
        ) from error

    started = time.monotonic()
    candidate_dir.mkdir(parents=True, exist_ok=True)
    features = extract_features.main(
        extract_features.confs["aliked-n16"], image_dir, candidate_dir
    )
    retrieval = extract_features.main(
        extract_features.confs["netvlad"], image_dir, candidate_dir
    )
    retrieval_pairs = candidate_dir / "pairs-retrieval.txt"
    pairs_from_retrieval.main(
        retrieval,
        retrieval_pairs,
        int(config.get("retrieval_neighbors", 10)),
    )
    pairs = candidate_dir / "pairs.txt"
    pair_summary = write_pair_graph(
        image_dir,
        pairs,
        int(config.get("sequential_overlap", 10)),
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
        "num_threads": int(config.get("num_threads", 4)),
    }
    result = reconstruction.main(
        model_dir,
        image_dir,
        pairs,
        features,
        matches_path,
        camera_mode=pycolmap.CameraMode.SINGLE,
        image_options={
            "camera_model": config.get("camera_model", "OPENCV"),
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
    }
