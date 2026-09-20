"""Import calibrated source geometry with explicitly selected replacement RGB."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from PIL import Image

from .config import load_config, write_resolved_config
from .run_state import RunState
from .stages import stage_names

IMPORTED_STAGES = ("frames", "preprocess", "sfm", "sfm_selection")
SOURCE_METADATA = (
    "run.json",
    "resolved-config.yaml",
    "frames/frames.json",
    "frames/extraction.json",
    "sfm/reconstruction.json",
    "sfm/candidates/candidates.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _compatible(source: dict[str, Any], config: dict[str, Any]) -> None:
    for field in ("seed", "frames", "preprocess", "sfm"):
        if source[field] != config[field]:
            raise ValueError(f"Imported workspace must retain source {field}")
    for field in ("data_factor", "test_every"):
        if source["nht_training"][field] != config["nht_training"][field]:
            raise ValueError(
                f"Imported workspace must retain source nht_training.{field}"
            )
    if (
        config["nht_training"]["data_factor"]
        != config["preprocess"]["training_image_factor"]
    ):
        raise ValueError("Training/preprocessing factors must match")


def validate_import_request(
    workspace: Path, config: dict[str, Any], from_stage: str
) -> None:
    """Do not destroy imported geometry through automatic upstream invalidation."""
    path = workspace / "import-provenance.json"
    if not path.exists():
        return
    provenance = json.loads(path.read_text())
    if stage_names().index(from_stage) < stage_names().index("nht_training"):
        raise ValueError("Imported appearance workspaces cannot rerun upstream stages")
    _compatible(provenance["source_config"], config)
    if config["nht_training"]["image_names"] != provenance["image_names"]:
        raise ValueError(
            "Imported image selection is immutable; prepare another workspace"
        )
    for relative, expected in provenance["copied_sha256"].items():
        if _sha256(workspace / relative) != expected:
            raise ValueError(f"Imported input changed: {relative}")


def import_workspace(
    source: Path,
    replacement_images: Path,
    workspace: Path,
    scene_id: str,
    config: dict[str, Any],
) -> RunState:
    """Prepare a new training-only workspace without altering source files."""
    import pycolmap

    source, replacement_images, workspace = (
        source.resolve(),
        replacement_images.resolve(),
        workspace.resolve(),
    )
    if (
        source == workspace
        or source in workspace.parents
        or workspace in source.parents
    ):
        raise ValueError("Source and imported workspaces must be separate")
    if workspace.exists() and any(
        path.name != ".pipeline.lock" for path in workspace.iterdir()
    ):
        raise ValueError("Imported workspace destination must be empty")
    source_state = RunState.load(source)
    for stage in IMPORTED_STAGES:
        if source_state.payload["stages"][stage]["status"] != "completed":
            raise ValueError(f"Source stage is not completed: {stage}")
        source_state.validate_outputs(stage)
    source_config = load_config(source / "resolved-config.yaml")
    _compatible(source_config, config)
    names = config["nht_training"]["image_names"]
    if not names:
        raise ValueError("Source import requires explicit nht_training.image_names")
    reconstruction = pycolmap.Reconstruction(source / "sfm/model")
    registered = {image.name: image for image in reconstruction.images.values()}
    full_names = sorted(registered)
    unknown = set(names) - set(full_names)
    if unknown:
        raise ValueError(f"Images are not registered by source SfM: {sorted(unknown)}")
    if not replacement_images.is_dir() or replacement_images.is_symlink():
        raise ValueError("replacement-images must be a regular directory")
    actual_names = {
        path.relative_to(replacement_images).as_posix()
        for path in replacement_images.rglob("*")
        if path.is_file()
    }
    if actual_names != set(names):
        raise ValueError("Replacement files must exactly match selected image names")
    frame_metadata = json.loads((source / "frames/frames.json").read_text())
    source_frames = {
        frame["filename"]: frame
        for frame in frame_metadata["frames"]
        if frame["accepted"]
    }
    if not set(names).issubset(source_frames):
        raise ValueError("Selected SfM images must have accepted source frame metadata")
    test_every = config["nht_training"]["test_every"]
    indices = [full_names.index(name) for name in names]
    validation_names = [
        name for name, index in zip(names, indices) if index % test_every == 0
    ]
    if not validation_names or len(validation_names) == len(names):
        raise ValueError(
            "Selection must retain both original train and validation splits"
        )
    factor = config["nht_training"]["data_factor"]
    replacement_hashes = {}
    for name in names:
        path = replacement_images / name
        if path.is_symlink():
            raise ValueError("Replacement images must not be symbolic links")
        camera = reconstruction.cameras[registered[name].camera_id]
        with Image.open(path) as image:
            image.load()
            if image.size != (camera.width, camera.height):
                raise ValueError(
                    f"Replacement size differs from calibrated source size: {name}"
                )
            if image.mode != "RGB":
                raise ValueError(f"Replacement must be RGB: {name}")
            if min(image.size) // factor < 1:
                raise ValueError("Training image factor produces an empty image")
        replacement_hashes[name] = _sha256(path)
    source_paths = [source / name for name in SOURCE_METADATA]
    source_paths.extend(
        sorted(path for path in (source / "sfm/model").rglob("*") if path.is_file())
    )
    if any(path.is_symlink() for path in source_paths):
        raise ValueError("Source metadata and SfM must be regular files")
    source_hashes = {
        str(path.relative_to(source)): _sha256(path) for path in source_paths
    }
    workspace.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".appearance-import-", dir=workspace))
    try:
        for relative in SOURCE_METADATA:
            destination = staging / "source-provenance" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        shutil.copytree(source / "sfm/model", staging / "sfm/model")
        for relative in (
            "frames/extraction.json",
            "sfm/reconstruction.json",
            "sfm/candidates/candidates.json",
        ):
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source / relative, destination)
        # Only timing/identity is inherited into generated-image records. Original
        # brightness/sharpness measurements stay under source-provenance/.
        transformed_frames = [
            {
                key: source_frames[name][key]
                for key in (
                    "filename",
                    "source_frame_index",
                    "source_time_seconds",
                    "accepted",
                )
            }
            for name in names
        ]
        _write_json(
            staging / "frames/frames.json",
            {
                "schema": "nht_frames_v1",
                "input_frame_count": len(names),
                "accepted_frame_count": len(names),
                "rejected_frame_count": 0,
                "frames": transformed_frames,
                "appearance_import": {
                    "source_frame_metadata": "source-provenance/frames/frames.json",
                    "image_quality_metrics": "not_measured",
                },
                "training_images": {
                    "factor": factor,
                    "source_count": len(names),
                    "output_count": len(names),
                },
            },
        )
        for name in names:
            image_path = staging / "frames/images" / name
            scaled_path = (
                staging / "frames/training-images" / Path(name).with_suffix(".png")
            )
            image_path.parent.mkdir(parents=True, exist_ok=True)
            scaled_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(replacement_images / name, image_path)
            with Image.open(image_path) as image:
                image.resize(
                    (image.width // factor, image.height // factor),
                    Image.Resampling.LANCZOS,
                ).save(scaled_path)
        for relative, expected in source_hashes.items():
            if _sha256(source / relative) != expected:
                raise ValueError(f"Source changed during import: {relative}")
        for name, expected in replacement_hashes.items():
            if _sha256(staging / "frames/images" / name) != expected:
                raise ValueError(f"Replacement changed during import: {name}")
        copied_hashes = {
            str(path.relative_to(staging)): _sha256(path)
            for directory in ("frames", "sfm")
            for path in sorted((staging / directory).rglob("*"))
            if path.is_file()
        }
        _write_json(
            staging / "import-provenance.json",
            {
                "schema": "nht_appearance_import_v1",
                "source_workspace": str(source),
                "source_scene_id": source_state.payload["scene_id"],
                "source_config": source_config,
                "replacement_images": str(replacement_images),
                "image_names": names,
                "source_indices": indices,
                "full_image_count": len(full_names),
                "validation_names": validation_names,
                "copy_method": "copy",
                "source_sha256": source_hashes,
                "replacement_sha256": replacement_hashes,
                "copied_sha256": copied_hashes,
            },
        )
        video = source_state.payload.get("input_video")
        state = RunState.create_or_load(
            staging, scene_id, Path(video) if video else None
        )
        for stage in IMPORTED_STAGES:
            state.validate_outputs(stage)
            state.mark_completed(
                stage,
                {
                    "origin": "imported",
                    "imported_from": str(source),
                    "copy_method": "copy",
                    "selected_image_count": len(names),
                    "provenance": "import-provenance.json",
                },
            )
        write_resolved_config(staging / "resolved-config.yaml", config)
        for path in list(staging.iterdir()):
            path.rename(workspace / path.name)
        return RunState.load(workspace)
    finally:
        shutil.rmtree(staging)
