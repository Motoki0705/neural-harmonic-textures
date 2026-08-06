"""Typed stage DAG and fixed workspace ownership."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StageDefinition:
    name: str
    dependencies: tuple[str, ...]
    required_inputs: tuple[Path, ...]
    owned_paths: tuple[Path, ...]
    fixed_outputs: tuple[Path, ...]


STAGES: tuple[StageDefinition, ...] = (
    StageDefinition(
        "frames",
        (),
        (),
        (Path("frames/raw"), Path("frames/extraction.json")),
        (Path("frames/extraction.json"),),
    ),
    StageDefinition(
        "preprocess",
        ("frames",),
        (Path("frames/raw"), Path("frames/extraction.json")),
        (
            Path("frames/images"),
            Path("frames/training-images"),
            Path("frames/frames.json"),
        ),
        (Path("frames/images"), Path("frames/frames.json")),
    ),
    StageDefinition(
        "sfm",
        ("preprocess",),
        (Path("frames/images"), Path("frames/frames.json")),
        (Path("sfm/candidates"),),
        (Path("sfm/candidates/candidates.json"),),
    ),
    StageDefinition(
        "sfm_selection",
        ("sfm",),
        (Path("sfm/candidates/candidates.json"),),
        (
            Path("sfm/model"),
            Path("sfm/reconstruction.json"),
            Path("sfm/diagnostics"),
        ),
        (Path("sfm/model"), Path("sfm/reconstruction.json")),
    ),
    StageDefinition(
        "nht_training",
        ("sfm_selection",),
        (
            Path("frames/images"),
            Path("frames/training-images"),
            Path("sfm/model"),
            Path("sfm/reconstruction.json"),
        ),
        (Path("3dgs"),),
        (
            Path("3dgs/training.json"),
            Path("3dgs/model"),
            Path("3dgs/diagnostics"),
            Path("3dgs/renders"),
        ),
    ),
    StageDefinition(
        "scene_export",
        ("nht_training",),
        (
            Path("frames/frames.json"),
            Path("sfm/model"),
            Path("sfm/reconstruction.json"),
            Path("3dgs/training.json"),
            Path("3dgs/model"),
        ),
        (Path("export"),),
        (
            Path("export/scene.json"),
            Path("export/cameras.json"),
            Path("export/points_scene.npy"),
            Path("export/images"),
            Path("export/model"),
        ),
    ),
    StageDefinition(
        "reconstruction_report",
        ("scene_export",),
        (
            Path("export/scene.json"),
            Path("export/cameras.json"),
            Path("export/points_scene.npy"),
        ),
        (Path("reconstruction-report.json"),),
        (Path("reconstruction-report.json"),),
    ),
)

STAGE_BY_NAME = {stage.name: stage for stage in STAGES}


def stage_names() -> tuple[str, ...]:
    return tuple(stage.name for stage in STAGES)


def descendants(name: str, include_self: bool = False) -> tuple[str, ...]:
    if name not in STAGE_BY_NAME:
        raise KeyError(f"Unknown stage: {name}")
    selected: set[str] = {name} if include_self else set()
    changed = True
    while changed:
        changed = False
        for stage in STAGES:
            if stage.name not in selected and any(
                dependency == name or dependency in selected
                for dependency in stage.dependencies
            ):
                selected.add(stage.name)
                changed = True
    return tuple(stage.name for stage in STAGES if stage.name in selected)


def execution_order(from_stage: str) -> tuple[str, ...]:
    return descendants(from_stage, include_self=True)
