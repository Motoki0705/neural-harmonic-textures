"""Stage implementations and canonical mutable workspace orchestration."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from .errors import classify_error
from .export import create_scene_export, validate_scene_export
from .frames import (
    downsample_images,
    extract_frames,
    list_images,
    preprocess_frames,
    write_json,
)
from .preflight import preflight_stage
from .run_state import RunState
from .sfm.classic import run_sift_incremental
from .sfm.learned import run_aliked_lightglue
from .sfm.metrics import evaluate_reconstruction, write_trajectory_diagnostics
from .sfm.select import NoValidCandidateError, select_candidate
from .stages import execution_order, stage_config
from .training import run_training
from .workspace import (
    capture_stage_log,
    cleanup_staging,
    invalidate_published_outputs,
    link_or_copy,
    publish_stage,
    remove_path,
    stage_staging_root,
)


@dataclass(frozen=True)
class PipelineContext:
    workspace: Path
    state: RunState
    config: dict[str, Any]
    repository_root: Path
    staging_root: Path | None = None

    @property
    def output_root(self) -> Path:
        return self.staging_root or self.workspace


def _run_frames(context: PipelineContext) -> dict[str, Any]:
    input_value = context.state.payload.get("input_video")
    if not input_value:
        raise ValueError("frames stage requires --input-video")
    config = context.config["frames"]
    summary = extract_frames(
        Path(input_value),
        context.output_root / "frames/raw",
        float(config["frames_per_second"]),
        int(config["jpeg_quality"]),
    )
    write_json(context.output_root / "frames/extraction.json", summary)
    return summary


def _run_preprocess(context: PipelineContext) -> dict[str, Any]:
    frame_config = context.config["frames"]
    config = context.config["preprocess"]
    summary = preprocess_frames(
        context.workspace / "frames/raw",
        context.output_root / "frames/images",
        float(frame_config["frames_per_second"]),
        float(config["absolute_minimum_sharpness"]),
        float(config["p05_sharpness_fraction"]),
        float(config["maximum_clipped_fraction"]),
        float(config["minimum_temporal_difference"]),
    )
    summary["training_images"] = downsample_images(
        context.output_root / "frames/images",
        context.output_root / "frames/training-images",
        int(config["training_image_factor"]),
    )
    write_json(context.output_root / "frames/frames.json", summary)
    return {key: value for key, value in summary.items() if key != "frames"}


SFM_BACKENDS: dict[
    str, Callable[[Path, Path, dict[str, Any], int], tuple[Any, dict[str, Any]]]
] = {
    "pycolmap_sift_incremental": run_sift_incremental,
    "hloc_aliked_lightglue": run_aliked_lightglue,
}


def _directory_size(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def _run_sfm(context: PipelineContext) -> dict[str, Any]:
    sfm_config = context.config["sfm"]
    image_dir = context.workspace / "frames/images"
    input_count = len(list_images(image_dir))
    candidate_records: list[dict[str, Any]] = []
    primary_accepted = False
    maximum = int(sfm_config["maximum_candidates"])
    for candidate_config in sfm_config["candidates"][:maximum]:
        identifier = candidate_config["id"]
        condition = candidate_config.get("run_condition", "always")
        if condition == "primary_rejected" and primary_accepted:
            candidate_records.append(
                {
                    "id": identifier,
                    "backend": candidate_config["backend"],
                    "status": "skipped",
                    "reason": "primary candidate passed every semantic gate",
                }
            )
            continue
        candidate_dir = context.output_root / "sfm/candidates" / identifier
        candidate_dir.mkdir(parents=True, exist_ok=True)
        backend_name = candidate_config["backend"]
        backend = SFM_BACKENDS.get(backend_name)
        if backend is None:
            raise ValueError(f"Unknown SfM backend: {backend_name}")
        record: dict[str, Any] = {
            "id": identifier,
            "backend": backend_name,
            "status": "running",
        }
        try:
            _, runtime = backend(
                image_dir,
                candidate_dir,
                candidate_config,
                int(context.config["seed"]),
            )
            metrics = evaluate_reconstruction(
                candidate_dir / "model",
                image_dir,
                input_count,
                int(sfm_config["minimum_supported_points_per_camera"]),
                (
                    {
                        **sfm_config["quality_gates"],
                        **sfm_config.get("short_clip_quality_gates", {}),
                    }
                    if input_count <= int(sfm_config.get("short_clip_max_images", 90))
                    else sfm_config["quality_gates"]
                ),
            )
            metrics["quality_profile"] = (
                "short_clip"
                if input_count <= int(sfm_config.get("short_clip_max_images", 90))
                else "standard"
            )
            write_trajectory_diagnostics(candidate_dir / "model", candidate_dir)
            runtime["stored_bytes"] = _directory_size(candidate_dir)
            write_json(candidate_dir / "metrics.json", metrics)
            record.update(
                {
                    "status": "completed",
                    "model": f"{identifier}/model",
                    "runtime": runtime,
                    "config": candidate_config,
                    "metrics": metrics,
                }
            )
            primary_accepted = primary_accepted or bool(metrics["accepted"])
        # A candidate failure is data: the explicit retry may still recover.
        except Exception as error:  # noqa: BLE001
            record.update(
                {
                    "status": "failed",
                    "error": {
                        "type": type(error).__name__,
                        "message": str(error),
                        "category": classify_error(error),
                    },
                }
            )
        candidate_records.append(record)
        write_json(candidate_dir / "candidate.json", record)

    manifest = {
        "schema": "nht_sfm_candidates_v1",
        "input_images": input_count,
        "candidates": candidate_records,
        "completed_candidates": sum(
            record["status"] == "completed" for record in candidate_records
        ),
        "effective_seeds": {
            "sfm": int(context.config["seed"]),
            "pair_generation": int(context.config["seed"]),
        },
    }
    write_json(context.output_root / "sfm/candidates/candidates.json", manifest)
    attempt = context.state.payload["stages"]["sfm"]["attempts"]
    diagnostic_path = (
        context.workspace / "logs/sfm" / f"attempt-{attempt}-candidates.json"
    )
    write_json(diagnostic_path, manifest)
    if not manifest["completed_candidates"]:
        failures = {
            record["id"]: record.get("error")
            for record in candidate_records
            if record["status"] == "failed"
        }
        raise RuntimeError(f"All SfM candidates failed to execute: {failures}")
    return manifest


def _run_sfm_selection(context: PipelineContext) -> dict[str, Any]:
    candidates_manifest = json.loads(
        (context.workspace / "sfm/candidates/candidates.json").read_text()
    )
    candidates = [
        record
        for record in candidates_manifest["candidates"]
        if record["status"] == "completed"
    ]
    diagnostics = context.output_root / "sfm/diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    comparison_path = diagnostics / "candidate-comparison.json"
    try:
        selection = select_candidate(candidates)
    except NoValidCandidateError as error:
        failure = {
            "schema": "nht_sfm_selection_failure_v1",
            "status": "failed",
            "selected_candidate": None,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        write_json(diagnostics / "selection.json", failure)
        write_json(
            comparison_path,
            {
                "schema": "nht_sfm_candidate_comparison_v1",
                "selection": failure,
                "candidates": candidates_manifest["candidates"],
            },
        )
        raise
    identifier = selection["selected_candidate"]
    source_model = context.workspace / "sfm/candidates" / identifier / "model"
    destination_model = context.output_root / "sfm/model"
    shutil.copytree(source_model, destination_model)
    write_json(diagnostics / "selection.json", selection)
    write_json(
        comparison_path,
        {
            "schema": "nht_sfm_candidate_comparison_v1",
            "selection": selection,
            "candidates": candidates_manifest["candidates"],
        },
    )
    reconstruction = {
        "schema": "nht_selected_sfm_v1",
        **selection,
        "model": "model",
    }
    write_json(context.output_root / "sfm/reconstruction.json", reconstruction)
    return {
        "selected_candidate": identifier,
        "selected_backend": selection["selected_backend"],
        "registered_images": selection["metrics"]["registered_images"],
        "supported_registered_images": selection["metrics"][
            "supported_registered_images"
        ],
        "sparse_points": selection["metrics"]["sparse_points"],
    }


def _run_nht_training(context: PipelineContext) -> dict[str, Any]:
    config = context.config["nht_training"]
    factor = int(config["data_factor"])
    preprocessing_factor = int(context.config["preprocess"]["training_image_factor"])
    if factor != preprocessing_factor:
        raise ValueError(
            "nht_training.data_factor must equal preprocess.training_image_factor"
        )
    dataset = context.output_root / "3dgs/dataset"
    sparse = dataset / "sparse/0"
    sparse.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(context.workspace / "sfm/model", sparse)
    link_methods = {
        "images": link_or_copy(
            context.workspace / "frames/images", dataset / "images", directory=True
        ),
        f"images_{factor}": link_or_copy(
            context.workspace / "frames/training-images",
            dataset / f"images_{factor}",
            directory=True,
        ),
    }
    output_root = context.output_root / "3dgs"
    training_config = {**config, "seed": int(context.config["seed"])}
    manifest = run_training(
        dataset,
        output_root,
        training_config,
        context.repository_root,
    )
    manifest["dataset_link_methods"] = link_methods
    diagnostics = output_root / "diagnostics"
    diagnostics.mkdir(parents=True, exist_ok=True)
    write_json(diagnostics / "training-summary.json", manifest)
    renders = output_root / "renders"
    try:
        renders.symlink_to(Path("model/renders"), target_is_directory=True)
        manifest["renders_link_method"] = "relative_symlink"
    except OSError:
        shutil.copytree(output_root / "model/renders", renders)
        manifest["renders_link_method"] = "copy"
    write_json(output_root / "training.json", manifest)
    return manifest


def _run_scene_export(context: PipelineContext) -> dict[str, Any]:
    scene = create_scene_export(
        context.workspace,
        context.state.payload["scene_id"],
        context.config["export"]["schema"],
        export_root=context.output_root / "export",
    )
    return {
        "scene": "export/scene.json",
        "cameras": "export/cameras.json",
        "camera_count": scene["camera_count"],
        "point_count": scene["point_cloud"]["shape"][0],
    }


def _run_report(context: PipelineContext) -> dict[str, Any]:
    export_validation = validate_scene_export(context.workspace / "export/scene.json")
    report = {
        "schema": "nht_reconstruction_report_v1",
        "scene_id": context.state.payload["scene_id"],
        "sfm": json.loads((context.workspace / "sfm/reconstruction.json").read_text()),
        "nht_training": json.loads(
            (context.workspace / "3dgs/training.json").read_text()
        ),
        "export_validation": export_validation,
    }
    write_json(context.output_root / "reconstruction-report.json", report)
    return export_validation


STAGE_EXECUTORS: dict[str, Callable[[PipelineContext], dict[str, Any]]] = {
    "frames": _run_frames,
    "preprocess": _run_preprocess,
    "sfm": _run_sfm,
    "sfm_selection": _run_sfm_selection,
    "nht_training": _run_nht_training,
    "scene_export": _run_scene_export,
    "reconstruction_report": _run_report,
}


def run_pipeline(
    context: PipelineContext, from_stage: str, through_stage: str | None = None
) -> None:
    invalidate_published_outputs(context.workspace, from_stage)
    cleanup_staging(context.workspace)
    stages = execution_order(from_stage)
    if through_stage is not None:
        if through_stage not in stages:
            raise ValueError(
                f"through_stage {through_stage} is before or unrelated to {from_stage}"
            )
        stages = stages[: stages.index(through_stage) + 1]
    for stage_name in stages:
        staging_root = stage_staging_root(context.workspace, stage_name)
        stage_context = replace(context, staging_root=staging_root)
        try:
            context.state.mark_running(
                stage_name, stage_config(context.config, stage_name)
            )
            attempt = context.state.payload["stages"][stage_name]["attempts"]
            log_path = (
                context.workspace / "logs" / stage_name / f"attempt-{attempt}.log"
            )
            with capture_stage_log(log_path):
                preflight_stage(
                    context.workspace,
                    stage_name,
                    context.config,
                    context.repository_root,
                )
                summary = STAGE_EXECUTORS[stage_name](stage_context)
                context.state.validate_outputs(stage_name, staging_root)
                if stage_name == "scene_export":
                    validate_scene_export(staging_root / "export/scene.json")
                publish_stage(context.workspace, staging_root, stage_name)
            summary["log"] = str(log_path.relative_to(context.workspace))
        except BaseException as error:
            remove_path(staging_root)
            cleanup_staging(context.workspace)
            context.state.mark_failed(stage_name, error)
            raise
        context.state.mark_completed(stage_name, summary)
    context.state.finish_request()
