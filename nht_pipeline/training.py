"""Mutable NHT training stage without commit, hash, or clean-tree gates."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_all_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_finite(item) for item in value)
    return True


def resolve_trainer(config: dict[str, Any], repository_root: Path) -> tuple[Path, Path]:
    # Keep a virtual environment's interpreter symlink intact: resolving it would
    # launch the base interpreter without that environment's site-packages.
    python = (
        Path(config["python"]).expanduser().absolute()
        if config.get("python")
        else Path(sys.executable).absolute()
    )
    if config.get("trainer"):
        trainer = Path(config["trainer"]).resolve()
    else:
        candidates = [
            repository_root / "gsplat/examples/simple_trainer_nht.py",
            repository_root / "upstream/gsplat/examples/simple_trainer_nht.py",
        ]
        trainer = next((path for path in candidates if path.is_file()), candidates[0])
    if not python.is_file():
        raise FileNotFoundError(f"NHT Python interpreter not found: {python}")
    if not trainer.is_file():
        raise FileNotFoundError(
            "NHT trainer not found; initialize gsplat or set nht_training.trainer"
        )
    return python, trainer


def run_training(
    dataset_dir: Path,
    output_root: Path,
    config: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    python, trainer = resolve_trainer(config, repository_root)
    factor = int(config["data_factor"])
    max_steps = int(config["max_steps"])
    result_dir = output_root / "model"
    result_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(python),
        str(trainer),
        "default",
        "--data_dir",
        str(dataset_dir),
        "--data_factor",
        str(factor),
        "--native_images_factor",
        "--result_dir",
        str(result_dir),
    ]
    if config.get("seed_argument"):
        command.extend([str(config["seed_argument"]), str(config["seed"])])
    command.extend(
        [
            "--test_every",
            str(config["test_every"]),
            "--max_steps",
            str(max_steps),
            "--eval_steps",
            str(max_steps),
            "--save_steps",
            str(max_steps),
            "--save_ply",
            "--ply_steps",
            str(max_steps),
            "--render_traj_path",
            "interp",
            "--lpips_net",
            str(config["lpips_net"]),
            "--use_color_correction_metric",
            "--strategy.cap-max",
            str(config["cap_max"]),
            "--disable_viewer",
            *[str(argument) for argument in config.get("extra_args", [])],
        ]
    )
    manifest: dict[str, Any] = {
        "schema": "nht_training_v1",
        "status": "running",
        "started_at_utc": _now(),
        "finished_at_utc": None,
        "command": command,
        "seed": config["seed"],
        "seed_control": (
            "command_argument" if config.get("seed_argument") else "trainer_internal"
        ),
        "max_steps": max_steps,
        "cap_max": config["cap_max"],
        "data_factor": factor,
        "checkpoint": None,
        "validation_metrics": [],
        "returncode": None,
        "elapsed_seconds": None,
        "training_metrics": [],
    }
    manifest_path = output_root / "training.json"
    _write_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment.setdefault("OMP_NUM_THREADS", "4")
    environment.setdefault("OPENBLAS_NUM_THREADS", "4")
    started_monotonic = time.monotonic()
    completed = subprocess.run(
        command, cwd=trainer.parent, env=environment, check=False
    )
    manifest["elapsed_seconds"] = time.monotonic() - started_monotonic
    manifest["returncode"] = completed.returncode
    manifest["finished_at_utc"] = _now()
    if completed.returncode != 0:
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise RuntimeError(f"NHT trainer exited with code {completed.returncode}")

    checkpoints = sorted((result_dir / "ckpts").glob("*.pt"))
    if not checkpoints:
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise RuntimeError("NHT training completed without a checkpoint")
    expected_step = max_steps - 1
    final_checkpoints = [
        path for path in checkpoints if f"ckpt_{expected_step}_" in path.name
    ]
    if not final_checkpoints:
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise RuntimeError(
            f"NHT training has no checkpoint for final step {expected_step}: "
            f"{[path.name for path in checkpoints]}"
        )
    final_checkpoint = final_checkpoints[-1]
    metric_paths = sorted((result_dir / "stats").glob("val_step*.json"))
    validation_metrics = []
    for path in metric_paths:
        if "per_image" in path.name:
            continue
        payload = json.loads(path.read_text())
        if not _all_finite(payload):
            manifest["status"] = "failed"
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"Non-finite validation metrics in {path}")
        validation_metrics.append(
            {"path": str(path.relative_to(output_root)), "metrics": payload}
        )
    if not validation_metrics:
        manifest["status"] = "failed"
        _write_json(manifest_path, manifest)
        raise RuntimeError("NHT training completed without validation metrics")
    training_metrics = []
    for path in sorted((result_dir / "stats").glob("train_step*.json")):
        payload = json.loads(path.read_text())
        if not _all_finite(payload):
            manifest["status"] = "failed"
            _write_json(manifest_path, manifest)
            raise RuntimeError(f"Non-finite training metrics in {path}")
        training_metrics.append(
            {"path": str(path.relative_to(output_root)), "metrics": payload}
        )
    manifest.update(
        {
            "status": "completed",
            "checkpoint": str(final_checkpoint.relative_to(output_root)),
            "validation_metrics": validation_metrics,
            "training_metrics": training_metrics,
        }
    )
    _write_json(manifest_path, manifest)
    return manifest
