"""Stage-aware dependency, device, and capacity diagnostics."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from .training import resolve_trainer


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _store_checks(workspace: Path, stage_name: str, checks: dict[str, Any]) -> None:
    path = workspace / "preflight.json"
    payload: dict[str, Any] = (
        json.loads(path.read_text())
        if path.exists()
        else {"schema": "nht_preflight_v1", "stages": {}}
    )
    payload["stages"][stage_name] = checks
    _atomic_json(path, payload)


def preflight_stage(
    workspace: Path,
    stage_name: str,
    config: dict[str, Any],
    repository_root: Path,
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    free_bytes = shutil.disk_usage(workspace).free
    required_bytes = int(float(config["operations"]["minimum_free_gb"]) * 1024**3)
    checks["disk"] = {
        "free_bytes": free_bytes,
        "required_bytes": required_bytes,
        "passed": free_bytes >= required_bytes,
    }
    if free_bytes < required_bytes:
        _store_checks(workspace, stage_name, checks)
        raise RuntimeError(
            f"Preflight disk space is below {config['operations']['minimum_free_gb']} GiB"
        )
    if stage_name == "frames":
        ffmpeg = shutil.which("ffmpeg")
        version = None
        if ffmpeg is not None:
            probe = subprocess.run(
                [ffmpeg, "-version"],
                check=False,
                capture_output=True,
                text=True,
            )
            version = probe.stdout.splitlines()[0] if probe.stdout else None
        checks["ffmpeg"] = {
            "path": ffmpeg,
            "version": version,
            "passed": ffmpeg is not None,
        }
        if ffmpeg is None:
            _store_checks(workspace, stage_name, checks)
            raise RuntimeError("Missing dependency: ffmpeg")
    if stage_name == "sfm":
        try:
            import pycolmap
        except ImportError as error:  # pragma: no cover - dependency environment
            checks["pycolmap"] = {"available": False, "passed": False}
            _store_checks(workspace, stage_name, checks)
            raise RuntimeError("Missing dependency: pycolmap") from error
        checks["pycolmap"] = {
            "available": True,
            "version": getattr(pycolmap, "__version__", None),
            "passed": True,
        }
        learned = any(
            candidate["backend"] == "hloc_aliked_lightglue"
            for candidate in config["sfm"]["candidates"]
        )
        if learned:
            hloc_available = importlib.util.find_spec("hloc") is not None
            checks["optional_learned_candidate"] = {
                "configured": True,
                "hloc_available": hloc_available,
                "passed": hloc_available,
                "interpretation": (
                    "primary may still complete; retry records missing_dependency if invoked"
                ),
            }
    if stage_name == "nht_training":
        training_config = {
            **config["nht_training"],
            "seed": int(config["seed"]),
        }
        adapter = (
            Path(training_config["adapter"]).resolve()
            if training_config.get("adapter")
            else repository_root / "nht_pipeline/nht_adapter.py"
        )
        try:
            python, trainer = resolve_trainer(training_config, repository_root)
        except (FileNotFoundError, ValueError) as error:
            checks["nht_runtime"] = {
                "passed": False,
                "error": str(error),
            }
            _store_checks(workspace, stage_name, checks)
            raise
        probe = subprocess.run(
            [
                str(python),
                str(adapter),
                "probe",
                "--trainer",
                str(trainer),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            checks["nht_runtime"] = {
                "python": str(python),
                "trainer": str(trainer),
                "adapter": str(adapter),
                "passed": False,
                "error": probe.stderr.strip(),
            }
            _store_checks(workspace, stage_name, checks)
            raise RuntimeError(
                "Missing dependency or invalid NHT runtime: " + probe.stderr.strip()
            )
        device = json.loads(probe.stdout.strip())
        configured_device = training_config["cuda_device"]
        available_devices = int(device["cuda_devices"])
        if not device["cuda_available"] or configured_device >= available_devices:
            checks["nht_runtime"] = {
                "python": str(python),
                "trainer": str(trainer),
                "adapter": str(adapter),
                **device,
                "configured_cuda_device": configured_device,
                "passed": False,
                "error": (
                    f"configured CUDA device {configured_device} is not present; "
                    f"detected {available_devices} device(s)"
                ),
            }
            _store_checks(workspace, stage_name, checks)
            raise RuntimeError(checks["nht_runtime"]["error"])
        selection_environment = os.environ.copy()
        selection_environment["CUDA_VISIBLE_DEVICES"] = str(configured_device)
        selected_probe = subprocess.run(
            [
                str(python),
                str(adapter),
                "probe",
                "--trainer",
                str(trainer),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=selection_environment,
        )
        selected_device = (
            json.loads(selected_probe.stdout.strip())
            if selected_probe.returncode == 0
            else {}
        )
        selected = bool(
            selected_probe.returncode == 0
            and selected_device.get("cuda_available")
            and selected_device.get("cuda_devices") == 1
        )
        checks["nht_runtime"] = {
            "python": str(python),
            "trainer": str(trainer),
            "adapter": str(adapter),
            **device,
            "configured_cuda_device": configured_device,
            "selected_device_probe": selected_device,
            "passed": selected,
        }
        if not checks["nht_runtime"]["passed"]:
            _store_checks(workspace, stage_name, checks)
            raise RuntimeError(
                f"NHT preflight could not select CUDA device {configured_device}: "
                f"{selected_probe.stderr.strip()}"
            )
    _store_checks(workspace, stage_name, checks)
    return checks
