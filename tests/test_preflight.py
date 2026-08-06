from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from nht_pipeline.config import load_config
from nht_pipeline.preflight import preflight_stage


def _runtime(tmp_path):
    python = tmp_path / "python"
    trainer = tmp_path / "trainer.py"
    python.write_text("")
    trainer.write_text("")
    return python, trainer


def test_nht_preflight_selects_the_configured_cuda_index(tmp_path, monkeypatch) -> None:
    config = load_config(None)
    config["operations"]["minimum_free_gb"] = 0
    config["nht_training"]["cuda_device"] = 1
    python, trainer = _runtime(tmp_path)
    monkeypatch.setattr(
        "nht_pipeline.preflight.resolve_trainer", lambda *_args: (python, trainer)
    )
    environments = []

    def fake_run(_command, **kwargs):
        environment = kwargs.get("env")
        environments.append(environment)
        payload = (
            {"cuda_available": True, "cuda_devices": 2, "cuda_device_names": ["a", "b"]}
            if environment is None
            else {"cuda_available": True, "cuda_devices": 1, "cuda_device_names": ["b"]}
        )
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("nht_pipeline.preflight.subprocess.run", fake_run)

    checks = preflight_stage(tmp_path, "nht_training", config, tmp_path)

    assert environments[1]["CUDA_VISIBLE_DEVICES"] == "1"
    assert checks["nht_runtime"]["configured_cuda_device"] == 1
    assert checks["nht_runtime"]["passed"] is True


def test_nht_preflight_rejects_a_missing_cuda_index(tmp_path, monkeypatch) -> None:
    config = load_config(None)
    config["operations"]["minimum_free_gb"] = 0
    config["nht_training"]["cuda_device"] = 2
    python, trainer = _runtime(tmp_path)
    monkeypatch.setattr(
        "nht_pipeline.preflight.resolve_trainer", lambda *_args: (python, trainer)
    )
    monkeypatch.setattr(
        "nht_pipeline.preflight.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0,
            stdout=json.dumps({"cuda_available": True, "cuda_devices": 2}),
            stderr="",
        ),
    )

    with pytest.raises(RuntimeError, match="device 2 is not present"):
        preflight_stage(tmp_path, "nht_training", config, tmp_path)


def test_nht_preflight_rejects_an_unselectable_cuda_index(
    tmp_path, monkeypatch
) -> None:
    config = load_config(None)
    config["operations"]["minimum_free_gb"] = 0
    python, trainer = _runtime(tmp_path)
    monkeypatch.setattr(
        "nht_pipeline.preflight.resolve_trainer", lambda *_args: (python, trainer)
    )
    responses = iter(
        [
            SimpleNamespace(
                returncode=0,
                stdout=json.dumps({"cuda_available": True, "cuda_devices": 1}),
                stderr="",
            ),
            SimpleNamespace(returncode=1, stdout="", stderr="device selection failed"),
        ]
    )
    monkeypatch.setattr(
        "nht_pipeline.preflight.subprocess.run",
        lambda *_args, **_kwargs: next(responses),
    )

    with pytest.raises(RuntimeError, match="could not select CUDA device 0"):
        preflight_stage(tmp_path, "nht_training", config, tmp_path)
