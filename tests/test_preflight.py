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


@pytest.mark.parametrize(
    "inherited_mask,configured_device,selected_token",
    [(None, 1, "1"), ("3", 0, "3"), ("2,5", 1, "5")],
)
def test_nht_preflight_selects_the_configured_logical_cuda_index(
    tmp_path, monkeypatch, inherited_mask, configured_device, selected_token
) -> None:
    config = load_config(None)
    config["operations"]["minimum_free_gb"] = 0
    config["nht_training"]["cuda_device"] = configured_device
    if inherited_mask is None:
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    else:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", inherited_mask)
    python, trainer = _runtime(tmp_path)
    monkeypatch.setattr(
        "nht_pipeline.preflight.resolve_trainer", lambda *_args: (python, trainer)
    )
    environments = []
    initial_device_count = (
        len(inherited_mask.split(",")) if inherited_mask is not None else 2
    )

    def fake_run(_command, **kwargs):
        environment = kwargs.get("env")
        payload = (
            {
                "cuda_available": True,
                "cuda_devices": initial_device_count,
                "cuda_device_names": ["visible"] * initial_device_count,
            }
            if not environments
            else {"cuda_available": True, "cuda_devices": 1, "cuda_device_names": ["b"]}
        )
        environments.append(environment)
        return SimpleNamespace(returncode=0, stdout=json.dumps(payload), stderr="")

    monkeypatch.setattr("nht_pipeline.preflight.subprocess.run", fake_run)

    checks = preflight_stage(tmp_path, "nht_training", config, tmp_path)

    assert environments[0].get("CUDA_VISIBLE_DEVICES") == inherited_mask
    assert environments[1]["CUDA_VISIBLE_DEVICES"] == selected_token
    assert checks["nht_runtime"]["configured_cuda_device"] == configured_device
    assert checks["nht_runtime"]["selected_cuda_token"] == selected_token
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
