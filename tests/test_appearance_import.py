from __future__ import annotations

import copy
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from nht_pipeline.config import (
    NhtTrainingConfig,
    earliest_affected_stage,
    load_config,
    write_resolved_config,
)
from nht_pipeline.import_workspace import (
    IMPORTED_STAGES,
    import_workspace,
    validate_import_request,
)
from nht_pipeline.run_state import RunState


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    config = load_config(None)
    config["nht_training"]["image_names"] = [
        "frame_000000.jpg",
        "frame_000001.jpg",
        "frame_000008.jpg",
        "frame_000009.jpg",
    ]
    source_config = copy.deepcopy(config)
    source_config["nht_training"]["image_names"] = None
    write_resolved_config(source / "resolved-config.yaml", source_config)
    state = RunState.create_or_load(source, "original", None)
    for relative in ("frames/images", "sfm/model", "sfm/candidates"):
        (source / relative).mkdir(parents=True)
    records = [
        {
            "filename": f"frame_{i:06d}.jpg",
            "source_frame_index": i,
            "source_time_seconds": float(i),
            "accepted": True,
            "brightness_mean": 99.0,
        }
        for i in range(10)
    ]
    (source / "frames/frames.json").write_text(json.dumps({"frames": records}))
    for relative in (
        "frames/extraction.json",
        "sfm/reconstruction.json",
        "sfm/candidates/candidates.json",
    ):
        (source / relative).write_text("{}")
    for name in ("cameras", "images", "points3D", "rigs", "frames"):
        (source / "sfm/model" / f"{name}.bin").write_bytes(name.encode())
    for stage in IMPORTED_STAGES:
        state.mark_completed(stage, {})
    images = {
        i: SimpleNamespace(name=row["filename"], camera_id=1)
        for i, row in enumerate(records)
    }
    model = SimpleNamespace(
        images=images, cameras={1: SimpleNamespace(width=32, height=18)}
    )
    monkeypatch.setitem(
        sys.modules, "pycolmap", SimpleNamespace(Reconstruction=lambda path: model)
    )
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    for name in config["nht_training"]["image_names"]:
        Image.new("RGB", (32, 18), (140, 70, 20)).save(replacement / name)
    return source, replacement, tmp_path / "destination", config


def test_import_copies_complete_sfm_and_only_selected_rgb(inputs):
    source, replacement, target, config = inputs
    before = {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    state = import_workspace(source, replacement, target, "clay", config)
    assert all(
        state.payload["stages"][stage]["status"] == "completed"
        for stage in IMPORTED_STAGES
    )
    assert state.payload["stages"]["nht_training"]["status"] == "pending"
    assert {p.name for p in (target / "sfm/model").iterdir()} == {
        "cameras.bin",
        "images.bin",
        "points3D.bin",
        "rigs.bin",
        "frames.bin",
    }
    assert (
        sorted(p.name for p in (target / "frames/images").iterdir())
        == config["nht_training"]["image_names"]
    )
    assert (target / "sfm/model/cameras.bin").stat().st_ino != (
        source / "sfm/model/cameras.bin"
    ).stat().st_ino
    with Image.open(target / "frames/training-images/frame_000000.png") as image:
        assert image.size == (16, 9)
    metadata = json.loads((target / "frames/frames.json").read_text())
    assert "brightness_mean" not in metadata["frames"][0]
    assert metadata["frames"][0]["source_time_seconds"] == 0.0
    provenance = json.loads((target / "import-provenance.json").read_text())
    assert provenance["validation_names"] == ["frame_000000.jpg", "frame_000008.jpg"]
    assert provenance["full_image_count"] == 10
    assert before == {
        str(p.relative_to(source)): p.read_bytes()
        for p in source.rglob("*")
        if p.is_file()
    }
    validate_import_request(target, config, "nht_training")
    with pytest.raises(ValueError, match="upstream"):
        validate_import_request(target, config, "sfm")
    (target / "frames/images/frame_000000.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Imported input changed"):
        validate_import_request(target, config, "nht_training")


def test_invalid_replacement_rejected_before_publishing(inputs):
    source, replacement, target, config = inputs
    Image.new("RGB", (31, 18)).save(replacement / "frame_000000.jpg")
    with pytest.raises(ValueError, match="calibrated source size"):
        import_workspace(source, replacement, target, "clay", config)
    assert not target.exists()


def test_nonempty_destination_and_upstream_config_change_rejected(inputs):
    source, replacement, target, config = inputs
    target.mkdir()
    (target / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="empty"):
        import_workspace(source, replacement, target, "clay", config)
    assert (target / "keep.txt").read_text() == "keep"
    config["seed"] += 1
    with pytest.raises(ValueError, match="source seed"):
        import_workspace(source, replacement, target.parent / "new", "clay", config)


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["../bad.jpg"],
        ["/bad.jpg"],
        ["a.jpg", "a.jpg"],
        "a.jpg",
        ["a//b.jpg"],
        ["a/./b.jpg"],
    ],
)
def test_image_selection_is_fail_closed(names):
    mapping = load_config(None)["nht_training"]
    mapping["image_names"] = names
    with pytest.raises((ValueError, TypeError)):
        NhtTrainingConfig.from_mapping(mapping)


def test_selection_invalidates_training_only():
    before = load_config(None)
    after = copy.deepcopy(before)
    after["nht_training"]["image_names"] = ["b.jpg", "a.jpg"]
    typed = NhtTrainingConfig.from_mapping(after["nht_training"])
    assert typed.to_mapping()["image_names"] == ["a.jpg", "b.jpg"]
    assert earliest_affected_stage(before, after) == "nht_training"


def test_prepare_only_cli_does_not_launch_training(inputs, monkeypatch):
    from nht_pipeline import __main__ as cli

    source, replacement, target, config = inputs
    path = target.parent / "config.yaml"
    write_resolved_config(path, config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "nht-reconstruct",
            "--scene-id",
            "clay",
            "--workspace",
            str(target),
            "--config",
            str(path),
            "--source-workspace",
            str(source),
            "--replacement-images",
            str(replacement),
            "--prepare-only",
            "--from-stage",
            "nht_training",
        ],
    )
    monkeypatch.setattr(
        cli, "run_pipeline", lambda *args: pytest.fail("prepare-only launched training")
    )
    cli.main()
    assert (target / "import-provenance.json").exists()
