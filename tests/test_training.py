from __future__ import annotations

import json
import sys

from nht_pipeline.training import resolve_trainer, run_training


def test_resolve_trainer_preserves_virtualenv_interpreter_symlink(tmp_path) -> None:
    interpreter = tmp_path / "venv/bin/python"
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    trainer = tmp_path / "trainer.py"
    trainer.write_text("# trainer\n")

    resolved_python, resolved_trainer = resolve_trainer(
        {"python": str(interpreter), "trainer": str(trainer)}, tmp_path
    )

    assert resolved_python == interpreter.absolute()
    assert resolved_python.is_symlink()
    assert resolved_trainer == trainer.resolve()


def test_training_selects_exact_final_checkpoint_and_stores_relative_paths(
    tmp_path,
) -> None:
    trainer = tmp_path / "fake_trainer.py"
    trainer.write_text("# adapter owns this fake trainer invocation\n")
    adapter = tmp_path / "fake_adapter.py"
    adapter.write_text(
        """\
import argparse
import json
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("command")
parser.add_argument("--trainer")
parser.add_argument("--seed", type=int)
parser.add_argument("--seed-output", type=Path)
parser.add_argument("--metadata-output", type=Path)
parser.add_argument("--observed-image-root", type=Path)
parser.add_argument("--runtime-output", type=Path)
args, remaining = parser.parse_known_args()
result_dir = Path(remaining[remaining.index("--result_dir") + 1])
(result_dir / "ckpts").mkdir(parents=True)
(result_dir / "stats").mkdir()
(args.observed_image_root).mkdir(parents=True)
(result_dir / "ckpts/ckpt_999_rank0.pt").write_bytes(b"old")
(result_dir / "ckpts/ckpt_29999_rank0.pt").write_bytes(b"final")
(result_dir / "stats/val_step29999.json").write_text(json.dumps({"psnr": 30.0}))
(result_dir / "stats/train_step29999_rank0.json").write_text(json.dumps({"loss": 0.1}))
args.seed_output.write_text(json.dumps({"effective_seed": args.seed}))
args.metadata_output.write_text(json.dumps({"schema": "nht_training_scene_v1"}))
args.runtime_output.write_text(json.dumps({"schema": "nht_runtime_config_v1"}))
"""
    )
    output_root = tmp_path / "output"
    manifest = run_training(
        tmp_path / "dataset",
        output_root,
        {
            "python": sys.executable,
            "trainer": str(trainer),
            "adapter": str(adapter),
            "seed": 42,
            "data_factor": 2,
            "max_steps": 30_000,
            "cap_max": 1_000_000,
            "test_every": 8,
            "lpips_net": "alex",
            "extra_args": [],
        },
        tmp_path,
    )

    assert manifest["checkpoint"] == "model/ckpts/ckpt_29999_rank0.pt"
    assert manifest["validation_metrics"][0]["path"].startswith("model/stats/")
    assert manifest["training_metrics"][0]["path"].startswith("model/stats/")
    persisted = json.loads((output_root / "training.json").read_text())
    assert persisted["status"] == "completed"
