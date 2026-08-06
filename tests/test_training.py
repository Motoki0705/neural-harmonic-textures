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
    trainer.write_text(
        """\
import json
import sys
from pathlib import Path

result_dir = Path(sys.argv[sys.argv.index("--result_dir") + 1])
(result_dir / "ckpts").mkdir(parents=True)
(result_dir / "stats").mkdir()
(result_dir / "ckpts/ckpt_999_rank0.pt").write_bytes(b"old")
(result_dir / "ckpts/ckpt_29999_rank0.pt").write_bytes(b"final")
(result_dir / "stats/val_step29999.json").write_text(json.dumps({"psnr": 30.0}))
(result_dir / "stats/train_step29999_rank0.json").write_text(json.dumps({"loss": 0.1}))
"""
    )
    output_root = tmp_path / "output"
    manifest = run_training(
        tmp_path / "dataset",
        output_root,
        {
            "python": sys.executable,
            "trainer": str(trainer),
            "seed": 42,
            "seed_argument": None,
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
