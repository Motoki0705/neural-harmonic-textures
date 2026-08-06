# Destructive migration from immutable runs

Issue #3 intentionally removes backward compatibility with the old isolated,
commit-pinned launcher.

| Removed path | Removed behavior |
|---|---|
| `train.py`, root `train.sh` | non-empty output refusal, commit and clean-tree gates, old manifest |
| `pins.env`, `lib.sh` | NHT/gsplat/GLM commit equality and pinned CUDA enforcement |
| `setup.sh`, `setup.ps1`, `requirements.in` | old runtime setup entrypoint and pin-oriented environment recipe |
| `smoke.py` | commit and CUDA binary SHA-256 reporting as the old setup gate |

No compatibility wrapper reads the old `nht-run.json` or immutable run layout.
The replacement is `python -m nht_pipeline`, a canonical mutable workspace,
`run.json`, and semantic scene validation. Runtime versions may be recorded as
diagnostics in the future, but are never execution or validity conditions.
