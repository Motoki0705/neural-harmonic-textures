# Video-to-NHT reconstruction pipeline

この repository の production entrypoint は、1本の動画、`scene_id`、1つの
mutable workspace から frame 抽出、堅牢な SfM、NHT 学習、標準 scene export
までを実行します。

```text
video → frames → preprocess → SfM candidates → selection
      → NHT training → standard scene export → report
```

Git commit、clean worktree、SHA-256、artifact fingerprint は実行条件では
ありません。同じ scene を再実行すると、指定 stage とその descendants を
同じ固定 path で置換します。

## Quick start

Python 3.11+、`ffmpeg`、COLMAP 対応の `pycolmap` が必要です。

```bash
uv sync --extra test

uv run python -m nht_pipeline \
  --scene-id tennis-court \
  --input-video data/tennis_court.mp4 \
  --workspace artifacts/tennis-court/reconstruction \
  --config configs/production.yaml
```

インストール後は同じ CLI を `nht-reconstruct` でも呼べます。

ALIKED/LightGlue retry を実行可能にする場合は、同じ環境へ公式 HLOC と
LightGlue を追加します。初回実行時には learned model の重みも取得されます。

```bash
uv sync --extra test --extra sfm-learned
```

既存の外部 checkout/runtime を使う場合は candidate config の `hloc_root`、
`lightglue_root`、`site_packages` でも import path を指定できます。これらは
install 方法であり、commit や hash を validity gate にはしません。

NHT 学習には、この repository の `gsplat` checkout または別の NHT runtime
が必要です。`nht_training.python` と `nht_training.trainer` に実行可能な Python
と `simple_trainer_nht.py` を設定します。commit や dirty state は任意であり、
一致しないことを理由に拒否しません。参照 trainer は seed 42 を内部設定するため、
別の seed 引数を持つ trainer では `seed_argument` も設定してください。

## Partial rerun

```bash
# SfM 以下を固定 path で置換
uv run python -m nht_pipeline \
  --scene-id tennis-court \
  --workspace artifacts/tennis-court/reconstruction \
  --from-stage sfm

# NHT 学習以下だけを置換
uv run python -m nht_pipeline \
  --scene-id tennis-court \
  --workspace artifacts/tennis-court/reconstruction \
  --from-stage nht_training
```

研究用の `--through-stage` は指定 stage で停止します。この場合 `run.json` の
top-level status は `pending` となり、未実行 descendants を completed と扱いません。

## Canonical workspace

```text
<workspace>/
├── run.json
├── resolved-config.yaml
├── frames/
├── sfm/
│   ├── candidates/
│   ├── diagnostics/
│   ├── model/
│   └── reconstruction.json
├── 3dgs/
│   ├── diagnostics/
│   ├── model/
│   ├── renders/
│   └── training.json
├── export/
│   ├── scene.json
│   ├── cameras.json
│   ├── points_scene.npy
│   ├── images/
│   └── model/
└── reconstruction-report.json
```

`run.json` は現在状態の正本です。各 stage は `pending`、`running`、
`completed`、`failed`、`invalidated`、`skipped` のいずれかを取り、失敗した
partial output は completed として公開されません。

## SfM policy

Production の一次系は 1 fps sampling、品質 filter、SIFT、temporal overlap 10、
quadratic overlap、COLMAP incremental mapper、`OPENCV` camera model です。
semantic gate に落ちた場合だけ ALIKED-N16 + LightGlue + temporal/NetVLAD pair
graph を1回実行します。全候補が不合格なら selection で失敗し、NHT/export
には進みません。

選定根拠と実測値は [SfM research](docs/sfm-research.md)、契約は
[Pipeline contracts](docs/contracts.md)、破壊的移行内容は
[Migration](docs/migration.md) を参照してください。

## Verification

```bash
uv run --extra test pytest -q
uv run --extra test ruff check nht_pipeline tests
uv run --extra test mypy nht_pipeline
```

CPU smoke は `tests/test_frames.py`、`tests/test_sfm_helpers.py`、
`tests/test_run_state.py`、`tests/test_export.py` が、frame quality、DAG、selection、
scene semantic validation を外部 GPU なしで検証します。実動画 integration の
結果は [2026-08-05 tennis-court experiment](research/experiments/2026-08-05-tennis-court.md)
に、派生 stress test は
[2026-08-06 SfM stress benchmark](research/experiments/2026-08-06-sfm-stress.md)
に記録しています。
