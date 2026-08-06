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
一致しないことを理由に拒否しません。pipeline adapter が top-level `seed` を
trainer の実際の乱数初期化へ注入し、requested/effective seed の不一致や、trainer
が seed hook を呼ばない実行を拒否します。

Production renderer 契約は `camera_model: pinhole`、`pose_opt: false`、
`post_processing: null` に限定されます。`near_plane` と `far_plane` は設定から
trainer、export runtime、`nht-render` の rasterizer まで同じ値を引き渡します。
未知の設定キー、型違い、未対応の trainer option は学習開始前に fail-closed で
拒否されます。`extra_args` は運用上必要な `--disable_video` と
`--num_workers N` だけを受理します。

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

各 stage は `.staging/<stage>` に生成し、fixed output と semantic validationを
通過した後だけ canonical path へ rename します。再実行要求時には対象 stage と
descendant の旧成果物を先に物理削除し、workspace lock は同時実行を拒否します。
consumer は `run.json` の該当 stage が `completed` のときだけ公開 path を読みます。

## SfM policy

Production の一次系は 1 fps sampling、品質 filter、SIFT、temporal overlap 10、
quadratic overlap、COLMAP incremental mapper、`OPENCV` camera model です。
semantic gate に落ちた場合だけ ALIKED-N16 + LightGlue + temporal/NetVLAD pair
graph を1回実行します。全候補が不合格なら selection で失敗し、NHT/export
には進みません。

高速移動により隣接frameの重なりが不足した場合は、scene固有の値を埋め込まず、
共通の [`configs/fast-motion-retry.yaml`](configs/fast-motion-retry.yaml) を明示的に
選びます。このprofileは5 fps、overlap 15へ増やし、同じcamera modelとquality
gateを維持します。両candidate比較用は
[`configs/fast-motion-benchmark.yaml`](configs/fast-motion-benchmark.yaml) です。
固定、区間共有、画像別intrinsicsの研究比較は
[`configs/camera-policy-benchmark.yaml`](configs/camera-policy-benchmark.yaml) で
明示的に実行します。これらの自由度はproductionの暗黙fallbackではありません。
合格した各candidateを同じ短縮NHT recipeへ接続する場合は
`scripts/run_candidate_nht_benchmark.py`を使用します。

この production policy は誤った再構成を後段へ渡さない high-precision / fail-safe
設計です。そのため、実選手が大きく動く映像、frame 間 overlap が少ない映像、
撮影中に optical zoom が変化する映像では、無理に scene を生成せず不合格にする
場合があり、recall は意図的に低くなります。

## Standard renderer

downstream consumer はNHT checkpoint内部を読まず、`export/scene.json`だけを入口に
独立processでRGB・alpha・depthを描画できます。

```bash
# export済みobserved camera
nht-render --scene WORKSPACE/export/scene.json \
  --camera-id frame_000000 --output artifacts/render-observed

# nht_render_request_v1 の任意camera
nht-render --scene WORKSPACE/export/scene.json \
  --cameras camera-request.json --output artifacts/render-arbitrary
```

出力はcameraごとのfloat32 `rgb.npy`、`alpha.npy`、`depth.npy`とpreview、および
`nht_render_result_v1`の`render.json`です。request/result schemaは
[`schemas/render-request.schema.json`](schemas/render-request.schema.json) と
[`schemas/render-result.schema.json`](schemas/render-result.schema.json) に固定しています。

選定根拠と実測値は [SfM research](docs/sfm-research.md)、契約は
[Pipeline contracts](docs/contracts.md)、破壊的移行内容は
[Migration](docs/migration.md) を参照してください。

## Verification

```bash
uv run --extra test pytest -q
uv run --extra test ruff check nht_pipeline tests scripts
uv run --extra test mypy nht_pipeline
uvx check-jsonschema --check-metaschema schemas/*.schema.json
uv build
```

CPU smoke は `tests/test_frames.py`、`tests/test_sfm_helpers.py`、
`tests/test_run_state.py`、`tests/test_export.py` が、frame quality、DAG、selection、
scene semantic validation を外部 GPU なしで検証します。実動画 integration の
結果は [2026-08-05 tennis-court experiment](research/experiments/2026-08-05-tennis-court.md)
に、派生 stress test は
[2026-08-06 SfM stress benchmark](research/experiments/2026-08-06-sfm-stress.md)
および
[independent-scene acceptance](research/experiments/2026-08-06-independent-scenes.md)
に記録しています。

実NHT/CUDA round-tripはmanual GPU workflowまたは次の固定commandで再実行します。
入力workspaceはcompleted SfM、NHT runtimeはCUDAと参照trainerを含む必要があります。

```bash
NHT_GPU_PYTHON=/path/to/nht/.venv/bin/python \
NHT_GPU_TRAINER=/path/to/gsplat/examples/simple_trainer_nht.py \
NHT_GPU_SOURCE_WORKSPACE=/path/to/completed/reconstruction \
NHT_GPU_REPORT=research/evidence/nht-gpu-round-trip.json \
/path/to/nht/.venv/bin/python -m pytest -q -m gpu \
  tests/test_nht_gpu_integration.py
```

合格条件は実trainer 1 step、非identity transform、export、新processからのobserved/
arbitrary RGB・alpha・depth、trainer validation renderとのMAE ≤ 0.01かつPSNR ≥ 40 dBです。
