# Pipeline contracts

## Command boundary

The only production command boundary is:

```text
python -m nht_pipeline --scene-id ID --input-video VIDEO --workspace WORKSPACE
python -m nht_pipeline --scene-id ID --workspace WORKSPACE --from-stage STAGE
```

The caller supplies the canonical workspace. The pipeline never searches another
scene or historical run. Downstream code consumes only `export/scene.json` and its
relative references; it does not import this package or inspect COLMAP/checkpoint
internals.

## Stage DAG and ownership

| Stage | Direct input | Owned output |
|---|---|---|
| `frames` | input video | `frames/raw`, `frames/extraction.json` |
| `preprocess` | extracted frames | `frames/images`, `frames/training-images`, `frames/frames.json` |
| `sfm` | accepted images | `sfm/candidates` |
| `sfm_selection` | candidate manifests/models | `sfm/model`, `sfm/reconstruction.json`, `sfm/diagnostics` |
| `nht_training` | selected SfM and images | `3dgs` |
| `scene_export` | SfM, frame metadata, NHT model | `export` |
| `reconstruction_report` | validated export | `reconstruction-report.json` |

`nht_pipeline.stages.STAGES` is the typed source of truth for dependencies,
required inputs, owned paths and fixed outputs. Executors are registered
individually in `nht_pipeline.pipeline.STAGE_EXECUTORS`.

Before a rerun, the requested stage and all DAG descendants are invalidated.
Each executor removes only its declared owned paths, writes the fixed outputs,
and is marked completed only after output and semantic checks pass. Missing
upstream outputs fail before execution. A failed stage may leave diagnostic
partial files, but `run.json` marks them failed and no downstream stage runs.

## Mutable run manifest

`run.json` uses schema `nht_pipeline_run_v1`. It records scene ownership, input,
resolved config path, requested range, timestamps, attempts, fixed outputs,
summaries and structured errors. Writes use atomic replace. Status values are:

```text
pending running completed failed invalidated skipped
```

The JSON shape is documented by [`schemas/run.schema.json`](../schemas/run.schema.json).

## Standard scene export

`export/` is self-contained and fixed:

```text
scene.json  cameras.json  points_scene.npy  images/  model/
```

Camera transforms are `camera_to_scene` homogeneous matrices. Camera coordinates
follow COLMAP (`x` right, `y` down, `z` forward); scene coordinates are the
selected right-handed COLMAP world coordinates. Intrinsics and image dimensions
refer to full-resolution exported images. `points_scene.npy` is finite float32
`N×6` data with XYZ and RGB normalized to `[0, 1]`.

The validator checks schema fields, conventions, shapes, dtype, finite values,
color range, unique IDs and image paths, positive focal length, homogeneous
proper orthonormal rotations, real image resolution, and an NHT checkpoint.
It intentionally does not check hashes, file-size identity or Git state.

See [`schemas/scene.schema.json`](../schemas/scene.schema.json) and
[`schemas/cameras.schema.json`](../schemas/cameras.schema.json).

## Downstream integration status

This repository now publishes the independent command/file boundary. The
consumer migration in `Motoki0705/tennis-lab#695` remains external and open: its
current provider path still reads COLMAP binaries and immutable hash manifests.
Until that issue replaces the old provider and completes alignment/dataset
generation from this `export/`, the cross-repository acceptance item is pending.
