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

Before a rerun, the effective stage and all DAG descendants are marked
`invalidated` and their declared published paths are physically removed. Each
executor writes only below `.staging/<stage>`, validates fixed outputs and semantic
invariants there, then renames owned paths to their canonical locations. The stage
is marked `completed` only after publication. A consumer must require `completed`;
an output briefly present while the stage record is still `running` is not public.
On failure, temporary outputs are deleted and canonical descendants stay absent.

The workspace lock is acquired before manifest/config inspection. A live owner is
rejected. A dead owner plus `running` record is recovered as an explicit
`process_interrupted` failure, stale staging is cleaned, and rerun publication
follows the same transaction.

Config ownership is defined by the DAG: `frames`, `preprocess`, `seed/sfm`,
`seed/nht_training`, and `export`. A structural comparison against the previous
resolved config expands a too-late request to the earliest owning stage. A changed
input video always expands to `frames`. Every attempt stores the exact owned config
subset used by that stage.

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
follow COLMAP (`x` right, `y` down, `z` forward). Canonical scene space is the
right-handed normalized world in which the exported NHT checkpoint Gaussian means
actually live. `scene_from_sfm` is the effective parser transform, including
camera normalization, principal-axis alignment and any upside-down correction;
`sfm_from_scene` is its inverse. `cameras.json`, `points_scene.npy`, and the NHT
model all use this same space.

Consumer cameras are the parser's effective undistorted `PINHOLE` cameras after
crop/downsampling. Their matrices, crop, source resolution and exported image
resolution are authoritative. Raw COLMAP camera IDs and transforms exist only
under each camera's `diagnostics` field. `points_scene.npy` is finite float32
`N×6` data with canonical XYZ and RGB normalized to `[0, 1]`.

The validator checks schema fields, conventions, shapes, dtype, finite values,
color range, unique IDs and image paths, positive focal length, homogeneous
proper orthonormal rotations, real image resolution, and an NHT checkpoint.
It intentionally does not check hashes, file-size identity or Git state.

See [`schemas/scene.schema.json`](../schemas/scene.schema.json) and
[`schemas/cameras.schema.json`](../schemas/cameras.schema.json).

## Rendering boundary

`nht-render --scene export/scene.json` is the stable subprocess boundary. The
caller may select exported observed camera IDs or supply an
`nht_render_request_v1` file containing arbitrary `camera_to_scene`, PINHOLE
intrinsics and resolution. The renderer resolves runtime/checkpoint paths only
through `scene.json`, starts with a clean staging output, and publishes
`nht_render_result_v1` only after every RGB/alpha/depth array is complete. NHT
module names, checkpoint keys and internal directories are not part of the caller
contract.

## Downstream integration status

Cross-repository status and replay evidence are recorded in the production
acceptance report rather than changing this file contract.
