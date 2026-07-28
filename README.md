# Isolated Neural Harmonic Textures runtime

This tracked directory owns the reproducible NHT checkout, CUDA environment, smoke
gate, and training launcher used by the 3DGS-native synthetic-data pipeline.
It does not reuse tennis-lab's `.venv` and does not copy the already-built
environment from `/home/kamimura/projects/gaussian-splating`.

The integration is pinned to:

- Neural Harmonic Textures `7de4cc07ba7f81ce90f7bd90f76ff0260c00c3d0`
- NHT's gsplat branch `20bc323d613258e5d169fdbc962c9ef27d55ca69`
- GLM `e7970a8b26732f1b0df9690f7180546f8c30e48e`
- Python 3.11, Torch 2.9.1 + CUDA 13.0
- Blackwell `sm_120+PTX`

All additional Git dependencies are content-pinned in `requirements.in`.
`pins.env` is the single declaration of checkout and runtime versions.

## Setup

The first build may take more than one scheduled research cycle because gsplat
and tiny-cuda-nn compile CUDA extensions.

```bash
third_party/nht/setup.sh
```

To seed Git objects from the completed reference checkout without sharing its
working tree or environment:

```bash
NHT_SEED_REPOSITORY=/home/kamimura/projects/gaussian-splating/third_party/neural-harmonic-textures \
  third_party/nht/setup.sh
```

The clone uses `--dissociate`, so the resulting checkout is independent. A
successful setup publishes the finite forward/backward report at
`third_party/nht/artifacts/smoke.json` and a resolved package inventory beside
it. These local build artifacts are intentionally ignored by Git.

## Train

The input is a COLMAP dataset containing `sparse/0` and a native
`images_<factor>` directory. The output directory must be absent or empty.

```bash
third_party/nht/train.sh \
  --data-dir /absolute/path/to/colmap-scene \
  --result-dir /absolute/path/to/results \
  --data-factor 2 \
  --max-steps 30000 \
  --cap-max 1000000
```

The launcher refuses a modified or unpinned checkout, records the exact command
and environment in `nht-run.json`, and never overwrites a non-empty run.
Additional upstream flags must be passed explicitly as repeatable
`--trainer-arg=<value>` arguments.

This runtime is a third-party boundary. Tennis-lab modules must communicate
with it through versioned files or subprocess requests and must not import its
packages in the main `.venv`.

## Tennis-lab integration boundary

This repository intentionally owns only the generic NHT runtime, training,
checkpoint loading, deferred shading, and rasterization stack. Dataset-specific
BLCS, PLCS, court, validation, and reporting modules live in tennis-lab under
`src/synthetic_data_generation`. Tennis-lab invokes those project-owned workers
with this repository's pinned Python environment through a shell-free subprocess
boundary. NHT must never import tennis-lab domain modules.
