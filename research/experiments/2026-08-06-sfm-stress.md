# Tennis-court SfM stress benchmarks — 2026-08-06

## Design

Four 60-second clips were taken from the supplied capture. All use 1 fps frame
sampling, the benchmark gates in `configs/sfm-benchmark.yaml`, and both
SIFT/COLMAP incremental and ALIKED-N16/LightGlue candidates. The transforms are
deterministic stress tests, not independent scene evidence:

- 120–180 s: temporal blur plus reduced brightness;
- 240–300 s: synthetic digital zoom;
- 360–420 s: unmodified control;
- 360–420 s: the same control plus two moving opaque rectangles.

All clips yielded 60 accepted preprocessing frames. A successful partial run has
top-level `pending` because it intentionally stopped at `sfm_selection`; a failed
selection has top-level `failed` and leaves NHT/export invalidated.
The measured runs used local HLOC/LightGlue import roots recorded in each
workspace's `resolved-config.yaml`; the portable equivalent is the
`sfm-learned` project extra.

## Results

| Clip | Candidate | Registered / supported | Points | Reprojection p95 px | Median track | Max/median step | Runtime s | Result |
|---|---|---:|---:|---:|---:|---:|---:|---|
| blur-lowlight | SIFT incremental | 60 / 60 | 20,935 | 2.076 | 3 | 1.184 | 158.7 | accepted, selected |
| blur-lowlight | ALIKED + LightGlue | 60 / 60 | 11,986 | 2.152 | 4 | 1.174 | 112.9 | accepted |
| digital zoom | SIFT incremental | 60 / 60 | 22,708 | 2.446 | 3 | 2.517 | 272.9 | accepted, selected |
| digital zoom | ALIKED + LightGlue | 60 / 60 | 13,687 | 2.376 | 4 | 1.744 | 228.7 | accepted |
| dynamic control | SIFT incremental | 3 / 0 | 23 | 1.654 | 2 | 1.669 | 93.1 | rejected |
| dynamic control | ALIKED + LightGlue | 3 / 3 | 366 | 1.962 | 2 | 1.166 | 49.7 | rejected |
| moving rectangles | SIFT incremental | 2 / 0 | 47 | 2.665 | 2 | 1.000 | 100.9 | rejected |
| moving rectangles | ALIKED + LightGlue | 3 / 3 | 1,155 | 2.119 | 2 | 1.218 | 55.6 | rejected |

Both accepted clips selected SIFT because supported registration tied and its
sparse-point density was higher. The learned candidate produced longer median
tracks and a smoother trajectory on digital zoom, but selection does not trade
away point density when every semantic gate already passes.

## Failure analysis

The 360–420 second control and occluded version both failed registration ratio,
supported registration ratio, and sparse-point gates. A contact sheet shows a
wide pan from near one position across repetitive, mostly planar courts: there is
little translational baseline for triangulation. Since the unmodified control
already failed, the experiment cannot attribute failure to the synthetic moving
rectangles. It does demonstrate that the pipeline rejects a small, low-parallax
model despite acceptable reprojection error, records both candidate diagnostics,
and does not run NHT.

The learned reconstruction call initially exposed an HLOC API compatibility bug:
the database path had been passed as an obsolete positional argument, duplicating
`camera_mode`. The production adapter now uses the current five positional inputs
(`sfm_dir`, `image_dir`, `pairs`, `features`, `matches`) and the reruns above both
completed.

## Limits

These clips share one camera, venue, day and source capture. Digital zoom is not
optical zoom; opaque rectangles are not deforming players; and the failed control
is not an independent low-parallax capture. Real multi-scene promotion evidence
is still required.
