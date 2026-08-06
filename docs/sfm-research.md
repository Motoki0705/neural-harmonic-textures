# Robust SfM research and production decision

This index connects the research question to primary evidence, implementations,
experiments, failure analysis and the production decision.

```text
repetitive/low-texture handheld tennis video
  ↔ literature.md
  ↔ SIFT incremental, SIFT global, ALIKED+LightGlue implementations
  ↔ experiments/2026-08-05-tennis-court.md
  ↔ experiments/2026-08-06-sfm-stress.md
  ↔ decisions/0001-sfm-policy.md
```

The measured scene inventory is tracked in
[`research/scene-inventory.yaml`](../research/scene-inventory.yaml). Candidate
artifacts keep their own model, config, metrics, trajectory CSV/SVG, runtime and
failure. `candidate-comparison.json` and `selection.json` state which eligible
candidate won and why; all-invalid selection writes the same diagnostics with a
structured failure before raising.

The current production policy is deliberately bounded: one explicit primary and
at most one learned retry. Global mapping is not an implicit fallback. If both
candidates fail any semantic gate, the pipeline fails before NHT or export.

## Metric and gate definitions

All ratios are dimensionless unless a unit is shown. A missing or non-finite
metric fails its gate; no candidate receives a default passing value.

| Gate | Calculation | Unit | Failure interpretation |
|---|---|---:|---|
| registration ratio | registered images / accepted input images | ratio | mapping did not cover the sampled video |
| supported registration ratio | registered images with at least the configured 3D observations / accepted input images | ratio | nominal registration is dominated by weak cameras |
| sparse points | reconstructed 3D point count | points | insufficient geometry at the profile's clip scale |
| points per supported camera | sparse points / supported registered cameras | points/camera | point-count success comes only from video length |
| median track length | median number of observing cameras per 3D point | cameras | correspondences do not persist through the sequence |
| p95 reprojection error | 95th percentile `pycolmap.Point3D.error` | pixels | long error tail indicates inconsistent geometry |
| trajectory step ratio | maximum translation per source-frame interval / median translation per interval | ratio | discontinuous pose jump or a collapsed median step |
| trajectory outliers | steps above `max(5×median, median + 6×MAD)` | steps | isolated camera teleportation |
| trajectory planarity | smallest / largest singular value of centred camera positions | ratio | values above the configured bound are implausible for the target capture policy |
| maximum rotation step | largest relative rotation per source-frame interval | degrees/frame | discontinuous orientation |
| mapping components | connected components of the registered image graph induced by real 3D tracks | components | one reported model contains disconnected camera support |
| near-duplicate fraction | steps no larger than `max(0.05×median, 1e-7)` / all steps | ratio | redundant or collapsed camera poses |
| focal/width bounds | minimum and maximum mean focal length divided by image width | ratio | degenerate or physically implausible intrinsics |
| intrinsics stability | standard deviation / median of per-image focal/width | ratio | the chosen sharing mode has uncontrolled focal variation |
| spatial voxel coverage | occupied cells in a 12³ grid over the robust p01–p99 point bounds / 1728 | ratio | points are concentrated in too little of the reconstructed volume |
| triangulation parallax | median maximum viewing-ray angle per sampled 3D point | degrees | depth is constrained by insufficient baseline |

The production profile targets long nominal captures. For at most 90 accepted
images, the short-clip profile changes only scale-sensitive gates (absolute point
count, points/camera, voxel coverage, and parallax); continuity, reprojection,
intrinsics, and topology gates remain unchanged. `metrics.json` stores the exact
threshold beside every value, so evidence remains interpretable if profiles are
revised.

The deterministic stress benchmark accepted blur/low-light and digital-zoom
clips with both candidates. A paired low-parallax control and synthetic-occlusion
clip rejected both candidates and exercised the all-invalid stop. These derived
clips improve failure coverage but do not replace independent captures.

The independent production-acceptance run is recorded in
[`research/experiments/2026-08-06-independent-scenes.md`](../research/experiments/2026-08-06-independent-scenes.md).
It includes twelve separately captured sources, four reconstruction-to-render
successes, fail-safe cases, real-player track-contamination measurements,
optical-zoom camera-policy evidence, and same-recipe candidate-to-NHT comparison.
