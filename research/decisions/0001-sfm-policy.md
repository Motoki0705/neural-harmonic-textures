# Decision 0001: bounded incremental SfM with one learned retry

Status: accepted on 2026-08-05 for the supplied tennis-court validation data.

## Primary

1 fps extraction → blur/exposure/near-duplicate filter → SIFT CPU extraction →
temporal overlap 10 with quadratic overlap → guided matching → COLMAP incremental
mapping and bundle adjustment with a single `OPENCV` camera.

## Retry

Run exactly once only if the primary fails a semantic gate: ALIKED-N16 +
LightGlue, union of temporal-overlap-10 and NetVLAD-top-10 pairs, followed by the
same incremental COLMAP geometry backend. Candidate configuration and failure
are stored separately.

## Selection and failure

Only candidates passing every gate are eligible. Eligible candidates are ranked
lexicographically by supported registration, sparse point density, p95
reprojection error, median track length and trajectory continuity. Gates also
cover mapping components, pose jumps/outliers, near-duplicate cameras, rotation,
ground-path planarity, intrinsics and minimum camera support. If none are
eligible, `sfm_selection` is failed and NHT/export remain invalidated.

## Rejected alternatives

SIFT global mapping is excluded because its apparently strong scalar metrics hid
a 60× trajectory step and 10 pose outliers. GLUEMAP and MASt3R remain research
options, not silent fallbacks, because their added runtime/integration cost has
not been justified across independent difficult captures.

## 2026-08-06 stress evidence

Both candidates passed 60-frame blur/low-light and digital-zoom clips; SIFT won
on sparse-point density. Both rejected a paired low-parallax control and
synthetic-occlusion clip, correctly stopping before NHT. Because every clip is
derived from the supplied capture, this evidence supports the bounded policy and
failure semantics but does not close the independent multi-scene evidence gap.
