# Decision 0001: bounded incremental SfM with one learned retry

Status: production candidate, updated 2026-08-06 from independent-scene evidence.

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

## 2026-08-06 independent-scene update

Twelve independent captures across venues/devices were evaluated with outcomes
frozen in advance. Four completed reconstruction through independent rendering;
the learned retry recovered Rural and Munich where classic formed only local
models. When both candidates passed on Sunset Park, classic's 3.8× greater point
density translated to higher PSNR and SSIM and lower LPIPS under the same
500-step NHT recipe, despite the learned candidate's longer tracks. This supports
retaining classic as primary and using learned matching as the one bounded retry.

The same downstream comparison was repeated on B00. Learned matching was
selected with 74/74 supported cameras and 19,961 points over SIFT's 72/74 and
12,018. It also won aggregate held-out PSNR, SSIM and LPIPS under the identical
500-step recipe; SIFT won one independently rendered observed camera by 0.824 dB.
This mixed single-view diagnostic does not override full supported coverage and
aggregate held-out quality, and the existing lexicographic selection is kept.

An explicit Halifax learned profile with overlap/retrieval 30 recovered 98/98
images and 15,336 points but required 968.4 seconds. Because primary SIFT already
accepted 98/98 there, expanded pairing remains a research setting rather than a
new automatic retry.

The optical-zoom arena source was rejected safely: fixed, ten-frame segment and
per-image policies registered only 5/60, 3/60 and 6/60 images. Per-image freedom
introduced a 14.2% focal coefficient of variation, 37.2% adjacent focal change,
24× pose-step ratio and 20% near-duplicate poses. It is not a rescue for weak
geometry. Production therefore keeps one shared camera for nominal
fixed-lens clips. Segment-shared or per-image cameras remain explicit research
profiles activated only when fixed-camera registration otherwise passes enough
of the clip to diagnose coherent focal drift; they are never an implicit retry.

Real-player analysis found less than 1% dynamic-box track contamination and full
static support in the learned Urban Match local model, with no pose outlier. Its
failure was insufficient global coverage, so dynamic masking is not enabled by
default. It may be tested as an explicit candidate only above 10% contaminated
tracks or after static support fails, with detector/mask runtime and failure
recorded.
