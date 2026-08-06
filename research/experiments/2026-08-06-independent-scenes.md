# Independent-scene production acceptance — 2026-08-06

## Frozen design

The source inventory and expected accept/fail-safe outcome were recorded before
these runs. Every source was captured independently; the set spans at least two
venues, devices and dates. The benchmark profile always enables classic
SIFT/COLMAP and ALIKED/LightGlue, while the production profile performs the
learned retry only after primary rejection. Per-scene code paths and hidden
threshold overrides are not used.

## Results

| Independent source | Main challenge | Classic / learned registered | Outcome |
|---|---|---:|---|
| supplied `tennis_court.mp4` | long handheld, repetitive courts | 491 / 492 raw (491 supported) | E2E accepted |
| Pexels Sunset Park 5015666 | low light, flare, exposure | 51 / 51 | E2E accepted; classic selected |
| Pexels Rural 20698836 | aerial motion, distant court | 4 / 86 | E2E accepted; learned selected |
| Pexels Munich 35014859 | winter light, multiple courts | 3 / 51 | E2E accepted; learned selected |
| Pexels Blue Courts 9105177 | repetitive texture | 2 / 7 | fail-safe rejection |
| Pexels Urban Match 14883508 | players, cars, shadows | 2 / 11 | fail-safe rejection |
| Pexels Cansu Overhead 35634647 | players, portrait descent | 2 / 3 | fail-safe rejection |
| Pexels Cansu Angled 35634600 | players, portrait approach | 4 / 3 | fail-safe rejection |
| Pexels Green Match 3858867 | players, top-down | no usable model | fail-safe execution failure |
| Pexels Red Multisport 14992855 | players, mixed markings | no usable model | fail-safe execution failure |
| Pexels Knoxville 17290119 | players, aerial motion | no reconstruction | fail-safe execution failure |
| Nitto optical-zoom arena 000003 | zoom, autofocus, low parallax | 5 / 3 / 6 for fixed/segment/per-image | fail-safe rejection |

The accepted sources continued through the same 500-step NHT acceptance
recipe, standard export and independent-process render validation. Their held-out
validation results were:

| Source | PSNR dB | SSIM | LPIPS-Alex | Independent observed render PSNR dB |
|---|---:|---:|---:|---:|
| supplied base (one-step coordinate round trip) | — | — | — | 51.504 trainer/export agreement |
| Sunset Park | 25.176 | 0.7670 | 0.3953 | 24.268 |
| Rural | 15.990 | 0.5253 | 0.5617 | 14.756 |
| Munich | 16.624 | 0.5166 | 0.6060 | 16.213 |
| B00 cross-repository selected learned candidate | 15.546 | 0.6069 | 0.6398 | 17.055 |

Every render produced finite float32 RGB, alpha and depth. Munich additionally
rendered a camera request translated 10 cm laterally, proving the arbitrary-camera
boundary. Every rejected scene stopped before NHT/export; failures remain in its
mutable run manifest and per-candidate diagnostics.

## Dynamic-object analysis

The Urban Match source compares both candidates under the same input and gates.
A frozen YOLO checkpoint detected people/cars in 8 registered-source images, and
`scripts/analyze_dynamic_tracks.py` measured reconstructed observations inside
those boxes separately from static-background support. Classic registered only
2 images. Learned registered 11, had at least 50 static observations in every
registered image, no pose-step outliers, a 1.402 maximum/median step ratio, and
only 31/3,151 tracks (0.984%) touched a dynamic box. Its rejection was caused by
global registration coverage, not dynamic-track domination.

Dynamic masking is therefore not enabled unconditionally: it would add detector
runtime and discard useful pixels without addressing the measured low-overlap
failure. It remains an explicit future candidate only when dynamic-box track
contamination exceeds 10% or static support drops below the configured
points-per-camera gate. Detection or masking failure must reject that candidate;
it must not silently fall back to unmasked matches. Machine-readable evidence is
stored in `research/evidence/2026-08-06-urban-dynamic-tracks.json`.

## Optical-zoom camera policy

The arena zoom source used identical 1 fps images and SIFT geometry while varying
only camera sharing. A single camera registered 5/60 images and produced a 56.4
pose-step ratio. Ten-frame segment sharing registered 3/60 with two cameras and
stable focal estimates. Per-image cameras registered 6/60, but introduced 14.2%
focal coefficient of variation, 37.2% p95 adjacent focal change, a 24.0 pose-step
ratio and 20% near-duplicate poses. All three correctly failed coverage gates.

Extra intrinsics freedom can slightly increase a local model while making pose
and focal estimates less identifiable; it is not an implicit rescue. Production
uses one shared `OPENCV` camera for nominal fixed-lens videos. Segment and
per-image modes are explicit research profiles only, activated after a
fixed-camera model has enough global coverage to distinguish coherent optical
drift from low-parallax failure. Exact results are in
`research/evidence/2026-08-06-optical-zoom-camera-policy.json`.

## Candidate-to-NHT downstream check

`scripts/run_candidate_nht_benchmark.py` isolates an accepted candidate model and
runs the same seed, frames, 500-step NHT recipe, export and renderer. On Sunset
Park, both candidates registered all 51 images:

| Candidate | Points / median track | SfM runtime s | PSNR dB | SSIM | LPIPS | Observed PSNR dB |
|---|---:|---:|---:|---:|---:|---:|
| SIFT incremental | 12,271 / 7 | 301.1 | 25.176 | 0.7670 | 0.3953 | 24.268 |
| ALIKED + LightGlue | 3,223 / 15 | 257.5 | 23.174 | 0.7504 | 0.4213 | 22.434 |

Longer learned tracks and slightly lower reprojection error did not compensate
for 3.8× lower point density; classic was better on every held-out NHT metric.
This supports the current ranking after semantic eligibility.

The comparison was repeated on the supplied B00 source through the real
`tennis-lab` boundary. Learned matching registered and supported all 74 images;
SIFT registered 73 and supported 72. Under the same recipe, learned produced
19,961 points and PSNR/SSIM/LPIPS of 15.525/0.6077/0.6389 versus SIFT's 12,018
points and 15.325/0.5961/0.7351. SIFT was 0.824 dB better on one independently
rendered observed camera, while learned won all aggregate held-out metrics and
full coverage. The semantic-first ranking therefore remains appropriate; a
single observed render is retained as a diagnostic rather than promoted above
coverage and held-out evaluation. Exact evidence is in
`research/evidence/2026-08-06-b00-candidate-nht.json`. The current production
contract replay, including the non-default near/far planes, CUDA selection,
accepted alignment and all three downstream datasets, is recorded in
`research/evidence/2026-08-06-b00-contract-replay.json`.

On Halifax, an explicit learned high-overlap research profile registered all
98 images with 15,336 points and median track length 22, but took 968.4 seconds.
The production classic candidate already registered all 98 with 14,264 points.
This shows expanded learned pairing can recover that source, but does not justify
adding a costly implicit retry after an accepted primary result.

## Failure interpretation

The expected-success labels intentionally remain visible for sources that were
rejected. They show the present policy's recall limit instead of retroactively
changing acceptance targets. Publication correctness is preserved: a local
two-to-eleven-view model with good reprojection error is not promoted when it
does not cover the video. Learned matching is valuable on Rural and Munich, but
is retained as a bounded retry because classic remains denser and better for NHT
when both are globally valid.
