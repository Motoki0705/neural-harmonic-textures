# Tennis-court base experiment — 2026-08-05

## Input and risks

- `data/tennis_court.mp4`: 1920×1080 H.264, 59.97 fps, 29,515 frames,
  492.17 seconds, handheld walk around multiple empty courts.
- Failure modes: repetitive court/fence/window texture, low-texture sky and court,
  motion blur, exposure/autofocus change, rolling shutter, long redundant video.
- Common input: 1 fps sampling. Quality preprocessing rejected only terminal
  `frame_000491.jpg` (Laplacian variance 4.99; threshold 15.46), leaving 491.

## Candidate results

| Candidate | Registered / supported | Sparse points | Reprojection p50 / p95 px | Track | Trajectory | Result |
|---|---:|---:|---:|---:|---|---|
| SIFT + COLMAP incremental | 491 / 491 | 217,115 | 0.647 / 1.974 | mean 4.79, median 3 | max/median step 1.35; 0 outliers; max rotation 28.88° | Accepted primary |
| SIFT + global mapper | 491 / 491 | 112,007 | 0.638 / — | median 4 | max/median step 60.12; 10 outliers; max rotation 138° | Rejected: catastrophic pose discontinuity |
| ALIKED-N16 + LightGlue + temporal 10 + NetVLAD top 10 + COLMAP incremental | 492 raw / 491 supported | 94,572 | — / 2.199 | mean 8.72 | smooth; median center difference from SIFT 0.0368 reference steps | Accepted retry; raw terminal blur camera had only 33 points |

The selected SIFT model also matched an independently generated earlier SIFT
baseline across 491 cameras: median aligned center error was 0.00154 world units,
or 0.0061 median reference steps.

## Downstream NHT validation

- Recipe: factor 2 native PNG, 30,000 steps, cap 1,000,000, seed 42.
- Training: 42m59s; checkpoint `ckpt_29999_rank0.pt`; 1,000,000 Gaussians.
- Validation: PSNR 29.444, SSIM 0.9131, LPIPS-Alex 0.1192.
- Color-corrected: PSNR 29.912, SSIM 0.9132, LPIPS 0.1181.
- Standard export: 491 cameras, 217,115 float32 colored points, images and NHT
  model; all semantic checks passed.
- Local canonical workspace size: frames 278 MiB, SfM 689 MiB, NHT 606 MiB,
  export 652 MiB.

The representative held-out render preserves court lines, net/fence geometry,
buildings and vegetation without a pose-induced scene split. NHT therefore
confirms the selected geometry is usable; it is not used to excuse rejected SfM.

## Failure analysis and next questions

Global SfM's high registration count hid an unusable camera path, demonstrating
why registration or reprojection error alone cannot select a model. The learned
retry strengthened tracks and recovered the blurred raw frame, but that camera
was not sufficiently supported for downstream use. Independent captured videos
with players, optical zoom and different motion remain necessary before promoting
a heavier primary or GLUEMAP escalation.
