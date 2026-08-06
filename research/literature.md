# Literature and official implementation review

Reviewed 2026-08-05. Links point to papers or official implementations.

| Work | Relevant evidence | Decision |
|---|---|---|
| [COLMAP](https://github.com/colmap/colmap) and [Structure-from-Motion Revisited](https://demuc.de/papers/schoenberger2016sfm.pdf) | Mature incremental SfM, robust verification and bundle adjustment. | SIFT incremental is the production primary because it produced the smoothest fully supported reconstruction on the supplied long video. |
| [Hierarchical Localization](https://github.com/cvg/Hierarchical-Localization) | Provides retrieval, learned local features/matching and COLMAP reconstruction with reproducible configurations. | Integration boundary for the learned retry. |
| [LightGlue](https://github.com/cvg/LightGlue) | Adaptive learned feature matching; official implementation supports ALIKED. | Used with ALIKED-N16 when the primary fails semantic gates. |
| [ALIKED](https://github.com/Shiaoming/ALIKED) | Lightweight keypoints/descriptors designed for deformable transformation robustness. | Chosen local feature for the retry; it registered every raw sampled frame in the base experiment, although the terminal blur frame was weak. |
| [GLOMAP](https://github.com/colmap/glomap) | Global SfM can improve speed/scalability, but the original repository is archived. | Experimental SIFT global run had a catastrophic camera jump; rejected from production. |
| [GLUEMAP](https://github.com/colmap/gluemap) | Current COLMAP global SfM successor using learned/robust components. | Research-only future escalation; not justified as an automatic retry without multi-scene evidence. |
| [MASt3R](https://github.com/naver/mast3r) | Dense matching and 3D reconstruction can help weak-texture imagery. | Not selected: substantially heavier integration/runtime and no evidence yet that it improves downstream NHT for these long videos. |

The central hypothesis was that repetitive court lines and fences need stronger
matching, but temporal continuity and conservative incremental mapping matter
more than raw registration count. The experiments support using learned matching
as a bounded retry, not as an unconditional primary.
