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

The deterministic stress benchmark accepted blur/low-light and digital-zoom
clips with both candidates. A paired low-parallax control and synthetic-occlusion
clip rejected both candidates and exercised the all-invalid stop. These derived
clips improve failure coverage but do not replace independent captures.
