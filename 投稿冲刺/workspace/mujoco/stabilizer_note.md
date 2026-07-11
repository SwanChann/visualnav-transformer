# MuJoCo Evidence and Stabilizer Audit

## What was found

- Sources with raw records: 9
- Raw records: 228
- Source-scoped configuration/map groups: 84
- Sources whose recorded runs are all successful: 6
- `success=true` records above 0.60 m final distance: 27
- Config/map identities repeated across sources: 14
- Trajectory-summary signatures repeated across distinct configs: 12

## Missing provenance

- `seed` missing in 228/228 records.
- `latency_ms` missing in 228/228 records.
- `stuck` missing in 228/228 records.
- `fall` missing in 228/228 records.
- `stabilizer_mode` missing in 228/228 records.

The current JSON cannot determine route-stabilizer mode, per-step inference latency, stuck/fall events, or randomized seed identity. Therefore the paper must not interpret these results as pure policy optimality or as a complete safety evaluation.

## Semantic warnings

- A success/final-distance inconsistency is flagged when a record is marked successful but its final distance exceeds the declared 0.6 m threshold. This may indicate a different historical success rule or post-processing; it must be resolved from the runner before using success rate.
- Repeated trajectory summaries across distinct encoders/samplers/CFG values indicate that the benchmark may be dominated by deterministic route/controller behavior. They are an audit signal, not proof of falsification.
- Source-scoped rows are not pooled because duplicate configurations occur in multiple benchmark directories with incomplete provenance.

## Figure availability

`fig_mujoco_paths.pdf` was not generated. Existing raw records contain scalar path lengths but no trajectory coordinates, so a path figure would be fabricated. Supply time-indexed positions from the original run logs before drawing it.
