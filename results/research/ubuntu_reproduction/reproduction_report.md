# Frozen Evidence Reproduction on Ubuntu

Recorded: `2026-07-11`

All outputs were regenerated under `/tmp/ral-ubuntu-repro-20260711`; no stored
result was overwritten.

## Offline Pareto

- Regenerated 131 unified configuration rows from 2,004 case records.
- Regenerated 11 representative Table II rows.
- `table_offline_pareto.md`: byte-identical.
- `table_budget_modes.md`: byte-identical.
- `offline_pareto_unified.csv`: content-identical after CRLF/LF normalization;
  the stored Windows artifact uses CRLF and the Ubuntu generator uses LF.
- `offline_metric_notes.md`: semantically identical except that regenerated source
  paths use POSIX `/`, while the stored Windows artifact contains `\` separators.
- Both PDFs have valid `%PDF` headers and nontrivial sizes (113,017 and 10,691
  bytes). Numeric source tables match; PDF metadata bytes were not required to match.

## Historical MuJoCo audit

- Sources: 9.
- Raw records: 228.
- Source-scoped config groups: 84.
- Success/final-distance semantic conflicts: 27.
- Missing stabilizer metadata: 228/228.
- Repeated trajectory signatures: 12.

`mujoco_audit.json`, `table_mujoco_results.md` and `stabilizer_note.md` are
byte-identical to the stored artifacts. The unified CSV is content-identical after
CRLF/LF normalization.

## Conclusion

U04 passes. Ubuntu reproduces the frozen numeric/text evidence. The only deviations
are declared platform path separators and line endings; historical warnings remain
unchanged. These historical MuJoCo records remain audit/integration evidence and
must not enter the new controlled aggregate.
