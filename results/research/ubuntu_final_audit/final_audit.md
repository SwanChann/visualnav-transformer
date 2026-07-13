# Ubuntu Final Audit — U00–U15

> Windows integration note (2026-07-13): this audit was written before the Ubuntu worktree was committed. The complete packet is now preserved on GitHub at `a49d37220a514dc7b9efd3820edc3d6fbf7ce70c`. The historical worktree/dirty-state statements below describe audit time, not current remote persistence. The authoritative cross-platform asset entry point is `PROJECT_ASSET_MAP.md`; text protocol hashes use LF-normalized UTF-8 semantics, while binary assets retain raw-byte hashes.

## Verdict

The authorized Ubuntu simulation/offline workstream is complete. Corrected protocol v0.3,
the 60-episode controlled matrix, independent offline evaluation, statistical
analysis, Table III/figures, evidence-bounded paper updates and checksum-bound
reproducibility package are present and verified.

This does **not** make the paper submission-ready. Submission remains NO-GO because
protocolized repeated real-robot outcomes, named target-device deadline/miss data,
and privacy/consent review are external/forbidden in this Ubuntu workstream.

## Controlled evidence

- 60 planned = 60 recorded = 25 success + 35 failure;
- 0 crash, 0 infrastructure invalid, 0 missing;
- two retained scenes, three diffusion seeds, five K=8 configurations, two separate
  stabilizer strata;
- policy-only success: DDPM-10 4/6; DDIM-2 5/6; DDIM-3 5/6; DDIM-2/TTS-8 5/6;
  DDIM-2/CFG-2/TTS-8 6/6;
- stabilizer-on success: 0/6 for every configuration, with 0.50 stuck rate;
- stabilizer-on trajectories are byte-identical across methods for each scene/seed;
  only latency differs, so these are not independent navigation outcomes;
- offline ADE versus policy-only success: rho=-0.67, exact p=0.30, reported as
  negative/descriptive evidence;
- no historical MuJoCo record enters the controlled aggregate.

A five-configuration v0.3 exact-key repeat on medium/seed-23 preserved every binary
outcome but produced up to 0.444 m common-prefix coordinate deviation. Primary
trials remain immutable, but their method ordering is explicitly treated as
stochastic and descriptive rather than deterministic.

U10 v2 is invalidated because it used 0.25 m instead of GO Stanford's configured
0.12 m scale. Corrected U10 v5 contains 75/75 valid records and is the only offline
source allowed in downstream correlation. U09 v0.2 is likewise audit-only because
it omitted policy-to-target metric scaling; v0.3 derives 0.1 m from the frozen
MuJoCo nominal contract of 0.4 m/s at 4 Hz and repeats blind scene gating and the
full matrix. Each implementation call advances 0.24 simulated seconds because
locomotion steps are integral; the reported `loop_hz_mean` is headless wall-clock
compute throughput, not a contradictory control frequency.

The policy backend now supports constructor injection through a framework-neutral
contract. A candidate-diversity adaptive router was tested and rejected rather than
integrated. The single-4090 TinyNavBrain plan and static validator are present; no
training, backward pass or optimizer was executed.

## Verification

- analysis tests 33/33, data tests 11/11, benchmark tests 7/7,
  policy-backend tests 5/5 and training-plan tests 4/4;
- 60/60 trial JSONs pass custom semantic validation and JSON Schema 2020-12;
- U11 clean regeneration is byte-identical by recursive diff;
- three PDFs are valid and Table III regenerates from stored trials;
- secret scan, new-artifact >10 MB scan (excluding pre-existing checkpoint/history
  archives), training-path scan and `git diff --check` pass;
- reproducibility manifest covers 141 code/protocol/raw/analysis/figure/paper/repeat-audit/offline/backend/training-plan files.

## Evidence boundary

With three seeds and six episodes per configuration/stratum, method comparisons are
descriptive. No significance, robustness-generalization, real-time guarantee,
cross-dataset result or real-robot success claim is permitted. The stabilizer result
is specific to the frozen scenes, controller and 15.5 s budget.

## Git and persistence

All deliverables remain intentionally uncommitted in the persistent
`agent/ubuntu-sim-handoff` worktree. No remote operation or branch switch occurred.
The user-owned dirty main worktree was not modified.
