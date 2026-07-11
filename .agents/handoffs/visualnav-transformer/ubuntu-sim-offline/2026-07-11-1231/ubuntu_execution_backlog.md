# Ubuntu Simulation/Offline Complete Execution Backlog

This is the authoritative task list for the Ubuntu Codex workstream. Execute it in
order. “Complete” means the acceptance condition is met and evidence is recorded;
writing a plan does not complete an experiment task.

## Global boundary

- Allowed: environment construction, dependency repair, read-only audits, code/tests,
  frozen-checkpoint inference, offline evaluation, MuJoCo/PyBullet simulation,
  analysis, figures/tables, paper-package updates and reproducibility packaging.
- Forbidden: model training, fine-tuning, distillation, checkpoint optimization,
  real-robot/ROS hardware execution, automatic real-video outcome labeling, and
  deleting or overwriting source results.
- A command that performs backpropagation or optimizer steps is training and must not
  be run, even if described as a “small test.” Synthetic forward/shape tests are okay.

## U00 — Import and transfer-integrity audit

- Dependencies: remote branch `agent/ubuntu-sim-handoff` fetched and checked out on
  Ubuntu (or an equivalent full-worktree transfer if the remote is unavailable).
- Actions:
  1. Locate root with `git rev-parse --show-toplevel`; do not assume a path.
  2. Read `CLAUDE.md`, `PROJECT_CONTEXT.md`, `ENVIRONMENT.md`,
     `.agents/workstreams/navigation-research-main/{AGENT.md,TASKS.md,log.md,handoff.md}`
     and the sibling `handoff.md` in this packet.
  3. Run `git status --short --branch` and the context-handoff project snapshot.
  4. Confirm `.agents/`, `results/research/`, `scripts/research/`, `后续研究内容/`
     and `投稿冲刺/workspace/` exist. If not, stop: the checkout is probably still on
     default `main` rather than `agent/ubuntu-sim-handoff`.
  5. Record Ubuntu OS, kernel, CPU, RAM, GPU, driver, CUDA, Python, Git and filesystem.
- Deliverable: `results/research/ubuntu_environment/transfer_audit.md`.
- Acceptance: fetched commit/branch and local dirty state recorded; every critical
  directory present; missing files explicitly block later tasks.

## U01 — Verify immutable inputs

- Dependencies: U00.
- Actions:
  1. Hash `deployment/model_weights/nomad/nomad.pth`; expected SHA-256:
     `66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79`.
  2. Hash `results/research/data_audit/dataset_manifest_grouped_v2.csv`; expected:
     `ad75c57369104e77eecaa752262159c38ec4e561729454ebfda7c3c45966c2dc`.
  3. Record Git/LFS/submodule status and external `diffusion_policy` path/commit.
  4. Inventory topomaps, MuJoCo assets, model config and encoder weights used by
     default inference. Hash every input selected for controlled experiments.
- Deliverable: `results/research/ubuntu_environment/input_inventory.json`.
- Acceptance: hashes match or discrepancy is explained before any result is mixed
  with source evidence; no silently substituted checkpoint/config/topomap.

## U02 — Build the no-training Ubuntu environment

- Dependencies: U01.
- Actions:
  1. Inspect `environment.sim.yaml`, `ENVIRONMENT.md` and
     `docs/nomad_environment_setup.md`; resolve Python/MuJoCo constraints explicitly.
  2. Use a fresh environment. Preserve `numpy<2`; install the repository editable
     package and external `real-stanford/diffusion_policy` at a recorded commit.
  3. Do not install or initialize a training tracker unless needed for imports; disable
     network logging during experiments.
  4. Select and record headless rendering backend (`egl` preferred with supported GPU,
     otherwise `osmesa`).
  5. Export an environment lock after success without embedding credentials or local
     absolute paths.
- Deliverables:
  - `results/research/ubuntu_environment/environment_report.md`
  - `results/research/ubuntu_environment/pip_freeze.txt` or conda lock/export
- Acceptance: imports for torch/NoMaD/diffusion_policy/OpenCV/MuJoCo/PyBullet work;
  GPU and renderer identity are explicit; no training run occurs.

## U03 — Cross-platform regression and path repair

- Dependencies: U02.
- Actions:
  1. Run all existing analysis, data, benchmark and feature-scaffold tests.
  2. Run `python scripts/tooling/env_check.py` and simulator `--help` commands.
  3. Audit new/relevant code for Windows drive letters, backslashes, locale-dependent
     CSV decoding, case-sensitive filenames and non-UTF-8 output.
  4. Repair only reproducible Linux compatibility defects with `pathlib`/UTF-8 and add
     regression tests. Do not rewrite unrelated legacy/vendor code.
- Deliverable: `results/research/ubuntu_environment/regression_report.md`.
- Acceptance: all applicable tests pass; skipped tests have dependency-backed reasons;
  `git diff --check` passes; no source result is changed by a path-only repair.

## U04 — Reproduce frozen local evidence on Ubuntu

- Dependencies: U03.
- Actions:
  1. Re-run `ral_offline_pareto.py` into a temporary directory and compare the four
     text/CSV outputs byte-for-byte with `投稿冲刺/workspace/offline_pareto/`.
  2. Validate both generated PDFs and compare underlying numeric tables; PDF metadata
     bytes need not be identical if the generator changes metadata.
  3. Re-run `ral_mujoco_summary.py` against historical results into a temporary
     directory; verify 228 records, 84 groups and 27 semantic conflicts.
  4. Do not recompute real-video material unless OpenCV portability itself is under
     test; real outcomes are outside this workstream.
- Deliverable: `results/research/ubuntu_reproduction/reproduction_report.md`.
- Acceptance: numeric/text outputs match; any platform deviation is explained and
  tested; historical warnings remain unchanged.

## U05 — Freeze the controlled-experiment protocol before results

- Dependencies: U04.
- Actions:
  1. Write a machine-readable protocol containing primary question, configurations,
     maps/scenes, seeds, goal seeds, checkpoint/config hashes, image preprocessing,
     waypoint bridge, success radius, timeout/max steps, failure taxonomy, stabilizer
     factor, latency scopes, exclusion rules, statistical unit and planned analysis.
  2. Minimum frozen configurations:
     - DDPM-10 / CFG-0 / TTS off baseline;
     - DDIM-2 / CFG-0 / TTS off;
     - DDIM-3 / CFG-0 / TTS off;
     - DDIM-2 / CFG-0 / TTS-8;
     - DDIM-2 / CFG-2 / TTS-8.
  3. Use identical candidate-K semantics. Explicitly distinguish TTS=0 standard batch,
     ranked TTS=8, and expanded budgets.
  4. Use at least three seeds per configuration/scene; five is preferred if runtime
     permits. Predeclare how failed/crashed runs are counted—never silently retry away
     a failure.
  5. Predeclare route-stabilizer comparison: policy-only off and system-with-stabilizer
     on are separate strata, never pooled.
- Deliverables:
  - `后续研究内容/benchmark/ral_mujoco_protocol_v0.1.json`
  - `后续研究内容/benchmark/ral_mujoco_protocol_v0.1.md`
- Acceptance: protocol validates as JSON, all required factors are fixed, and its
  SHA-256 is recorded before viewing the full matrix results.

## U06 — Implement the controlled-run schema and batch runner

- Dependencies: U05.
- Actions:
  1. Inspect the state-machine/session result writer first; extend or wrap it instead
     of duplicating simulator logic.
  2. Add a JSON schema and validator for raw trials and batch manifests.
  3. Every trial must record:
     - run/trial ID, UTC time, Git commit+dirty status;
     - protocol/checkpoint/config/topomap/scene hashes;
     - device, precision, simulator/renderer versions;
     - map, spawn, goal, seed, goal seed;
     - encoder, scheduler, steps, CFG, TTS budget/top-k/verifier, candidate K;
     - preprocessing, waypoint index, bridge/controller clamps;
     - stabilizer mode, success definition/radius, timeout/max steps;
     - success, final distance, path length/SPL, collision, fall, stuck, timeout,
       intervention, exception and exit code;
     - raw trajectory coordinates, per-call inference timing and full-loop timing.
  4. Write atomically into a new timestamped directory. Support resume by detecting a
     complete trial key; never overwrite or mutate historical outputs.
  5. Add synthetic/unit tests for schema failure, seed/config expansion, duplicate
     trial prevention, crash recording, atomic writes and summary consistency.
- Suggested locations:
  - `scripts/experiments/ral_mujoco_controlled.py`
  - `scripts/analysis/validate_ral_mujoco_trial.py`
  - `后续研究内容/benchmark/ral_mujoco_trial_schema_v0.1.json`
- Acceptance: tests pass; every missing provenance field fails validation; a crash is
  retained as a failed trial record rather than disappearing.

## U07 — Simulator mechanical and determinism smoke tests

- Dependencies: U06.
- Actions:
  1. Run headless stand/walk mechanics without policy comparison.
  2. Generate/verify domain-matched topomaps without using dataset trajectories as
     simulated goals.
  3. Run one configuration, one scene, one seed with timing/trajectory capture.
  4. Repeat the exact trial and quantify deterministic tolerance. If GPU diffusion or
     physics is nondeterministic, record the source and switch the analysis to seeded
     stochastic replication rather than claiming bitwise determinism.
  5. Confirm collision/fall/stuck/timeout triggers with synthetic or deliberately
     constrained smoke cases; do not use these cases in the main comparison.
- Deliverable: `results/research/ral_mujoco_controlled/<run_id>/smoke_report.md`.
- Acceptance: valid trial JSON, trajectory and timing arrays are non-empty; outcome
  agrees with final-distance semantics; failure detectors are exercised.

## U08 — Scene calibration without method leakage

- Dependencies: U07.
- Actions:
  1. Evaluate scene mechanics using the baseline or method-blind aggregate only.
  2. Reject scenes that are universally saturated, impossible, unstable due to broken
     physics, or dominated by a hidden route stabilizer.
  3. Freeze the retained scenes and hashes before comparing configurations.
  4. Do not tune obstacles, goal radius, timeout or controller separately per method.
- Deliverable: `results/research/ral_mujoco_controlled/<run_id>/scene_gate.json`.
- Acceptance: each retained scene has nontrivial outcome/efficiency variance and
  identical rules for all configurations; all exclusions have predeclared reasons.

## U09 — Execute the frozen controlled MuJoCo matrix

- Dependencies: U08.
- Actions:
  1. Run the exact Cartesian product frozen in U05/U08.
  2. Monitor disk, GPU temperature/memory and failures without changing the protocol.
  3. Resume only missing trial keys after interruption; preserve crashes/timeouts.
  4. Validate every trial immediately and validate batch completeness at the end.
- Deliverable: new immutable directory under
  `results/research/ral_mujoco_controlled/<run_id>/`.
- Acceptance: planned trial count equals valid+failed/crashed recorded count; no
  missing seed/stabilizer/timing/trajectory fields; no post-hoc configuration edits.

## U10 — Run independent offline inference evaluation

- Dependencies: U03 and immutable inputs; can run parallel to U07–U09 after protocol
  freeze, but must not train.
- Actions:
  1. Freeze held-out leakage-safe case/session IDs and identical random seeds across
     configurations.
  2. Add independent metrics not reused by the TTS verifier: waypoint/action ADE and
     FDE in meters are primary candidates. Keep trajectory proxy components only as
     secondary diagnostics.
  3. Capture per-call inference latency after warm-up, including p50/p95/p99, device,
     precision, batch/K/NFE and synchronization method.
  4. Compare single-sample and candidate-selection settings fairly; never compare
     best-of-K against K=1 without labeling the extra compute.
  5. Report every configuration/case and invalid result; no best-only filtering.
- Suggested locations:
  - `scripts/experiments/ral_offline_independent.py`
  - `results/research/ral_offline_independent/<run_id>/`
- Acceptance: metrics are in physical units where possible, case pairing is exact,
  latency is per-call rather than percentile of case means, and no optimizer/backward
  call appears in the execution path.

## U11 — Statistical analysis and robustness checks

- Dependencies: U09 and U10.
- Actions:
  1. Analyze simulation at the independent trial/seed level, not individual control
     ticks. Report numerator/denominator and uncertainty for success/failures.
  2. Use paired/cluster-aware bootstrap by scene/goal/seed where appropriate. Treat
     three seeds as descriptive if power is inadequate; do not overclaim significance.
  3. Report path/SPL/final distance, collision/fall/stuck/timeout and latency jointly.
  4. Analyze stabilizer on/off separately and test whether rankings change.
  5. Compare offline independent-metric ranking with closed-loop ranking; report flat
     or reversed correlations as negative evidence.
  6. Check sensitivity to success radius/timeout only as a labeled robustness analysis,
     never to redefine the primary outcome after results.
- Deliverables:
  - `scripts/analysis/ral_controlled_results.py` plus tests
  - controlled unified CSV/JSON, audit report and statistical notes
- Acceptance: analysis regenerates from raw immutable trials; no historical records
  enter the controlled aggregate; all denominators and exclusions reconcile.

## U12 — Produce RA-L figures and tables

- Dependencies: U11.
- Actions:
  1. Generate Table III from controlled simulation only.
  2. Generate path/failure plots only from stored trajectory coordinates; never
     reconstruct paths from scalar summaries.
  3. Update Fig. 2/Table II only if the independent offline experiment materially
     changes the evidence; preserve source-scoped historical plots as an audit trail.
  4. Create captions that state seeds, scenes, stabilizer stratum, metric units,
     uncertainty and evidence limits.
- Deliverables under `投稿冲刺/workspace/controlled_results/` and updates to
  `投稿冲刺/workspace/paper_package/03_figures_tables_captions.md`.
- Acceptance: every number traces to raw data+protocol+script; tables regenerate;
  negative/dominated configurations remain visible where scientifically relevant.

## U13 — Update claims and paper readiness objectively

- Dependencies: U12.
- Actions:
  1. Update `claim_evidence_matrix.csv/.md`, method/results skeleton, inclusion rules
     and reviewer checklist.
  2. Mark G4 complete only if controlled simulation passes provenance and statistical
     gates. Keep G5/real-robot gates incomplete: Ubuntu cannot satisfy them.
  3. If TTS/CFG/DDIM effects vanish or reverse, narrow contributions and title instead
     of explaining the result away.
  4. Reassess RA-L novelty/solidity separately: clean simulation improves solidity,
     not algorithmic novelty.
- Deliverable: revised `投稿冲刺/workspace/paper_package/` plus decision note.
- Acceptance: no unsupported abstract/contribution claim; submission remains NO-GO
  if real-robot protocol evidence is still insufficient.

## U14 — Reproducibility and repository hygiene

- Dependencies: U13.
- Actions:
  1. Add exact reproduction commands, environment lock, protocol hashes, expected
     output tree and estimated runtime/storage.
  2. Keep raw videos, huge caches and credentials out of Git. Use Git LFS or an
     artifact manifest only when repository policy and remote support are confirmed.
  3. Run tests, JSON/schema validation, `git diff --check`, path/secret/large-file
     audit and a clean temporary regeneration test.
  4. Do not delete historical data. Clearly separate `historical_audit` from
     `controlled_results` in paths and paper language.
- Deliverable: `results/research/ral_mujoco_controlled/<run_id>/REPRODUCE.md` and final
  verification report.
- Acceptance: a fresh Ubuntu agent can regenerate analysis from retained raw trials;
  repository contains no secret or accidental giant artifact.

## U15 — Final adversarial review and handoff refresh

- Dependencies: U14.
- Actions:
  1. Perform an adversarial reviewer pass over causal attribution, fairness, statistics,
     saturation, stabilizer confounding, timing scope and claims.
  2. Fix valid P0/P1 defects and rerun checks.
  3. Update this packet's `handoff.md`, the persistent workstream log/task ledger and
     a next-session prompt with actual Ubuntu commit/branch/worktree state.
  4. Stop at the environment boundary. List training and real-robot requirements as
     external blockers; do not mark them complete.
- Acceptance: all U00–U15 locally permitted tasks are complete or have an evidence-
  backed hard blocker; no unresolved P0/P1 review defect; handoff matches repository.

## Final completion definition

The Ubuntu workstream is complete only when:

1. environment and transfer are auditable;
2. frozen historical evidence reproduces;
3. a new controlled simulation matrix with complete provenance finishes;
4. independent offline metrics and true per-call timing finish;
5. controlled Table III/plots and claims ledger regenerate from raw data;
6. all tests/reproducibility/critic gates pass; and
7. remaining training/real-robot gaps are explicitly external, not silently ignored.
