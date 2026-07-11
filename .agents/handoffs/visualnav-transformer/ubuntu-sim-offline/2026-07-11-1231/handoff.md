# 1. Copyable Prompt For New Conversation

```text
请使用 $context-handoff 读取本交接包。目标 handoff_id 是 `visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`。先在 Ubuntu 项目根目录运行 project snapshot 与 git status，核对完整工作树和关键文件；然后复述证据边界，再继续“下一步行动”：搭建并执行不训练的、带完整 provenance 的 RA-L MuJoCo 受控闭环实验，以及独立离线评估。本环境允许离线实验和仿真实验，禁止模型训练和真机实验。不要回滚、清理或覆盖未提交/未跟踪文件。
```

# 2. Handoff Identity

- `handoff_id`: `visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`
- `created_at`: `2026-07-11T12:31:00+08:00`
- `source_conversation`: Windows Codex 全权托管研究收口；Codex 主控，Volcengine GLM-5.2 workers 只读 producer/critic
- `agent_role`: `ubuntu-simulation-offline-agent`
- `workstream`: 当前 RA-L 的受控 MuJoCo 补证、独立离线评估和证据集成
- `scope`: `scripts/analysis/`, `scripts/experiments/`, `scripts/simulation/`, `scripts/shared/`, `results/`, `topomaps/`, `deployment/model_weights/nomad/`, `投稿冲刺/workspace/`, `.agents/workstreams/navigation-research-main/`, `后续研究内容/benchmark/`
- `not_scope`: 模型训练或微调、真机运行、ROS 真机部署、自动填写真机 outcome、Git commit/push/pull/fetch/branch cleanup（除非用户另行授权）
- `related_handoffs`: `visualnav-transformer/ral-submission/2026-07-09-1846`；`.agents/workstreams/navigation-research-main/handoff.md`

# 3. Project Overview

- Project root on source machine: `F:\codespace\visualnav-transformer`. On Ubuntu,
  discover the actual root with `git rev-parse --show-toplevel`; never retain the
  Windows drive prefix or backslashes in commands.
- Project purpose: NoMaD/ViNT visual goal navigation and exploration, with diffusion
  waypoint generation, Lite3 quadruped integration, offline configuration analysis,
  and MuJoCo closed-loop evaluation.
- Main flow: RGB observation/goal -> frozen NoMaD-style encoder/fusion -> diffusion
  waypoint chunk -> waypoint-to-velocity bridge -> Lite3 state machine/simulator.
- Main entry points:
  - `scripts/simulation/nomad_mujoco_lite3_state_machine.py` — preferred state-machine simulator; exposes seed, goal seed, scheduler, CFG, TTS and route-stabilizer flags.
  - `scripts/simulation/nomad_mujoco_lite3_nav.py` — legacy/integration entry with easy/medium/hard scenes and topomap generation.
  - `scripts/analysis/ral_offline_pareto.py` — Fig. 2/Table II offline unification.
  - `scripts/analysis/ral_mujoco_summary.py` — conservative historical-result audit.
  - `scripts/shared/nomad_inference.py`, `scripts/shared/nomad_eval_common.py` — frozen policy inference/evaluation utilities.
- Important instructions: read `CLAUDE.md`, `ENVIRONMENT.md`,
  `docs/nomad_environment_setup.md`, `.agents/workstreams/navigation-research-main/AGENT.md`, `TASKS.md`, `log.md`, and `handoff.md` before editing or running experiments.
- Environment starting points: `environment.sim.yaml` and `ENVIRONMENT.md`. The
  external `real-stanford/diffusion_policy` package is required and is not stored in
  this repository. On Ubuntu, resolve its location explicitly and verify its commit.
- Relevant skill: `$context-handoff`. Use repository facts over this packet if they
  differ after transfer.

# 4. Previous Conversation Summary

- Durable goal: objectively determine whether the project can support an RA-L paper,
  close all locally possible evidence gaps, and develop a future multi-dataset
  benchmark + small generative navigation-policy direction without inventing results.
- Decisions:
  - Current paper is draft-ready but submission `NO-GO` until clean simulation and
    protocolized repeated real-robot outcomes exist.
  - Defensible current claim is frozen-policy deployment characterization, not a new
    navigation method, SOTA, cross-dataset generalization, or independently proven
    TTS improvement.
  - Future TinyNavBrain is a falsifiable candidate, not a presumed winner. Training
    remains out of scope in the Ubuntu workstream.
- Constraints/preferences: objective and skeptical assessment; preserve all dirty
  files; do not turn filenames or automatic `status=success` into ground truth;
  distinguish case-mean latency tails, per-call latency, and full-loop latency.
- Agent workflow: Codex made all edits and decisions. Volc workers only read files.
  The final critic's valid objections were integrated; its incorrect assertion that
  no local manifest existed was rejected using repository evidence.
- Initial source Git state at packet creation:
  - branch `main`, tracking `origin/main`
  - HEAD `0f1284acea885d661f171413d58f713ef6c00767`
  - one dirty worktree with important untracked artifacts
- Transfer state: the complete packet and research artifacts are published on remote
  branch `agent/ubuntu-sim-handoff`. On Ubuntu, fetch and check out that branch, then
  record its actual tip with `git rev-parse HEAD`; do not use the old base hash above
  as the handoff commit.
- Snapshot command was run successfully under Windows with UTF-8 mode. Without
  `PYTHONUTF8=1`, the skill script failed on Chinese filenames; this is a Windows-only
  encoding incident, not a repository error.

## Files created or materially changed

- Current-paper evidence: `投稿冲刺/workspace/claim_evidence_matrix.*`,
  `offline_pareto/`, `mujoco/`, `real_robot/`, `paper_package/`.
- Analysis: `scripts/analysis/ral_offline_pareto.py`,
  `ral_mujoco_summary.py`, `ral_real_robot_timing.py`,
  `ral_real_video_review.py` and their tests.
- Future research: `后续研究内容/data/`, `benchmark/`, `model/`, `paper/`,
  `external_execution/`.
- Research code/results: `scripts/research/`, `results/research/data_audit/`.
- Persistent governance: `.agents/workstreams/navigation-research-main/`.

# 5. Current Implementation State

## Already implemented

- Offline Pareto: 131 source–configuration records, 91 distinct settings, 2,004
  case records, 23 within-source Pareto points and 11 Table II representatives.
- Fig. 2/Table II artifacts and metric notes are reproducible. Pareto dominance is
  only within source. All data are Go Stanford.
- Offline p50/p90/p95 are percentiles across case-level mean latencies, not per-call
  tail latency.
- TTS verifier and quality proxy reuse progress/lateral/smoothness terms; TTS proxy
  gains are partly circular by construction.
- Historical MuJoCo audit: 9 sources, 228 records, 84 source-scoped groups. All 228
  lack explicit seed, latency, stuck, fall and stabilizer metadata; 27 stored success
  values conflict with final-distance semantics. Historical tables are integration/
  audit evidence only.
- Real evidence: 528 sampled timing records, 20 videos across 14 trials, contact
  sheets and AI-assisted review. Formal outcomes remain `needs_human_goal_check`.
- Data/benchmark: grouped Go manifest and leakage audit, benchmark protocol/schema/
  validator, TinyNavBrain feature-level scaffold. These are contracts and code tests,
  not model-effect evidence.
- Current checkpoint expected at `deployment/model_weights/nomad/nomad.pth`; source
  SHA-256 was `66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79`.
- Grouped-v2 manifest SHA-256 was
  `ad75c57369104e77eecaa752262159c38ec4e561729454ebfda7c3c45966c2dc`.

## Partially implemented

- MuJoCo simulation entry points expose seeds and route-stabilizer control, but no
  accepted batch runner currently guarantees the complete RA-L result schema,
  non-saturated scene selection, immutable config/checkpoint hashes, and failure
  taxonomy in every record.
- The paper package has Fig. 2/Table II but Table III remains blocked.
- Independent offline navigation metrics are not yet sufficient to break the
  verifier/proxy circularity.

## Not implemented yet

- A clean controlled MuJoCo matrix with frozen seeds/config/checkpoint, explicit
  stabilizer mode, per-call/full-loop timing, collision/fall/stuck/timeout reasons,
  trajectories and raw provenance.
- A conservative Table III and path/failure plots generated only from that clean run.
- Independent offline metrics such as held-out action ADE/FDE or other measures not
  reused by TTS selection, under matched case/config sets.
- Any new training or real-robot evidence; both remain prohibited in this workstream.

# 6. Current Task State

- Active task: convert the historical-simulation audit into a new, fully auditable
  non-training MuJoCo experiment and add independent offline evidence.
- Last completed step: current/future paper packages, external runbooks, video review,
  receipt hardening, final critic review and regression verification.
- Current blocker on source machine: missing coherent NoMaD/diffusion simulation
  environment. Ubuntu is expected to remove this environment blocker.
- Evidence gathered:
  - `投稿冲刺/workspace/paper_package/05_reviewer_checklist.md`
  - `投稿冲刺/workspace/mujoco/mujoco_audit.json`
  - `投稿冲刺/workspace/offline_pareto/offline_metric_notes.md`
  - `投稿冲刺/workspace/claim_evidence_matrix.csv`
  - `后续研究内容/benchmark/benchmark_protocol_v0.1.md`

# 7. Next Actions

The authoritative exhaustive Ubuntu task ledger is
`ubuntu_execution_backlog.md` in this directory. It defines U00–U15, dependencies,
deliverables, acceptance criteria and the final completion definition. The steps
below are the condensed execution route; the consuming agent must not stop after the
environment or smoke-test phase and call the workstream complete.

## P0 — reconstruct and protect the transferred state

1. From the Ubuntu project root run:

   ```bash
   git rev-parse --show-toplevel
   git status --short --branch
   git worktree list --porcelain
   python /path/to/context-handoff/scripts/project_snapshot.py --root .
   ```

2. Fetch and check out `origin/agent/ubuntu-sim-handoff`. Confirm `.agents/`,
   `results/research/`, `scripts/research/`, `后续研究内容/`, and `投稿冲刺/workspace/`
   exist. A clone left on default `main` will not contain this handoff until the
   branch is merged. If any critical directory is missing after checkout, stop and
   compare the remote branch tree before running experiments.
3. Verify the two SHA-256 values above and record the actual Ubuntu paths. Inspect
   Git LFS/submodules/external `diffusion_policy` state; do not silently download a
   different checkpoint.

## P1 — build and verify the no-training simulation environment

1. Inspect rather than blindly applying `environment.sim.yaml`; its name says
   Python 3.8.5 while the comments note MuJoCo compatibility constraints. Prefer a
   fresh environment and pin `numpy<2`.
2. Verify imports:

   ```bash
   python scripts/tooling/env_check.py
   python -c "import torch, cv2, numpy, yaml, diffusers, mujoco, pybullet, onnxruntime; print('core deps OK')"
   python -c "from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D; print('diffusion_policy OK')"
   python -c "from vint_train.models.nomad.nomad import NoMaD; print('NoMaD OK')"
   python -c "import scripts.shared.nomad_inference; print('inference OK')"
   python scripts/simulation/nomad_mujoco_lite3_state_machine.py --help
   ```

3. Run analysis/data unit tests before simulator mutation. TinyNavBrain tests are
   optional because no training/method experiment is authorized.

## P2 — implement the controlled MuJoCo batch protocol, then execute it

1. First inspect `nomad_mujoco_lite3_state_machine.py` and its session/result writer.
   Add or wrap a batch runner only if required. Every raw trial must contain:
   `run_id`, UTC time, Git commit+dirty state, checkpoint/config SHA-256, map/scene
   version, spawn/goal, `seed`, `goal_seed`, scheduler/steps/CFG/TTS/K/verifier,
   `stabilizer_mode`, image preprocessing, max steps/timeout, success definition,
   final distance, path length/SPL, collision, fall, stuck, intervention/exception,
   per-call inference samples, full-loop samples, and trajectory coordinates.
2. Freeze a compact matrix before looking at results. Minimum candidates:
   DDPM-10/CFG-0/TTS-off baseline; DDIM-2/CFG-0/TTS-off; DDIM-3/CFG-0/TTS-off;
   DDIM-2/CFG-0/TTS-8; DDIM-2/CFG-2/TTS-8. Use identical checkpoint, encoder,
   preprocessing, waypoint bridge, candidate K semantics, maps and >=3 seeds.
3. Treat route stabilizer as a controlled factor. Do not mix on/off results. Ideally
   report policy-only (off) and system-with-stabilizer (on) separately. If off causes
   universal failure, that is a systems result, not permission to hide the factor.
4. Begin with `walk-test`/topomap smoke tests, then a tiny one-config/one-seed run.
   Run full matrix only after its JSON validates. Replace or redesign scenes whose
   success is saturated across all candidates; never tune scenes per method.
5. Keep outputs in a new timestamped directory such as
   `results/research/ral_mujoco_controlled/<run_id>/`; never overwrite historical
   `results/benchmark/mujoco/` data.

## P3 — independent offline experiment and evidence integration

1. On held-out, leakage-safe cases, compute metrics not used by the TTS verifier;
   prioritize action/waypoint ADE/FDE in meters and per-call latency distributions.
   Keep identical case IDs and random seeds across configurations.
2. Report all configs and seeds, not best-only results. Use paired or cluster-aware
   uncertainty by trajectory/session; do not reuse the current normal case-mean CI as
   a training-seed CI.
3. Extend `scripts/analysis/ral_mujoco_summary.py` or add a separate controlled-run
   analyzer. Do not weaken its historical-data warnings.
4. Generate Table III and path/failure plots only when raw trajectories and complete
   provenance pass the schema. Update the claim ledger and paper package with both
   positive and negative findings.

# 8. Verification

## Checks already run on source machine

- `python -m unittest discover -s scripts/analysis -p 'test_*.py'`: 5/5.
- `python -m unittest discover -s scripts/research/data -p 'test_*.py'`: 11/11.
- benchmark validator tests: 7/7; example JSON passed.
- TinyNavBrain scaffold in `prp`: 5/5.
- dataset registry JSON parse and `git diff --check`: passed.
- No model training, new simulation or robot execution was run during that batch.

## Checks still needed on Ubuntu

- Full dirty-tree presence and checkpoint/manifest hashes.
- Core NoMaD + diffusion_policy + MuJoCo imports.
- Headless render viability (`MUJOCO_GL=egl` or `osmesa` as appropriate), GPU/device
  identity, deterministic seeding and simulator version.
- Controlled result schema tests, one-trial smoke run, then full frozen matrix.
- Re-run all applicable tests after Linux path/encoding changes.

## Expected success criteria

- Zero missing provenance fields and zero success/final-distance semantic conflicts.
- Same seed/scene/config is reproducible within declared tolerance.
- Every trial retains raw timing and trajectory data; failures are explicit outcomes.
- Table III is generated from the new controlled directory only.
- Claims remain valid if the method does not win; negative results trigger narrowing,
  not selective deletion.

# 9. Risks And Guardrails

- Do not repeat: treating historical MuJoCo summaries as a controlled comparison;
  comparing Pareto points globally across different sources; calling TTS proxy gain
  independent quality improvement; interpreting sampled timing `N` as trial count.
- Do not undo: transferred research artifacts, historical results, grouped splits,
  manual outcome blanks, or current evidence warnings.
- Requires confirmation: Git commit/push/pull/fetch, branch/worktree deletion,
  destructive dataset/result cleanup, paid downloads, or any scope expansion into
  training or real robot.
- Unknowns: exact Ubuntu GPU/driver/MuJoCo rendering stack; external
  `diffusion_policy` revision; whether the state-machine writer already exposes all
  required fields; whether current easy/medium/hard scenes are non-saturated.
- Cross-stream risk: future TinyNavBrain/multi-dataset training is a separate
  method-paper stream. Do not mix its untrained scaffold into the current RA-L
  simulation comparison.
- Privacy: historical robot videos contain visible people; they are not needed for
  this Ubuntu simulation/offline workstream.
- Portability: use UTF-8 and `pathlib`; avoid hard-coded Chinese-path assumptions,
  Windows drive letters and backslashes.

# 10. Raw Notes

- Base HEAD before publication: `0f1284acea885d661f171413d58f713ef6c00767`.
- Published transfer branch: `agent/ubuntu-sim-handoff`; use its fetched tip as the
  authoritative transfer commit.
- Current RA-L verdict: draft-ready, submission NO-GO.
- Historical audit highlights: 228 MuJoCo records; 228 missing seed/latency/stuck/
  fall/stabilizer; 27 success-distance conflicts.
- Offline highlights: 131 source–configuration rows, 91 distinct settings, 2,004
  case records, all Go Stanford.
- Real highlights (context only): 528 sampled loop records; 20 videos/14 trials;
  0 protocol-confirmed outcomes.
- Primary task ledger: `.agents/workstreams/navigation-research-main/TASKS.md`.
- Paper hard gates: `投稿冲刺/workspace/paper_package/05_reviewer_checklist.md`.
