# Current Handoff — 2026-07-11 Ubuntu U00–U18 Complete

## Worktree and boundary

- Current worktree: `/home/swanchan/visualnav-transformer-worktrees/agent-ubuntu-sim-handoff`
- Branch/HEAD: `agent/ubuntu-sim-handoff` @ `463ac70129c6afec71fa84e3eba457d436781642`, tracking origin 0/0, intentionally dirty with uncommitted deliverables.
- Main worktree: `/home/swanchan/visualnav-transformer`, `main@0f1284a`, user dirty; untouched.
- Environment: `/home/swanchan/anaconda3/envs/nomad_train`, `MUJOCO_GL=egl`.
- Forbidden remains: training/backward/optimizer, real robot, automatic real outcome labels, remote Git without separate authorization.

## Completed artifacts

- Protocol v0.3: `后续研究内容/benchmark/ral_mujoco_protocol_v0.3.json`, SHA `102bf5192a404f8585153aecf4d5c8885d8bc6940e5105e5f19b02f6e40b4c1b`, target action scale 0.1 m.
- Controlled run: `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/`, 60/60 primary trials, 0 missing/crash/infra invalid. U09 v0.2 is audit-only.
- Corrected offline run: `results/research/ral_offline_independent/u10-independent-20260711-v5/`, 75/75 valid at GO Stanford scale 0.12 m. U10 v2 is invalidated.
- U11 analysis: controlled episode CSV, per-stratum summary, Clopper-Pearson/bootstrap, paired effects, offline correlation, robustness, trajectory-equivalence, provenance and audit.
- U12: `投稿冲刺/workspace/controlled_results/` Table III + 3 PDFs.
- U13: updated paper package and claim-evidence matrix; simulation is descriptive only.
- U14: `REPRODUCE.md` + checksum-bound reproducibility manifest; analysis clean regeneration diff empty.
- U15: `results/research/ubuntu_final_audit/`.

## Scientific interpretation

- Primary policy-only realization: DDPM10 4/6; DDIM2 5/6; DDIM3 5/6; DDIM2/TTS8 5/6; DDIM2/CFG2/TTS8 6/6.
- Stabilizer-on: 0/6 for every configuration, but trajectories are byte-identical across methods per scene/seed; interpret as one controller override behavior with different compute latency, not five independent navigation results.
- Exact v0.3 repeat preserved all five outcomes but showed up to 0.444 m trajectory deviation. Treat all counts as a seeded-stochastic immutable realization; no stable ranking/significance claim.
- Offline ADE vs policy-only success rho=-0.67, exact p=0.30: offline ranking does not validate closed-loop superiority.

## Status

Ubuntu U00–U18 is complete, including policy backend injection, rejected adaptive routing study, corrected U09/U10 and the static single-4090 TinyNavBrain plan. Paper submission remains NO-GO solely on external gates: protocolized repeated real-robot outcomes, named target-device sustained deadline/miss measurements, privacy/consent review, and future training/data acquisition. More local simulation cannot close those gates.

## Resume prompt

```text
读取 `.agents/workstreams/navigation-research-main/{AGENT.md,TASKS.md,log.md,handoff.md}` 和 `results/research/ubuntu_final_audit/`。Ubuntu U00–U18 已完成，不要重跑或覆盖 primary U09 v0.3/U10 v5。只有获得新的外部数据/训练/目标设备 timing/真机人工 outcome 后继续。保持 Codex 主控、Volc glm-5.2 只读；主 worktree 不得清理或改动。
```
