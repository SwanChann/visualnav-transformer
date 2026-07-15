# 1. Copyable Prompt For New Conversation

```text
请使用 $context-handoff 导入本交接包。目标 handoff_id 是 `visualnav-transformer/pretraining-4090/2026-07-15-1641`，文件位于 `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md`。工作环境是远程 RTX 4090 服务器。先从当前仓库重建事实：核对分支、完整 Git SHA、dirty 状态、PROJECT_ASSET_MAP.md、ENVIRONMENT.md、任务队列和本包；然后只执行本包 Phase 0 的只读 live snapshot，并按“已验证事实 / 推断 / 未验证假设 / 阻塞门”报告。不要下载外部数据，不要运行 forward/backward、optimizer、训练、离线评测或仿真，等待用户在新一轮明确授权。不要重复 Ubuntu U00–U18，不要覆盖 U09 v0.3 或 U10 v5，不要把静态合同、shape test、200-step 计划或 loss 下降写成训练/方法结果。任何 commit、push、pull/fetch、分支/worktree 操作、外部数据获取和训练执行均需用户当轮授权；Windows 原项目目录及 PROJECT_ASSET_MAP.md 始终是最终人工入口。
```

# 2. Handoff Identity

- `handoff_id`: `visualnav-transformer/pretraining-4090/2026-07-15-1641`
- `created_at`: `2026-07-15T16:41:35+08:00`
- `source_conversation`: Windows 主控完成三环境职责冻结、无训练 pre-training 基础设施和 4090 交付审查
- `agent_role`: `rtx4090-pretraining-executor`
- `workstream`: 4090 live audit、DATA-PILOT、真实 adapter、B0 smoke、多数据集 baseline/H1 与跨数据集离线门
- `scope`: `ENVIRONMENT.md`、`PROJECT_ASSET_MAP.md`、`.agents/workstreams/navigation-research-main/`、`scripts/research/{pretraining,data,models,training_plans}/`、`后续研究内容/{data,model,benchmark,external_execution}/`，以及 4090 上由用户明确指定的数据/checkpoint 根目录
- `not_scope`: Windows 论文编辑；重复 Ubuntu U00–U18；改写 U09 v0.3/U10 v5；未授权的数据下载、训练、仿真、真机、Git 远端或 worktree 操作
- `related_handoffs`: `visualnav-transformer/navigation-research-main/2026-07-14-2002`（历史主线，只作背景）；`visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`（已完成 Ubuntu 流，不合并执行）

# 3. Project Overview

- Project root: 4090 上的实际路径未知；不得照抄 Windows 的 `F:\codespace\visualnav-transformer`。进入服务器已有 clone 后先运行 `pwd`。
- Project purpose: 基于 NoMaD/ViNT 的视觉目标导航研究；下一候选问题是在固定数据、encoder、参数、样本数与推理预算下，比较多数据集 mixture 策略、确定性 H0 与低 NFE rectified-flow H1。
- Canonical human entry: Windows 原项目目录和根目录 `PROJECT_ASSET_MAP.md`。4090 是数据/训练/离线执行端，不是论文权威入口。
- Git delivery: branch `agent/ubuntu-sim-handoff`；Windows 审查 source base 为 `d8a238e169a626cae7eb69fe40e18421ea628162`。实际交付 SHA 是包含本文件的提交，4090 检出后必须用 `git rev-parse HEAD` 记录，不得继续使用 source-base SHA 代替。
- Data residency: 用户规定完整数据集只驻留 4090；Windows 和 Ubuntu 不保存副本。服务器 live 数据清单、路径、完整性和许可状态尚未验证。
- Important instructions: 静态基础设施不是训练结果；失败/负结果必须保留；跨环境产物必须绑定完整 Git SHA、manifest/split/model-contract/config SHA-256。

# 4. Previous Conversation Summary

- User's durable goal: 将多数据集训练与 TinyNavBrain 探索转成可执行、可否证的 pre-training backlog，先完成不需要实际训练的基础设施和测试，再决定外部数据与训练。
- Environment decision:
  - Windows：唯一人工/Git/论文集成入口；静态合同和测试。
  - 4090：数据唯一驻留；真实 adapter、smoke、训练和 IID/mixed/LODO/corruption 离线评估。
  - Ubuntu：只接收通过 `OFFLINE-GATE` 的 checkpoint 做新非饱和闭环 `SIM-GATE`。
- Authoritative existing results: U09 v0.3 为 60/60 有效 controlled records；U10 v5 为 75/75 有效离线 cases。它们是历史权威证据，不是新模型训练结果，也不是 4090 待重跑任务。
- Scientific boundary: 当前论文仍为 submission `NO-GO`；仓库没有多数据集训练结果、H1 方法收益、新非饱和闭环结果或协议化重复真机结果。
- Current-turn authorization boundary: Windows 本轮只获准生成、审查、commit 和 push 本交付。该授权不自动延伸到 4090 的 pull/fetch、数据访问、下载或训练；服务器会话需要用户重新授权。
- Files added/changed: 三环境文档、任务和资产入口；`scripts/research/pretraining/` 静态合同/backlog/tests；4090 计划验证器；外部执行顺序；本交接包。

# 5. Current Implementation State

## Already implemented and locally verified on Windows

- Canonical seven-item ViNT batch adapter contract：显式 meter scale、goal mask、dataset/trajectory IDs。
- Past-only action history：不得读取 observation time 之后的 pose，padding 必须为零且单独 mask。
- Proportional/balanced sampler 概率合同。
- H0 SmoothL1 和 H1 rectified-flow 的 masked dataset-macro loss 纯函数合同。
- Checkpoint provenance/exact-resume 元数据校验：Git、manifest、split、model contract、config、RNG、sampler state。
- `pretraining_backlog_v0.1.yaml` 与 `smoke_config_v0.1.yaml`；二者明确 `training_performed: false` / `planned_no_training_executed`。
- `tinynavbrain_4090_plan.yaml`：shared EfficientNet-B0、6 observation frames、4-step history、8-step horizon、deterministic H0、rectified-flow H1、NFE 1/2/4/8；显存预算只是 analytic partial accounting。

## Partially implemented

- TinyNavBrain 仍是 feature-level scaffold 和 shape tests；尚无真实图像/数据 adapter 接线、完整 train loop、实际 checkpoint/export。
- 数据 registry、manifest、group-safe split、leakage audit 和 receipt 工具已存在，但服务器真实数据尚未接入本合同。
- Repo 记录 Go Stanford 已 materialized；这不能证明 4090 当前文件完整或路径正确。

## Not implemented or not executed

- 4090 live snapshot、服务器依赖/磁盘/数据/checkpoint 核验。
- 合规 RECON/HuRoN pilot、真实 adapter 和三域 group-safe splits。
- B0 200-step smoke、baseline matrix、H1 训练、多 seed 与离线门。
- 新 Ubuntu `SIM-GATE`、目标设备持续 timing、重复真机和隐私/同意审查。

# 6. Current Task State

- Active task: 把已审查的 Windows 静态基础设施交给 4090，但先只重建服务器事实。
- Last completed step: Windows 对全部 tracked/untracked 交付内容做 diff、路径、术语、秘密扫描和本地测试审查，并生成本包。
- Current blocker: 4090 的实际 clone 路径、HEAD/dirty 状态、Python/CUDA/PyTorch、磁盘、数据根目录、数据完整性、旧 checkpoint 和可用训练环境均未知。
- Evidence gathered: `ENVIRONMENT.md`、`PROJECT_ASSET_MAP.md`、任务队列、pre-training backlog、4090 plan、外部执行 runbook，以及 Windows 测试记录。

# 7. Next Actions

## Phase 0 — only after the user authorizes read-only server inspection

1. 在 4090 已有项目 clone 中记录 Git 状态，不修改工作区：

   ```bash
   pwd
   git status --short --branch
   git worktree list --porcelain
   git rev-parse HEAD
   git log -1 --oneline --decorate
   ```

2. 核验本文件确实存在，并读取：

   ```bash
   test -f .agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md
   sed -n '1,240p' PROJECT_ASSET_MAP.md
   sed -n '1,240p' ENVIRONMENT.md
   sed -n '1,240p' .agents/workstreams/navigation-research-main/TASKS.md
   ```

3. 只读采集软件/GPU/磁盘事实，不输出环境变量或凭据：

   ```bash
   uname -a
   python --version
   command -v python
   nvidia-smi --query-gpu=name,driver_version,memory.total,memory.free --format=csv,noheader
   df -h
   python -c 'import torch; print({"torch": torch.__version__, "cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "device_count": torch.cuda.device_count()})'
   ```

4. 不扫描整个服务器。先由用户给出允许检查的 `DATA_ROOT` 和 `CHECKPOINT_ROOT`；仅在这些根内列目录、大小、mtime 和 checksum/receipt/manifest 存在性。不得移动、复制、转换或下载数据。
5. 输出 `4090 live snapshot` 报告，严格分为：已验证事实、推断、未验证假设、阻塞门。到此停止，等待用户决定数据接入和训练。

## Phase 1 — after separate data-access/acquisition authorization

1. 按 `后续研究内容/external_execution/01_data_pilot.md` 核验许可、artifact receipt 和 checksum。
2. 在 quarantine 中做真实 adapter；生成 manifest、trajectory-grouped split 与 zero-leakage audit。
3. 至少三域可审计前，禁止 cross-dataset claim；SCAND/TartanDrive 载荷许可未确认前不得下载。

## Phase 2 — after separate training authorization

1. 先验证静态合同，再执行冻结的 200-step B0 smoke；记录实际 peak VRAM、吞吐、finite loss、meter sanity 和 exact next-batch resume。
2. smoke 通过后才执行 `baseline matrix`，至少 3 seeds；固定数据曝光、encoder、参数/步数、candidate K 和 bridge。
3. baseline 稳定后才训练 H1；若 H1 不超过 sampler-matched H0 或收益不超过 seed variance，停止生成式 head 主张。
4. 4090 完成 IID/mixed/LODO/corruption `OFFLINE-GATE` 后，才向 Ubuntu 晋级少量 checksum-bound checkpoints。

# 8. Verification

- Checks already run on Windows:
  - pre-training contracts: 10/10 passed。
  - training-plan validator tests: 7/7 passed。
  - existing TinyNavBrain scaffold tests: 5/5 passed。
  - existing data contract/receipt tests: 11/11 passed。
  - backlog + smoke CLI validator: passed。
  - 4090 plan CLI validator: passed；预算类型为 `analytic_partial-accounting_not_measured`。
  - Python `compileall`、`git diff --check`、路径/旧术语一致性与秘密模式扫描通过。
- Checks still needed on 4090: live snapshot；确认可用 Python env 后重跑静态测试；真实数据 adapter/manifest/leakage；之后才可能 smoke。
- Expected Phase-0 success: 形成可审计服务器事实表，不改变数据、checkpoint、Git 工作区或实验状态。
- Expected training success: 不是“loss 下降”，而是同等预算下多 seed、分域、worst-domain/LODO 和非饱和闭环证据超过对应 baseline，且完整 provenance 可复现。

# 9. Risks And Guardrails

- Do not repeat: Ubuntu U00–U18；U09 v0.2/U10 v2 的已知无效路径；在 4090 重跑 U09 v0.3/U10 v5。
- Do not undo: `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/`、`results/research/ral_offline_independent/u10-independent-20260711-v5/`、作废标记、checksum/audit 链。
- Do not misstate: scaffold、静态 tests、计划、200-step smoke、单 seed 或 loss 下降都不是方法效果；repo 记录不是 live 服务器盘点。
- Requires confirmation: pull/fetch/push/commit/branch/worktree；任何外部数据下载/许可接受；数据移动/删除/转换；forward/backward/optimizer/训练；仿真、真机和材料发布。
- Unknowns: 4090 clone/data/checkpoint 路径、dirty changes 所有权、服务器依赖与容量、数据许可/完整性、旧 checkpoint provenance、H1 是否有效。
- Cross-stream risk: 4090 只负责数据/训练/离线；不要把新离线结果直接写成 Ubuntu 闭环或真机证据。Windows 回收结果后才更新论文 go/no-go。

# 10. Raw Notes

- Windows review source base: `d8a238e169a626cae7eb69fe40e18421ea628162`。
- Delivery branch: `agent/ubuntu-sim-handoff` tracking `origin/agent/ubuntu-sim-handoff`。
- Resolve delivery commit after checkout: `git rev-parse HEAD`；确认该 commit 包含本 handoff。
- Pre-training backlog: `scripts/research/pretraining/pretraining_backlog_v0.1.yaml`。
- Smoke contract: `scripts/research/pretraining/smoke_config_v0.1.yaml`。
- 4090 plan: `scripts/research/training_plans/tinynavbrain_4090_plan.yaml`。
- Fair baselines: `后续研究内容/model/baseline_matrix_v0.1.md`。
- Data runbook: `后续研究内容/external_execution/01_data_pilot.md`。
- Training order: `后续研究内容/external_execution/02_training_run_order.md`。
- Current paper gate: `后续研究内容/paper/go_no_go_v0.1.md` — submission `NO-GO`。
