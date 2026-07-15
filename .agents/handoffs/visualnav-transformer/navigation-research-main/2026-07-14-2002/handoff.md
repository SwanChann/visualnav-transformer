# Historical packet — superseded for 4090 execution

> 本包保留 2026-07-14 的主线交接事实，但不再是 4090 执行入口。其中的
> `residual flow` 等训练术语已被后续合同修正。4090 环境必须改用
> `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md`，
> 不得从本历史包直接启动数据获取或训练。

# 1. Copyable Prompt For New Conversation

```text
请使用 $context-handoff 导入本交接包。目标 handoff_id 是 `visualnav-transformer/navigation-research-main/2026-07-14-2002`，文件位于 `.agents/handoffs/visualnav-transformer/navigation-research-main/2026-07-14-2002/handoff.md`。先在 `F:\codespace\visualnav-transformer` 运行 project snapshot、git status、git worktree list，并核对 `PROJECT_ASSET_MAP.md`、任务队列和权威 U09/U10 结果；然后复述“已验证事实 / 推断 / 未验证假设 / 投稿硬门”。不要重复 Ubuntu U00–U18，不要覆盖 U09 v0.3 或 U10 v5，不要把 scaffold/计划写成训练结果。下一阶段聚焦：把多数据集训练与 TinyNavBrain 探索转成可执行、可否证的 pre-training backlog；先完成无需实际训练的本地基础设施和测试，再由用户决定外部数据获取与训练执行。任何 commit、push、worktree 删除或远端操作均需按用户当轮授权执行；最终人工入口始终为原项目目录和 `PROJECT_ASSET_MAP.md`。
```

# 2. Handoff Identity

- `handoff_id`: `visualnav-transformer/navigation-research-main/2026-07-14-2002`
- `created_at`: `2026-07-14T20:02:42+08:00`
- `source_conversation`: 2026-07-10 起的长 Codex 项目线程；包含 Windows 主控、Ubuntu 离线/仿真回收、Volc workers 只读协作、Git 收拢和训练路线审查
- `agent_role`: `navigation-research-main-agent`
- `workstream`: RA-L 研究主线、证据治理、多数据集训练准备、TinyNavBrain 候选方法与后续实验晋级
- `scope`: 项目根目录；重点为 `.agents/`、`PROJECT_ASSET_MAP.md`、`results/research/`、`scripts/analysis/`、`scripts/experiments/`、`scripts/research/`、`后续研究内容/`、`投稿冲刺/workspace/`
- `not_scope`: 未经授权的模型训练、付费数据下载、真机运行、自动真机 outcome 标注、破坏性 Git 清理，以及把计划/静态测试当作方法效果证据
- `related_handoffs`: `visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`；`.agents/workstreams/navigation-research-main/handoff.md`

# 3. Project Overview

- Project root: `F:\codespace\visualnav-transformer`
- Project purpose: 基于 NoMaD/ViNT 的视觉目标导航与 Lite3 部署；当前研究主线是统一 benchmark、多数据集训练和小型低 NFE 生成式导航 policy。
- Canonical human entry: `PROJECT_ASSET_MAP.md`。不要把临时 worktree 当成人工入口。
- Current Git: `agent/ubuntu-sim-handoff` @ `d8a238e169a626cae7eb69fe40e18421ea628162`；tracking `origin/agent/ubuntu-sim-handoff`；交接创建前 clean；仅一个注册 worktree。
- Important instructions: 遵守用户的 Global Epistemic Discipline；把用户和 agent 的主张当假设，区分已验证事实、证据推断、推测和价值判断。
- Relevant local skills: `$context-handoff`、`$codex-volc-orchestrator`、`$codex-volc-worktree-workflow`。后两者已在本机更新为低频等待、任务交付契约前置、final-structured-report-only；worker 禁止返回 reasoning/JSON event stream/完整命令日志。

# 4. Previous Conversation Summary

- Durable goal: 客观判断并推进 RA-L 可投稿工作；不迎合预期，不伪造结果；把 benchmark、多数据集训练、框架/范式改进和小型生成式导航大脑结合成可否证研究。
- Important decisions:
  - Ubuntu 授权 U00–U18 已完成，但整篇论文仍为 submission `NO-GO`。
  - 当前工程/复现质量较高，科学证据与创新性证明较弱。此前审计估计：协作约 7.6/10；工程约 85%；研究证据约 35%；草稿约 45%；投稿约 5–10%。这些是审计判断，不是客观测量。
  - 需要训练才能成为方法论文，但不能立刻大规模训练。必须先完成合法多数据集、统一 meter-space contract、泄漏安全 split、训练 smoke 和冻结评测。
  - 不需要推倒重写框架。优先补齐 image/dataset adapter、action history、sampler、loss/train/checkpoint/export 和 backend 接入；更需要重建非饱和仿真 benchmark。
  - 必须训练受控最小 baseline 矩阵：单数据集 specialists、Multi-NoMaD proportional/balanced、H0 deterministic、低 NFE DDIM、H1 residual flow、scale/action-history ablations。DDPM/DDIM/CFG/TTS 多数是推理配置，不应伪装成独立训练 policy。
  - 正确顺序：研究问题与数据/评测协议冻结 → 训练 smoke → baseline → H1 → 离线 IID/mixed/LODO/corruption → 非饱和仿真 → 目标设备 sustained timing → 协议化真机 → 论文冻结。
  - 用户要求 agent 与 Git 联动；临时 worktree 只作施工区，验证后应在获得授权时收拢回原项目目录、push 并退休。
- Files created/changed in the long thread: 以 `PROJECT_ASSET_MAP.md` 和 Git commits `a49d372`、`ac49e99`、`d8a238e` 为索引。交接包本身是本轮新增文件，尚未 commit/push。
- Local skill changes outside this repository: `C:\Users\Modes\.agents\skills\codex-volc-{orchestrator,worktree-workflow}\` 已更新并通过 quick validation；不属于项目 Git。

# 5. Current Implementation State

## Already implemented and verified

- 权威受控 MuJoCo：`results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/`，60/60 records，25 success、35 failure、0 crash/invalid/missing。
- 权威独立离线：`results/research/ral_offline_independent/u10-independent-20260711-v5/`，15 cases × 5 configs = 75，invalid=0，GO Stanford scale 0.12 m。
- 141-file reproducibility checksum bundle、分析重生成、论文 Table III 和 3 张 controlled-result PDF。
- Policy-only single realization: DDPM10 4/6；DDIM2 5/6；DDIM3 5/6；DDIM2/TTS8 5/6；DDIM2/CFG2/TTS8 6/6。
- Stabilizer-on 每配置 0/6，且同 scene/seed 跨方法轨迹 byte-identical，只能视为 controller override 行为。
- Exact repeat 保持二元 outcome，但共同轨迹最大差异 0.4439 m；仅 seeded-stochastic 描述性证据。
- Offline ADE vs policy-only success: rho=-0.67，exact p=0.30；不支持离线指标预测闭环优越。
- 数据 registry/manifest/leakage audit、benchmark schema/validator、NavigationPolicyBackend、TinyNavBrain feature scaffold 和静态训练计划已存在。

## Partially implemented

- TinyNavBrain 只有 architecture contract、feature-level scaffold、shape tests 和机器可读计划；缺完整图像 adapter、dataset adapter、loss/train loop、checkpoint、部署导出。
- 训练与多数据集执行 runbook 已写，但仅 Go Stanford 完整 materialized。
- 当前仿真可审计，但 2 scenes × 3 seeds 且 hard 被 gate 淘汰，不足以验证新方法。

## Not implemented

- RECON/HuRoN 等至少三个合法、完整、trajectory-disjoint domain。
- Multi-NoMaD、H0、H1 的多 seed 训练和公平对比。
- 新的非饱和 procedural/variant closed-loop benchmark。
- 指定目标设备 sustained deadline/miss、功耗/温度测试。
- 协议化重复真机 outcome 与隐私/同意审查。

# 6. Current Task State

- Active task: 长线程收口并为下一线程移交训练准备/研究决策。
- Last completed step: 回收 Ubuntu GitHub 成果；建立资产地图；收拢到原工作区；push 到 `d8a238e`；审查训练必要性和正确实验顺序；更新本地 agent skills。
- Current blocker: 本地环境不能执行训练、仿真和真机；外部训练需数据和算力授权/环境。Ubuntu 可做离线与仿真，但训练未授权/未执行。
- Evidence gathered: `PROJECT_ASSET_MAP.md`、`.agents/workstreams/navigation-research-main/TASKS.md`、`results/research/ubuntu_final_audit/`、`后续研究内容/{03_单卡4090训练路线.md,05_policy后端与小模型研究.md,model/,paper/}`。
- Context reason for handoff: 当前长线程曾有 58 次 active prompt >272K，峰值 312,521；3 次 compaction；交接前最近约 200K。另一个项目相关线程有 4 次 >272K。累计 token 不是阈值依据。

# 7. Next Actions

1. 新线程先重建状态：snapshot、Git、资产地图、TASKS、权威 U09/U10；确认交接包新增导致的 dirty 状态，不要误删。
2. 将训练探索转成新的机器可验收 backlog，明确 `DATA-PILOT → TRAIN-INFRA → B0 smoke → baseline matrix → H1 → offline gate → sim gate → device/robot gate`。
3. 优先完成无需训练的本地代码：canonical dict batch adapter、manifest metadata 接入、action-history 构造与泄漏测试、sampler contracts、H0/H1 train-step contract、checkpoint/resume schema、smoke config 和单元测试。
4. 在看到任何 H1 结果前，设计并冻结新的非饱和仿真 benchmark；执行可留给 Ubuntu。
5. 数据外部任务：合规获取 RECON/HuRoN pilot，生成 receipt/license snapshot/manifest/group-safe split；少于 3 domains 时禁止 cross-dataset claim。
6. 获得外部训练条件后先做 200-step smoke；再按至少 3 seeds 执行 baseline，只有 baseline 稳定后训练 H1。
7. 修正治理债务时保持历史：`final_audit.json` 是旧 Git 快照；`log.md` 有旧 U10 数值。应增加 current-state overlay/说明，不得篡改历史实验事实。
8. 用户确认后再 commit/push 本交接包或后续实现；远端操作默认关闭。

# 8. Verification

- Checks already run after Ubuntu integration: controlled runner 8/8；controlled results 6/6；data 11/11；benchmark 7/7；policy backend 5/5；training plan 4/4；`git diff --check` pass。
- Handoff snapshot: `PYTHONUTF8=1 python C:\Users\Modes\.agents\skills\context-handoff\scripts\project_snapshot.py --root .` 可在 Windows 正常处理中文；未设置 UTF-8 时脚本可能因 GBK 输出失败。
- Checks still needed: 校验本交接文件结构、重新运行 `git status`、确认没有其他本轮文件变化。
- Future training success: 不是 loss 能下降，而是固定数据/encoder/steps/seed/compute 下 H1 超过 H0，收益超过 seed variance，并转化到冻结非饱和闭环。

# 9. Risks And Guardrails

- Do not repeat: U09 v0.2 action-scale 错误、U10 v2 0.25 m 尺度错误、把 stabilizer-on 当五个独立 policy、把 inference sampler 配置当独立训练模型。
- Do not undo: U09 v0.3/U10 v5、作废标记、历史审计链、用户原始文件、权威资产地图。
- Requires confirmation: commit、push/pull/fetch、branch/worktree 删除、外部数据下载、训练、仿真执行、真机、隐私材料发布。
- Unknowns: 可用训练机器/4090 的真实环境和吞吐；RECON/HuRoN 许可/下载完整性；新 benchmark 的区分度；H1 是否提供任何方法收益。
- Epistemic guardrail: 工程测试通过不等于方法有效；Ubuntu scope 100% 不等于项目/论文 100%；负结果必须保留。
- Context guardrail: 建议在全局 Codex config 设置 `model_auto_compact_token_limit = 210000` 和 `model_auto_compact_token_limit_scope = "total"`，把 220K 视为上限而不是目标；不要把 `model_context_window` 伪装成硬上限。另保持有限的 `tool_output_token_limit` 和 final-only worker transport。该建议尚未写入配置。

# 10. Raw Notes

- Current canonical HEAD/remote: `d8a238e169a626cae7eb69fe40e18421ea628162`。
- Current branch: `agent/ubuntu-sim-handoff`。
- Current paper verdict: engineering/research package可继续发展，RA-L method submission `NO-GO`。
- Primary entry: `PROJECT_ASSET_MAP.md`。
- Task ledger: `.agents/workstreams/navigation-research-main/TASKS.md`。
- Current handoff: `.agents/workstreams/navigation-research-main/handoff.md`。
- Training plan: `scripts/research/training_plans/tinynavbrain_4090_plan.yaml`。
- Fair baselines: `后续研究内容/model/baseline_matrix_v0.1.md`。
- Go/no-go: `后续研究内容/paper/go_no_go_v0.1.md`。
