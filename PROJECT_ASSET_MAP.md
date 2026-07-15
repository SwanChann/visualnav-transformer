# 项目资产地图

> 统一入口，更新于 2026-07-16。当前已推送基线：`agent/ubuntu-sim-handoff@a537c3e2a079fe094b5a5c5dba15b83120bc7d4f`。

## 1. 当前结果一览

| 结果 | 状态 | 权威位置 |
|---|---|---|
| Ubuntu U00–U18 审计 | 完成 | `results/research/ubuntu_final_audit/final_audit.md`、`final_audit.json` |
| 受控 MuJoCo 主矩阵 | 完成、描述性 | `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/` |
| 独立离线主评估 | 完成、描述性 | `results/research/ral_offline_independent/u10-independent-20260711-v5/` |
| 受控统计与关联分析 | 完成 | U09 v0.3 下的 `analysis/` |
| 论文 Table III 与 3 张 PDF | 完成 | `投稿冲刺/workspace/controlled_results/` |
| RA-L 写作包 | 可继续写作；投稿 NO-GO | `投稿冲刺/workspace/`、`后续研究内容/paper/` |
| 4090 Go Stanford live DATA-PILOT | 内容/adapter/DataLoader 通过；仅单域，未运行模型 | `results/research/data_pilot/go_stanford_live_20260715/` |
| TinyNavBrain IMAGE-FORWARD | GPU 0 inference-only 通过；未 backward/训练 | `scripts/research/models/tinynavbrain_image_policy.py`、`results/research/data_pilot/go_stanford_live_20260715/image_forward_readiness.md` |
| TinyNavBrain TRAIN-STEP | 两步集成与 exact-resume 通过；不是 B0 | `scripts/research/pretraining/train_step_runtime.py`、`results/research/pretraining/train_step_gate_20260716/readiness.md` |
| TinyNavBrain Go Stanford B0 | H0/H1 各 200-step plumbing smoke 与 exact-resume 通过；无评测/效果主张 | `results/research/pretraining/b0_go_stanford_v0.1/b0_audit.md` |
| RECON/HuRoN local DATA-PILOT | 授权工作区内 0/2 materialized；只有处理器，数据/receipt/manifest/许可快照均缺失 | `results/research/data_pilot/recon_huron_presence_20260716/audit.md` |
| 多数据集训练 | 未执行，外部阻塞 | `后续研究内容/external_execution/02_training_run_order.md` |
| 目标设备/协议化真机 | 未执行，外部阻塞 | `后续研究内容/external_execution/03_sim_robot_promotion.md` |

U09 v0.3 共 60 个计划/记录：25 success、35 failure、0 crash、0 infrastructure-invalid、0 missing。U10 v5 共 75/75 valid。文件级可复现清单在 `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/reproducibility/checksums.json`，共 141 个条目。

## 2. 原始结果与分析资产

| 资产类别 | 位置 | 使用规则 |
|---|---|---|
| U09 原始 trial JSON | `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/trials/` | 闭环主证据，不修改 |
| U09 batch/plan | 同目录 `batch_manifest.json`、`selected_plan.json` | 矩阵完整性与 provenance |
| U09 runtime/timing | 同目录 `runtime/` | 仅对应当前 Ubuntu/硬件/runner 口径 |
| U09 分析 | 同目录 `analysis/` | episode、分层汇总、paired effects、robustness、轨迹等价、离线关联 |
| U09 复现说明 | 同目录 `REPRODUCE.md` | 重生成分析而非重写原始 trial |
| U10 case 数据 | `results/research/ral_offline_independent/u10-independent-20260711-v5/` | 离线主证据；物理尺度 0.12 m |
| 历史离线 Pareto | `results/ral_offline/` | Tier A 历史证据，不与 U09/U10 混算 |
| 历史 MuJoCo 汇总 | `results/ral_mujoco_summary/`、`results/nomad_mujoco/` | 历史/诊断证据，不进入受控主矩阵 |
| 历史真机 timing/视频 | `results/real_robot_timing/`、`results/real_robot_video_audit/` | timing 可报告；outcome 未按冻结协议确认 |

## 3. 论文呈现资产

| 资产 | 位置 |
|---|---|
| Table III 数据/Markdown | `投稿冲刺/workspace/controlled_results/table_iii.csv`、`table_iii.md` |
| 闭环成功率图 | `投稿冲刺/workspace/controlled_results/fig_controlled_success.pdf` |
| latency/SPL 图 | `投稿冲刺/workspace/controlled_results/fig_controlled_latency_spl.pdf` |
| 轨迹图 | `投稿冲刺/workspace/controlled_results/fig_controlled_trajectories.pdf` |
| 当前 evidence package | `投稿冲刺/workspace/` |
| claims ledger / figure-table plan / go-no-go | `后续研究内容/paper/` |
| 最终 Ubuntu 审计 | `results/research/ubuntu_final_audit/` |

## 4. 协议、代码与研究设计

| 资产类别 | 位置 |
|---|---|
| 冻结 benchmark/protocol/schema/metric | `后续研究内容/benchmark/`；主协议为 `ral_mujoco_protocol_v0.3.json` |
| 受控 MuJoCo runner | `scripts/experiments/ral_mujoco_controlled.py` |
| 独立离线 runner | `scripts/experiments/ral_offline_independent.py` |
| RA-L 分析/校验/复现工具 | `scripts/analysis/` |
| 数据 registry、manifest、split、receipt | `后续研究内容/data/`、`scripts/research/data/` |
| Go Stanford 真实 adapter / 内容审计 | `scripts/research/pretraining/go_stanford_adapter.py`、`audit_go_stanford_pilot.py` |
| policy backend 探索 | `scripts/research/policy_backend/`、`后续研究内容/05_policy后端与小模型研究.md` |
| TinyNavBrain 合同/scaffold/image policy/训练计划 | `后续研究内容/model/`、`scripts/research/models/`、`scripts/research/training_plans/` |
| Pre-training backlog/本地合同测试 | `scripts/research/pretraining/`；仅静态基础设施，不是训练结果 |
| 后续研究总览 | `后续研究内容/README.md`、`01_后续研究总览.md` 至 `04_RA-L后续研究投稿规划.md` |
| 外部执行 runbook | `后续研究内容/external_execution/` |

预训练 checkpoint `nomad.pth` 由 `.gitignore` 排除，因此新 detached worktree 默认没有该文件；冻结期望 SHA-256 以协议/审计记录为准（`66edf390…`）。执行新的模型推理前必须恢复并完整核验，不能把“文件缺失”误记为科学失败。

## 5. 作废、替代与仅审计资产

| 位置 | 状态/原因 |
|---|---|
| `results/research/ral_mujoco_controlled/u09-v02-controlled-20260711/` | 作废为主证据；缺少正确 action scale，保留审计 |
| `results/research/ral_mujoco_controlled/u09-v02-repeat-audit-20260711/` | 仅重复性审计 |
| `results/research/ral_offline_independent/u10-independent-20260711-v2/` | 作废；错误使用 0.25 m 尺度 |
| U10 未版本目录、v3、v4 | 被 v5 替代或为失败中间产物 |
| U08 v0.2 及 U06/U07 smoke | gate/smoke 证据，不是最终主结果 |
| `results/day*`、旧 benchmark/nomad_mujoco | 历史 Tier A，不与受控 RA-L 结果聚合 |

所有下游论文数字只允许绑定 U09 v0.3 与 U10 v5。历史目录保留以维持审计链，不删除。

## 6. 治理与任务

| 资产 | 位置 |
|---|---|
| 当前任务队列 | `.agents/workstreams/navigation-research-main/TASKS.md` |
| 三环境职责与执行链 | `ENVIRONMENT.md`；Windows 主控、4090 数据/训练/离线、Ubuntu 晋级仿真 |
| 4090 pre-training 交接 | `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md`；先 live snapshot，数据获取与训练仍需另行授权 |
| 工作日志 | `.agents/workstreams/navigation-research-main/log.md` |
| 当前交接 | `.agents/workstreams/navigation-research-main/handoff.md` |
| 旧证据索引 | `results/evidence_index.md`（只覆盖早期 Tier A；以本文件为总入口） |

## 7. 口径与剩余门

- 文本协议的内容身份按 UTF-8、LF 规范化后计算 SHA-256，以消除 Windows CRLF checkout 差异；checkpoint、数据和其他二进制资产仍按原始字节计算。
- stabilizer-on 跨方法轨迹相同，不构成五个独立策略的导航证据。
- 当前 counts 与相关性仅为 seeded-stochastic、描述性结果，不支持稳定排名、显著优越或真机泛化主张。
- Ubuntu 范围已完成；4090 对两个授权工作区的只读盘点确认 RECON/HuRoN 为 0/2 materialized，`recon_datavis` 只是可视化源码。全项目尚缺 RECON/HuRoN 原始 artifact、许可快照、receipt、manifest/split、多数据集训练、目标设备持续 timing、协议化重复真机 outcome 和隐私/同意。RA-L 当前可写但不可提交。
- 完整数据集只驻留 4090；Go Stanford 已在 4090 完成 3696 轨迹 / 198126 图片的内容级审计，但 raw receipt、per-frame timestamp 和 processor version 仍缺失。真实数据接线、训练和跨数据集离线评估不得分配给 Windows/Ubuntu。环境职责以 `ENVIRONMENT.md` 为准。
