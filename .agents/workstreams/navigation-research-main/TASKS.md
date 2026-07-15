# Navigation Research Task Queue

状态：`ready` / `in_progress` / `blocked_external` / `done` / `stopped`。

## 已完成

| ID | 状态 | 任务 | 证据/交付物 |
|---|---|---|---|
| GOV-01 | done | 主控、worktree、worker 与审计治理 | `AGENT.md`、本文件、`log.md` |
| RAL-01..04 | done | 既有证据、离线 Pareto、历史 MuJoCo、真机 timing 与标注入口整理 | `results/ral_offline/`、`results/ral_mujoco_summary/`、`results/real_robot_timing/` |
| DATA-01..03 | done | 数据源/许可审计、manifest 与泄漏检查、外部执行收据 | `后续研究内容/data/`、`scripts/research/data/`；3 个核心 manifest/split 工具，另有 1 个 artifact receipt 工具 |
| BENCH-01 | done | 诊断 benchmark v0.1 | `后续研究内容/benchmark/`、`scripts/research/validate_benchmark_result.py` |
| MODEL-01..02 | done | TinyNavBrain 合同、baseline 与 feature-level scaffold | `后续研究内容/model/`、`scripts/research/models/` |
| PAPER-01..02 | done | 当前/后续 RA-L 证据包、claims 与 go/no-go | `投稿冲刺/workspace/`、`后续研究内容/paper/` |
| VIDEO-01 | done | 20 个历史真机视频的辅助审查 | 只证明可见运动；14 个正式 outcome 仍需人工按协议判定 |
| U00..U18 | done | Ubuntu 环境、离线实验、受控 MuJoCo、分析、论文资产、复现、纠错、后端探索、静态训练计划 | `results/research/ubuntu_final_audit/`；U09 v0.3 60/60、U10 v5 75/75、141 文件 checksum bundle |
| EXP-02 | done | 冻结协议下的闭环仿真 benchmark | `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/`；2 场景×3 seeds×5 configs×2 strata，60/60 有效 |
| CURATE-01 | done | GitHub/Ubuntu 成果回收、统一资产地图与任务状态校正 | 根目录 `PROJECT_ASSET_MAP.md` |
| PORT-01 | done | Windows CRLF 下协议哈希与 crash 单测可移植性 | 文本协议按 LF 规范化哈希；二进制输入仍按原始字节哈希 |
| TRAIN-INFRA-01 | done | 冻结 pre-training canonical batch、past-only action history、sampler、H0/H1 loss 与 checkpoint/resume 合同 | `scripts/research/pretraining/`；仅静态合同与单元测试，不是训练结果 |
| TRAIN-INFRA-02 | done | 建立可执行、可否证的 pre-training backlog 与 200-step smoke 配置 | `scripts/research/pretraining/pretraining_backlog_v0.1.yaml`、`smoke_config_v0.1.yaml` |
| ENV-01 | done | 冻结 Windows、Ubuntu、4090 的职责、数据驻留和跨环境交付链 | `ENVIRONMENT.md`、`PROJECT_ASSET_MAP.md` |
| HANDOFF-4090 | done | 导出 4090 pre-training 执行交接，明确先 snapshot、后授权数据/训练 | `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md` |

## 当前真实阻塞项

| ID | 优先级 | 状态 | 执行环境 | 任务 | 验收标准 |
|---|---:|---|---|---|---|
| DATA-04 | P0 | blocked_external | 4090 | 盘点并合规登记 RECON/HuRoN；数据不离开 4090 | 原始 artifact、许可快照、receipt、group-safe manifest 均可审计 |
| EXP-01 | P0 | blocked_external | 4090 | 真实 adapter 接线、B0 smoke、多数据集 baseline 与 TinyNavBrain 训练及离线门 | 多 seed、固定预算、完整日志、失败记录、模型与数据 provenance |
| SIM-01 | P0 | blocked_external | Ubuntu | 对通过离线门的 checkpoint 执行新的非饱和闭环仿真 | 冻结 scenes/seeds/bridge，结果不饱和，原始 rollout 可审计 |
| DEVICE-01 | P0 | blocked_external | 待指定目标设备 | 指定目标设备的 sustained deadline/miss timing | 明确设备、热身、持续时长、deadline、miss rate、功耗/温度口径 |
| EXP-03 | P0 | blocked_external | 待指定真机环境 | 协议化重复真机闭环验证 | 安全协议、原始日志、人工 outcome、成功率/效率/失败模式 |
| PRIV-01 | P0 | blocked_external | Windows 主控 + 真机环境 | 真机数据隐私/同意审查 | 场地、人员、视频发布与匿名化记录完备 |
| PAPER-03 | P0 | blocked_external | Windows | RA-L 最终投稿门 | 上述外部门全部闭合，主张、图表和统计重新冻结并复核 |

4090 的 Go Stanford 单域 live pilot 已通过：3696 轨迹、198126 图片全量可读，
group-safe manifest 覆盖完整，真实 canonical adapter/DataLoader 可产生米制 batch；
train split 有 131950 个可用 sample，冻结 5% 子集为 6598。该结果不改变
`DATA-04` / `EXP-01` 的 `blocked_external` 状态；RECON/HuRoN 未授权，且未执行训练。

`IMAGE-FORWARD` gate 已在 GPU 0 通过：随机初始化且不下载权重的共享
EfficientNet-B0 image policy 共 8513117 个可训练参数，合成数据测试与真实 batch 的
H0/H1 inference-only 前向均通过。当前可以实现 train loop，但 loss/train-step、
checkpoint exact-resume、Git 冻结和任何 backward/optimizer/B0 仍需后续门控。

## 执行顺序

Windows 静态实现与 Ubuntu U00–U18 均已完成。4090 必须先检出包含 `HANDOFF-4090` 的冻结 Git SHA，后续顺序是：

`4090 环境/数据 snapshot → DATA-PILOT → B0 smoke → baseline/H1 → 4090 离线门 → Ubuntu 仿真门 → 目标设备/真机 → Windows 论文冻结`

当前论文可继续写作和内部审稿，但仍为 **submission NO-GO**。不能用更多同类仿真替代目标设备和真机硬门。

Pre-training 的细粒度依赖、验收证据和否证条件以
`scripts/research/pretraining/pretraining_backlog_v0.1.yaml` 为准。静态基础设施完成后，
`DATA-PILOT`、`B0-SMOKE`、`BASELINE-MATRIX`、`H1-TRAIN` 仍需用户分别授权外部数据与训练执行。
完整数据集只驻留 4090；Ubuntu 不承担新的跨数据集离线评估，Windows 不承担真实数据验证或训练。
