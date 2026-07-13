# Navigation Research Task Queue

状态：`ready` / `in_progress` / `blocked_external` / `done` / `stopped`。

## 已完成

| ID | 状态 | 任务 | 证据/交付物 |
|---|---|---|---|
| GOV-01 | done | 主控、worktree、worker 与审计治理 | `AGENT.md`、本文件、`log.md` |
| RAL-01..04 | done | 既有证据、离线 Pareto、历史 MuJoCo、真机 timing 与标注入口整理 | `results/ral_offline/`、`results/ral_mujoco_summary/`、`results/real_robot_timing/` |
| DATA-01..03 | done | 数据源/许可审计、manifest 与泄漏检查、外部执行收据 | `后续研究内容/data/`、`scripts/research/data/`；3 个核心 manifest/split 工具，另有 1 个 artifact receipt 工具 |
| BENCH-01 | done | 诊断 benchmark v0.1 | `后续研究内容/benchmark/`、`scripts/research/validate_benchmark_result.py` |
| MODEL-01..02 | done | TinyNavBrain 合同、baseline 与 feature-level scaffold | `后续研究内容/model/`、`scripts/research/tinynavbrain/` |
| PAPER-01..02 | done | 当前/后续 RA-L 证据包、claims 与 go/no-go | `投稿冲刺/workspace/`、`后续研究内容/paper/` |
| VIDEO-01 | done | 20 个历史真机视频的辅助审查 | 只证明可见运动；14 个正式 outcome 仍需人工按协议判定 |
| U00..U18 | done | Ubuntu 环境、离线实验、受控 MuJoCo、分析、论文资产、复现、纠错、后端探索、静态训练计划 | `results/research/ubuntu_final_audit/`；U09 v0.3 60/60、U10 v5 75/75、141 文件 checksum bundle |
| EXP-02 | done | 冻结协议下的闭环仿真 benchmark | `results/research/ral_mujoco_controlled/u09-v03-controlled-20260711/`；2 场景×3 seeds×5 configs×2 strata，60/60 有效 |
| CURATE-01 | done | GitHub/Ubuntu 成果回收、统一资产地图与任务状态校正 | 根目录 `PROJECT_ASSET_MAP.md` |
| PORT-01 | done | Windows CRLF 下协议哈希与 crash 单测可移植性 | 文本协议按 LF 规范化哈希；二进制输入仍按原始字节哈希 |

## 当前真实阻塞项

| ID | 优先级 | 状态 | 任务 | 验收标准 |
|---|---:|---|---|---|
| DATA-04 | P0 | blocked_external | 获取并合规登记 RECON/HuRoN 小样本 | 原始 artifact、许可快照、receipt、group-safe manifest 均可审计 |
| EXP-01 | P0 | blocked_external | 在统一多数据集协议下训练 baseline 与 TinyNavBrain | 多 seed、固定预算、完整日志、失败记录、模型与数据 provenance |
| DEVICE-01 | P0 | blocked_external | 指定目标设备的 sustained deadline/miss timing | 明确设备、热身、持续时长、deadline、miss rate、功耗/温度口径 |
| EXP-03 | P0 | blocked_external | 协议化重复真机闭环验证 | 安全协议、原始日志、人工 outcome、成功率/效率/失败模式 |
| PRIV-01 | P0 | blocked_external | 真机数据隐私/同意审查 | 场地、人员、视频发布与匿名化记录完备 |
| PAPER-03 | P0 | blocked_external | RA-L 最终投稿门 | 上述外部门全部闭合，主张、图表和统计重新冻结并复核 |

## 执行顺序

Ubuntu 授权范围 U00–U18 已完成，本地整理也已完成。后续顺序是：

`RECON/HuRoN 数据试点 → 多数据集训练 → 目标设备持续 timing → 重复真机实验 → 隐私/同意 → 最终论文冻结`

当前论文可继续写作和内部审稿，但仍为 **submission NO-GO**。不能用更多同类仿真替代目标设备和真机硬门。
