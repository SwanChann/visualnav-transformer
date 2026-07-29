# 三环境统一工作进度

更新日期：2026-07-29

统一人工入口：`PROJECT_ASSET_MAP.md`

活跃分支：`agent/ubuntu-sim-handoff`

4090 进展回收基线：`9fb58e7e1c03d5813bf580ad91d25680f5999a72`

## 1. 状态口径

三个环境统一使用下列状态。只有证据进入权威 Git 仓库并在 Windows 原项目目录复核后，
项目总进度才随之更新。

| 状态 | 含义 | 进入条件 |
|---|---|---|
| `done_verified` | 已完成且证据可在权威仓库复核 | 有 Git SHA、输入/配置身份、产物和验证记录 |
| `done_not_promoted` | 有限 gate 已通过，但不足以晋级或支撑方法主张 | 明确保留限制、缺口与下一门 |
| `ready` | 本环境前置条件已满足，可以等待执行授权 | 依赖门已通过，执行位置和验收标准明确 |
| `in_progress` | 正在执行 | 有环境、负责人、开始时间和任务 ID 回执 |
| `blocked_external` | 受其他环境、数据、硬件或用户授权阻塞 | 阻塞条件和解除条件明确 |
| `unknown_no_receipt` | 可能在外部环境做过，但权威仓库没有回执 | 不允许据此宣称开始、完成或失败 |
| `stopped` | 按否证/止损规则停止 | 保留失败证据和停止原因 |

计划、scaffold、静态 shape test、配置校验、单 seed、loss 下降和短 smoke 都不能直接
记为模型有效或训练完成。训练/评测结论必须有原始日志、provenance、冻结预算和对照结果。

## 2. 三环境当前总览

| 环境 | 已完成且已验证 | 当前阶段 | 当前状态 | 下一动作 | 权威证据 |
|---|---|---|---|---|---|
| Windows | Git/论文/证据治理；三环境职责；pre-training 静态合同与测试；已审查并吸收 4090 的 12 个提交 | 维护统一总账、审查外部回执和论文资产 | 已提交范围 `done_verified`；本轮统一文档待提交 | 提交并推送本轮整理；随后等待 4090 新回执 | 83/83 当前研究基础设施 tests、3/3 validators、compileall；本轮日志 |
| Ubuntu | U00–U18；U09 v0.3 60/60；U10 v5 75/75；141-file 复现清单 | 等待通过 4090 `OFFLINE-GATE` 的 checkpoint | 历史授权范围 `done_verified`；后续 `blocked_external` | 暂停；收到 checksum-bound checkpoint 后才建立新的非饱和 `SIM-GATE` | `results/research/ubuntu_final_audit/`、权威 U09/U10 目录 |
| RTX 4090 | Phase 0；Go Stanford 内容/adapter；IMAGE-FORWARD；两步 TRAIN-STEP；单域 H0/H1 各 200-step B0；RECON/HuRoN raw 与 conversion pilots；2026-07-29 只读复核 | 恢复 GPU live gate并决定数据晋级边界 | 已完成 gates 为 `done_verified` 或 `done_not_promoted`；新 CUDA 与多数据集训练 `blocked_external` | 先恢复 `/dev/nvidia*` 与 driver communication；再处理或接受 RECON/HuRoN metadata/holdout 限制 | `results/research/4090_progress_audit/20260729/` 及其引用的 receipts |

### 已验证事实

- Windows 已执行 `git fetch --all --prune`，确认远端同名分支从 `ffa3533` 前进到
  `9fb58e7`，并以 fast-forward 吸收；该范围未修改权威 U09 v0.3/U10 v5 结果。
- 集成后的 pretraining、training-plan、models、data、policy backend 和 benchmark
  validator 共 83 个不同测试全部通过；三个配置 validator 与 `compileall` 通过。
  历史 analysis 测试另有 20/24 通过，4 个只因 Windows 缺完整 NoMaD 依赖而无法导入。
- 4090 审计记录的软件环境为 Python 3.8.5、PyTorch 2.4.1+cu121、torchvision
  0.19.1+cu121；历史执行时有 2×RTX 4090 收据。2026-07-29 当前状态是 NVIDIA
  580.173.02 模块已加载，但 `/dev/nvidia*` 缺失、`nvidia-smi` 失败、PyTorch CUDA
  不可用且可见设备数为 0。
- Go Stanford 已审计 3696 条轨迹、198126 张图片、train split 131950 个可用 sample，
  冻结 5% 子集 6598。IMAGE-FORWARD、两步 TRAIN-STEP/exact-resume、H0/H1 各
  200 optimizer steps 的 B0 plumbing smoke 有回执。
- RECON/HuRoN 原始 pilot 和隔离转换存在且完整性审计通过；RECON 为 14 帧/1 window，
  HuRoN 为 66 帧/53 windows。两者仍为 `promotion_ready=false`。
- Ubuntu 的完成范围仍只到 U00–U18；没有新的晋级 checkpoint、非饱和仿真或真机回执。

### 证据支持的推断

- 当前不是继续补 Windows 基础设施的时机；新的可执行主线由 4090 的 GPU live gate 和
  数据晋级决定。
- 历史 GPU receipts 证明当时完成了有限 plumbing gates，但不证明 2026-07-29 GPU 可用，
  也不证明训练收敛、H1 优于 H0、跨数据集泛化或论文方法有效。

### 未验证假设

- 修复 device nodes/driver communication 后，两张 4090 是否能在当前环境稳定执行完整
  多数据集训练，尚未验证。
- RECON 的运动学推断 dt、缺失 collection policy/version 和 train-only pilot 是否可被
  论文/实验设计接受，尚未由用户决定；也没有独立评测 holdout。
- 142 GB 历史 `train/logs` 尚无逐 checkpoint provenance，不能并入当前证据链。

## 3. 跨环境主链

| Gate | 执行环境 | 统一状态 | 通过或解除条件 | 证据 |
|---|---|---|---|---|
| G0 Windows freeze | Windows | `done_verified` | 代码、协议、配置、handoff 和静态测试冻结 | `ffa3533` 及之前提交 |
| G1 4090 checkout + live snapshot | 4090 | `done_verified` | full SHA/dirty、软件/GPU/磁盘、指定数据/checkpoint 清单 | `results/research/4090_progress_audit/20260729/` |
| G2a Go Stanford DATA-PILOT | 4090 | `done_verified` | 内容、manifest/split、adapter/DataLoader 审计 | `results/research/data_pilot/go_stanford_live_20260715/` |
| G2b RECON/HuRoN acquisition + conversion | 4090 | `done_not_promoted` | raw receipts 和 conversion integrity 已通过；晋级仍缺 metadata/holdout 决策 | `results/research/data_conversion/recon_huron_pilot_20260720/` |
| G3 IMAGE-FORWARD + TRAIN-STEP + B0 | 4090 | `done_not_promoted` | 有限 forward、2-step resume、单域 seed-0 smoke 通过 | `results/research/pretraining/`；只证明 plumbing |
| G4 GPU live gate | 4090 | `blocked_external` | `nvidia-smi` 成功、`/dev/nvidia*` 存在、PyTorch 可见两张 GPU | 尚未闭合 |
| G5 data promotion | 4090 + 用户决策 | `blocked_external` | 时间戳、collection policy/version、独立 holdout 可审计，或明确接受限制 | 尚未闭合 |
| G6 multi-dataset baseline + H1 | 4090 | `blocked_external` | G4/G5；固定预算、至少 3 seeds、失败保留、完整 provenance | 未执行 |
| G7 OFFLINE-GATE | 4090 | `blocked_external` | G6；IID/mixed/LODO/corruption 分域与 worst-domain 结果 | 未执行 |
| G8 SIM-GATE | Ubuntu | `blocked_external` | G7；非饱和冻结场景、多 seed 原始 rollout、相同 bridge/stabilizer | 未执行 |
| G9 device/robot + paper freeze | 待指定 + Windows | `blocked_external` | sustained timing、协议化真机、隐私/同意、claims/表图/统计重冻结 | 未执行 |

当前只应按顺序推进 **G4 → G5**。G4/G5 未闭合前，不应启动 G6，也不应在 Ubuntu
重做 U09/U10 或把更多静态文档计为训练进度。

## 4. 跨环境回执合同

每次环境工作完成或暂停后，必须向 Windows 返回一个小型 Git 可追踪回执；原始/
processed 数据继续只留在 4090。建议路径：

```text
.agents/handoffs/visualnav-transformer/<windows|ubuntu|4090>-progress/<YYYY-MM-DD-HHMM>/handoff.md
```

回执至少包含：

1. `environment`、`task_id`、`status`、开始/结束时间。
2. 完整 Git SHA、dirty 状态；若 dirty，列出文件所有权和未提交原因。
3. 输入、manifest、split、model contract、config 和 checkpoint 的 SHA-256。
4. 实际执行命令、退出码、关键运行环境和硬件信息。
5. 原始日志/结果路径、失败记录、最窄验证结果。
6. 已验证事实、证据支持的推断、未验证假设、下一门。
7. 是否执行过数据获取、forward/backward、optimizer、训练、评测、仿真或真机。

Windows 只在回执通过路径、hash、状态和科学口径审查后更新本文件、任务队列和
`PROJECT_ASSET_MAP.md`。环境内的口头结论不能直接进入论文。

## 5. 当前人类控制点

需要用户在 4090 环境决定的只有两类事项：

1. 授权并执行 GPU live gate 修复与最小复核；这属于服务器/驱动运维，不在 Windows
   本轮整理范围内。
2. 决定 RECON/HuRoN 的逐帧时间戳、collection policy/version 和独立 holdout 缺口是
   补齐、限制使用，还是停止晋级。

两项闭合后，才讨论多数据集 baseline/H1 的具体训练授权。Windows 和 Ubuntu 当前都不
需要执行新的实验；论文可以继续内部写作，但仍为 **submission NO-GO**。
