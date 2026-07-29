# Current Handoff — 2026-07-29 Three-Environment Progress Sync

## 当前快照

- 人工与后续工作的唯一入口：`F:/codespace/visualnav-transformer`。
- 统一资产入口：根目录 `PROJECT_ASSET_MAP.md`；三环境总账：`PROJECT_PROGRESS.md`。
- 当前分支：`agent/ubuntu-sim-handoff`。
- Windows 已执行 `git fetch --all --prune`，审查并 fast-forward 吸收远端 4090 的
  12 个提交；进展回收基线为
  `9fb58e7e1c03d5813bf580ad91d25680f5999a72`。
- 集成后当前研究基础设施 83/83 tests、3/3 validators 与 `compileall` 通过；历史
  analysis 测试为 20/24，另 4 个仅因 Windows `prp` 缺完整 NoMaD 依赖而无法导入。
- 临时集成 worktree `F:/codespace/visualnav-transformer-worktrees/ubuntu-results-integration`
  已退休，不再作为项目入口。
- 4090 执行交接仍位于
  `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md`；
  其早期阶段已由远端回执部分闭合，当前状态以 2026-07-29 audit 和本 handoff 为准。

## 三环境职责与进度

- Windows：唯一人工/Git/论文集成入口。负责协议、代码审查、小型回执、资产地图、
  统一总账和论文；没有训练数据，不运行真实数据验证、训练、仿真或真机。
- 4090：唯一数据驻留与多数据集训练/离线评估环境。Phase 0、Go Stanford DATA-PILOT、
  IMAGE-FORWARD、两步 TRAIN-STEP、单域 B0，以及 RECON/HuRoN raw/conversion pilots
  已有回执。当前因 `/dev/nvidia*` 缺失、`nvidia-smi` 失败而禁止新 CUDA 执行；数据
  promotion 也受 metadata/holdout 缺口阻塞。
- Ubuntu：U00–U18 已完成。只接收通过 4090 离线门的 checkpoint，执行新的非饱和闭环
  仿真；不得重复 U09/U10，不承担需要完整数据的离线评估。
- 目标设备/真机：执行位置尚未指定，不得默认归给 Ubuntu 或 4090。

## 已验证事实

- U09 v0.3 为 60/60 有效受控记录，U10 v5 为 75/75 有效离线记录；两套权威目录未被
  4090 远端提交修改。
- 4090 Go Stanford 审计为 3696 条轨迹、198126 张图片、train split 131950 个 sample，
  冻结 5% 子集 6598。
- TinyNavBrain IMAGE-FORWARD 使用随机初始化 EfficientNet-B0，共 8513117 个可训练参数；
  TRAIN-STEP 只有 2 个 optimizer/backward steps；B0 是 H0/H1 各 200 optimizer steps、
  单域、seed 0、无评测的 plumbing smoke。
- RECON 处理 pilot 为 14 帧/1 window，HuRoN 为 66 帧/53 windows；完整性审计通过，
  但 `promotion_ready=false`。
- 2026-07-29 4090 live audit 记录磁盘 3.7 TiB、已用 3.0 TiB、剩余 576 GiB；
  NVIDIA 580.173.02 模块已加载，但 device nodes 缺失，PyTorch 可见设备数为 0。

## 推断、未验证假设与投稿硬门

- 推断：下一主线应先恢复 GPU live gate，再处理 RECON/HuRoN 晋级条件；继续补 Windows
  基础设施或在 Ubuntu 重复旧实验不会解除当前阻塞。
- 未验证：修复驱动/device nodes 后能否稳定运行完整训练；RECON/HuRoN 的缺失 metadata
  是否可补齐或可被限制性接受；142 GB 历史训练日志是否能建立可信 provenance。
- 历史 forward、2-step 和 B0 receipts 只证明接口、恢复和短程 plumbing；不证明收敛、
  H1 优于 H0、跨数据集泛化或方法有效。
- 论文仍为 **submission NO-GO**。至少还需多数据集 baseline/H1、分域离线门、Ubuntu
  新非饱和仿真、目标设备 sustained timing、协议化真机 outcome 与隐私/同意审查。

## 恢复指令

先读 `PROJECT_ASSET_MAP.md`、`PROJECT_PROGRESS.md`、`ENVIRONMENT.md`、
`.agents/workstreams/navigation-research-main/{TASKS.md,log.md,handoff.md}` 和
`results/research/4090_progress_audit/20260729/`。保留 U09 v0.3/U10 v5 原始结果，不覆盖
历史审计目录。下一服务器动作是闭合 `ENV-4090-GPU`，然后由用户决定 `DATA-04` 的
metadata/holdout 处理；在两门闭合前不要启动多数据集训练。
