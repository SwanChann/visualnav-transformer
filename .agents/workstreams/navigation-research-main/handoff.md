# Current Handoff — 2026-07-15 Three-Environment Freeze

## 当前快照

- 人工与后续工作的唯一入口：`F:/codespace/visualnav-transformer`。
- 当前分支：`agent/ubuntu-sim-handoff`；已包含 Ubuntu 审计提交 `a49d372` 和 Windows 资产整理提交 `ac49e99`，并同步至同名远端分支。
- 临时集成 worktree `F:/codespace/visualnav-transformer-worktrees/ubuntu-results-integration` 已完成使命，在成果提交、快进、验证和 push 后退休；不要再把它当作项目入口。
- 统一入口：根目录 `PROJECT_ASSET_MAP.md`。
- 本轮审查的 source base：`d8a238e169a626cae7eb69fe40e18421ea628162`。4090 的交付版本是包含 `.agents/handoffs/visualnav-transformer/pretraining-4090/2026-07-15-1641/handoff.md` 的提交；检出后必须用 `git rev-parse HEAD` 记录实际完整 SHA。

## 三环境职责

- Windows：唯一人工/Git/论文集成入口；静态合同与测试已完成，当前只做 diff、验证和版本交付收口。无数据集，不运行真实数据验证、训练、仿真或真机。
- 4090：唯一数据驻留和下一阶段主执行环境；负责 live snapshot、DATA-PILOT、真实 adapter、B0 smoke、baseline/H0/H1、多数据集离线门。当前服务器实际状态尚未在本会话核验。
- Ubuntu：U00–U18 已完成；只接收通过 4090 离线门的 checkpoint，执行新的非饱和闭环仿真。不得重复 U09/U10，不承担需要完整数据的离线评估。
- 目标设备/真机：执行位置尚未指定，不得默认归给 Ubuntu 或 4090。

## 已完成结论

Ubuntu 授权的 U00–U18 全部完成：环境与复现、受控 MuJoCo、独立离线评估、统计/图表、论文包、复现包、全盘纠错、policy backend 探索和单卡训练静态计划。主结果为 U09 v0.3（60/60）和 U10 v5（75/75）；旧 U09 v0.2、U10 v2 不得作为主证据。

## 科学结论边界

- policy-only 成功数：DDPM10 4/6；DDIM2 5/6；DDIM3 5/6；DDIM2/TTS8 5/6；DDIM2/CFG2/TTS8 6/6。
- stabilizer-on 每配置均 0/6，而且同 scene/seed 跨方法轨迹 byte-identical，只能解释为 controller override 行为。
- 精确重复保持二元 outcome，但共同轨迹最大偏差 0.444 m；结果是 seeded-stochastic 单次实现，不支持稳定排序或显著性主张。
- 离线 ADE 与 policy-only 成功率 Spearman rho=-0.67、exact p=0.30，是描述性负证据。

## 尚未完成

4090 live snapshot、RECON/HuRoN 合规数据获取、真实 adapter、B0 smoke、多数据集训练和离线门尚未执行；Ubuntu 新仿真门、目标设备持续 deadline/miss timing、协议化重复真机结果、隐私/同意审查也未完成。论文因此仍是 submission NO-GO。

## 恢复指令

先读 `PROJECT_ASSET_MAP.md`、`ENVIRONMENT.md` 与 `.agents/workstreams/navigation-research-main/{TASKS.md,log.md,handoff.md}`。保留 U09 v0.3/U10 v5 原始结果，不覆盖历史审计目录；新外部证据必须按 `后续研究内容/external_execution/` 登记和晋级。Windows 经授权形成冻结 commit 后，下一实际执行位置是 4090，而不是 Ubuntu。
