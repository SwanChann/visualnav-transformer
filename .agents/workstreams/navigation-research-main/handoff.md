# Current Handoff — 2026-07-13 Windows Integration

## 当前快照

- GitHub 来源：`origin/agent/ubuntu-sim-handoff` @ `a49d37220a514dc7b9efd3820edc3d6fbf7ce70c`（`Complete Ubuntu navigation research audit`）。
- Windows 集成 worktree：`F:/codespace/visualnav-transformer-worktrees/ubuntu-results-integration`，detached HEAD。
- 原工作区：`F:/codespace/visualnav-transformer`，保持在 `agent/ubuntu-sim-handoff@463ac70`，未拉取、未覆盖。
- 统一入口：根目录 `PROJECT_ASSET_MAP.md`。

## 已完成结论

Ubuntu 授权的 U00–U18 全部完成：环境与复现、受控 MuJoCo、独立离线评估、统计/图表、论文包、复现包、全盘纠错、policy backend 探索和单卡训练静态计划。主结果为 U09 v0.3（60/60）和 U10 v5（75/75）；旧 U09 v0.2、U10 v2 不得作为主证据。

## 科学结论边界

- policy-only 成功数：DDPM10 4/6；DDIM2 5/6；DDIM3 5/6；DDIM2/TTS8 5/6；DDIM2/CFG2/TTS8 6/6。
- stabilizer-on 每配置均 0/6，而且同 scene/seed 跨方法轨迹 byte-identical，只能解释为 controller override 行为。
- 精确重复保持二元 outcome，但共同轨迹最大偏差 0.444 m；结果是 seeded-stochastic 单次实现，不支持稳定排序或显著性主张。
- 离线 ADE 与 policy-only 成功率 Spearman rho=-0.67、exact p=0.30，是描述性负证据。

## 尚未完成

RECON/HuRoN 合规数据获取、多数据集训练、目标设备持续 deadline/miss timing、协议化重复真机结果、隐私/同意审查均需外部条件。论文因此仍是 submission NO-GO。

## 恢复指令

先读 `PROJECT_ASSET_MAP.md` 与 `.agents/workstreams/navigation-research-main/{TASKS.md,log.md,handoff.md}`。保留 U09 v0.3/U10 v5 原始结果，不覆盖历史审计目录；新外部证据必须按 `后续研究内容/external_execution/` 登记和晋级。
