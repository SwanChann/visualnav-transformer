# Navigation Research Agent Log

## 2026-07-13 — GitHub 回收与 Windows 集成整理

- `git fetch origin --prune` 后确认 Ubuntu 完整成果已推送到 `origin/agent/ubuntu-sim-handoff@a49d372`，相对旧本地 HEAD 增加 692 个文件、131,574 行；在 detached integration worktree 整理，原工作区未拉取或覆盖。
- 独立审计确认 U00–U18 在 Ubuntu 授权边界内完整：U09 v0.3 60/60、U10 v5 75/75、分析/数据/benchmark/backend/training-plan 测试组分别记录为 33/33、11/11、7/7、5/5、4/4，复现清单 141 文件。
- 新建根目录 `PROJECT_ASSET_MAP.md`，统一映射主结果、论文图表、协议代码、数据/模型、历史证据、作废版本、治理文件与外部阻塞项；校正 `EXP-02` 为已完成，并将旧任务中的 138-file 更正为 141-file。
- 修复 Windows CRLF checkout 导致的 protocol overlay 哈希误报：文本协议按 UTF-8/LF 内容身份哈希，二进制 immutable input 继续按原始字节哈希；runtime crash 单测显式隔离 gitignored checkpoint preflight。
- 火山终审 critic 以 `glm-5.2`/read-only 启动，持续约 8 分钟未产出正式 final，主控按编排时限终止并记录为 timeout；未采纳任何不完整输出，改由本地确定性计数与测试完成终验。
- 结论不变：Ubuntu 工作完成，但整篇 RA-L 仍受外部数据/训练、目标设备 sustained timing、重复真机 outcome、隐私/同意门阻塞。

## 2026-07-10 — Onboarding

- 读取 `codex-volc-orchestrator`、`codex-volc-worktree-workflow`、`start-project-agent` 完整技能说明。
- 运行项目 onboarding scan；确认仓库根目录为 `F:/codespace/visualnav-transformer`。
- 确认当前唯一注册 worktree 即主工作区，`main` 位于 `0f1284a`，存在上一阶段需保留的修改与未跟踪成果。
- 决策：当前阶段复用该 worktree，避免从 `HEAD` 新建 detached worktree 时丢失尚未锚定的研究成果；不提交、不切分支、不执行远程 Git 操作。
- 预检 `volc-worker`：模型 `glm-5.2`、provider `volcengine-agent-plan`、sandbox `read-only`，provider 与 `ARK_API_KEY` 均已配置；未读取或输出密钥。
- 本机能力边界：不执行模型训练、仿真或真机实验；允许离线分析、代码/配置/数据协议建设、静态与合成验证、论文准备。
- 下一步：启动两个只读 worker，分别审计当前 RA-L 和未来研究的本地任务；主控合并后更新队列并执行首项。

## 2026-07-10 — First execution batch

- 完成 2 个只读 Volc producer 审计；A 聚焦当前 RA-L，B 聚焦多数据集/TinyNavBrain。主控接受证据边界与数据 Gate A 判断，拒绝把当前/未来两套 03/04 文档视为重复文件。
- 独立重跑 `ral_offline_pareto.py`：4 个文本/CSV 与现有产物逐字节一致；131 配置行、2004 case records、11 个 Table II 代表行；PDF 头有效。
- 新增 RA-L claims ledger，明确 p95 的 case-mean 口径、TTS verifier/proxy 循环性、Go Stanford 单数据集边界和真机人工复核要求。
- 新增 `ral_mujoco_summary.py` 与 2 个合成单元测试；汇总 9 个已有 source、228 条 raw records、84 个 source-scoped groups，不运行仿真。
- MuJoCo 审计发现：228/228 均缺 seed、per-step latency、stuck、fall、stabilizer_mode；27 条 success/final-distance 语义冲突；14 个跨 source 重复配置；12 个跨配置重复轨迹摘要。没有轨迹坐标，因此拒绝伪造 path PDF。
- 修正判断：seeded encoder benchmark 有四种 encoder，但闭环行为指标饱和/重复，主要只能支撑 wall-time 差异。
- 下一步：核验候选多数据集官方来源与许可，建立机器可读 registry；随后实现 manifest/leakage audit。

## 2026-07-10 — Data Gate A groundwork

- 建立 5 数据集 registry：Go Stanford/RECON/HuRoN 为 Tier A；SCAND 因官方许可不可见而阻塞；TartanDrive 因 payload 许可边界、单包约 100 GB 和高速 ATV domain gap 降为 Tier C。
- 实现 trajectory-level manifest builder、schema/license/leakage auditor 和 group-safe split generator；合成测试 4/4 通过。
- 对本地 Go Stanford 生成 3696 行 manifest，统计 2956 train / 740 test。
- 旧 split 按 `source_session=去掉末尾 segment` 审计失败：238 个 session 跨 train/test，涉及 1333 行；`train/data_split.py` 证实其按 trajectory 文件夹随机切分、没有 group 约束。
- 在新目录生成 group-safe 候选：2106 groups，2957 train / 739 test；重新审计 error=0、passed=true。未覆盖历史 split。
- 保留 warning：Go Stanford 是 CC BY-NC-SA 3.0；本地 processed copy 的 processor version 未知。
- 下一步：冻结 benchmark v0.1 protocol 与 result schema；禁止在协议冻结前写 TinyNavBrain 模型实现。

## 2026-07-11 — Protocol, model scaffold, and frozen real evidence

- 冻结 benchmark v0.1：IID/mixed/LODO/corruption/closed-loop、baseline 顺序、公平预算、meter-scale metrics、latency scope 和 result JSON schema。
- 实现 benchmark result validator；critic 后补齐 data exposure key、head/NFE、corruption vocabulary、context/action-history/encoder-sharing 规则；7 个测试通过。
- Volc critic 判定 DATA-02/BENCH-01 需修、MODEL-01 接受。主控采用 strip/val canonicalization/比例门/session 断言，拒绝无条件 case-fold；修订后 data tests 6/6，grouped v2 audit error=0。
- 冻结 TinyNavBrain contract 与 baseline matrix；实现 feature-level PyTorch scaffold。`prp` 环境测试 5/5，通过 fusion/H0/H1/progress shapes、seed reproducibility、NFE/scale guard；默认 scaffold 4,505,569 trainable params（不含视觉 encoder）。
- 新增 frozen real timing 聚合：7 timing files、528 sampled loop records、14 annotation rows；2 个合成测试通过。自动 status 全为 success，人工 outcome 保持空白。
- Timing 代表值：DDIM2/CFG0/TTS8 6.61 Hz、p95 170.1 ms；DDIM2/CFG2/TTS8 3.43 Hz、p95 317.5 ms；DDIM3/CFG0/TTS0 5.90 Hz、p95 187.1 ms；`ddpm` 2.89 Hz 但 exact steps/CFG/TTS 未记录。
- Worker 调度：7 次尝试，3 次完整返回、4 次 404/timeout；最大并发 2。完整返回用量合计 input 435,922（cached 378,176）、output 30,697 tokens。结论：worker 只用于里程碑 producer/critic。
- 下一步：PAPER-01 当前 RA-L 写作证据包；需要人工完成 14 个视频 outcome，外部训练/仿真/真机仍阻塞。

## 2026-07-11 — Paper packages, video audit, and external execution handoff

- 逐一审查 20 段历史视频的 12-frame contact sheets，覆盖 14 trials。画面可支持真实运动/多视角执行，但没有冻结 goal、成功半径、timeout、intervention 或 terminal trigger；正式 outcome 全部保持 `needs_human_goal_check`，不从文件夹名计算成功率。
- 完成当前 RA-L paper package：窄化标题、evidence-bounded abstract、method/results skeleton、Fig. 1–4 / Table II–IV inclusion rules、hard-gate reviewer checklist。客观结论为 draft-ready / submission NO-GO。
- 完成未来 RA-L claims ledger（CL1–CL6）、go/no-go、图表计划和外部 data/training/sim/robot 分阶段执行包；TinyNavBrain 被视为待证候选，不预设优于 H0/DDIM。
- 新增原始数据 artifact receipt 工具；绑定 registry/entry/artifact/license snapshot SHA-256、官方 HTTPS 域名、许可名、operator 与 Git provenance，许可状态采用 allowlist，拒绝覆盖和跨 registry 重登记。数据测试增至 11/11。
- Volc `glm-5.2` 最终 critic 只读运行 510.3 s，主控采纳 131 source–configuration / 91 distinct-setting 口径、无效绝对时钟 timing 限制及 receipt 完整性修复；拒绝其“本地 manifest 不存在”判断，并把现有 grouped-v2 manifest 路径与 SHA 写入 registry。
- 回归：analysis 5/5、data 11/11、benchmark validator 7/7 + example passed、TinyNavBrain 5/5（`prp` env）、registry JSON 与 `git diff --check` 通过。未训练、未仿真、未运行真机、未提交/推送/切分支。
- 本地授权范围内任务结束；外部 G4/G5 与 EXP-01..03 仍为真实阻塞，不可用文档或 AI 审查替代。

## 2026-07-11 — Ubuntu simulation/offline handoff export

- 按 `context-handoff` export workflow 生成独立交接包 `visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`。
- 新消费环境允许离线与 MuJoCo 仿真实验，禁止训练和真机；首要任务改为完整 provenance 的受控闭环矩阵和独立离线指标。
- 包中记录 Windows -> Ubuntu 路径重建、环境/import/headless renderer 预检、checkpoint/manifest SHA-256、stabilizer 分层、非饱和场景门、输出隔离和 Table III 验收条件。
- 明确迁移风险：`.agents/`、`results/research/`、`scripts/research/`、`后续研究内容/`、`投稿冲刺/workspace/` 等关键成果未被当前 HEAD 跟踪；只 clone `0f1284a` 会丢失，必须转移完整 dirty worktree。

## 2026-07-11 — Ubuntu U00–U10 execution

- 在独立 worktree `/home/swanchan/visualnav-transformer-worktrees/agent-ubuntu-sim-handoff`、分支 `agent/ubuntu-sim-handoff`、HEAD `463ac70` 执行；主 worktree `main@0f1284a` 保持用户原有 dirty 状态且未触碰。未提交、推送、拉取或切分支。
- U00–U04：完成 Ubuntu 20.04 / RTX 3050 Ti / CUDA 12.1 / EGL / `nomad_train` 环境审计；从主 worktree 恢复并核验 checkpoint SHA `66edf390...`；解释 grouped manifest CRLF/LF 哈希；analysis/data/benchmark/model 测试与历史离线/MuJoCo 数值复现通过。
- 火山协作：两个 `glm-5.2` read-only producer 完成环境与 runner gap 审计；第三个 critic 因 provider reasoning-event 噪声持续超过 10 分钟被主控中止，没有伪造 final review。主控独立完成测试与真实 smoke。
- 冻结 U05 protocol v0.1，SHA-256 `0dffb341642027af89a51bf81df0053e6682647e062e41a7db0e75a4ef68e62f`；配置固定 K=8、5 methods、3 scenes、3 seeds、stabilizer off/on，共 90 planned keys。
- U06 实现严格 trial/batch schema、语义 validator、原子/断点 runner、source-diff provenance、collision/fall/stuck/timeout latch、CUDA sampler timing 和 full-loop timing。修复 `legacy_bridge.py` 的 Linux 路径错误，受控 runner 不再向 legacy 历史结果目录写新摘要。
- U07：stand smoke 通过；DDIM2/easy/seed11/stabilizer-off 在 43 loops 成功，final distance 0.4572 m；完全重复的 42 点轨迹最大差 0.0 m；四类失败合成触发通过。
- U08：DDPM10 baseline-only 校准 18 episodes，全部 JSON 有效。easy/medium/hard 在两个 stabilizer strata 均为 3/3 成功；三场景全部 success 饱和，且 stabilizer-on 在 easy/hard 均触发 3/3 stuck。按预声明 gate 全部拒绝，U09 禁止启动。
- U10：agent worktree 缺 dataset images 的第一次启动在推理前安全失败并保留；显式使用主 worktree 只读完整 dataset 后，grouped-v2 test split 冻结 15 cases，5 configs 共 75 records、invalid=0。DDIM2 mean-K8 ADE 0.6018 m / 9.28 ms；DDPM10 0.6004 m / 39.98 ms；heuristic TTS8 0.6944 m，为负证据。
- 最终回归：analysis 21/21、data 11/11、benchmark 7/7；JSON Schema 2020-12、所有 trial/manifest、`git diff --check` 通过。U11–15 因 U09 无 retained scene 而保持真实阻塞。

## 2026-07-11 — Ubuntu v0.2 U09–U15 completion

- 继续使用 single persistent worktree；Codex/OpenAI 为唯一 writer。4 次只读 `glm-5.2 / volcengine-agent-plan` 调用：scene producer、U11–15 producer、integration critic、revision verifier；峰值并发 2。worker 未获写权限。
- 基于 baseline-only v0.1 timing 冻结 calibration overlay SHA `3c2eec...`，统一 timeout 为 62 cycles / 15.5 s；18-trial gate 保留 easy/medium、淘汰 hard。最终 v0.2 overlay SHA `4df19b...` 在任何跨方法结果前冻结，计划 60 trials。
- U09 完成 60/60：24 success、36 navigation failure、0 crash、0 infrastructure invalid、0 missing。policy-only primary matrix为 DDPM10 3/6、DDIM2 5/6、DDIM3 5/6、DDIM2+TTS8 5/6、DDIM2+CFG2+TTS8 6/6；stabilizer-on 全部 0/6。
- integration critic 发现 stabilizer-on 对每个 scene/seed 的五方法轨迹 byte-identical；主控新增机器检测、表图脚注和 claims 降级，只把它解释为 controller override 的负证据与 latency 差异，不视为独立方法结果。
- critic 对 primary success 的质疑经代码复核被部分驳回：MuJoCo mission 有 `goal_position` 时只采用 physics `<0.5 m`，且 tick-62 timeout 在 goal check 前终止，符合 frozen wording。真正缺陷是 post-hoc robustness 把 terminal 坐标重算成 primary success；已修为 primary condition 精确复用 raw outcome。
- 精确 repeat audit 显示 medium/seed23 的 DDIM3 从 success 翻为 failure、最大轨迹差 3.30 m；主结果保留但明确为 seeded-stochastic single realization，禁止稳定 ranking/显著性主张。
- U11 完成 2,000-replicate paired cluster bootstrap、Clopper-Pearson、paired deltas、offline correlation、robustness、trajectory equivalence 和 provenance map；U10 ADE vs policy-only success rho=-0.67、exact p=0.30，为负证据。
- U12 生成 Table III 与 3 个有效 PDF；U13 更新 abstract/method/captions/evidence/checklist/claim matrix，submission 仍 NO-GO；U14 生成 120-file checksum bundle、checkpoint/U10/secret checks与 byte-identical analysis regeneration；U15 final audit 明确 Ubuntu complete / external robot gates remain。

## 2026-07-11 — U16–U18 full audit, policy backend and scale corrections

- 全盘审计发现 U10 v2 错用 RECON 0.25 m 尺度评估 GO Stanford；修复 runner 为从冻结 training data config 读取 0.12 m，U10 v5 重跑 75/75 valid。v2 明确作废，下游 analysis 绑定 v5 summary SHA。
- 进一步发现 MuJoCo v0.2 把 NoMaD native action units 直接当作 meter waypoint；新增显式 target action scale，按 Lite3 0.4 m/s / 4 Hz 冻结为 0.1 m。blind DDPM gate 仍保留 easy/medium、拒绝 hard；v0.3 在跨方法结果前冻结，60/60 完成：25 success、35 failure、0 crash/invalid/missing。
- v0.3 policy-only：DDPM10 4/6、DDIM2 5/6、DDIM3 5/6、DDIM2+TTS8 5/6、DDIM2+CFG2+TTS8 6/6；stabilizer-on 全部 0/6 且同 scene/seed 轨迹跨方法 byte-identical。medium/seed23 exact repeat 5/5 binary preserved，但连续轨迹最大差 0.444 m。
- 新增 framework-neutral `NavigationPolicyBackend`、Lite3System constructor injection、candidate diagnostics 和 adaptive-compute replay。corrected U10 v5 上 TTS routing 被支配，DDIM3/DDPM 只有后验精度/延迟交换，router 拒绝合并；接口和负结果保留。
- 新建单 RTX 4090 `TinyNavBrain-ScaleAdaptive` 机器可读计划：shared B0、physical scale/action history tokens、deterministic NFE1 baseline、residual-flow NFE1/2/4；validator 明确所有显存为未实测 partial accounting，未运行训练/backward/optimizer。
- 最终 revision worker 质疑 `loop_hz_mean` 与 4 Hz 合同；主控逐路径核验确认每次 navigation call 固定推进 12 x 20 ms = 0.24 s 仿真时间，而 11--19 Hz 是 headless 墙钟计算吞吐率。文档补充名义 4 Hz、离散后的 4.167 calls/s 以及 timing 字段语义；该 worker 超过 10 分钟硬上限、未产出正式报告，按 orchestrator 协议终止并记 timeout。
