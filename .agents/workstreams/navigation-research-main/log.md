# Navigation Research Agent Log

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
