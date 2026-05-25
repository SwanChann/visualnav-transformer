# RA-L 投稿全流程任务分解

生成日期：2026-05-17  
更新说明：2026-05-25 已确定主投 RA-L，并把论文主线重构为“预算感知的时间-质量推理调度 + 跨形态 zero-shot 机器人适配”。新的执行方案和两位同学的平行分工以 [05_RA-L最终选题重构与双人并行分工.md](05_RA-L最终选题重构与双人并行分工.md) 为准；本文件作为第一版全流程参考保留。
目标 venue：IEEE Robotics and Automation Letters (RA-L)  
建议投稿窗口：2026-07-15 前后完成首投

## 0. RA-L 硬约束

官方要求和对本项目的影响：

| 规则 | RA-L 要求 | 对本项目的影响 |
|---|---|---|
| 论文类型 | concise account of innovative research ideas and application results | 不能写成长篇毕业论文，要压成“短而硬”的机器人实证论文。 |
| 页数 | 6 页正文为标准，最多额外 2 页且需付费 | 推荐目标 7-8 页，避免 6 页塞不下实验。 |
| 补充材料 | 视频允许；不能把额外正文、图表、appendix 当作 multimedia 补充 | 所有关键实验表和图必须进正文。 |
| 评审 | double-anonymous peer review | 初稿要匿名，视频/仓库/文件名都要去作者身份。 |
| 审稿周期 | 首轮决定通常 3 个月内；R&R 必须 30 天内返回；最终 6 个月内接受或拒稿 | 实验和代码证据首投前就要准备好，不能指望返修时慢慢补。 |
| 双投 | 不能同时投 RA-L 和会议 | 投稿 RA-L 后不要把同一篇同时投 ICRA/IROS。 |
| 会议展示 | RA-L 接收后可在规定窗口内 transfer 到 RAS 会议展示 | 先 RA-L，再考虑 ICRA/IROS/CASE/Humanoids 等展示机会。 |

官方链接：

- RA-L author information: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
- RA-L FAQ: https://www.ieee-ras.org/publications/ra-l/faq/
- RA-L overview: https://www.ieee-ras.org/publications/ieee-robotics-and-automation-letters/

## 1. 总目标

把当前毕业设计压缩成一篇 RA-L 风格论文：

> Real-Time Diffusion Policy Visual Goal Navigation on a Quadruped Robot

核心 claim：

> We present and evaluate a real-time deployment stack that makes NoMaD-style diffusion-policy visual goal navigation executable on a resource-constrained quadruped robot, with accelerated diffusion inference and real-robot validation.

不能主张：

- 我们提出了一个全新的视觉导航基础模型。
- 我们解决了四足机器人通用视觉导航。
- outdoor zero-shot 已经鲁棒。
- MuJoCo stabilizer 结果等价于真实策略性能。

## 2. 团队分工总览

默认三人：

- **你 / 第一作者 / 总负责人**：论文主线、贡献边界、实验设计、最终写作、投稿。
- **同学 A：推理调度线负责人**：围绕 DDIM、CFG、TTS、预算感知选择器做实验、实现、分析、写作和图表。
- **同学 B：跨形态适配线负责人**：围绕机器人平台抽象、不同形态仿真、Lite3 实机映射做实验、实现、分析、写作和图表。

两位同学的工作性质应当相近：都要读文献、写代码、跑实验、整理数据、画图、写自己负责的小节。区别只在技术主题不同，不能一个人只写文章、另一个人只做实验。

## 3. 阶段一：现状调研与资产盘点

建议时间：2026-05-17 至 2026-05-22

### 3.1 项目资产盘点

负责人：你  
协作：同学 A

要做：

- [ ] 建立 `投稿冲刺/RA-L工作台/` 或等价目录，集中放实验表、图、视频脚本、论文草稿。
- [ ] 列出现有代码模块：训练、推理、MuJoCo、Lite3 实机桥接、实验脚本。
- [ ] 列出现有模型权重：EfficientNet-B0、DINOv2、ConvNeXt、ResNet、baseline NoMaD。
- [ ] 列出现有实验结果：DDIM、CFG、TTS、encoder、MuJoCo、实机。
- [ ] 列出现有视频/图片素材：Lite3 实机、室内场景、室外失败、MuJoCo 轨迹。
- [ ] 列出现有论文图表是否能复用。
- [ ] 标出证据缺口：哪些 claim 还没有原始日志或统计支撑。

交付物：

- `asset_inventory.md`
- `claim_evidence_matrix.xlsx` 或 `.csv`

### 3.2 RA-L 近三年同类论文调研

负责人：同学 B  
协作：你

检索关键词：

- `RA-L visual navigation robot learning`
- `RA-L diffusion policy robot navigation`
- `RA-L quadruped visual navigation`
- `RA-L topological navigation`
- `RA-L real-time robot learning deployment`

要做：

- [ ] 找 20 篇 RA-L/ICRA/IROS/CoRL 相关论文。
- [ ] 每篇记录：题目、venue、年份、任务、机器人平台、方法、实验规模、与本论文差异。
- [ ] 重点读 GNM、ViNT、NoMaD、Diffusion Policy、NavDP/NaviBridger/SIDP、四足视觉导航相关工作。
- [ ] 总结审稿人可能认为“已有”的部分。
- [ ] 总结我们真正不同的部分：Lite3 四足部署、实时扩散采样权衡、完整系统闭环、实机 safety bridge。

交付物：

- `related_work_table.md`
- `related_work_bib_candidates.bib`
- `gap_summary.md`

### 3.3 论文 claim 锁定

负责人：你  
协作：同学 A、同学 B

要做：

- [ ] 写出 3 条 contribution，每条必须能被实验或系统实现支撑。
- [ ] 写出 4 个 research questions。
- [ ] 写出不能说的 claim 清单。
- [ ] 确定最终 target configuration，例如 `EfficientNet-B0 + DDIM-2 + TTS-8 + CFG=0`。

交付物：

- `claim_lock_v1.md`

建议最终 research questions：

1. Can accelerated diffusion sampling make NoMaD-style visual navigation run in real time on Jetson Orin?
2. How do DDIM, CFG, TTS, and visual encoders trade off latency and action quality?
3. Can the system complete closed-loop navigation in MuJoCo and on a real Lite3 quadruped?
4. Where does the current system fail under outdoor domain shift?

## 4. 阶段二：补实验与数据闭环

建议时间：2026-05-23 至 2026-06-16

### 4.1 离线推理消融

负责人：同学 A  
协作：你

必须完成：

- [ ] DDPM baseline。
- [ ] DDIM steps：2、3、4、8。
- [ ] TTS budget：1、4、8、16。
- [ ] CFG weight：0、1、2。
- [ ] encoder：EfficientNet-B0、ResNet50、ConvNeXt、DINOv2。
- [ ] 所有结果统一报告 mean、std、95% CI。

指标：

- latency / Hz。
- forward progress proxy。
- lateral absolute displacement。
- smoothness。
- action diversity。
- endpoint distance proxy。

交付物：

- `offline_ablation_raw.csv`
- `offline_ablation_summary.csv`
- `offline_ablation_table.tex`
- `latency_tradeoff_plot.pdf`

### 4.2 Jetson Orin 端实时性剖析

负责人：同学 A  
协作：你

必须完成：

- [ ] 相机读取耗时。
- [ ] 图像预处理耗时。
- [ ] visual encoder 耗时。
- [ ] distance prediction 耗时。
- [ ] diffusion sampling 耗时。
- [ ] TTS scoring 耗时。
- [ ] PD adapter 耗时。
- [ ] twist command send 耗时。
- [ ] end-to-end control frequency。

推荐至少测 1000 个 loop，报告 mean / median / p90 / p95。

交付物：

- `orin_timing_profile.csv`
- `orin_timing_profile_table.tex`
- `timing_breakdown_plot.pdf`

### 4.3 MuJoCo 闭环实验

负责人：同学 A  
协作：你

必须完成：

- [ ] easy / medium / hard 三类场景。
- [ ] 每个场景至少 10 次 trial。
- [ ] 对比 DDPM、DDIM-2、DDIM-2 + TTS-8、DDIM-2 + CFG/TTS。
- [ ] 如果使用 route stabilizer，必须额外记录 stabilizer on/off 或在表注中明确限定解释。

指标：

- success rate。
- final distance。
- time to goal。
- path length。
- number of interventions。
- fall/stuck count。

交付物：

- `mujoco_closed_loop_raw.csv`
- `mujoco_closed_loop_summary.csv`
- `mujoco_results_table.tex`
- `mujoco_trajectory_figure.pdf`

### 4.4 真实 Lite3 室内导航实验

负责人：你  
协作：同学 A

这是 RA-L 成败关键。

最低要求：

- [ ] 至少 3 个室内场景。
- [ ] 每个场景至少 3 个目标点。
- [ ] 每个配置至少 10 次 trial。
- [ ] 总 trials 建议不少于 60；理想不少于 100。
- [ ] 记录失败原因，不只记录成功率。

建议配置：

| 配置 | 目的 |
|---|---|
| DDPM baseline | 原始扩散采样对照 |
| DDIM-2 | 加速核心对照 |
| DDIM-2 + TTS-8 | 推荐部署配置 |
| DDIM-2 + CFG=2 + TTS-8 | 检查 guidance 是否值得 |

指标：

- success rate。
- time to goal。
- path length 或 odom-integrated distance。
- mean control Hz。
- intervention count。
- collision/contact。
- stuck/fall/estop。
- final distance。

交付物：

- `real_robot_trials_raw.csv`
- `real_robot_summary.csv`
- `real_robot_results_table.tex`
- `real_robot_scene_figure.pdf`
- 每个代表性 trial 的视频片段。

### 4.5 Outdoor zero-shot stress test

负责人：你  
协作：同学 A、同学 B

目的不是证明 outdoor 已解决，而是展示 limitation。

要做：

- [ ] 简单室外直道/坡道成功案例。
- [ ] 强光、草地、复杂纹理、开放区域失败案例。
- [ ] 每类至少 3 次 trial。
- [ ] 保存失败帧和机器人行为。
- [ ] 写清楚失败归因：visual representation OOD、topomap mismatch、goal ambiguity、illumination shift。

交付物：

- `outdoor_stress_test_summary.md`
- `outdoor_failure_frames.pdf`
- 视频素材。

## 5. 阶段三：图表与视频

建议时间：2026-06-10 至 2026-06-24，可与实验并行

### 5.1 必须图

负责人：同学 B  
协作：你

- [ ] Fig. 1：系统总览图，从 camera/topomap 到 diffusion policy 到 PD/twist 到 Lite3。
- [ ] Fig. 2：Lite3 实机平台和软件栈。
- [ ] Fig. 3：DDIM/TTS/CFG latency-performance trade-off。
- [ ] Fig. 4：MuJoCo 三场景轨迹或结果可视化。
- [ ] Fig. 5：真实机器人实验场景和 outdoor failure cases。

图的原则：

- RA-L 页数紧，Fig. 1 必须信息密度高。
- 所有图中文字必须能在双栏下看清。
- 实机照片不要花哨，要证明“真的跑了”。
- failure figure 要像科学分析，不像宣传失败。

### 5.2 必须表

负责人：同学 A  
协作：同学 B

- [ ] Table I：平台和系统参数。
- [ ] Table II：offline ablation。
- [ ] Table III：MuJoCo closed-loop results。
- [ ] Table IV：real robot results。

### 5.3 Supplementary video

负责人：同学 B  
协作：你、同学 A

建议结构：

1. 10s：任务和平台简介。
2. 20s：系统 pipeline 动画或图示。
3. 40s：室内真实导航成功案例。
4. 20s：不同采样策略频率对比。
5. 30s：MuJoCo 闭环案例。
6. 30s：outdoor zero-shot 成功与失败。
7. 10s：limitations。

注意：

- 双匿名阶段不要出现学校、姓名、实验室 logo。
- 视频文件名不要暴露身份。
- 如果上传外链，也要匿名。

交付物：

- `supp_video_script.md`
- `supp_video_anonymous.mp4`
- `video_caption.txt`

## 6. 阶段四：论文写作

建议时间：2026-06-17 至 2026-07-05

### 6.1 LaTeX 项目搭建

负责人：同学 B  
协作：你

要做：

- [ ] 使用 IEEE RAS conference style 做初稿格式。
- [ ] 建立 `main.tex`、`sections/`、`figures/`、`tables/`、`refs.bib`。
- [ ] 使用匿名作者信息。
- [ ] 配好 `latexmk` 或 VSCode 编译。

交付物：

- 可编译的 RA-L 初稿工程。

### 6.2 Abstract

负责人：你

要求：

- 150-200 词左右。
- 第一半讲问题和系统。
- 第二半讲实验和数字。
- 必须包含 2-3 个最硬数字：加速倍数、Hz、真实机器人成功率/次数。

### 6.3 Introduction

负责人：你  
协作：同学 B

结构：

1. 视觉目标导航和四足机器人应用价值。
2. diffusion policy 有潜力但部署慢。
3. 现有 NoMaD/ViNT 等方法到真实四足平台仍有系统缺口。
4. 本文做了什么。
5. 三条 contribution。

### 6.4 Related Work

负责人：同学 B  
协作：你

分四段：

- Visual goal and topological navigation。
- Diffusion policies for robotics。
- Learning-based quadruped navigation。
- Real-time robotic deployment。

目标长度：0.75-1 页。

### 6.5 Method / System

负责人：你  
协作：同学 B

章节建议：

- Problem formulation。
- NoMaD-style policy overview。
- Topological goal selection。
- Accelerated diffusion inference。
- Waypoint-to-velocity adapter。
- Real robot safety bridge。

写作重点：

- NoMaD 只写必要背景，不抢贡献。
- DDIM/TTS/CFG 的公式和流程要清楚。
- Lite3 bridge 要写出工程难点：heartbeat、twist refresh、safety timeout、fall/stale monitor。

### 6.6 Experiments

负责人：你  
协作：同学 A

结构：

- Experimental setup。
- Offline inference ablation。
- MuJoCo closed-loop evaluation。
- Real robot indoor navigation。
- Outdoor zero-shot analysis。

所有实验都要回答前面 research questions。

### 6.7 Discussion and Limitations

负责人：你

必须写：

- Pre-collected topological maps。
- 单平台 Lite3。
- outdoor domain shift。
- MuJoCo stabilizer 的解释边界。
- 当前不是新 diffusion architecture。

### 6.8 Conclusion

负责人：你

短写：

- 重新概括系统和结论。
- 给出未来方向：online topomap、domain adaptation、larger outdoor trials、multi-robot validation。

## 7. 阶段五：内部审稿与压缩

建议时间：2026-07-06 至 2026-07-12

### 7.1 第一轮自审

负责人：你  
协作：同学 A、同学 B

检查问题：

- [ ] 每个 contribution 是否有实验支撑。
- [ ] 每个表格是否能独立理解。
- [ ] 是否过度声称算法创新。
- [ ] 是否解释 NoMaD 继承关系。
- [ ] 是否有统计显著性或置信区间。
- [ ] 是否诚实报告失败。
- [ ] 是否匿名。

### 7.2 模拟审稿人问题

负责人：同学 B 先列，你回答

至少准备回答：

- Why is this more than an engineering implementation of NoMaD?
- Why should DDIM/TTS be considered a contribution?
- How many real robot trials are enough?
- Does the route stabilizer inflate MuJoCo performance?
- Is the topological map manually curated?
- Does the method generalize outdoors?
- Why not compare against ViNT/GNM or classical navigation?

交付物：

- `reviewer_questions_and_answers.md`

### 7.3 页数压缩

负责人：同学 B  
协作：你

目标：

- 7-8 页以内。
- 图表优先保留，弱文字删掉。
- Related Work 不超过 1 页。
- Method 不超过 2 页。
- Experiments 至少 2.5 页。

## 8. 阶段六：投稿准备

建议时间：2026-07-13 至 2026-07-15

负责人：你  
协作：同学 B

要做：

- [ ] PaperPlaza 账号确认。
- [ ] ORCID 和作者信息确认。
- [ ] 选择 2-5 个 RA-L keywords。
- [ ] 检查 PDF 格式。
- [ ] 检查双匿名。
- [ ] 检查引用格式。
- [ ] 检查图表编号和正文引用。
- [ ] 检查视频匿名。
- [ ] 确认没有与会议双投。
- [ ] 确认所有作者贡献和署名顺序。
- [ ] 保存投稿版本 hash、PDF、source zip。

推荐 keywords：

- Visual-Based Navigation
- Legged Robots
- Motion and Path Planning
- Deep Learning for Visual Perception
- Learning from Demonstration / Imitation Learning

最终 keywords 要以 RA-L 官方 keyword 列表为准。

## 9. 阶段七：首轮结果后的处理

### 如果 Accept

负责人：你、同学 B

- [ ] 两周内提交 camera-ready。
- [ ] 修正格式。
- [ ] 处理版权和 IEEE 文件。
- [ ] 准备 conference presentation transfer。

### 如果 Revise and Resubmit

负责人：你  
协作：同学 A、同学 B

RA-L R&R 只有 30 天，非常紧。

分工：

- 你：逐条回复审稿人，决定补实验优先级。
- 同学 A：7-10 天内补完所有可补实验。
- 同学 B：维护 diff、修改图表、压页数。

交付物：

- `response_to_reviewers.pdf`
- highlighted diff PDF。
- revised manuscript。

### 如果 Reject

负责人：你

选择：

- 深改后重投 RA-L。
- 扩展为 T-RL/T-RO/RAS journal。
- 改投 IROS/ICRA conference。

必须保留：

- 审稿意见归类表。
- 下一版修改计划。

## 10. 两位同学的详细任务包

更新说明：2026-05-25 后，两位同学不再按“实验/写作”拆分，而按两个技术子线平行拆分。每个人都要做调研、代码、实验、数据、图表和自己小节的英文初稿。详细任务以 [05_RA-L最终选题重构与双人并行分工.md](05_RA-L最终选题重构与双人并行分工.md) 为准。

### 同学 A：推理调度与时间-质量优化线

核心使命：

> 把 DDIM、CFG、TTS 统一成一个实时预算下的时间-质量优化问题，并用 Pareto frontier 和 budget-aware selector 证明。

任务清单：

- [ ] 调研 anytime inference、test-time sampling、diffusion acceleration、budget-aware robot inference。
- [ ] 统一 DDIM、CFG、TTS 的实验配置和日志格式。
- [ ] 设计 quality score 与 latency-quality Pareto 分析。
- [ ] 跑 DDIM steps、CFG weight、TTS budget 和组合消融。
- [ ] 实现 Fast / Balanced / Quality 三类 budget-aware selector。
- [ ] 生成 Pareto frontier、latency breakdown 和配置选择表。
- [ ] 写 `Budget-Aware Diffusion Inference` 小节和对应实验结果初稿。

每周交付：

- 周一：本周调研/实现/实验计划。
- 周三：中间结果、raw csv、阻塞点。
- 周五：图表、结果摘要、论文段落 draft。

不能只交截图；必须交 raw data、生成脚本和可复现结果。

### 同学 B：跨形态适配与仿真迁移线

核心使命：

> 把同一个高层视觉导航策略通过不同 robot adapter zero-shot 接到不同形态平台，并用仿真实验证明适配边界。

任务清单：

- [ ] 调研 cross-embodiment transfer、morphology-agnostic policy、robot adapter、四足/轮式导航部署。
- [ ] 抽象 `RobotAdapter` 接口。
- [ ] 整理 Lite3 adapter 与现有真实机器人控制链路。
- [ ] 实现 differential-drive adapter。
- [ ] 实现 holonomic adapter。
- [ ] 可选实现 Ackermann adapter。
- [ ] 设计跨形态仿真 benchmark。
- [ ] 跑 Lite3 / differential-drive / holonomic 仿真实验。
- [ ] 生成跨形态轨迹图、adapter 参数表和成功率统计。
- [ ] 写 `Morphology-Specific Robot Adapters` 小节和对应实验结果初稿。

每周交付：

- 周一：本周调研/实现/实验计划。
- 周三：仿真中间结果、raw csv、阻塞点。
- 周五：图表、结果摘要、论文段落 draft。

不能只做接口包装；必须证明哪些形态 zero-shot 可行、哪些失败、失败原因是什么。

## 11. 你自己的任务

你不能把最关键的部分外包。

必须亲自负责：

- [ ] 最终论文主张。
- [ ] 贡献边界。
- [ ] 实验设计。
- [ ] 真实机器人关键实验。
- [ ] Introduction。
- [ ] Method 核心段落。
- [ ] Experiments 解释。
- [ ] Discussion 和 limitation。
- [ ] 投稿和作者确认。
- [ ] 审稿回复。

每天 30 分钟做 team sync：

- 昨天完成什么。
- 今天做什么。
- 当前阻塞是什么。
- 哪个 claim 还没有证据。

## 12. 最终交付清单

投稿前必须齐：

- [ ] `paper.pdf`
- [ ] `paper_source.zip`
- [ ] `refs.bib`
- [ ] 所有 figures 源文件。
- [ ] 所有 tables 的 raw csv。
- [ ] `supp_video_anonymous.mp4`
- [ ] `claim_evidence_matrix`
- [ ] `reviewer_questions_and_answers.md`
- [ ] `submission_checklist.md`
- [ ] 代码版本 hash。
- [ ] 实验数据备份。

## 13. 最短可执行时间表

| 日期 | 里程碑 |
|---|---|
| 2026-05-22 | 完成现状调研、资产盘点、claim lock |
| 2026-05-31 | 完成 offline ablation 和 timing profile |
| 2026-06-09 | 完成 MuJoCo 实验 |
| 2026-06-16 | 完成真实 Lite3 室内实验 |
| 2026-06-20 | 完成 outdoor stress test |
| 2026-06-24 | 完成所有图表和视频初版 |
| 2026-06-30 | 完成英文初稿 v1 |
| 2026-07-05 | 完成英文初稿 v2 |
| 2026-07-12 | 完成内部审稿和压缩 |
| 2026-07-15 | RA-L 首投 |

## 14. 最重要的判断

RA-L 不会因为“我做了很多工程”就接受论文。它要看到一个清晰、简洁、可验证的机器人研究贡献。

所以这篇论文的成败不是写得多，而是能否把下面这句话证明扎实：

> Accelerated diffusion inference, when integrated with topological image-goal selection and a safety-aware quadruped control bridge, enables practical real-time visual goal navigation on a resource-constrained quadruped platform.
