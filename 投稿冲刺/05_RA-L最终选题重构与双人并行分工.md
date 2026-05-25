# RA-L 最终选题重构与双人并行分工

生成日期：2026-05-25  
目标 venue：IEEE Robotics and Automation Letters (RA-L)  
当前决定：确定主投 RA-L，不再以 CoRL/会议为主线。

## 1. 新论文主线

原来的主线是：

> 把 NoMaD/扩散视觉导航部署到 Lite3 四足机器人上，并通过 DDIM、CFG、TTS 提升实时性和质量。

现在建议升级为：

> 在统一的视觉目标导航框架中，研究扩散策略推理的时间-质量权衡，并验证同一高层策略通过形态相关 adapter 在不同机器人平台上的 zero-shot 适配能力。

一句话版本：

> Budget-aware diffusion inference and morphology-agnostic robot adaptation for real-time visual goal navigation.

这比“我把模型部署到 Lite3”更像 RA-L 研究贡献，因为它有两个可被实验检验的问题：

1. 在固定实时预算下，DDIM、CFG、TTS 怎么组合才能得到最好的导航质量？
2. 同一个高层视觉导航策略能否不重新训练，通过不同 robot adapter 泛化到不同运动形态？

## 2. 建议标题

首选：

> Budget-Aware Diffusion Inference for Cross-Morphology Visual Goal Navigation

备选：

> Real-Time Visual Goal Navigation with Budget-Aware Diffusion Inference and Cross-Morphology Robot Adaptation

更保守的 RA-L 标题：

> Budget-Aware Diffusion Visual Navigation on Quadruped and Simulated Robot Platforms

如果跨形态实验最后不够强，标题应降级为第三个，避免过度声称。

## 3. 最终贡献设计

### Contribution 1：时间-质量统一推理调度

你的理解是对的：

- **DDIM**：用更少反向扩散步数换速度，本质是质量换时间。
- **CFG**：额外计算 conditional/unconditional guidance，通常是时间换质量。
- **TTS / best-of-N**：采更多候选再选择，明显是时间换质量。

论文不应把 DDIM、CFG、TTS 分散写成三个技巧，而应统一成一个核心问题：

> Given a real-time compute budget, choose a diffusion inference configuration that maximizes navigation quality under latency constraints.

最小可实现形式：

- 先离线扫 `ddim_steps x cfg_weight x tts_budget`。
- 得到 latency-quality Pareto frontier。
- 定义一个预算感知选择器：
  - 输入：目标周期或最大延迟 `B`，例如 80 ms、120 ms、200 ms。
  - 输出：`(scheduler, steps, cfg, tts_budget)`。
  - 目标：在 `latency <= B` 条件下最大化质量分数。

质量分数可以定义为：

```text
Q = w1 * forward_progress
  - w2 * lateral_deviation
  - w3 * smoothness_cost
  - w4 * collision_or_stuck_proxy
```

真实机器人结果中则用：

```text
Q_real = success_rate
       - alpha * normalized_time_to_goal
       - beta * intervention_rate
       - gamma * collision_or_stuck_rate
```

RA-L 里不用把这个包装成复杂学习算法。一个清晰、可复现、可验证的 budget-aware selector 就够了。

### Contribution 2：跨形态 zero-shot 适配框架

核心思想：

> NoMaD-style high-level policy predicts body-frame waypoints. These waypoints are not tied to a specific robot morphology. Morphology-specific adapters convert the same waypoint command into platform-specific controls.

统一接口：

```text
camera/topomap -> visual navigation policy -> waypoint sequence
waypoint sequence -> morphology-specific adapter -> robot command
robot command -> simulator or real robot
```

不同形态的 adapter 示例：

| 平台形态 | Adapter 输出 | 可做实验 |
|---|---|---|
| Lite3 quadruped | body-frame `vx, vy, wz` | MuJoCo + 真机 |
| Differential-drive robot | `v, omega` | 2D/3D 仿真 |
| Holonomic base | `vx, vy, wz` | 2D/3D 仿真 |
| Ackermann/car-like robot | steering + throttle | 2D/3D 仿真，可选 |

zero-shot 的定义必须严格：

- 同一个冻结视觉导航模型。
- 不对新形态重新训练 policy。
- 只替换 robot adapter 和底层运动模型。
- 同一套目标图像/拓扑导航逻辑。

### Contribution 3：Lite3 实机验证

真机只有 Lite3 没问题，但要把它定位为：

> real-world anchor validation

而不是全部跨形态结论的唯一依据。

真机证明：

- 系统能上真实四足机器人。
- budget-aware 推理能满足 onboard real-time constraints。
- adapter 和 safety bridge 在真实平台可运行。

仿真证明：

- 框架可以跨不同机器人形态迁移。
- 不同形态下 adapter 的成功率、轨迹质量和控制饱和情况不同。

## 4. 必须重新设计的实验矩阵

### Experiment A：推理时间-质量 Pareto 实验

目的：

> 证明不是随便用了 DDIM/TTS/CFG，而是在实时预算下系统优化了扩散推理配置。

配置：

| 变量 | 候选 |
|---|---|
| scheduler | DDPM, DDIM |
| DDIM steps | 2, 3, 4, 8 |
| CFG weight | 0, 1, 2 |
| TTS budget | 1, 4, 8, 16 |
| encoder | 先固定 EfficientNet-B0；encoder 另做次要实验 |

指标：

- latency mean / p90 / p95。
- inferred control Hz。
- offline quality score。
- action diversity。
- smoothness。
- forward progress。
- Pareto optimality。

图：

- latency-quality scatter。
- Pareto frontier。
- 不同预算下选择出的配置表。

预期结论：

- DDIM-2/3 是低预算下优势配置。
- TTS-8 可能在中等预算下最划算。
- CFG 如果质量收益不明显但延迟增加，就诚实写为不推荐默认开启。

### Experiment B：预算感知选择器验证

目的：

> 证明“综合平均优化”不仅是分析图，而是能变成一个系统策略。

预算设置：

| 模式 | 预算 | 目标 |
|---|---:|---|
| Fast | 80 ms | 保证高频控制 |
| Balanced | 120 ms | 默认部署 |
| Quality | 200 ms | 复杂场景提高候选质量 |

要做：

- [ ] 从 Experiment A 的结果表中选出每个预算下最优配置。
- [ ] 在 MuJoCo 和 Lite3 实机中跑 Fast / Balanced / Quality。
- [ ] 比较固定配置 vs budget-aware selector。

如果时间不够：

- 至少做静态 budget selector。
- 动态 selector 可以作为 future work。

### Experiment C：跨形态仿真 zero-shot

目的：

> 证明同一高层视觉导航策略通过不同 adapter 可跨机器人形态迁移。

建议最小配置：

| Morphology | 仿真方式 | 必做程度 |
|---|---|---|
| Quadruped Lite3 | 现有 MuJoCo | 必做 |
| Differential-drive | 简单 2D/3D 仿真或现有导航 sim | 必做 |
| Holonomic base | 简单 2D/3D 仿真 | 推荐 |
| Ackermann | 简单 2D/3D 仿真 | 可选 |

指标：

- success rate。
- final distance。
- time to goal。
- path length。
- command saturation ratio。
- tracking error between desired waypoint and executed motion。
- failure mode。

关键控制变量：

- 冻结同一个视觉导航 policy。
- 同一个 topomap/goal selection。
- 相同起点终点。
- 不同的只是 robot dynamics 和 adapter。

重要提醒：

- 如果仿真不是 visual-in-the-loop，而是只验证 waypoint adapter，要在论文中说清楚它验证的是 morphology adaptation layer，不要把它写成完整视觉 zero-shot。
- 最好至少一个仿真平台有视觉闭环，否则跨形态 claim 要降级为“adapter-level zero-shot transfer”。

### Experiment D：Lite3 真实机器人主实验

目的：

> 证明该系统在真实四足平台上可用，并验证预算调度带来的实际收益。

配置：

| 配置 | 作用 |
|---|---|
| DDPM baseline | 原始慢速基线 |
| DDIM-2 | 低延迟基线 |
| Balanced selector | 主方法 |
| Quality selector | 复杂场景对比 |

至少做：

- 3 个室内场景。
- 每个场景 3 个目标。
- 每个配置至少 10 次 trial。
- 总 trials 最低 60，理想 100+。

指标：

- success rate。
- time to goal。
- mean / p90 inference latency。
- control Hz。
- intervention count。
- collision/stuck/fall。
- final distance。

### Experiment E：Outdoor zero-shot stress test

目的：

> 给 RA-L 一个诚实的泛化边界，而不是夸大 outdoor。

做法：

- 简单室外路面成功例。
- 强光失败。
- 草地/复杂纹理失败。
- 开放区域目标混淆失败。

写法：

> These results indicate that the proposed deployment and inference framework transfers to the real quadruped platform, but visual representation robustness remains a limiting factor under severe outdoor domain shift.

## 5. 从今天到写完论文的完整任务表

### 第 0 阶段：选题冻结

时间：2026-05-25 至 2026-05-27  
负责人：你  
协作：同学 A、同学 B

- [ ] 决定最终标题。
- [ ] 决定是否把跨形态写成强 claim 还是弱 claim。
- [ ] 决定预算感知选择器是静态还是动态。
- [ ] 写出三条 contribution。
- [ ] 写出四个 research questions。
- [ ] 写出所有不能说的 claim。
- [ ] 更新论文工作目录。
- [ ] 开一个 shared progress log。

交付物：

- `claim_lock_20260527.md`
- `experiment_matrix_20260527.md`

### 第 1 阶段：现状调研

时间：2026-05-25 至 2026-05-31

要做：

- [ ] 查 RA-L 近三年视觉导航、四足导航、robot learning deployment 论文。
- [ ] 查 diffusion policy / NoMaD / NavDP / NaviBridger / SIDP 等方法。
- [ ] 查 cross-morphology / embodiment transfer / robot adapter 相关工作。
- [ ] 查 anytime inference / budget-aware inference / test-time sampling 相关工作。
- [ ] 整理与本文最接近的 20 篇论文。
- [ ] 每篇写清楚：任务、平台、是否真机、是否跨形态、是否实时、本文差异。

交付物：

- `related_work_matrix.md`
- `gap_summary_for_ral.md`
- `refs_ral.bib`

### 第 2 阶段：代码和实验框架改造

时间：2026-05-28 至 2026-06-07

要做：

- [ ] 把 DDIM/CFG/TTS 配置统一成 `InferenceConfig`。
- [ ] 输出每次推理的 latency、quality proxy、config id。
- [ ] 写 Pareto 分析脚本。
- [ ] 写 budget-aware selector。
- [ ] 抽象 `RobotAdapter` 接口。
- [ ] 保留 Lite3 adapter。
- [ ] 新增 differential-drive adapter。
- [ ] 新增 holonomic adapter。
- [ ] 可选新增 Ackermann adapter。
- [ ] 统一实验日志 schema。

交付物：

- `budget_selector.py`
- `robot_adapters.py`
- `experiment_logger.py`
- `pareto_analysis.py`

### 第 3 阶段：离线和仿真实验

时间：2026-06-03 至 2026-06-17

要做：

- [ ] 跑 Experiment A：推理 Pareto。
- [ ] 跑 Experiment B：预算选择器。
- [ ] 跑 Experiment C：跨形态仿真。
- [ ] 整理 raw csv。
- [ ] 生成 summary csv。
- [ ] 生成 LaTeX tables。
- [ ] 生成 Pareto 图和轨迹图。
- [ ] 写每组实验的 result note。

交付物：

- `inference_pareto_raw.csv`
- `inference_pareto_summary.csv`
- `budget_selector_results.csv`
- `cross_morphology_sim_results.csv`
- `fig_pareto_frontier.pdf`
- `fig_cross_morphology_paths.pdf`

### 第 4 阶段：Lite3 真实实验

时间：2026-06-10 至 2026-06-24

要做：

- [ ] 检查 Lite3、Orin、相机、网络、电池、安全场地。
- [ ] 固定室内场景和 topomap。
- [ ] 先跑 5 次 pilot，修日志。
- [ ] 正式跑 DDPM baseline。
- [ ] 正式跑 DDIM-2。
- [ ] 正式跑 Balanced selector。
- [ ] 正式跑 Quality selector。
- [ ] 记录每次 trial 的视频、日志、失败原因。
- [ ] 汇总统计。

交付物：

- `real_robot_raw_trials.csv`
- `real_robot_summary.csv`
- `real_robot_failure_cases.md`
- `fig_real_robot_scenes.pdf`
- `table_real_robot.tex`

### 第 5 阶段：图表、视频、论文素材

时间：2026-06-18 至 2026-06-30

要做：

- [ ] Fig. 1 系统总图：budget-aware inference + robot adapter。
- [ ] Fig. 2 推理时间-质量 Pareto frontier。
- [ ] Fig. 3 跨形态仿真轨迹。
- [ ] Fig. 4 Lite3 实机场景和结果。
- [ ] Table I 平台和 adapter 参数。
- [ ] Table II 推理配置 Pareto 结果。
- [ ] Table III 跨形态仿真结果。
- [ ] Table IV 真实 Lite3 结果。
- [ ] supplementary video 初版。

交付物：

- 所有 figure 源文件和 PDF。
- 所有 table tex。
- `supp_video_v1.mp4`。

### 第 6 阶段：英文论文初稿

时间：2026-06-25 至 2026-07-08

要写：

- [ ] Abstract。
- [ ] Introduction。
- [ ] Related Work。
- [ ] Problem Formulation。
- [ ] Budget-Aware Diffusion Inference。
- [ ] Morphology-Specific Robot Adapters。
- [ ] Experimental Setup。
- [ ] Inference Pareto Results。
- [ ] Cross-Morphology Simulation Results。
- [ ] Real Robot Results。
- [ ] Discussion and Limitations。
- [ ] Conclusion。

交付物：

- `paper_v1.pdf`
- `paper_v1_source/`

### 第 7 阶段：内部审稿和压页数

时间：2026-07-09 至 2026-07-18

要做：

- [ ] 检查 RA-L 页数。
- [ ] 检查匿名。
- [ ] 检查每个 contribution 是否有证据。
- [ ] 检查跨形态 claim 是否过度。
- [ ] 检查所有图表是否正文引用。
- [ ] 模拟 10 个审稿问题并写回答。
- [ ] 删除弱 claim。
- [ ] 压缩 Related Work。
- [ ] 强化 Method 和 Experiments。

交付物：

- `paper_v2.pdf`
- `reviewer_qna.md`

### 第 8 阶段：定稿

时间：2026-07-19 至 2026-07-25

要做：

- [ ] 完成语言润色。
- [ ] 完成图表最终版。
- [ ] 完成视频匿名版。
- [ ] 完成参考文献格式。
- [ ] 完成作者贡献确认。
- [ ] 完成 RA-L 投稿检查。

交付物：

- `paper_final.pdf`
- `paper_source_final.zip`
- `supp_video_anonymous.mp4`
- `submission_checklist.md`

## 6. 两位同学的平行分工

原则：

- 两个人都不是单纯“写文章”或“做实验”。
- 两个人都必须承担：调研、代码、实验、数据、图表、段落写作。
- 两个人的区别是技术子线不同。

### 同学 A：推理调度与时间-质量优化线

核心问题：

> 如何在实时预算下选择 DDIM、CFG、TTS 的组合，让扩散视觉导航达到最优时间-质量折中？

具体任务：

- [ ] 调研 anytime inference、test-time sampling、diffusion acceleration、budget-aware robot inference。
- [ ] 整理 DDIM、CFG、TTS 在本文中的统一解释。
- [ ] 实现或整理统一 `InferenceConfig`。
- [ ] 设计 `quality score`。
- [ ] 跑 DDIM steps 消融。
- [ ] 跑 CFG weight 消融。
- [ ] 跑 TTS budget 消融。
- [ ] 跑组合 Pareto 实验。
- [ ] 实现静态 budget-aware selector。
- [ ] 输出 Fast / Balanced / Quality 三种配置。
- [ ] 生成 Pareto frontier 图。
- [ ] 生成 latency breakdown 图。
- [ ] 写论文中 `Budget-Aware Diffusion Inference` 小节初稿。
- [ ] 写 Experiments 中 inference ablation 结果初稿。

交付物：

- `inference_related_work.md`
- `inference_config_table.md`
- `inference_pareto_raw.csv`
- `inference_pareto_summary.csv`
- `budget_selector_results.csv`
- `fig_pareto_frontier.pdf`
- `sec_budget_inference_draft.tex`

### 同学 B：跨形态适配与仿真迁移线

核心问题：

> 同一个高层视觉导航策略，能否通过不同形态的 adapter zero-shot 接入不同机器人平台？

具体任务：

- [ ] 调研 cross-embodiment transfer、morphology-agnostic policy、robot adapter、四足/轮式导航部署。
- [ ] 整理现有 Lite3 adapter 和控制链路。
- [ ] 抽象 `RobotAdapter` 接口。
- [ ] 实现 differential-drive adapter。
- [ ] 实现 holonomic adapter。
- [ ] 可选实现 Ackermann adapter。
- [ ] 设计跨形态仿真实验场景。
- [ ] 跑 Lite3 MuJoCo 仿真。
- [ ] 跑 differential-drive 仿真。
- [ ] 跑 holonomic 仿真。
- [ ] 记录 success、tracking error、command saturation。
- [ ] 生成跨形态轨迹图。
- [ ] 生成 adapter 参数表。
- [ ] 写论文中 `Morphology-Specific Robot Adapters` 小节初稿。
- [ ] 写 Experiments 中 cross-morphology results 初稿。

交付物：

- `morphology_related_work.md`
- `robot_adapter_interface.md`
- `cross_morphology_sim_raw.csv`
- `cross_morphology_sim_summary.csv`
- `fig_cross_morphology_paths.pdf`
- `table_adapter_parameters.tex`
- `sec_robot_adapters_draft.tex`

## 7. 两位同学共同要做的事

每个人都要参与：

- [ ] 每周读 3-5 篇相关论文。
- [ ] 每周至少提交一次可复现结果。
- [ ] 所有实验必须有 raw data。
- [ ] 所有图表必须能从 raw data 重生成。
- [ ] 每人负责自己小节的英文初稿。
- [ ] 每人负责自己实验的审稿问题回答。

每周同步格式：

```text
1. 本周完成：
2. 当前数据/图表：
3. 遇到的问题：
4. 下周计划：
5. 哪个论文 claim 被支撑/被削弱：
```

## 8. 你自己的职责

你是第一作者，必须亲自负责：

- [ ] 最终 contribution 锁定。
- [ ] 实验矩阵取舍。
- [ ] 是否弱化或强化 cross-morphology claim。
- [ ] Lite3 真实实验总控。
- [ ] Introduction。
- [ ] Abstract。
- [ ] Discussion and Limitations。
- [ ] 整篇论文统一口径。
- [ ] 投稿。

你还要做两个关键决策：

1. 如果跨形态仿真结果很强，就把题目写成 `Cross-Morphology Visual Goal Navigation`。
2. 如果跨形态仿真只证明 adapter 可接入，就把题目降级为 `Budget-Aware Diffusion Visual Navigation on Quadruped and Simulated Robot Platforms`。

## 9. RA-L 论文建议结构

```text
I. Introduction
II. Related Work
III. Problem Formulation
IV. Method
    A. NoMaD-Style Visual Goal Navigation
    B. Budget-Aware Diffusion Inference
    C. Morphology-Specific Robot Adapters
    D. Real-Time Deployment on Lite3
V. Experiments
    A. Experimental Setup
    B. Time-Quality Trade-off of Diffusion Inference
    C. Budget-Aware Selector Evaluation
    D. Cross-Morphology Simulation
    E. Real-Robot Lite3 Navigation
VI. Discussion and Limitations
VII. Conclusion
```

## 10. 最后判断

这版 RA-L 论文的胜负手不是“做了多少工程”，而是能否把两个 claim 证明扎实：

1. **Budget-aware inference claim**：DDIM、CFG、TTS 不是三个零散技巧，而是一个可调度的时间-质量优化空间。
2. **Cross-morphology claim**：视觉导航策略输出的 waypoint 可通过形态相关 adapter zero-shot 接到不同机器人平台，真实 Lite3 是实机锚点，其他形态用仿真补足。

如果这两件事做实，这篇就从毕业设计部署报告，变成了一篇 RA-L 能认真审的机器人系统与学习论文。

