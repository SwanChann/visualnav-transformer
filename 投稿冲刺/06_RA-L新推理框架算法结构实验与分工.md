# RA-L 新推理框架、算法结构、实验矩阵与分工

生成日期：2026-05-25  
目标：主投 IEEE Robotics and Automation Letters (RA-L)  
本文档作用：作为后续 RA-L 重构的主方案文件。

## 0. 当前决策

这篇论文不能再写成“把 NoMaD 部署到 Lite3 的系统报告”。RA-L 需要一个更明确的研究结构。

建议把论文重构为：

> 面向实时机器人导航的预算感知扩散推理框架，在固定计算预算下统一调度 DDIM 加速、CFG 引导和测试时候选扩展，并通过仿真与 Lite3 真机验证速度和结果质量的双重约束。

建议框架名：

> **TQ-Nav: Time-Quality Aware Diffusion Navigation**

TQ-Nav 的主张：

1. **扩散推理不是固定采样流程，而是可调度的时间-质量优化过程。**
2. **DDIM、CFG、best-of-N/TTS 不是零散 trick，而是同一个预算分配问题的三个旋钮。**
3. **推理层应从单次采样器升级为“候选生成 + 质量验证 + 预算控制 + 安全回退”的闭环决策模块。**
4. **仿真和真机实验要同时证明：该框架更快，并且在给定预算下不牺牲导航成功率。**

## 1. 前沿调研结论

### 1.1 你的论文已有文献清单

以下来自 `论文/refs.bib` 和正文引用。后续 RA-L 写作时应保留核心文献，删除只服务毕业论文背景的弱相关中文综述或放到少量引用中。

#### 扩散模型与扩散策略

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `ddpm2020` | Denoising Diffusion Probabilistic Models | DDPM 基础扩散模型，作为原始采样链基础。 |
| `nichol2021improved` | Improved Denoising Diffusion Probabilistic Models | 噪声调度和扩散模型改进背景。 |
| `ddim2020` | Denoising Diffusion Implicit Models | 少步采样加速基础，是 TQ-Nav 的速度旋钮。 |
| `cfg2022` | Classifier-Free Diffusion Guidance | 条件增强基础，是 TQ-Nav 的质量旋钮。 |
| `ttsdiffusion2025` | Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps | 测试时扩展思想：用额外候选搜索提升结果质量。 |
| `ttssearch2025` | Inference-time Scaling of Diffusion Models through Classical Search | 将扩散测试时控制表述为搜索问题，支撑我们的 verifier/selector 设计。 |
| `diffuser2022` | Planning with Diffusion for Flexible Behavior Synthesis | 轨迹级扩散规划代表工作。 |
| `chi2025diffusion` | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion | 动作扩散策略代表工作，说明扩散可用于视觉运动控制。 |

#### 视觉导航与拓扑导航

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `gnm2022` | GNM: A General Navigation Model to Drive Any Robot | 跨机器人视觉导航基线思想。 |
| `vint2023` | ViNT: A Foundation Model for Visual Navigation | NoMaD 的视觉导航基础结构来源。 |
| `nomad2023` | NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration | 本文高层策略基础，必须明确继承关系。 |
| `shah2021ving` | ViNG: Learning Open-World Navigation with Visual Goals | 视觉目标导航和拓扑导航背景。 |
| `savinov2018sptm` | Semi-Parametric Topological Memory for Navigation | 拓扑记忆导航背景。 |
| `chaplot2020nts` | Neural Topological SLAM for Visual Navigation | 学习式拓扑导航背景。 |
| `zhu2017targetdriven` | Target-Driven Visual Navigation in Indoor Scenes Using Deep Reinforcement Learning | 早期图像目标导航背景。 |
| `ddppo2020` | DD-PPO | 点目标导航和大规模 RL 背景。 |
| `anderson2018evaluation` | On Evaluation of Embodied Navigation Agents | embodied navigation 评价指标背景。 |

#### 视觉编码器和模型基础

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `transformer2017` | Attention Is All You Need | Transformer 基础。 |
| `vit2020` | An Image is Worth 16x16 Words | ViT 视觉编码背景。 |
| `dit2022` | Scalable Diffusion Models with Transformers | diffusion + Transformer 背景。 |
| `efficientnet2019` | EfficientNet | NoMaD/ViNT 默认视觉骨干。 |
| `resnet2016` | ResNet | 视觉骨干对比。 |
| `convnext2022` | ConvNeXt | 视觉骨干对比。 |
| `dinov2_2023` | DINOv2 | 自监督视觉特征对比。 |
| `timm2019` | PyTorch Image Models | 视觉骨干实现来源。 |

#### 机器人系统、SLAM、仿真与四足平台

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `thrun2005probabilistic` | Probabilistic Robotics | 机器人导航基础。 |
| `orbslam3_2021` | ORB-SLAM3 | 传统视觉 SLAM 背景。 |
| `cartographer2016` | Cartographer | 传统 2D LiDAR SLAM 背景。 |
| `mujoco2012` | MuJoCo | 仿真实验平台基础。 |
| `hwangbo2019anymal` | Learning Agile and Dynamic Motor Skills for Legged Robots | 四足运动控制背景。 |
| `lee2020learning` | Learning Quadrupedal Locomotion over Challenging Terrain | 四足地形适应背景。 |
| `rl_locomotion_rudin2022` | Learning to Walk in Minutes Using Massively Parallel Deep RL | 四足 RL 运动控制背景。 |
| `miki2022perceptive` | Learning Robust Perceptive Locomotion for Quadrupedal Robots in the Wild | 野外感知运动背景。 |
| `ren2023hvn` | Hierarchical Vision Navigation System for Quadruped Robots with Foothold Adaptation Learning | 四足视觉导航相关工作。 |
| `kareer2023vinl` | ViNL: Visual Navigation and Locomotion Over Obstacles | 视觉导航和运动结合背景。 |
| `dippest2024` | DiPPeST | 四足机器人扩散路径规划直接相关工作。 |
| `quarvla2024` | QUAR-VLA | 四足 VLA 相关工作。 |
| `saro2025` | SARO | VLM/空间感知四足系统相关工作。 |

#### 2025-2026 视觉导航扩散新工作

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `navibridger2025` | NaviBridger: Visual Navigation via Denoising Diffusion Bridge Models | 指出从高斯噪声到动作分布可能低效，支持我们讨论采样效率。 |
| `navdp2025` | NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance | 强相关竞争工作：sim-to-real、跨形态、diffusion navigation。 |
| `sidp2026` | Self-Imitated Diffusion Policy for Efficient and Robust Visual Navigation | 强相关竞争工作：减少 generate-then-filter 依赖，提升效率。 |

#### 中文综述与背景

| Bib key | 文献 | 在本文中的作用 |
|---|---|---|
| `pei2026embodied_nav` | 具身导航概念、方法与挑战 | 中文毕业论文背景，RA-L 中可不引或少引。 |
| `mao2022slam_review` | 惯性/视觉/激光雷达 SLAM 技术综述 | 中文背景，RA-L 中可不引。 |
| `chen2025object_nav_review` | 面向具身人工智能的物体目标导航综述 | 中文背景，RA-L 中可不引。 |
| `sima2023vln_review` | 视觉语言导航研究进展 | 中文背景，RA-L 中可不引。 |
| `yang2019quadruped_review` | 四足机器人研究综述 | 中文背景，RA-L 中可不引。 |

### 1.2 前沿 diffusion policy 与测试时扩展方向

下面是后续前期调研必须细读的文献组。

#### A. 动作扩散策略基础

- **Diffusion Policy**：把扩散过程放在 robot action space，解决多模态动作分布和长时序动作生成问题。对本文的启发是：NoMaD 的 waypoint diffusion 可以被视为低维 action diffusion，适合做采样调度和候选筛选。
- **Diffuser**：把 trajectory generation 表述为扩散规划问题。对本文的启发是：verifier 可以被看作一种测试时 reward/guidance，但我们不训练新 planner。
- **NoMaD**：goal-masked diffusion policy 同时处理导航和探索。对本文的启发是：已有 goal mask 使 CFG 和 goal-conditioned/unconditioned 分支天然可用。

#### B. 快速扩散策略

- **DDIM**：通过跳步采样降低扩散步数，不改训练权重；适合我们当前代码和模型。
- **Consistency Policy**：把 diffusion policy 蒸馏成 consistency model，实现单步或少步控制。对本文启发大，但需要重新训练/蒸馏，短期可以作为 discussion 或 future work。
- **One-Step Diffusion Policy (OneDP)**：从预训练 diffusion policy 蒸馏单步 action generator，报告显著速度提升。对本文启发是：长期可以把 TQ-Nav 的最优配置蒸馏成一阶段快速模型。
- **One-Step Flow Policy / Flow matching policy**：2026 年继续强化单步生成趋势。对本文启发是：RA-L 首稿不建议引入新训练目标，否则工作量过大。

#### C. 测试时扩展和搜索

- **Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps**：不仅增加 denoising steps，还可以搜索更好的初始噪声和候选。
- **Inference-time Scaling through Classical Search**：将扩散推理控制组织为 local/global search。对本文启发是：TTS 不应只是 blind best-of-N，而应有预算、分支、早停和 verifier。
- **Dynamic Search / Feynman-Kac steering 类方法**：说明测试时 reward/verifier 可以动态影响粒子筛选。短期不实现复杂版本，但可作为理论背景。

#### D. 导航扩散新工作

- **NaviBridger**：指出从高斯噪声开始生成动作可能增加冗余 denoising，并引入 bridge prior。对本文启发：我们可用上一时刻动作、拓扑方向、PD 可执行性作为候选先验，而不是每次盲采。
- **NavDP**：用模拟 privileged information 训练 navigation diffusion policy，报告跨 quadruped、wheeled、humanoid 的 zero-shot real-world 泛化。它是最强竞争工作之一。我们的差异必须写清楚：我们不主张大规模训练，而主张在已有 NoMaD-style policy 上做实时预算调度和真实四足部署。
- **SIDP**：通过 reward-guided self-imitation 让 policy 自身更稳定，减少 generate-then-filter 需求。它说明单纯 TTS 不是终点。我们的短期策略是：先把测试时调度做实；长期可以把 verifier 选择结果反哺训练。
- **DiPPeST**：四足机器人扩散路径规划，证明 diffusion planning 可以进入 quadruped 场景。我们的差异是视觉目标导航、拓扑目标选择和实时预算调度。

### 1.3 参考来源链接

访问日期：2026-05-25

- Diffusion Policy IJRR: https://journals.sagepub.com/doi/10.1177/02783649241273668
- NoMaD project: https://general-navigation-models.github.io/nomad/
- NoMaD arXiv: https://arxiv.org/abs/2310.07896
- DDIM OpenReview: https://openreview.net/forum?id=St1giarCHLP
- Inference-Time Scaling for Diffusion Models: https://arxiv.org/abs/2501.09732
- Inference-time Scaling through Classical Search: https://arxiv.org/abs/2505.23614
- Classical search project page: https://diffusion-inference-scaling.github.io/
- Consistency Policy: https://arxiv.org/abs/2405.07503
- One-Step Diffusion Policy: https://arxiv.org/abs/2410.21257
- NaviBridger CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Ren_Prior_Does_Matter_Visual_Navigation_via_Denoising_Diffusion_Bridge_Models_CVPR_2025_paper.html
- NavDP: https://arxiv.org/abs/2505.08712
- SIDP: https://arxiv.org/abs/2601.22965
- DiPPeST: https://arxiv.org/abs/2405.19232
- GNM: https://arxiv.org/abs/2210.03370
- ViNT: https://arxiv.org/abs/2306.14846
- Open X-Embodiment: https://huggingface.co/papers/2310.08864
- XSkill: https://arxiv.org/abs/2307.09955

## 2. 新框架：TQ-Nav

### 2.1 框架目标

TQ-Nav 要解决的问题：

> 在边缘计算平台上，扩散视觉导航每次决策的计算预算有限。系统必须在给定预算内输出可执行、高质量、安全的局部航点，而不是固定使用某一种采样器。

输入：

- 历史观察图像 `O_t = {I_{t-K}, ..., I_t}`。
- 拓扑子目标图像 `G_t`。
- 当前拓扑窗口和距离预测。
- 机器人状态和上一时刻动作。
- 实时预算 `B_t`，例如 80 ms / 120 ms / 200 ms。

输出：

- 被选中的局部航点序列 `A_t = {a_t^1, ..., a_t^H}`。
- 中层执行 waypoint `a_t^k`。
- 推理配置和质量评分日志。

核心模块：

```text
Observation / Goal
    -> Visual Condition Encoder
    -> Distance and Progress Estimator
    -> Budget Controller
    -> Multi-Fidelity Diffusion Sampler
    -> Candidate Verifier
    -> Budgeted Selector
    -> Safety Fallback
    -> Robot Adapter
```

### 2.2 与当前 NoMaD 推理模块的区别

当前推理模块主要是：

```text
fixed config -> sample actions -> optional TTS -> choose candidate -> execute
```

新模块应改为：

```text
state + budget + risk
    -> choose sampling schedule
    -> generate candidates progressively
    -> verify candidates
    -> early stop if quality sufficient or budget nearly exhausted
    -> execute best safe candidate
    -> log result for Pareto calibration
```

关键变化：

1. 采样器从固定配置变成 `SamplerBank`。
2. TTS 从固定 best-of-N 变成 progressive candidate expansion。
3. CFG 从默认开关变成风险触发的质量增强项。
4. verifier 从简单启发式分数变成可消融的候选质量模型。
5. 系统加入 hard latency guard 和 safe fallback。

### 2.3 预算控制器

预算控制器输入：

- 当前目标距离。
- topomap 指针是否稳定。
- 距离预测置信度或窗口内距离差。
- 上一时刻候选质量。
- 机器人是否接近障碍/转角/开阔区域。
- 最近几帧推理延迟。

输出模式：

| 模式 | 预算 | 默认配置 | 使用场景 |
|---|---:|---|---|
| Fast | 80 ms | DDIM-2, CFG=0, TTS=1 或 4 | 直线、低风险、需要高频控制 |
| Balanced | 120 ms | DDIM-2/3, CFG=0, TTS=8 | 默认部署 |
| Quality | 200 ms | DDIM-3/4, CFG=1/2, TTS=8/16 | 转角、目标混淆、距离预测不稳定 |
| Fallback | 20 ms | 上一安全动作或停止/低速 | 超预算、候选全失败、相机异常 |

第一版建议实现静态选择器：

```text
if latency_budget <= 80 ms:
    Fast
elif risk_score < threshold:
    Balanced
else:
    Quality
```

第二版再考虑动态策略：

```text
risk_score = f(distance_margin, topomap_progress, action_smoothness, verifier_conflict)
budget = g(risk_score, current_control_rate, safety_state)
```

### 2.4 多保真扩散采样器

SamplerBank 包含：

- `DDPMSampler`：原始基线。
- `DDIMSampler(S)`：少步采样，S = 1/2/3/4/8。
- `GuidedSampler(w)`：CFG，w = 0/1/2。
- `ProgressiveTTSSampler(N)`：逐批生成候选，N = 1/4/8/16。
- `WarmStartSampler`：可选，用上一时刻候选或拓扑方向初始化噪声/动作。

Progressive TTS 的逻辑：

```text
best = None
for batch in [1, 3, 4, 8]:
    candidates += sample(batch)
    scores = verifier(candidates)
    best = argmax(scores)
    if score(best) > tau_good:
        break
    if elapsed_time > budget_margin:
        break
return best
```

这样可以避免固定 TTS-16 在简单场景中浪费计算。

### 2.5 候选验证器

Verifier 不是学习模型也可以先成立。第一版用可解释启发式：

```text
score(A) =
    + w_forward * forward_progress(A)
    - w_lateral * lateral_deviation(A)
    - w_smooth * curvature_or_jerk(A)
    - w_reverse * backward_motion(A)
    - w_saturation * adapter_saturation(A)
    - w_topomap * topomap_regression_risk(A)
```

候选必须通过 hard filters：

- 非 NaN/Inf。
- waypoint 范围不超过中层控制能力。
- yaw/linear command 不超过限幅。
- 不出现明显后退或横向跳变。
- 与上一执行命令差异不超过安全阈值。

后续可加 learned critic：

- 输入视觉条件、candidate trajectory、topomap distance。
- 输出 success/progress/risk。
- 训练标签来自仿真 rollout 或真实 trial。

RA-L 首稿建议使用 heuristic verifier + 消融，避免新增训练负担。

### 2.6 速度和结果双重保证

这里的“保证”应写成工程意义上的 bounded behavior，不要写成理论最优保证。

速度保证：

- hard latency budget。
- progressive sampling early stop。
- 超预算时 fallback 到上一安全动作或低速停止。
- 每次记录 p50/p90/p95 latency。

结果质量保证：

- 候选必须通过 verifier hard filters。
- 在预算内最大化 quality score。
- 简单场景不浪费计算，困难场景使用额外预算。
- 通过 offline Pareto、simulation rollout、real robot trials 三层验证。

### 2.7 建议代码结构

建议新增目录：

```text
scripts/shared/tq_nav/
    config.py
    engine.py
    budget.py
    samplers.py
    verifier.py
    selector.py
    adapters.py
    logger.py
```

建议职责：

| 文件 | 职责 |
|---|---|
| `config.py` | 定义 `InferenceConfig`, `BudgetMode`, `TQNavConfig`。 |
| `engine.py` | 顶层 `TQNavEngine`，接收图像/目标/预算，输出 waypoint。 |
| `budget.py` | 风险估计和预算模式选择。 |
| `samplers.py` | 封装 DDPM/DDIM/CFG/TTS/progressive sampling。 |
| `verifier.py` | 候选打分和 hard filters。 |
| `selector.py` | Pareto 表加载、候选选择、early stop。 |
| `adapters.py` | Lite3/differential-drive/holonomic adapter。 |
| `logger.py` | 统一记录 latency、score、config、candidate、result。 |

需要改造现有文件：

| 文件 | 改造内容 |
|---|---|
| `scripts/shared/nomad_inference.py` | 保留底层 encode / distance / diffusion primitive，减少直接承担策略选择。 |
| `scripts/deployment/nomad_navigation_host.py` | 从调用固定 NoMaD inference 改为调用 `TQNavEngine`。 |
| `scripts/simulation/lite3_system/interfaces.py` | 允许接入不同 `RobotAdapter` 和预算模式。 |
| `scripts/nomad_eval_common.py` | 增加 TQ-Nav offline eval hooks。 |

## 3. 实验设计

### Experiment 1：离线 Pareto 前沿

目的：

> 找到扩散推理配置在 latency-quality 空间中的有效前沿，为 budget selector 提供依据。

配置矩阵：

| 维度 | 取值 |
|---|---|
| sampler | DDPM, DDIM |
| DDIM steps | 1, 2, 3, 4, 8 |
| CFG weight | 0, 1, 2 |
| TTS budget | 1, 4, 8, 16 |
| progressive TTS | off, on |
| verifier | forward only, full heuristic |

指标：

- latency mean/p50/p90/p95。
- candidate score。
- forward progress。
- lateral deviation。
- smoothness。
- action diversity。
- invalid candidate rate。
- Pareto dominated ratio。

输出：

- `table_inference_pareto.tex`
- `fig_latency_quality_pareto.pdf`
- `fig_budget_modes.pdf`

成功标准：

- 至少找出 3 个预算模式下的非支配配置。
- 证明固定 DDPM 或固定 DDIM/TTS 不是全局最优。
- 证明 progressive TTS 能在相近质量下降低平均延迟，或在相近延迟提高质量。

### Experiment 2：Budget-aware selector 离线回放

目的：

> 验证 TQ-Nav 的选择器能根据场景风险和预算选择不同推理配置。

做法：

- 使用已有 dataset cases。
- 构造风险等级：straight / turn / ambiguous goal / low distance margin。
- 对每个 case 运行 fixed configs 和 TQ-Nav selector。

对比：

| 方法 | 说明 |
|---|---|
| DDPM baseline | 原始慢速高质量基线。 |
| DDIM-2 fixed | 固定快速基线。 |
| DDIM-2 + TTS-8 fixed | 当前强基线。 |
| TQ-Nav static | 根据预算表选择配置。 |
| TQ-Nav progressive | 根据 verifier 和预算早停。 |

指标：

- normalized quality。
- latency。
- quality per millisecond。
- risk-conditioned success proxy。

### Experiment 3：MuJoCo 闭环仿真

目的：

> 验证离线 Pareto 结论能否转化为闭环导航性能。

场景：

- easy corridor。
- medium turn。
- hard long corridor / ambiguous segment。
- 可选：open-space / distractor goal。

方法：

- DDPM baseline。
- DDIM-2 fixed。
- DDIM-2 + TTS-8 fixed。
- TQ-Nav static。
- TQ-Nav progressive。

指标：

- success rate。
- final distance。
- steps/time to goal。
- path length。
- stuck/fall count。
- average inference latency。
- budget violation rate。
- verifier rejection rate。

注意：

- 如果使用 route stabilizer，必须在表注中明确。
- 最好补一个 stabilizer off 或 weak-stabilizer 实验，否则仿真结果只能说明系统集成，不说明纯策略能力。

### Experiment 4：跨形态 adapter 仿真

目的：

> 验证高层 waypoint diffusion policy 与机器人形态解耦，TQ-Nav 输出可以通过 adapter 接到不同平台。

最小平台：

| 平台 | 状态 |
|---|---|
| Lite3 quadruped | 现有 MuJoCo 或当前系统。 |
| Differential-drive | 新增简化仿真。 |
| Holonomic base | 新增简化仿真。 |

统一输入：

- 同一拓扑目标序列。
- 同一冻结视觉导航模型。
- 同一 budget selector。

只变：

- kinematics/dynamics。
- adapter。
- control limits。

指标：

- success rate。
- final distance。
- tracking error。
- command saturation ratio。
- path efficiency。
- failure mode。

写作边界：

- 如果新仿真不是视觉闭环，只能声称 `adapter-level cross-morphology validation`。
- 如果能接入视觉闭环，才能更强地写 `cross-morphology visual navigation`。

### Experiment 5：Jetson Orin 推理剖析

目的：

> 验证 TQ-Nav 满足真实板端实时预算。

指标：

- image capture。
- preprocessing。
- visual encoder。
- distance prediction。
- diffusion sampling。
- verifier。
- selector。
- adapter。
- end-to-end。
- p50/p90/p95 latency。

方法：

- 每种配置至少 1000 次循环。
- 分别测 Fast/Balanced/Quality。
- 报告预算违例率。

### Experiment 6：Lite3 真机导航

目的：

> 证明 TQ-Nav 在真实四足机器人上能够达到速度和结果质量的双重约束。

配置：

| 方法 | 作用 |
|---|---|
| DDPM baseline | 原始扩散基线。 |
| DDIM-2 fixed | 快速基线。 |
| DDIM-2 + TTS-8 fixed | 当前强基线。 |
| TQ-Nav static | 预算表选择器。 |
| TQ-Nav progressive | 最终方法。 |

场景：

- 室内直线。
- 室内转角。
- 室内长走廊。
- 可选：开阔区域或光照变化。

指标：

- success rate。
- time to goal。
- final distance。
- intervention count。
- stuck/collision/fall。
- average control Hz。
- p90 inference latency。
- budget violation rate。

最低规模：

- 每种方法每个场景 5 次，最低 75 trials。
- 更稳规模：每种方法每个场景 10 次，150 trials。

### Experiment 7：Outdoor zero-shot stress test

目的：

> 不是证明 outdoor 完全可用，而是给出真实泛化边界。

要做：

- 简单室外路面成功例。
- 强光失败。
- 草地/复杂纹理失败。
- 开阔区域目标混淆失败。

写作方式：

> Outdoor trials show that the inference and deployment stack remains executable, but visual representation robustness remains the limiting factor under severe domain shift.

## 4. RA-L 论文结构

建议标题：

> TQ-Nav: Time-Quality Aware Diffusion Inference for Real-Time Visual Navigation

备选：

> Budgeted Test-Time Scaling for Diffusion-Based Visual Navigation on Quadruped Robots

章节：

```text
I. Introduction
II. Related Work
    A. Diffusion Policies for Robotics
    B. Test-Time Scaling and Fast Diffusion Inference
    C. Visual Navigation and Quadruped Deployment
III. Method
    A. Problem Formulation
    B. NoMaD-Style Action Diffusion Backbone
    C. Time-Quality Aware Inference
    D. Candidate Verification and Budgeted Selection
    E. Robot Adapters and Real-Time Deployment
IV. Experiments
    A. Offline Pareto Analysis
    B. Budget-Aware Selector Evaluation
    C. MuJoCo Closed-Loop Simulation
    D. Cross-Morphology Adapter Simulation
    E. Jetson Orin and Lite3 Real-Robot Evaluation
V. Discussion and Limitations
VI. Conclusion
```

## 5. 两位同学的新分工

### 5.1 总原则

你的要求是合理的：两位同学不能一个只写文章、一个只做实验。新的分工原则：

1. 前期调研两个人都参与。
2. 推理算法架构设计两个人都参与。
3. 中期代码实现两个人都参与，但模块不同。
4. 后期实验分开：一个主离线推理实验，一个主仿真实验。
5. 真机实验由你主导，两人按各自模块参与。
6. 每个人都要产出文献综述、代码、实验数据、图表和论文小节。

### 5.2 前期共同任务

时间：2026-05-25 至 2026-06-02

两人共同完成：

- [ ] 读完你论文现有 refs.bib，按本文件第 1.1 节分类。
- [ ] 每人至少精读 10 篇核心文献。
- [ ] 一起整理 `related_work_matrix.md`。
- [ ] 一起输出 `gap_to_tq_nav.md`。
- [ ] 一起讨论 TQ-Nav 的模块边界。
- [ ] 一起画第一版算法框图。
- [ ] 一起确定 verifier score。
- [ ] 一起确定实验矩阵和日志字段。

共同交付物：

- `related_work_matrix.md`
- `tq_nav_gap_analysis.md`
- `tq_nav_algorithm_diagram_v1.pdf`
- `experiment_matrix_v1.md`
- `logging_schema_v1.md`

### 5.3 调研细分

两人都参与，但各有主责。

#### 同学 A 主责：扩散推理与测试时扩展

必读：

- DDPM。
- Improved DDPM。
- DDIM。
- CFG。
- Diffuser。
- Diffusion Policy。
- NoMaD。
- Inference-Time Scaling beyond denoising steps。
- Inference-time Scaling through Classical Search。
- Consistency Policy。
- One-Step Diffusion Policy。
- SIDP。

输出：

- 每篇 300-500 字中文总结。
- 论文与 TQ-Nav 的关系。
- 可实现点和不可实现点。
- 推荐引用句。

#### 同学 B 主责：视觉导航、跨平台和机器人系统

必读：

- GNM。
- ViNT。
- NoMaD。
- ViNG。
- SPTM。
- Neural Topological SLAM。
- NavDP。
- NaviBridger。
- DiPPeST。
- Open X-Embodiment。
- XSkill。
- ViNL。
- QUAR-VLA。
- 四足感知运动相关文献。

输出：

- 每篇 300-500 字中文总结。
- 与 TQ-Nav 的差异。
- 机器人平台、实验规模、是否真机。
- 可作为 baseline 或 discussion 的点。

### 5.4 算法设计共同任务

时间：2026-06-03 至 2026-06-08

两人共同参与：

- [ ] 定义 `TQNavEngine` 输入输出。
- [ ] 定义 `InferenceConfig`。
- [ ] 定义 `BudgetController`。
- [ ] 定义 `SamplerBank`。
- [ ] 定义 `CandidateVerifier`。
- [ ] 定义 `BudgetedSelector`。
- [ ] 定义 `RobotAdapter`。
- [ ] 写伪代码。
- [ ] 写方法部分初稿。

你负责最终拍板：

- 是否做 learned critic。
- 是否做 warm start。
- 是否做 dynamic budget。
- 是否把跨形态作为强 claim。

### 5.5 后期任务拆分

#### 同学 A：离线推理实验负责人

时间：2026-06-09 至 2026-06-24

职责：

- [ ] 实现 offline Pareto eval。
- [ ] 实现 progressive TTS。
- [ ] 实现 budget selector 回放。
- [ ] 跑 DDIM/CFG/TTS/Verifier 消融。
- [ ] 跑 Orin timing profile。
- [ ] 生成 Pareto frontier 图。
- [ ] 生成 budget mode 表。
- [ ] 写 `Time-Quality Aware Inference` 方法小节。
- [ ] 写 offline experiments 小节。

交付物：

- `inference_pareto_raw.csv`
- `inference_pareto_summary.csv`
- `budget_replay_results.csv`
- `orin_timing_profile.csv`
- `fig_pareto_frontier.pdf`
- `table_budget_modes.tex`
- `sec_time_quality_inference.tex`

#### 同学 B：仿真实验与 adapter 负责人

时间：2026-06-09 至 2026-06-24

职责：

- [ ] 实现或整理 `RobotAdapter`。
- [ ] 跑 Lite3 MuJoCo。
- [ ] 做 differential-drive 简化仿真。
- [ ] 做 holonomic 简化仿真。
- [ ] 跑 TQ-Nav vs fixed configs 的闭环对比。
- [ ] 统计 success/final distance/tracking error/saturation。
- [ ] 生成跨形态轨迹图。
- [ ] 写 `Robot Adapters and Deployment` 方法小节。
- [ ] 写 simulation experiments 小节。

交付物：

- `mujoco_tqnav_raw.csv`
- `mujoco_tqnav_summary.csv`
- `cross_morphology_raw.csv`
- `cross_morphology_summary.csv`
- `fig_mujoco_paths.pdf`
- `fig_cross_morphology_paths.pdf`
- `table_adapter_results.tex`
- `sec_robot_adapters.tex`

### 5.6 真机实验分工

时间：2026-06-25 至 2026-07-05

你主导：

- 场地、安全、任务定义、最终运行、失败判定。

同学 A：

- 负责真机推理配置、latency 记录、budget violation、日志检查。

同学 B：

- 负责 topomap/场景记录、adapter 参数、视频、轨迹和失败案例标注。

共同交付：

- `real_robot_tqnav_raw.csv`
- `real_robot_tqnav_summary.csv`
- `real_robot_failure_cases.md`
- `fig_real_robot_results.pdf`
- `supp_video_anonymous.mp4`

## 6. 时间表

| 日期 | 任务 |
|---|---|
| 2026-05-25 至 2026-06-02 | 前期共同调研，整理文献矩阵和 gap。 |
| 2026-06-03 至 2026-06-08 | 共同设计 TQ-Nav 算法结构和接口。 |
| 2026-06-09 至 2026-06-16 | 完成核心代码改造和第一批离线/仿真结果。 |
| 2026-06-17 至 2026-06-24 | 完成完整离线 Pareto、budget selector、MuJoCo 和跨形态仿真。 |
| 2026-06-25 至 2026-07-05 | Lite3 真机实验和 outdoor stress test。 |
| 2026-07-06 至 2026-07-12 | 图表、视频、论文初稿。 |
| 2026-07-13 至 2026-07-20 | 内部审稿、压页数、补缺口。 |
| 2026-07-21 至 2026-07-25 | RA-L 投稿准备。 |

## 7. 必须避免的风险

1. **不能把 best-of-N 写成算法创新本身。** 创新应是预算感知、渐进式候选扩展、verifier 和闭环验证。
2. **不能过度声称理论保证。** 写 bounded latency 和 empirical quality guarantee。
3. **不能把跨形态仿真写过头。** 如果不是视觉闭环，必须叫 adapter-level validation。
4. **不能只做离线分数。** RA-L 必须有闭环仿真和真实机器人。
5. **不能把两位同学拆成“文书”和“苦力”。** 每个人都必须有技术产出。

## 8. 最终论文 claim 草案

可以写：

> We propose TQ-Nav, a time-quality aware inference framework for diffusion-based visual navigation. Instead of using a fixed diffusion sampler, TQ-Nav treats DDIM acceleration, classifier-free guidance, and test-time candidate expansion as budgeted inference actions. A progressive candidate generator and verifier select executable waypoint trajectories under a hard latency budget. We validate the framework through offline Pareto analysis, closed-loop simulation, cross-morphology adapter tests, and real-world Lite3 quadruped navigation.

不要写：

> We propose a new diffusion policy architecture.

除非你真的重新训练了模型结构。

不要写：

> We guarantee optimal navigation quality.

应该写：

> We empirically improve the latency-quality trade-off while enforcing a hard inference budget.

