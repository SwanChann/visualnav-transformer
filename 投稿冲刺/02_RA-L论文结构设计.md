# RA-L 论文结构设计

更新日期：2026-07-07
目标：把本文设计成一篇短、硬、诚实、可审稿的 RA-L 论文，而不是毕业设计说明书或大而散的综述。

## 0. 论文一句话

> We study how a pretrained NoMaD-style diffusion visual navigation policy can be configured and deployed on a resource-constrained Lite3 quadruped by measuring the latency-quality trade-off of inference choices across offline, simulation, and frozen real-robot evidence.

中文意思：

> 我们不是重新训练一个新模型，而是研究已有 diffusion 视觉导航策略在 Lite3 上怎么选推理配置、怎么接控制栈、什么配置真正值得花算力。

## 1. 推荐题目

首选：

> Time-Quality Aware Diffusion Visual Navigation on a Quadruped Robot

更部署导向：

> Real-Time Deployment Analysis of Diffusion Visual Navigation on Lite3

更保守：

> Offline, Simulation, and Real-World Evidence for Diffusion Visual Navigation on a Quadruped Robot

不建议在标题中使用：

- foundation model
- cross-embodiment
- zero-shot quadruped navigation
- general robot navigation

除非后续补出远超当前证据的实验。

## 2. Abstract 结构

Abstract 应该按四句写。

第一句：背景和问题。

```text
Diffusion-based visual navigation policies can generate multimodal waypoint trajectories, but their iterative sampling cost makes real-time deployment on legged robots difficult.
```

第二句：本文场景。

```text
This paper studies the no-retraining deployment setting, where a pretrained NoMaD-style diffusion policy must be executed on a resource-constrained Lite3 quadruped under practical latency and safety constraints.
```

第三句：方法和实验。

```text
We unify DDIM sampling steps, classifier-free guidance, test-time candidate budget, and visual encoder choice into a time-quality inference space, and evaluate the resulting configurations through offline Pareto analysis, MuJoCo closed-loop validation, and frozen real-robot trials.
```

第四句：结论和边界。

```text
The results identify low-step DDIM with moderate candidate sampling as a practical deployment trade-off, while outdoor visual domain shift and limited real-robot statistics remain important limitations.
```

正式投稿前把 `practical deployment trade-off` 替换成最强数字，例如 `X ms`, `Y Hz`, `Z/Z MuJoCo success`, `N representative real-robot runs`。

## 3. Contributions

最终只写三条。

```text
This paper makes three contributions:
1) We present a Lite3 quadruped deployment stack for NoMaD-style diffusion visual goal navigation, including topological image-goal selection, waypoint-to-velocity adaptation, and safety-aware command execution.
2) We provide a systematic time-quality analysis of low-modification inference choices, including DDIM steps, classifier-free guidance, test-time candidate budget, and visual encoder variants.
3) We validate the selected configurations through offline Pareto analysis, MuJoCo closed-loop quadruped simulation, and frozen real-robot trials, with explicit discussion of failure modes and visual domain shift.
```

不要写第四条贡献。RA-L 篇幅很短，贡献越多越像摊大饼。

## 4. 全文结构和页数

目标 7 页左右，最多 8 页。

| 章节 | 页数 | 作用 |
|---|---:|---|
| I. Introduction | 0.8 | 提出问题、gap、贡献。 |
| II. Related Work | 0.8 | 三段式定位，不写大综述。 |
| III. System and Method | 1.6 | 系统、推理空间、adapter、安全桥。 |
| IV. Experiments | 2.6 | offline、MuJoCo、real robot、failure。 |
| V. Discussion and Limitations | 0.5 | 主动处理硬伤。 |
| VI. Conclusion | 0.2 | 收束。 |
| References | 1.0-1.3 | 关键近作必须引用。 |

## 5. Introduction 设计

Introduction 四段即可。

### 第一段：机器人视觉导航背景

写视觉目标导航和 visual navigation models 的意义。点到 GNM、ViNT、NoMaD 这一谱系，但不要展开细节。

核心句：

```text
Visual navigation models have made it possible to learn goal-conditioned navigation behaviors from heterogeneous robot trajectories, reducing the need for explicit mapping and task-specific policy design.
```

### 第二段：diffusion policy 的机会和问题

写 diffusion 能表达多模态 waypoint，但迭代采样导致 latency，四足机器人上尤其明显。

核心句：

```text
On legged robots, however, the benefit of multimodal action generation must be balanced against stale observations, command latency, and lower-level controller timing.
```

### 第三段：现有工作已经很强，但本文 gap 不同

写 NaviBridger、NavDP、SIDP、Fisher 等通过新训练、prior、critic、自模仿或 guidance 提升性能；本文关注固定策略、固定真机数据预算下的部署选择。

核心句：

```text
For many practical deployments, retraining a large navigation policy or collecting additional real-robot data is not feasible; the available question is which inference settings are worth their latency cost.
```

### 第四段：本文方案和贡献

引出 Lite3、DDIM/CFG/TTS/encoder、offline/MuJoCo/frozen real robot。

## 6. Related Work 设计

只写三段。

### 6.1 Visual Navigation Models

引用 GNM、ViNT、NoMaD。结尾落到：

```text
Our work builds on this line and uses a NoMaD-style policy as the backbone rather than proposing a new visual navigation model.
```

### 6.2 Diffusion Navigation and Inference-Time Adaptation

引用 Diffusion Policy、NaviBridger、NavDP、SIDP、NaviDiffusor、SRIF、Fisher-Preserving Guidance、RTI-DP。

结尾落到：

```text
In contrast to methods that retrain the policy or introduce additional guidance models, we study low-modification inference choices available in an existing navigation stack.
```

### 6.3 Legged Navigation and Deployment

引用 ANYmal Parkour、Motion Anisotropy Awareness、X-Nav、CE-Nav、Beyond Imitation。

结尾落到：

```text
We treat the quadruped interface as a deployment and evaluation problem, not as a claim of general cross-embodiment navigation.
```

## 7. Method 设计

### 7.1 Problem Formulation

定义变量：

```text
O_t: RGB observation history
G_t: selected topological goal image
A_t: generated waypoint sequence
a_t: selected executable waypoint
u_t = (v_x, v_y, omega_z): Lite3 command
B: latency budget
c = (encoder, sampler, steps, cfg_weight, tts_budget): inference configuration
```

优化视角：

```text
select c to maximize Q(c) subject to latency_p95(c) <= B
```

### 7.2 System Overview

Fig. 1 必须对应这条链：

```text
RGB camera + topomap
  -> goal image selection
  -> NoMaD-style visual encoder and distance head
  -> diffusion waypoint generator
  -> time-quality configuration table
  -> waypoint-to-velocity adapter
  -> Lite3 safety bridge and low-level control
```

### 7.3 NoMaD-Style Backbone

必须写清楚：

```text
The policy architecture and pretrained weights are not the contribution of this paper.
```

这里说明输入、输出和 topomap 机制即可，不要把 NoMaD 重新讲成自己的方法。

### 7.4 Time-Quality Inference Space

把四个旋钮讲清楚：

| 旋钮 | 作用 | 风险 |
|---|---|---|
| DDIM steps | 降低去噪步数和延迟。 | 步数过低可能损害 waypoint 质量。 |
| CFG weight | 增强目标条件影响。 | 权重过大可能牺牲平滑和鲁棒性。 |
| TTS budget | 多候选采样，提高找到好轨迹概率。 | 延迟线性或近线性增加。 |
| Encoder | 改变视觉表征和速度。 | 强 encoder 未必闭环更稳。 |

质量 proxy：

```text
Q = w_p * forward_progress
  - w_s * smoothness_cost
  - w_i * invalid_or_saturation
  + w_d * useful_diversity
```

### 7.5 Adapter and Safety Bridge

写工程要点：

- waypoint 转 body-frame velocity。
- 速度限幅。
- 角速度限幅。
- 低通滤波。
- NaN/Inf guard。
- stale frame protection。
- command timeout。
- heartbeat。
- emergency stop。

这部分是 RA-L 机器人味道最强的地方。

## 8. Experiments 设计

四个 research questions。

```text
RQ1: Which inference configurations form the latency-quality Pareto frontier offline?
RQ2: Do offline-selected configurations remain executable in Lite3 MuJoCo closed loop?
RQ3: What onboard timing and qualitative evidence are available from frozen real-robot trials?
RQ4: Where does the system fail, and which failures require retraining, guidance, or world models?
```

### 8.1 Offline Pareto

输入：

- DDIM results。
- CFG results。
- TTS results。
- encoder results。
- joint ablation results。

输出：

- Fig. 2：latency-quality Pareto。
- Table II：代表配置和指标。

核心结论写法：

```text
Low-step DDIM provides the largest latency reduction, while moderate candidate sampling improves trajectory quality within a usable control frequency. Higher guidance or larger candidate budgets should be treated as quality modes rather than default deployment settings.
```

### 8.2 MuJoCo Closed-Loop

输入：

- DDIM step sweep。
- CFG sweep。
- encoder closed-loop sweep。
- easy/medium/hard scenes。

输出：

- Fig. 3：路径或 final distance。
- Table III：success/final distance/path length/time/stuck。

必须写：

```text
MuJoCo results validate closed-loop integration and relative configuration behavior; they should not be interpreted as replacing large-scale real-world evaluation.
```

### 8.3 Frozen Real-Robot Evidence

输入：

- `history-doc/毕设冲刺/video/`
- `timing_profile.csv`
- videos/keyframes。

输出：

- Table IV：scene、configuration、manual outcome、latency、Hz、failure mode。
- Fig. 4：Lite3 platform、成功帧、失败帧、timing 小图。

必须写：

```text
The real-robot trials are frozen representative validation, not a statistically conclusive benchmark.
```

### 8.4 Failure and Domain Shift

分类：

- strong illumination。
- grass / complex texture。
- repetitive corridor。
- open-space goal ambiguity。
- topomap mismatch。
- low control frequency。
- stale observation / command delay。

结尾落到 future work：

- RL fine-tuning。
- real-to-sim adaptation。
- cost-guided diffusion。
- ranking / verifier。
- world-model foresight。

## 9. 图表结构

| 编号 | 内容 | 目的 |
|---|---|---|
| Fig. 1 | 系统总览 | 让审稿人一眼知道系统链路。 |
| Fig. 2 | Offline latency-quality Pareto | 本文主结果。 |
| Fig. 3 | MuJoCo closed-loop paths/results | 证明离线推荐能闭环执行。 |
| Fig. 4 | Frozen real robot frames and timing | 真实系统 anchor validation。 |
| Table I | Platform/system parameters | 说明硬件和控制频率。 |
| Table II | Offline inference configs | 支撑 time-quality claim。 |
| Table III | MuJoCo closed-loop | 支撑 closed-loop claim。 |
| Table IV | Frozen real robot evidence | 支撑真实部署 claim。 |

## 10. Discussion 和 Limitations

必须主动承认：

1. NoMaD-style policy 不是本文训练的。
2. 真机实验是冻结代表性 trial，不是大规模统计。
3. MuJoCo route stabilizer 如果启用，只证明系统闭环集成，不证明纯 policy 最优。
4. Outdoor domain shift 没有被解决。
5. 本文不 claim cross-embodiment zero-shot。

推荐句：

```text
This study should be read as a deployment-time analysis of a frozen diffusion visual navigation policy, rather than as a new navigation foundation model or a large-scale real-world benchmark.
```

## 11. 审稿问题防御

| 问题 | 回答方向 |
|---|---|
| Is this just NoMaD engineering? | NoMaD trains the policy; this paper studies Lite3 deployment, inference budget, and offline-sim-real evidence. |
| Why are DDIM/CFG/TTS contributions? | They are not claimed as new algorithms; the contribution is measured deployment selection under latency constraints. |
| Why no more real robot trials? | Real trials are frozen; quantitative conclusions come from offline and simulation, real trials anchor deployment. |
| Why not compare against NavDP/SIDP? | They require different training/data pipelines; we address fixed-policy no-retraining deployment. |
| Does MuJoCo stabilizer inflate results? | It is reported as an integration aid; claims are framed as closed-loop validation, not pure policy optimality. |
| Does this generalize across robots? | No broad claim; Lite3 adapter is platform-specific. |

## 12. 最小可投标准

缺任何一项都不建议投稿：

- [ ] Fig. 1 系统图。
- [ ] Fig. 2 offline Pareto。
- [ ] Table II 代表推理配置。
- [ ] Table III MuJoCo 闭环结果。
- [ ] Table IV 冻结真机 evidence。
- [ ] Limitation 段落。
- [ ] Related Work 覆盖 GNM/ViNT/NoMaD/NavDP/SIDP/NaviBridger/Fisher 或 SRIF/ANYmal。
- [ ] 全文没有 cross-embodiment 或 SOTA 过强 claim。
