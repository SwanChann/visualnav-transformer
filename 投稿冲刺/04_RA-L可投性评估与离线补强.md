# RA-L 可投性评估与离线补强

更新时间：2026-07-09

## 0. 总判断

结论：**黄灯可投，不建议立刻换成全新工作。**

当前工作不能写成新 foundation model、跨形态 zero-shot 或 SOTA navigation policy，但可以写成一篇诚实的 RA-L 实证论文：

> fixed NoMaD-style diffusion visual navigation policy + Lite3 quadruped deployment + inference-time latency-quality analysis + offline / MuJoCo / frozen real-robot evidence.

它的录用机会取决于三件事：

1. 离线结果是否能形成清楚的 Pareto frontier，而不是零散消融。
2. MuJoCo 是否能证明离线推荐配置在 Lite3 闭环中可执行，同时诚实说明 route stabilizer 的边界。
3. 真机材料是否能作为 representative anchor validation，而不是被写成大规模统计结论。

如果这三件事整理完整，这篇稿子有 RA-L 投稿价值；如果 Fig. 2 / Table II / Table III 做不扎实，就不建议投稿。

## 1. 为什么不是绿灯

审稿人最容易质疑：

- **novelty 偏系统实证**：DDIM、CFG、TTS、encoder 都不是新算法。
- **真机统计不足**：已有 Lite3 视频和 timing 可以证明跑通过，但不能支撑强泛化结论。
- **MuJoCo 结果受 stabilizer 影响**：它能支撑 closed-loop integration，不能证明 policy 本身最优。
- **前沿工作很强**：NavDP、SIDP、NaviBridger、NaviDiffusor、Fisher Guidance、SRIF 等已经在训练、guidance、critic、self-imitation 或 safety ranking 上走得更远。

所以当前论文必须把 claim 压窄：

> We study deployment-time inference configuration for a frozen diffusion visual navigation policy under quadruped latency and safety constraints.

不要把论文写成：

> We improve diffusion navigation policy.

## 2. 为什么不是红灯

本地已经有足够多的资产支撑一篇实证型论文的骨架：

| 证据层 | 已有资产 | 当前价值 | 风险 |
|---|---|---|---|
| Offline | DDIM、CFG、joint DDIM-CFG、TTS、encoder、encoder joint CSV/JSON | 支撑 time-quality Pareto 主结果 | 需要统一指标和质量代理 |
| MuJoCo | seeded DDIM / CFG / encoder benchmark 的 raw、aggregated、report | 支撑 Lite3 closed-loop executability | 必须说明 stabilizer 和仿真边界 |
| Real robot | 多个 Lite3 视频、interactive summary、部分 timing profile | 支撑真实部署和失败案例 | 只能做代表性证据，不能做统计 benchmark |
| System | Lite3 bridge、adapter、topomap、safety guard 代码链路 | RA-L 机器人味道来源 | 需要图和表说明工程约束 |

这说明现在不需要马上改做 navigation brain / ranker / distillation。那些是下一篇或 future work，不应把当前 RA-L 主线搅散。

## 3. 当前 RA-L 的最小可录用证据门槛

投稿前必须满足：

| 门槛 | 文件/图表 | 通过标准 |
|---|---|---|
| G1 | Fig. 1 system diagram | 清楚区分 pretrained NoMaD-style backbone、本工作部署栈、inference knobs、Lite3 safety bridge |
| G2 | Fig. 2 offline Pareto | 至少覆盖 DDPM/DDIM、CFG、TTS、encoder/joint ablation；标出 Fast/Balanced/Quality |
| G3 | Table II offline configs | 每行含 latency mean/p50/p90/p95 或可解释替代、quality proxy、num cases、CI |
| G4 | Table III MuJoCo | 对离线推荐配置做 closed-loop 汇总，含 success/final distance/path length/wall time/stabilizer note |
| G5 | Table IV real robot | 只列 frozen representative trials，人工标注 success/partial/failure/unusable 和 failure mode |
| G6 | Limitations | 明确不 claim new policy、large-scale real-world benchmark、cross-embodiment 或 solved domain shift |

缺 G2 或 G3：不投。

缺 G4：不投 RA-L，可以转 workshop / technical report。

缺 G5：论文仍能写，但 RA-L 机器人说服力会明显下降。

## 4. 离线实验必须补强什么

### 4.1 统一离线配置表

把以下结果合成一个统一表，而不是分散引用：

```text
results/day2/20260410_143136_ddim_stat_experiment/ddim_overall_summary.csv
results/day3/20260410_143225_cfg_stat_experiment/cfg_overall_summary.csv
results/day4/20260410_143709_joint_ddim_cfg_experiment/overall_summary.csv
results/day4/20260421_140601_tts_stat_experiment/tts_overall_summary.csv
results/day4/20260421_151228_encoder_comparison_experiment/encoder_overall_summary.csv
results/day5/20260422_171021_encoder_joint_ablation_experiment/joint_overall_summary.csv
```

输出：

```text
投稿冲刺/workspace/offline_pareto/offline_pareto_unified.csv
投稿冲刺/workspace/offline_pareto/table_offline_pareto.md
投稿冲刺/workspace/offline_pareto/fig_offline_pareto.pdf
投稿冲刺/workspace/offline_pareto/budget_modes.md
```

### 4.2 质量代理不要只用一个分数

正文可以画一个主质量分数，但表格必须保留分项：

```text
forward_progress
smoothness
lateral_abs
diversity
invalid_or_saturation
endpoint_norm
```

如果某个结果没有 `invalid_or_saturation`，就写 `NA`，不要临时编一个指标。

主质量分数建议只作为排序辅助：

```text
Q = z(forward_progress)
  - z(smoothness)
  - z(lateral_abs)
  + 0.5 * z(useful_diversity)
```

`useful_diversity` 不能简单等于越大越好；如果 diversity 大但 endpoint 或 lateral drift 变差，要在讨论里说明。

### 4.3 必须给出 latency budget modes

不要只报告“最佳配置”。RA-L 更容易接受“部署预算下怎么选”：

| 模式 | 作用 | 候选来源 |
|---|---|---|
| Baseline | 原始扩散采样参考 | DDPM / DDIM-10 |
| Fast | 控制频率优先 | DDIM-1/2, CFG=0, TTS=0/1 |
| Balanced | 默认部署推荐 | DDIM-2, CFG=0, TTS=8 或离线 Pareto 等价配置 |
| Quality | 复杂场景候选 | DDIM-2, CFG=1/2, TTS=8/16 |

正文写法应该是：

> The recommended configuration depends on the latency budget; we therefore report budget modes rather than a single universally optimal setting.

### 4.4 TTS 只能写成 budgeted candidate search

已有 TTS 结果很有用，但不要 claim 新 verifier。当前 `tts_verifier=heuristic`，所以合理 claim 是：

> Moderate candidate budgets improve offline trajectory proxies within an acceptable latency budget, but large budgets show diminishing returns.

不要写：

> We propose a learned verifier or safe ranker.

learned ranker 可以作为下一篇工作，不进入当前 RA-L 的 contribution。

### 4.5 Encoder 结果要谨慎解释

离线 encoder 结果和 MuJoCo encoder 结果可能不完全一致。建议写成：

> Encoder choice affects latency and offline trajectory proxies, but closed-loop performance can be dominated by control interface and simulation stabilizer effects.

因此 encoder 不应作为本文最强 claim，只作为 inference configuration 的一个维度。

## 5. 是否引入新工作

### 不建议现在引入为主贡献

| 方向 | 当前 RA-L 是否引入 | 原因 |
|---|---|---|
| learned ranker / verifier | 不作为主贡献 | 会把论文从 deployment analysis 改成新算法论文，需闭环证明 |
| few-step / consistency distillation | 不作为主贡献 | 需要训练、teacher/student、额外消融，容易拖垮首投 |
| navigation brain / world-action model | 不引入 | 这是下一阶段课题，不适合塞进 7-8 页 RA-L |
| RL / preference fine-tuning | 不引入 | 工程量和不稳定性高，且改变 fixed-policy 设定 |

### 可以放入 Discussion / Future Work

把失败案例自然引到：

- learned verifier/ranker for TTS candidates
- cost-guided diffusion
- world-model foresight
- few-step distillation for real-time diffusion navigation
- RL/preference fine-tuning for domain shift

但正文实验保持当前主线：

> no retraining, low-modification inference configuration, Lite3 deployment evidence.

## 6. 最小投稿版实验包

只做整理，不新增真机、不训练：

1. 生成 `offline_pareto_unified.csv`、Fig. 2、Table II。
2. 汇总 seeded MuJoCo benchmark，生成 Table III。
3. 人工标注 frozen real-robot trials，生成 Table IV。
4. 从已有视频抽关键帧，生成 Fig. 4。
5. 写 limitation，主动说明每层证据的边界。

这个版本可以投稿，但 reviewer 可能会认为 novelty 较弱。

## 7. 增强投稿版实验包

在不改变主线的前提下，最多补这些：

1. **Offline sensitivity**：用 bootstrap 或 seed split 检查 Pareto frontier 是否稳定。
2. **Budget ablation**：固定 DDIM-2，比较 `TTS=0/8/16/32` 的边际收益和 latency。
3. **Cross-level consistency**：把 Table II 的 Fast/Balanced/Quality 映射到 MuJoCo Table III。
4. **Failure-linked analysis**：把真机失败帧对应到 offline/MuJoCo 无法覆盖的因素，如 illumination、grass、open-space ambiguity。

不要补会改变论文类型的东西。

## 8. 如果补强失败怎么办

如果统一离线表做不出稳定 Pareto，或者 MuJoCo 结果只能证明“靠 stabilizer 走到目标”，那么不建议硬投 RA-L。

替代路线：

1. 把当前工作收束成技术报告 / workshop。
2. 用现有 TTS 和 Pareto 资产启动下一篇：learned verifier / candidate ranker。
3. 或做 few-step / consistency distillation，把当前 latency-quality 研究升级为真正算法贡献。

但截至当前本地资产检查，**还没到必须换题的程度**。
