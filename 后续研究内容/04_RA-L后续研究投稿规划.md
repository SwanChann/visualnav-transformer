# 下一篇 RA-L 投稿规划

更新日期：2026-07-10

## 0. 投稿判断

这是一篇 **后续 RA-L** 的规划，不是当前 投稿冲刺/ 中 frozen-policy 部署稿的扩展章节。

推荐状态：

> 研究方向黄绿灯，论文尚未立项完成。先通过数据和 baseline gate，再正式进入模型开发。

RA-L 接受创新方法和有意义的 application case study，但要求简洁、原创、有完整结果。当前计划只有在“跨数据集诊断 + 小型生成策略 + 闭环验证”三者形成因果链时才适合 RA-L。

## 1. 推荐题目

主标题：

> Compact Generative Visual Navigation Policies Under Cross-Dataset and Compute Constraints

备选：

> TinyNavBrain: Diagnostic Multi-Dataset Training for Resource-Aware Visual Navigation

不建议在标题中使用：

- foundation model。
- generalist robot brain。
- universal navigation。
- zero-shot cross-embodiment。
- world model。

## 2. 一句话问题

> How should a compact generative visual navigation policy be trained and evaluated across heterogeneous robot datasets so that transfer gains survive visual shift, action-scale mismatch, compute limits, and closed-loop deployment?

## 3. 三条贡献

### Contribution 1：training-side diagnostic benchmark

一个控制训练数据、模型规模和推理预算的跨数据集诊断协议，分离：

- visual domain shift。
- action scale/dt。
- observation/action history。
- data mixture。
- generative head/NFE。

不能写“首个通用 VNM benchmark”；2026 年已有直接 VNM 真实评测。

### Contribution 2：compact generative policy

一个小型 observation-goal-action-conditioned generative waypoint policy，在相同参数量和计算预算下支持：

- one/few-step generation。
- multi-modal waypoint chunks。
- explicit action-history/physical-scale conditioning。
- calibrated progress/confidence。

action history、scale conditioning 和 few-step 本身不是独立创新；贡献必须来自整体训练/模型设计带来的跨数据集与闭环收益。

### Contribution 3：evidence chain

用三层证据回答同一问题：

1. 多数据集 offline/held-out transfer。
2. corruption 与 failure diagnostics。
3. randomized closed-loop simulation + end-device timing。

若未来允许新增真机，补充最小 Lite3 controlled trials；若仍禁止新增真机，只能使用冻结素材作 anchor，并在 limitation 中承认真实统计不足。

## 4. 强 baseline

正文至少包含：

| Baseline | 目的 |
|---|---|
| published frozen NoMaD | 现有模型锚点 |
| Go-Stanford-only NoMaD | 单数据 specialist |
| multi-dataset NoMaD | 排除“只是多数据带来收益” |
| compact deterministic model | 排除“生成式并非必要” |
| DDIM-2/3 same-budget | 延迟基线 |
| action-history/scale baseline | 对齐 VISTA 问题 |
| TinyNavBrain NFE=1/2/4 | 主模型 |

可选：

- consistency/OneDP-style student。
- learned ranker。
- encoder adapter。

它们不能取代 multi-dataset NoMaD 和 deterministic same-budget baseline。

## 5. 实验矩阵

### E1：数据诊断

- dataset size。
- action magnitude/turn distribution。
- dt/waypoint spacing。
- image statistics。
- goal distance distribution。
- mixture sampling exposure。

### E2：single vs multi-dataset

~~~
single specialist
proportional mixture
balanced mixture
temperature mixture
~~~

报告所有 dataset 的结果，不只挑平均分。

### E3：leave-one-dataset-out

至少 3 个数据集才成立。核心结果：

~~~
train on N-1 -> test on held-out dataset
~~~

如果只有 2 个数据集，这更接近 cross-domain transfer case study，不能称 multi-dataset benchmark。

### E4：生成范式

- deterministic。
- DDPM/DDIM。
- flow/rectified flow。
- NFE=1/2/4/8。

固定 encoder、参数量和训练 steps。

### E5：corruption

- blur。
- sunflare/brightness。
- crop/FOV。
- stale frame/frame drop。
- action-scale mismatch。

### E6：closed-loop

- 20+ evaluation seeds。
- turn/obstacle/recovery scenes。
- success、SPL、collision、stuck、fall。
- latency injection。
- end-to-end control rate。

### E7：deployment

- workstation GPU latency。
- Jetson Orin end-to-end latency。
- total/trainable parameters。
- memory and NFE。

## 6. 图表规划

| 编号 | 内容 |
|---|---|
| Fig. 1 | TinyNavBrain + data/benchmark/system overview |
| Fig. 2 | dataset action/visual statistics and split design |
| Fig. 3 | cross-dataset transfer matrix |
| Fig. 4 | accuracy/coverage vs NFE/latency Pareto |
| Fig. 5 | closed-loop success/collision/failure cases |
| Table I | datasets, splits, scales, licenses |
| Table II | same-budget model comparison |
| Table III | leave-one-dataset-out results |
| Table IV | closed-loop and deployment timing |

RA-L 篇幅有限，主文不应把所有方向都放入。完整 CSV 和代码可发布到外部 repository，但关键定义、baseline 和主结果必须在正文内。

## 7. 论文结构

建议 6–8 页：

1. Introduction：0.7 页。
2. Related Work：0.5 页。
3. Problem and Diagnostic Protocol：1.0 页。
4. Compact Generative Policy：1.2 页。
5. Experiments：2.5–3.5 页。
6. Limitations and Conclusion：0.4 页。

实验应占最大篇幅。

## 8. Go / No-Go gates

### Gate A：数据

Go：

- 至少 3 个 processed datasets。
- split 和许可清楚。
- 有 held-out dataset。

No-Go：

- 只有 Go Stanford。
- 数据无法重现或存在严重泄漏。

### Gate B：baseline

Go：

- multi-dataset NoMaD 和 deterministic baseline 稳定。
- 至少 3 seeds。

No-Go：

- baseline 本身无法复现。
- 改善小于 seed variance。

### Gate C：方法

Go：

- 至少两个 held-out/shift setting 稳定提升。
- same-budget 比较成立。
- confidence 或 generative diversity 有独立价值。

No-Go：

- 只在 IID 提升。
- 仅靠更多参数/steps。
- 只比 published checkpoint，不比 retrained baseline。

### Gate D：闭环

Go：

- success/SPL/collision 至少一项有统计改善。
- 结果在 randomized seeds 下稳定。

No-Go：

- 继续使用当前饱和直线走廊作为主要闭环证据。
- offline 提升无法转化为行为收益。

### Gate E：投稿

Go：

- 三条贡献都有直接证据。
- 代码、split、配置和 seeds 可复现。
- 主图在 6–8 页内可读。

No-Go：

- 论文必须依赖大量补充材料才能解释方法。
- 仍需用“brain/foundation/generalist”包装有限结果。

## 9. 风险与降级路线

| 失败情况 | 降级路线 |
|---|---|
| 无法获得 3+ datasets | 做 corruption/scale diagnostic，不写 multi-dataset |
| generative head 不优于 deterministic | 转成 negative study 或停止模型论文 |
| flow 只提升 latency | 写 efficiency study，弱化 brain claim |
| offline 好、closed-loop 无提升 | 暂缓 RA-L，修 benchmark/控制接口 |
| 无新真机 | 明确 simulation-primary，冻结真机仅 anchor |
| benchmark 与现有工作重叠 | 强化 training-mixture/compute-controlled diagnostic，不写通用评测 |

## 10. 24 周执行计划

| 周期 | 目标 | 交付物 |
|---|---|---|
| W1–4 | 数据获取与 audit | manifest、license、splits、stats |
| W5–8 | baseline | single/multi NoMaD、deterministic |
| W9–12 | TinyNavBrain pilot | model、NFE ablation、sanity results |
| W13–16 | cross-dataset/corruption | transfer matrix、failure analysis |
| W17–20 | closed-loop/deployment | randomized benchmark、timing |
| W21–24 | paper | figures、tables、RA-L draft |

时间表是 gate-driven，不是承诺。任何 gate 失败时先修问题，不按日期强行进入下一阶段。

## 11. 当前第一步

下一步不是训练：

1. 建立数据 acquisition/license matrix。
2. 选择 RECON + SACSoN/SCAND 中至少两个可获取数据集。
3. 编写统一 manifest/audit 设计。
4. 建立 multi-dataset NoMaD baseline config。
5. 重新设计非饱和 closed-loop benchmark。

完成这些后，再实现 TinyNavBrain。

## 12. 官方与核心参考

- RA-L Information for Authors: https://www.ieee-ras.org/publications/ra-l/ra-l-information-for-authors/
- ViNT: https://arxiv.org/abs/2306.14846
- NoMaD: https://arxiv.org/abs/2310.07896
- NavDP: https://arxiv.org/abs/2505.08712
- Can Vision Foundation Models Navigate?: https://arxiv.org/abs/2603.25937
- VISTA: https://arxiv.org/abs/2606.17294
- NavOL: https://arxiv.org/abs/2605.11762
- NavWAM: https://arxiv.org/abs/2606.13494
