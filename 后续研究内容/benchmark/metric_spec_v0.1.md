# Benchmark Metric Specification v0.1

## Offline primary metrics

| Metric | Definition | Unit | Aggregation |
|---|---|---:|---|
| `action_ade_m` | 预测 waypoint 与 GT 在 horizon 上的平均 L2；生成式模型以固定 K 的 minADE@K 另报 | m | trajectory mean -> dataset macro |
| `action_fde_m` | 最后一个 waypoint 的 L2 | m | trajectory mean -> dataset macro |
| `heading_error_rad` | wrapped heading absolute error | rad | trajectory mean -> dataset macro |
| `progress_error_m` | predicted vs GT forward progress absolute error | m | trajectory mean -> dataset macro |
| `minade_k_m` | 固定 `candidate_count=K` 的 best-of-K ADE | m | 与 K、选择规则一起报告 |
| `candidate_diversity_m` | candidate waypoint pairwise distance 的均值 | m | 仅作多模态诊断，不定义越大越好 |

所有 action metric 必须转换回 meter/radian。禁止直接跨数据集平均 normalized action error。

## Robustness metrics

- `clean_relative_degradation = (corrupt_metric - clean_metric) / max(abs(clean_metric), eps)`，只用于“越低越好”的误差。
- 每个 corruption/severity 同时保留 absolute metric。
- 结果按 dataset/corruption/severity 分层，macro 平均不能掩盖单数据集崩溃。

## Closed-loop metrics

| Metric | Definition |
|---|---|
| `success_rate` | 规定 timeout 内进入 success radius，且无 disqualifying safety event |
| `spl` | 标准 success weighted by path length；最短路定义和地图版本固定 |
| `collision_rate` | 至少一次碰撞的 episode 比例；contact threshold 固定 |
| `stuck_rate` | 连续窗口内进度小于阈值的 episode 比例 |
| `fall_rate` | 触发姿态/高度 fall rule 的 episode 比例 |
| `recovery_rate` | 进入 recovery 的 episode 比例，不能当成功改进隐藏 |
| `time_to_goal_s` | 成功 episode 的 wall-clock time；失败另报 timeout，不删除 |
| `path_efficiency` | shortest path / executed path，范围 [0,1] |
| `loop_hz_mean` | 完整 perception-policy-bridge loop 的平均频率 |
| `loop_latency_p95_ms` | 每个完整 loop 单次 latency 的 p95；不得由 case mean 再取 p95 |

## Latency scope

`latency_scope` 必须是以下之一：

- `model_forward_per_call`
- `sampler_per_call`
- `full_loop_per_call`
- `case_level_mean`

`case_level_mean` 不能称为 tail latency。不同 scope 不能画在同一 Pareto axis 上而不分面。

## Statistical reporting

- offline CI：trajectory/session cluster bootstrap；至少 1000 bootstrap replicates。
- closed-loop CI：episode bootstrap，并保留 scene/seed pairing。
- training：报告 seed-level 原始值、mean、std/CI；不把 frame 数当独立样本数。
- primary test 在协议冻结时指定；其他 metric 标 secondary/exploratory，避免事后挑选。

## Invalid results

以下任一项使对比无效：

- meter scale、`dt` 或 action horizon 缺失；
- train/test leakage group 交叉；
- held-out dataset 参与 normalization 或 model selection；
- latency 缺 device/batch/precision/NFE/scope；
- closed-loop success 定义或 stabilizer mode 缺失；
- 只保留 aggregate、没有 per-trajectory/per-episode raw artifact。
