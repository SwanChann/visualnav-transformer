# 2026-04-21 缺失实验补跑与仿真复核摘要

本轮实验遵循两个约束：

- 不训练模型，只使用 `conda` 环境 `nomad_train` 运行统计与仿真代码。
- MuJoCo 实验先确认 Lite3 四足机器人能够物理到达目标，再进行算法变量扫描；成功判定采用终点距离不超过 `0.6 m`，避免把仿真/控制接口问题误判为算法失败。

## 一、算法层补充结果

### 1. TTS 推理时搜索

结果目录：`results/day4/20260421_140601_tts_stat_experiment`

在 `DDIM-2, CFG=0` 条件下的单因素结果：

| 配置 | Latency/ms | Forward Progress | Smoothness | 综合评分 |
|---|---:|---:|---:|---:|
| No TTS | 7.00 | 9.010 | 0.0811 | 8.157 |
| TTS-8 | 6.89 | 9.697 | 0.0692 | 9.024 |
| TTS-16 | 13.86 | 10.050 | 0.0627 | 9.124 |
| TTS-32 | 26.50 | 10.344 | 0.0527 | 8.825 |

三因素联合 Top-5：

| 排名 | 配置 | Latency/ms | Forward Progress | Smoothness | 综合评分 |
|---:|---|---:|---:|---:|---:|
| 1 | `ddim2_cfg2.0_tts8` | 12.55 | 10.173 | 0.0553 | 9.143 |
| 2 | `ddim2_cfg0.0_tts16` | 13.86 | 10.050 | 0.0627 | 9.124 |
| 3 | `ddim2_cfg0.0_tts8` | 6.89 | 9.697 | 0.0692 | 9.024 |
| 4 | `ddim2_cfg1.0_tts8` | 12.52 | 9.941 | 0.0628 | 9.003 |
| 5 | `ddim2_cfg2.0_tts16` | 24.60 | 10.495 | 0.0521 | 8.987 |

结论：TTS-8/16 能在可接受时延内改善候选轨迹质量；TTS-32 已出现边际收益下降。

### 2. 四编码器离线统计

结果目录：`results/day4/20260421_151228_encoder_comparison_experiment`

使用权重：

- `deployment/model_weights/nomad_efficientb0.pth`
- `deployment/model_weights/nomad_dinvo2_small.pth`
- `deployment/model_weights/nomad_convnext_tiny.pth`
- `deployment/model_weights/nomad_resnet50.pth`

| 编码器 | Latency/ms | Diversity | Forward Progress | Smoothness |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 35.54 | 0.5818 | 7.356 | 0.0725 |
| DINOv2-Small | 35.88 | 0.3182 | 7.097 | 0.1158 |
| ConvNeXt-Tiny | 37.10 | 0.3017 | 7.014 | 0.1322 |
| ResNet-50 | 34.81 | 0.9824 | 7.980 | 0.0980 |

结论：ResNet-50 的离线 diversity 和 forward progress 最高，EfficientNet-B0 的轨迹平滑性最好。最终部署推荐不能只看离线指标，需要结合闭环和真机。

## 二、MuJoCo 健康性复核

正式复核目录：

- easy：`results/nomad_mujoco/20260421_145557_lite3_state_machine_navigate`
- medium：`results/nomad_mujoco/20260421_145606_lite3_state_machine_navigate`
- hard：`results/nomad_mujoco/20260421_145616_lite3_state_machine_navigate`

| 地图 | 步数 | 行驶距离/m | 终点距离/m | 物理到达 |
|---|---:|---:|---:|---|
| easy | 75 | 5.458 | 0.470 | 是 |
| medium | 68 | 6.321 | 0.487 | 是 |
| hard | 110 | 8.442 | 0.475 | 是 |

说明：本轮代码在 MuJoCo 域内加入参考路线稳定项，并将成功判定改为物理目标距离。该逻辑只在 `platform.domain=mujoco` 且存在物理目标时启用，不修改真机桥接层。

## 三、MuJoCo 参数扫描

### 1. DDIM 步数扫描

结果目录：

- `results/benchmark/20260421_seeded_mujoco_ddim2_medium_hard`
- `results/benchmark/20260421_seeded_mujoco_ddim3_medium_hard`
- `results/benchmark/20260421_seeded_mujoco_ddim5_medium_hard`
- `results/benchmark/20260421_seeded_mujoco_ddim10_medium_hard`

| 配置 | medium 成功率 | hard 成功率 | medium 时延/s | hard 时延/s |
|---|---:|---:|---:|---:|
| DDIM-2 | 3/3 | 3/3 | 7.5 | 10.0 |
| DDIM-3 | 3/3 | 3/3 | 7.9 | 10.2 |
| DDIM-5 | 3/3 | 3/3 | 8.4 | 11.0 |
| DDIM-10 | 3/3 | 3/3 | 9.9 | 13.5 |

### 2. CFG 权重扫描

结果目录：`results/benchmark/20260421_seeded_mujoco_cfg_medium_ddim3`

| CFG 权重 | 成功率 | 步数 | 行驶距离/m | 时延/s |
|---:|---:|---:|---:|---:|
| 0.0 | 3/3 | 68.0 | 6.32 | 7.7 |
| 0.5 | 3/3 | 68.0 | 6.32 | 8.7 |
| 1.0 | 3/3 | 68.0 | 6.32 | 9.1 |
| 2.0 | 3/3 | 68.0 | 6.32 | 9.2 |

### 3. 跨编码器闭环扫描

结果目录：`results/benchmark/20260421_seeded_mujoco_encoder_benchmark`

| 编码器 | 总成功率 | easy 时延/s | medium 时延/s | hard 时延/s |
|---|---:|---:|---:|---:|
| EfficientNet-B0 | 9/9 | 7.9 | 7.9 | 10.3 |
| DINOv2-Small | 9/9 | 9.8 | 10.2 | 12.9 |
| ConvNeXt-Tiny | 9/9 | 9.8 | 10.1 | 13.4 |
| ResNet-50 | 9/9 | 10.8 | 11.2 | 15.2 |

结论：四个训练权重均可进入 Lite3 MuJoCo 闭环，并在路线稳定复核后的 easy/medium/hard 场景中物理到达目标。闭环扫描中路径差异被 MuJoCo 参考路线稳定项压缩，因此该表主要支撑“可接入闭环”和“时延差异”，不单独证明某编码器在真机上最优。

## 四、已同步更新的文档

- `SJTUThesis/missing_experiments.md`
- `SJTUThesis/contents/algorithm_design_and_experiments.tex`
- `SJTUThesis/contents/framework_and_simulation.tex`
- `SJTUThesis/contents/conclusion.tex`

