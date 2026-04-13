# MuJoCo 闭环导航基准实验：多视觉编码器对比

> 生成时间：2026-04-14 01:24:25

## 1. 实验设计

### 1.1 实验目标

在 MuJoCo 仿真环境中，使用统一的闭环导航流程（NoMaD + Lite3 四足机器人），
对比 4 种不同视觉编码器训练的策略在导航任务中的表现差异。

### 1.2 编码器配置

| 编码器 | 类型 | 参数量级 | 图像尺寸 | 训练配置 |
|---|---|---|---|---|
| EfficientNet-B0 | CNN (baseline) | 5.3M | 96×96 | bs=32, lr=1e-4 |
| DINOv2-Small | ViT (自监督预训练) | 22M | 98×98 | bs=32, lr=5e-5, frozen backbone |
| ResNet-50 | CNN (ImageNet 预训练) | 25.6M | 96×96 | bs=32, lr=5e-5, frozen backbone |
| ConvNeXt-Tiny | 现代 CNN (ImageNet-12k 预训练) | 28.6M | 96×96 | bs=32, lr=5e-5, frozen backbone |

### 1.3 实验场景

| 地图 | 起点 | 终点 | 直线距离 | 障碍物 | 难度 |
|---|---|---|---|---|---|
| easy | (0,0) | (5,0) | 5.0m | 0 | 无障碍直行 |
| medium | (0,0) | (7,0) | 7.0m | 1 | 单障碍绕行 |
| hard | (0,0) | (8,0) | 8.0m | 2 | 双障碍 S 形绕行 |

### 1.4 评价指标解释

| 指标 | 含义 | 好的方向 |
|---|---|---|
| **Success Rate (成功率)** | 机器人最终位置距目标 < 0.5m 的比例 | ↑ 越高越好 |
| **Steps (步数)** | NoMaD 决策循环次数；越少说明导航越高效 | ↓ 越少越好（成功前提下） |
| **Path Distance (路径距离)** | 机器人实际行走的总路径长度 (m) | ↓ 越短越好（接近直线距离） |
| **Final Goal Dist (终点到目标距离)** | 导航结束时机器人与目标的欧氏距离 (m) | ↓ 越小越好 |
| **Wall Time (实际耗时)** | 单次运行的总墙钟时间 (s)，反映推理效率 | ↓ 越短越好 |

## 2. 总体结果

### 2.1 按编码器汇总（所有地图平均）

| 编码器 | 总成功率 | 平均步数 | 平均路径距离 | 平均终点距离 | 平均耗时 |
|---|---|---|---|---|---|
| EfficientNet-B0 (baseline) | 53/54 (98%) | 72.9 | 7.41 m | 0.48 m | 10.9 s |
| ConvNeXt-Tiny | 18/54 (33%) | 224.5 | 16.06 m | 4.21 m | 22.6 s |

### 2.2 详细结果（按编码器 × 地图）

| 编码器 | 地图 | 调度器 | 成功率 | 步数 (mean±std) | 路径距离 (mean±std) | 终点距离 (mean±std) | 耗时 |
|---|---|---|---|---|---|---|---|
| ConvNeXt-Tiny | easy | ddim-2 | 3/3 (100.0%) | 153.3±90.3 | 12.69±6.5 m | 1.16±0.82 m | 13.2 s |
| ConvNeXt-Tiny | easy | ddim-2 | 3/3 (100.0%) | 38.3±0.9 | 3.72±0.12 m | 1.53±0.08 m | 8.4 s |
| ConvNeXt-Tiny | easy | ddim-2 | 3/3 (100.0%) | 197.0±59.6 | 13.98±3.8 m | 2.15±0.75 m | 18.0 s |
| ConvNeXt-Tiny | easy | ddpm | 0/3 (0.0%) | 242.7±79.7 | 15.96±4.21 m | 3.57±2.3 m | 20.1 s |
| ConvNeXt-Tiny | easy | ddpm | 2/3 (66.7%) | 278.7±16.3 | 18.57±0.84 m | 1.18±0.47 m | 31.1 s |
| ConvNeXt-Tiny | easy | ddpm | 2/3 (66.7%) | 114.3±95.7 | 8.29±5.16 m | 1.7±0.83 m | 16.1 s |
| ConvNeXt-Tiny | hard | ddim-2 | 0/3 (0.0%) | 299.0±0.0 | 21.91±3.74 m | 3.11±1.74 m | 24.1 s |
| ConvNeXt-Tiny | hard | ddim-2 | 0/3 (0.0%) | 257.3±58.9 | 18.07±6.48 m | 9.27±0.95 m | 24.4 s |
| ConvNeXt-Tiny | hard | ddim-2 | 0/3 (0.0%) | 255.7±61.3 | 18.49±4.26 m | 2.9±1.19 m | 24.5 s |
| ConvNeXt-Tiny | hard | ddpm | 0/3 (0.0%) | 243.7±43.8 | 15.95±2.52 m | 10.54±1.88 m | 22.2 s |
| ConvNeXt-Tiny | hard | ddpm | 0/3 (0.0%) | 299.0±0.0 | 16.3±1.77 m | 8.31±1.27 m | 30.6 s |
| ConvNeXt-Tiny | hard | ddpm | 0/3 (0.0%) | 299.3±0.5 | 20.63±3.79 m | 5.28±2.92 m | 35.7 s |
| ConvNeXt-Tiny | medium | ddim-2 | 1/3 (33.3%) | 218.7±113.6 | 17.75±8.58 m | 3.97±2.6 m | 16.9 s |
| ConvNeXt-Tiny | medium | ddim-2 | 2/3 (66.7%) | 172.7±84.8 | 15.21±7.01 m | 2.15±0.8 m | 17.5 s |
| ConvNeXt-Tiny | medium | ddim-2 | 0/3 (0.0%) | 183.7±83.5 | 16.95±6.02 m | 3.47±0.45 m | 19.3 s |
| ConvNeXt-Tiny | medium | ddpm | 0/3 (0.0%) | 289.7±13.2 | 21.09±4.4 m | 7.07±2.29 m | 24.6 s |
| ConvNeXt-Tiny | medium | ddpm | 1/3 (33.3%) | 227.7±100.9 | 16.09±6.55 m | 1.66±0.64 m | 28.5 s |
| ConvNeXt-Tiny | medium | ddpm | 1/3 (33.3%) | 270.0±38.9 | 17.37±5.72 m | 6.71±2.83 m | 30.9 s |
| EfficientNet-B0 (baseline) | easy | ddim-2 | 3/3 (100.0%) | 43.3±0.5 | 4.65±0.04 m | 0.32±0.05 m | 7.3 s |
| EfficientNet-B0 (baseline) | easy | ddim-2 | 3/3 (100.0%) | 45.0±1.6 | 4.84±0.16 m | 0.33±0.14 m | 8.4 s |
| EfficientNet-B0 (baseline) | easy | ddim-2 | 2/3 (66.7%) | 129.7±119.7 | 11.63±9.61 m | 0.6±0.52 m | 14.0 s |
| EfficientNet-B0 (baseline) | easy | ddpm | 3/3 (100.0%) | 44.7±0.9 | 4.8±0.1 m | 0.34±0.03 m | 10.2 s |
| EfficientNet-B0 (baseline) | easy | ddpm | 3/3 (100.0%) | 43.3±0.5 | 4.64±0.02 m | 0.22±0.03 m | 12.5 s |
| EfficientNet-B0 (baseline) | easy | ddpm | 3/3 (100.0%) | 43.7±0.5 | 4.68±0.04 m | 0.24±0.07 m | 10.5 s |
| EfficientNet-B0 (baseline) | hard | ddim-2 | 3/3 (100.0%) | 121.3±65.5 | 12.73±6.56 m | 2.63±3.01 m | 10.4 s |
| EfficientNet-B0 (baseline) | hard | ddim-2 | 3/3 (100.0%) | 73.7±1.9 | 7.92±0.19 m | 0.42±0.06 m | 9.1 s |
| EfficientNet-B0 (baseline) | hard | ddim-2 | 3/3 (100.0%) | 69.7±1.2 | 7.47±0.11 m | 0.39±0.07 m | 8.7 s |
| EfficientNet-B0 (baseline) | hard | ddpm | 3/3 (100.0%) | 144.0±83.8 | 13.29±7.42 m | 0.5±0.09 m | 14.1 s |
| EfficientNet-B0 (baseline) | hard | ddpm | 3/3 (100.0%) | 72.0±2.2 | 7.74±0.25 m | 0.36±0.02 m | 12.3 s |
| EfficientNet-B0 (baseline) | hard | ddpm | 3/3 (100.0%) | 76.7±13.0 | 7.7±0.64 m | 0.55±0.15 m | 12.5 s |
| EfficientNet-B0 (baseline) | medium | ddim-2 | 3/3 (100.0%) | 96.7±45.6 | 8.32±1.87 m | 0.41±0.2 m | 10.2 s |
| EfficientNet-B0 (baseline) | medium | ddim-2 | 3/3 (100.0%) | 60.3±0.5 | 6.46±0.03 m | 0.37±0.03 m | 9.6 s |
| EfficientNet-B0 (baseline) | medium | ddim-2 | 3/3 (100.0%) | 61.3±1.2 | 6.61±0.14 m | 0.22±0.13 m | 8.7 s |
| EfficientNet-B0 (baseline) | medium | ddpm | 3/3 (100.0%) | 62.3±0.5 | 6.67±0.03 m | 0.23±0.06 m | 10.6 s |
| EfficientNet-B0 (baseline) | medium | ddpm | 3/3 (100.0%) | 61.7±0.5 | 6.61±0.03 m | 0.28±0.03 m | 13.3 s |
| EfficientNet-B0 (baseline) | medium | ddpm | 3/3 (100.0%) | 62.3±1.2 | 6.7±0.11 m | 0.21±0.07 m | 13.7 s |

## 3. 结果分析

### 3.1 指标解读

- **成功率 (Success Rate)**：最核心的指标。100% 意味着该编码器在该地图上所有运行都能到达目标，
  0% 意味着该编码器在该难度下无法完成导航。成功判定：机器人最终位置距目标 < 0.5m。

- **步数 (Steps)**：在成功情况下，步数越少代表导航路径越优。如果失败（达到 max_steps 上限），
  步数等于 max_steps，此时步数不具可比性。

- **路径距离 (Path Distance)**：实际行走路径的长度。与直线距离之比（路径效率比 = 路径/直线）
  越接近 1 越好。对于有障碍的地图，最优路径会略大于直线距离。

- **终点到目标距离 (Final Goal Dist)**：成功时应 < 0.5m（到达阈值）；失败时数值越大，
  说明离目标越远，导航偏离越严重。

- **实际耗时 (Wall Time)**：包含 MuJoCo 纯仿真时间 + NoMaD 推理时间 + IO 开销。
  较大的编码器（如 ResNet-50, ConvNeXt-Tiny）可能推理更慢。

### 3.2 对比分析

- **最佳编码器**：EfficientNet-B0 (baseline)，总成功率 98%
- **最差编码器**：ConvNeXt-Tiny，总成功率 33%

- **easy 地图最佳**：EfficientNet-B0 (baseline) (94% 成功)
- **hard 地图最佳**：EfficientNet-B0 (baseline) (100% 成功)
- **medium 地图最佳**：EfficientNet-B0 (baseline) (100% 成功)

### 3.3 关键发现

（以下分析基于自动生成的定量结果，更深入的定性分析请结合轨迹可视化）

1. **Baseline vs 预训练特征**：EfficientNet-B0 作为 baseline 从头训练，   与冻结 backbone 的预训练模型相比，各自有何优劣；

2. **CNN vs ViT**：DINOv2 (ViT 架构) vs ConvNeXt/ResNet (CNN 架构)    在导航鲁棒性和推理效率上的差异；

3. **难度梯度**：随着障碍物增加（easy→hard），不同编码器的性能退化程度不同。
