# 论文证据索引表

> 生成时间：2026-04-09
> 本表汇总所有实验的脚本入口、结果路径和核心指标，供论文各章节直接引用。

## 第三章：NoMaD 基线与系统架构

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| 环境检查 | `scripts/env_check.py` | `results/tools/` | 数据集/权重/目录完整性 | 3.x 实验环境 |
| 路径审计 | `scripts/path_audit.py` | `results/tools/` | 硬编码路径统计 | 3.x 工程规范 |
| 数据集检查 | `scripts/check_dataset.py` | `results/day1/` | 3696 条轨迹，采样正常 | 3.x 数据集说明 |
| 离线基线推理 | `scripts/offline_inference.py` | `results/day1/20260409_221944_offline_inference/` | 19,049,675 参数，12.2 Hz，DDPM 0.0817s | 3.x 基线性能 |

## 第四章：DDIM 加速与 CFG 引导优化

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| DDIM 快速验证 | `scripts/ddim_experiment.py` | `results/day2/` | DDIM-5: 14.17ms (2× 加速), MSE 0.000063 | 4.x 快速验证 |
| DDIM 统计实验 | `scripts/ddim_stat_experiment.py` | `results/day2/20260409_222333_ddim_stat_experiment/` | 6 配置 × 24 cases × 8 runs，含 CI95 | 4.x 正式结果表 |
| CFG 快速验证 | `scripts/cfg_experiment.py` | `results/day3/` | w=1.0 最佳导航引导 | 4.x 快速验证 |
| CFG 统计实验 | `scripts/cfg_stat_experiment.py` | `results/day3/20260409_222424_cfg_stat_experiment/` | 6 引导强度 × 24 cases × 8 runs，含 CI95 | 4.x 正式结果表 |
| 编码器对比 | `scripts/encoder_comparison_experiment.py` | `results/day4/20260409_222739_encoder_comparison_experiment/` | EfficientNet-b0 vs DINOv2-small 推理对比 | 4.x 视觉编码器 |

### DDIM 推荐配置结论

| 配置 | 延迟 (ms) | 加速比 | MSE vs 基线 | 结论 |
|---|---:|---:|---:|---|
| DDPM-10 (基线) | 28.55 | 1.00× | 0.0000 | 基线 |
| DDIM-5 (推荐) | 14.18 | 2.01× | 0.0048 | **部署推荐** |
| DDIM-1 (极端) | 3.18 | 8.97× | 3.5168 | 不推荐，精度下降过大 |

### DDIM × CFG 联合优化实验 (NEW)

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| 联合网格搜索 | `scripts/joint_ddim_cfg_experiment.py` | `results/day4/20260409_225810_joint_ddim_cfg_experiment/` | 20 配置 × 24 cases × 8 runs, 综合评分排名 | 4.x 联合优化 |

#### 联合优化 Top-5 配置

| 排名 | 配置 | 综合评分 | 延迟(ms) | 加速比 | 前进距离 | 平滑度 |
|---:|---|---:|---:|---:|---:|---:|
| 1 | **DDIM-2 w=0.0** | **5.6477** | **6.18** | **4.84×** | 8.089 | 0.1237 |
| 2 | DDIM-3 w=0.0 | 5.3648 | 9.12 | 3.28× | 8.076 | 0.1075 |
| 3 | DDIM-2 w=0.5 | 4.9877 | 11.39 | 2.63× | 8.106 | 0.1246 |
| 4 | DDIM-2 w=1.0 | 4.9853 | 11.44 | 2.62× | 8.122 | 0.1259 |
| 5 | DDIM-5 w=0.0 | 4.9851 | 14.94 | 2.00× | 8.032 | 0.1043 |

**最优部署配置: DDIM-2 + 无 CFG (6.18ms, 4.84× 加速)**

### CFG 推荐配置结论

| 引导强度 w | 延迟 (ms) | Diversity | Shift vs w=0 | 结论 |
|---:|---:|---:|---:|---|
| 0.0 (基线) | 28.65 | 0.2879 | 0.0000 | 标准导航 |
| 1.0 (推荐) | 53.20 | 0.2774 | 0.0476 | **适度引导，轨迹更集中** |
| 4.0 (过度) | 53.12 | 0.3041 | 0.5133 | 过度引导，偏移过大 |

### 编码器对比结论

| 编码器 | 延迟 (ms) | Diversity | Forward Progress | 结论 |
|---|---:|---:|---:|---|
| EfficientNet-b0 (原始) | 29.98 | 0.2668 | 6.87 | 基线 |
| DINOv2-small | 28.94 | 0.5779 | 8.51 | 多样性更高，前进距离更优 |

## 第五章：Lite3 MuJoCo 联合导航

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| Topomap 生成 (easy) | `scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map easy` | `topomaps/easy/` | 20 节点 | 5.x topomap 构建 |
| Topomap 生成 (medium) | `scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map medium` | `topomaps/medium/` | 20 节点 | 5.x topomap 构建 |
| Topomap 生成 (hard) | `scripts/nomad_mujoco_lite3_nav.py --mode generate-topomap --map hard` | `topomaps/hard/` | 20 节点 | 5.x topomap 构建 |
| DDPM 导航 easy | `--mode navigate --map easy` | `results/nomad_mujoco/20260409_222936_navigate_mujoco/` | SUCCESS, 43 步, 0 恢复, 4.565m | 5.x 结果表 |
| DDPM 导航 medium | `--mode navigate --map medium` | `results/nomad_mujoco/20260409_222956_navigate_mujoco/` | SUCCESS, 60 步, 0 恢复, 6.388m | 5.x 结果表 |
| DDPM 导航 hard | `--mode navigate --map hard` | `results/nomad_mujoco/20260409_223018_navigate_mujoco/` | SUCCESS, 70 步, 0 恢复, 7.467m | 5.x 结果表 |
| DDIM-5 导航 easy | `--mode navigate --map easy --scheduler ddim --ddim-steps 5` | `results/nomad_mujoco/20260409_223041_navigate_mujoco/` | SUCCESS, 43 步, 0 恢复, 4.517m | 5.x 结果表 |
| DDIM-5 导航 medium | `--mode navigate --map medium --scheduler ddim --ddim-steps 5` | `results/nomad_mujoco/20260409_223054_navigate_mujoco/` | SUCCESS, 61 步, 0 恢复, 6.401m | 5.x 结果表 |
| DDIM-5 导航 hard | `--mode navigate --map hard --scheduler ddim --ddim-steps 5` | `results/nomad_mujoco/20260409_223109_navigate_mujoco/` | SUCCESS, 71 步, 0 恢复, 7.560m | 5.x 结果表 |
| 探索模式 medium | `--mode explore --map medium` | `results/nomad_mujoco/20260409_223125_explore_mujoco/` | 200 步, 2 次恢复, 18.730m | 5.x 探索模式 |
| 纯运动测试 | `--mode walk-test` | `results/nomad_mujoco/20260409_223153_walk_test/` | 前进 7.09m, locomotion 正常 | 5.x 底层验证 |
| 状态机导航 hard | `nomad_mujoco_lite3_state_machine.py --map hard --scheduler ddim --ddim-steps 5` | `results/nomad_mujoco/20260409_223341_lite3_state_machine_navigate/` | SUCCESS, 72 步, 7.734m | 5.x 状态机 |
| **DDIM-2 导航 easy** | `--scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230132_navigate_mujoco/` | SUCCESS, 43 步, 0 恢复 | 5.x 最优配置验证 |
| **DDIM-2 导航 medium** | `--scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230158_navigate_mujoco/` | SUCCESS, 85 步, 0 恢复 | 5.x 最优配置验证 |
| **DDIM-2 导航 hard** | `--scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230215_navigate_mujoco/` | SUCCESS, 70 步, 0 恢复 | 5.x 最优配置验证 |
| **DDIM-2+CFG 导航 easy** | `--scheduler ddim --ddim-steps 2 --cfg-weight 1.0` | `results/nomad_mujoco/20260409_230237_navigate_mujoco/` | SUCCESS, 42 步, 0 恢复 | 5.x CFG 对比 |
| **DDIM-2+CFG 导航 medium** | `--scheduler ddim --ddim-steps 2 --cfg-weight 1.0` | `results/nomad_mujoco/20260409_230246_navigate_mujoco/` | SUCCESS, 61 步, 0 恢复 | 5.x CFG 对比 |
| **DDIM-2+CFG 导航 hard** | `--scheduler ddim --ddim-steps 2 --cfg-weight 1.0` | `results/nomad_mujoco/20260409_230257_navigate_mujoco/` | SUCCESS, 70 步, 0 恢复 | 5.x CFG 对比 |
| **DDIM-2 探索 easy** | `--mode explore --scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230327_explore_mujoco/` | 100 步, 2 恢复, ~6.6m | 5.x 探索模式 |
| **状态机 DDIM-2** | `state_machine --scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230434_lite3_state_machine_navigate/` | SUCCESS, 44 步, 4.673m | 5.x 状态机 |
| **多目标 mission** | `state_machine --mode mission --scheduler ddim --ddim-steps 2` | `results/nomad_mujoco/20260409_230505_lite3_state_machine_mission/` | SUCCESS, 152 步, 13.466m | 5.x 多目标导航 |

### Lite3 导航结果汇总

| 地图 | 调度器 | 结果 | 步数 | 恢复次数 | 路径距离 |
|---|---|---|---:|---:|---:|
| easy | DDPM | SUCCESS | 43 | 0 | 4.565m |
| medium | DDPM | SUCCESS | 60 | 0 | 6.388m |
| hard | DDPM | SUCCESS | 70 | 0 | 7.467m |
| easy | DDIM-5 | SUCCESS | 43 | 0 | 4.517m |
| medium | DDIM-5 | SUCCESS | 61 | 0 | 6.401m |
| hard | DDIM-5 | SUCCESS | 71 | 0 | 7.560m |
| **easy** | **DDIM-2** | **SUCCESS** | **43** | **0** | — |
| **medium** | **DDIM-2** | **SUCCESS** | **85** | **0** | — |
| **hard** | **DDIM-2** | **SUCCESS** | **70** | **0** | — |
| **easy** | **DDIM-2+CFG** | **SUCCESS** | **42** | **0** | — |
| **medium** | **DDIM-2+CFG** | **SUCCESS** | **61** | **0** | — |
| **hard** | **DDIM-2+CFG** | **SUCCESS** | **70** | **0** | — |

### 多目标导航结果

| 模式 | 目标序列 | 配置 | 步数 | 距离 | 结果 |
|---|---|---|---:|---:|---|
| mission | easy → medium | DDIM-2 | 152 | 13.466m | ✅ SUCCESS |

## 第五章：Tron1 实施状态

| 项目 | 脚本路径 | 结果路径 | 状态 |
|---|---|---|---|
| 资产验证 | `scripts/nomad_mujoco_tron1_nav.py --mode validate-assets` | 终端输出 | NoMaD 权重 OK，MuJoCo 模型缺失 |
| 实施清单 | `scripts/nomad_mujoco_tron1_nav.py --mode export-manifest --save` | `results/day6/20260409_223422_tron1_tron1_plan/` | 6 阶段实施路线已导出 |

## 部署清单

| 平台 | 脚本路径 | 结果路径 |
|---|---|---|
| Lite3 | `scripts/nomad_real_deployment_checklist.py --platform lite3 --save` | `results/deployment/20260409_223446_lite3_checklist/` |
| Tron1 | `scripts/nomad_real_deployment_checklist.py --platform tron1 --save` | `results/deployment/20260409_223452_tron1_checklist/` |

## 最终推荐配置

1. **模型**：NoMaD（19M 参数）
2. **调度器**：DDIM-5（2× 加速，精度损失可忽略）
3. **引导强度**：CFG w=1.0（适度引导，轨迹更集中）
4. **视觉编码器**：DINOv2-small 表现优于 EfficientNet-b0（需训练验证）
5. **Lite3 仿真**：3 地图 × 2 调度器 = 6/6 成功率
