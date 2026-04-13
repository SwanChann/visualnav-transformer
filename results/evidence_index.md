# 论文证据索引表

> 生成时间：2026-04-13
> 说明：本表按 `new` 分支当前代码结构整理，统一使用分类后的 canonical script path，并优先指向正式统计结果与 2026-04-13 补充生成的仿真/部署证据。

## 第三章：环境、数据与基线

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| 环境检查 | `scripts/tooling/env_check.py` | `results/tools/env_check_20260413_150644.txt` | Ubuntu 20.04 / Python 3.8.5 / Torch 2.4.1 / CUDA 可用 | 3.x 实验环境 |
| 路径审计 | `scripts/tooling/path_audit.py` | `results/tools/path_audit_20260413_150647.txt` | 78 个路径发现项，已剔除 `results/` 与构建缓存噪声 | 3.x 工程规范 |
| 数据集检查 | `scripts/analysis/check_dataset.py` | `results/day1/20260410_143018_dataset_check.txt` | 3696 条轨迹，抽查 5 条，0 异常 | 3.x 数据集说明 |
| 离线基线推理 | `scripts/analysis/offline_inference.py` | `results/day1/20260410_143032_offline_inference/summary.txt` | 19,049,675 参数，DDPM 0.1250 s，8.0 Hz | 3.x 基线性能 |
| 统一汇总 | `scripts/analysis/thesis_result_summary.py` | `results/summary/thesis_result_summary_20260413_150642.md` | 汇总基线、统计实验、Lite3、导航主机与 Tron1 资产状态 | 3.x 证据总览 |

## 第四章：DDIM、CFG 与视觉编码器

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| DDIM 快速验证 | `scripts/experiments/ddim_experiment.py` | `results/day2/20260410_143043_ddim_results.txt` | DDIM-2/3/5 与 DDPM-10 的时延与轨迹误差对比 | 4.x 快速验证 |
| DDIM 统计实验 | `scripts/experiments/ddim_stat_experiment.py` | `results/day2/20260410_143136_ddim_stat_experiment/ddim_overall_summary.csv` | 6 配置 × 24 cases × 8 runs，含 CI95 | 4.x 正式结果 |
| CFG 快速验证 | `scripts/experiments/cfg_experiment.py` | `results/day3/results.txt` | 不同 `w` 的快速效果对比 | 4.x 快速验证 |
| CFG 统计实验 | `scripts/experiments/cfg_stat_experiment.py` | `results/day3/20260410_143225_cfg_stat_experiment/cfg_overall_summary.csv` | 6 引导强度 × 24 cases × 8 runs | 4.x 正式结果 |
| DDIM × CFG 联合优化 | `scripts/joint_ddim_cfg_experiment.py` | `results/day4/20260410_143709_joint_ddim_cfg_experiment/overall_summary.csv` | 综合最优 `ddim2_w0.0` | 4.x 部署配置 |
| 编码器对比 | `scripts/experiments/encoder_comparison_experiment.py` | `results/day4/20260410_143729_encoder_comparison_experiment/encoder_overall_summary.csv` | `dinov2_small` 优于 `efficientnet_b0` | 4.x 编码器升级 |

### 当前推荐配置

| 模块 | 推荐 | 依据 |
|---|---|---|
| 采样调度 | `DDIM-2` | `results/day2/20260410_143136_ddim_stat_experiment/ddim_overall_summary.csv` |
| CFG | 默认 `w=0.0`，障碍增强 `w=1.0` | `results/day3/20260410_143225_cfg_stat_experiment/cfg_overall_summary.csv` 与 Lite3 medium 地图结果 |
| 视觉编码器 | `dinov2_small` | `results/day4/20260410_143729_encoder_comparison_experiment/encoder_overall_summary.csv` |

## 第五章：Lite3 MuJoCo 联合导航

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| Topomap 生成 | `scripts/simulation/nomad_mujoco_lite3_nav.py --mode generate-topomap` | `topomaps/easy/`, `topomaps/medium/`, `topomaps/hard/` | 每张地图 20 节点 | 5.x topomap 构建 |
| DDIM-2 导航 easy | `scripts/simulation/nomad_mujoco_lite3_nav.py` | `results/nomad_mujoco/20260410_143820_navigate_mujoco/summary.txt` | 45 步，4.705 m，成功到达 | 5.x 基础导航 |
| DDIM-2 导航 medium | `scripts/simulation/nomad_mujoco_lite3_nav.py` | `results/nomad_mujoco/20260410_143829_navigate_mujoco/summary.txt` | 87 步，7.360 m，成功到达 | 5.x 基础导航 |
| DDIM-2 导航 hard | `scripts/simulation/nomad_mujoco_lite3_nav.py` | `results/nomad_mujoco/20260410_143840_navigate_mujoco/summary.txt` | 70 步，7.474 m，成功到达 | 5.x 基础导航 |
| DDIM-2 + CFG 导航 easy | `scripts/simulation/nomad_mujoco_lite3_nav.py --cfg-weight 1.0` | `results/nomad_mujoco/20260410_144025_navigate_mujoco/summary.txt` | 42 步，4.354 m | 5.x CFG 对比 |
| DDIM-2 + CFG 导航 medium | `scripts/simulation/nomad_mujoco_lite3_nav.py --cfg-weight 1.0` | `results/nomad_mujoco/20260410_144035_navigate_mujoco/summary.txt` | 61 步，6.433 m | 5.x CFG 对比 |
| DDIM-2 + CFG 导航 hard | `scripts/simulation/nomad_mujoco_lite3_nav.py --cfg-weight 1.0` | `results/nomad_mujoco/20260410_144047_navigate_mujoco/summary.txt` | 70 步，7.403 m | 5.x CFG 对比 |
| 单脚本探索 | `scripts/simulation/nomad_mujoco_lite3_nav.py --mode explore` | `results/nomad_mujoco/20260410_144106_explore_mujoco/summary.txt` | 100 步，7.763 m | 5.x 探索能力 |
| 单脚本纯运动 | `scripts/simulation/nomad_mujoco_lite3_nav.py --mode walk-test` | `results/nomad_mujoco/20260410_144120_walk_test/summary.txt` | 48 步，7.094 m | 5.x 底层链路 |
| 状态机导航 | `scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode navigate` | `results/nomad_mujoco/20260410_144134_lite3_state_machine_navigate/summary.txt` | 43 步，4.622 m，成功到达 | 5.x 分层系统 |
| 状态机 mission | `scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode mission` | `results/nomad_mujoco/20260410_144148_lite3_state_machine_mission/summary.txt` | 129 步，12.601 m，成功完成 | 5.x 多目标导航 |
| 状态机探索 | `scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode explore` | `results/nomad_mujoco/20260413_150349_lite3_state_machine_explore/summary.txt` | 99 步，8.711 m | 5.x 分层探索 |
| 状态机纯运动 | `scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode walk-test` | `results/nomad_mujoco/20260413_150349_lite3_state_machine_walk-test/summary.txt` | 48 步，6.398 m | 5.x 分层底层验证 |
| 导航主机 dry-run | `scripts/deployment/nomad_navigation_host.py --backend mujoco --plan-file ... --dry-run --save-plan` | `results/deployment/20260413_150346_lite3_mujoco_navigation_host/host_summary.md` | `navigate + mission` 两任务计划通过校验 | 5.x 仿真到真机调度层 |

## 第六章：Tron1 与真机部署准备

| 实验名称 | 脚本路径 | 结果路径 | 核心指标 | 章节用途 |
|---|---|---|---|---|
| Tron1 资产导出 | `scripts/simulation/nomad_mujoco_tron1_nav.py --mode export-manifest --save` | `results/day6/20260413_150410_tron1_tron1_plan/tron1_manifest.json` | 基础权重存在，MuJoCo 模型缺失 | 6.x Tron1 资产规范 |
| Lite3 部署清单 | `scripts/deployment/nomad_real_deployment_checklist.py --platform lite3 --save` | `results/deployment/20260413_150410_lite3_checklist/deployment_checklist.md` | Lite3 必需资产齐全 | 6.x Lite3 部署准备 |
| Tron1 部署清单 | `scripts/deployment/nomad_real_deployment_checklist.py --platform tron1 --save` | `results/deployment/20260413_150410_tron1_checklist/deployment_checklist.md` | Tron1 缺失 `assets/tron1/mujoco/tron1.xml` | 6.x Tron1 部署准备 |

## 当前未闭环项

| 项目 | 当前状态 | 原因 |
|---|---|---|
| 训练相关流程 | 本轮未执行 | 当前要求明确不做训练验收 |
| ConvNeXt / ResNet 正式结果 | 缺失 | 无现成训练权重，不在 Ubuntu 20.04 部署环境补训 |
| Tron1 MuJoCo 闭环导航 | 缺失 | `assets/tron1/mujoco/tron1.xml` 尚未补齐 |
| Lite3 / Tron1 真机低速测试日志 | 缺失 | 当前仓库保留接口与清单，未保存真机运行记录 |
