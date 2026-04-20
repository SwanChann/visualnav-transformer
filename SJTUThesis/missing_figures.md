# 缺失图片清单

> 基于对毕业论文各章节的全面审查，列出所有使用 \fbox 占位符的图位。
> 生成时间：2026-04-20

---

## 第一章：绪论

| 图号 | Label | 描述 | 优先级 |
| --- | --- | --- | --- |
| 图1.1 | fig:intro-background-placeholder | 研究背景与目标场景示意图 | **高** |
| 图1.2 | fig:intro-roadmap-placeholder | 本文总体技术路线与章节关系图 | **高** |

---

## 第二章：预备知识

| 图号 | Label | 描述 | 优先级 |
| --- | --- | --- | --- |
| 图2.1 | fig:prelim-nav-pipeline-placeholder | 导航任务输入输出与 topomap 关系示意图 | **高** |
| 图2.2 | fig:prelim-diffusion-placeholder | 扩散模型、扩散策略与 NoMaD 关系示意图 | **高** |
| 图2.3 | fig:prelim-lite3-placeholder | Lite3 平台与控制链路示意图 | **高** |

---

## 第三章：算法设计与实验

| 图号 | Label | 描述 | 优先级 |
| --- | --- | --- | --- |
| 图3.1 | fig:algo-overview-placeholder | NoMaD 算法研究对象与数据流示意图 | **高** |
| 图3.2 | fig:algo-ddim-placeholder | DDIM 加速实验结果图（柱状图/折线图） | **高** |
| 图3.3 | fig:algo-cfg-placeholder | CFG 引导实验结果图 | 中 |
| 图3.4 | fig:algo-tts-placeholder | TTS 联合实验结果图 | 低（实验未完成） |
| 图3.5 | fig:algo-encoder-placeholder | 视觉编码器升级实验结果图 | 中 |

---

## 第四章：算法框架设计与仿真环境验证

| 图号 | Label | 描述 | 优先级 |
| --- | --- | --- | --- |
| 图4.1 | fig:framework-modules-placeholder | 统一推理模块、状态机与平台后端关系图 | **高** |
| 图4.2 | fig:framework-fsm-placeholder | 状态机状态流转图 | **高** |
| 图4.3 | fig:framework-mujoco-env-placeholder | MuJoCo 仿真环境、Lite3 模型与 topomap 示例图 | **高** |
| 图4.4 | fig:framework-mujoco-results-placeholder | MuJoCo 单目标导航与多目标任务运行结果图 | **高** |

---

## 第五章：真机实验

| 图号 | Label | 描述 | 优先级 |
| --- | --- | --- | --- |
| 图5.1 | fig:real-overview-placeholder | Orin 与 Lite3 真机部署总体架构图 | **高** |
| 图5.2 | fig:real-bridge-placeholder | Orin 推理、桥接层与 Lite3 运动主机数据流图 | **高** |
| 图5.3 | fig:real-topomap-placeholder | 真实环境 topomap 采集与目标图像组织示意图 | 中（需真机采集） |
| 图5.4 | fig:real-visualization-placeholder | 真机运行中的实时图像、目标图像与 capture 示例图 | 中（需真机运行） |
| 图5.5 | fig:real-pending-placeholder | 真机实验照片、运行截图与失败案例图 | 中（需真机实验） |

---

## 总结

| 章节 | 占位图数量 | 高优先 | 中/低优先 |
| --- | --- | --- | --- |
| 第一章 绪论 | 2 | 2 | 0 |
| 第二章 预备知识 | 3 | 3 | 0 |
| 第三章 算法设计与实验 | 5 | 2 | 3 |
| 第四章 框架设计与仿真 | 4 | 4 | 0 |
| 第五章 真机实验 | 5 | 2 | 3 |
| **合计** | **19** | **13** | **6** |

## 图片制作建议

1. **架构图/流程图**（共 7 张）：建议使用 TikZ 或 draw.io 制作，包括技术路线图、系统架构图、状态机流转图等
2. **实验结果图**（共 5 张）：建议使用 Python matplotlib/pgfplots 生成柱状图或折线图
3. **平台/环境截图**（共 4 张）：需要 MuJoCo 仿真截图和 Lite3 实物照片
4. **真机运行图**（共 3 张）：需要真机实验时拍摄
