# 缺失图片清单

> 基于当前论文正文重新扫描，列出仍使用 `\fbox` 占位或 caption 标注“待补”的图位。
> 更新时间：2026-04-28

## 分类口径

- **自绘图**：架构图、流程图、机制图、状态机图。建议用 draw.io、TikZ、PPT/Keynote 或 AI 生成后再人工修正。
- **实验图**：由实验结果、仿真截图、真机截图、现场照片或代码绘图生成。用户后续自行准备，不在自绘提示词文档中展开。
- 已经存在并被正文引用的实验图不再列为缺失，例如 `tts_combo_tradeoff_20260421.png` 和 `encoder_comparison_20260421.png`。

---

## 第一章：绪论

| 图号 | Label | 描述 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 图1.1 | `fig:intro-background-placeholder` | 研究背景与目标场景示意图 | 自绘图 | 高 | 缺失 |
| 图1.2 | `fig:intro-roadmap-placeholder` | 本文总体技术路线与章节关系图 | 自绘图 | 高 | 缺失 |

---

## 第二章：预备知识

| 图号 | Label | 描述 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 图2.1 | `fig:prelim-nav-pipeline-placeholder` | 导航任务输入输出与 topomap 关系示意图 | 自绘图 | 高 | 缺失 |
| 图2.2 | `fig:prelim-diffusion-placeholder` | 扩散模型、扩散策略、ViT 编码器与 NoMaD 技术演化关系 | 自绘图 | 高 | 缺失 |
| 图2.3 | `fig:prelim-lite3-placeholder` | Lite3 平台与控制链路示意图 | 自绘图 | 高 | 缺失 |

---

## 第三章：算法设计与实验

| 图号 | Label | 描述 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 图3.1 | `fig:algo-overview-placeholder` | NoMaD 算法研究对象与数据流示意图 | 自绘图 | 高 | 缺失 |
| 图3.2 | `fig:algo-ddim-placeholder` | DDIM 加速实验结果图 | 实验图/代码生成 | 高 | 缺失 |
| 图3.3 | `fig:algo-cfg-placeholder` | CFG 引导实验结果图 | 实验图/代码生成 | 中 | 缺失 |
| 图3.4 | `fig:algo-tts-tradeoff` | TTS 联合实验的时延与轨迹质量折中 | 实验图/已有 | - | 已存在：`figures/tts_combo_tradeoff_20260421.png` |
| 图3.5 | `fig:algo-encoder-comparison` | 四种视觉编码器离线统计对比 | 实验图/已有 | - | 已存在：`figures/encoder_comparison_20260421.png` |

---

## 第四章：算法框架设计与仿真环境验证

| 图号 | Label | 描述 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 图4.1 | `fig:framework-modules-placeholder` | 统一推理模块、状态机与平台后端关系图 | 自绘图 | 高 | 缺失 |
| 图4.2 | `fig:framework-fsm-placeholder` | 状态机状态流转图 | 自绘图 | 高 | 缺失 |
| 图4.3 | `fig:framework-mujoco-env-placeholder` | MuJoCo 仿真环境、Lite3 模型与 topomap 示例图 | 实验图/截图 | 高 | 缺失 |
| 图4.4 | `fig:framework-mujoco-results-placeholder` | MuJoCo 单目标导航与多目标任务运行结果图 | 实验图/代码生成或截图 | 高 | 缺失 |

---

## 第五章：真机实验

| 图号 | Label | 描述 | 类型 | 优先级 | 状态 |
| --- | --- | --- | --- | --- | --- |
| 图5.1 | `fig:real-overview-placeholder` | Orin 与 Lite3 真机部署总体架构图 | 自绘图 | 高 | 缺失 |
| 图5.2 | `fig:real-bridge-placeholder` | Orin 推理、桥接层与 Lite3 运动主机数据流图 | 自绘图 | 高 | 缺失 |
| 图5.3 | `fig:real-topomap-placeholder` | 真实环境 topomap 采集与目标图像组织示意图 | 自绘图 | 高 | 缺失 |
| 图5.4 | `fig:real-visualization-placeholder` | 真机运行中的实时前视图像、实时子目标节点图像与 capture 示例图 | 实验图/真机截图 | 中 | 缺失 |
| 图5.5 | `fig:real-pending-placeholder` | 真机实验照片、运行截图与失败案例图 | 实验图/现场照片 | 中 | 缺失 |

---

## 汇总

| 章节 | 当前仍缺图数量 | 自绘图 | 实验图 | 已补图 |
| --- | ---: | ---: | ---: | ---: |
| 第一章 绪论 | 2 | 2 | 0 | 0 |
| 第二章 预备知识 | 3 | 3 | 0 | 0 |
| 第三章 算法设计与实验 | 3 | 1 | 2 | 2 |
| 第四章 框架设计与仿真 | 4 | 2 | 2 | 0 |
| 第五章 真机实验 | 5 | 3 | 2 | 0 |
| **合计** | **17** | **11** | **6** | **2** |

## 制作建议

1. **优先补自绘图 11 张**：这些图承担论文结构解释功能，且不依赖进一步实验。建议先生成草图，再按论文术语人工校对。
2. **实验图 6 张用户自行准备**：包括 DDIM/CFG 结果图、MuJoCo 截图或结果图、真机运行截图/照片。
3. **第三章已有两张实验图**：TTS 与编码器对比图已存在，不需要重复列入缺失清单。
4. **第五章图5.1建议优先重画为“双链路总览”**：当前正文开头已经突出横向实时推理链路与纵向分层控制链路，图5.1应与该内容匹配。
