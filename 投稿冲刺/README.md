# 投稿冲刺

更新日期：2026-07-07
项目位置：`f:\codespace\visualnav-transformer`

本文件夹已经重构为 RA-L 投稿冲刺的最小作战包。旧的阶段性笔记、重复路线图和过强 claim 文档已经合并删除，现在只保留一个入口和三个核心文档。

## 当前主线

> Time-quality aware deployment and evaluation of NoMaD-style diffusion visual goal navigation on a resource-constrained Lite3 quadruped robot.

一句话说人话：

> 不再把本科毕设包装成新 foundation model 或跨形态 zero-shot 论文，而是把已有 NoMaD-style diffusion 视觉导航系统在 Lite3 上的部署、推理时间-质量权衡、MuJoCo 闭环和冻结真机证据整理成一篇 RA-L 实证论文。

## 文件索引

- [01_贡献点与前沿对照.md](01_贡献点与前沿对照.md)：逐点解释本文每个 contribution，以及 GNM、ViNT、NoMaD、NavDP、SIDP、X-Nav、Fisher Guidance 等相关工作已经做到什么程度。
- [02_RA-L论文结构设计.md](02_RA-L论文结构设计.md)：设计论文题目、摘要、贡献、章节、图表、Related Work、Method、Experiments 和 limitation。
- [03_实验与写作任务清单.md](03_实验与写作任务清单.md)：布置接下来具体要做的离线实验、MuJoCo 汇总、冻结真机整理、图表制作和英文写作任务。

## 硬约束

1. 不再新增 Lite3 真机采集；已有视频、日志、失败案例全部冻结。
2. 不做大规模重新训练；不把论文改成 NavDP/SIDP/CE-Nav/X-Nav 那种新训练范式论文。
3. 不 claim cross-embodiment zero-shot；Lite3 adapter 只作为平台部署组件。
4. 不把 TTS/CFG/DDIM 说成新算法；它们是 measured inference configuration space。
5. 真机结果只作为 representative anchor validation；统计结论主要来自离线和 MuJoCo。

## 最小可投稿目标

首投版本至少要有：

- Fig. 1：系统总览。
- Fig. 2：offline latency-quality Pareto。
- Fig. 3：MuJoCo closed-loop results。
- Fig. 4：frozen real-robot frames and timing。
- Table II：offline inference configurations。
- Table III：MuJoCo closed-loop summary。
- Table IV：frozen real-robot evidence。
- 一个明确、诚实、不躲闪的 limitations 段落。

## 下一步

按照 [03_实验与写作任务清单.md](03_实验与写作任务清单.md) 执行：先做 evidence matrix，再做 offline Pareto，然后整理 MuJoCo 和真机，最后写英文正文。
