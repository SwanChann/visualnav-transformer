# 后续研究内容

更新日期：2026-07-10

## 当前结论

这个文件夹已经从“罗列所有可能方向”收敛为一条可执行主线：

> 面向跨数据集泛化与端侧部署，训练一个小型生成式视觉导航策略，并建立能诊断视觉域、动作尺度、历史条件和计算预算失效原因的 benchmark。

工作名：

> **TinyNavBrain：Cross-Dataset Diagnostic Training of Compact Generative Visual Navigation Policies**

这里的“导航大脑”不是 VLM、AGI 或大型视频世界模型，而是视觉导航系统中的生成式策略核心：

~~~
observation history + goal image + action history + physical scale
    -> compact generative policy
    -> multimodal waypoint/action chunk + progress/confidence
    -> existing Lite3 adapter and safety bridge
~~~

## 为什么改变原路线

原文件夹把 learned ranker、mode selector、few-step distillation、world model、VLM 和 RL 同时列为主线，范围过宽，也低估了相关工作：

- NavDP 已使用 diffusion + critic。
- SIDP 已针对 generate-then-filter 低效问题做 self-imitation。
- NaviDiffusor、SRIF 已覆盖 cost guidance 或 candidate ranking。
- Navigation World Models、NavWAM 已覆盖未来视觉和 world-action 建模。
- Can Vision Foundation Models Navigate? 已做多模型、双平台、五环境及扰动评测。
- VISTA 已指出 normalized action scaling 的部署问题并引入 action-history conditioning。

因此：

- **不再把 ranker-first 当唯一主线**；ranker 只保留为 baseline 或后续模块。
- **不单独做 benchmark 论文**；benchmark 必须服务于一个明确训练问题。
- **不把动作归一化、action history、few-step sampling 单独声称为新贡献**。
- **不把 VLM/world model/RL 同时塞入一篇 RA-L**。

## 仓库事实

当前真正可直接训练的数据只有：

~~~
nomad_dataset/go_stanford
~~~

本地统计为 3696 条 processed trajectories、201822 张图像。recon_datavis 和 tartan_drive 目前是代码/可视化仓库，不是符合 ViNT_Dataset 格式的训练数据。

多数据集研究的第一项工作不是训练模型，而是完成数据获取、许可、处理、统一 schema 和无泄漏 split 审计。

## 文件索引

- [01_后续研究总览.md](01_后续研究总览.md)：现有方向的前沿性评分、保留/降级/删除判断和最终 research gap。
- [02_导航策略大脑方向.md](02_导航策略大脑方向.md)：TinyNavBrain 的问题定义、模型结构、训练目标和对比边界。
- [03_单卡4090训练路线.md](03_单卡4090训练路线.md)：多数据集获取、benchmark 协议、4090 训练阶段和止损条件。
- [04_RA-L后续研究投稿规划.md](04_RA-L后续研究投稿规划.md)：面向下一篇 RA-L 的贡献、实验、图表、里程碑与 go/no-go 标准。
- [06_毕业论文后工作进展详解.md](06_毕业论文后工作进展详解.md)：用非缩写式名称解释毕业论文之后已完成的实验审计、数据接线、训练基础设施和仍未完成的科学结果。
- [07_具身导航Agent方向与单卡4090真机可行性调研.md](07_具身导航Agent方向与单卡4090真机可行性调研.md)：判断 NoMaD 在 2026 年的角色、单卡 RTX 4090 与 Lite3 真机的分层可行性，并提出可证伪的异步 Agent 研究方向。
- [references/后续研究文献矩阵.md](references/后续研究文献矩阵.md)：核验后的核心文献与本文差异。
- [data/dataset_acquisition_audit.md](data/dataset_acquisition_audit.md)：候选数据集的官方来源、许可、格式、获取优先级和 Gate A。
- [data/dataset_registry.json](data/dataset_registry.json)：机器可读数据集注册表；未知许可默认阻塞。
- [benchmark/benchmark_protocol_v0.1.md](benchmark/benchmark_protocol_v0.1.md)：冻结的五条评测 track、公平预算、baseline 和 go/no-go。
- [benchmark/metric_spec_v0.1.md](benchmark/metric_spec_v0.1.md)：meter-scale offline、robustness、closed-loop 与 latency 口径。
- [benchmark/result_schema_v0.1.json](benchmark/result_schema_v0.1.json)：每个外部实验必须满足的机器可读结果契约。
- [model/TinyNavBrain_contract_v0.1.md](model/TinyNavBrain_contract_v0.1.md)：canonical batch、共享 encoder、deterministic/flow heads 与停止条件。
- [model/tinynavbrain_v0.1.json](model/tinynavbrain_v0.1.json)：设计配置；状态明确为 `design_only_not_trained`。
- [model/baseline_matrix_v0.1.md](model/baseline_matrix_v0.1.md)：固定数据、sampler、encoder、NFE 和 candidate budget 的公平对照矩阵。

原 roadmaps/一年研究路线图.md 已删除。原因是它重复其他文档、默认一年产出三篇论文，并把 ranker、world model、VLM、RL 串成线性路线，目标不符合 RA-L 的单一问题约束。

## 与当前 RA-L 的关系

必须区分两篇工作：

1. **当前 RA-L 冲刺**：frozen NoMaD-style policy 的 Lite3 部署和 latency-quality 分析。继续在 投稿冲刺/ 完成，不加入新训练方法。
2. **后续 RA-L 研究**：多数据集诊断 benchmark + 小型生成式策略。需要新数据处理、重新训练和新的闭环实验。

不要为了“创新”把后续模型塞回当前稿件，否则会同时削弱两篇论文。

## 唯一推荐主线

~~~
Phase A  多数据集 benchmark 和强 baseline
Phase B  小型生成式策略 TinyNavBrain
Phase C  leave-one-dataset-out + corruption + compute-budget 诊断
Phase D  有区分度的闭环仿真
Phase E  Lite3 端侧 timing 与最小真实验证（若未来允许）
~~~

只有当 TinyNavBrain 在公平参数量和计算预算下，稳定超过单数据集 NoMaD、同规模确定性策略和多数据集 NoMaD baseline，并在闭环 collision/SPL 上改善，才适合投稿 RA-L。
