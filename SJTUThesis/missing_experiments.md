# 缺失实验清单（2026-04-22 更新）

> 基于对整本论文各章节的梳理，列出论文中存在但尚未完成或未填入数据的实验项。
> 2026-04-22 已把算法层的编码器联合消融补齐，并把第三章框架介绍的若干事实错误（双编码器结构、token 池化方式、冻结策略口径）按代码实际行为修正；所有已补实验都在 `results/` 下有时间戳目录，供复核。

---

## 一、算法层（第三章）

### 1.1 已完成
| 实验 | 证据位置 | 状态 |
|---|---|---|
| 基线推理复现 (DDPM-10) | §3.4 tab:baseline-inference | ✅ 已填，并按 40 ms 纯采样时延修正 |
| DDIM 6 配置（ddpm_10/ddim_10/5/3/2/1） | §3.6 tab:ddim-results | ✅ 已填，Diversity 列已补入 |
| CFG 6 配置（$w\in\{-1,0,0.5,1,2,4\}$） | §3.7 tab:cfg-results | ✅ 数据已填 |
| DDIM×CFG 联合 Top-5 | §3.9 tab:joint-ddim-cfg | ✅ 数据已填 |
| 视觉编码器第一阶段（EfficientNet / DINOv2 / ConvNeXt / ResNet） | §3.8 tab:encoder-results | ✅ 数据已填 |
| TTS 单因素、DDIM×TTS、CFG×TTS、DDIM×CFG×TTS | §3.8 tab:tts-results / tab:tts-joint-top | ✅ `results/day4/20260421_140601_tts_stat_experiment` |
| 四编码器离线对比 | §3.8 tab:encoder-results | ✅ `results/day4/20260421_151228_encoder_comparison_experiment` |
| 编码器 × DDIM 联合消融（4 编码器 × DDIM-2/3/5, 12 case × 3 runs） | §3.9 tab:encoder-ddim-joint | ✅ `results/day5/20260422_171021_encoder_joint_ablation_experiment` |
| 编码器 × TTS 联合消融（4 编码器 × TTS-0/8/16, 12 case × 3 runs） | §3.9 tab:encoder-tts-joint | ✅ 同上 |
| 编码器 × CFG 联合消融（EfficientNet-B0 / ResNet-50 × CFG-0/1/2, 12 case × 3 runs） | §3.9 tab:encoder-cfg-joint | ✅ 同上 |
| 算法层事实校正（双编码器、池化方式、冻结口径、时延基准） | §3.2 / §3.4 | ✅ 2026-04-22 按代码核对后修订 |

### 1.2 待补
| 实验 | 章节 | 优先级 | 说明 |
|---|---|---|---|
| 真机最终配置对照（DDPM-10 vs DDIM-2 vs DDIM-2+TTS-8） | §5.6 / 结论 | 高 | 需要 Lite3 真机；验证第三章默认方案能否迁移到真机，记录成功率、时延、轨迹长度和失败模式 |
| DDIM-2 在 Orin 上的单独 wall-clock 测量 | §3.4 末段（现仅为推算） | 低 | 目前"2--5 Hz"为按 RTX 4090 算力 1/5 推算，缺 Orin 实测 DDIM-2 时延 |

---

## 二、仿真层（第四章）

### 2.1 已完成
| 实验 | 证据位置 | 条数 |
|---|---|---|
| Lite3 MuJoCo 单目标导航 | §4.4 tab:mujoco-results | 6 条（全部到达） |
| Lite3 MuJoCo 探索模式 | §4.4 tab:mujoco-results | 1 条（100 步） |
| Lite3 MuJoCo 步态测试 | §4.4 tab:mujoco-results | 2 条 |
| 状态机完整导航 | §4.4 tab:mujoco-results | 1 条 |
| 状态机多目标任务 | §4.4 tab:mujoco-results | 1 条（12.601 m） |
| 状态机探索 | §4.4 tab:mujoco-results | 1 条 |
| MuJoCo 健康性复核（easy/medium/hard 物理到达） | §4.4 / §4.5 | ✅ `results/nomad_mujoco/20260421_145557_*`、`145606_*`、`145616_*` |
| 不同 DDIM 步数 MuJoCo 闭环扫描（2/3/5/10） | §4.5 tab:mujoco-ddim-closed-loop | ✅ `results/benchmark/20260421_seeded_mujoco_ddim*_medium_hard` |
| 不同 CFG 权重 MuJoCo 闭环扫描（0/0.5/1/2） | §4.5 tab:mujoco-cfg-closed-loop | ✅ `results/benchmark/20260421_seeded_mujoco_cfg_medium_ddim3` |
| 跨视觉编码器 MuJoCo 闭环对比 | §4.5 tab:mujoco-encoder-closed-loop | ✅ `results/benchmark/20260421_seeded_mujoco_encoder_benchmark` |

### 2.2 待补
| 实验 | 章节 | 优先级 | 说明 |
|---|---|---|---|
| 失败案例分析（带图像） | §4.5（已预留占位） | 中 | 文本已讨论推理超时/轨迹震荡/到达误触发三类失败，但尚无可视化证据 |
| MuJoCo 运行截图 + 状态机界面截图 | §4（多处 figure 占位） | 中 | `fig:framework-mujoco-env-placeholder` 等多个图位未替换 |
| Tron1 MuJoCo 仿真 | §4.6（预留） | 低 | 需要 Tron1 MuJoCo 资产与中层映射，当前未做 |

---

## 三、真机层（第五章）

### 3.1 已完成
| 实验 | 证据位置 | 状态 |
|---|---|---|
| Orin 阶段一独立调试 | §5.2.2 tab:orin-stage1-results | ✅ 实测数据已填（相机/模型/DDIM-5/流水线/Bridge/Checklist 全通过） |

### 3.2 待补（高优先·阻塞论文定稿）
| 实验 | 章节 | 说明 |
|---|---|---|
| **Orin + Lite3 阶段二联动** | §5.3.3 | 网络连通 → UDP 心跳 → standup → 低速 twist → 交互式导航；当前只写流程，无实测数据 |
| **Lite3 真机导航正式统计** | §5.6.1（待补小节） | 需要多次重复实验，报告成功率、轨迹长度、CI95、失败模式分布 |
| **Lite3 真机探索正式统计** | §5.6.2（待补小节） | 探索模式持续步数、覆盖范围、图像证据链 |
| **真实环境 topomap 采集与验证** | §5.3.4 | 需要落实 1–2 m 节点间距采集规范并记录节点图像质量 |

### 3.3 待补（中低优先）
| 实验 | 章节 | 优先级 | 说明 |
|---|---|---|---|
| 多目标真机任务 | §5.6.3（待补小节） | 中 | 任务队列真机端执行情况，与仿真 12.601 m 对照 |
| 真机运行照片与 Orin 安装位置照片 | §5 多处 figure 占位 | 中 | 论文多处 `fig:real-*-placeholder` 未替换 |
| 真机 capture/实时/目标三类图像示例 | fig:real-visualization-placeholder | 中 | 论文描述了记录机制但缺示例 |
| Tron1 真机部署 | §5.6.4（预留） | 低 | 需 Tron1 平台与中层映射适配 |

---

## 四、摘要 / 结论所依赖但尚未有数据支撑的结论

以下说法出现在摘要或结论中，完成对应实验后需回过头来校准措辞：

1. "DDIM-2 在保持轨迹质量的同时实现约 4.84× 加速" — 离线已验证；闭环已补 DDIM-2/3/5/10 扫描；现已扩展为跨四编码器稳定（见 tab:encoder-ddim-joint）。
2. "所有 MuJoCo 导航任务均成功到达目标" — 已用 2026-04-21 健康性复核与 3 次重复 benchmark 支撑；论文中明确限定于 MuJoCo 路线稳定复核后的闭环配置。
3. "Orin 阶段一独立调试已通过验证" — 已落实。
4. "真机部署具备分阶段联调流程" — 方法成立，但缺阶段二联动实测数据。
5. "分层结构可迁移到 Tron1" — 纯结构论证，无实测证据。
6. **新增**："TTS-8 的筛选收益对编码器选择具有鲁棒性" — 本日新数据表明结论**需要弱化**：TTS-8 的 Forward Progress 增益与初始 Diversity 正相关（ResNet-50 +16.4%，DINOv2 仅 +6.0%），四个编码器上增益都存在但幅度差异明显。正文 §3.9 已按实际数据表述，摘要/结论若引用此点应避免"鲁棒"一词。

---

## 五、优先级建议

按"毕业论文成稿需要"的先后顺序：

1. **必做（阻塞论文定稿）**：
   - Orin 阶段二联调实测（否则第五章只有方法无结果）
   - Lite3 真机导航/探索统计表（否则第五章的"真机"名不副实）

2. **强烈建议**（决定结论可信度）：
   - 真机最终配置对照（DDPM-10 vs DDIM-2 vs DDIM-2+TTS-8），用于直接支撑第三章最终算法框架的对外可迁移性
   - MuJoCo 失败案例图像证据链

3. **可选**：
   - DDIM-2 在 Orin 上的独立 wall-clock 测量，用以替换正文中按算力比推算的边缘设备时延
   - Tron1 仿真/真机（未来工作兜底）
   - 真机运行照片与 Orin 安装位置照片

---

## 六、统计与证据规范建议

- 对所有新增的 MuJoCo 与真机实验，每个配置至少运行 3 次，报告均值与 CI95
- 失败案例必须保留：输入图像序列、推理 waypoint 序列、速度指令日志、状态机状态轨迹
- 真机成功率的阈值判定需固定：到达距离 < 0.6 m 且在 200 步内完成
- 每轮实验写入 `results/<date>_<experiment>/` 目录，并在论文表格中引用该目录，便于复现
- 正文中若引用某个离线数值（如 Forward Progress、Diversity），必须确保其来自对应表格的同一实验批次，避免跨批次借数
