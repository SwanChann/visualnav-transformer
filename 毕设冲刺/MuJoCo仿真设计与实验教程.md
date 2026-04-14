# MuJoCo 仿真设计与实验教程

## 1. 文档定位

这份文档单独整理 Lite3 为主的 MuJoCo 仿真设计与实验，不再把仿真设计、部署设计、推理模块设计混写在同一处。它主要服务论文第五章中“系统实现与仿真实验”部分。

---

## 2. 仿真工作的目标

MuJoCo 仿真的作用不是简单“让机器人在画面里走起来”，而是承担四项任务：

1. 验证 NoMaD 推理配置是否能迁移到闭环控制。
2. 验证状态机与导航主机是否能在任务级别稳定运行。
3. 验证多目标导航、探索模式等系统能力是否成立。
4. 为真机部署提供低风险预演环境。

---

## 3. 仿真系统组成

当前 Lite3 MuJoCo 仿真由五层组成：

1. 场景与 topomap：`scripts/simulation/lite3_system/topomap.py`
2. 高层 NoMaD 推理：`scripts/shared/nomad_inference.py`
3. 中层 waypoint 到控制量映射：`scripts/simulation/lite3_system/interfaces.py`
4. 底层平台接口：`scripts/simulation/lite3_system/interfaces.py`
5. 状态机与系统循环：`scripts/simulation/lite3_system/states.py`、`scripts/simulation/lite3_system/system.py`

对应的主要入口有四个：

1. 单脚本闭环：`scripts/simulation/nomad_mujoco_lite3_nav.py`
2. 状态机闭环：`scripts/simulation/nomad_mujoco_lite3_state_machine.py`
3. 导航主机：`scripts/deployment/nomad_navigation_host.py`
4. 视觉编码器闭环基准：`scripts/mujoco_encoder_benchmark.py`

---

## 4. 仿真任务设计

当前 MuJoCo 仿真已经覆盖四类任务：

1. 单目标导航。
2. 探索模式。
3. 状态机导航。
4. 同地图多目标导航。

### 4.1 单目标导航

用于验证给定 topomap 和推理配置时，机器人能否完成从当前位置到单一目标点的闭环导航。

### 4.2 探索模式

用于验证 goal mask 为空时，高层推理是否仍能输出稳定 waypoint 序列，而不是系统直接失效。

### 4.3 状态机导航

用于验证 NoMaD 不只是单脚本调用，而是能够接入 `stand -> navigate -> recovery -> completed/failed` 这种完整状态流。

### 4.4 同地图多目标导航

用于验证导航主机能否基于同一地图连续执行多个目标任务，而不需要为每个目标手工重启一次流程。

---

## 5. 基础闭环实验结果

结果目录主要位于：

1. `results/nomad_mujoco/`
2. `results/benchmark/`
3. `results/deployment/`

### 5.1 Lite3 基础闭环结果

| 场景 | 配置 | 结果文件 | 步数 | 总距离(m) | 是否到达 |
|---|---|---|---:|---:|---|
| easy | DDIM-2 | `results/nomad_mujoco/20260410_143820_navigate_mujoco/summary.txt` | 45 | 4.705 | True |
| medium | DDIM-2 | `results/nomad_mujoco/20260410_143829_navigate_mujoco/summary.txt` | 87 | 7.360 | True |
| hard | DDIM-2 | `results/nomad_mujoco/20260410_143840_navigate_mujoco/summary.txt` | 70 | 7.474 | True |
| easy | DDIM-2 + CFG `w=1.0` | `results/nomad_mujoco/20260410_144025_navigate_mujoco/summary.txt` | 42 | 4.354 | True |
| medium | DDIM-2 + CFG `w=1.0` | `results/nomad_mujoco/20260410_144035_navigate_mujoco/summary.txt` | 61 | 6.433 | True |
| hard | DDIM-2 + CFG `w=1.0` | `results/nomad_mujoco/20260410_144047_navigate_mujoco/summary.txt` | 70 | 7.403 | True |
| easy | 探索 DDIM-2 | `results/nomad_mujoco/20260410_144106_explore_mujoco/summary.txt` | 100 | 7.763 | N/A |
| easy | 状态机导航 DDIM-2 | `results/nomad_mujoco/20260410_144134_lite3_state_machine_navigate/summary.txt` | 43 | 4.622 | True |
| easy 三目标点 | 同地图多目标导航 DDIM-2 | `results/nomad_mujoco/20260413_191644_lite3_mujoco_navigation_host_navigate/summary.txt` | 48 | 5.280 | True |

### 5.2 指标含义

1. `步数`：实际执行的闭环控制步数。
2. `总距离(m)`：机器人累计实际路径长度，不是起终点直线距离。
3. `是否到达`：是否满足到达条件或 mission 完成条件。

### 5.3 结果分析

1. 实验设计分析：这一组实验用于验证离线优化得到的推理配置是否能迁移到 MuJoCo 闭环，并检验状态机与导航主机是否真的形成系统能力。
2. 结果维度分析：`DDIM-2` 已经能在 easy、medium、hard 三张地图全部完成闭环；`CFG w=1.0` 在 medium 地图上收益最明显，说明适度引导主要在中等障碍复杂度场景下更有价值；状态机导航和三目标点导航均成功，说明系统已经具备任务级闭环组织能力。
3. 结论：MuJoCo 已经不只是“单目标 demo 环境”，而是完整的 NoMaD 仿真验证平台。

---

## 6. 视觉编码器闭环基准

### 6.1 实验目的

验证视觉编码器替换能否真正迁移到多步闭环导航，而不是只在离线代理指标上看起来更好。

### 6.2 实验设计

固定以下因素：

1. 地图：`easy`、`medium`、`hard`
2. 重复次数：每图 3 次
3. 调度器：`DDIM-2`
4. CFG：`w=0.0`
5. 平台：Lite3 MuJoCo

主变量只保留“编码器不同”。

### 6.3 当前结论

1. `efficientnet_b0` 是当前唯一稳定可靠的闭环部署编码器。
2. `convnext_tiny` 是可继续优化的候选。
3. `dinov2_small` 和 `resnet50` 在当前替换方法下闭环失败。

### 6.4 这一组结果是否还有用

有用，但用途要说准：

1. 它们是筛选阶段的闭环初筛结果。
2. 它们是失败模式分析证据。
3. 它们暂时不是最终公平排名。

原因在于当前 `efficientnet_b0` 与其他编码器并非完全来自同一训练框架，仍然叠加了原始 NoMaD 链路与 backbone-suite 链路的差异。

---

## 7. DDIM / CFG 跨编码器闭环消融

### 7.1 实验目的

验证 DDIM 与 CFG 的最优组合是否具有编码器依赖性，而不是默认对所有编码器都成立。

### 7.2 当前关键结论

1. 对 `EfficientNet-B0`，`DDIM-2 + CFG=0.5` 是当前最稳妥的闭环配置。
2. 对 `ConvNeXt-Tiny`，DDIM 明显优于 DDPM，且 `CFG=0.5` 比 `CFG=1.0` 更稳。
3. 超参数最优点会随着编码器变化而变化，不能把单一编码器上的最优值直接外推给全部模型。

### 7.3 这部分在论文中的意义

这组实验的价值在于说明：

1. 闭环最优配置不完全等同于离线统计最优配置。
2. DDIM 与 CFG 不只是“推理加速参数”，它们和视觉编码器、地图复杂度之间存在耦合。
3. MuJoCo 仿真在这里承担了从离线统计结论到部署配置结论之间的桥梁。

---

## 8. 推荐命令

### 8.1 状态机导航

```bash
python scripts/simulation/nomad_mujoco_lite3_state_machine.py \
  --mode navigate --map easy \
  --scheduler ddim --ddim-steps 2 --cfg-weight 0.5
```

### 8.2 多目标导航主机

```bash
python scripts/deployment/nomad_navigation_host.py \
  --backend mujoco \
  --mode navigate --map easy \
  --goal-source random_points --num-goals 3 \
  --scheduler ddim --ddim-steps 2 --cfg-weight 0.5
```

### 8.3 视觉编码器闭环基准

```bash
python scripts/mujoco_encoder_benchmark.py \
  --encoders efficientnet_b0 convnext_tiny dinov2_small resnet50 \
  --maps easy medium hard \
  --runs 3 --max-steps 300 \
  --scheduler ddim --ddim-steps 2 --cfg-weight 0.0
```

---

## 9. 最终结论

MuJoCo 仿真部分当前已经形成三层结论：

1. 算法层面：NoMaD 的 DDIM/CFG 配置可以通过闭环实验继续优化，不应只看离线表。
2. 系统层面：状态机、导航主机、多目标任务编排已经在仿真中跑通。
3. 迁移层面：MuJoCo 结果已经足以作为真机部署前的必要验证环节。
