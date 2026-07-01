# 缺失实验清单（2026-04-26 更新）

> 基于对整本论文各章节的梳理，列出论文中存在但尚未完成或未填入数据的实验项。
>
> **2026-04-22**：算法层编码器联合消融已补齐；第三章框架介绍的若干事实错误（双编码器结构、token 池化方式、冻结策略口径）按代码实际行为修正；所有已补实验都在 `results/` 下有时间戳目录。
> **2026-04-26**：剔除全部 Tron1 相关条目；按"完整性"角度新增多组仿真消融；将真机层重组为"室内 / 室外"两块；**指标体系全面精简，真机实验统一收敛到"成功率 + 前进长度 + 跌倒/干预次数"四个指标**。

---

## 〇、指标定义与公式约定（全文统一）

为避免口径漂移，先固定下面这套指标。后面所有"待补"条目都直接引用，不再每条重复公式。

### 0.1 真机核心指标（4 个，足够覆盖论文需要）

每条 run 只需要测以下 4 个值：

| 指标 | 来源 | 公式 / 取数方式 |
|---|---|---|
| **成功率 SR** | 计数 | $\text{SR} = N_{\text{success}} / N_{\text{runs}}$；单次 success 的判定：物理到达目标 < 0.6 m 半径圆，且全程无人工干预、无跌倒 |
| **前进长度** | 系统日志 | `path_distance`（m），由 `lite3_system` 在 `results.json` 中自动累加每周期位移；机器人沿走过的轨迹的弧长 |
| **跌倒次数** | 人工记录 | 全程触发任意一次 `is_fallen()` 或人工扶起，计 1 次 |
| **人工干预次数** | 人工记录 | 急停、托扶、推动、改方向，每次计 1 |

可选补充（仅 1.2.1 真机算法对照实验需要）：

| 指标 | 来源 | 公式 |
|---|---|---|
| 平均周期时延 $\bar L_{ms}$ | Orin 日志 `inference_log.csv` 中 `t_ms` 字段 | $\overline{t_{ms}}$ |

### 0.2 仿真核心指标（4 个）

| 指标 | 来源 | 公式 |
|---|---|---|
| 成功率 SR | `mujoco_encoder_benchmark` 输出 | $N_{\text{success}}/N_{\text{runs}}$；single-run success = `final_goal_dist < 0.5 m` 且未跌倒 |
| 前进长度 | run 输出 | `path_distance`（m）|
| 闭环时延 | run 输出 | `wall_time / steps`（s/周期）|
| 重复均值与置信区间 | 多种子聚合 | $\bar v = \frac{1}{n}\sum_k v_k$；$\text{CI}_{95} = 1.96 \cdot s/\sqrt{n}$，$s$ 取无偏样本标准差 |

闭环判定参数来自 [scripts/simulation/lite3_system/system.py:762-984](scripts/simulation/lite3_system/system.py#L762-L984)：物理到达阈值 `GOAL_REACH_DIST=0.5 m`，跌倒由 `is_fallen()` 触发，步数上限实测设为 $10^9$（无效约束）。

### 0.3 离线推理指标（仅算法层第三章使用，已完成不再修改）

来自 [scripts/nomad_eval_common.py:289-309](scripts/nomad_eval_common.py#L289-L309)，包括 Forward Progress、Diversity、Path Length、Latency 等 8 项；公式见原代码注释。本节只是声明它们与第三章已有表格保持一致，新增实验不再使用这套口径。

### 0.4 真机测量操作约定

- **起终点位置**：激光卷尺贴地测量直线距离即可，不要求高精度坐标。
- **前进长度**：直接读 `results.json` 的 `path_distance`，无需人工测量。
- **跌倒/干预**：双人现场记录，事后视频核对。
- **数据落盘**：每条 run 的 `results.json` + 相机视频 MP4 都保留在 `results/deployment/<timestamp>_<label>/` 中，复核时直接打开。

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
| 编码器 × DDIM 联合消融 | §3.9 tab:encoder-ddim-joint | ✅ `results/day5/20260422_171021_encoder_joint_ablation_experiment` |
| 编码器 × TTS 联合消融 | §3.9 tab:encoder-tts-joint | ✅ 同上 |
| 编码器 × CFG 联合消融（仅 EfficientNet-B0 / ResNet-50） | §3.9 tab:encoder-cfg-joint | ✅ 同上 |
| 算法层事实校正（双编码器、池化方式、冻结口径、时延基准） | §3.2 / §3.4 | ✅ 2026-04-22 按代码核对后修订 |


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
| 跨视觉编码器 MuJoCo 闭环对比（仅 DDPM-10 默认） | §4.5 tab:mujoco-encoder-closed-loop | ✅ `results/benchmark/20260421_seeded_mujoco_encoder_benchmark` |

### 2.2 待补

> 已有的"跨编码器闭环对比"只在 DDPM-10 默认配置下、每格 1 个种子完成，不能直接支撑"鲁棒"或"可推广"等措辞。下面 2.2.1 / 2.2.2 是用户要求的"不同视觉编码器接入后的仿真表现"实验的细化与扩展；2.2.3--2.2.6 是从实验完整性角度补的实验。

> 仿真所有指标统一按 §0.2，每条记录用脚本自动落到 JSON，不需人工读数。

#### 2.2.1 编码器 × 调度器闭环深扫（高优先，新增）
- **目的**：把第三章 `tab:encoder-ddim-joint` 的离线结论平移到 MuJoCo 闭环，回答"DDIM-2 提速对编码器选择是否敏感"。
- **配置矩阵**：4 编码器 × {DDPM-10, DDIM-5, DDIM-3, DDIM-2} × 3 张地图 × 3 种子 = 144 条。
- **指标**：SR、前进长度、闭环时延（mean ± CI95）。
- **脚本**：[scripts/mujoco_encoder_benchmark.py](scripts/mujoco_encoder_benchmark.py) 加 `--schedulers ddpm ddim --ddim-steps 10 5 3 2 --runs 3`。
- **章节**：§4.5（新增 `tab:mujoco-encoder-ddim-closed-loop`）。

#### 2.2.2 编码器 × TTS 闭环深扫（中优先，新增）
- **目的**：在 MuJoCo 闭环中验证 TTS-8 对不同编码器的增益方向是否与离线一致。
- **配置矩阵**：4 编码器 × {TTS-1, TTS-8} × medium × 3 种子 = 24 条。
- **指标**：SR；并报告 $\Delta\text{SR}_e=\text{SR}_{e,\text{TTS}=8}-\text{SR}_{e,\text{TTS}=1}$。
- **脚本**：[scripts/simulation/nomad_mujoco_lite3_state_machine.py](scripts/simulation/nomad_mujoco_lite3_state_machine.py) 加 `--tts --tts-budget 8`；批量驱动复用 `mujoco_encoder_benchmark.py`。
- **章节**：§4.5（新增 `tab:mujoco-encoder-tts-closed-loop`）。

#### 2.2.3 闭环重复实验补 CI95（高优先，新增完整性）
- **目的**：现有 `tab:mujoco-{ddim,cfg,encoder}-closed-loop` 多数为单种子，无法回答"是否显著"。
- **方法**：每格重复 3 次（`--goal-seed [7, 19, 31]`），其余配置不变。
- **指标**：SR ($k/3$)、前进长度（mean ± CI95）。
- **章节**：§4.5 同表刷新。

#### 2.2.4 topomap 节点密度敏感性（中优先，新增完整性）
- **目的**：当前 `--topomap-step` 默认为 5，未量化"节点稀疏 / 稠密"对成功率的影响；真机端最容易踩的坑。
- **配置矩阵**：medium × `--topomap-step ∈ {2, 3, 5, 8, 12}` × DDIM-3 × 3 种子 = 15 条。
- **指标**：SR、前进长度。
- **章节**：§4.5（新增 `tab:mujoco-topomap-density`）。


## 三、真机层（第五章）

> 用户要求把真机部分拆成"室内"与"室外"两类，每类独立给出实验内容与测量方法。下面 3.2 是阶段性必做，3.3 / 3.4 分别为室内 / 室外的具体实验单。

### 3.1 已完成
| 实验 | 证据位置 | 状态 |
|---|---|---|
| Orin 阶段一独立调试 | §5.2.2 tab:orin-stage1-results | ✅ 实测数据已填（相机/模型/DDIM-5/流水线/Bridge/Checklist 全通过） |

### 3.2 阻塞论文定稿的真机实验（高优先，必做）

#### 3.2.1 Orin + Lite3 阶段二联动验证
- **目的**：当前 §5.3.3 只写流程不给数据；需要给出网络连通 → UDP 心跳 → standup → 低速 twist → 交互式导航的完整时序证据。
- **测量数据**：
  - 网络连通 ping 平均/最大延时（`ping -c 1000`）；
  - UDP 心跳序列号丢失率（`lite3_command.py` 日志中 `seq` 字段）；
  - standup 完成时间 $t_{\text{standup}}$（站立完成回报到的 tick）；
  - 第一次低速 twist 实际响应延时 $t_{\text{cmd→motion}}$（命令时间戳 vs IMU 速度突变时间戳）。
- **指标与公式**：
  - 心跳丢包率 $\text{PLR}=N_{\text{missing}}/N_{\text{sent}}$；
  - 命令响应延时 $\Delta t = t_{\text{IMU突变}}-t_{\text{cmd下发}}$，从 `cmd_log.csv` / `imu_log.csv` 用最近邻时间戳对齐计算。
- **测量方法**：上位机/Orin 日志即可，无需人工掐表。
- **章节**：§5.3.3。

#### 3.2.2 真实环境 topomap 采集与质量验证
- **目的**：落实 1--2 m 节点间距的采集规范，并用第一章定义的指标量化节点图像质量。
- **测量数据**：每条 topomap 采集时记录 (节点编号, GPS/卷尺 $(x,y)$, 节点图像路径, 采集时光照)。
- **指标与公式**：
  - 节点间距 $d_i=\lVert p_{i+1}-p_i\rVert$，记录均值和方差；
  - 节点图像可识别性：用 NoMaD 视觉编码器对相邻节点 (i, i+1) 的 token 余弦相似度 $\cos\langle f_i, f_{i+1}\rangle$ 必须 $<0.95$（防止重复点位）。
- **测量方法**：采集时人工卷尺；图像相似度由 [scripts/analysis/check_dataset.py](scripts/analysis/check_dataset.py) 离线扫描即可。
- **章节**：§5.3.4。

### 3.3 室内真机实验（必做+建议）

#### 3.3.1 直走廊导航（必做）
- **环境**：实验室主走廊或宿舍走廊，单段直线 $\ge 15$ m，两侧连续墙体。
- **目的**：基础导航能力底线；用最简单的视觉条件验证整链能跑通。
- **测量数据**：每条 run 记录起点与目标点 $(x,y)$（卷尺）、`success`、`final_goal_dist`、`path_distance`、`wall_time`、人工干预次数、跌倒次数、视频。
- **指标与公式**：
  - 成功率 $\text{SR}=k/N$（$N\ge 8$ 次）；
  - 路径相对偏长 $\eta=\bar L_{\text{actual}}/L_{\text{geo}}$；
  - 平均决策时延 $\bar L$（来自 Orin `inference_log.csv`）。
- **测量方法**：起终点用激光卷尺；轨迹长度用第三方 SLAM（若有）或事后视频对地砖计数估算（地砖边长当尺）；其他全部由系统日志给出。
- **章节**：§5.6.1（待补小节）。

#### 3.3.2 L 型 / U 型走廊导航（必做）
- **环境**：包含一处 90°或 180° 拐弯的走廊（实验室拐角，长度各 6--8 m）。
- **目的**：测试转弯阶段视觉编码 + topomap 切换的稳定性。
- **测量数据**：除 3.3.1 外额外记录拐弯处的 `closest_node` 切换序列、出现 `recovery` 的次数。
- **指标与公式**：
  - 拐弯成功率 $\text{SR}_{\text{turn}}$（拐弯阶段未触发跌倒/recovery 即视为通过）；
  - 拐点滞后步数 $\Delta n_{\text{turn}}$：实际开始转向的 tick 与几何拐点对应 tick 的差。
- **测量方法**：人工标注几何拐点对应的视频帧（即录制视频中机器人走到拐角时的帧号），再读取 `state_log.csv` 中机器人的 yaw rate 跃变 tick。
- **章节**：§5.6.1。

#### 3.3.3 多目标室内任务（建议）
- **环境**：单层办公楼，含 3 个标定目标点（A → B → C），节点总跨度约 25 m，含至少 2 次拐弯。
- **目的**：与仿真 12.601 m 多目标对照，验证状态机 mission 切换在真机上的健壮性。
- **测量数据**：每个目标段独立记录 (起点, 终点, success, 路径长度, 时延)；任务总成功率 = 所有段都 success 的次数比。
- **指标与公式**：
  - 段内 SR；任务级 SR；
  - 段间切换间隙 $t_{\text{switch}}$：上一段 `completed` 到下一段 `navigate` 之间的墙钟差。
- **测量方法**：起点终点卷尺；其余日志。
- **章节**：§5.6.3（待补小节）。

#### 3.3.4 室内光照鲁棒性（建议）
- **环境**：3.3.1 的同一走廊，分别在 (上午自然光, 下午背光, 夜间荧光) 三种光照下重复同一 topomap 任务。
- **目的**：度量光照变化对成功率的影响（真机第一性问题）。
- **测量数据**：三种光照下各 5 条 run 的 `success`、`final_goal_dist`、`recovery_count_total`、相机平均亮度（OpenCV `cv2.mean(frame)[0]`）。
- **指标与公式**：
  - 各光照成功率 SR；
  - 光照差异 $\Delta\text{lux} = $（白天平均亮度 - 夜间平均亮度），与 SR 做相关性分析。
- **测量方法**：相机亮度从 `images/` 目录每条 run 抽 30 张算均值；光照变化用手机照度计辅助记录定性分级。
- **章节**：§5.6.1 子条目。

#### 3.3.5 室内动态障碍干扰（可选）
- **环境**：3.3.1 走廊中预先安排 1 名行人在第 5--10 m 处缓慢横穿。
- **目的**：度量 NoMaD 对未训练过的动态障碍的反应（推理是否会偏向避让）。
- **测量数据**：是否触发 stuck/recovery、是否撞到行人（人工旁观+视频回看）、行人通过后是否能恢复任务。
- **指标与公式**：避让成功率 $\text{SR}_{\text{avoid}}=$（未碰撞 ∧ 任务完成）的次数比。
- **测量方法**：视频人工判定。**安全条件**：行人佩戴防撞护具，机器人速度限到 0.4 m/s。
- **章节**：§5.6.1 子条目。

### 3.4 室外真机实验（必做+建议）

> 室外是真机部分最有论文价值的实验，但风险也最高。下面所有实验均要求：(1) 风速 $<5$ m/s 且无降水；(2) 地面平整无落叶/积水；(3) 人工陪护至少 2 人，1 人持遥控急停。

#### 3.4.1 校园人行道直线导航（必做）
- **环境**：校园主路人行道，长度 30--50 m，目标点放在尽头树下。
- **目的**：室外基线，验证视觉模型在真实纹理（地砖、绿化、阳光）下能否导航。
- **测量数据**：与 3.3.1 完全相同；额外记录采集时刻天气（晴 / 阴 / 多云）。
- **指标与公式**：与 3.3.1 完全相同；并报告与 3.3.1 的成功率差 $\Delta\text{SR}=\text{SR}_{\text{outdoor}}-\text{SR}_{\text{indoor}}$。
- **测量方法**：起终点用 GPS（精度 $\pm 1$ m）或卷尺；其余系统日志。
- **章节**：§5.6.1（室外子节）。

#### 3.4.2 室外开阔场地导航（必做）
- **环境**：操场或停车场，目标点距起点 20--30 m，视野中地标稀疏。
- **目的**：测试在缺乏强结构线索（如墙壁）时的退化表现。
- **测量数据**：同 3.3.1，额外记录每周期 `predicted_distance`（NoMaD 距离头输出，来自 `inference_log.csv`）。
- **指标与公式**：
  - 距离头一致性 $\rho_{\text{dist}}=\text{corr}(\text{predicted\_distance}, \text{ground\_truth\_distance})$，其中 ground truth 由起点到当前位置的 GPS/卷尺距离的差给出。
  - SR、$\eta$、$\bar L$。
- **测量方法**：每条 run 在起点贴坐标基线，事后用视频对每秒一帧标注机器人位置（地砖或操场跑道线作刻度）。
- **章节**：§5.6.1（室外）。

#### 3.4.3 室外多段任务（建议）
- **环境**：教学楼 A 出口 → 道路拐角 → 教学楼 B 入口，含一次 90° 拐弯，全长 40--60 m。
- **目的**：与 3.3.3 室内多目标对照，验证室外长程任务可行性。
- **测量数据**：同 3.3.3；额外记录从 GPS 漂移导致的目标偏移（如有）。
- **指标与公式**：段内 SR、任务级 SR、段间切换时间 $t_{\text{switch}}$。
- **测量方法**：日志 + GPS。
- **章节**：§5.6.3（室外子节）。

#### 3.4.4 光照与天气鲁棒性（建议）
- **环境**：3.4.1 的同一路径，分别在 (中午直射, 傍晚低角度光, 阴天) 三种条件下用同一 topomap 任务。
- **目的**：度量户外光照对视觉编码器的冲击（与 3.3.4 室内对照）。
- **测量数据**：每条 run 拍摄前 5 帧 FPV 图像（用于事后计算亮度方差）、`success`、`recovery_count_total`、相机自动曝光时间（如能从 ROS 相机话题读取）。
- **指标与公式**：
  - SR by condition；
  - 图像方差 $\sigma^2(I)$ 在三种条件下的分布；
  - 跨条件成功率方差 $\text{Var}(\text{SR})$ 作为"鲁棒度"指标。
- **测量方法**：图像统计离线计算；天气定性记录。
- **章节**：§5.6.1（室外）子条目。

#### 3.4.5 室外 topomap 时间偏移测试（建议）
- **环境**：3.4.1 路径采集 topomap 后，间隔 24 / 72 小时再用同一 topomap 重测。
- **目的**：考察自然环境（光照/落叶/车辆移动）随时间漂移对节点匹配的影响。
- **测量数据**：每个时刻 5 条 run 的 SR、节点匹配距离 `predicted_distance` 序列、节点跳跃 $\bar j$（同 2.2.4）。
- **指标与公式**：
  - SR 衰减 $\Delta\text{SR}_t=\text{SR}_0-\text{SR}_t$；
  - 节点匹配下降率 $\Delta\rho = \overline{\text{predicted\_distance}_t} - \overline{\text{predicted\_distance}_0}$。
- **测量方法**：日志统计。
- **章节**：§5.6.1（室外）子条目，挂"附录"占位也可。

### 3.5 通用真机统计与照片证据（中低优先）
- **目的**：解决论文多处 `fig:real-*-placeholder` 与 capture/realtime/goal 三类图像缺示例问题。
- **要做**：
  - 全部室内+室外实验均需录像；每条 run 抽 (capture, realtime, goal) 三类各 1 张，按时间戳归档到 `results/figures/real_robot/`。
  - 整理 Orin 安装位置、Lite3 真机外观、采集 topomap 现场各 2 张照片。
- **章节**：§5 多处占位。

---

## 四、摘要 / 结论所依赖但尚未有数据支撑的结论

以下说法出现在摘要或结论中，完成对应实验后需回过头来校准措辞：

1. "DDIM-2 在保持轨迹质量的同时实现约 4.84× 加速" — 离线已验证；闭环已补 DDIM-2/3/5/10 扫描；现已扩展为跨四编码器稳定（见 tab:encoder-ddim-joint）。**还需要 2.2.1 的闭环数据进一步固化"对编码器选择不敏感"这一推论**。
2. "所有 MuJoCo 导航任务均成功到达目标" — 已用 2026-04-21 健康性复核与 3 次重复 benchmark 支撑；论文中明确限定于 MuJoCo 路线稳定复核后的闭环配置。
3. "Orin 阶段一独立调试已通过验证" — 已落实。
4. "真机部署具备分阶段联调流程" — 方法成立，但缺阶段二联动实测数据（见 3.2.1）。
5. "TTS-8 的筛选收益对编码器选择具有鲁棒性" — 离线数据表明结论**需要弱化**：TTS-8 的 Forward Progress 增益与初始 Diversity 正相关（ResNet-50 +16.4%，DINOv2 仅 +6.0%），四个编码器上增益都存在但幅度差异明显。正文 §3.9 已按实际数据表述，摘要/结论若引用此点应避免"鲁棒"一词。**闭环验证待 2.2.2 完成后再决定要不要重写**。

---

## 五、优先级建议

按"毕业论文成稿需要"的先后顺序：

1. **必做（阻塞论文定稿）**
   - 3.2.1 Orin 阶段二联调实测（否则第五章只有方法无结果）
   - 3.3.1 / 3.3.2 室内直走廊与 L 型走廊导航（真机最小可信结果集）
   - 3.4.1 室外人行道直线导航（"室内+室外"两类至少各一条）
   - 2.2.3 闭环重复实验补 CI95（既有表的可信度补强，工作量低）

2. **强烈建议（决定结论可信度）**
   - 1.2.1 真机最终配置对照（DDPM-10 vs DDIM-2 vs DDIM-2+TTS-8）
   - 2.2.1 编码器 × 调度器闭环深扫（响应用户"增加视觉编码器接入仿真表现"要求的核心）
   - 3.3.3 / 3.4.3 多目标室内/室外任务（与仿真 12.601 m 对照）
   - 2.2.6 失败案例图像证据链

3. **可选**
   - 1.2.2 DDIM-2 在 Orin 上的独立 wall-clock 测量
   - 2.2.2 编码器 × TTS 闭环深扫
   - 2.2.4 topomap 节点密度敏感性
   - 2.2.5 Recovery 子机有效性消融
   - 3.3.4 / 3.4.4 室内/室外光照鲁棒性
   - 3.3.5 室内动态障碍干扰（注意安全成本）
   - 3.4.5 室外 topomap 时间偏移

---

## 六、统计与证据规范建议

- 所有新增 MuJoCo 与真机实验，每个配置至少运行 3 次；指标按 0.2 节给出 mean ± CI95。
- 失败案例必须保留：输入图像序列、推理 waypoint 序列、速度指令日志、状态机状态轨迹。
- 真机成功率的阈值判定固定为：到达距离 $<0.6$ m，无人工干预，无跌倒。
- 每轮实验写入 `results/<date>_<experiment>/` 目录，并在论文表格中引用该目录，便于复现。
- 正文引用某个离线数值（如 Forward Progress、Diversity）时，必须确保其来自对应表格的同一实验批次，避免跨批次借数。
- 真机视频须保存原始 MP4（不剪辑），文件名约定 `YYYYMMDD_HHMM_<location>_<run-id>.mp4`，便于回看与复核。
