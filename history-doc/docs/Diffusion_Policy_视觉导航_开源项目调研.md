# 扩散策略视觉导航开源项目调研报告

> 调研日期：2026年3月24日  
> 关键词：Diffusion Policy, Visual Navigation, NoMaD, GNM, ViNT, 足式机器人, 四足机器人

---

## 一、核心项目家族（Berkeley BAIR 系列）

### 1. robodhruv/visualnav-transformer ⭐ 核心仓库

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/robodhruv/visualnav-transformer |
| **Stars** | ⭐ 1.2k |
| **Forks** | 177 |
| **License** | MIT |
| **最后更新** | ~2024年（2 years ago from 2026） |
| **活跃度** | ⚠️ 不再活跃更新，但社区仍在使用 |

**包含三篇论文的实现：**

#### 1.1 GNM: A General Navigation Model to Drive Any Robot
- **论文日期**：2022年10月（ICRA 2023）
- **作者**：Dhruv Shah, Ajay Sridhar, Arjun Bhorkar, Noriaki Hirose, Sergey Levine
- **论文链接**：https://arxiv.org/abs/2210.03370
- **项目主页**：https://sites.google.com/view/drive-any-robot
- **简介**：第一个通用导航模型，在多种机器人的多样化数据上训练，可zero-shot控制不同机器人。使用目标条件的行为克隆方法。
- **已部署机器人**：LoCoBot, Clearpath Jackal, DJI Tello, Unitree A1, TurtleBot2, Vizbot, CARLA仿真

#### 1.2 ViNT: A Foundation Model for Visual Navigation
- **论文日期**：2023年6月（CoRL 2023 Oral）
- **作者**：Dhruv Shah, Ajay Sridhar, Nitish Dashora, Kyle Stachowicz, Kevin Black, Noriaki Hirose, Sergey Levine
- **论文链接**：https://arxiv.org/abs/2306.14846
- **项目主页**：https://general-navigation-models.github.io/vint/index.html
- **简介**：视觉导航基础模型，使用Transformer架构的视觉编码器进行跨具身体导航。支持微调适配新机器人。

#### 1.3 NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration
- **论文日期**：2023年10月（🏆 ICRA 2024 Best Paper Award）
- **作者**：Ajay Sridhar, Dhruv Shah, Catherine Glossop, Sergey Levine
- **论文链接**：https://arxiv.org/abs/2310.07896
- **项目主页**：https://general-navigation-models.github.io/nomad/index.html
- **简介**：使用扩散策略（Diffusion Policy）生成导航动作，引入目标掩码（Goal Masking）机制，使模型可同时执行目标导向导航和无目标探索。使用DDPM噪声调度器，ConditionalUnet1D作为噪声预测网络。
- **核心创新**：
  - Goal Masking: 通过掩码机制在单一模型中统一导航和探索
  - Diffusion Action Head: 用扩散模型替代确定性动作预测
  - 距离预测网络: 同时预测到目标的距离
  - 使用EfficientNet-B0 + Multi-Head Attention编码器
- **与GNM/ViNT的关键区别**：GNM/ViNT使用确定性动作预测，NoMaD使用扩散策略生成多模态动作分布

---

### 2. robodhruv/drive-any-robot（GNM 独立仓库，已归档）

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/robodhruv/drive-any-robot |
| **Stars** | ⭐ 316 |
| **Forks** | 44 |
| **状态** | ⚠️ 已被 visualnav-transformer 取代 |
| **简介** | GNM 的原始独立代码仓库，现在推荐使用 visualnav-transformer |

---

## 二、核心依赖项目

### 3. real-stanford/diffusion_policy

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/real-stanford/diffusion_policy |
| **Stars** | ⭐ 3.9k |
| **Forks** | 717 |
| **论文** | Diffusion Policy: Visuomotor Policy Learning via Action Diffusion (RSS 2023) |
| **作者** | Cheng Chi, Siyuan Feng, Yilun Du, Zhenjia Xu, Eric Cousineau, Benjamin Burchfiel, Shuran Song |
| **机构** | Columbia University, Toyota Research Institute, MIT |
| **最后更新** | ~2024年 |
| **活跃度** | ⚠️ 低活跃（代码稳定） |

- **简介**：扩散策略的原始实现。NoMaD 的 `ConditionalUnet1D` 噪声预测网络直接依赖于该项目。这是将扩散模型引入机器人策略学习的奠基性工作。
- **与NoMaD关系**：NoMaD安装时需要 `pip install -e diffusion_policy/`，复用了其条件UNet架构。
- **核心贡献**：证明扩散模型可以处理多模态动作分布，优于传统行为克隆方法。

---

## 三、GNM/NoMaD 作者的后续工作

### 4. kylestach/lifelong-nav-rl（LiReN）

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/kylestach/lifelong-nav-rl |
| **Stars** | ⭐ 11 |
| **Forks** | 0 |
| **论文** | LiReN: Lifelong Autonomous Improvement of Navigation Foundation Models in the Wild |
| **作者** | Kyle Stachowicz（GNM/ViNT共同作者）, Lydia Ignatova |
| **最后更新** | ~2025年 |
| **活跃度** | ⚠️ 低活跃 |

- **简介**：导航基础模型的终身自主改进框架。使用Conservative Q-Learning（CQL）替代行为克隆训练，支持在线微调。包含自主探索、自动充电、自动求助等完整自主改进管线。
- **与NoMaD关键区别**：
  - 使用JAX实现而非PyTorch
  - 使用强化学习（CQL）替代扩散策略
  - 专注于在线自主改进能力
  - 支持自动精细调参（在iRobot Create上验证）
- **使用数据**：基于GNM数据集（RLDS格式）

### 5. catglossop/CAST

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/catglossop/CAST |
| **Stars** | ⭐ 32 |
| **Forks** | 2 |
| **论文** | CAST: Counterfactual Labels Improve Instruction Following in Vision-Language-Action Models |
| **作者** | Catherine Glossop（NoMaD共同作者）, William Chen, Arjun Bhorkar, Dhruv Shah, Sergey Levine |
| **最后更新** | 2026年1月 |
| **活跃度** | ✅ 相对活跃 |

- **简介**：视觉-语言-动作（VLA）模型的反事实标签增强，提升指令跟随能力。通过反事实数据增强改善导航VLA模型的语言指令执行。
- **与NoMaD关键区别**：
  - 从diffusion policy转向VLA模型架构
  - 关注语言指令导航（不仅是视觉目标导航）
  - 使用反事实数据增强提升泛化性
- **包含子模块**：visualnav-transformer, diffusion_policy, prismatic-vlms

### 6. catglossop/hierarchical_lifelong_learning

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/catglossop/hierarchical_lifelong_learning |
| **Stars** | ⭐ 0 |
| **Forks** | 0 |
| **作者** | Catherine Glossop（NoMaD共同作者） |
| **最后更新** | ~2024年 |
| **活跃度** | ⚠️ 实验性项目 |

- **简介**：层次化终身学习导航框架，结合VLM引导的无条件探索策略。包含 visualnav-transformer 作为子模块。
- **与NoMaD关键区别**：探索层次化策略学习 + VLM语言引导

### 7. blazejosinski/lm_nav

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/blazejosinski/lm_nav |
| **Stars** | ⭐ 264 |
| **Forks** | 26 |
| **论文** | LM-Nav: Robotic Navigation with Large Pre-Trained Models of Language, Vision, and Action |
| **作者** | Dhruv Shah（GNM/NoMaD作者）等 |
| **最后更新** | 较早 |
| **活跃度** | ⚠️ 不活跃 |

- **简介**：结合大语言模型（LLM）、视觉模型（CLIP/ViLD）和导航模型（ViNG）的机器人导航系统。将自然语言指令转化为导航路径。
- **与NoMaD关键区别**：关注语言指令到导航的转换管线，不使用扩散策略

---

## 四、2024-2025年新兴项目

### 8. InternRobotics/InternNav ⭐ 重点推荐

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/InternRobotics/InternNav |
| **Stars** | ⭐ 740 |
| **Forks** | 93 |
| **论文** | InternVLA-N1: An Open Dual-System Navigation Foundation Model (ICLR 2026) |
| **最后更新** | 2026年3月（2周前！） |
| **活跃度** | ✅ **非常活跃**（17位贡献者） |
| **License** | MIT |

- **简介**：面向通用化导航的一体化开源工具箱，基于PyTorch + Habitat + Isaac Sim。支持最全面的导航基准测试和模型库。
- **核心亮点**：
  - **NavDP**：Navigation Diffusion Policy，在InternVLA-N1中作为System 1使用
  - **InternVLA-N1**：双系统导航基础模型（System 1: 低级视觉导航 + System 2: 高级语言理解）
  - **InternData-N1**：3k+场景、830k VLN数据的大规模导航数据集
  - 内建支持 GNM、ViNT、NoMaD 等基线模型
  - **已支持 Unitree Go2 部署**（含3D打印摄像头支架文件！）
  - 社区部署教程覆盖 Go2 和 G1 系列
- **与NoMaD关键区别**：
  - 规模更大的数据集和模型
  - 双系统架构（快/慢思考）
  - 支持VLN（视觉语言导航）
  - 更完善的仿真评估管线
  - 活跃维护中

### 9. ai4ce/CityWalker ⭐ 重点推荐

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/ai4ce/CityWalker |
| **Stars** | ⭐ 202 |
| **Forks** | 11 |
| **论文** | CityWalker: Learning Embodied Urban Navigation from Web-Scale Videos (CVPR 2025) |
| **作者** | Xinhao Liu, Jintong Li 等 (NYU) |
| **最后更新** | ~2025年9月 |
| **活跃度** | ✅ 活跃 |
| **License** | Apache-2.0 |

- **简介**：利用数千小时的城市行走/驾驶网络视频，通过可扩展的模仿学习训练城市导航智能体。
- **核心亮点**：
  - 从YouTube城市行走视频中学习导航
  - 数据驱动的模仿学习方法
  - 支持 **四足机器人（quadruped）** 部署
  - 户外城市环境导航
  - 直接致谢引用了 ViNT 和 NoMaD
- **与NoMaD关键区别**：
  - 数据来源从机器人数据扩展到网络视频
  - 专注户外城市场景（NoMaD主要室内）
  - 使用视频学习替代传统机器人数据收集
  - 规模化能力更强（web-scale data）

### 10. AnjieCheng/NaVILA ⭐ 重点推荐（足式机器人方向）

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/AnjieCheng/NaVILA |
| **Stars** | ⭐ 556 |
| **Forks** | N/A |
| **论文** | NaVILA: Legged Robot Vision-Language-Action Model for Navigation (RSS 2025) |
| **作者** | An-Chieh Cheng, Yandong Ji, Zhaojing Yang, Xueyan Zou, Jan Kautz, Erdem Bıyık, Hongxu Yin, Sifei Liu, Xiaolong Wang |
| **机构** | UC San Diego, USC, NVIDIA |
| **最后更新** | 2025年8月 |
| **活跃度** | ✅ 活跃 |

- **简介**：专为足式机器人设计的视觉-语言-动作（VLA）导航模型。两级框架：VLA生成高级语言命令 + 实时运动策略保证避障安全。
- **核心亮点**：
  - **已在 Unitree Go2（四足）+ Unitree G1（双足人形）+ Booster T1 上实机验证**
  - 端到端视觉运动策略（无teacher-student蒸馏）
  - 使用LiDAR训练直接减少sim-to-real gap
  - 从YouTube人类旅游视频中学习导航
  - 提出VLN-CE-Isaac高保真物理仿真基准
  - 无需模拟器预训练的waypoint预测器
- **与NoMaD关键区别**：
  - **专注足式机器人**（四足+双足），NoMaD主要是轮式
  - 使用VLA架构而非纯Diffusion Policy
  - 两级层次化架构（高级规划+低级运动）
  - 已在多种足式机器人上验证
  - 视觉运动策略直接处理崎岖地形

---

## 五、其他相关项目

### 11. kylestach/fastrlap-release

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/kylestach/fastrlap-release |
| **Stars** | ⭐ 59 |
| **作者** | Kyle Stachowicz（GNM/ViNT共同作者） |
| **简介** | 快速强化学习自主赛车策略 |

### 12. facebookresearch/home-robot

| 项目 | 详情 |
|------|------|
| **GitHub** | https://github.com/facebookresearch/home-robot |
| **Stars** | ⭐ 1.2k |
| **简介** | Meta的移动操作研究工具（Dhruv Shah参与pin） |

---

## 六、关键对比总结

| 项目 | 年份 | 方法 | 机器人类型 | Stars | 活跃度 | 扩散策略 |
|------|------|------|-----------|-------|--------|---------|
| **GNM** | 2022 | 行为克隆 | 轮式多平台 | 316 | ❌ | ❌ |
| **ViNT** | 2023 | Transformer BC | 轮式多平台 | (1.2k合) | ❌ | ❌ |
| **NoMaD** | 2023 | **Diffusion Policy** + Goal Masking | 轮式(LoCoBot) | (1.2k合) | ❌ | ✅ |
| **Diffusion Policy** | 2023 | Diffusion DDPM | 机械臂 | 3.9k | ⚠️ | ✅ |
| **LiReN** | 2024 | CQL + 在线微调 | 轮式(iRobot) | 11 | ⚠️ | ❌ |
| **CAST** | 2025 | VLA + 反事实 | 轮式 | 32 | ✅ | ❌ |
| **InternNav** | 2025-26 | **NavDP** + VLA双系统 | 轮式+**Go2四足** | 740 | ✅✅ | ✅ |
| **CityWalker** | 2025 | 视频模仿学习 | **四足机器人** | 202 | ✅ | ❌ |
| **NaVILA** | 2025 | **VLA** + 运动策略 | **Go2四足+G1双足** | 556 | ✅ | ❌ |

---

## 七、与你的研究方向最相关的推荐

### 🎯 如果你关注「扩散策略 + 导航」：
1. **NoMaD** (robodhruv/visualnav-transformer) — 奠基性工作，必读
2. **InternNav** (InternRobotics/InternNav) — NavDP是NoMaD的升级，活跃维护
3. **Diffusion Policy** (real-stanford/diffusion_policy) — 底层依赖，理解核心原理

### 🎯 如果你关注「足式机器人 + 视觉导航」：
1. **NaVILA** (AnjieCheng/NaVILA) — **最直接相关**，Go2+G1实机验证
2. **InternNav** — 已有Go2/G1部署支持和社区教程
3. **CityWalker** — 四足机器人户外导航

### 🎯 如果你关注「扩散策略 + 足式机器人运动」：
- 目前**直接结合扩散策略和足式机器人导航的开源项目非常稀少**
- NoMaD的扩散策略输出的是2D waypoint（适合轮式），需要额外的低级运动控制器
- NaVILA等使用VLA架构替代了纯扩散策略
- **这是一个有价值的研究空白**：将NoMaD的扩散导航策略与足式机器人的运动控制器相结合

### 🎯 如果你想找「ViNL (Visual Navigation and Locomotion)」：
- ViNL是CMU提出的概念，将视觉导航与足式运动解耦为分层策略
- 目前没有找到名为"ViNL"的独立GitHub仓库
- 最接近的实现是 **NaVILA** 的两级框架（VLA导航 + locomotion policy）

---

## 八、未找到的项目说明

| 项目名称 | 状态 |
|----------|------|
| LNM (Large Navigation Model) | ❌ 未找到以此命名的独立项目 |
| ViNL 独立仓库 | ❌ 无独立GitHub仓库（概念存在于论文中） |
| jaredmejia/llm_nav | ❌ 404 未找到 |
| SPINE | ❌ 未找到与导航直接相关的项目 |

---

## 九、技术演进路线图

```
GNM (2022, BC)
  │
  ├── ViNT (2023, Transformer BC, 基础模型)
  │     │
  │     ├── NoMaD (2023, Diffusion Policy + Goal Masking) ◄── Diffusion Policy (Stanford, 2023)
  │     │     │
  │     │     ├── InternNav/NavDP (2025, 大规模双系统) ← 融合VLM
  │     │     └── CAST (2025, VLA + 反事实增强)
  │     │
  │     └── LiReN (2024, RL在线微调)
  │
  └── LM-Nav (2022, LLM + 导航)

独立发展线：
CityWalker (2025, 网络视频学习导航，支持四足)
NaVILA (2025, VLA足式机器人导航，Go2+G1) ← 最直接的足式导航方案
```

---

*本报告基于2026年3月24日的GitHub公开信息整理，项目状态可能随时变化。*
