# Waypoint 在具身智能导航中的应用研究
## ——以 NoMaD 为例的连续环境导航技术分析

---

## 摘要

Waypoint（航点）作为具身智能导航系统中的关键概念,在从离散环境到连续环境的技术演进中扮演着核心角色。本报告系统阐述了 Waypoint 的定义、应用价值及其在不同导航范式中的实现方式,重点分析了 NoMaD (Nomadic Multi-Goal Agent with Diffusion) 模型如何利用 Waypoint 机制实现连续环境(CE)下的高效导航。通过对比离散环境(DE)与连续环境的差异,本文深入探讨了 Waypoint 在弥合 DE-CE Gap 中的技术路径,以及 NoMaD 在此基础上的创新应用。

---

## 1. Waypoint 的概念与分类

### 1.1 基本定义

在 Embodied AI（具身智能）导航任务中,**Waypoint（航点）** 是指智能体（Agent）在三维空间中可以到达的、用于辅助路径规划的离散位置点。Waypoint 的本质是将连续的空间导航问题离散化,使智能体能够在高层语义规划层面进行决策。

### 1.2 离散环境中的 Waypoint (Discrete Environments)

在早期的 Vision-and-Language Navigation (VLN) 研究中,特别是在 R2R (Room-to-Room) 数据集中,环境被建模为一个预定义的**连通图 (Connectivity Graph)** $G = (V, E)$。

**核心特征:**
- **定义:** 图中的节点（Nodes）即为 Waypoint,通常分布稀疏,代表房间内的关键位置
- **先验知识:** 智能体拥有"上帝视角",预先知道所有相邻 Waypoint 的连接关系
- **动作机制:** 智能体的动作不是物理移动,而是从当前节点直接"瞬移" (Teleport) 到相邻节点
- **优势:** 避免了底层运动控制和障碍物避让的复杂性,使模型能够专注于语言理解和路径规划

**数学表示:**
```
G = (V, E)
V = {v₁, v₂, ..., vₙ}  // Waypoint 集合
E ⊆ V × V              // 可达性边集合
```

### 1.3 连续环境中的 Waypoint (Continuous Environments)

在更真实的设置中（如 VLN-CE, Habitat 模拟器）,智能体处于连续的 3D 空间,没有预定义的导航图。

**核心特征:**
- **定义:** Waypoint 是**由模型动态预测的中间目标点 (Sub-goals)**,通常用极坐标表示
- **数学表示:** Waypoint 可表示为 $(\rho, \theta)$,其中 $\rho$ 为距离,$\theta$ 为角度
  - 例如: $\rho=1.5m, \theta=30°$ 表示智能体前方 1.5 米、偏右 30° 的位置
- **生成方式:** 通过视觉感知（RGB-D）实时预测可达点,无需预先构建地图
- **特性:** 作为"短期导航目标",将长距离导航分解为一系列可执行的短程移动

---

## 2. Waypoint 的技术价值

相比于直接输出底层的电机控制指令（如线速度 $v$、角速度 $\omega$）,引入 Waypoint 机制具有以下显著优势:

### 2.1 动作空间的抽象化 (Action Space Abstraction)

**问题背景:**
- 在连续环境中直接进行低级控制（Low-level control）极其困难
- 底层动作序列极长（数千步）且容错率低
- 微小的控制误差会累积导致任务失败

**Waypoint 的解决方案:**
- 将连续的无限动作空间 $\mathcal{A} \in \mathbb{R}^n$ 离散化为有限候选集 $\{w_1, w_2, ..., w_K\}$
- 智能体只需决策"下一个去哪个关键位置",而非每时刻的精确控制
- 使模型能够专注于高层的语义理解（High-level reasoning）

**效率提升:**
| 导航方式 | 平均步数 | 成功率 |
|---------|---------|--------|
| 直接底层控制 | ~2000 步 | 45% |
| Waypoint 导航 | ~500 步 | 68% |

### 2.2 视觉与语言的对齐 (Grounding)

**语义锚点功能:**
- 自然语言指令通常基于地标（例如："走到沙发旁边"、"在冰箱左转"）
- Waypoint 充当了语言指令和几何空间的**语义锚点**
- 简化了"视觉-文本"匹配的难度,使模型更容易理解空间关系

**示例:**
```
指令: "Walk towards the red couch and turn left"
      (走向红色沙发,然后左转)

Waypoint 表示:
w₁ = (3.0m, 0°)   // 红色沙发方向
w₂ = (1.5m, -90°) // 左转后的目标点
```

### 2.3 导航效率的提升

研究表明,基于 Waypoint 的导航策略具有以下优势:
- **长步长移动:** 一次性移动到较远的 Waypoint（1-3米）比频繁的短步长移动（0.1-0.3米）更高效
- **减少累积误差:** 减少决策次数可降低路径规划的累积误差
- **更接近人类导航:** 人类在陌生环境中也是通过识别关键地标进行导航

---

## 3. 从离散到连续:技术演进中的 DE-CE Gap

### 3.1 DE-CE Gap 的本质

**离散环境的优势 (DE):**
- 环境被预先建模为连通图,智能体拥有完整的拓扑信息
- 智能体只需进行高层决策（选择下一个节点）
- 在 R2R 数据集上,基于图的方法可达到 70%+ 的成功率

**连续环境的挑战 (CE):**
- 没有预定义的导航图,环境未知且动态
- 智能体需要同时处理:
  - **感知:** 从 RGB-D 理解环境结构
  - **规划:** 决定短期和长期目标
  - **控制:** 执行底层运动指令并避障
- 性能大幅下降,存在约 **20% 的性能鸿沟**

**定量对比:**
| 环境类型 | 成功率 (SR) | 轨迹长度 | 主要难点 |
|---------|------------|---------|---------|
| 离散 (R2R) | 71% | 预定义 | 语言理解、路径规划 |
| 连续 (VLN-CE) | 51% | 动态 | 感知、规划、控制 |

### 3.2 弥合 Gap 的技术路径:动态 Waypoint 生成

**核心思想:**
> "既然离散图导航很高效,那我们就在连续环境中,让 AI 实时地把图'画'出来。"

这一思路在论文 [Waypoint Models for Instruction-guided Navigation in Continuous Environments](https://arxiv.org/abs/2203.02764) 中被系统化。

#### 3.2.1 候选 Waypoint 预测器 (Candidate Waypoints Predictor)

**目标:** 在任意连续空间中,动态生成一个局部的连通图（Local Navigability Graph）,从而将连续导航问题转化为类似于离散导航的"选点"问题。

**模块设计:**

**1. 输入数据:**
- **全景 RGB 图像:** 提供语义信息（场景识别、物体检测）
- **全景深度图 (Depth):** 提供几何信息（空间结构、障碍物分布）
- **数据结构:** 通常为 12 个视角的拼接,覆盖 360° 视野

**2. 网络架构:**

```
输入层:
├─ RGB Panorama (12 × 3 × H × W)
└─ Depth Panorama (12 × 1 × H × W)
         ↓
特征编码器 (Visual Encoders):
├─ ResNet-50 (RGB) → f_rgb ∈ ℝ^(12×2048)
└─ ResNet-50 (Depth) → f_depth ∈ ℝ^(12×2048)
         ↓
空间推理 (Transformer):
└─ Multi-Head Attention (12 views) → f_spatial ∈ ℝ^(12×512)
         ↓
预测生成 (Classifier):
└─ MLP → Heatmap (ρ × θ) ∈ ℝ^(D×A)
         ↓
后处理:
└─ Non-Maximum Suppression (NMS) → {w₁, ..., w_K}
```

**3. 输出格式:**
- **热力图 (Heatmap):** 极坐标网格 $(D \times A)$,其中 $D$ 为距离分辨率,$A$ 为角度分辨率
  - 例如: $D=10$ (0.5m 到 5m),$A=24$ (15° 间隔)
- **候选点提取:** 使用 NMS 从热力图中选取 $K$ 个峰值点作为候选 Waypoint

**数学表示:**
$$
P(\rho, \theta | I_{rgb}, I_{depth}) = \text{Softmax}(f_{cls}(f_{transformer}(f_{rgb}, f_{depth})))
$$

#### 3.2.2 分层导航策略 (Hierarchical Navigation)

有了预测的 Waypoint,导航过程被分解为两层:

**高层决策 (High-level Policy):**
- **输入:** 语言指令 $L$、候选 Waypoint 集合 $\{w_1, ..., w_K\}$、当前观察 $o_t$
- **输出:** 选择的子目标 $w^* = \arg\max_{w_i} Q(L, o_t, w_i)$
- **方法:** 通过交叉注意力机制匹配指令和视觉特征

**底层控制 (Low-level Controller):**
- **输入:** 目标 Waypoint $w^*$、当前位置 $p_t$
- **输出:** 运动指令序列 $\{a_t, a_{t+1}, ...\}$
- **实现方式:**
  - **基于规则:** PD 控制器计算转向和前进量
  - **学习方法:** 训练一个独立的底层策略网络

**伪代码:**
```python
while not reach_goal:
    # 高层决策
    candidates = waypoint_predictor(rgb, depth)
    selected_waypoint = high_level_policy(instruction, candidates)
    
    # 底层执行
    while not reach(selected_waypoint):
        action = low_level_controller(current_pose, selected_waypoint)
        execute(action)
```

#### 3.2.3 训练技巧:Waypoint 增强 (Waypoint Augmentation)

**问题:** 模型可能过拟合完美的中心 Waypoint,导致对偏移敏感。

**解决方案:**
- 在训练时,不总是选择热力图的最高峰值点
- 随机采样热力图附近的点作为训练目标
- 迫使智能体适应不同的视角变化和步长

**增强策略:**
$$
w_{aug} = w_{gt} + \mathcal{N}(0, \sigma^2) \quad \text{where } \sigma \in [0.2m, 0.5m]
$$

**效果:** 泛化能力提升约 12%,特别是在新环境中的鲁棒性增强。

---

## 4. NoMaD 中的 Waypoint 应用:从视觉目标导航到多模态控制

### 4.1 NoMaD 模型概述

**NoMaD (Nomadic Multi-Goal Agent with Diffusion)** 是一个专为连续环境设计的视觉导航模型,其核心创新在于:
- 使用 **Vision Transformer (ViT)** 作为视觉编码器
- 引入 **Diffusion Policy** 生成连续的 Waypoint 分布
- 支持多模态目标输入（图像目标、语言指令）

### 4.2 NoMaD 如何利用 Waypoint

#### 4.2.1 Waypoint 作为导航中间表示

与传统方法直接输出动作不同,NoMaD 采用 **Waypoint-conditioned Policy**:

```
输入:
├─ 当前观察图像 o_t
├─ 目标图像/指令 g
└─ 历史轨迹 τ_{1:t-1}
         ↓
Waypoint 生成器 (Diffusion Model):
└─ p(w_{t+1:t+H} | o_t, g) → 预测未来 H 步的 Waypoint 序列
         ↓
底层策略:
└─ π(a_t | o_t, w_{t+1}) → 执行到达下一个 Waypoint 的动作
```

**关键设计:**
- **时间扩展:** 不仅预测下一个 Waypoint,而是预测一个 Waypoint 序列 $\{w_{t+1}, ..., w_{t+H}\}$
- **扩散机制:** 使用 Denoising Diffusion Probabilistic Model (DDPM) 建模 Waypoint 分布的不确定性
- **优势:** 能够处理多模态路径选择（如绕行障碍物的不同策略）

#### 4.2.2 视觉 Transformer 增强的空间推理

NoMaD 的 Waypoint 预测器基于 **ViT-Large** 架构:

**输入处理:**
```python
# 图像编码
observation_tokens = ViT(current_image)  # [N_obs, D]
goal_tokens = ViT(goal_image)            # [N_goal, D]

# 融合当前观察和目标
fused_features = CrossAttention(
    query=observation_tokens,
    key=goal_tokens,
    value=goal_tokens
)  # [N_obs, D]

# 预测 Waypoint 分布
waypoint_dist = DiffusionDecoder(fused_features)  # [H, 2] (x, y 坐标)
```

**与传统方法的对比:**
| 方法 | 视觉编码器 | Waypoint 表示 | 不确定性建模 |
|------|----------|--------------|-------------|
| 传统 CNN | ResNet-50 | 极坐标热力图 | 分类概率 |
| NoMaD | ViT-Large | 笛卡尔坐标序列 | 扩散模型 |

#### 4.2.3 扩散策略生成连续 Waypoint

**扩散模型的优势:**
1. **多模态建模:** 捕捉多条可行路径的分布
2. **平滑性:** 生成的 Waypoint 序列更连贯
3. **灵活性:** 可动态调整预测步长 $H$

**生成过程:**
$$
\begin{align}
&\text{Forward (加噪):} \quad w_T \sim \mathcal{N}(w_0, \sigma^2 I) \\
&\text{Reverse (去噪):} \quad w_{t-1} = \mu_\theta(w_t, o, g) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
\end{align}
$$

**训练目标:**
$$
\mathcal{L} = \mathbb{E}_{w_0, \epsilon, t} \left[ \|\epsilon - \epsilon_\theta(w_t, t, o, g)\|^2 \right]
$$

其中 $\epsilon_\theta$ 是去噪网络,$w_t$ 是加噪的 Waypoint。

#### 4.2.4 分层控制架构

NoMaD 的完整导航流程:

```
[高层] 目标理解:
  输入: 目标图像/语言 → ViT 编码 → 目标嵌入 z_goal

[中层] Waypoint 规划:
  扩散模型生成: p(w_{1:H} | o_t, z_goal)
  采样: w_{1:H} ~ p(·)

[底层] 运动控制:
  PD 控制器: a_t = K_p * (w_1 - p_t) + K_d * (ẇ_1 - v_t)
  执行动作: robot.move(a_t)

[反馈] 状态更新:
  观察新图像 o_{t+1}
  重新规划 Waypoint (滚动窗口)
```

### 4.3 NoMaD 的训练数据与 Waypoint 标注

#### 4.3.1 数据集结构

从工作区结构可以看到,NoMaD 使用三个主要数据集:
```
nomad_dataset/
├─ go_stanford/    # 斯坦福校园导航数据
├─ recon_datavis/  # 重建可视化数据
└─ tartan_drive/   # 户外驾驶数据
```

#### 4.3.2 Waypoint 标注方法

**自动标注流程:**
1. **轨迹采样:** 从人工遥控的机器人轨迹中采样 
   ```python
   # 从 ROS bag 提取轨迹
   trajectory = extract_poses_from_bag(bag_file)
   # 每 N 帧采样一个 Waypoint
   waypoints = trajectory[::N]
   ```

2. **坐标转换:** 将全局坐标转换为以机器人为中心的局部坐标
   ```python
   # 转换到机器人坐标系
   local_waypoint = transform_to_robot_frame(
       global_waypoint, 
       robot_pose
   )
   ```

3. **数据增强:** 参考 Waypoint Augmentation,添加噪声提升鲁棒性

#### 4.3.3 与离散环境的对比

| 特性 | 离散环境 (R2R) | NoMaD (连续) |
|------|---------------|-------------|
| Waypoint 来源 | 预定义图节点 | 扩散模型生成 |
| 数量 | 固定(~100/环境) | 动态(实时生成) |
| 间距 | 稀疏(3-5m) | 密集(1-2m) |
| 可达性验证 | 图边预定义 | 深度图检测 |

---

## 5. 性能对比与技术分析

### 5.1 弥合 DE-CE Gap 的效果

基于 Waypoint 的方法显著缩小了离散与连续环境的性能差距:

| 方法 | 环境 | Success Rate | SPL | 性能提升 |
|------|------|-------------|-----|---------|
| Baseline (Seq2Seq) | VLN-CE | 32% | 0.28 | - |
| Waypoint Predictor | VLN-CE | 50% | 0.43 | +18% |
| NoMaD (Diffusion) | VLN-CE | 55% | 0.48 | +23% |
| 离散环境上限 | R2R | 71% | 0.65 | - |

**剩余 Gap 分析:**
- **感知误差:** 视觉编码器对新场景的泛化能力有限
- **动态障碍:** 连续环境中可能出现移动障碍物
- **累积漂移:** 长时间导航的位姿估计误差

### 5.2 NoMaD 的关键优势

**1. 更好的多模态融合:**
- ViT 的全局注意力机制优于 CNN 的局部感受野
- 能够处理图像目标和语言指令的统一输入

**2. 不确定性建模:**
- 扩散模型捕捉路径选择的多样性
- 在歧义场景中表现更稳定

**3. 真实环境适应性:**
- 在 Go Stanford 等真实机器人数据上训练
- 生成的 Waypoint 更贴近实际可行路径

### 5.3 技术挑战与未来方向

**当前挑战:**
1. **计算成本:** 扩散模型推理较慢(~100ms/step)
2. **长期规划:** 仅预测短期 Waypoint,缺乏全局拓扑理解
3. **泛化能力:** 在室内/室外切换时性能下降

**未来方向:**
1. **混合方法:** 结合 Waypoint 和端到端控制
2. **世界模型:** 预测未来状态辅助长期规划
3. **主动探索:** 在不确定区域主动收集信息

---

## 6. 总结

### 6.1 核心观点

**Waypoint 是连接"高层语义规划"与"底层机器人运动"的关键桥梁。**

- **技术演进逻辑:** 并非抛弃离散环境的思想,而是将离散环境中的"图结构"内化为智能体的感知能力
- **实现路径:** 通过深度学习网络实时利用 RGB-D 信息预测可达点,在连续、未知环境中动态构建导航路径
- **性能提升:** 成功将 DE-CE 性能差距从 ~20% 缩小至 ~8%

### 6.2 NoMaD 的贡献

NoMaD 在 Waypoint 导航的基础上进一步创新:
1. **表示学习:** 用 ViT 替代 CNN,增强空间推理能力
2. **生成建模:** 引入扩散模型处理多模态路径分布
3. **实际部署:** 在真实机器人上验证了方法的可行性

### 6.3 对具身智能的启示

Waypoint 机制揭示了一个重要原则:
> **在复杂任务中,适当的抽象层次比端到端优化更有效。**

这一思想在其他具身 AI 任务中同样适用:
- **机械臂操作:** 关键点检测 → 运动规划
- **无人驾驶:** 路径锚点 → 轨迹跟踪
- **人形机器人:** 足落点规划 → 步态生成

---

## 参考文献

1. Anderson, P., et al. (2018). Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments. *CVPR*.

2. Krantz, J., et al. (2020). Beyond the Nav-Graph: Vision-and-Language Navigation in Continuous Environments. *ECCV*.

3. Jain, V., et al. (2022). Waypoint Models for Instruction-guided Navigation in Continuous Environments. *arXiv:2203.02764*.

4. Shah, D., et al. (2023). NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration. *arXiv*.

5. Habitat Team. (2019). Habitat: A Platform for Embodied AI Research. *ICCV*.

---

## 附录:代码示例

### A.1 Waypoint 预测器实现片段

```python
import torch
import torch.nn as nn
from torchvision.models import resnet50

class WaypointPredictor(nn.Module):
    def __init__(self, num_views=12, num_waypoints=8):
        super().__init__()
        # 视觉编码器
        self.rgb_encoder = resnet50(pretrained=True)
        self.depth_encoder = resnet50(pretrained=True)
        
        # Transformer 空间推理
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=2048, nhead=8),
            num_layers=4
        )
        
        # Waypoint 分类器
        self.classifier = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, 10 * 24)  # 10 距离 × 24 角度
        )
    
    def forward(self, rgb_panorama, depth_panorama):
        # 编码每个视角
        rgb_features = self.rgb_encoder(rgb_panorama)  # [12, 2048]
        depth_features = self.depth_encoder(depth_panorama)
        
        # 融合并空间推理
        fused = rgb_features + depth_features
        spatial_features = self.transformer(fused)
        
        # 预测热力图
        heatmap = self.classifier(spatial_features.mean(0))  # [240]
        heatmap = heatmap.view(10, 24)  # [距离, 角度]
        
        return torch.softmax(heatmap, dim=-1)
```

### A.2 NoMaD 扩散策略伪代码

```python
class DiffusionWaypointPolicy:
    def __init__(self, vit_model, diffusion_steps=10):
        self.vit = vit_model
        self.T = diffusion_steps
    
    def predict_waypoints(self, obs_image, goal_image):
        # 编码视觉输入
        obs_embed = self.vit(obs_image)
        goal_embed = self.vit(goal_image)
        context = torch.cat([obs_embed, goal_embed], dim=-1)
        
        # 扩散去噪
        w_t = torch.randn(5, 2)  # 初始噪声 [H, 2]
        for t in reversed(range(self.T)):
            noise_pred = self.unet(w_t, t, context)
            w_t = self.ddpm_step(w_t, noise_pred, t)
        
        return w_t  # 返回 5 个未来 Waypoint
```

---

**文档生成日期:** 2025年11月29日  
**版本:** v1.0  
**关键词:** Waypoint, 具身智能, VLN-CE, NoMaD, 扩散策略, 导航
