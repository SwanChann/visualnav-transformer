# NoMaD技术详解：实现原理与Action机制

## 论文信息

**标题**: NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration

**作者**: Ajay Sridhar, Dhruv Shah, Catherine Glossop, Sergey Levine

**发表时间**: 2023年10月11日

**arXiv链接**: https://arxiv.org/abs/2310.07896

**项目主页**: https://general-navigation-models.github.io/nomad/

**所属领域**: 机器人学(cs.RO)、计算机视觉(cs.CV)、机器学习(cs.LG)

---

## 一、NoMaD核心思想

### 1.1 研究动机

NoMaD旨在解决机器人在未知环境中导航的两大核心任务：
- **目标导向导航(Goal-directed Navigation)**: 当机器人定位到目标后，需要导航到该目标位置
- **目标无关探索(Goal-agnostic Exploration)**: 在新环境中搜索目标的能力

传统方法通常使用**两个独立的模型**来处理这两个任务，例如通过子目标提案、路径规划或不同的导航策略。NoMaD的创新点在于：**使用单一的统一扩散策略(Unified Diffusion Policy)来同时处理这两个任务**。

### 1.2 核心创新

1. **Goal Masking机制**: 
   - 训练时随机以50%的概率遮蔽目标图像
   - 当目标被遮蔽时，模型学习探索行为
   - 当目标可见时，模型学习目标导向导航

2. **Diffusion Policy**:
   - 使用扩散模型来生成机器人的动作轨迹
   - 相比潜变量模型(latent variable models)具有更好的性能
   - 能够生成多模态的动作分布

3. **大规模Transformer架构**:
   - 使用基于Transformer的视觉编码器
   - 在多个地面机器人数据集上训练
   - 具有更好的泛化能力

---

## 二、NoMaD模型架构

### 2.1 整体架构图

```
输入:
├── 观察图像(Observation Images) [Context: 3帧历史图像]
└── 目标图像(Goal Image) [可能被mask]

↓

[Vision Encoder - NoMaD_ViNT]
├── EfficientNet-B0 (图像特征提取)
├── Multi-Head Attention Layers (时序融合)
└── 输出: obsgoal_cond [256维embedding]

↓                               ↓

[Distance Predictor]           [Diffusion Policy Network]
└── 3层MLP                      ├── Conditional UNet1D
    └── 预测到目标的距离           ├── DDPM Scheduler (10步扩散)
                               └── 输出: Waypoint轨迹 [8个点 × (x,y)]

↓

输出:
├── 预测的距离
└── 动作轨迹(Action Trajectory)
```

### 2.2 核心组件详解

#### 2.2.1 Vision Encoder (NoMaD_ViNT)

```python
class NoMaD_ViNT(nn.Module):
    def __init__(self, 
                 obs_encoding_size=256,
                 context_size=3,
                 mha_num_attention_heads=4,
                 mha_num_attention_layers=4,
                 mha_ff_dim_factor=4):
```

**功能**: 
- 处理观察图像(context: 3帧)和目标图像
- 使用EfficientNet-B0作为backbone提取图像特征
- 通过Multi-Head Attention融合时序信息
- 支持Goal Masking机制

**输入**:
- `obs_img`: 观察图像 [batch, 9, 96, 96] (3帧×3通道)
- `goal_img`: 目标图像 [batch, 3, 96, 96]
- `input_goal_mask`: 遮蔽标记 [batch] (0=不遮蔽, 1=遮蔽)

**输出**:
- `obsgoal_cond`: 融合的特征向量 [batch, 256]

#### 2.2.2 Noise Prediction Network (Conditional UNet1D)

```python
ConditionalUnet1D(
    input_dim=2,              # 每个waypoint的维度(x, y)
    global_cond_dim=256,      # 条件向量的维度
    down_dims=[64, 128, 256], # UNet的下采样维度
    cond_predict_scale=False,
    prediction_type='epsilon'  # 预测噪声
)
```

**功能**: 
- 使用扩散模型生成waypoint轨迹
- 接收视觉特征作为条件
- 通过去噪过程逐步生成清晰的轨迹

**扩散过程**:
1. 初始化: 从高斯噪声开始 `[num_samples, 8, 2]`
2. 迭代去噪: 10步DDPM去噪
3. 输出: 清晰的waypoint轨迹

#### 2.2.3 Distance Predictor (DenseNetwork)

```python
class DenseNetwork(nn.Module):
    def __init__(self, embedding_dim=256):
        self.network = nn.Sequential(
            nn.Linear(256, 64),   # 256 → 64
            nn.ReLU(),
            nn.Linear(64, 16),    # 64 → 16
            nn.ReLU(),
            nn.Linear(16, 1)      # 16 → 1
        )
```

**功能**: 
- 预测当前位置到目标的距离
- 用于选择合适的子目标
- 辅助导航决策

---

## 三、关键技术机制

### 3.1 Goal Masking机制

在训练配置文件`nomad.yaml`中:

```yaml
# mask 
goal_mask_prob: 0.5  # 50%的概率遮蔽目标
```

**工作原理**:

1. **训练阶段**:
   ```python
   # 随机生成mask
   if np.random.rand() < 0.5:
       input_goal_mask = torch.ones(batch_size)  # 遮蔽目标
   else:
       input_goal_mask = torch.zeros(batch_size)  # 显示目标
   ```

2. **推理阶段**:
   - 探索模式: `mask = 1` (遮蔽目标) → 模型输出探索性动作
   - 导航模式: `mask = 0` (显示目标) → 模型输出目标导向动作

3. **效果**:
   - 单一模型同时学习两种行为
   - 无需切换模型或策略
   - 更加统一和高效

### 3.2 扩散策略(Diffusion Policy)

**DDPM调度器配置**:

```python
noise_scheduler = DDPMScheduler(
    num_train_timesteps=10,        # 扩散步数
    beta_schedule='squaredcos_cap_v2',  # beta调度策略
    clip_sample=True,              # 裁剪样本
    prediction_type='epsilon'      # 预测噪声类型
)
```

**推理流程**:

```python
# 1. 初始化噪声
noisy_action = torch.randn((num_samples, 8, 2))  # [样本数, 轨迹长度, xy坐标]

# 2. 设置时间步
noise_scheduler.set_timesteps(10)

# 3. 迭代去噪
for k in noise_scheduler.timesteps:
    # 预测噪声
    noise_pred = model('noise_pred_net',
                      sample=noisy_action,
                      timestep=k,
                      global_cond=obs_cond)
    
    # 去噪一步
    noisy_action = noise_scheduler.step(
        model_output=noise_pred,
        timestep=k,
        sample=noisy_action
    ).prev_sample

# 4. 输出清晰的轨迹
final_trajectory = noisy_action  # [num_samples, 8, 2]
```

**优势**:
- 能够生成多样化的轨迹(通过采样多个样本)
- 比单点预测更鲁棒
- 能够处理多模态分布

### 3.3 负样本挖掘(Negative Mining)

```yaml
datasets:
  go_stanford:
    negative_mining: True  # 启用负样本挖掘
    goals_per_obs: 2       # 每个观察采样2个目标
```

**原理**:
- 来自ViNG论文(Shah et al.)的技术
- 训练时随机采样不相关的目标图像作为负样本
- 帮助模型学习区分相关和不相关的目标
- 提高泛化能力

---

## 四、Action机制深度解析

### 4.1 Action的本质是什么？

**核心答案**: Action **不是**直接的线速度和角速度，而是**相对坐标系下的waypoint轨迹**。

### 4.2 Waypoint表示

从代码`vint_dataset.py`中的实现可以看出:

```python
def _compute_actions(self, traj_data, curr_time, goal_time):
    # 提取未来的位置和航向
    start_index = curr_time
    end_index = curr_time + self.len_traj_pred * self.waypoint_spacing + 1
    yaw = traj_data["yaw"][start_index:end_index:self.waypoint_spacing]
    positions = traj_data["position"][start_index:end_index:self.waypoint_spacing]
    
    # 转换到局部坐标系
    waypoints = to_local_coords(positions, positions[0], yaw[0])
    
    # 动作是相对位移
    if self.learn_angle:
        yaw = yaw[1:] - yaw[0]
        actions = np.concatenate([waypoints[1:], yaw[:, None]], axis=-1)
        # actions shape: [8, 3] (x, y, delta_yaw)
    else:
        actions = waypoints[1:]
        # actions shape: [8, 2] (x, y)
    
    # 归一化
    if self.normalize:
        actions[:, :2] /= self.data_config["metric_waypoint_spacing"]
```

**Waypoint格式**:

1. **基础版本** (默认, `learn_angle=False`):
   ```
   Shape: [8, 2]
   每个waypoint: [dx, dy]
   - dx: 相对于机器人当前位置的x方向位移(前方)
   - dy: 相对于机器人当前位置的y方向位移(左侧)
   ```

2. **包含角度版本** (`learn_angle=True`):
   ```
   Shape: [8, 3] → 经过sin/cos转换后 [8, 4]
   每个waypoint: [dx, dy, sin(delta_yaw), cos(delta_yaw)]
   - dx, dy: 同上
   - delta_yaw: 相对于当前航向的角度变化
   ```

### 4.3 Action配置参数

在`nomad.yaml`中:

```yaml
# action output params
len_traj_pred: 8           # 预测8个waypoint
learn_angle: False         # 不学习角度

# distance bounds for action predictions
action:
  min_dist_cat: 3         # 最小动作距离: 3个waypoint间隔
  max_dist_cat: 20        # 最大动作距离: 20个waypoint间隔

# normalization for the action space
normalize: True           # 归一化动作
```

**参数含义**:

- `len_traj_pred=8`: 每次预测未来8个waypoint的轨迹
- `waypoint_spacing=1`: waypoint之间的时间间隔(数据采样频率)
- `min_action_distance=3`: 只有当目标距离>3时才学习动作
- `max_action_distance=20`: 当目标距离>20时使用特殊处理

### 4.4 从Waypoint到速度控制

虽然模型输出的是waypoint，但在实际部署时需要转换为速度命令。这通过**PD控制器**实现(在`pd_controller.py`中):

```python
def pd_controller(waypoint: np.ndarray) -> Tuple[float]:
    """
    将waypoint转换为线速度和角速度
    
    输入: waypoint = [dx, dy] 或 [dx, dy, hx, hy]
           - dx: 前方距离
           - dy: 侧向距离
           - hx, hy: 目标航向(可选)
    
    输出: (v, w)
           - v: 线速度 (m/s)
           - w: 角速度 (rad/s)
    """
    
    if len(waypoint) == 2:
        dx, dy = waypoint
    else:
        dx, dy, hx, hy = waypoint
    
    # 情况1: 如果有航向信息且位移接近0，使用航向控制
    if len(waypoint) == 4 and np.abs(dx) < EPS and np.abs(dy) < EPS:
        v = 0
        w = clip_angle(np.arctan2(hy, hx)) / DT
    
    # 情况2: 如果只有侧向位移
    elif np.abs(dx) < EPS:
        v = 0
        w = np.sign(dy) * np.pi / (2 * DT)
    
    # 情况3: 一般情况 - PD控制
    else:
        v = dx / DT                    # 线速度 = 前向距离 / 时间步长
        w = np.arctan(dy / dx) / DT    # 角速度 = arctan(侧向/前向) / 时间步长
    
    # 限制速度范围
    v = np.clip(v, 0, MAX_V)         # MAX_V = 0.5 m/s
    w = np.clip(w, -MAX_W, MAX_W)    # MAX_W = 1.0 rad/s
    
    return v, w
```

**控制参数** (在`robot.yaml`中):

```yaml
max_v: 0.5        # 最大线速度 0.5 m/s
max_w: 1.0        # 最大角速度 1.0 rad/s
frame_rate: 10    # 控制频率 10Hz
```

### 4.5 完整的Action流程

```
1. 模型推理
   ↓
   Diffusion Policy → 生成轨迹 [num_samples=8, horizon=8, dim=2]
   
2. 轨迹选择
   ↓
   选择第一个waypoint: chosen_waypoint = trajectory[0]
   或选择距离相关的waypoint: chosen_waypoint = trajectory[waypoint_idx]
   
3. PD控制器转换
   ↓
   waypoint [dx, dy] → (v, w)
   
4. 发布ROS消息
   ↓
   Twist.linear.x = v
   Twist.angular.z = w
   
5. 机器人执行
   ↓
   底层控制器执行速度命令
```

### 4.6 Waypoint选择策略

在`navigate.py`中的实际使用:

```python
# 生成多个样本轨迹(如8个)
naction = generate_trajectory()  # shape: [8, 8, 2]

# 方法1: 选择固定索引的waypoint
chosen_waypoint = naction[args.waypoint]  # 例如选择第2个waypoint

# 方法2: 基于距离选择waypoint
distances = predict_distances()
min_dist_idx = np.argmin(distances)
chosen_waypoint = naction[min_dist_idx][args.waypoint]
```

**选择策略的意义**:
- 较近的waypoint(如第1-2个): 更精确的短期控制
- 较远的waypoint(如第5-8个): 更具前瞻性的规划
- 通过`args.waypoint`参数调整行为特性

---

## 五、训练流程

### 5.1 数据集要求

NoMaD使用多个机器人导航数据集:
- **RECON**: 室内导航数据
- **Go Stanford**: 斯坦福校园导航
- **Cory Hall**: 室内走廊数据
- **TartanDrive**: 越野驾驶数据
- **SACSoN**: 多样化场景

**数据结构**:
```
trajectory_folder/
├── traj_data.pkl          # 包含position, yaw等
└── images/
    ├── 0.jpg
    ├── 1.jpg
    └── ...
```

**traj_data.pkl内容**:
```python
{
    'position': np.array([[x1, y1], [x2, y2], ...]),  # 位置序列
    'yaw': np.array([yaw1, yaw2, ...]),               # 航向序列
}
```

### 5.2 训练配置

```yaml
# 训练设置
batch_size: 256
epochs: 100
lr: 1e-4
optimizer: adamw
scheduler: "cosine"
warmup: True
warmup_epochs: 4

# 模型参数
model_type: nomad
vision_encoder: nomad_vint
encoding_size: 256
obs_encoder: efficientnet-b0

# 扩散模型参数
num_diffusion_iters: 10

# Goal masking
goal_mask_prob: 0.5

# 上下文
context_type: temporal
context_size: 3  # 使用3帧历史图像
```

### 5.3 损失函数

NoMaD使用两个主要损失:

1. **扩散损失(Diffusion Loss)**:
   ```python
   # 预测的噪声与真实噪声的MSE
   diffusion_loss = F.mse_loss(noise_pred, noise_target)
   ```

2. **距离预测损失(Distance Loss)**:
   ```python
   # 预测距离与真实距离的交叉熵
   distance_loss = F.cross_entropy(pred_distance, true_distance)
   ```

### 5.4 数据增强

- 图像resize到 [96, 96]
- 颜色抖动
- 随机裁剪
- 归一化

---

## 六、推理与部署

### 6.1 推理流程

```python
# 1. 加载模型
model = load_model(checkpoint_path)
model.eval()

# 2. 准备输入
obs_images = get_context_images()  # [1, 9, 96, 96] (3帧历史)
goal_image = get_goal_image()      # [1, 3, 96, 96]
mask = torch.zeros(1)              # 0=显示目标(导航模式)

# 3. 编码视觉特征
obsgoal_cond = model('vision_encoder', 
                     obs_img=obs_images,
                     goal_img=goal_image,
                     input_goal_mask=mask)

# 4. 预测距离
distance = model('dist_pred_net', obsgoal_cond=obsgoal_cond)

# 5. 生成轨迹(扩散过程)
noisy_action = torch.randn((num_samples, 8, 2))
noise_scheduler.set_timesteps(10)

for k in noise_scheduler.timesteps:
    noise_pred = model('noise_pred_net',
                      sample=noisy_action,
                      timestep=k,
                      global_cond=obsgoal_cond)
    noisy_action = noise_scheduler.step(
        model_output=noise_pred,
        timestep=k,
        sample=noisy_action
    ).prev_sample

# 6. 选择waypoint
trajectory = noisy_action[0]  # [8, 2]
chosen_waypoint = trajectory[2]  # 选择第3个waypoint

# 7. 转换为速度命令
v, w = pd_controller(chosen_waypoint)

# 8. 发布控制命令
publish_velocity(v, w)
```

### 6.2 实时性能

- **推理时间**: ~0.1秒 (取决于num_samples和扩散步数)
- **控制频率**: 10Hz (每100ms发布一次速度命令)
- **扩散步数**: 10步 (可调整，步数越多越慢但质量更好)

### 6.3 探索与导航模式切换

```python
# 探索模式: 寻找目标
mask = torch.ones(1)  # 遮蔽目标
trajectory = model.predict(obs_images, goal_image, mask)

# 导航模式: 到达已知目标
mask = torch.zeros(1)  # 显示目标
trajectory = model.predict(obs_images, goal_image, mask)
```

---

## 七、NoMaD的优势

### 7.1 相比传统方法

1. **统一策略**: 单一模型处理探索和导航，无需切换
2. **端到端**: 从图像直接到动作，无需手工特征
3. **数据驱动**: 从大规模数据中学习，泛化能力强

### 7.2 相比其他学习方法

1. **扩散模型优势**: 比VAE等潜变量模型生成质量更好
2. **多模态**: 能生成多样化的轨迹，提高鲁棒性
3. **更小的模型**: 使用EfficientNet-B0，比某些SOTA方法更轻量

### 7.3 实验效果

- 在真实机器人平台上验证有效性
- 在未见过的环境中表现良好
- 碰撞率更低
- 相比5种基线方法有显著提升

---

## 八、关键代码位置

### 8.1 模型定义
- 主模型: `train/vint_train/models/nomad/nomad.py`
- 视觉编码器: `train/vint_train/models/nomad/nomad_vint.py`

### 8.2 数据处理
- 数据集: `train/vint_train/data/vint_dataset.py`
- 数据配置: `train/vint_train/data/data_config.yaml`

### 8.3 训练
- 训练脚本: `train/train.py`
- 训练循环: `train/vint_train/training/train_eval_loop.py`
- 配置文件: `train/config/nomad.yaml`

### 8.4 部署
- 导航节点: `deployment/src/navigate.py`
- PD控制器: `deployment/src/pd_controller.py`
- 配置: `deployment/config/robot.yaml`

---

## 九、总结

### 9.1 NoMaD的本质

NoMaD是一个**基于扩散模型的视觉导航系统**，通过Goal Masking机制实现了探索和导航的统一。其核心是：

1. **输入**: 历史观察图像 + 目标图像(可遮蔽)
2. **编码**: Transformer-based视觉编码器提取特征
3. **预测**: 
   - 扩散模型生成waypoint轨迹
   - MLP预测到目标的距离
4. **输出**: 相对坐标系下的waypoint序列
5. **控制**: PD控制器将waypoint转换为速度命令

### 9.2 Action的真相

**Action并非直接的速度命令，而是waypoint轨迹**:
- 模型输出: `[8, 2]` 的waypoint坐标 (相对位置)
- 中间层: PD控制器
- 最终输出: `(v, w)` 线速度和角速度

这种设计的优势:
- **更高层次的抽象**: waypoint比速度更容易学习
- **解耦规划与控制**: 学习负责轨迹规划，PD控制器负责执行
- **更好的泛化性**: 不同机器人可以使用相同的waypoint但不同的控制器

### 9.3 适用场景

- ✅ 室内/室外导航
- ✅ 未知环境探索
- ✅ 视觉目标导航
- ✅ 需要同时探索和导航的任务
- ❌ 需要极高速响应的场景(扩散推理较慢)
- ❌ 没有视觉传感器的场景

### 9.4 未来方向

1. 减少扩散步数以提高实时性
2. 扩展到3D导航
3. 结合语言指令
4. 多机器人协同导航

---

## 参考资料

1. **论文**: NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration (https://arxiv.org/abs/2310.07896)
2. **项目主页**: https://general-navigation-models.github.io/nomad/
3. **相关技术**:
   - DDPM (Denoising Diffusion Probabilistic Models)
   - ViNT (Visual Navigation Transformer)
   - ViNG (Visual Neural Graph)
4. **代码库**: 当前项目 `visualnav-transformer`

---

**文档创建时间**: 2025年12月1日

**作者**: GitHub Copilot

**版本**: 1.0
