# NoMaD中Diffusion Policy的训练与推理详解

## 概述

本文档详细解析NoMaD框架中Diffusion Policy的完整实现，包括训练流程、推理机制、关键代码位置及核心算法原理。通过逐文件、逐函数的代码解读，帮助读者深入理解扩散策略在视觉导航中的应用。

---

## 一、代码文件结构总览

### 1.1 核心文件位置

```
visualnav-transformer/
├── train/
│   ├── train.py                              # 主训练入口
│   ├── config/
│   │   └── nomad.yaml                        # NoMaD配置文件
│   └── vint_train/
│       ├── models/
│       │   └── nomad/
│       │       ├── nomad.py                  # NoMaD主模型定义
│       │       └── nomad_vint.py             # 视觉编码器
│       ├── training/
│       │   ├── train_eval_loop.py            # 训练评估循环
│       │   └── train_utils.py                # 训练工具函数(核心!)
│       └── data/
│           ├── vint_dataset.py               # 数据集定义
│           └── data_config.yaml              # 数据配置
│
├── deployment/
│   └── src/
│       ├── navigate.py                       # 导航推理主程序
│       ├── pd_controller.py                  # PD控制器
│       └── utils.py                          # 模型加载工具
│
└── diffusion_policy/                         # 扩散策略库(外部依赖)
    └── model/
        └── diffusion/
            └── conditional_unet1d.py         # 条件UNet1D噪声预测网络
```

### 1.2 文件职责划分

| 文件 | 职责 | 关键函数/类 |
|------|------|-------------|
| `train.py` | 训练入口，模型构建 | `main()` |
| `nomad.py` | NoMaD模型封装 | `NoMaD`, `DenseNetwork` |
| `nomad_vint.py` | 视觉编码器 | `NoMaD_ViNT` |
| `train_utils.py` | **扩散训练核心** | `train_nomad()`, `model_output()` |
| `train_eval_loop.py` | 训练循环调度 | `train_eval_loop_nomad()` |
| `navigate.py` | 实时推理 | `main()` |
| `conditional_unet1d.py` | 噪声预测网络 | `ConditionalUnet1D` |

---

## 二、模型架构详解

### 2.1 NoMaD模型结构 (`train/vint_train/models/nomad/nomad.py`)

```python
class NoMaD(nn.Module):
    """
    NoMaD主模型：封装三个子网络
    """
    def __init__(self, vision_encoder,    # 视觉编码器 (NoMaD_ViNT)
                       noise_pred_net,     # 噪声预测网络 (ConditionalUnet1D)
                       dist_pred_net):     # 距离预测网络 (DenseNetwork)
        super(NoMaD, self).__init__()
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.dist_pred_net = dist_pred_net
    
    def forward(self, func_name, **kwargs):
        """
        统一的前向接口，通过func_name选择调用哪个子网络
        
        使用方式:
            model("vision_encoder", obs_img=..., goal_img=..., input_goal_mask=...)
            model("noise_pred_net", sample=..., timestep=..., global_cond=...)
            model("dist_pred_net", obsgoal_cond=...)
        """
        if func_name == "vision_encoder":
            output = self.vision_encoder(kwargs["obs_img"], 
                                         kwargs["goal_img"], 
                                         input_goal_mask=kwargs["input_goal_mask"])
        elif func_name == "noise_pred_net":
            output = self.noise_pred_net(sample=kwargs["sample"], 
                                         timestep=kwargs["timestep"], 
                                         global_cond=kwargs["global_cond"])
        elif func_name == "dist_pred_net":
            output = self.dist_pred_net(kwargs["obsgoal_cond"])
        else:
            raise NotImplementedError
        return output
```

**关键设计**：使用字符串调度的方式调用不同子网络，便于在训练和推理中灵活组合。

### 2.2 距离预测网络 (`DenseNetwork`)

```python
class DenseNetwork(nn.Module):
    """
    简单的3层MLP，用于预测到目标的距离
    
    输入: obsgoal_cond [B, 256]
    输出: distance [B, 1]
    """
    def __init__(self, embedding_dim):
        super(DenseNetwork, self).__init__()
        self.embedding_dim = embedding_dim  # 256
        self.network = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim//4),  # 256 → 64
            nn.ReLU(),
            nn.Linear(self.embedding_dim//4, self.embedding_dim//16),  # 64 → 16
            nn.ReLU(),
            nn.Linear(self.embedding_dim//16, 1)  # 16 → 1
        )
    
    def forward(self, x):
        x = x.reshape((-1, self.embedding_dim))
        output = self.network(x)
        return output
```

### 2.3 噪声预测网络 (`ConditionalUnet1D`)

来源：`diffusion_policy/model/diffusion/conditional_unet1d.py`

```python
# 在train.py中的初始化
noise_pred_net = ConditionalUnet1D(
    input_dim=2,                          # waypoint维度 (x, y)
    global_cond_dim=config["encoding_size"],  # 条件向量维度 (256)
    down_dims=config["down_dims"],        # 下采样维度 [64, 128, 256]
    cond_predict_scale=config["cond_predict_scale"],  # False
)
```

**架构特点**：
- **1D UNet结构**：适合处理时序waypoint数据
- **全局条件注入**：将视觉条件融入每一层
- **时间步嵌入**：对扩散时间步进行编码

---

## 三、训练流程详解

### 3.1 训练入口 (`train/train.py`)

#### 3.1.1 模型构建

```python
# 位置: train.py 第170-220行

elif config["model_type"] == "nomad":
    # 1. 构建视觉编码器
    if config["vision_encoder"] == "nomad_vint":
        vision_encoder = NoMaD_ViNT(
            obs_encoding_size=config["encoding_size"],      # 256
            context_size=config["context_size"],            # 3
            mha_num_attention_heads=config["mha_num_attention_heads"],  # 4
            mha_num_attention_layers=config["mha_num_attention_layers"], # 4
            mha_ff_dim_factor=config["mha_ff_dim_factor"],  # 4
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)  # BatchNorm → GroupNorm
    
    # 2. 构建噪声预测网络 (Diffusion核心)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,                              # waypoint (x, y)
        global_cond_dim=config["encoding_size"],  # 256
        down_dims=config["down_dims"],            # [64, 128, 256]
        cond_predict_scale=config["cond_predict_scale"],  # False
    )
    
    # 3. 构建距离预测网络
    dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
    
    # 4. 组装NoMaD模型
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )

    # 5. 创建DDPM噪声调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],  # 10
        beta_schedule='squaredcos_cap_v2',  # cosine schedule
        clip_sample=True,
        prediction_type='epsilon'  # 预测噪声
    )
```

#### 3.1.2 调用训练循环

```python
# 位置: train.py 第350-380行

train_eval_loop_nomad(
    train_model=config["train"],
    model=model,
    optimizer=optimizer,
    lr_scheduler=scheduler,
    noise_scheduler=noise_scheduler,
    train_loader=train_loader,
    test_dataloaders=test_dataloaders,
    transform=transform,
    goal_mask_prob=config["goal_mask_prob"],  # 0.5
    epochs=config["epochs"],
    device=device,
    project_folder=project_folder,
    print_log_freq=config.get("print_log_freq", 100),
    wandb_log_freq=config.get("wandb_log_freq", 10),
    image_log_freq=config.get("image_log_freq", 1000),
    num_images_log=config.get("num_images_log", 8),
    current_epoch=current_epoch,
    alpha=float(config["alpha"]),  # 1e-4
    use_wandb=config["use_wandb"],
)
```

### 3.2 训练循环 (`train/vint_train/training/train_eval_loop.py`)

```python
# 位置: train_eval_loop.py 第147-270行

def train_eval_loop_nomad(
    train_model: bool,
    model: nn.Module,
    optimizer: Adam, 
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    noise_scheduler: DDPMScheduler,
    train_loader: DataLoader,
    test_dataloaders: Dict[str, DataLoader],
    transform: transforms,
    goal_mask_prob: float,
    epochs: int,
    device: torch.device,
    project_folder: str,
    ...
):
    """NoMaD的训练评估主循环"""
    
    # 创建EMA模型 (指数移动平均，用于推理)
    ema_model = EMAModel(model=model, power=0.75)
    
    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}")
            
            # 核心训练函数
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                goal_mask_prob=goal_mask_prob,
                project_folder=project_folder,
                epoch=epoch,
                ...
            )
            lr_scheduler.step()
        
        # 保存EMA模型 (用于推理)
        torch.save(ema_model.averaged_model.state_dict(), f"{project_folder}/ema_{epoch}.pth")
        
        # 保存普通模型
        torch.save(model.state_dict(), f"{project_folder}/{epoch}.pth")
        
        # 评估
        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    ...
                )
```

### 3.3 核心训练函数 (`train/vint_train/training/train_utils.py`)

这是**Diffusion Policy训练的核心代码**，位于`train_utils.py`第520-700行：

```python
def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,
    optimizer: Adam,
    dataloader: DataLoader,
    transform: transforms,
    device: torch.device,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,
    project_folder: str,
    epoch: int,
    alpha: float = 1e-4,  # 距离损失权重
    ...
):
    """
    NoMaD的单epoch训练
    
    核心流程:
    1. 获取观察图像和目标图像
    2. 随机生成goal_mask (50%概率遮蔽目标)
    3. 编码视觉特征
    4. 对干净动作添加噪声
    5. 训练噪声预测网络
    6. 计算并反传损失
    """
    goal_mask_prob = torch.clip(torch.tensor(goal_mask_prob), 0, 1)
    model.train()
    
    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            # ========== 1. 数据准备 ==========
            (obs_image,      # [B, 12, H, W] (context_size=3 + 1 = 4帧 × 3通道)
             goal_image,     # [B, 3, H, W]
             actions,        # [B, 8, 2] ground truth waypoints
             distance,       # [B] 到目标的距离
             goal_pos,       # [B, 2] 目标位置
             dataset_idx,    # [B] 数据集索引
             action_mask,    # [B] 动作有效掩码
            ) = data
            
            # 图像预处理
            obs_images = torch.split(obs_image, 3, dim=1)  # 分成4帧
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)
            batch_goal_images = transform(goal_image).to(device)
            
            B = actions.shape[0]  # batch size
            
            # ========== 2. Goal Masking ==========
            # 以goal_mask_prob(50%)的概率遮蔽目标
            goal_mask = (torch.rand((B,)) < goal_mask_prob).long().to(device)
            # goal_mask[i] = 1 表示遮蔽第i个样本的目标
            # goal_mask[i] = 0 表示不遮蔽
            
            # ========== 3. 视觉特征编码 ==========
            obsgoal_cond = model("vision_encoder", 
                                 obs_img=batch_obs_images, 
                                 goal_img=batch_goal_images, 
                                 input_goal_mask=goal_mask)
            # obsgoal_cond: [B, 256] 视觉条件向量
            
            # ========== 4. 动作归一化 ==========
            # 转换为增量表示并归一化到[-1, 1]
            deltas = get_delta(actions)  # 计算相邻waypoint的增量
            ndeltas = normalize_data(deltas, ACTION_STATS)  # 归一化
            naction = from_numpy(ndeltas).to(device)  # [B, 8, 2]
            
            # ========== 5. 距离预测损失 ==========
            distance = distance.float().to(device)
            dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            dist_loss = nn.functional.mse_loss(dist_pred.squeeze(-1), distance)
            # 只对未遮蔽目标的样本计算距离损失
            dist_loss = (dist_loss * (1 - goal_mask.float())).mean() / (1e-2 + (1 - goal_mask.float()).mean())
            
            # ========== 6. 扩散训练核心 ==========
            # 6.1 采样随机噪声
            noise = torch.randn(naction.shape, device=device)  # [B, 8, 2]
            
            # 6.2 采样随机时间步
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,  # [0, 10)
                (B,), device=device
            ).long()
            
            # 6.3 前向扩散：添加噪声到干净动作
            noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
            # noisy_action = sqrt(α_t) * naction + sqrt(1-α_t) * noise
            
            # 6.4 预测噪声
            noise_pred = model("noise_pred_net", 
                              sample=noisy_action,      # 带噪声的动作
                              timestep=timesteps,       # 时间步
                              global_cond=obsgoal_cond) # 视觉条件
            # noise_pred: [B, 8, 2] 预测的噪声
            
            # ========== 7. 计算扩散损失 ==========
            def action_reduce(unreduced_loss: torch.Tensor):
                """考虑action_mask的损失归约"""
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return (unreduced_loss * action_mask).mean() / (action_mask.mean() + 1e-2)
            
            # L2损失：预测噪声 vs 真实噪声
            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            
            # ========== 8. 总损失 & 优化 ==========
            # 总损失 = α * 距离损失 + (1-α) * 扩散损失
            loss = alpha * dist_loss + (1 - alpha) * diffusion_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ========== 9. 更新EMA模型 ==========
            ema_model.step(model)
            
            # 日志记录
            wandb.log({"total_loss": loss.item()})
            wandb.log({"dist_loss": dist_loss.item()})
            wandb.log({"diffusion_loss": diffusion_loss.item()})
```

### 3.4 训练损失函数详解

#### 3.4.1 扩散损失 (Diffusion Loss)

```python
# 核心思想：训练网络预测添加到数据中的噪声

# 1. 前向扩散过程
noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)
# 公式: x_t = sqrt(α_t) * x_0 + sqrt(1-α_t) * ε

# 2. 噪声预测
noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

# 3. L2损失
diffusion_loss = F.mse_loss(noise_pred, noise, reduction="none")
# 公式: L = ||ε - ε_θ(x_t, t, c)||²
```

#### 3.4.2 距离损失 (Distance Loss)

```python
# 预测当前位置到目标的距离
dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
dist_loss = F.mse_loss(dist_pred.squeeze(-1), distance)

# 只对未遮蔽目标的样本计算
dist_loss = (dist_loss * (1 - goal_mask.float())).mean()
```

#### 3.4.3 总损失

```python
# α = 1e-4 (距离损失权重很小)
loss = alpha * dist_loss + (1 - alpha) * diffusion_loss
# 实际上: loss ≈ diffusion_loss
```

### 3.5 动作归一化处理

```python
# 位置: train_utils.py 第920-960行

# 动作统计信息 (来自data_config.yaml)
ACTION_STATS = {
    'min': np.array([-1.0, -1.0]),
    'max': np.array([1.0, 1.0])
}

def get_delta(actions):
    """将绝对位置转换为相对增量"""
    # actions: [B, 8, 2] 绝对waypoint位置
    # 返回: [B, 8, 2] 相邻waypoint的增量
    ex_actions = np.concatenate([np.zeros((actions.shape[0], 1, actions.shape[-1])), actions], axis=1)
    delta = ex_actions[:, 1:] - ex_actions[:, :-1]
    return delta

def normalize_data(data, stats):
    """归一化到[-1, 1]"""
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])  # → [0, 1]
    ndata = ndata * 2 - 1  # → [-1, 1]
    return ndata

def unnormalize_data(ndata, stats):
    """反归一化"""
    ndata = (ndata + 1) / 2  # → [0, 1]
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data

def get_action(diffusion_output, action_stats=ACTION_STATS):
    """从扩散输出恢复动作"""
    ndeltas = diffusion_output  # [B, 8, 2] 归一化的增量
    ndeltas = to_numpy(ndeltas)
    ndeltas = unnormalize_data(ndeltas, action_stats)  # 反归一化
    actions = np.cumsum(ndeltas, axis=1)  # 增量累加得到绝对位置
    return actions
```

---

## 四、推理流程详解

### 4.1 模型加载 (`deployment/src/utils.py`)

```python
def load_model(model_path: str, config: dict, device: torch.device) -> nn.Module:
    """
    加载NoMaD模型权重
    
    注意: 部署时加载的是EMA模型权重 (ema_latest.pth)
    """
    if config["model_type"] == "nomad":
        # 构建模型结构 (与训练时相同)
        vision_encoder = NoMaD_ViNT(...)
        noise_pred_net = ConditionalUnet1D(
            input_dim=2,
            global_cond_dim=config["encoding_size"],
            down_dims=config["down_dims"],
            cond_predict_scale=config["cond_predict_scale"],
        )
        dist_pred_network = DenseNetwork(embedding_dim=config["encoding_size"])
        
        model = NoMaD(
            vision_encoder=vision_encoder,
            noise_pred_net=noise_pred_net,
            dist_pred_net=dist_pred_network,
        )
        
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint, strict=False)
    
    model.to(device)
    return model
```

### 4.2 实时推理 (`deployment/src/navigate.py`)

```python
# 位置: navigate.py 第120-200行

def main(args):
    # ========== 1. 加载模型 ==========
    model = load_model(ckpth_path, model_params, device)
    model.eval()
    
    # ========== 2. 创建噪声调度器 ==========
    if model_params["model_type"] == "nomad":
        num_diffusion_iters = model_params["num_diffusion_iters"]  # 10
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon'
        )
    
    # ========== 3. 主导航循环 ==========
    while not rospy.is_shutdown():
        if len(context_queue) > model_params["context_size"]:
            
            # 3.1 图像预处理
            obs_images = transform_images(context_queue, model_params["image_size"])
            obs_images = torch.cat(torch.split(obs_images, 3, dim=1), dim=1)
            obs_images = obs_images.to(device)
            
            # 3.2 设置goal mask (导航模式 = 0)
            mask = torch.zeros(1).long().to(device)
            
            # 3.3 获取子目标图像
            goal_image = transform_images(topomap[closest_node], ...)
            
            # 3.4 编码视觉特征
            obsgoal_cond = model('vision_encoder', 
                                obs_img=obs_images.repeat(len(goal_image), 1, 1, 1),
                                goal_img=goal_image, 
                                input_goal_mask=mask.repeat(len(goal_image)))
            
            # 3.5 预测距离，选择最近节点
            dists = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
            min_idx = np.argmin(dists.flatten())
            obs_cond = obsgoal_cond[min_idx].unsqueeze(0)
            
            # ========== 4. 扩散采样 (核心!) ==========
            with torch.no_grad():
                # 4.1 复制条件向量
                if len(obs_cond.shape) == 2:
                    obs_cond = obs_cond.repeat(args.num_samples, 1)  # [8, 256]
                
                # 4.2 从高斯噪声初始化
                noisy_action = torch.randn(
                    (args.num_samples, model_params["len_traj_pred"], 2), 
                    device=device
                )
                # noisy_action: [8, 8, 2]  (8个样本, 8个waypoint, 2维坐标)
                naction = noisy_action
                
                # 4.3 设置扩散时间步
                noise_scheduler.set_timesteps(num_diffusion_iters)  # 10步
                
                # 4.4 迭代去噪 (逆扩散)
                for k in noise_scheduler.timesteps:  # [9, 8, 7, ..., 0]
                    # 预测噪声
                    noise_pred = model(
                        'noise_pred_net',
                        sample=naction,        # 当前带噪声的动作
                        timestep=k,            # 当前时间步
                        global_cond=obs_cond   # 视觉条件
                    )
                    
                    # 去噪一步
                    naction = noise_scheduler.step(
                        model_output=noise_pred,
                        timestep=k,
                        sample=naction
                    ).prev_sample
                # naction: [8, 8, 2] 去噪后的清晰轨迹
            
            # ========== 5. 选择waypoint ==========
            naction = to_numpy(get_action(naction))  # 反归一化
            # naction: [8, 8, 2]
            
            naction = naction[0]  # 选择第一个样本 [8, 2]
            chosen_waypoint = naction[args.waypoint]  # 选择第2个waypoint [2,]
            
            # ========== 6. 发布控制命令 ==========
            waypoint_msg = Float32MultiArray()
            waypoint_msg.data = chosen_waypoint
            waypoint_pub.publish(waypoint_msg)
```

### 4.3 扩散去噪过程详解

```python
# DDPM逆扩散公式:
# x_{t-1} = (1/sqrt(α_t)) * (x_t - (1-α_t)/sqrt(1-ᾱ_t) * ε_θ(x_t, t)) + σ_t * z

# 在代码中通过noise_scheduler.step()实现:
for k in noise_scheduler.timesteps:  # [9, 8, ..., 0]
    # 1. 预测当前时间步的噪声
    noise_pred = model('noise_pred_net', sample=naction, timestep=k, global_cond=obs_cond)
    
    # 2. 根据预测的噪声去噪一步
    naction = noise_scheduler.step(
        model_output=noise_pred,  # 预测的噪声
        timestep=k,               # 当前时间步
        sample=naction            # 当前样本
    ).prev_sample                 # 返回去噪后的样本
```

### 4.4 推理时的Goal Masking

```python
# 导航模式: mask = 0 (不遮蔽目标)
mask = torch.zeros(1).long().to(device)
obsgoal_cond = model('vision_encoder', ..., input_goal_mask=mask)
# 此时Attention可以看到目标图像，输出导航行为

# 探索模式: mask = 1 (遮蔽目标)
mask = torch.ones(1).long().to(device)
obs_cond = model('vision_encoder', ..., input_goal_mask=mask)
# 此时Attention看不到目标图像，输出探索行为
```

---

## 五、训练与推理的对应关系

### 5.1 数据流对比

| 阶段 | 训练 | 推理 |
|------|------|------|
| **输入图像** | DataLoader批量加载 | ROS实时订阅 |
| **上下文** | 从数据集采样4帧 | context_queue维护4帧 |
| **目标图像** | 数据集中的goal_image | topomap中的节点图像 |
| **Goal Mask** | 随机50%遮蔽 | 0(导航)或1(探索) |
| **扩散过程** | 前向加噪 → 预测噪声 → L2损失 | 初始化噪声 → 迭代去噪 → 输出轨迹 |
| **采样数** | 无(直接比较噪声) | num_samples=8个轨迹 |
| **输出** | 损失值(反传) | waypoint坐标(控制) |

### 5.2 关键差异

```python
# ========== 训练时 ==========
# 1. 前向扩散: 给干净动作添加噪声
noise = torch.randn_like(naction)
timesteps = torch.randint(0, 10, (B,))
noisy_action = noise_scheduler.add_noise(naction, noise, timesteps)

# 2. 预测噪声
noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, ...)

# 3. 计算损失
loss = F.mse_loss(noise_pred, noise)

# ========== 推理时 ==========
# 1. 从纯噪声开始
naction = torch.randn((num_samples, 8, 2))

# 2. 迭代去噪 (10步)
for k in [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]:
    noise_pred = model("noise_pred_net", sample=naction, timestep=k, ...)
    naction = noise_scheduler.step(noise_pred, k, naction).prev_sample

# 3. 输出清晰轨迹
final_trajectory = naction
```

---

## 六、配置文件详解 (`train/config/nomad.yaml`)

```yaml
# ========== 项目信息 ==========
project_name: nomad
run_name: nomad

# ========== 训练设置 ==========
use_wandb: True
train: True
batch_size: 256
epochs: 100
gpu_ids: [0]
num_workers: 12
lr: 1e-4
optimizer: adamw

# ========== 学习率调度 ==========
scheduler: "cosine"
warmup: True 
warmup_epochs: 4

# ========== 模型参数 ==========
model_type: nomad
vision_encoder: nomad_vint
encoding_size: 256           # 视觉特征维度
obs_encoder: efficientnet-b0 # 图像backbone

# Transformer参数
mha_num_attention_heads: 4
mha_num_attention_layers: 4
mha_ff_dim_factor: 4

# UNet参数
down_dims: [64, 128, 256]    # 下采样通道数
cond_predict_scale: False

# ========== 扩散模型参数 ==========
num_diffusion_iters: 10      # 扩散步数 (关键!)

# ========== Goal Masking ==========
goal_mask_prob: 0.5          # 遮蔽概率

# ========== 动作参数 ==========
normalize: True              # 动作归一化
len_traj_pred: 8             # 预测8个waypoint
learn_angle: False           # 不学习角度

# ========== 上下文 ==========
context_type: temporal
context_size: 3              # 使用3帧历史

# ========== 距离范围 ==========
distance:
  min_dist_cat: 0
  max_dist_cat: 20
action:
  min_dist_cat: 3            # 只有距离>3时才学习动作
  max_dist_cat: 20

# ========== 损失权重 ==========
alpha: 1e-4                  # 距离损失权重 (很小)

# ========== 数据集 ==========
image_size: [96, 96]
datasets:
  go_stanford:
    data_folder: /path/to/go_stanford
    train: /path/to/train/
    test: /path/to/test/
    negative_mining: True
    goals_per_obs: 2
```

---

## 七、关键代码函数索引

### 7.1 训练相关

| 函数 | 文件 | 行号 | 功能 |
|------|------|------|------|
| `train_nomad()` | train_utils.py | 520-700 | 单epoch训练核心 |
| `train_eval_loop_nomad()` | train_eval_loop.py | 147-270 | 训练循环调度 |
| `evaluate_nomad()` | train_utils.py | 720-900 | 评估函数 |
| `_compute_losses_nomad()` | train_utils.py | 448-520 | 损失计算 |
| `get_delta()` | train_utils.py | 940-945 | 动作增量计算 |
| `normalize_data()` | train_utils.py | 925-930 | 动作归一化 |
| `unnormalize_data()` | train_utils.py | 932-935 | 动作反归一化 |
| `get_action()` | train_utils.py | 947-955 | 恢复动作 |
| `model_output()` | train_utils.py | 958-1035 | 推理输出 |

### 7.2 模型相关

| 类/函数 | 文件 | 功能 |
|---------|------|------|
| `NoMaD` | nomad.py | 主模型封装 |
| `DenseNetwork` | nomad.py | 距离预测MLP |
| `NoMaD_ViNT` | nomad_vint.py | 视觉编码器 |
| `ConditionalUnet1D` | conditional_unet1d.py | 噪声预测网络 |

### 7.3 推理相关

| 函数 | 文件 | 功能 |
|------|------|------|
| `main()` | navigate.py | 导航主程序 |
| `load_model()` | utils.py | 模型加载 |
| `transform_images()` | utils.py | 图像预处理 |
| `pd_controller()` | pd_controller.py | waypoint→速度 |

---

## 八、DDPM调度器详解

### 8.1 DDPMScheduler配置

```python
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

noise_scheduler = DDPMScheduler(
    num_train_timesteps=10,          # 扩散步数
    beta_schedule='squaredcos_cap_v2', # β调度策略
    clip_sample=True,                # 裁剪样本到[-1,1]
    prediction_type='epsilon'        # 预测噪声类型
)
```

### 8.2 β调度策略

```python
# 'squaredcos_cap_v2': 改进的余弦调度
# 相比线性调度，在噪声较小时更平滑

# β_t 从小到大，定义噪声添加量
# α_t = 1 - β_t
# ᾱ_t = ∏(α_i) 累积乘积
```

### 8.3 关键方法

```python
# 前向扩散 (训练时)
noisy_sample = noise_scheduler.add_noise(original_sample, noise, timesteps)

# 逆扩散一步 (推理时)
noise_scheduler.set_timesteps(num_inference_steps)
for t in noise_scheduler.timesteps:
    output = noise_scheduler.step(noise_pred, t, sample)
    sample = output.prev_sample
```

---

## 九、EMA模型详解

### 9.1 什么是EMA？

**指数移动平均(Exponential Moving Average)**：
- 训练时维护模型参数的滑动平均
- 推理时使用EMA模型，更稳定

### 9.2 代码实现

```python
from diffusers.training_utils import EMAModel

# 创建EMA模型
ema_model = EMAModel(model=model, power=0.75)

# 每步训练后更新
ema_model.step(model)

# 保存EMA权重 (用于部署)
torch.save(ema_model.averaged_model.state_dict(), "ema_latest.pth")

# 推理时加载EMA权重
model.load_state_dict(torch.load("ema_latest.pth"))
```

### 9.3 更新公式

```python
# θ_ema = decay * θ_ema + (1 - decay) * θ
# decay = 1 - (1 / (step + 1))^power
```

---

## 十、可视化与调试

### 10.1 训练可视化

```python
# 位置: train_utils.py 第1040-1175行

def visualize_diffusion_action_distribution(
    ema_model,
    noise_scheduler,
    batch_obs_images,
    batch_goal_images,
    ...
):
    """
    可视化扩散模型生成的动作分布
    
    输出:
    - 30个采样轨迹的可视化
    - UC (Unconditional/探索) vs GC (Goal-Conditioned/导航) 对比
    - 与Ground Truth的比较
    """
    # 采样30条轨迹
    model_output_dict = model_output(ema_model, ..., num_samples=30)
    
    # 绘制
    for i in range(num_images_log):
        fig, ax = plt.subplots(1, 3)
        # ax[0]: 轨迹分布图
        # ax[1]: 观察图像
        # ax[2]: 目标图像
        plt.savefig(f"sample_{i}.png")
```

### 10.2 调试技巧

```python
# 1. 检查噪声预测
print(f"noise_pred shape: {noise_pred.shape}")  # 应为 [B, 8, 2]
print(f"noise_pred range: [{noise_pred.min():.3f}, {noise_pred.max():.3f}]")

# 2. 检查动作范围
print(f"naction range: [{naction.min():.3f}, {naction.max():.3f}]")
# 应在 [-1, 1] 左右

# 3. 检查损失值
print(f"diffusion_loss: {diffusion_loss.item():.6f}")
print(f"dist_loss: {dist_loss.item():.6f}")

# 4. 可视化去噪过程
for k in noise_scheduler.timesteps:
    noise_pred = model("noise_pred_net", ...)
    naction = noise_scheduler.step(...).prev_sample
    print(f"Step {k}: action std = {naction.std():.4f}")
```

---

## 十一、总结

### 11.1 核心要点

1. **训练目标**：预测添加到数据中的噪声 ε_θ(x_t, t, c)
2. **训练过程**：前向加噪 → 预测噪声 → L2损失
3. **推理过程**：纯噪声 → 10步迭代去噪 → 清晰轨迹
4. **Goal Masking**：50%概率遮蔽，统一探索和导航
5. **EMA模型**：部署时使用，更稳定

### 11.2 关键超参数

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_diffusion_iters` | 10 | 扩散步数 |
| `goal_mask_prob` | 0.5 | 目标遮蔽概率 |
| `encoding_size` | 256 | 条件向量维度 |
| `len_traj_pred` | 8 | 预测waypoint数 |
| `alpha` | 1e-4 | 距离损失权重 |
| `num_samples` | 8 | 推理时采样数 |

### 11.3 代码阅读顺序建议

1. `train/config/nomad.yaml` - 理解配置
2. `train/train.py` - 理解模型构建
3. `train/vint_train/models/nomad/nomad.py` - 理解模型结构
4. `train/vint_train/training/train_utils.py` 的 `train_nomad()` - **核心训练逻辑**
5. `deployment/src/navigate.py` - 理解推理流程
6. `train/vint_train/training/train_utils.py` 的 `model_output()` - 理解采样过程

---

**文档创建时间**: 2026年1月9日

**版本**: 1.0

**关键词**: Diffusion Policy, DDPM, NoMaD, 训练, 推理, 噪声预测
