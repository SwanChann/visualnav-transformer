# NoMaD 项目完整复现指南

> **重要说明**: 本指南面向**Linux (Ubuntu)系统**。所有命令均为Linux Bash命令。
> 如果你在Windows系统通过SSH/VS Code Remote-SSH连接Ubuntu服务器,请在远程Linux终端执行这些命令。

## 目录
1. [项目简介](#项目简介)
2. [系统要求](#系统要求)
3. [环境准备](#环境准备)
4. [数据准备](#数据准备)
5. [模型训练](#模型训练)
6. [机器人部署](#机器人部署)
7. [常见问题](#常见问题)
8. [验证清单](#验证清单)

---

## 项目简介

**NoMaD (Navigation with Goal Masked Diffusion)** 是一个用于机器人导航和探索的目标掩码扩散策略模型。它是通用导航模型（GNM）系列的最新成员，能够：

- 基于视觉的端到端导航
- 支持探索模式（无需预定义目标）
- 跨机器人平台零样本迁移
- 适应动态障碍物和光照变化

**论文**: [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html) (2023年10月)

**核心特性**:
- 扩散模型架构（10次迭代）
- 目标掩码概率：50%
- 预测轨迹长度：8个时间步
- 图像尺寸：96x96

---

## 系统要求

### 硬件要求

#### 训练环境
- **操作系统**: Ubuntu 18.04 / 20.04 / 22.04 (Linux)
- **GPU**: NVIDIA GPU (显存 ≥ 8GB, 推荐 ≥ 16GB)
- **CUDA**: 10.0+
- **内存**: ≥ 32GB RAM
- **存储**: ≥ 100GB (用于数据集和模型)

#### 部署环境（可选）
- **机器人**: LoCoBot / TurtleBot2 / Jackal / 其他ROS兼容机器人
- **计算平台**: NVIDIA Jetson Orin Nano 或类似设备（Linux）
- **摄像头**: 广角RGB摄像头（如ELP鱼眼摄像头）
- **控制器**: 支持Linux的游戏手柄或键盘

### 软件要求
- Python 3.8.5
- PyTorch 1.x (CUDA支持)
- ROS Noetic (仅部署需要)
- Conda / Miniconda

**注意**: 本指南所有命令均为Linux系统命令。如果你在Windows系统通过远程连接(SSH/Remote-SSH)访问Ubuntu服务器,请在Ubuntu终端执行这些命令。

---

## 环境准备

### 步骤1：克隆项目仓库

```bash
# 克隆项目
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer

# 检查项目结构
ls -la
```

预期目录结构：
```
visualnav-transformer/
├── train/                  # 训练相关代码
├── deployment/            # 部署相关代码
├── LICENSE
└── README.md
```

### 步骤2：创建训练环境

```bash
# 创建conda环境（使用训练配置）
cd train
conda env create -f train_environment.yml

# 激活环境
conda activate nomad_train

# 验证环境
python --version  # 应显示 Python 3.8.5
python -c "import torch; print(torch.__version__)"
python -c "import torch; print(torch.cuda.is_available())"  # 应返回 True
```

### 步骤3：安装项目依赖

```bash
# 确保在 visualnav-transformer/ 根目录

# 安装 vint_train 包
pip install -e train/

# 验证安装
python -c "import vint_train; print('vint_train installed successfully')"
```

### 步骤4：安装 Diffusion Policy 依赖

```bash
# 重要：先安装兼容版本的 huggingface_hub
pip install huggingface_hub==0.10.1

# 然后安装 diffusers
pip install diffusers==0.11.1

# 克隆 diffusion_policy 仓库
cd ..  # 返回上级目录或选择合适的位置
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy

# 安装
pip install -e .

# 验证安装
python -c "import diffusers; print('diffusers version:', diffusers.__version__)"  # 应显示 0.11.1
python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"  # 应显示 0.10.1

# 测试导入（重要！）
python -c "import diffusers; from diffusers import  DiffusionPipeline; print('diffusers imported successfully!')"
```

⚠️ **常见错误**：如果看到 `ImportError: cannot import name 'cached_download'`，说明 `huggingface_hub` 版本太新。请运行：
```bash
pip uninstall huggingface_hub -y
pip install huggingface_hub==0.10.1
```

### 环境验证检查清单

运行以下命令验证环境：

```bash
# 检查Python包
pip list | grep -E "torch|numpy|diffusers|wandb|efficientnet|huggingface"

# 预期输出应包含：
# torch                 1.x.x
# torchvision          0.x.x
# numpy                1.x.x
# diffusers            0.11.1
# wandb                0.12.18
# efficientnet-pytorch x.x.x
# huggingface-hub      0.10.1  # 重要：必须是兼容版本
```

**重要验证步骤**：

```bash
# 测试 diffusers 能否正常导入
python -c "import diffusers; print('diffusers imported successfully')"

# 如果出现 ImportError: cannot import name 'cached_download'
# 说明 huggingface_hub 版本不兼容，参见下方常见问题 Q0
```

---

## 数据准备

### 数据集概述

NoMaD在以下公开数据集上训练：

| 数据集 | 描述 | 链接 | 规模 |
|--------|------|------|------|
| **RECON** | 室内机器人导航 | [链接](https://sites.google.com/view/recon-robot/dataset) | ~100轨迹 |
| **TartanDrive** | 越野驾驶 | [GitHub](https://github.com/castacks/tartan_drive) | 大规模 |
| **SCAND** | 校园导航 | [链接](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links) | 中等 |
| **GoStanford2** | 室内导航(修改版) | [Google Drive](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing) | 中等 |
| **SACSoN** | 多环境 | [链接](https://sites.google.com/view/sacson-review/huron-dataset) | 大规模 |

### 步骤1：下载数据集

#### 什么是ROS Bag？

**ROS Bag (rosbag)** 是ROS (Robot Operating System) 中用于记录和回放数据的文件格式：

- 📦 **文件扩展名**：`.bag`
- 🎥 **内容**：记录了机器人运行时的传感器数据（图像、里程计、激光雷达等）
- ⏱️ **时间戳**：每条消息都带有精确的时间戳
- 🔄 **可回放**：可以像"录像"一样重放机器人的运行过程
- 📊 **数据结构**：包含多个"话题"(topics)，每个话题存储不同类型的数据

**ROS Bag的典型结构**：
```
example_trajectory.bag
├── /camera/image_raw          # 图像话题
│   └── sensor_msgs/Image      # 图像消息
├── /odom                       # 里程计话题
│   └── nav_msgs/Odometry      # 位置、速度信息
└── /imu                        # IMU话题（可选）
    └── sensor_msgs/Imu        # 加速度、角速度
```

**为什么需要处理ROS Bag？**
- NoMaD需要的是图像序列（.jpg）+ 位置信息（.pkl）
- ROS Bag是原始的传感器数据记录
- 需要提取图像并同步对齐位置信息

---

#### 以GoStanford2数据集为例：完整下载和验证流程

> ⚠️ **重要说明**：GoStanford2数据集是**已经处理好的数据**（图像序列+pickle文件），不是ROS Bag格式！  
> **不需要进行ROS Bag处理步骤**，可以直接用于训练。

##### 1) 下载GoStanford2数据集

**GoStanford2 特点**：
- 📍 **环境**：斯坦福大学室内环境（走廊、办公室）
- 🤖 **机器人**：LoCoBot平台
- 📷 **传感器**：RGB摄像头（鱼眼或广角）
- 📊 **轨迹数量**：约1,000条轨迹
- 📏 **轨迹长度**：每条轨迹68-200帧图像
- 💾 **总大小**：约50GB
- 📦 **数据格式**：已处理的图像序列（.jpg） + pickle文件（traj_data.pkl）

```bash
# Linux命令
# 创建数据集目录
mkdir -p ~/visualnav-transformer/nomad_dataset
cd ~/visualnav-transformer/nomad_dataset

# 方式1：使用gdown工具（推荐）
pip install gdown

# 下载整个文件夹（注意：文件很大，约50GB）
gdown --folder https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_

# 下载后会得到 go_stanford 文件夹

# 方式2：使用wget批量下载（如果gdown失败）
# 需要先从Google Drive获取分享链接,然后使用wget下载

# 方式3：从Google Drive手动下载（备选）
# 访问：https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing
# 下载所有文件,然后使用scp上传到服务器:
# scp -r go_stanford/ user@server:~/visualnav-transformer/nomad_dataset/
```

##### 2) 验证下载的数据

```bash
# 查看数据集根目录
ls ~/visualnav-transformer/nomad_dataset/go_stanford

# 应该看到很多轨迹文件夹，例如:
# no10vcF_10_0/ no10vcF_11_0/ no10vcF_11_1/ ...
# sim1hF_0_0/ sim1hF_0_1/ ...
# 共约1000个文件夹

# 统计轨迹数量
ls -d ~/visualnav-transformer/nomad_dataset/go_stanford/*/ | wc -l
```

**查看单个轨迹文件夹内容**：

```bash
# 查看一个轨迹
ls ~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/

# 应该看到:
# 0.jpg 1.jpg 2.jpg ... 67.jpg  <- 图像序列（68张图片）
# traj_data.pkl                  <- 轨迹数据（位置和朝向）

# 统计图像数量
ls ~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/*.jpg | wc -l
```

**使用Python验证pickle文件**：

```python
import pickle
import numpy as np
import os

# 方法1: 使用os.path.expanduser展开~符号(推荐)
pkl_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl")

# 方法2: 直接使用绝对路径
# pkl_path = "/home/your_username/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl"

# 方法3: 使用环境变量
# pkl_path = os.path.join(os.environ['HOME'], "visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl")

# 检查文件是否存在
if not os.path.exists(pkl_path):
    print(f"错误: 文件不存在: {pkl_path}")
    print(f"请检查路径是否正确")
else:
    print(f"找到文件: {pkl_path}")
    
    # 读取轨迹数据
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    
    print("\nKeys:", data.keys())
    # 输出: dict_keys(['position', 'yaw'])
    
    print("Position shape:", data['position'].shape)  # (N, 2) - xy坐标
    print("Yaw shape:", data['yaw'].shape)            # (N,) - 朝向角度
    print("Number of waypoints:", len(data['position']))
    
    # 示例输出:
    # Keys: dict_keys(['position', 'yaw'])
    # Position shape: (68, 2)
    # Yaw shape: (68,)
    # Number of waypoints: 68
    
    # 检查数据范围
    print("\n数据范围:")
    print(f"  Position X: [{data['position'][:, 0].min():.2f}, {data['position'][:, 0].max():.2f}]")
    print(f"  Position Y: [{data['position'][:, 1].min():.2f}, {data['position'][:, 1].max():.2f}]")
    
    # 处理yaw可能是对象数组的情况
    yaw = np.array(data['yaw'], dtype=float)  # 确保转换为浮点数
    print(f"  Yaw: [{yaw.min():.2f}, {yaw.max():.2f}] radians")
    print(f"  Yaw: [{np.degrees(yaw.min()):.1f}, {np.degrees(yaw.max()):.1f}] degrees")
    
    # 显示数据类型信息
    print("\n数据类型:")
    print(f"  position dtype: {data['position'].dtype}")
    print(f"  yaw dtype: {data['yaw'].dtype}")
```

⚠️ **重要：如果看到 dtype 为 `object`，请参考下方的"数据类型修复"章节！**

**快速验证脚本** (保存为`check_gostanford.py`):

```python
#!/usr/bin/env python3
import pickle
import numpy as np
import os
import sys

def check_trajectory(traj_path):
    """检查单个轨迹的完整性"""
    traj_path = os.path.expanduser(traj_path)
    
    if not os.path.exists(traj_path):
        print(f"❌ 轨迹目录不存在: {traj_path}")
        return False
    
    # 检查pickle文件
    pkl_file = os.path.join(traj_path, "traj_data.pkl")
    if not os.path.exists(pkl_file):
        print(f"❌ 缺少traj_data.pkl: {traj_path}")
        return False
    
    # 检查图像文件
    jpg_files = sorted([f for f in os.listdir(traj_path) if f.endswith('.jpg')])
    if len(jpg_files) == 0:
        print(f"❌ 没有图像文件: {traj_path}")
        return False
    
    # 读取pickle数据
    with open(pkl_file, "rb") as f:
        data = pickle.load(f)
    
    num_images = len(jpg_files)
    num_positions = len(data['position'])
    
    print(f"✅ {os.path.basename(traj_path)}: {num_images}张图像, {num_positions}个位置点")
    
    if num_images != num_positions:
        print(f"   ⚠️  警告: 图像数量({num_images})与位置点数量({num_positions})不匹配")
    
    return True

if __name__ == "__main__":
    # 检查单个轨迹
    traj_path = "~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0"
    check_trajectory(traj_path)
    
    print("\n" + "="*50)
    print("检查整个数据集...")
    print("="*50)
    
    # 检查整个数据集
    dataset_root = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford")
    
    if not os.path.exists(dataset_root):
        print(f"❌ 数据集根目录不存在: {dataset_root}")
        sys.exit(1)
    
    trajectories = sorted([d for d in os.listdir(dataset_root) 
                          if os.path.isdir(os.path.join(dataset_root, d))])
    
    print(f"找到 {len(trajectories)} 个轨迹目录")
    print(f"检查前5个轨迹...\n")
    
    for i, traj_name in enumerate(trajectories[:5]):
        traj_path = os.path.join(dataset_root, traj_name)
        check_trajectory(traj_path)
    
    print(f"\n总计: {len(trajectories)} 个轨迹")
```

运行验证:

```bash
# 创建验证脚本
cd ~/visualnav-transformer
nano check_gostanford.py  # 或使用 vim, gedit 等编辑器

# 将上面的脚本内容粘贴进去,保存

# 添加执行权限
chmod +x check_gostanford.py

# 运行验证
python check_gostanford.py
```

---

#### 数据类型修复（重要！）

> ⚠️ **如果训练时遇到以下错误，必须执行此步骤！**

**错误现象**：
```
ValueError: setting an array element with a sequence. 
The requested array has an inhomogeneous shape after 2 dimensions. 
The detected shape was (3, 3) + inhomogeneous part.
```

**原因分析**：

GoStanford2 数据集中的 `traj_data.pkl` 文件的数据类型可能是 `object`（对象类型），而不是 `float64`（浮点数）。这会导致训练代码在加载数据时失败。

**检查数据类型**：

```bash
cd ~/visualnav-transformer

# 创建检查脚本
cat > check_dtype.py << 'EOF'
#!/usr/bin/env python3
import pickle
import numpy as np
import os

# 检查一个轨迹的数据类型
traj_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl")

with open(traj_path, 'rb') as f:
    data = pickle.load(f)

print("数据类型检查:")
print(f"  position dtype: {data['position'].dtype}")
print(f"  yaw dtype: {data['yaw'].dtype}")

if data['position'].dtype == 'object' or data['yaw'].dtype == 'object':
    print("\n❌ 检测到 object 类型！需要修复！")
else:
    print("\n✅ 数据类型正确")
EOF

python check_dtype.py
```

**如果输出显示 `dtype: object`，执行以下修复步骤：**

##### 方法1：批量转换所有轨迹数据（推荐）

```bash
cd ~/visualnav-transformer

# 创建批量转换脚本
cat > fix_go_stanford_dtype.py << 'EOF'
#!/usr/bin/env python3
"""
修复 GoStanford2 数据集的数据类型问题
将 object 类型转换为 float64 类型
"""
import pickle
import numpy as np
import os
from tqdm import tqdm
import shutil

def fix_trajectory_dtype(traj_folder):
    """修复单个轨迹的数据类型"""
    pkl_file = os.path.join(traj_folder, "traj_data.pkl")
    
    if not os.path.exists(pkl_file):
        return False, "pkl文件不存在"
    
    # 备份原文件
    backup_file = pkl_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy2(pkl_file, backup_file)
    
    try:
        # 读取数据
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 检查并转换数据类型
        needs_fix = False
        
        if 'position' in data and data['position'].dtype == 'object':
            data['position'] = np.array(data['position'], dtype=np.float64)
            needs_fix = True
        
        if 'yaw' in data and data['yaw'].dtype == 'object':
            data['yaw'] = np.array(data['yaw'], dtype=np.float64)
            needs_fix = True
        
        # 如果需要修复，保存修复后的数据
        if needs_fix:
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            return True, "已修复"
        else:
            return True, "无需修复"
    
    except Exception as e:
        return False, str(e)

def main():
    dataset_root = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford")
    
    if not os.path.exists(dataset_root):
        print(f"❌ 数据集不存在: {dataset_root}")
        return
    
    # 获取所有轨迹文件夹
    traj_folders = []
    for item in os.listdir(dataset_root):
        traj_path = os.path.join(dataset_root, item)
        if os.path.isdir(traj_path):
            traj_folders.append(traj_path)
    
    print(f"找到 {len(traj_folders)} 个轨迹文件夹")
    print("开始修复数据类型...\n")
    
    success_count = 0
    fixed_count = 0
    error_count = 0
    
    # 处理每个轨迹
    for traj_folder in tqdm(traj_folders, desc="处理进度"):
        success, message = fix_trajectory_dtype(traj_folder)
        
        if success:
            success_count += 1
            if message == "已修复":
                fixed_count += 1
        else:
            error_count += 1
            tqdm.write(f"❌ {os.path.basename(traj_folder)}: {message}")
    
    print("\n" + "="*50)
    print("修复完成！")
    print(f"  成功处理: {success_count}/{len(traj_folders)}")
    print(f"  已修复: {fixed_count}")
    print(f"  无需修复: {success_count - fixed_count}")
    print(f"  失败: {error_count}")
    print("="*50)
    
    if fixed_count > 0:
        print("\n✅ 数据类型已修复！")
        print(f"   备份文件保存在各轨迹文件夹中（*.backup）")
    
    # 验证修复结果
    print("\n验证修复结果...")
    sample_traj = traj_folders[0]
    pkl_file = os.path.join(sample_traj, "traj_data.pkl")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    print(f"  position dtype: {data['position'].dtype}")
    print(f"  yaw dtype: {data['yaw'].dtype}")
    
    if data['position'].dtype == np.float64 and data['yaw'].dtype == np.float64:
        print("\n✅ 验证通过！数据类型正确")
    else:
        print("\n⚠️  验证失败，请检查")

if __name__ == "__main__":
    main()
EOF

# 安装依赖（如果没有）
pip install tqdm

# 运行修复脚本
python fix_go_stanford_dtype.py
```

**预期输出**：

```
找到 1000 个轨迹文件夹
开始修复数据类型...

处理进度: 100%|████████████| 1000/1000 [00:30<00:00, 33.21it/s]

==================================================
修复完成！
  成功处理: 1000/1000
  已修复: 1000
  无需修复: 0
  失败: 0
==================================================

✅ 数据类型已修复！
   备份文件保存在各轨迹文件夹中（*.backup）

验证修复结果...
  position dtype: float64
  yaw dtype: float64

✅ 验证通过！数据类型正确
```

##### 方法2：单个轨迹修复（用于测试）

```python
#!/usr/bin/env python3
import pickle
import numpy as np
import os
import shutil

# 选择一个轨迹进行修复
traj_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0")
pkl_file = os.path.join(traj_path, "traj_data.pkl")

# 备份原文件
backup_file = pkl_file + ".backup"
if not os.path.exists(backup_file):
    shutil.copy2(pkl_file, backup_file)
    print(f"✅ 已备份: {backup_file}")

# 读取数据
with open(pkl_file, 'rb') as f:
    data = pickle.load(f)

print("修复前:")
print(f"  position dtype: {data['position'].dtype}")
print(f"  yaw dtype: {data['yaw'].dtype}")

# 转换数据类型
data['position'] = np.array(data['position'], dtype=np.float64)
data['yaw'] = np.array(data['yaw'], dtype=np.float64)

# 保存修复后的数据
with open(pkl_file, 'wb') as f:
    pickle.dump(data, f)

print("\n修复后:")
# 重新读取验证
with open(pkl_file, 'rb') as f:
    data_fixed = pickle.load(f)
print(f"  position dtype: {data_fixed['position'].dtype}")
print(f"  yaw dtype: {data_fixed['yaw'].dtype}")

print("\n✅ 修复完成！")
```

##### 方法3：如果数据量很大，使用并行处理

```bash
cat > fix_go_stanford_parallel.py << 'EOF'
#!/usr/bin/env python3
import pickle
import numpy as np
import os
import shutil
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def fix_single_trajectory(traj_folder):
    """修复单个轨迹（用于并行处理）"""
    pkl_file = os.path.join(traj_folder, "traj_data.pkl")
    
    if not os.path.exists(pkl_file):
        return traj_folder, False, "pkl不存在"
    
    backup_file = pkl_file + ".backup"
    if not os.path.exists(backup_file):
        shutil.copy2(pkl_file, backup_file)
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        needs_fix = False
        if 'position' in data and data['position'].dtype == 'object':
            data['position'] = np.array(data['position'], dtype=np.float64)
            needs_fix = True
        
        if 'yaw' in data and data['yaw'].dtype == 'object':
            data['yaw'] = np.array(data['yaw'], dtype=np.float64)
            needs_fix = True
        
        if needs_fix:
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            return traj_folder, True, "已修复"
        return traj_folder, True, "无需修复"
    
    except Exception as e:
        return traj_folder, False, str(e)

def main():
    dataset_root = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford")
    
    # 获取所有轨迹文件夹
    traj_folders = [
        os.path.join(dataset_root, item)
        for item in os.listdir(dataset_root)
        if os.path.isdir(os.path.join(dataset_root, item))
    ]
    
    print(f"找到 {len(traj_folders)} 个轨迹")
    print(f"使用 {cpu_count()} 个CPU核心进行并行处理\n")
    
    # 使用进程池并行处理
    with Pool(cpu_count()) as pool:
        results = list(tqdm(
            pool.imap(fix_single_trajectory, traj_folders),
            total=len(traj_folders),
            desc="修复进度"
        ))
    
    # 统计结果
    success = sum(1 for _, s, _ in results if s)
    fixed = sum(1 for _, s, msg in results if s and msg == "已修复")
    errors = sum(1 for _, s, _ in results if not s)
    
    print(f"\n修复完成: 成功 {success}, 已修复 {fixed}, 失败 {errors}")

if __name__ == "__main__":
    main()
EOF

python fix_go_stanford_parallel.py
```

**修复完成后，再次验证**：

```bash
python check_dtype.py

# 应该看到:
# 数据类型检查:
#   position dtype: float64
#   yaw dtype: float64
#
# ✅ 数据类型正确
```

**常见问题**：

**Q: 为什么会出现 object 类型？**
A: 某些数据处理工具在保存 pickle 文件时，如果数据维度不一致或包含 Python 对象，NumPy 会将其保存为 object 类型。

**Q: 修复会丢失数据吗？**
A: 不会。脚本会先备份原文件（`.backup`），然后只是转换数据类型，数值内容完全不变。

**Q: 如果修复失败怎么办？**
A: 使用备份文件恢复：
```bash
cd ~/visualnav-transformer/nomad_dataset/go_stanford
# 恢复所有备份
find . -name "*.backup" -exec sh -c 'mv "$1" "${1%.backup}"' _ {} \;
```

**Q: 可以跳过修复直接训练吗？**
A: 不可以。如果数据类型是 object，训练代码会立即报错。必须先修复。

---

##### 3) 其他数据集的下载方式（可选）

如果你需要训练更强的模型,可以添加其他数据集:

```bash
# TartanDrive（ROS Bag格式，需要从官方申请）
# 访问：https://github.com/castacks/tartan_drive
# 填表后会收到下载链接，包含户外越野数据

# RECON（HDF5格式，不是bag）
# 访问：https://sites.google.com/view/recon-robot/dataset
# 直接下载HDF5文件，包含室内办公环境

# SCAND（ROS Bag格式）
# 访问：https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links
# 下载bag文件，包含楼梯、坡道等复杂场景

# SACSoN（ROS Bag格式）
# 访问：https://sites.google.com/view/sacson-review/huron-dataset
# 下载bag文件，包含水下机器人数据
```

---

### 步骤2：创建数据分割（GoStanford直接开始）

> ✅ **GoStanford用户从这里开始！**  
> GoStanford数据集已经是处理好的格式,跳过ROS Bag处理步骤。

#### 1) 什么是数据分割?

**数据分割**将轨迹分成**训练集**和**测试集**：
- **训练集**（通常80-90%）：用于训练模型
- **测试集**（通常10-20%）：用于评估模型性能

**为什么需要分割？**
- 防止模型"背答案"（过拟合）
- 测试集评估模型在未见过数据上的泛化能力

#### 2.2 运行数据分割脚本

```bash
# Linux命令
# 进入项目训练目录
cd ~/visualnav-transformer/train

# 激活nomad_train环境
conda activate nomad_train

# 运行数据分割（80%训练，20%测试）
python data_split.py \
  --data-dir ~/codespace/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.8

# 参数说明:
# --data-dir: 数据集根目录（包含轨迹文件夹的目录）
# --dataset-name: 数据集名称（用于创建分割文件的子目录）
# --split: 训练集比例（0.8=80%训练，20%测试）
# --data-splits-dir: (可选)分割文件输出目录，默认为 vint_train/data/data_splits
```

#### 2.3 验证分割结果

运行成功后，会在 `vint_train/data/data_splits/go_stanford/` 目录下生成训练集和测试集文件：

```bash
# 查看生成的分割文件目录
ls ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/

# 应该看到:
# train/
# test/

# 查看训练集和测试集文件
ls ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/
# 应该看到: traj_names.txt

ls ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/test/
# 应该看到: traj_names.txt
```

**查看文件内容**：

```bash
# 查看训练集（前10条）
head -n 10 ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/traj_names.txt

# 示例输出:
# no10vcF_10_0
# no10vcF_11_0
# sim1hF_0_0
# sim1hF_0_1
# ...

# 统计训练集和测试集数量
wc -l ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/traj_names.txt
wc -l ~/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/test/traj_names.txt

# 预期输出示例:
# 800 .../train/traj_names.txt
# 200 .../test/traj_names.txt
```

#### 4) 调整分割比例（可选）

如果需要不同的训练/测试比例：

```bash
# 90%训练, 10%测试（更多训练数据）
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.9

# 70%训练, 30%测试（更多测试数据）
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.7
```

---

### 步骤3：处理其他数据集（可选 - 仅针对ROS Bag格式）

> ⏭️ **GoStanford用户可以跳过此步骤，直接进入步骤4配置数据集元信息！**

如果你下载了**TartanDrive**、**SCAND**等ROS Bag格式的数据集，需要进行以下处理。

#### 1) 理解数据处理流程

**数据处理做了什么？**

```
原始ROS Bag                          处理后的数据
┌─────────────────┐                  ┌─────────────────┐
│ example.bag     │                  │ trajectory_001/ │
│ ├─/camera/image │  ───处理───>    │ ├─ 0.jpg        │
│ │  (1000 msgs)  │                  │ ├─ 1.jpg        │
│ ├─/odom         │                  │ ├─ ...          │
│ │  (1000 msgs)  │                  │ ├─ 150.jpg      │
│ └─/imu          │                  │ └─ traj_data.pkl│
└─────────────────┘                  └─────────────────┘
                                     
                                     traj_data.pkl 包含:
                                     - position: [150, 2]  # xy坐标
                                     - yaw: [150]          # 朝向角度
```

**处理过程中发生的事情**：
1. **提取图像**：从`/camera/image`话题提取图像，保存为JPEG
2. **提取位置**：从`/odom`话题提取位置和朝向
3. **时间对齐**：确保图像和位置信息的时间戳匹配
4. **下采样**：根据`sample_rate`降低数据频率
5. **过滤反向**：移除机器人后退的部分
6. **轨迹分割**：如果轨迹有中断，分割成多个子轨迹

#### 2) 配置处理脚本（TartanDrive示例）

TartanDrive数据集已有默认配置，查看 `train/vint_train/process_data/process_bags_config.yaml`：

```yaml
# TartanDrive配置（已存在）
tartan_drive:
  odomtopics: "/odometry/filtered"       # 里程计话题名称
  imtopics: "/zed2/zed_node/left/image_rect_color"  # 图像话题
  ang_offset: 0.0                        # 角度偏移
  img_process_func: "process_tartan_img" # 图像处理函数
  odom_process_func: "nav_to_xy_yaw"     # 里程计处理函数
```

**如果需要添加新数据集配置**：

```yaml
# 在文件末尾添加
my_dataset:
  odomtopics: "/odom"                    # 里程计话题名称
  imtopics: "/camera/image_raw"          # 图像话题名称
  ang_offset: 0.0                        # 角度偏移（弧度）
  img_process_func: "process_locobot_img"  # 图像处理函数
  odom_process_func: "nav_to_xy_yaw"     # 里程计处理函数
```

**配置参数说明**：

| 参数 | 说明 | 如何确定 |
|------|------|----------|
| `odomtopics` | 里程计话题 | 运行 `rosbag info xxx.bag` 查看 |
| `imtopics` | 图像话题 | 运行 `rosbag info xxx.bag` 查看 |
| `ang_offset` | 角度偏移 | 根据机器人坐标系，通常0或π/2 |
| `img_process_func` | 图像处理函数 | 根据图像格式选择（见下表） |
| `odom_process_func` | 里程计处理 | 通常用 `nav_to_xy_yaw` |


**图像处理函数选择**：

| 函数名 | 适用场景 | 图像格式 |
|--------|----------|----------|
| `process_locobot_img` | LoCoBot, 原始图像 | sensor_msgs/Image |
| `process_tartan_img` | TartanDrive | sensor_msgs/Image (特殊) |
| `process_scand_img` | SCAND, 压缩图像 | sensor_msgs/CompressedImage |
| `process_sacson_img` | SACSoN, JPEG压缩 | sensor_msgs/CompressedImage |

#### 3) 执行数据处理（以其他ROS Bag数据集为例）

```bash
cd visualnav-transformer/train

# 处理TartanDrive数据集
python process_bags.py \
    --dataset-name tartan_drive \
    --input-dir ~/nomad_dataset/tartan_drive_raw/ \
    --output-dir ~/nomad_dataset/tartan_drive/ \
    --num-trajs -1 \
    --sample-rate 4

# 参数说明：
# --dataset-name: 必须与process_bags_config.yaml中的名称匹配
# --input-dir: 包含.bag文件的目录（会递归搜索）
# --output-dir: 输出处理后数据的目录
# --num-trajs: 处理的轨迹数量，-1表示全部
# --sample-rate: 采样率（每N帧取1帧），默认1
```

**处理过程输出示例**：

```bash
Bags processed: 100%|████████████| 15/15 [12:30<00:00, 50.00s/bag]

Processing floor4_loop1.bag...
  - Found 10100 images
  - Found 10100 odom messages
  - Sampled to 2525 frames (rate=4)
  - Filtered backwards movement: 2500 frames remaining
  - Split into 1 trajectory
  - Saved to: ~/nomad_dataset/tartan_drive/floor4_loop1_0/

Total trajectories created: 18
```

（更多ROS Bag处理的详细内容请参考原文档中的第三步完整说明）

---

### 步骤4：配置数据集元信息（以go_stanford为例）

> 🎯 **GoStanford用户必读部分！**

数据集元信息配置用于告诉训练脚本**路标点之间的距离**，这对于模型学习正确的导航距离至关重要。

#### 1) 理解`metric_waypoints_distance`参数

**`metric_waypoints_distance`** 表示数据集中相邻路标点（waypoints）之间的平均物理距离（米）。

- 该参数影响模型对距离的理解
- 不同数据集的路标点间距不同，取决于采样率和机器人速度
- GoStanford2数据集的推荐值是 **0.4米**

#### 2) 编辑`data_config.yaml`文件

打开文件：`train/vint_train/data/data_config.yaml`

**添加go_stanford配置**：

```bash
# 在终端中编辑（使用nano）
cd ~/visualnav-transformer/train
nano vint_train/data/data_config.yaml
```

在文件中添加以下内容：

```yaml
# 其他数据集配置...
tartan_drive:
    metric_waypoints_distance: 0.5  # 米

recon:
    metric_waypoints_distance: 0.3

# ===== 添加go_stanford配置 =====
go_stanford:
    metric_waypoints_distance: 0.4  # GoStanford2数据集的路标点间距

# 如果你有其他数据集，也可以添加
# my_custom_dataset:
#     metric_waypoints_distance: 0.5
```

**保存并退出**：
- 按 `Ctrl+O` 保存
- 按 `Enter` 确认
- 按 `Ctrl+X` 退出

#### 3) 验证配置

```bash
# 检查配置文件内容
cat vint_train/data/data_config.yaml

# 或者用Python验证
python -c "
import yaml
with open('vint_train/data/data_config.yaml', 'r') as f:
    config = yaml.safe_load(f)
    print('已配置的数据集:')
    for dataset_name, params in config.items():
        print(f'  {dataset_name}: {params[\"metric_waypoints_distance\"]}m')
"
```

**预期输出**：
```
已配置的数据集:
  tartan_drive: 0.5m
  recon: 0.3m
  go_stanford: 0.4m
```

#### 4) 如何确定其他数据集的距离参数？

如果你要添加自己的数据集，可以这样确定`metric_waypoints_distance`：

```python
import pickle
import numpy as np
import os

# 选择一个典型的轨迹
traj_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl")

with open(traj_path, 'rb') as f:
    data = pickle.load(f)

# 计算相邻位置点之间的距离
positions = data['position']
distances = np.sqrt(np.sum(np.diff(positions, axis=0)**2, axis=1))

# 计算平均距离和中位数
avg_distance = np.mean(distances)
median_distance = np.median(distances)

print(f"平均路标点距离: {avg_distance:.3f}m")
print(f"中位数距离: {median_distance:.3f}m")
print(f"建议使用: {round(median_distance, 1)}m")

# 示例输出:
# 平均路标点距离: 0.387m
# 中位数距离: 0.402m
# 建议使用: 0.4m
```

---

### 步骤5：更新训练配置（以go_stanford为例）

> 🎯 **GoStanford用户必读部分！**

训练配置文件定义了使用哪些数据集、数据存放位置、以及训练的详细参数。

#### 1) 打开训练配置文件

编辑文件：`train/config/nomad.yaml`

```bash
cd ~/visualnav-transformer/train
nano config/nomad.yaml
```

#### 2) 理解配置文件结构

`nomad.yaml` 文件包含三个主要部分：
1. **项目设置** - 项目名称、实验名称、日志配置
2. **模型参数** - 扩散模型的架构和超参数
3. **数据集配置** - 各个数据集的路径和加载参数

#### 3) 添加go_stanford数据集配置

在 `datasets:` 部分添加go_stanford的配置：

```yaml
# ===== 项目设置 =====
project_name: nomad
run_name: nomad_go_stanford_experiment  # 给你的实验起个名字

# ===== 模型参数 =====
use_wandb: True              # 是否使用Weights & Biases记录日志
batch_size: 256              # 根据GPU显存调整 (8GB显存用64, 16GB用128, 24GB+用256)
epochs: 100                  # 训练轮数
lr: 1e-4                     # 学习率
optimizer: adamw             # 优化器
scheduler: "cosine"          # 学习率调度器
warmup_epochs: 4             # 预热轮数

model_type: nomad
vision_encoder: nomad_vint
encoding_size: 256
num_diffusion_iters: 10      # 扩散迭代次数（NoMaD核心参数）
goal_mask_prob: 0.5          # 目标掩码概率（NoMaD核心参数）

len_traj_pred: 8             # 预测轨迹长度
image_size: [96, 96]         # 输入图像尺寸

# ===== 数据集配置 =====
datasets:
  # ===== 添加go_stanford数据集 =====
  go_stanford:
    # 数据集根目录（包含所有轨迹文件夹的目录）
    # ⚠️ 注意：替换<your_username>为你的Linux用户名
    data_folder: /home/<your_username>/visualnav-transformer/nomad_dataset/go_stanford
    
    # 训练集分割文件路径（相对于train/目录）
    train: data/data_splits/go_stanford/train/
    
    # 测试集分割文件路径（相对于train/目录）
    test: data/data_splits/go_stanford/test/
    
    # 轨迹末尾的松弛帧数（用于避免边界问题）
    end_slack: 3
    
    # 每个观测的目标数量
    goals_per_obs: 1
    
    # 是否启用负样本挖掘（提高训练质量）
    negative_mining: True
  
  # ===== 如果你还有其他数据集，继续添加 =====
  # recon:
  #   data_folder: /home/<your_username>/visualnav-transformer/nomad_dataset/recon
  #   train: data/data_splits/recon/train/
  #   test: data/data_splits/recon/test/
  #   end_slack: 3
  #   goals_per_obs: 1
  #   negative_mining: True
  
  # tartan_drive:
  #   data_folder: /home/<your_username>/visualnav-transformer/nomad_dataset/tartan_drive
  #   train: data/data_splits/tartan_drive/train/
  #   test: data/data_splits/tartan_drive/test/
  #   end_slack: 3
  #   goals_per_obs: 1
  #   negative_mining: True
```

#### 4) 重要：修改路径为你的实际路径

**必须修改的地方**：

```yaml
# 将 <your_username> 替换为你的Linux用户名
# 例如：如果你的用户名是 john，则修改为：
data_folder: /home/john/visualnav-transformer/nomad_dataset/go_stanford

# 或者使用绝对路径，例如：
data_folder: /mnt/data/nomad_dataset/go_stanford
```

**如何获取你的用户名和完整路径**：

```bash
# 方法1：查看用户名
echo $USER

# 方法2：查看数据集的完整绝对路径
cd ~/visualnav-transformer/nomad_dataset/go_stanford
pwd
# 输出类似: /home/john/visualnav-transformer/nomad_dataset/go_stanford
# 这就是你应该填写的路径
```

#### 5) 各参数详细说明

**数据集配置参数解释**：

| 参数 | 说明 | go_stanford推荐值 | 备注 |
|------|------|------------------|------|
| `data_folder` | 数据集根目录绝对路径 | 你的实际路径 | 必须存在 |
| `train` | 训练集分割文件相对路径 | `data/data_splits/go_stanford/train/` | 包含`traj_names.txt` |
| `test` | 测试集分割文件相对路径 | `data/data_splits/go_stanford/test/` | 包含`traj_names.txt` |
| `end_slack` | 轨迹末尾忽略帧数 | `3` | 避免轨迹结束时的不稳定数据 |
| `goals_per_obs` | 每个观测的目标点数 | `1` | 通常保持为1 |
| `negative_mining` | 是否使用负样本挖掘 | `True` | 提高训练质量，推荐开启 |

**训练超参数调整建议**：

```yaml
# 根据你的硬件调整batch_size
# GPU显存 8GB
batch_size: 64
num_workers: 4

# GPU显存 16GB  
batch_size: 128
num_workers: 8

# GPU显存 24GB+
batch_size: 256
num_workers: 12

# 如果只用go_stanford数据集（数据量较小），可以：
epochs: 150              # 增加训练轮数
lr: 5e-5                 # 降低学习率防止过拟合
warmup_epochs: 6         # 增加预热轮数
```

#### 6) 保存并验证配置

**保存文件**：
- 按 `Ctrl+O` 保存
- 按 `Enter` 确认
- 按 `Ctrl+X` 退出

**验证配置正确性**：

```bash
# 验证YAML语法
python -c "
import yaml
import os

# 读取配置文件
with open('config/nomad.yaml', 'r') as f:
    config = yaml.safe_load(f)

print('✅ YAML语法正确')
print(f'\\n项目名称: {config[\"project_name\"]}')
print(f'实验名称: {config[\"run_name\"]}')
print(f'批次大小: {config[\"batch_size\"]}')
print(f'训练轮数: {config[\"epochs\"]}')

print('\\n配置的数据集:')
for dataset_name, dataset_config in config['datasets'].items():
    print(f'\\n  数据集: {dataset_name}')
    data_folder = dataset_config['data_folder']
    train_split = dataset_config['train']
    test_split = dataset_config['test']
    
    # 检查数据文件夹是否存在
    data_folder_expanded = os.path.expanduser(data_folder)
    if os.path.exists(data_folder_expanded):
        num_trajs = len([d for d in os.listdir(data_folder_expanded) 
                        if os.path.isdir(os.path.join(data_folder_expanded, d))])
        print(f'    ✅ 数据文件夹存在: {num_trajs} 个轨迹')
    else:
        print(f'    ❌ 数据文件夹不存在: {data_folder}')
        print(f'       请检查路径是否正确！')
    
    # 检查分割文件
    train_file = os.path.join(train_split, 'traj_names.txt')
    test_file = os.path.join(test_split, 'traj_names.txt')
    
    if os.path.exists(train_file):
        with open(train_file) as f:
            train_count = len(f.readlines())
        print(f'    ✅ 训练集: {train_count} 个轨迹')
    else:
        print(f'    ❌ 训练集文件不存在: {train_file}')
    
    if os.path.exists(test_file):
        with open(test_file) as f:
            test_count = len(f.readlines())
        print(f'    ✅ 测试集: {test_count} 个轨迹')
    else:
        print(f'    ❌ 测试集文件不存在: {test_file}')

print('\\n配置验证完成！')
"
```

**预期输出示例**：

```
✅ YAML语法正确

项目名称: nomad
实验名称: nomad_go_stanford_experiment
批次大小: 256
训练轮数: 100

配置的数据集:

  数据集: go_stanford
    ✅ 数据文件夹存在: 58 个轨迹
    ✅ 训练集: 46 个轨迹
    ✅ 测试集: 12 个轨迹

配置验证完成！
```

#### 7) 常见配置错误排查

**❌ 错误1：路径不存在**
```
❌ 数据文件夹不存在: /home/<your_username>/visualnav-transformer/nomad_dataset/go_stanford
```
**解决方法**：
- 检查是否忘记替换`<your_username>`
- 使用`pwd`确认数据集的实际绝对路径
- 确保数据集已经下载到该位置

**❌ 错误2：分割文件不存在**
```
❌ 训练集文件不存在: data/data_splits/go_stanford/train/traj_names.txt
```
**解决方法**：
```bash
# 重新运行数据分割
cd ~/visualnav-transformer/train
python data_split.py \
  --data-dir ~/visualnav-transformer/nomad_dataset/go_stanford \
  --dataset-name go_stanford \
  --split 0.8
```

**❌ 错误3：YAML语法错误**
```
yaml.scanner.ScannerError: mapping values are not allowed here
```
**解决方法**：
- 检查缩进是否正确（使用空格，不要用Tab）
- 检查冒号后面是否有空格
- 使用在线YAML验证器检查语法

---

### 配置完成检查清单

在开始训练之前，确保以下所有步骤都已完成：

- [ ] ✅ 数据集已下载到 `nomad_dataset/go_stanford/`
- [ ] ✅ 数据集包含轨迹文件夹，每个文件夹有`.jpg`图像和`traj_data.pkl`
- [ ] ✅ 运行了`data_split.py`，生成了训练集和测试集分割
- [ ] ✅ `train/traj_names.txt` 和 `test/traj_names.txt` 文件存在
- [ ] ✅ 编辑了`vint_train/data/data_config.yaml`，添加了`go_stanford`配置
- [ ] ✅ `metric_waypoints_distance: 0.4` 已设置
- [ ] ✅ 编辑了`config/nomad.yaml`，添加了`go_stanford`数据集配置
- [ ] ✅ `data_folder`路径已修改为实际绝对路径
- [ ] ✅ `train`和`test`路径指向正确的分割文件目录
- [ ] ✅ 运行验证脚本，所有检查都通过

**现在你可以开始训练了！** 🚀

---

## 模型训练

配置完成后，现在可以开始训练NoMaD模型了！本节将详细讲解训练代码、训练流程和相关配置。

---

### 训练前准备

#### 步骤1：配置W&B（可选但推荐）

```bash
# 安装并登录Weights & Biases
pip install wandb
wandb login

# 输入你的API key（从 https://wandb.ai/settings 获取）

# 或者禁用W&B
# 在 nomad.yaml 中设置: use_wandb: False
```

编辑 `train/train.py` 第395行，修改W&B entity：

```python
wandb.init(
    project=config["project_name"],
    settings=wandb.Settings(start_method="fork"),
    entity="your_wandb_username",  # 改为你的用户名
)
```

#### 步骤2：调整训练配置（根据硬件）

根据GPU显存调整批次大小（已在步骤5的nomad.yaml中配置）：

```yaml
# 显存 8GB
batch_size: 64
num_workers: 4

# 显存 16GB
batch_size: 128
num_workers: 8

# 显存 24GB+
batch_size: 256
num_workers: 12
```

---

### 训练代码详解

#### 主训练脚本 (`train/train.py`) 架构

**1. 导入与初始化 (Lines 1-40)**

```python
import torch
import wandb
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from vint_train.models.nomad.nomad import NoMaD, DenseNetwork
from vint_train.data.vint_dataset import ViNT_Dataset
```

主要模块：
- **PyTorch核心**: 深度学习框架
- **Diffusers**: 扩散模型调度器
- **NoMaD模型**: 视觉编码器 + 扩散策略网络
- **数据集**: 导航轨迹数据加载

**2. 数据加载流程 (Lines 72-130)**

```python
# 遍历配置中的所有数据集
for dataset_name in config["datasets"]:
    data_config = config["datasets"][dataset_name]
    
    # 创建数据集实例
    dataset = ViNT_Dataset(
        data_folder=data_config["data_folder"],       # 图像和轨迹数据
        data_split_folder=data_config["train"],       # train/test分割
        dataset_name=dataset_name,
        image_size=config["image_size"],              # [96, 96]
        waypoint_spacing=data_config["waypoint_spacing"],  # 路标点间隔
        min_dist_cat=config["distance"]["min_dist_cat"],   # 最小距离类别
        max_dist_cat=config["distance"]["max_dist_cat"],   # 最大距离类别
        len_traj_pred=config["len_traj_pred"],        # 预测轨迹长度: 8
        context_size=config["context_size"],          # 时间上下文: 3帧
        normalize=config["normalize"],                # 归一化动作
    )
```

**数据集关键参数解析**:

| 参数 | NoMaD默认值 | 作用 |
|------|------------|------|
| `image_size` | `[96, 96]` | 输入图像尺寸 |
| `context_size` | `3` | 使用前3帧作为观察上下文 |
| `len_traj_pred` | `8` | 预测未来8个路标点 |
| `waypoint_spacing` | `1` | 每帧采样一个路标点 |
| `min_dist_cat` | `0` | 最近目标距离0米 |
| `max_dist_cat` | `20` | 最远目标距离20米 |
| `negative_mining` | `True` | 采样负样本（困难样本） |
| `goals_per_obs` | `1-2` | 每个观察采样的目标数 |

**3. 模型构建 (Lines 155-220)**

NoMaD模型由三个核心组件组成：

```python
if config["model_type"] == "nomad":
    # 组件1: 视觉编码器 (Vision Encoder)
    vision_encoder = NoMaD_ViNT(
        obs_encoding_size=config["encoding_size"],        # 256维特征
        context_size=config["context_size"],              # 3帧上下文
        mha_num_attention_heads=config["mha_num_attention_heads"],  # 4个注意力头
        mha_num_attention_layers=config["mha_num_attention_layers"], # 4层Transformer
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)   # BatchNorm → GroupNorm
    
    # 组件2: 噪声预测网络 (Noise Prediction Network)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,                                      # 输入: (x, y) 位置
        global_cond_dim=config["encoding_size"],          # 条件: 256维视觉特征
        down_dims=config["down_dims"],                    # [64, 128, 256] UNet维度
        cond_predict_scale=config["cond_predict_scale"],  # 条件预测缩放
    )
    
    # 组件3: 距离预测网络 (Distance Prediction Network)
    dist_pred_network = DenseNetwork(
        embedding_dim=config["encoding_size"]             # 256维输入
    )
    
    # 组合成完整模型
    model = NoMaD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        dist_pred_net=dist_pred_network,
    )
    
    # 扩散调度器
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],  # 10次迭代
        beta_schedule='squaredcos_cap_v2',                   # 余弦调度
        clip_sample=True,
        prediction_type='epsilon'                            # 预测噪声
    )
```

**模型架构详解**:

```
输入: 观察图像 (3x96x96) + 目标图像 (3x96x96)
  ↓
[视觉编码器 - NoMaD_ViNT]
  - EfficientNet-B0 提取特征
  - 多头自注意力融合观察和目标
  - 输出: 256维全局特征向量
  ↓
[扩散策略分支]                    [距离预测分支]
  ↓                                ↓
[条件UNet1D]                      [Dense Network]
  - 输入: 随机噪声                 - 输入: 256维特征
  - 条件: 256维视觉特征            - 3层全连接网络
  - 10次迭代去噪                   - 输出: 距离预测 (1维)
  - 输出: 轨迹预测 (8x2)
```

**4. 优化器和学习率调度 (Lines 230-275)**

```python
# 优化器配置
lr = float(config["lr"])  # 1e-4
if config["optimizer"] == "adamw":
    optimizer = AdamW(model.parameters(), lr=lr)

# 学习率调度器
if config["scheduler"] == "cosine":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["epochs"]  # 100 epochs
    )

# Warmup调度器 (前4个epoch逐渐增加学习率)
if config["warmup"]:
    scheduler = GradualWarmupScheduler(
        optimizer,
        multiplier=1,
        total_epoch=config["warmup_epochs"],  # 4 epochs
        after_scheduler=scheduler,
    )
```

**学习率策略**:
- **初始学习率**: 1e-4
- **Warmup**: 前4个epoch线性增长到1e-4
- **主训练**: 余弦退火，逐渐降低到0
- **总训练时长**: 100 epochs

---

### 训练流程详解

#### NoMaD训练循环 (`train_nomad` 函数)

**1. 单个Epoch训练流程**

```python
def train_nomad(
    model: nn.Module,
    ema_model: EMAModel,              # 指数移动平均模型
    optimizer: Adam,
    dataloader: DataLoader,
    noise_scheduler: DDPMScheduler,
    goal_mask_prob: float,            # 目标掩码概率: 0.5
    device: torch.device,
    epoch: int,
    alpha: float = 1e-4,              # 距离损失权重
):
    model.train()
    
    for i, data in enumerate(dataloader):
        # 步骤1: 解包数据
        obs_image, goal_image, actions, distance, goal_pos, action_mask = data
        
        # 步骤2: 数据预处理
        obs_images = torch.split(obs_image, 3, dim=1)  # 分离上下文帧
        obs_images = [transform(img).to(device) for img in obs_images]
        obs_image = torch.cat(obs_images, dim=1)       # 拼接: [B, 9, 96, 96]
        goal_image = transform(goal_image).to(device)  # [B, 3, 96, 96]
        
        # 步骤3: 目标掩码 (Goal Masking)
        # 50%概率掩盖目标,训练探索能力
        B = obs_image.shape[0]
        goal_mask = torch.rand(B) > goal_mask_prob     # 随机掩码
        goal_mask = goal_mask.long().to(device)
        
        # 步骤4: 视觉特征提取
        obsgoal_cond = model(
            "vision_encoder",
            obs_img=obs_image,
            goal_img=goal_image,
            input_goal_mask=goal_mask
        )  # [B, 256]
        
        # 步骤5: 扩散训练 - 添加噪声
        pred_horizon = actions.shape[1]  # 8
        action_dim = actions.shape[2]     # 2 (x, y)
        
        # 计算delta (相邻路标点的增量)
        ndeltas = get_delta(actions.cpu().numpy())
        ndeltas = normalize_data(ndeltas, ACTION_STATS)  # 归一化到[-1, 1]
        ndeltas = torch.from_numpy(ndeltas).float().to(device)
        
        # 采样噪声时间步
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (B,), device=device
        ).long()
        
        # 添加噪声
        noise = torch.randn(ndeltas.shape, device=device)
        noisy_actions = noise_scheduler.add_noise(
            ndeltas, noise, timesteps
        )
        
        # 步骤6: 预测噪声
        noise_pred = model(
            "noise_pred_net",
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=obsgoal_cond
        )
        
        # 步骤7: 计算损失
        # 动作损失 (扩散损失)
        action_loss = F.mse_loss(noise_pred, noise, reduction='none')
        action_loss = (action_loss * action_mask.unsqueeze(-1)).mean()
        
        # 距离损失
        dist_pred = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
        dist_loss = F.mse_loss(dist_pred, distance.unsqueeze(-1))
        
        # 总损失
        total_loss = action_loss + alpha * dist_loss  # alpha=1e-4
        
        # 步骤8: 反向传播和优化
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # 步骤9: 更新EMA模型
        ema_model.step(model)
```

**2. 关键训练技巧**

**① 目标掩码 (Goal Masking)**
```python
# 50%概率掩盖目标图像
# 目的: 让模型学会无目标探索
goal_mask = torch.rand(B) > 0.5

if goal_mask[i]:
    # 使用真实目标图像
    vision_features = encode(obs, goal)
else:
    # 掩盖目标,只使用观察
    vision_features = encode(obs, masked_goal)
```

**② 动作归一化**
```python
# 将动作归一化到[-1, 1]范围
def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

# 预测时反归一化
def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    data = ndata * (stats['max'] - stats['min']) + stats['min']
    return data
```

**③ 指数移动平均 (EMA)**
```python
# 使用EMA平滑模型参数,提升推理稳定性
ema_model = EMAModel(model=model, power=0.75)

# 每次更新后
ema_model.step(model)

# 推理时使用EMA模型
ema_model.averaged_model.eval()
```

**3. 评估流程 (`evaluate_nomad`)**

```python
def evaluate_nomad(ema_model, dataloader, noise_scheduler, device):
    ema_model.averaged_model.eval()
    
    with torch.no_grad():
        for data in dataloader:
            obs_image, goal_image, actions_gt, distance_gt = data
            
            # 特征提取
            obsgoal_cond = ema_model.averaged_model(
                "vision_encoder", 
                obs_img=obs_image,
                goal_img=goal_image,
                input_goal_mask=torch.ones(B).long()  # 评估时不掩码
            )
            
            # 扩散采样 (10次迭代)
            noisy_action = torch.randn((B, 8, 2), device=device)
            
            for t in noise_scheduler.timesteps:
                noise_pred = ema_model.averaged_model(
                    "noise_pred_net",
                    sample=noisy_action,
                    timestep=t,
                    global_cond=obsgoal_cond
                )
                
                # 去噪
                noisy_action = noise_scheduler.step(
                    noise_pred, t, noisy_action
                ).prev_sample
            
            # 反归一化得到最终轨迹
            actions_pred = get_action(noisy_action)
            
            # 计算指标
            action_loss = F.mse_loss(actions_pred, actions_gt)
            cos_sim = F.cosine_similarity(actions_pred, actions_gt)
```

---

### 训练配置详解

#### NoMaD核心配置 (`config/nomad.yaml`)

**1. 模型架构配置**

```yaml
# 模型类型和架构
model_type: nomad                    # 使用NoMaD模型
vision_encoder: nomad_vint           # 视觉编码器类型
encoding_size: 256                   # 特征向量维度

# Transformer配置
mha_num_attention_heads: 4           # 多头注意力头数
mha_num_attention_layers: 4          # Transformer层数
mha_ff_dim_factor: 4                 # 前馈网络维度因子

# UNet配置
down_dims: [64, 128, 256]           # UNet下采样维度
cond_predict_scale: False            # 条件预测缩放

# 扩散参数
num_diffusion_iters: 10              # 扩散迭代次数
goal_mask_prob: 0.5                  # 目标掩码概率
```

**配置参数影响分析**:

| 参数 | 作用 | 增大效果 | 减小效果 |
|------|------|---------|---------|
| `encoding_size` | 特征维度 | 表达能力↑, 显存↑ | 速度↑, 容量↓ |
| `mha_num_attention_heads` | 注意力头数 | 多角度特征↑ | 计算量↓ |
| `mha_num_attention_layers` | Transformer深度 | 抽象能力↑ | 训练速度↑ |
| `num_diffusion_iters` | 扩散步数 | 质量↑, 推理慢 | 速度↑, 质量↓ |
| `goal_mask_prob` | 掩码概率 | 探索能力↑ | 目标导航↑ |

**2. 训练超参数配置**

```yaml
# 训练设置
batch_size: 256                      # 批次大小 (根据显存调整)
epochs: 100                          # 训练轮数
lr: 1e-4                            # 学习率
optimizer: adamw                     # 优化器
seed: 0                             # 随机种子

# 学习率调度
scheduler: "cosine"                  # 余弦退火
warmup: True                         # 启用warmup
warmup_epochs: 4                     # warmup轮数

# 正则化
clipping: False                      # 梯度裁剪
max_norm: 1.0                       # 裁剪范数

# 数据加载
num_workers: 12                      # 数据加载线程数
```

**3. 数据配置**

```yaml
# 图像配置
image_size: [96, 96]                # 输入图像尺寸 (宽, 高)
normalize: True                      # 归一化动作空间

# 时间上下文
context_type: temporal               # 上下文类型: 时间序列
context_size: 3                      # 使用3帧历史

# 距离范围
distance:
  min_dist_cat: 0                   # 最小距离类别
  max_dist_cat: 20                  # 最大距离类别 (20米)
action:
  min_dist_cat: 3                   # 动作最小距离
  max_dist_cat: 20                  # 动作最大距离

# 轨迹预测
len_traj_pred: 8                    # 预测8个路标点
learn_angle: False                   # 不学习朝向角

# 损失权重
alpha: 1e-4                         # 距离损失权重
```

**4. 数据集配置示例**

```yaml
datasets:
  go_stanford:
    data_folder: /home/<username>/nomad_dataset/go_stanford
    train: /home/<username>/data_splits/go_stanford/train/
    test: /home/<username>/data_splits/go_stanford/test/
    
    # 数据增强
    end_slack: 0                    # 轨迹末尾忽略帧数
    goals_per_obs: 2                # 每个观察采样的目标数
    negative_mining: True            # 启用负样本挖掘
    waypoint_spacing: 1              # 路标点采样间隔
  
  recon:
    data_folder: /home/<username>/nomad_dataset/recon
    train: /home/<username>/data_splits/recon/train/
    test: /home/<username>/data_splits/recon/test/
    end_slack: 3                    # RECON数据集末尾有碰撞
    goals_per_obs: 1
    negative_mining: True
```

**5. 日志配置**

```yaml
# Weights & Biases
use_wandb: True                      # 启用W&B日志
project_name: nomad                  # W&B项目名
run_name: nomad                      # 实验名称

# 日志频率
print_log_freq: 100                  # 每100步打印
wandb_log_freq: 10                   # 每10步记录W&B
image_log_freq: 1000                 # 每1000步记录图像
num_images_log: 8                    # 每次记录8张图像
eval_freq: 1                         # 每1轮评估一次
eval_fraction: 0.25                  # 评估数据集的25%
```

---

### 日志保存机制详解

NoMaD训练系统使用**多层次日志保存机制**，包括本地文件保存和云端W&B日志。以下是完整的日志保存逻辑：

#### 1. 项目文件夹结构

训练开始时，系统会自动创建带时间戳的项目文件夹：

```python
# train/train.py (Lines 377-391)
# 生成唯一的运行名称
config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")

# 创建项目文件夹
config["project_folder"] = os.path.join(
    "logs", config["project_name"], config["run_name"]
)
os.makedirs(config["project_folder"])

# ✨ 新增功能: 保存配置文件到项目文件夹
config_save_path = os.path.join(config["project_folder"], "config.yaml")
with open(config_save_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"Config saved to {config_save_path}")
```

**文件夹结构示例**:
```
logs/
└── nomad/                           # 项目名
    └── nomad_2023_11_25_10_30_00/  # 运行名称_时间戳
        ├── config.yaml              # ✨ 训练配置快照 (新增)
        ├── latest.pth               # 最新模型检查点
        ├── 0.pth                    # Epoch 0 模型
        ├── 1.pth                    # Epoch 1 模型
        ├── ...
        ├── 99.pth                   # Epoch 99 模型
        ├── ema_latest.pth           # EMA模型 (最新)
        ├── ema_0.pth                # EMA Epoch 0
        ├── ...
        ├── optimizer_latest.pth     # 优化器状态
        ├── scheduler_latest.pth     # 学习率调度器状态
        └── visualize/               # 可视化结果
            ├── train/
            │   └── epoch0/
            │       └── action_prediction/
            │           ├── img_0.png
            │           └── ...
            └── go_stanford_test/
                └── epoch0/
                    └── action_prediction/
```

#### 2. 模型检查点保存 (Checkpoint Saving)

**ViNT/GNM模型保存逻辑** (train_eval_loop.py Lines 118-141):

```python
def train_eval_loop(...):
    for epoch in range(current_epoch, current_epoch + epochs):
        # ... 训练和评估 ...
        
        # 构造检查点字典
        checkpoint = {
            "epoch": epoch,                      # 当前轮次
            "model": model,                      # 完整模型
            "optimizer": optimizer,              # 优化器状态
            "avg_total_test_loss": np.mean(avg_total_test_loss),  # 平均测试损失
            "scheduler": scheduler               # 学习率调度器
        }
        
        # 保存最新检查点
        latest_path = os.path.join(project_folder, f"latest.pth")
        torch.save(checkpoint, latest_path)
        
        # 保存编号检查点 (每个epoch)
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(checkpoint, numbered_path)
        
        print(f"✓ Checkpoint saved: {numbered_path}")
```

**NoMaD模型保存逻辑** (train_eval_loop.py Lines 225-245):

```python
def train_eval_loop_nomad(...):
    for epoch in range(current_epoch, current_epoch + epochs):
        # ... 训练 ...
        
        # 1. 保存EMA模型 (推理用)
        ema_numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), ema_numbered_path)
        
        ema_latest_path = os.path.join(project_folder, f"ema_latest.pth")
        torch.save(ema_model.averaged_model.state_dict(), ema_latest_path)
        print(f"✓ Saved EMA model to {ema_numbered_path}")
        
        # 2. 保存训练模型
        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        
        latest_path = os.path.join(project_folder, f"latest.pth")
        torch.save(model.state_dict(), latest_path)
        print(f"✓ Saved model to {numbered_path}")
        
        # 3. 保存优化器状态
        optimizer_path = os.path.join(project_folder, f"optimizer_{epoch}.pth")
        optimizer_latest = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), optimizer_latest)
        
        # 4. 保存学习率调度器
        scheduler_path = os.path.join(project_folder, f"scheduler_{epoch}.pth")
        scheduler_latest = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), scheduler_latest)
```

**检查点内容对比**:

| 文件类型 | 包含内容 | 大小 | 用途 |
|---------|---------|------|------|
| `{epoch}.pth` | 模型权重 | ~200MB | 训练恢复 |
| `ema_{epoch}.pth` | EMA模型权重 | ~200MB | 推理部署 |
| `optimizer_latest.pth` | 优化器状态 | ~400MB | 训练恢复 |
| `scheduler_latest.pth` | 学习率状态 | ~1KB | 训练恢复 |
| `latest.pth` | 最新检查点 | ~200MB | 快速恢复 |
| `config.yaml` | 训练配置 | ~5KB | 复现实验 |

#### 3. Weights & Biases (W&B) 云端日志

**初始化W&B** (train.py Lines 394-402):

```python
if config["use_wandb"]:
    wandb.login()
    wandb.init(
        project=config["project_name"],         # 项目名: "nomad"
        settings=wandb.Settings(start_method="fork"),
        entity="your_username",                 # ⚠️ 修改为你的W&B用户名
    )
    wandb.save(args.config, policy="now")       # 保存配置文件
    wandb.run.name = config["run_name"]         # 设置运行名称
    
    # 更新W&B配置
    if wandb.run:
        wandb.config.update(config)
```

**训练期间记录指标** (train_utils.py):

```python
# 每 wandb_log_freq 步记录一次 (默认10步)
if use_wandb and (i % wandb_log_freq == 0):
    wandb.log({
        # 损失指标
        "uc_action_loss": uc_action_loss.item(),
        "gc_action_loss": gc_action_loss.item(),
        "gc_dist_loss": gc_dist_loss.item(),
        "total_loss": total_loss.item(),
        
        # 轨迹相似度
        "uc_action_waypts_cos_sim": uc_cos_sim.mean().item(),
        "gc_action_waypts_cos_sim": gc_cos_sim.mean().item(),
        
        # 训练进度
        "epoch": epoch,
        "batch": i,
        "learning_rate": optimizer.param_groups[0]['lr'],
        
        # 系统指标
        "gpu_memory_allocated": torch.cuda.memory_allocated() / 1e9,
        "gpu_memory_reserved": torch.cuda.memory_reserved() / 1e9,
    }, commit=True)
```

**可视化记录** (每 image_log_freq 步):

```python
if use_wandb and (i % image_log_freq == 0):
    # 生成轨迹可视化图像
    fig = visualize_trajectory_prediction(
        obs_image, goal_image, pred_waypoints, label_waypoints
    )
    
    # 记录到W&B
    wandb.log({
        "trajectory_visualization": wandb.Image(fig),
        "observation": wandb.Image(obs_image[0]),
        "goal": wandb.Image(goal_image[0]),
    }, commit=False)
```

#### 4. 控制台日志 (Console Logging)

**训练进度打印** (每 print_log_freq 步):

```python
if i % print_log_freq == 0:
    print(f"Epoch [{epoch}/{epochs}] "
          f"Batch [{i}/{len(dataloader)}] "
          f"Loss: {total_loss.item():.4f} "
          f"UC_Action: {uc_action_loss.item():.4f} "
          f"GC_Action: {gc_action_loss.item():.4f} "
          f"GC_Dist: {gc_dist_loss.item():.4f} "
          f"LR: {optimizer.param_groups[0]['lr']:.6f} "
          f"GPU Mem: {torch.cuda.memory_allocated()/1e9:.2f}GB")
```

**完整训练输出示例**:
```
Start ViNT DP Training Epoch 0/99
Using cuda devices: 0
Building LMDB cache for go_stanford: 100%|██████████| 5234/5234
Epoch [0/100] Batch [0/256] Loss: 0.1234 UC_Action: 0.0567 GC_Action: 0.0432 GC_Dist: 0.0235 LR: 0.000025 GPU Mem: 8.34GB
Epoch [0/100] Batch [100/256] Loss: 0.0987 UC_Action: 0.0445 GC_Action: 0.0321 GC_Dist: 0.0221 LR: 0.000050 GPU Mem: 8.45GB
...
✓ Saved EMA model to logs/nomad/nomad_2023_11_25_10_30_00/ema_0.pth
✓ Saved model to logs/nomad/nomad_2023_11_25_10_30_00/0.pth
✓ Config saved to logs/nomad/nomad_2023_11_25_10_30_00/config.yaml
```

#### 5. 可视化结果保存机制详解

NoMaD训练系统包含**完整的可视化pipeline**，从训练循环到文件保存都有详细的记录。以下是完整的函数调用关系和参数说明。

---

##### A. 可视化调用链路图

```
训练主循环 (train.py)
    ↓
train_eval_loop_nomad() (train_eval_loop.py)
    ↓
train_nomad() (train_utils.py)
    ↓ 每 image_log_freq 步
visualize_diffusion_action_distribution() (train_utils.py)
    ↓ 生成图像
    ├─ model_output() ← 使用EMA模型生成预测
    ├─ plot_trajs_and_points() ← 绘制轨迹
    └─ plt.savefig() ← 保存到本地
    ↓
wandb.log() ← 上传到W&B
```

**对应文件位置**:
- `train/train.py` (Lines 328-348): 主训练入口
- `train/vint_train/training/train_eval_loop.py` (Lines 147-247): 训练循环管理
- `train/vint_train/training/train_utils.py` (Lines 525-720): NoMaD训练逻辑
- `train/vint_train/training/train_utils.py` (Lines 1038-1178): 可视化函数
- `train/vint_train/visualizing/action_utils.py`: 轨迹绘制工具

---

##### B. 核心可视化函数：`visualize_diffusion_action_distribution()`

**函数位置**: `train/vint_train/training/train_utils.py` Lines 1038-1178

**调用时机**: 
```python
# train_nomad() 函数中 (Line 700)
if image_log_freq != 0 and i % image_log_freq == 0:
    visualize_diffusion_action_distribution(
        ema_model.averaged_model,      # EMA模型(用于推理)
        noise_scheduler,                # DDPM调度器
        batch_obs_images,               # 观察图像(transform后)
        batch_goal_images,              # 目标图像(transform后)
        batch_viz_obs_images,           # 可视化用观察图像(原始尺寸)
        batch_viz_goal_images,          # 可视化用目标图像(原始尺寸)
        actions,                        # Ground Truth动作
        distance,                       # Ground Truth距离
        goal_pos,                       # 目标位置
        device,                         # CUDA设备
        "train",                        # 模式: "train" 或 "go_stanford_test"
        project_folder,                 # 日志保存路径
        epoch,                          # 当前epoch
        num_images_log,                 # 保存图像数量(默认8)
        30,                             # 采样数量(生成30条轨迹)
        use_wandb,                      # 是否上传W&B
    )
```

**函数签名详解**:

```python
def visualize_diffusion_action_distribution(
    ema_model: nn.Module,                    # EMA模型,参数更平滑
    noise_scheduler: DDPMScheduler,          # 扩散调度器
    batch_obs_images: torch.Tensor,          # [B, 9, 96, 96] 观察图像(3帧×3通道)
    batch_goal_images: torch.Tensor,         # [B, 3, 96, 96] 目标图像
    batch_viz_obs_images: torch.Tensor,      # [B, 3, H, W] 原始观察(用于显示)
    batch_viz_goal_images: torch.Tensor,     # [B, 3, H, W] 原始目标(用于显示)
    batch_action_label: torch.Tensor,        # [B, 8, 2] Ground Truth轨迹
    batch_distance_labels: torch.Tensor,     # [B] Ground Truth距离
    batch_goal_pos: torch.Tensor,            # [B, 2] 目标位置(x, y)
    device: torch.device,                    # cuda:0
    eval_type: str,                          # "train" 或 "go_stanford_test"
    project_folder: str,                     # "logs/nomad/<run_name>"
    epoch: int,                              # 0, 1, 2, ...
    num_images_log: int = 8,                 # 每次保存8张图
    num_samples: int = 30,                   # 每个观察采样30条轨迹
    use_wandb: bool = True,                  # 是否上传W&B
):
```

**参数详细说明**:

| 参数名 | 形状 | 说明 | 示例值 |
|-------|------|------|-------|
| `batch_obs_images` | `[B, 9, 96, 96]` | 观察图像,包含3帧历史 | 经过transform的张量 |
| `batch_goal_images` | `[B, 3, 96, 96]` | 目标图像 | 经过transform的张量 |
| `batch_viz_obs_images` | `[B, 3, H, W]` | 用于显示的原始观察图像 | H=480, W=640 |
| `batch_viz_goal_images` | `[B, 3, H, W]` | 用于显示的原始目标图像 | H=480, W=640 |
| `batch_action_label` | `[B, 8, 2]` | 真实轨迹(8个waypoint) | [[dx1, dy1], ...] |
| `batch_distance_labels` | `[B]` | 到目标的真实距离 | [5.2, 8.7, ...] |
| `batch_goal_pos` | `[B, 2]` | 目标相对位置 | [[gx, gy], ...] |
| `eval_type` | `str` | 数据集类型 | "train", "go_stanford_test" |
| `num_samples` | `int` | 采样轨迹数量 | 30条(用于可视化分布) |

---

##### C. 可视化生成流程

**步骤1: 模型推理** (`model_output()` 函数)

```python
def model_output(
    model, noise_scheduler, 
    batch_obs_images, batch_goal_images,
    pred_horizon=8, action_dim=2, num_samples=30, device
):
    # 1. 无目标条件推理 (uc)
    goal_mask = torch.ones((B,)).long().to(device)  # 掩盖目标
    obs_cond = model("vision_encoder", 
                     obs_img=batch_obs_images, 
                     goal_img=batch_goal_images, 
                     input_goal_mask=goal_mask)    # [B, 256]
    
    # 2. 扩散采样 (无目标)
    noisy_action = torch.randn((B*num_samples, 8, 2), device=device)
    for k in noise_scheduler.timesteps:
        noise_pred = model("noise_pred_net", 
                          sample=noisy_action, 
                          timestep=k, 
                          global_cond=obs_cond.repeat_interleave(num_samples, dim=0))
        noisy_action = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=noisy_action
        ).prev_sample
    uc_actions = get_action(noisy_action)  # [B*num_samples, 8, 2]
    
    # 3. 有目标条件推理 (gc)
    no_mask = torch.zeros((B,)).long().to(device)  # 显示目标
    obsgoal_cond = model("vision_encoder", 
                         obs_img=batch_obs_images, 
                         goal_img=batch_goal_images, 
                         input_goal_mask=no_mask)
    
    # 4. 扩散采样 (有目标)
    noisy_action = torch.randn((B*num_samples, 8, 2), device=device)
    for k in noise_scheduler.timesteps:
        noise_pred = model("noise_pred_net", 
                          sample=noisy_action, 
                          timestep=k, 
                          global_cond=obsgoal_cond.repeat_interleave(num_samples, dim=0))
        noisy_action = noise_scheduler.step(
            model_output=noise_pred, timestep=k, sample=noisy_action
        ).prev_sample
    gc_actions = get_action(noisy_action)  # [B*num_samples, 8, 2]
    
    # 5. 距离预测
    gc_distance = model("dist_pred_net", obsgoal_cond=obsgoal_cond)
    
    return {
        'uc_actions': uc_actions,    # [B*30, 8, 2] 无目标轨迹
        'gc_actions': gc_actions,    # [B*30, 8, 2] 有目标轨迹
        'gc_distance': gc_distance,  # [B] 距离预测
    }
```

**关键点**:
- 对**每个观察**生成`num_samples=30`条轨迹
- `uc_actions`: 探索模式的轨迹分布
- `gc_actions`: 导航模式的轨迹分布
- 通过扩散模型的10步去噪生成

**步骤2: 轨迹绘制** (`plot_trajs_and_points()`)

```python
# 对每张图像
for i in range(num_images_log):  # 默认8张
    fig, ax = plt.subplots(1, 3, figsize=(18.5, 10.5))
    
    # 左图: 轨迹分布图
    uc_actions = uc_actions_list[i]      # [30, 8, 2]
    gc_actions = gc_actions_list[i]      # [30, 8, 2]
    action_label = batch_action_label[i] # [8, 2]
    
    # 合并所有轨迹
    traj_list = np.concatenate([
        uc_actions,        # 30条红色轨迹(探索)
        gc_actions,        # 30条绿色轨迹(导航)
        action_label[None],# 1条品红轨迹(Ground Truth)
    ], axis=0)
    
    # 颜色和透明度
    traj_colors = ["red"] * 30 + ["green"] * 30 + ["magenta"]
    traj_alphas = [0.1] * 60 + [1.0]  # 预测轨迹半透明,GT不透明
    
    # 绘制机器人和目标位置
    point_list = [np.array([0, 0]), batch_goal_pos[i]]
    point_colors = ["green", "red"]
    
    plot_trajs_and_points(
        ax[0], traj_list, point_list, 
        traj_colors, point_colors, 
        traj_alphas=traj_alphas
    )
    
    # 中图: 观察图像
    ax[1].imshow(to_numpy(batch_viz_obs_images[i]).transpose(1, 2, 0))
    
    # 右图: 目标图像
    ax[2].imshow(to_numpy(batch_viz_goal_images[i]).transpose(1, 2, 0))
    
    # 设置标题
    ax[0].set_title("diffusion action predictions")
    ax[1].set_title("observation")
    ax[2].set_title(f"goal: label={batch_distance_labels[i]:.1f} "
                    f"gc_dist={gc_distances_avg[i]:.2f}±{gc_distances_std[i]:.2f}")
    
    # 保存
    save_path = os.path.join(visualize_path, f"sample_{i}.png")
    plt.savefig(save_path)
    wandb_list.append(wandb.Image(save_path))
    plt.close(fig)
```

**步骤3: 文件保存**

```python
# 创建保存路径
visualize_path = os.path.join(
    project_folder,                    # logs/nomad/<run_name>
    "visualize",
    eval_type,                         # train 或 go_stanford_test
    f"epoch{epoch}",                   # epoch0, epoch1, ...
    "action_sampling_prediction",
)
os.makedirs(visualize_path, exist_ok=True)

# 保存图像
save_path = os.path.join(visualize_path, f"sample_{i}.png")
plt.savefig(save_path)
```

**生成的文件结构**:

```
logs/nomad/<run_name>/
└── visualize/
    ├── train/
    │   ├── epoch0/
    │   │   └── action_sampling_prediction/
    │   │       ├── sample_0.png  ← 第1张可视化
    │   │       ├── sample_1.png  ← 第2张可视化
    │   │       ├── ...
    │   │       └── sample_7.png  ← 第8张可视化
    │   ├── epoch1/
    │   │   └── action_sampling_prediction/
    │   └── ...
    └── go_stanford_test/
        └── epoch0/
            └── action_sampling_prediction/
                ├── sample_0.png
                └── ...
```

**步骤4: W&B上传**

```python
if len(wandb_list) > 0 and use_wandb:
    wandb.log({
        f"{eval_type}_action_samples": wandb_list
    }, commit=False)
```

**W&B中的显示**:
- 键名: `train_action_samples` 或 `go_stanford_test_action_samples`
- 内容: 8张可视化图像的gallery
- 可在W&B网页端查看、对比

---

##### D. 其他可视化函数

**1. ViNT/GNM模型的可视化** (非NoMaD)

**函数**: `visualize_traj_pred()` (action_utils.py Lines 27-113)

```python
# 调用位置: train_utils.py Line 150
visualize_traj_pred(
    to_numpy(obs_image),         # [B, H, W, 3]
    to_numpy(goal_image),        # [B, H, W, 3]
    to_numpy(dataset_index),     # [B] 数据集索引
    to_numpy(goal_pos),          # [B, 2] 目标位置
    to_numpy(action_pred),       # [B, 8, 2] 预测轨迹
    to_numpy(action_label),      # [B, 8, 2] GT轨迹
    mode,                        # "train" 或 "<dataset>_test"
    normalized,                  # True/False
    project_folder,
    epoch,
    num_images_log,
    use_wandb=use_wandb,
)
```

**保存路径**:
```
logs/<project>/<run>/visualize/<mode>/epoch<N>/action_prediction/
```

**2. 距离预测可视化**

**函数**: `visualize_dist_pred()` (distance_utils.py Lines 9-66)

```python
# 调用位置: train_utils.py Line 139
visualize_dist_pred(
    to_numpy(obs_image),         # 观察图像
    to_numpy(goal_image),        # 目标图像
    to_numpy(dist_pred),         # 距离预测
    to_numpy(dist_label),        # 距离GT
    mode,
    project_folder,
    epoch,
    num_images_log,
    use_wandb=use_wandb,
)
```

**保存路径**:
```
logs/<project>/<run>/visualize/<mode>/epoch<N>/dist_classification/
```

**显示内容**:
- 并排显示观察图像和目标图像
- 标题显示预测距离 vs 真实距离
- 如果误差 > 3.0，标题变红色警告

---

##### F. 可视化频率控制

**配置参数** (nomad.yaml):

```yaml
image_log_freq: 1000          # 每1000个batch保存一次
num_images_log: 8             # 每次保存8张图像
```

**实际触发**:

```python
# 在 train_nomad() 中
for i, data in enumerate(train_loader):
    # ... 训练逻辑 ...
    
    if image_log_freq != 0 and i % image_log_freq == 0:
        # 第0, 1000, 2000, 3000, ... 个batch触发
        visualize_diffusion_action_distribution(...)
```

**示例时间线** (假设1050个batch/epoch):

```
Epoch 0:
├─ Batch 0    → 生成可视化 (8张图)
├─ Batch 1000 → 生成可视化 (8张图)
└─ Batch 1049 → 训练结束

Epoch 1:
├─ Batch 0    → 生成可视化 (8张图)
├─ Batch 1000 → 生成可视化 (8张图)
└─ Batch 1049 → 训练结束

每个epoch生成 2 × 8 = 16 张可视化图像
```

**存储估算**:

| 图像 | 分辨率 | 大小 | 数量(100 epochs) | 总计 |
|------|--------|------|-----------------|------|
| 单张可视化 | 1850×1050 | ~500KB | 2×8×100 = 1600 | ~800MB |
| 评估可视化 | 1850×1050 | ~500KB | 1×8×100 = 800 | ~400MB |
| **总计** | - | - | - | **~1.2GB** |

**优化建议**:

```yaml
# 训练早期: 高频率监控
image_log_freq: 100         # 每100步一次
num_images_log: 4           # 只保存4张

# 训练中期: 标准频率
image_log_freq: 1000        # 每1000步一次
num_images_log: 8           # 保存8张

# 训练后期: 低频率
image_log_freq: 5000        # 每5000步一次
num_images_log: 8           # 保存8张
```

#### 6. 日志查看与分析

**查看本地日志**:

```bash
# 查看所有实验
ls logs/nomad/

# 查看特定实验的文件
ls -lh logs/nomad/nomad_2023_11_25_10_30_00/

# 查看配置文件
cat logs/nomad/nomad_2023_11_25_10_30_00/config.yaml

# 查看可视化结果
eog logs/nomad/nomad_2023_11_25_10_30_00/visualize/train/epoch0/action_prediction/img_0.png
```

**查看W&B日志**:

1. 访问 `https://wandb.ai/<your_username>/nomad`
2. 查看关键图表：
   - **Loss曲线**: `uc_action_loss`, `gc_action_loss`, `gc_dist_loss`
   - **学习率**: `learning_rate` (余弦退火曲线)
   - **相似度**: `uc_action_waypts_cos_sim`, `gc_action_waypts_cos_sim`
   - **系统指标**: `gpu_memory_allocated`
   - **可视化**: `trajectory_visualization`

**使用W&B API查询**:

```python
import wandb

# 登录
wandb.login()

# 获取运行记录
api = wandb.Api()
runs = api.runs("your_username/nomad")

# 查看最佳模型
best_run = min(runs, key=lambda run: run.summary.get("gc_action_loss", float('inf')))
print(f"Best run: {best_run.name}")
print(f"Best loss: {best_run.summary['gc_action_loss']}")

# 下载检查点
best_run.file("model.pth").download(replace=True)
```

#### 7. 恢复训练

**从检查点恢复** (train.py Lines 283-301):

```python
# 在配置文件中添加
load_run: "nomad/nomad_2023_11_25_10_30_00"

# 训练脚本会自动加载
if "load_run" in config:
    load_project_folder = os.path.join("logs", config["load_run"])
    latest_path = os.path.join(load_project_folder, "latest.pth")
    
    # 加载检查点
    latest_checkpoint = torch.load(latest_path)
    load_model(model, config["model_type"], latest_checkpoint)
    
    # 恢复epoch
    if "epoch" in latest_checkpoint:
        current_epoch = latest_checkpoint["epoch"] + 1
    
    # 恢复优化器和调度器
    if "optimizer" in latest_checkpoint:
        optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
    if "scheduler" in latest_checkpoint:
        scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    
    print(f"✓ Resumed from epoch {current_epoch}")
```

#### 8. 日志保存最佳实践

**配置建议**:

```yaml
# 高频率监控 (调试阶段)
print_log_freq: 10
wandb_log_freq: 5
image_log_freq: 100

# 标准监控 (正式训练)
print_log_freq: 100
wandb_log_freq: 10
image_log_freq: 1000

# 低频率监控 (长期训练)
print_log_freq: 500
wandb_log_freq: 50
image_log_freq: 5000
```

**存储管理**:

```bash
# 定期清理旧检查点 (保留最近10个epoch)
cd logs/nomad/nomad_2023_11_25_10_30_00/
ls -t *.pth | tail -n +11 | xargs rm

# 压缩可视化结果
tar -czf visualize.tar.gz visualize/
rm -rf visualize/

# 备份最佳模型
cp ema_latest.pth ../../best_models/nomad_best.pth
```

**日志分析脚本**:

```python
# analyze_logs.py
import torch
import glob

def analyze_checkpoints(project_folder):
    checkpoints = sorted(glob.glob(f"{project_folder}/*.pth"))
    
    for ckpt_path in checkpoints:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        if isinstance(ckpt, dict) and "avg_total_test_loss" in ckpt:
            epoch = ckpt["epoch"]
            loss = ckpt["avg_total_test_loss"]
            print(f"Epoch {epoch}: Loss = {loss:.4f}")

# 使用
analyze_checkpoints("logs/nomad/nomad_2023_11_25_10_30_00")
```

---

### 配置文件保存功能说明

**新增功能**: 训练开始时自动保存完整配置到日志文件夹

**位置**: `train/train.py` Lines 383-387

**代码**:
```python
# 保存配置文件到项目文件夹
config_save_path = os.path.join(config["project_folder"], "config.yaml")
with open(config_save_path, "w") as f:
    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
print(f"Config saved to {config_save_path}")
```

**优势**:
1. ✅ **实验可复现**: 每次训练的完整配置都被保存
2. ✅ **参数追溯**: 无需记忆使用了哪些参数
3. ✅ **对比分析**: 轻松对比不同实验的配置差异
4. ✅ **快速恢复**: 直接使用保存的配置文件重新训练

**使用示例**:

```bash
# 1. 训练模型
python train.py -c config/nomad.yaml
# 输出: Config saved to logs/nomad/nomad_2023_11_25_10_30_00/config.yaml

# 2. 查看保存的配置
cat logs/nomad/nomad_2023_11_25_10_30_00/config.yaml

# 3. 使用保存的配置重新训练 (完全相同的设置)
python train.py -c logs/nomad/nomad_2023_11_25_10_30_00/config.yaml
```

---

### 训练过程详解

#### 训练输出示例与解读

当你运行 `python train.py -c config/nomad.yaml` 后，会看到详细的训练日志。下面是基于**实际训练1个epoch**的完整输出及其含义解析：

##### 1) 初始化阶段

```bash
$ python train.py -c config/nomad.yaml

# W&B初始化
wandb: Currently logged in as: securitycfs (mvasl). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.12.18
wandb: Run data is saved locally in /home/yifei/codespace/visualnav-transformer/train/wandb/run-20251201_164322-ctu0lccn
wandb: Syncing run earnest-fog-5
wandb: ⭐️ View project at https://wandb.ai/mvasl/nomad
wandb: 🚀 View run at https://wandb.ai/mvasl/nomad/runs/ctu0lccn

# 配置参数打印
{'project_name': 'nomad', 
 'run_name': 'nomad-sample_2025_12_01_16_43_21', 
 'use_wandb': True, 
 'batch_size': 128,                    # 批次大小
 'epochs': 1,                          # 训练轮数
 'lr': '1e-4',                         # 学习率
 'optimizer': 'adamw', 
 'model_type': 'nomad', 
 'vision_encoder': 'nomad_vint', 
 'encoding_size': 256,                 # 特征编码维度
 'num_diffusion_iters': 10,            # 扩散迭代次数
 'goal_mask_prob': 0.5,                # 目标掩码概率
 'len_traj_pred': 8,                   # 预测轨迹长度
 'image_size': [96, 96],               # 输入图像尺寸
 'datasets': {
   'go_stanford': {
     'data_folder': '/home/yifei/codespace/visualnav-transformer/nomad_dataset/go_stanford',
     'train': '/home/yifei/codespace/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/train/',
     'test': '/home/yifei/codespace/visualnav-transformer/train/vint_train/data/data_splits/go_stanford/test/',
     'end_slack': 0, 
     'goals_per_obs': 2, 
     'negative_mining': True
   }
 },
 'project_folder': 'logs/nomad/nomad-sample_2025_12_01_16_43_21'
}

Using cuda devices: 0
Using cosine annealing with T_max 1
Using warmup scheduler
```

**初始化阶段关键信息：**
- 📊 **W&B项目**: `mvasl/nomad`，运行ID: `ctu0lccn`
- 🎯 **实验名称**: `nomad-sample_2025_12_01_16_43_21`
- 💾 **日志保存**: `logs/nomad/nomad-sample_2025_12_01_16_43_21/`
- ⚙️ **GPU**: 使用CUDA设备0
- 📈 **学习率调度**: 余弦退火 + 预热

---

##### 2) 训练阶段 (Epoch 0 - 完整过程)

```bash
Start ViNT DP Training Epoch 0/0

# ========== Batch 0 (初始批次) ==========
Train Batch:   0%|                                        | 0/1050 [00:09<?, ?it/s, loss=1.09]

(epoch 0) (batch 0/1049) uc_action_loss (train): 49.1917 
  (100pt moving_avg: 49.1917) (avg: 49.1917)
(epoch 0) (batch 0/1049) uc_action_waypts_cos_sim (train): 0.39
(epoch 0) (batch 0/1049) uc_multi_action_waypts_cos_sim (train): 0.3977

(epoch 0) (batch 0/1049) gc_dist_loss (train): 151.9019
(epoch 0) (batch 0/1049) gc_action_loss (train): 43.4216
(epoch 0) (batch 0/1049) gc_action_waypts_cos_sim (train): 0.5466
(epoch 0) (batch 0/1049) gc_multi_action_waypts_cos_sim (train): 0.6026

# ========== Batch 100 ==========
Train Batch:  10%|█████████▏                              | 100/1050 [00:33<03:19, 4.77it/s, loss=1.18]

(epoch 0) (batch 100/1049) uc_action_loss (train): 40.8705 
  (100pt moving_avg: 45.0311) (avg: 45.0311)
(epoch 0) (batch 100/1049) uc_action_waypts_cos_sim (train): 0.489
(epoch 0) (batch 100/1049) uc_multi_action_waypts_cos_sim (train): 0.5799

(epoch 0) (batch 100/1049) gc_dist_loss (train): 155.4245 
  (100pt moving_avg: 153.6632) (avg: 153.6632)
(epoch 0) (batch 100/1049) gc_action_loss (train): 43.6779
(epoch 0) (batch 100/1049) gc_action_waypts_cos_sim (train): 0.3727
(epoch 0) (batch 100/1049) gc_multi_action_waypts_cos_sim (train): 0.4224

# ========== Batch 500 (中期) ==========
Train Batch:  48%|█████████████████████████████▋          | 500/1050 [01:59<01:55, 4.76it/s, loss=1.12]

(epoch 0) (batch 500/1049) uc_action_loss (train): 42.1491 
  (100pt moving_avg: 45.3422) (avg: 45.3422)
(epoch 0) (batch 500/1049) uc_action_waypts_cos_sim (train): 0.4844
(epoch 0) (batch 500/1049) uc_multi_action_waypts_cos_sim (train): 0.5207

(epoch 0) (batch 500/1049) gc_dist_loss (train): 141.1024 
  (100pt moving_avg: 152.3188) (avg: 152.3188)
(epoch 0) (batch 500/1049) gc_action_loss (train): 44.2803
(epoch 0) (batch 500/1049) gc_action_waypts_cos_sim (train): 0.4765
(epoch 0) (batch 500/1049) gc_multi_action_waypts_cos_sim (train): 0.5051

# ========== Batch 1000 (末期) ==========
Train Batch:  95%|██████████████████████████████████████▍ | 1000/1050 [03:48<00:10, 4.60it/s, loss=1.18]

(epoch 0) (batch 1000/1049) uc_action_loss (train): 40.632 
  (100pt moving_avg: 44.0944) (avg: 44.0944)
(epoch 0) (batch 1000/1049) uc_action_waypts_cos_sim (train): 0.3803
(epoch 0) (batch 1000/1049) uc_multi_action_waypts_cos_sim (train): 0.4241

(epoch 0) (batch 1000/1049) gc_dist_loss (train): 140.782 
  (100pt moving_avg: 150.0024) (avg: 150.0024)
(epoch 0) (batch 1000/1049) gc_action_loss (train): 46.4176
(epoch 0) (batch 1000/1049) gc_action_waypts_cos_sim (train): 0.473
(epoch 0) (batch 1000/1049) gc_multi_action_waypts_cos_sim (train): 0.4838

# 保存模型检查点
Saved EMA model to logs/nomad/nomad-sample_2025_12_01_16_43_21/ema_latest.pth
Saved model to logs/nomad/nomad-sample_2025_12_01_16_43_21/0.pth
```

**训练阶段指标解读：**

| 指标名称 | 含义 | 初始值 | 第1000批次 | 趋势 |
|---------|------|--------|-----------|------|
| `uc_action_loss` | **无目标条件**的动作损失 | 49.19 | 40.63 | ✅ 下降 |
| `uc_action_waypts_cos_sim` | 无目标条件下路标点余弦相似度 | 0.39 | 0.38 | ➡️ 波动 |
| `gc_dist_loss` | **有目标条件**的距离预测损失 | 151.90 | 140.78 | ✅ 下降 |
| `gc_action_loss` | **有目标条件**的动作损失 | 43.42 | 46.42 | ⚠️ 波动 |
| `gc_action_waypts_cos_sim` | 有目标条件下路标点余弦相似度 | 0.55 | 0.47 | ⚠️ 下降 |

**关键术语说明：**
- **`uc`** (unconditioned): 无目标条件，模型被要求在**没有明确目标**的情况下预测轨迹（探索模式）
- **`gc`** (goal conditioned): 有目标条件，模型被给定**明确目标图像**来预测导航轨迹
- **`100pt moving_avg`**: 最近100个批次的移动平均值，用于平滑噪声
- **`avg`**: 从训练开始到当前的全局平均值

**训练速度：**
- 处理速度: ~4.6-4.8 it/s (每秒约4.7个批次)
- 每100批次耗时: ~21-22秒
- 完整epoch (1050批次): 约3分48秒

---

##### 3) 验证阶段 (Epoch 0 - 测试集评估)

```bash
Start go_stanford_test ViNT DP Testing Epoch 0/0

# ========== 验证开始 ==========
Evaluating go_stanford_test for epoch 0:   0%|              | 0/21 [00:03<?, ?it/s, loss=1.11]

(epoch 0) (batch 0/20) uc_action_loss (go_stanford_test): 46.921
(epoch 0) (batch 0/20) uc_action_waypts_cos_sim (go_stanford_test): 0.3898
(epoch 0) (batch 0/20) uc_multi_action_waypts_cos_sim (go_stanford_test): 0.4443

(epoch 0) (batch 0/20) gc_dist_loss (go_stanford_test): 151.2758
(epoch 0) (batch 0/20) gc_action_loss (go_stanford_test): 47.5994
(epoch 0) (batch 0/20) gc_action_waypts_cos_sim (go_stanford_test): 0.4017
(epoch 0) (batch 0/20) gc_multi_action_waypts_cos_sim (go_stanford_test): 0.435

# 验证完成
FINISHED TRAINING
```

**验证结果对比 (训练集 vs 测试集)：**

| 指标 | 训练集 (最后) | 测试集 | 差异 | 状态 |
|------|------------|--------|------|------|
| `uc_action_loss` | 40.63 | 46.92 | +6.29 | ⚠️ 测试损失较高 |
| `uc_action_waypts_cos_sim` | 0.38 | 0.39 | +0.01 | ✅ 接近 |
| `gc_dist_loss` | 140.78 | 151.28 | +10.50 | ⚠️ 测试损失较高 |
| `gc_action_loss` | 46.42 | 47.60 | +1.18 | ✅ 相对接近 |
| `gc_action_waypts_cos_sim` | 0.47 | 0.40 | -0.07 | ⚠️ 测试表现稍差 |

**分析：**
- 由于**只训练了1个epoch**，模型还未充分收敛
- 测试集损失高于训练集是正常的（模型尚未见过测试数据）
- 需要继续训练至少50-100个epoch才能看到明显改善

---

##### 4) 训练完成总结

```bash
wandb: Waiting for W&B process to finish... (success).

# ========== W&B记录的指标历史 ==========
wandb: Run history:
wandb:   diffusion_eval_loss (goal masking) ▅▇▁▁▆▅▅▅▄▄▆▅█▃▄▂▅▇▆▄▇
wandb:   diffusion_eval_loss (no masking) ▅▇▁▁▆▅▅▅▄▄▆▅█▃▄▂▅▇▆▄▇
wandb:   diffusion_loss ▁▄▇▁█▄▄▄▄▆▅▅▅▅▂▅▄▂▅▃▄▃▂▄▃▄▅▄▄▄▅▄▁▃▅▂▇▃▂▃
wandb:   dist_loss ▂▄▃▃▅▄▆▄▆▁▃▆▆▅▄▄▆▇▄▅▂█▃▅▅▆▅▃▅▄▅█▆▁▅▄▅▃▅▃
wandb:   gc_action_loss (train) ▂▂▃▇▃▃█▃▁▃▆
wandb:   gc_dist_loss (train) ▅▆▅▅█▃█▅▅▁▃
wandb:   uc_action_loss (train) █▃▇▆▇▃▃▄█▁▂
wandb:   lr ▁

# ========== 最终指标总结 ==========
wandb: Run summary:
wandb:   diffusion_eval_loss (goal masking) 1.13049
wandb:   diffusion_eval_loss (no masking) 1.13504
wandb:   diffusion_eval_loss (random masking) 1.13241
wandb:   diffusion_loss 1.21677
wandb:   dist_loss 140.37523
wandb:   
wandb:   gc_action_loss (go_stanford_test) 47.59941
wandb:   gc_action_loss (train) 46.41763
wandb:   gc_action_waypts_cos_sim (go_stanford_test) 0.40173
wandb:   gc_action_waypts_cos_sim (train) 0.47304
wandb:   gc_dist_loss (go_stanford_test) 151.2758
wandb:   gc_dist_loss (train) 140.78198
wandb:   gc_multi_action_waypts_cos_sim (go_stanford_test) 0.43503
wandb:   gc_multi_action_waypts_cos_sim (train) 0.48383
wandb:   
wandb:   uc_action_loss (go_stanford_test) 46.92104
wandb:   uc_action_loss (train) 40.63201
wandb:   uc_action_waypts_cos_sim (go_stanford_test) 0.38977
wandb:   uc_action_waypts_cos_sim (train) 0.38033
wandb:   uc_multi_action_waypts_cos_sim (go_stanford_test) 0.44431
wandb:   uc_multi_action_waypts_cos_sim (train) 0.42414
wandb:   
wandb:   lr 5e-05                              # 学习率(预热阶段)
wandb:   total_loss 1.23069

# ========== 同步完成 ==========
wandb: Synced earnest-fog-5: https://wandb.ai/mvasl/nomad/runs/ctu0lccn
wandb: Synced 6 W&B file(s), 24 media file(s), 0 artifact file(s) and 1 other file(s)
wandb: Find logs at: ./wandb/run-20251201_164322-ctu0lccn/logs
```

**W&B可视化说明：**
- 📊 **ASCII图表**: `▁▂▃▄▅▆▇█` 表示指标随时间的变化趋势
  - `▁` = 最低值
  - `█` = 最高值
- 🔍 **diffusion_loss**: 扩散模型的总体损失，显示训练过程中的波动
- 📈 **dist_loss**: 距离预测损失，从高到低有明显变化
- 🎯 **lr**: 学习率保持在底部（预热中）

**模型保存位置：**
```
logs/nomad/nomad-sample_2025_12_01_16_43_21/
├── ema_latest.pth          # 指数移动平均模型(推理用)
└── 0.pth                   # Epoch 0 检查点
```

---

#### Weights & Biases (WandB) 指标详解

训练过程中，WandB会实时记录和可视化各种指标。基于实际训练输出，以下是每个指标的详细含义：

##### 📊 **1. 核心损失指标 (Core Loss Metrics)**

NoMaD模型记录了**两种训练模式**的损失：

**A. 无目标条件 (Unconditioned - `uc_*`)** 

模型在**没有明确目标图像**的情况下预测轨迹（探索模式）

| 指标名称 | 含义 | 初始值 | 期望最终值 |
|---------|------|--------|-----------|
| `uc_action_loss` | 无目标条件下的动作预测损失 | ~45-50 | < 10 (100 epochs后) |
| `uc_action_waypts_cos_sim` | 无目标条件下路标点余弦相似度 | ~0.35-0.40 | > 0.90 |
| `uc_multi_action_waypts_cos_sim` | 无目标多模态轨迹相似度 | ~0.40-0.45 | > 0.85 |

**B. 有目标条件 (Goal Conditioned - `gc_*`)**

模型接收**目标图像**作为条件来预测导航轨迹

| 指标名称 | 含义 | 初始值 | 期望最终值 |
|---------|------|--------|-----------|
| `gc_action_loss` | 有目标条件下的动作预测损失 | ~43-48 | < 8 (100 epochs后) |
| `gc_dist_loss` | 到达目标的距离预测损失 | ~150-155 | < 50 |
| `gc_action_waypts_cos_sim` | 有目标条件下路标点余弦相似度 | ~0.40-0.55 | > 0.92 |
| `gc_multi_action_waypts_cos_sim` | 有目标多模态轨迹相似度 | ~0.43-0.60 | > 0.88 |

**C. 扩散模型总体损失**

| 指标名称 | 含义 | Epoch 0 | 期望最终值 |
|---------|------|---------|-----------|
| `diffusion_loss` | 扩散模型噪声预测损失 | ~1.2 | < 0.3 |
| `dist_loss` | 距离预测分支损失 | ~140-155 | < 50 |
| `total_loss` | 总损失 (扩散损失 + 距离损失) | ~1.23 | < 0.5 |

**关键术语解释:**

- **`uc` (Unconditioned)**: 
  - 无目标条件，训练模型的**探索能力**
  - 场景：机器人在不知道目标位置时如何移动
  - 通过 `goal_mask_prob=0.5` 控制，50%的训练样本会掩盖目标

- **`gc` (Goal Conditioned)**:
  - 有目标条件，训练模型的**导航能力**
  - 场景：机器人知道目标图像，规划到达路径
  - 这是部署时的主要使用模式

- **`100pt moving_avg`**: 
  - 最近100个批次的移动平均值
  - 用于平滑训练过程中的噪声波动

- **`avg`**: 
  - 从epoch开始到当前的全局平均值


##### 📈 **2. 评估指标 (Evaluation Metrics)**

在测试集上评估时，WandB还记录了不同掩码策略下的扩散损失：

| 指标名称 | 含义 | 用途 |
|---------|------|------|
| `diffusion_eval_loss (goal masking)` | 使用目标掩码的评估损失 | 评估探索能力 |
| `diffusion_eval_loss (no masking)` | 不使用掩码的评估损失 | 评估纯导航能力 |
| `diffusion_eval_loss (random masking)` | 随机掩码的评估损失 | 混合能力评估 |

**性能指标:**

| 指标 | 含义 | 初始值 (Epoch 0) | 优秀值 (收敛后) |
|------|------|-----------------|----------------|
| `action_waypts_cos_sim` | 路标点方向相似度 | 0.38-0.40 | > 0.95 |
| `multi_action_waypts_cos_sim` | 多模态轨迹相似度 | 0.42-0.44 | > 0.90 |

**余弦相似度解读:**

余弦相似度衡量预测轨迹与真实轨迹的**方向一致性**：

- **1.0**: 完全一致（完美预测）
- **0.9-0.95**: 优秀（方向基本正确）
- **0.7-0.9**: 良好（大致方向正确）
- **0.4-0.7**: 一般（有明显偏差）
- **< 0.4**: 较差（方向错误）

初始epoch的0.38-0.44是**正常的**，随着训练应逐步提升至0.90+

---

### 开始训练

#### 步骤1：启动训练

```bash
cd ~/visualnav-transformer/train

# 激活训练环境
conda activate nomad_train

# 从头开始训练
python train.py -c config/nomad.yaml
```

#### 步骤2：从检查点恢复训练（可选）

如果训练中断，可以从检查点恢复：

```bash
# 1. 在 nomad.yaml 中添加：
load_run: nomad/nomad_2023_11_25_10_30_00

# 2. 确保检查点存在：
ls logs/nomad/nomad_2023_11_25_10_30_00/latest.pth

# 3. 重新运行训练
python train.py -c config/nomad.yaml
```

#### 步骤3：使用预训练模型（快速开始）

如果你想跳过训练，直接使用官方预训练模型：

```bash
# 1. 下载预训练权重
mkdir -p ../deployment/model_weights
cd ../deployment/model_weights

# 从Google Drive下载 nomad.pth
# 链接：https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing

# 2. 验证文件
ls -lh nomad.pth
# 应显示 ~100-200MB 的文件
```

---

### 训练监控

#### 使用W&B监控

访问 `https://wandb.ai/your_username/nomad` 查看：
- **Loss曲线**: `uc_action_loss`, `gc_action_loss`, `gc_dist_loss`
- **学习率变化**: 余弦退火曲线
- **预测轨迹可视化**: 观察、目标、预测轨迹对比
- **余弦相似度**: 轨迹方向一致性
- **GPU使用率**: 显存和计算利用率

**W&B关键指标解读**:

| 指标 | 含义 | 理想值 |
|------|------|--------|
| `uc_action_loss` | 无目标动作损失 | < 0.05 |
| `gc_action_loss` | 有目标动作损失 | < 0.03 |
| `gc_dist_loss` | 距离预测损失 | < 2.0 |
| `uc_action_waypts_cos_sim` | 路标点方向相似度 | > 0.9 |
| `gc_action_waypts_cos_sim` | 有目标方向相似度 | > 0.95 |

#### 本地日志

```bash
# 查看训练日志
tail -f logs/nomad/<run_name>/train.log

# 查看检查点
ls -lh logs/nomad/<run_name>/*.pth

# 查看可视化结果
ls logs/nomad/<run_name>/visualize/*/epoch*/
```

---

### 训练时间和资源估计

#### 硬件需求

| GPU型号 | 显存 | 批次大小 | 每轮时间 | 100轮总时间 |
|---------|------|---------|---------|-------------|
| RTX 3060 | 12GB | 64 | ~40分钟 | ~67小时 |
| RTX 3090 | 24GB | 256 | ~20分钟 | ~33小时 |
| RTX 4090 | 24GB | 256 | ~15分钟 | ~25小时 |
| A100 | 40GB | 512 | ~10分钟 | ~17小时 |

#### 存储需求

- **数据集**: 50-200GB (取决于数据集数量)
- **检查点**: ~200MB/epoch × 100 = 20GB
- **日志和可视化**: ~5GB
- **总计**: ~100-250GB

#### 训练阶段时间分配

```
总训练时间 (以RTX 3090为例, 100 epochs):
├── Warmup (4 epochs): ~1.5小时
├── 主训练 (96 epochs): ~31小时
└── 评估和保存: ~0.5小时

单个Epoch细分:
├── 数据加载: 10%
├── 前向传播: 40%
├── 反向传播: 35%
├── 优化器更新: 10%
└── 日志和可视化: 5%
```

### 评估训练结果

```bash
# 训练完成后，模型保存在：
ls logs/nomad/<run_name>/

# 文件包括：
# - latest.pth          # 最新检查点
# - best.pth            # 最佳模型（验证loss最低）
# - config.yaml         # 训练配置
# - metrics.json        # 训练指标
```
---


## 机器人部署

### 部署环境准备

#### 第一步：设置部署环境（在机器人上）

```bash
# 如果是新机器，克隆仓库
git clone https://github.com/robodhruv/visualnav-transformer.git
cd visualnav-transformer

# 创建部署环境
conda env create -f deployment/deployment_environment.yaml
conda activate vint_deployment

# 安装依赖
pip install -e train/

# 安装 diffusion_policy
cd ..
git clone https://github.com/real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
cd visualnav-transformer
```

#### 第二步：安装ROS依赖（Ubuntu 20.04）

```bash
# 安装ROS Noetic
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
sudo apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654
sudo apt update
sudo apt install ros-noetic-desktop-full

# 安装ROS包
sudo apt-get install ros-noetic-usb-cam ros-noetic-joy

# 安装Kobuki驱动（LoCoBot）
sudo apt install ros-noetic-kobuki-*

# 安装tmux
sudo apt install tmux

# 初始化ROS
echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### 第三步：配置机器人参数

编辑 `deployment/config/robot.yaml`：

```yaml
# LoCoBot配置示例
max_v: 0.5                    # 最大线速度 (m/s)
max_w: 1.0                    # 最大角速度 (rad/s)

# ROS话题
joy_topic: "/joy"
vel_teleop_topic: "/cmd_vel_teleop"
vel_navi_topic: "/cmd_vel_navi"
waypoint_topic: "/waypoint"

# 控制参数
linear_vel_scale: 0.5
angular_vel_scale: 0.5
```

编辑 `deployment/config/camera.yaml`：

```yaml
# 摄像头参数（根据实际摄像头调整）
image_width: 640
image_height: 480
fps: 30
camera_index: 0               # USB摄像头索引
```

编辑 `deployment/config/joystick.yaml`：

```yaml
# 手柄按钮映射（Logitech F710示例）
deadman_button: 4             # LB按钮
enable_button: 5              # RB按钮
```

#### 第四步：配置模型

编辑 `deployment/config/models.yaml`：

```yaml
nomad: 
  config_path: "../../train/config/nomad.yaml"
  ckpt_path: "../model_weights/nomad.pth"
```

确保模型权重文件存在：

```bash
ls -lh deployment/model_weights/nomad.pth
```

### 收集拓扑地图

拓扑地图是机器人导航的参考轨迹。

#### 第一步：记录ROS Bag

```bash
cd deployment/src

# 启动记录脚本
./record_bag.sh my_office_map

# 这会打开3个tmux窗口：
# 1. roslaunch (摄像头、手柄、机器人驱动)
# 2. joy_teleop.py (手柄控制)
# 3. rosbag record (等待你按Enter开始录制)
```

操作步骤：
1. 在第3个窗口按Enter开始录制
2. 使用手柄遥控机器人沿期望路径行驶
3. 到达终点后，Ctrl+C停止录制
4. 终止所有tmux窗口

#### 第二步：生成拓扑地图

```bash
# 从ROS bag创建拓扑地图
./create_topomap.sh my_office_map my_office_map.bag

# 这会打开3个tmux窗口：
# 1. roscore
# 2. create_topomap.py (等待提示后按Enter)
# 3. rosbag play (等待提示)

# 操作：
# 1. 等待第2个窗口显示 "Waiting for messages..."
# 2. 在第3个窗口按Enter播放bag
# 3. 当bag播放完成后，Ctrl+C停止create_topomap.py
# 4. 终止tmux会话
```

验证生成的地图：

```bash
ls ../topomaps/images/my_office_map/
# 应看到：0.jpg, 1.jpg, 2.jpg, ..., N.jpg

# 查看图像数量
ls ../topomaps/images/my_office_map/*.jpg | wc -l
# 根据路径长度，应有10-100张图像
```

### 运行导航

#### 导航到目标

```bash
cd deployment/src

# 启动导航
./navigate.sh "--model nomad --dir my_office_map"

# 这会打开4个tmux窗口：
# 1. roslaunch (硬件驱动)
# 2. navigate.py (NoMaD推理)
# 3. joy_teleop.py (手柄控制，可随时接管)
# 4. pd_controller.py (PD控制器)
```

导航流程：
1. 机器人启动后，会显示当前位置和目标
2. 使用手柄的方向键选择目标节点（0到N）
3. 按确认键，机器人开始自主导航
4. 导航过程中可随时用手柄接管
5. 到达目标后，可选择新目标或停止

#### 探索模式

NoMaD特有的探索模式（无需预定义目标）：

```bash
cd deployment/src

# 启动探索模式
./explore.sh "--model nomad"

# 窗口布局与导航相同
# 机器人会自主探索环境，避开障碍物
```

探索参数调整（在 `explore.py` 中）：

```python
# 探索距离范围
EXPLORATION_HORIZON = 5.0  # 米

# 随机性
TEMPERATURE = 1.0          # 越大越随机
```

### 部署调试

#### 检查摄像头

```bash
# 测试摄像头
rosrun usb_cam usb_cam_node

# 在另一个终端查看图像
rosrun image_view image_view image:=/usb_cam/image_raw

# 应该能看到实时摄像头画面
```

#### 检查手柄

```bash
# 测试手柄
rosrun joy joy_node

# 在另一个终端查看输入
rostopic echo /joy

# 按下手柄按钮，应看到数据变化
```

#### 调试导航

```bash
# 查看导航日志
# 在navigate.py窗口查看输出

# 常见输出：
# - "Received observation" - 收到图像
# - "Goal: X/N" - 当前目标
# - "Action: [x, y, theta]" - 预测的动作
# - "Distance to goal: X.Xm" - 到目标的距离
```

#### 性能优化

如果导航速度慢：

```python
# 在 navigate.py 中调整：

# 1. 降低扩散迭代次数（牺牲精度换速度）
NUM_DIFFUSION_ITERS = 5  # 默认10

# 2. 调整图像大小
IMAGE_SIZE = [64, 64]    # 默认[96, 96]

# 3. 使用GPU加速（Jetson）
device = torch.device("cuda:0")
```

### 安全注意事项

⚠️ **重要安全提示**：

1. **始终准备紧急停止**：手柄的deadman按钮松开即停止
2. **测试区域**：在开阔、安全的区域测试
3. **速度限制**：初次测试时降低 `max_v` 和 `max_w`
4. **监控电池**：确保机器人电量充足
5. **避开楼梯**：模型无法识别楼梯等危险区域

---

## 常见问题

### 环境相关

**Q0: diffusers 导入失败 - ImportError: cannot import name 'cached_download'**

这是最常见的环境问题，由于 `diffusers==0.11.1` 与新版本 `huggingface_hub` 不兼容导致。

```bash
# 症状：
python -c "import diffusers"
# ImportError: cannot import name 'cached_download' from 'huggingface_hub'

# 解决方案1：安装兼容版本的 huggingface_hub
pip uninstall huggingface_hub -y
pip install huggingface_hub==0.10.1

# 解决方案2：如果方案1不行，尝试更早的版本
pip install huggingface_hub==0.8.1

# 解决方案3：完全重建环境（推荐用于严重问题）
conda deactivate
conda env remove -n nomad_train
conda env create -f train/train_environment.yml
conda activate nomad_train

# 手动安装正确版本
pip install diffusers==0.11.1
pip install huggingface_hub==0.10.1

# 验证
python -c "import diffusers; print('Success! diffusers version:', diffusers.__version__)"
python -c "import huggingface_hub; print('huggingface_hub version:', huggingface_hub.__version__)"
```

**推荐的安装顺序**（避免依赖冲突）：

```bash
# 在激活 nomad_train 环境后
pip install huggingface_hub==0.10.1
pip install diffusers==0.11.1
pip install transformers==4.25.1  # 如果需要

# 然后再安装其他包
pip install -e train/
cd ../diffusion_policy
pip install -e .
```

### 数据相关

**Q1: 训练报错 - ValueError: setting an array element with a sequence / inhomogeneous shape**

这是 **GoStanford2 数据集最常见的问题**！

```bash
# 症状：
ValueError: setting an array element with a sequence. 
The requested array has an inhomogeneous shape after 2 dimensions. 
The detected shape was (3, 3) + inhomogeneous part.
```

**原因**：数据集中 `traj_data.pkl` 的数据类型是 `object` 而不是 `float64`。

**解决方案**：参考本指南 **"数据准备 → 步骤1 → 数据类型修复"** 章节，使用提供的脚本批量转换数据类型。

快速修复：
```bash
cd ~/visualnav-transformer

# 下载修复脚本（从指南复制）
# 运行批量转换
python fix_go_stanford_dtype.py

# 验证修复
python check_dtype.py
# 应显示: position dtype: float64, yaw dtype: float64
```

**关键步骤**：
1. 检查数据类型：`position.dtype` 和 `yaw.dtype`
2. 如果是 `object`，运行修复脚本
3. 脚本会自动备份原文件（`.backup`）
4. 转换所有轨迹为 `float64` 类型
5. 验证修复成功后再开始训练

**详细修复步骤**：请查看指南第 "数据准备" 章节中的 "数据类型修复" 部分。

---

### 训练相关

**Q2: 训练时显存不足 (CUDA Out of Memory)**

```bash
# 解决方案：
# 1. 减小批次大小
batch_size: 64  # 在 nomad.yaml 中

# 2. 减少worker数量
num_workers: 4

# 3. 使用梯度累积（在train.py中添加）
accumulation_steps = 4
```

**Q3: 训练loss不下降**

```bash
# 检查：
# 1. 数据是否正确加载
# 2. 学习率是否合适
# 3. 是否启用了warmup

# 解决方案：
# - 降低学习率：lr: 5e-5
# - 延长warmup：warmup_epochs: 8
# - 检查数据标准化
```

**Q3: W&B登录失败**

```bash
# 离线模式
export WANDB_MODE=offline

# 或在配置中禁用
use_wandb: False
```

### 数据相关

**Q4: 找不到数据集**

```bash
# 检查路径
python -c "import os; print(os.path.exists('/home/<username>/nomad_dataset/recon'))"

# 修正路径
# 在 nomad.yaml 中使用绝对路径
data_folder: /full/path/to/nomad_dataset/recon
```

**Q5: rosbag处理失败**

```bash
# 安装rosbag依赖
pip install --extra-index-url https://rospypi.github.io/simple/ rosbag roslz4

# 检查bag文件
rosbag info your_file.bag

# 确认话题名称
# 在 process_bags_config.yaml 中修改
```

### 部署相关

**Q6: 摄像头无法打开**

```bash
# 检查摄像头设备
ls /dev/video*

# 测试摄像头
ffplay /dev/video0

# 修改摄像头索引
# 在 camera.yaml 中
camera_index: 0  # 尝试 0, 1, 2...
```

**Q7: 手柄无法识别**

```bash
# 检查手柄连接
ls /dev/input/js*

# 测试手柄
jstest /dev/input/js0

# 修改权限
sudo chmod a+rw /dev/input/js0
```

**Q8: 机器人不移动**

```bash
# 检查话题
rostopic list

# 查看速度命令
rostopic echo /cmd_vel

# 检查机器人驱动
rosnode list | grep kobuki

# 手动发送速度命令测试
rostopic pub /cmd_vel geometry_msgs/Twist "linear:
  x: 0.1
  y: 0.0
  z: 0.0
angular:
  x: 0.0
  y: 0.0
  z: 0.0"
```

**Q9: 导航不稳定/抖动**

```python
# 调整PD控制器参数（pd_controller.py）
KP_LINEAR = 0.5   # 线速度比例增益
KP_ANGULAR = 1.0  # 角速度比例增益
KD_LINEAR = 0.1   # 线速度微分增益
KD_ANGULAR = 0.2  # 角速度微分增益

# 或增加平滑
SMOOTH_FACTOR = 0.3  # 动作平滑系数
```

### 性能相关

**Q10: 推理速度慢**

```bash
# Jetson优化：
# 1. 使用TensorRT
pip install torch2trt

# 2. 降低图像分辨率
image_size: [64, 64]

# 3. 减少扩散迭代
num_diffusion_iters: 5

# 4. 使用FP16
model.half()
```

---

## 验证清单

### 训练验证

- [ ] 环境创建成功：`conda activate nomad_train`
- [ ] CUDA可用：`python -c "import torch; print(torch.cuda.is_available())"`
- [ ] **关键**：`diffusers`能正常导入：`python -c "import diffusers; from diffusers import DiffusionPipeline"`
- [ ] `huggingface_hub`版本正确：`pip show huggingface_hub` 显示 0.10.1 或 0.8.1
- [ ] 数据集下载完成
- [ ] 数据处理完成，生成`.jpg`和`.pkl`文件
- [ ] 数据分割创建，`traj_names.txt`存在
- [ ] `data_config.yaml`配置正确
- [ ] `nomad.yaml`路径更新
- [ ] W&B配置（可选）
- [ ] 训练启动无错误
- [ ] Loss正常下降
- [ ] 模型检查点保存：`logs/nomad/<run_name>/latest.pth`

### 部署验证

- [ ] 部署环境创建：`conda activate vint_deployment`
- [ ] ROS安装：`roscore`能启动
- [ ] 摄像头工作：`rostopic echo /usb_cam/image_raw`
- [ ] 手柄连接：`rostopic echo /joy`
- [ ] 机器人驱动：`rostopic list`显示相关话题
- [ ] 模型权重存在：`deployment/model_weights/nomad.pth`
- [ ] 配置文件更新：`robot.yaml`, `camera.yaml`, `models.yaml`
- [ ] ROS bag录制成功
- [ ] 拓扑地图生成：`topomaps/images/<map_name>/`有图像
- [ ] 导航脚本启动无错误
- [ ] 机器人能够移动
- [ ] 能够到达目标
- [ ] 探索模式工作（NoMaD特有）

---

## 进阶使用

### 自定义数据集训练

```bash
# 1. 准备你的数据（rosbag或图像序列）
# 2. 处理数据
python process_bags.py --dataset-name my_dataset \
    --input-dir /path/to/rosbags \
    --output-dir ~/nomad_dataset/my_dataset

# 3. 创建分割
python data_split.py \
    --data-dir ~/nomad_dataset/my_dataset \
    --dataset-name my_dataset \
    --data-splits-dir ./vint_train/data/data_splits \
    --split 0.8

# 4. 更新data_config.yaml
# my_dataset:
#     metric_waypoints_distance: 0.5

# 5. 更新nomad.yaml
# datasets:
#   my_dataset:
#     data_folder: ~/nomad_dataset/my_dataset
#     train: data/data_splits/my_dataset/train/
#     test: data/data_splits/my_dataset/test/
#     end_slack: 3
#     goals_per_obs: 1
#     negative_mining: True

# 6. 训练
python train.py -c config/nomad.yaml
```

### 微调预训练模型

```yaml
# 在 nomad.yaml 中添加
load_run: nomad/nomad_pretrained

# 降低学习率进行微调
lr: 1e-5
epochs: 20

# 只在你的数据集上训练
datasets:
  my_custom_environment:
    # ... 配置
```

### 多机器人适配

```yaml
# 为不同机器人创建配置文件
# deployment/config/robot_jackal.yaml
# deployment/config/robot_turtlebot.yaml

# 在导航时指定
./navigate.sh "--model nomad --dir map --robot-config robot_jackal.yaml"
```

### 性能分析

```bash
# 使用PyTorch Profiler
python train.py -c config/nomad.yaml --profile

# 分析推理时间
python -c "
import torch
import time
from vint_train.models.nomad.nomad import NoMaD

model = NoMaD(...)
model.eval()

# 预热
for _ in range(10):
    model(dummy_input)

# 计时
start = time.time()
for _ in range(100):
    model(dummy_input)
print(f'Avg inference time: {(time.time()-start)/100*1000:.2f}ms')
"
```

---

## 资源链接

### 官方资源
- **项目主页**: https://general-navigation-models.github.io/nomad/
- **代码仓库**: https://github.com/robodhruv/visualnav-transformer
- **论文**: https://arxiv.org/abs/2310.xxxx
- **预训练模型**: [Google Drive](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

### 数据集
- **RECON**: https://sites.google.com/view/recon-robot/dataset
- **TartanDrive**: https://github.com/castacks/tartan_drive
- **SCAND**: https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html
- **GoStanford2**: https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing
- **SACSoN**: https://sites.google.com/view/sacson-review/huron-dataset

### 相关项目
- **GNM**: https://sites.google.com/view/drive-any-robot
- **ViNT**: https://general-navigation-models.github.io/vint/
- **Diffusion Policy**: https://github.com/real-stanford/diffusion_policy

### 社区支持
- **GitHub Issues**: https://github.com/robodhruv/visualnav-transformer/issues
- **联系作者**: shah@cs.berkeley.edu

---

## 引用

如果你在研究中使用NoMaD，请引用：

```bibtex
@article{sridhar2023nomad,
  author  = {Ajay Sridhar and Dhruv Shah and Catherine Glossop and Sergey Levine},
  title   = {{NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration}},
  journal = {arXiv pre-print},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.xxxx}
}
```

---

## 更新日志

- **2025-11-25**: 创建完整复现指南
- 包含训练、数据处理、部署全流程
- 添加常见问题和故障排除
- 提供详细的验证清单

---

## 许可证

本项目遵循原始仓库的许可证。详见 [LICENSE](LICENSE) 文件。

---

**祝你复现顺利！如有问题，欢迎提Issue。** 🚀
