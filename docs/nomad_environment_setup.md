# NoMaD Environment Setup

本文档整理当前仓库在 `nomad_train` 环境中实际验证过的依赖，并给出标准 conda `environment.yaml` 配置。原则是以当前可运行环境为准，而不是完全照搬上游 README 中较旧的 `train/train_environment.yml`。

## 环境文件

仓库根目录提供两个 conda 环境文件：

- `environment.yaml`：训练、离线推理和离线实验环境。
- `environment.sim.yaml`：在 `environment.yaml` 基础上增加 MuJoCo/PyBullet/ONNX Runtime 仿真实验依赖。

创建环境前，先确保 `diffusion_policy` 在本仓库同级目录：

```bash
cd /home/yifei/codespace
git clone https://github.com/real-stanford/diffusion_policy.git
```

创建训练/离线实验环境：

```bash
cd /home/yifei/codespace/visualnav-transformer
conda env create -f environment.yaml
conda activate nomad_train_repro
```

创建 MuJoCo 仿真实验环境：

```bash
cd /home/yifei/codespace/visualnav-transformer
conda env create -f environment.sim.yaml
conda activate nomad_train_sim
```

## 当前已验证的基线

检查对象：现有 conda 环境 `nomad_train`。

关键版本：

| Package | Version |
| --- | --- |
| Python | 3.8.5 |
| torch | 2.4.1+cu121 |
| torchvision | 0.19.1+cu121 |
| diffusers | 0.11.1 |
| huggingface-hub | 0.10.1 |
| numpy | 1.24.3 |
| opencv-python | 4.6.0.66 |
| timm | 1.0.26 |
| wandb | 0.12.18 |
| lmdb | 1.7.5 |
| h5py | 3.6.0 |
| Pillow | 10.4.0 |
| matplotlib | 3.7.2 |

已通过顶层 import 检查的范围：

- 训练核心：`train/train.py`、`train/vint_train/data`、`train/vint_train/models`、`train/vint_train/training`
- 训练扩展：`scripts/training/train_dinov2.py`、`scripts/training/train_backbone_suite.py`
- 离线/实时推理：`scripts/shared/nomad_inference.py`、`scripts/analysis/offline_inference.py`、`scripts/analysis/realtime_inference.py`
- 离线实验：`scripts/experiments/ablation_experiment.py`、`cfg_*`、`ddim_*`、`tts_*`、encoder 对比与消融脚本
- 工具：`scripts/tooling/env_check.py`、`path_audit.py`、`project_paths.py`

## 依赖分类

### 1. 当前正在使用的基础依赖

这些包构成当前 `nomad_train` 的可运行基础，不建议随意升级：

- Python 与打包：`python==3.8.5`、`pip==24.2`、`setuptools==75.1.0`、`wheel==0.44.0`
- PyTorch：`torch==2.4.1`、`torchvision==0.19.1`，CUDA 12.1 wheel
- 基础数值/图像：`numpy==1.24.3`、`opencv-python==4.6.0.66`、`Pillow==10.4.0`、`matplotlib==3.7.2`
- 配置/工具：`PyYAML==6.0.3`、`tqdm==4.64.0`、`requests==2.32.4`、`gdown==5.2.0`
- 本项目包：`vint_train` editable install from `train/`
- 扩散策略包：`diffusion_policy` editable install

### 2. 训练、离线推理与实验依赖

训练主线和 NoMaD 离线实验需要：

- Diffusion：`diffusers==0.11.1`、`huggingface-hub==0.10.1`
- NoMaD/GNM/ViNT 模型：`efficientnet-pytorch==0.7.1`、`vit-pytorch==1.15.7`、`positional-encodings==6.0.3`
- 新视觉编码器实验：`timm==1.0.26`、`einops==0.8.1`
- 训练日志和缓存：`wandb==0.12.18`、`prettytable==3.11.0`、`lmdb==1.7.5`、`warmup-scheduler==0.3.2`
- 数据文件：`h5py==3.6.0`
- rosbag 数据处理：`rosbag==1.15.11`、`roslz4==1.14.3.post2`、`rospkg==1.6.0`、`std-msgs==0.5.13.post0`

说明：

- `diffusers==0.11.1` 需要较旧的 `huggingface-hub==0.10.1`，否则可能缺少旧 API。
- `wandb==0.12.18` 与 `protobuf==3.20.3` 搭配使用，避免新版 protobuf 兼容性问题。
- `train/check_train_config.py` 使用相对路径，从 `train/` 目录运行更符合它的假设。

### 3. MuJoCo / 仿真实验依赖

仿真脚本分两类：

- `scripts/simulation/nomad_mujoco_lite3_nav.py`
- `scripts/simulation/nomad_mujoco_lite3_state_machine.py`
- `scripts/experiments/mujoco_encoder_benchmark.py`
- `scripts/simulation/lite3_sim.py`

当前 `nomad_train` 顶层 import 检查显示：

- MuJoCo 相关脚本的部分顶层 import 能通过，但真正运行 MuJoCo 环境时需要 `mujoco` 和 `onnxruntime`。
- `scripts/simulation/lite3_sim.py` 直接 import `pybullet`，当前 `nomad_train` 没装，所以会失败。

因此仿真组建议安装：

- `mujoco==3.2.3`
- `onnxruntime==1.19.2`
- `pybullet==3.2.7`

这些包由 `environment.sim.yaml` 安装。环境文件会在仿真包之后继续固定 `numpy==1.24.3` 和 `protobuf==3.20.3`，避免 `onnxruntime` 解析依赖时升级到可能影响旧版 `wandb` 的 protobuf 5.x。MuJoCo 图形窗口还可能依赖系统 OpenGL/EGL/显卡驱动环境，环境文件只负责 Python 依赖。

## 未纳入主环境的依赖

以下包属于真机部署、ROS 消息、TartanDrive 历史工具或 RECON 可视化工具，不放进默认环境，避免扩大依赖面和版本冲突：

- 真机/ROS 消息：`sensor_msgs`、`geometry_msgs`、`nav_msgs`、`ackermann_msgs`、`grid_map_msgs`、`racepak`
- TartanDrive / wheeledsim：`gym`、`pandas`、`sklearn`、`tabulate`、`tslearn`、`pytorch3d`、`wheeledSim`、`wheeledRobots`
- RECON 可视化：`utm`、旧版 `pandas` 等
- 真机部署路径：`deployment/`、`scripts/deployment/` 中 real backend 相关依赖

如果后续要恢复这些历史数据集工具，建议单独建环境，而不是混入训练/离线实验环境。

## 验证命令

创建环境后可以运行：

```bash
conda activate nomad_train_repro
python -c "import torch; print(torch.cuda.is_available(), torch.version.cuda)"
python -c "import diffusers, huggingface_hub, cv2, timm, wandb, lmdb, h5py; import vint_train; print('core ok')"
python -c "from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D; print('diffusion_policy ok')"
```

仿真可选依赖验证：

```bash
python -c "import mujoco, onnxruntime, pybullet; print('sim ok')"
```

训练入口示例：

```bash
cd train
python train.py -c config/nomad.yaml
```

离线实验入口示例：

```bash
python scripts/analysis/offline_inference.py --help
python scripts/experiments/ddim_stat_experiment.py --help
```

MuJoCo 仿真入口示例：

```bash
python scripts/simulation/nomad_mujoco_lite3_nav.py --mode walk-test --no-gui
python scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode navigate --map easy --no-gui
```
