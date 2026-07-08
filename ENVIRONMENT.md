# Environment Notes

更新日期：2026-07-08

## 1. 当前结论

当前这台 Windows 本机上**不存在一个已经配置好、可以直接跑通本项目核心代码 import 的环境**。

更具体地说：

- `base` 环境是 Python 3.12.3，缺 `torch`，并且 `numpy==2.2.6` 与已有 `matplotlib/h5py` 二进制包不兼容。
- `RPF` 环境能 import `torch/timm/opencv`，但缺 `diffusers`、`diffusion_policy`、`efficientnet_pytorch`、`vit_pytorch`、`pybullet`、`mujoco` 等关键包。
- `prp` 环境最接近 RA-L 离线/仿真需求：有 `torch 2.4.1`、CUDA、`mujoco`、`onnxruntime`，但仍缺 `diffusers`、`diffusion_policy`、`efficientnet_pytorch`、`vit_pytorch`、`pybullet`、`lmdb`、`h5py` 等。
- `cv` 环境有 NumPy/MKL DLL import 问题，不适合作为本项目环境。
- `part_hoe` 环境有 torch/opencv，但缺 NoMaD/diffusion 关键依赖。
- `vlm-test` 环境缺 torch/opencv/matplotlib/yaml 等基础依赖。

此外，仓库当前没有 `diffusion_policy/` 目录。NoMaD 相关核心代码需要：

```python
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
```

所以即使把普通 pip 包补齐，如果不安装 `real-stanford/diffusion_policy`，NoMaD 推理和训练代码仍然无法 import。

## 2. 现有环境配置文件

仓库里已有这些配置入口：

```text
train/train_environment.yml
deployment/deployment_environment.yaml
deployment/src/deployment_environment.yml
scripts/requirements.txt
scripts/tooling/env_check.py
```

它们不是同一个目标：

| 文件 | 用途 | 当前问题 |
|---|---|---|
| `train/train_environment.yml` | 原始 ViNT/NoMaD 训练环境，Python 3.8.5，CUDA 10/旧依赖。 | 偏旧，适合复现原项目，不覆盖当前 MuJoCo/Lite3 分析脚本。 |
| `deployment/deployment_environment.yaml` | 原始 ROS 部署环境。 | 面向 Ubuntu/ROS，不适合 Windows 一键跑通。 |
| `deployment/src/deployment_environment.yml` | 部署环境旧副本。 | 内容更少，不建议作为主入口。 |
| `scripts/requirements.txt` | 中期冲刺后整理的实验脚本依赖。 | 比较贴近当前离线实验，但缺 `diffusion_policy`、`mujoco`、`pybullet`、`onnxruntime` 等运行部分入口需要的包。 |
| `scripts/tooling/env_check.py` | 项目结构、torch、数据集、权重检查。 | 检查较宽松，不能证明所有核心 import 都通过。 |

## 3. 不建议追求“一个环境跑全仓库”

本仓库混合了几类代码：

1. NoMaD/ViNT 训练和离线推理。
2. Lite3 MuJoCo 仿真。
3. Lite3 真机部署桥。
4. 原始 ROS/LoCoBot 部署代码。
5. TartanDrive/RECON 数据集工具。
6. 第三方 Lite3 SDK、Eigen、gamepad、MotionSDK 等 vendor 代码。

因此“全仓库所有 `.py` 都能 import”不是一个合理目标。全仓库静态扫描会碰到：

- ROS 包：`rospy`, `rosbag`, `geometry_msgs`, `sensor_msgs`, `std_msgs`。
- 数据集工具包：`ackermann_msgs`, `grid_map_msgs`, `pytorch3d`, `minio`, `tslearn`。
- 第三方 SDK 调试脚本：`gdb`, `lldb`。
- Python 2 风格第三方脚本。

更合理的目标是拆成两个环境：

1. **RA-L 离线/仿真环境**：用于当前投稿需要的 offline Pareto、NoMaD inference、MuJoCo/Lite3 simulation、图表统计。
2. **Ubuntu/ROS 真机部署环境**：用于原始 ROS deployment 和真实机器人节点，不要求在 Windows 本机完整 import。

## 4. 推荐环境：`nomad_ral`

建议新建一个专门环境，不要继续修补 `base`。

### 4.1 Windows/离线仿真推荐流程

```powershell
conda create -n nomad_ral python=3.10 -y
conda activate nomad_ral
```

安装 PyTorch。按当前机器 CUDA 选择；如果用 CUDA 12.1：

```powershell
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
```

安装项目 Python 依赖：

```powershell
pip install -r scripts/requirements.txt
pip install mujoco pybullet onnx onnxruntime colorama urdfpy einops
pip install -e train/
```

安装 NoMaD 需要的 `diffusion_policy`：

```powershell
git clone https://github.com/real-stanford/diffusion_policy.git
pip install -e diffusion_policy/
```

如果不希望把外部仓库放在项目根目录，可以放到 `third_party/diffusion_policy/`，但需要保持 `pip install -e` 后能 import：

```powershell
python -c "from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D; print('diffusion_policy OK')"
```

### 4.2 推荐锁定版本

当前项目对这些版本比较敏感：

```text
diffusers==0.11.1
huggingface-hub==0.10.1
opencv-python==4.6.0.66
tqdm==4.64.0
h5py==3.6.0
numpy<2.0
```

不要使用 `numpy>=2`，当前 `base` 环境的 `matplotlib/h5py` 问题就是一个警告。

## 5. 验证命令

创建环境后先做基础检查：

```powershell
python scripts/tooling/env_check.py
```

再做核心 import 检查：

```powershell
python -c "import torch, torchvision, cv2, numpy, yaml, diffusers; print('basic deps OK')"
python -c "from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D; print('diffusion_policy OK')"
python -c "from vint_train.models.nomad.nomad import NoMaD; print('vint_train OK')"
python -c "import scripts.shared.nomad_inference; print('nomad_inference OK')"
python -c "import scripts.simulation.nomad_mujoco_lite3_state_machine; print('mujoco state machine OK')"
```

如果要跑 PyBullet/Lite3 仿真，再检查：

```powershell
python -c "import mujoco, pybullet, onnxruntime; print('simulation deps OK')"
```

ROS 部署不要用 Windows 本机作为验收标准。ROS 相关 import 应在 Ubuntu/ROS Noetic 环境中检查：

```bash
python -c "import rospy, rosbag, geometry_msgs, sensor_msgs, std_msgs; print('ROS deps OK')"
```

## 6. 当前本机环境体检摘要

| 环境 | Python | torch | CUDA | 核心问题 |
|---|---:|---|---|---|
| `base` | 3.12.3 | 缺失 | 否 | 缺 torch；NumPy 2 与 matplotlib/h5py 不兼容。 |
| `RPF` | 3.9.23 | 2.5.0+cpu | 否 | 缺 diffusion/NoMaD 依赖和仿真依赖。 |
| `cv` | 3.7.12 | 1.13.1 | 未验证 | NumPy/MKL DLL import 失败。 |
| `part_hoe` | 3.7.16 | 1.13.1+cpu | 否 | 缺 diffusion/NoMaD 依赖。 |
| `prp` | 3.12.3 | 2.4.1 | 是 | 最接近，但缺 diffusion/NoMaD/PyBullet/数据依赖；Python 3.12 对旧包风险较高。 |
| `vlm-test` | 3.10.19 | 缺失 | 否 | 缺大部分项目依赖。 |

## 7. 建议后续清理

建议下一步做三件事：

1. 新建 `nomad_ral`，不要在 `base` 上修。
2. 把 `scripts/requirements.txt` 升级为当前 RA-L 实验依赖清单，或新增 `requirements-ral.txt`。
3. 在 `scripts/tooling/` 增加一个更严格的 import checker，分组检查：
   - offline/training
   - MuJoCo simulation
   - real robot bridge
   - ROS legacy deployment

当前最重要的 blocker 是：

```text
diffusion_policy 不在仓库中，也未安装到任何本机环境。
```

这个不解决，NoMaD 推理和训练入口都无法完整 import。
