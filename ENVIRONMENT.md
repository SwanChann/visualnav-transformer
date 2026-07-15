# Environment Notes

更新日期：2026-07-15

## 1. 当前三环境权威划分

本项目固定使用三个环境，不再追求任一机器“跑通全仓库”。用户确认：完整数据集只存放在远程 RTX 4090 服务器，Windows 与 Ubuntu 不保存训练数据副本。

| 环境 | 已验证条件 | 当前职责 | 明确不做 | 当前状态 |
|---|---|---|---|---|
| Windows 原项目目录 | 论文、证据包、Git 治理、资产地图、静态 pre-training 合同与单元测试已存在；完整 NoMaD 训练依赖不齐；无训练数据 | 唯一人工/Git/论文集成入口；维护协议、任务、代码审查、小型结果和论文；形成供其他环境执行的冻结 commit/config | 不保存数据集；不做真实数据 adapter 验证、训练、跨数据集离线评估、仿真或真机 | 静态实现与测试完成；跨环境执行只认包含 4090 交接包的冻结 Git SHA |
| Ubuntu 自有电脑 | U00–U18、U09 v0.3、U10 v5 的离线/仿真执行与审计已完成 | 接收通过 4090 `OFFLINE-GATE` 的少量候选 checkpoint；执行新的非饱和闭环 `SIM-GATE`；返回原始 rollout、统计与日志 | 不重复 U00–U18；不保存训练数据；不做多数据集训练或需要完整数据的 LODO/corruption 离线评估 | 历史授权范围完成；等待 4090 晋级 checkpoint |
| 远程 RTX 4090 服务器 | 已核验 `agent/ubuntu-sim-handoff`、2x RTX 4090、`nomad_train`、Go Stanford 全量内容与冻结 `nomad.pth`；GPU 0 IMAGE-FORWARD 与两步 TRAIN-STEP/exact-resume 通过 | 数据盘点与合规登记、真实 dataset adapter 接线、manifest/split/leakage、B0 smoke、baseline/H0/H1 训练、IID/mixed/LODO/corruption 离线评估、显存与吞吐测量 | 不直接改论文结论；不把 forward/两步集成/smoke、loss 下降或单 seed 当方法结果；不绕过 Windows 权威入口 | Phase 0、Go Stanford 单域 pilot、随机初始化 image forward 与两步 plumbing 完成；RECON/HuRoN、旧训练目录 provenance 和 200-step B0 仍阻塞 |

执行链固定为：

```text
Windows 冻结代码/协议/配置
  -> 4090 DATA-PILOT -> adapter integration -> B0-SMOKE
  -> 4090 BASELINE-MATRIX -> H1-TRAIN -> OFFLINE-GATE
  -> Ubuntu SIM-GATE
  -> 目标设备/真机环境（尚未指定）
  -> Windows 证据回收、论文和 go/no-go
```

跨环境只交换 Git 版本、配置、manifest/split hash、checkpoint、日志和小型结果；原始/processed 数据不离开 4090。三端执行必须绑定完整 Git SHA、manifest SHA-256、split SHA-256、model-contract SHA-256 和 config SHA-256。

环境状态的证据边界：

- Windows 和 Ubuntu 状态已由当前仓库与 U00–U18 审计核验。
- 4090 Phase 0 已核验 Git、Python/CUDA/GPU/磁盘、Go Stanford 和冻结 `nomad.pth`。Go Stanford 内容级 pilot 通过，但 raw receipt、per-frame timestamp、pinned processor version 和独立物理标定仍缺失；142 GB 旧训练目录也未建立逐 checkpoint provenance。
- 目标设备与真机执行位置尚未确定，不能默认归入 Ubuntu 或 4090。

## 2. Windows 环境历史体检

以下内容保留 Windows 兼容性和依赖审计，但不再构成下一阶段“在 Windows 建完整训练环境”的要求。

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

## 3. 现有环境配置文件

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

## 4. 不建议追求“一个环境跑全仓库”

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

## 5. 历史可选环境：`nomad_ral`

仅当未来明确要求 Windows 重跑 NoMaD/仿真时再创建该环境；当前三环境分工不要求执行以下安装步骤。

建议新建一个专门环境，不要继续修补 `base`。

### 5.1 Windows/离线仿真历史推荐流程

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

### 5.2 推荐锁定版本

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

## 6. 历史环境验证命令

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

## 7. Windows 本机环境体检摘要

| 环境 | Python | torch | CUDA | 核心问题 |
|---|---:|---|---|---|
| `base` | 3.12.3 | 缺失 | 否 | 缺 torch；NumPy 2 与 matplotlib/h5py 不兼容。 |
| `RPF` | 3.9.23 | 2.5.0+cpu | 否 | 缺 diffusion/NoMaD 依赖和仿真依赖。 |
| `cv` | 3.7.12 | 1.13.1 | 未验证 | NumPy/MKL DLL import 失败。 |
| `part_hoe` | 3.7.16 | 1.13.1+cpu | 否 | 缺 diffusion/NoMaD 依赖。 |
| `prp` | 3.12.3 | 2.4.1 | 是 | 最接近，但缺 diffusion/NoMaD/PyBullet/数据依赖；Python 3.12 对旧包风险较高。 |
| `vlm-test` | 3.10.19 | 缺失 | 否 | 缺大部分项目依赖。 |

## 8. Windows 后续边界

Windows 当前只需：

1. 完成现有静态基础设施、环境职责文档和 handoff 的 diff 审核。
2. 获得用户授权后 commit/push，给 4090 一个明确的冻结 Git SHA。
3. 在 4090/Ubuntu 返回结果后做证据审查、资产地图和论文更新。

`diffusion_policy` 未安装在 Windows 仍是事实，但不是当前 blocker；真实训练环境应在 4090 核验和修复。除非职责再次改变，不在 Windows 新建完整训练/仿真环境。
