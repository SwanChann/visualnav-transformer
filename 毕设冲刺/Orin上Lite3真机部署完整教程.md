# Orin 上 Lite3 真机部署完整教程

## 1. 文档定位

本文档用于指导在 Jetson Orin 上部署 NoMaD 视觉导航，并与云深处 Lite3 真机联动。本文档只保留当前项目中已经具备代码支撑、命令可以直接对应到脚本、并且能够形成明确预期结果的内容。

整个部署分为两个阶段：

1. 阶段一：Orin 独立调试  
   不连接 Lite3，只验证相机、模型加载、推理速度和端到端推理链路。
2. 阶段二：Orin + Lite3 联动  
   连接 Lite3 运动主机，完成起立、低速移动、交互式探索和基于真实 topomap 的目标导航。

本文档默认仓库根目录为：

```bash
cd /path/to/visualnav-transformer
```

---

## 2. 代码与系统对应关系

### 2.1 系统链路

```text
Orin Camera -> NoMaDInferenceModule -> waypoint sequence
           -> PD middle layer -> Twist command
           -> Lite3RealBridge -> Lite3Controller(UDP)
           -> Lite3 motion host(jy_exe)
```

### 2.2 关键文件

| 文件 | 作用 |
|------|------|
| `scripts/deployment/orin_standalone_test.py` | Orin 端独立调试 |
| `scripts/deployment/lite3_real_bridge.py` | Lite3 真机桥接 |
| `scripts/deployment/nomad_navigation_host.py` | 导航主机 |
| `scripts/configs/navigation_host/lite3_real_bridge_config.json` | 真机桥接配置 |
| `scripts/shared/nomad_inference.py` | NoMaD 推理模块 |
| `scripts/simulation/lite3_system/system.py` | 闭环状态机系统 |
| `scripts/simulation/lite3_system/session.py` | `fpv/`、`goal_views/`、`captures/` 保存 |
| `lite3_host_control/lite3_controller.py` | UDP 上位机控制器 |
| `scripts/nomad_real_deployment_checklist.py` | 真机部署检查清单生成器 |

### 2.3 当前代码已经支持的真机能力

1. Orin 上读取 USB/CSI 相机。
2. 加载 NoMaD checkpoint，在 Orin 上执行 DDIM 推理。
3. 通过 `Lite3RealBridge` 把高层命令转成 Lite3 的 UDP Twist 控制。
4. 用导航主机启动交互式 `stand`、`explore`、`navigate`、`estop`。
5. 保存运行目录下的 `captures/`、`goal_views/`、`fpv/`。

### 2.4 当前代码不应误写的限制

1. `real backend` 的 `navigate` 必须显式提供真实世界 `--topomap-dir`，不会自动回退到仿真 topomap。
2. 交互式主机不是一次命令就自动完成“多目标真机任务编排”的最终形态；当前稳定链路是单目标导航和探索。
3. 模型权重文件不会自动下载到 `deployment/model_weights/`，需要手工准备。

---

## 3. 部署前准备

### 3.1 硬件准备

需要以下硬件：

1. Jetson Orin NX 或 Orin Nano。
2. 一台可被 Orin 读取的相机：
   USB 相机或 CSI 相机均可。
3. Lite3 机器人本体。
4. Orin 与 Lite3 之间的网络连接：
   推荐有线直连；也可使用 Lite3 热点。
5. 独立供电。

### 3.2 软件环境

在 Orin 上执行。

这里先明确一个关键事实：Jetson Orin 没有桌面平台那种独立 RTX 显卡，但它自带可用于 CUDA 推理的 NVIDIA 集成 GPU。对本项目来说，“Orin 能不能推理”不取决于有没有额外显卡，而取决于当前 JetPack 对应的 PyTorch 是否能把这块集成 GPU 正确识别为 `cuda`。

你现在已经确认 Orin 侧 CUDA 版本是 `11.4`。这通常意味着当前环境属于 JetPack 5.x 体系，安装时不要照搬训练机上的桌面 GPU 方式。`scripts/requirements.txt` 是按训练/桌面环境整理的依赖清单，不能直接当成 Orin 上的 PyTorch 安装方案。推荐按照“先确认 JetPack / Python 版本，再装 Jetson 版 PyTorch，最后装其余依赖”的顺序进行。

```bash
conda create -n nomad-deploy python=3.8 -y
conda activate nomad-deploy

cd /path/to/visualnav-transformer

# 先确认当前 Orin 环境
python -V
nvcc --version
dpkg-query --show nvidia-jetpack

# 安装 PyTorch 所需系统依赖
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev

# 对于 CUDA 11.4 且 Python 3.8 的 Orin，优先使用 NVIDIA Jetson 官方 wheel
# 如果 dpkg-query 显示 JetPack 5.1.1，可直接使用下面这条
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python -m pip install --upgrade pip
# 当前部署环境为 Python 3.8，NumPy 使用兼容版本
python -m pip install numpy==1.24.4
python -m pip install --no-cache-dir $TORCH_INSTALL

# 如果你的 JetPack 不是 5.1.1，不要直接照抄上面的 v511 路径
# 改成 NVIDIA 官方文档给出的通式：
# https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION/pytorch/$PYT_VERSION

# 再安装项目运行所需的其余依赖

pip install diffusers==0.11.1 huggingface-hub==0.10.1
pip install efficientnet-pytorch
pip install prettytable lmdb warmup-scheduler
pip install opencv-python pillow matplotlib tqdm h5py "numpy<2"
pip install timm

cd train && pip install -e . && cd ..
```

然后验证 `torch` 是否正常导入。注意，这一步的目标不是看有没有桌面 RTX 卡，而是看 Jetson 集成 GPU 是否被 PyTorch 正确识别。

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

当前 Lite3 + NoMaD 基线真机部署路径已经去掉了对 `torchvision` 的运行时强依赖，因此不需要为了真机部署在 Orin 上继续安装 `torchvision`、`vit-pytorch` 或 `positional-encodings`。
如果 `python -m pip install numpy==1.26.1` 这类命令报 `No matching distribution found`，说明当前 Python 版本不支持该 NumPy 版本。对本文当前的 `Python 3.8` Orin 部署环境，应改用 `numpy==1.24.4`。

如果你后续确实需要在 Orin 上安装 `torchvision`，要特别注意“版本匹配”和“安装来源”是两件事。对于本文当前使用的 Jetson 版 `torch 2.0.0+nv23.05`，应按 `torchvision 0.15.1` 这一版本线处理；但在 Jetson 上不要直接使用通用 PyPI 的 `pip install torchvision`，否则很容易把 NVIDIA 提供的 CUDA 版 `torch` 替换成通用版本，导致 `cuda False` 或 `torch._custom_ops` 这类错误。

更稳妥的做法有两种：

1. 部署环境中默认不安装 `torchvision`，保持当前最小运行时依赖。
2. 如果研究脚本必须用到 `torchvision`，单独建立实验环境，并按 Jetson 兼容方式安装：

```bash
python -m pip uninstall -y torch torchvision
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python -m pip install --no-cache-dir $TORCH_INSTALL

git clone --branch v0.15.1 --depth 1 https://github.com/pytorch/vision torchvision-src
cd torchvision-src
export BUILD_VERSION=0.15.1
python setup.py install
cd ..
```

安装完成后验收：

```bash
python -c "import torch, torchvision; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('torchvision', torchvision.__version__)"
```

理想结果应为：

```text
torch 2.0.0+nv23.05
cuda True
torchvision 0.15.1
```

如果你之前已经执行过 `pip install vit-pytorch positional-encodings[torch]` 或通用 `pip install torchvision`，先回滚到 Jetson 版 `torch` 再继续。不要在真机部署环境里混装会触发 PyPI 重新解析 `torch` / `torchvision` 的包。

### 3.3 预期结果

执行下面命令检查关键脚本是否存在：

```bash
ls scripts/deployment/orin_standalone_test.py
ls scripts/deployment/lite3_real_bridge.py
ls scripts/deployment/nomad_navigation_host.py
ls scripts/configs/navigation_host/lite3_real_bridge_config.json
ls lite3_host_control/lite3_controller.py
```

预期结果：

1. 上述文件均能列出。
2. 没有 `No such file or directory`。
3. 上一条 `python -c` 检查中，理想结果应为：

```text
torch <版本号>
cuda True
device <Jetson Orin 对应的 CUDA 设备名>
```

如果 `lite3_host_control/` 不存在，则说明当前仓库不完整，先不要进入真机部署阶段。
如果 `cuda False`，对 Orin 来说通常不是“没有显卡”，而是 JetPack / CUDA / PyTorch 版本没有对齐，这时先不要继续后面的真机步骤。

### 3.4 模型文件准备

当前代码默认不会自动生成以下文件，因此必须先人工准备：

```bash
mkdir -p deployment/model_weights/nomad

# 你最终实际使用哪一份 checkpoint，就把它放到这里
ls deployment/model_weights/nomad/nomad.pth
```

同时确认配置文件存在：

```bash
ls scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml
```

预期结果：

1. 配置文件存在。
2. 权重文件存在。

如果你要使用其他视觉编码器，只需要替换：

1. `--policy-config`
2. `--policy-checkpoint`

桥接层和状态机不需要修改。

### 3.5 真实 topomap 准备

真实导航必须使用真实环境采集的 topomap。建议为每个场景建立独立目录。

```bash
mkdir -p deployment/topomaps/images/real_hallway
```

采集方法：

1. 先确定起点和目标点。
2. 手持相机沿着机器人未来将要行走的路线缓慢前进。
3. 每隔约 0.5 m 到 1.0 m 拍一张图。
4. 图像按路线顺序重命名为：

```text
000.png
001.png
002.png
...
```

检查命令：

```bash
ls deployment/topomaps/images/real_hallway | head
```

预期结果：

1. 能看到 `000.png`、`001.png` 这类顺序图像。
2. 图像不是空文件。
3. 路径中不要混入 MuJoCo 生成的仿真图像。

---

## 4. 阶段一：Orin 独立调试

本阶段不连接 Lite3，只验证 Orin 侧软硬件链路。

### 4.1 步骤一：相机读取测试

USB 相机：

```bash
python scripts/deployment/orin_standalone_test.py \
  --test camera \
  --camera-device 0
```

CSI 相机：

```bash
python scripts/deployment/orin_standalone_test.py \
  --test camera \
  --use-csi
```

预期结果：

1. 终端输出 `测试一：相机读取`。
2. 前几帧会打印类似：

```text
帧 0: size=(640, 480), latency=xx.xms
```

3. 最后会出现：

```text
✅ 相机正常 | 平均延迟: ...
```

如果失败：

1. USB 相机检查 `ls /dev/video*`。
2. 换设备号，比如 `--camera-device 1`。
3. 如果是 CSI，相机链路和 GStreamer 没准备好时会直接报 `相机打开失败`。

### 4.2 步骤二：模型加载测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test model \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

预期结果：

1. 输出 `测试二：模型加载`。
2. 输出 `PyTorch` 版本与 `CUDA available`。
3. 成功时出现：

```text
✅ 模型加载成功
```

4. 同时打印：
   `Device`、`Image size`、`Context size`、`Trajectory length`。
5. 在 Orin 上的理想结果应是 `CUDA available: True`，并且 `Device` 为 `cuda`。

如果失败：

1. 优先检查 `--policy-config` 路径。
2. 再检查 `--policy-checkpoint` 路径。
3. 若 `CUDA available: False`，脚本会退回 CPU，但这不满足正常真机闭环要求。
4. 对 Orin 而言，`CUDA available: False` 更常见的原因是 Jetson 版 PyTorch 没装对，或者当前环境没有正确继承 JetPack 对应的 CUDA 运行时，而不是“Orin 没有 NVIDIA 显卡”。

### 4.3 步骤三：推理基准测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test benchmark \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5
```

预期结果：

1. 输出 `测试三：推理基准测试`。
2. 先进行 `预热 (5 次)...`。
3. 然后输出 `结果 (DDIM-5)`，包括：
   视觉编码时间、扩散采样时间、端到端时间、推理频率。

建议判断标准：

1. 端到端延迟在 `120 ms - 300 ms` 区间内，说明可以继续做真机尝试。
2. 如果低于 `3 Hz`，先不要进入真机闭环。
3. 如果本步骤显示运行设备是 `cpu`，即使数值偶尔可跑，也不要把它当作正式 Orin 部署结果。

优化方法：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

然后重测，并尝试：

```bash
python scripts/deployment/orin_standalone_test.py \
  --test benchmark \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 2
```

### 4.4 步骤四：端到端流水线测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test pipeline \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5
```

预期结果：

1. 输出 `测试四：完整流水线 (相机 + 推理)`。
2. 先提示填充帧缓冲。
3. 然后每轮打印一个 waypoint，例如：

```text
[1/10] 185.2ms | waypoint=(0.231, -0.014)
```

4. 最后输出平均延迟和推理频率。

这个步骤通过后，说明相机输入、张量构造、NoMaD 推理和 waypoint 输出已经连通。

### 4.5 步骤五：桥接模块相机独立测试

这个步骤仍然不连接 Lite3，只验证桥接模块的相机部分。

```bash
python scripts/deployment/lite3_real_bridge.py \
  --test-camera-only \
  --camera-device 0
```

预期结果：

1. 输出 `=== 相机独立测试 ===`。
2. 连续打印 30 帧图像尺寸。
3. 最后输出 `相机测试完成`。

### 4.6 步骤六：部署清单检查

```bash
python scripts/deployment/nomad_real_deployment_checklist.py \
  --platform lite3
```

预期结果：

1. 打印一张 Markdown 表格。
2. 表格中会列出 checkpoint、导航主机、桥接链路等文件。
3. 如果存在必需文件缺失，命令可能以非零状态退出，这是正常的提醒机制。

### 4.7 阶段一通过标准

进入阶段二之前，至少满足：

1. 相机读取正常。
2. 模型可以成功加载。
3. 模型加载时设备为 `cuda`，而不是 `cpu`。
4. 端到端推理至少达到约 `3 Hz`。
5. 推理输出的 waypoint 数值不是全零，也不是明显发散的异常值。

---

## 5. 阶段二：Orin + Lite3 联动

### 5.1 网络连接

推荐有线直连。

```bash
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0
ping 192.168.1.120
```

预期结果：

1. `ping` 可以收到回复。
2. 延迟稳定，无大量丢包。

如果使用 WiFi，则把桥接配置中的 `robot_ip` 改成机器人实际地址。

### 5.2 桥接配置核对

当前项目的标准桥接配置文件是：

```bash
cat scripts/configs/navigation_host/lite3_real_bridge_config.json
```

配置格式应类似：

```json
{
  "description": "Lite3 真机桥接配置 — Orin 部署用",
  "bridge_kwargs": {
    "robot_ip": "192.168.1.120",
    "robot_port": 43893,
    "local_port": 43897,
    "camera_device": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "use_csi": false,
    "csi_sensor_id": 0,
    "csi_flip": 0,
    "twist_hz": 25.0,
    "max_linear_x": 0.2,
    "max_yaw_rate": 0.6,
    "gait": "low",
    "standup_wait": 3.0
  }
}
```

注意：仓库自带 `lite3_real_bridge_config.json` 的默认值是 `max_linear_x=0.4`、`max_yaw_rate=0.8`，对首次真机调试偏激进。首次必须先手动改小后再启动桥接：

1. `max_linear_x` 改为 `0.2`
2. `max_yaw_rate` 改为 `0.6`
3. `gait` 保持 `low`
4. `standup_wait` 字段的实际等待时间为 `max(0, standup_wait - 3.0)` 秒（因为底层 `prepare_for_twist_control` 已内置 3 秒等待）；若希望额外延长起立稳定时间，应设 `> 3.0`，例如 `4.5`。

### 5.3 通讯与相机联通测试

这一步会连接机器人控制器，但还不会主动起立。

```bash
python scripts/deployment/lite3_real_bridge.py \
  --robot-ip 192.168.1.120 \
  --camera-device 0
```

预期结果：

1. 输出：

```text
[Lite3RealBridge] 控制器已连接: 192.168.1.120:43893
=== 测试相机 ===
图像尺寸: (640, 480)
```

2. 最后打印当前位置与 yaw。

如果这一步失败：

1. 先查网络。
2. 再查 Lite3 运动主机是否已经正常启动。

### 5.4 起立测试

```bash
python scripts/deployment/lite3_real_bridge.py \
  --robot-ip 192.168.1.120 \
  --camera-device 0 \
  --test-standup
```

预期结果：

1. 输出 `=== 测试起立 ===`。
2. 输出 `[Lite3RealBridge] 执行起立序列...`。
3. 最后输出 `[Lite3RealBridge] 起立完成，已进入自主+移动模式`。

安全要求：

1. 机器人四周至少留出 `1 m` 安全空间。
2. 一人扶持，一人操作。
3. 遥控器急停始终可用。

### 5.5 低速前进测试

```bash
python scripts/deployment/lite3_real_bridge.py \
  --robot-ip 192.168.1.120 \
  --camera-device 0 \
  --test-standup \
  --test-twist
```

预期结果：

1. 起立后开始低速前进约 2 秒。
2. 终端打印：

```text
=== 测试低速前进 (2秒) ===
停止
```

3. 机器人前进距离应较短且可控。

如果移动过快或不稳定，立刻降低配置中的 `max_linear_x`。

---

## 6. 交互式导航主机调试

### 6.1 启动交互式主机

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

预期结果：

1. 终端打印交互帮助。
2. 平台初始化后进入：

```text
[Host] IDLE — 等待命令
```

3. 如果 `--camera on`，会出现实时相机窗口。

⚠️ 安全警告：当前交互式主机在进入 idle 前会**自动调用一次 `_ensure_standing()`，机器人在启动瞬间就会起立**（见 `nomad_navigation_host.py` 中 `InteractiveNavigationHost.run()`）。因此启动该命令前必须：

1. 机器人四周至少留出 `1 m` 安全空间。
2. 一人扶持，一人按回车启动。
3. 遥控急停始终可用。
4. 桥接配置中的 `max_linear_x / max_yaw_rate` 已按首次调试值改小。

由于启动时已自动完成起立，§6.2 中第一条 `stand --stand-steps 20` 实质上只是再次维持站立，不是真正的"起立动作"，请勿把它当作安全测试的替代。

### 6.2 推荐的首次交互顺序

进入交互模式后，按下面顺序执行：

```text
stand --stand-steps 20
status
capture
explore --max-steps 30 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
estop
```

每一步的预期结果如下。

#### 1. `stand --stand-steps 20`

说明：交互式主机在启动时已自动起立（见 §6.1 安全警告），此处下发的 `stand` 是作为"冷启动后的维持站立步骤"使用，用于观察状态机链路是否正常，而非真正的首次起立动作。

预期结果：

1. 机器人保持站立。
2. 任务结束后主机返回 idle。
3. 终端出现：

```text
[Host] Task #1 finished: success
```

#### 2. `status`

预期结果：

1. 输出当前位置、yaw、高度。
2. 输出当前已完成任务数量和 capture 数量。

#### 3. `capture`

预期结果：

1. 当前相机图像会被保存到 `captures/`。
2. 终端打印类似：

```text
[Capture] #0: ...
```

#### 4. `explore --max-steps 30 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0`

预期结果：

1. 机器人做短程探索，不依赖真实 topomap。
2. 探索完成后返回 idle。
3. 若窗口开启，可看到实时相机画面刷新。

#### 5. `estop`

预期结果：

1. 机器人停止。
2. 软件层记录急停。

---

## 7. 基于真实 topomap 的目标导航

### 7.1 先检查真实 topomap 目录

```bash
ls deployment/topomaps/images/real_hallway | head
```

预期结果：

1. 至少能看到若干连续编号图像。
2. 图像内容与真实环境一致。

### 7.2 启动导航主机

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

### 7.3 在交互模式中下发导航命令

```text
navigate --topomap-dir deployment/topomaps/images/real_hallway --max-steps 200 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
```

预期结果：

1. 系统会加载真实 topomap。
2. 运行期间实时画面会显示当前图像与目标图像。
3. 当前任务对应的目标图像会保存到运行目录下的 `goal_views/`。
4. 任务结束后回到 idle，并打印任务成功或失败。

如果这一步直接报错：

1. 报 `Real backend requires an explicit real-world --topomap-dir; dataset fallback is disabled.`  
   来自 `scripts/simulation/lite3_system/topomap.py`，说明命令里漏写了 `--topomap-dir`。
2. 报 `Real backend requires --bridge-module and --bridge-class.`  
   来自 `scripts/deployment/nomad_navigation_host.py`，说明启动主机时缺了桥接参数。
3. 报目录不存在  
   说明 topomap 路径写错。
4. 报域不匹配（domain mismatch）  
   说明你传入的可能是仿真 topomap，而不是实拍 topomap。

---

## 8. 可视化模式、无可视化模式与图像保存

### 8.1 可视化模式

用于调试，命令如下：

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --save-fpv
```

预期结果：

1. 出现实时相机窗口。
2. 执行 `navigate` 或 `explore` 时，会把过程帧保存到 `fpv/`。
3. `capture` 会保存到 `captures/`。
4. `navigate` 会把目标图像保存到 `goal_views/`。

### 8.2 无可视化模式

用于正式跑实验时减少 GUI 开销：

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --no-gui \
  --camera off \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --save-fpv
```

预期结果：

1. 不弹出任何 GUI 窗口。
2. 仍可执行 `capture`。
3. 仍可保存：
   `fpv/`、`captures/`、`goal_views/`。

### 8.3 图像保存目录

每次主机启动后会创建一个新的运行目录：

```text
results/deployment/<timestamp>_lite3_real_interactive/
```

目录说明：

| 目录 | 生成条件 | 内容 |
|------|----------|------|
| `captures/` | 执行 `capture` | 当前相机图像 |
| `goal_views/` | 执行 `navigate` 且任务有目标图像 | 当前任务目标图像 |
| `fpv/` | 主机启动时带 `--save-fpv` | 运行过程中的实时图像 |

检查命令：

```bash
ls results/deployment
```

进入最新目录后可继续检查：

```bash
find results/deployment/<latest_run_dir> -maxdepth 2 -type f | head
```

---

## 9. 常见问题与处理方法

### 9.1 相机打不开

处理顺序：

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

如果 USB 相机不在 `video0`，改成：

```bash
python scripts/deployment/orin_standalone_test.py --test camera --camera-device 1
```

### 9.2 Orin 上推理太慢

先执行：

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

再改成更快的推理参数，例如：

```text
explore --max-steps 30 --scheduler ddim --ddim-steps 2 --cfg-weight 0.0
```

### 9.3 导航主机启动时报桥接参数错误

当前代码支持两种 bridge 配置方式：

1. 直接写成原始 kwargs 字典。
2. 写成带 `bridge_kwargs` 包装的 JSON。

建议继续使用项目自带的：

```bash
scripts/configs/navigation_host/lite3_real_bridge_config.json
```

### 9.4 `navigate` 一启动就报 topomap 错误

这是最常见问题。检查三件事：

1. 命令里是否显式写了 `--topomap-dir`。
2. 目录是否真实存在。
3. 目录中的图像是不是实拍图，而不是 MuJoCo 图。

### 9.5 没有保存 `fpv/`

要满足两个条件：

1. 启动导航主机时带 `--save-fpv`。
2. 实际执行了 `explore` 或 `navigate` 这类会循环运行的任务。

仅仅进入 idle 或只执行 `capture`，不会产生整段 `fpv/`。

---

## 10. 推荐的完整执行顺序

建议严格按下面顺序执行：

```bash
# 1. 环境和文件检查
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3

# 2. Orin 独立相机测试
python scripts/deployment/orin_standalone_test.py --test camera --camera-device 0

# 3. 模型加载测试
python scripts/deployment/orin_standalone_test.py \
  --test model \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth

# 4. 推理基准
python scripts/deployment/orin_standalone_test.py \
  --test benchmark \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5

# 5. 桥接通讯测试
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.1.120 --camera-device 0

# 6. 起立测试
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.1.120 --camera-device 0 --test-standup

# 7. 低速前进测试
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.1.120 --camera-device 0 --test-standup --test-twist

# 8. 交互式主机
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --save-fpv
```

进入交互模式后依次执行：

```text
stand --stand-steps 20
status
capture
explore --max-steps 30 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
navigate --topomap-dir deployment/topomaps/images/real_hallway --max-steps 200 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
estop
quit
```

---

## 11. 最终结论

经过核对，当前项目中的 Orin + Lite3 真机部署教程，必须满足以下口径才是正确的：

1. 真实导航一定要提供真实 `topomap-dir`。
2. `bridge-config` 使用项目自带 JSON 即可，当前代码已兼容其 `bridge_kwargs` 包装格式；但默认 `max_linear_x=0.4 / max_yaw_rate=0.8` 偏激进，首次真机前需先手动改小（见 §5.2）。
3. 若要保存运行过程中的实时图像，启动导航主机时必须带 `--save-fpv`。
4. `captures/`、`goal_views/`、`fpv/` 三类图像都已经有对应代码路径，不是纯文档设计。
5. **交互式主机启动即自动起立**：`nomad_navigation_host.py --interactive` 的 `run()` 会在进入 idle 之前调用一次 `_ensure_standing()`，所以必须在启动命令回车之前就完成安全准备，不能指望"启动后再有时间反应"。
6. 当前最稳的真机流程是：
   先独立调试，再桥接通讯（仅连接），再用 `--test-standup/--test-twist` 手工验证起立与低速前进，再打开交互式主机（注意其会自动起立），再探索，最后再做基于真实 topomap 的单目标导航。
