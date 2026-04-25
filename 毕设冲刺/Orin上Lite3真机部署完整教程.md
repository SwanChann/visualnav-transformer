# Orin 上 Lite3 真机部署完整教程

## 目录

1. [文档说明](#1-文档说明)
2. [软件流水线全图(先读这一节)](#2-软件流水线全图先读这一节)
3. [部署前准备](#3-部署前准备)
4. [阶段一:Orin 独立调试](#4-阶段一orin-独立调试)
5. [阶段二:Orin + Lite3 联动](#5-阶段二orin--lite3-联动)
6. [交互式导航主机](#6-交互式导航主机)
7. [navigate 模式:真实 topomap 目标导航](#7-navigate-模式真实-topomap-目标导航)
8. [运行结果保存](#8-运行结果保存)
9. [常见问题排查](#9-常见问题排查)
10. [快速启动序列](#10-快速启动序列)
11. [关键结论与检查清单](#11-关键结论与检查清单)

---

## 1. 文档说明

本文档指导在 Jetson Orin 上部署 NoMaD 视觉导航,并与云深处 Lite3 真机联动。所有命令、文件路径、预期结果均来源于当前仓库代码,可直接对应到具体脚本。

### 1.1 部署阶段划分

| 阶段 | 是否连接 Lite3 | 目标 |
|---|---|---|
| 阶段一 | 否 | 验证相机、模型加载、推理速度、端到端推理 |
| 阶段二 | 是 | 起立、低速移动、交互式探索、基于 topomap 的目标导航 |

本文档默认仓库根目录为:

```bash
cd /path/to/visualnav-transformer
```

### 1.2 关键文件对照

| 文件 | 作用 |
|---|---|
| `scripts/deployment/orin_standalone_test.py` | Orin 端独立调试入口 |
| `scripts/deployment/capture_real_topomap.py` | Orin 相机采集真实 topomap |
| `scripts/deployment/lite3_real_bridge.py` | Lite3 真机桥接(相机 + 控制) |
| `scripts/deployment/nomad_navigation_host.py` | 导航主机入口 |
| `scripts/configs/navigation_host/lite3_real_bridge_config.json` | 真机桥接配置 |
| `scripts/shared/nomad_inference.py` | NoMaD 推理模块 |
| `scripts/simulation/lite3_system/system.py` | 闭环状态机系统 |
| `scripts/simulation/lite3_system/interfaces.py` | NoMaD 高层 / PD 中层 / 平台抽象 |
| `scripts/simulation/lite3_system/topomap.py` | MissionQueue / topomap 校验 |
| `scripts/simulation/lite3_system/session.py` | `videos/` `goal_views/` `captures/` 落盘 |
| `lite3_host_control/lite3_controller.py` | UDP 上位机控制器 |
| `scripts/nomad_real_deployment_checklist.py` | 部署检查清单生成器 |

### 1.3 当前已支持的真机能力

1. Orin 上读取 USB / CSI 相机(后台线程持续刷新最新帧)。
2. 在 Orin 上加载 NoMaD checkpoint,执行 DDIM/DDPM 推理。
3. 用 Orin 相机采集真实环境 topomap 并写入 `topomap_meta.json`。
4. 通过 `Lite3RealBridge` 把高层命令转成 Lite3 的 UDP Twist。
5. 用导航主机启动交互式 `stand` / `keyboard` / `explore` / `navigate` / `estop`。
6. 在运行目录下保存 `captures/` `goal_views/` `videos/`。

### 1.4 已知限制

1. 真机 `navigate` 必须显式提供真实世界 `--topomap-dir`,不会自动回退到仿真 topomap。
2. 交互式主机当前的稳定能力是单目标导航和探索;多目标编排尚未稳定。
3. 模型权重不会自动下载到 `deployment/model_weights/`,需手工准备。

---

## 2. 软件流水线全图(先读这一节)

> **本节是为定位"导航不准"或"图像滞后"问题提供的代码层事实。** 调试时先在这里确认软件不是嫌疑对象,再去查模型或硬件。

### 2.1 端到端数据流

```text
┌─────────────────────────────┐
│  Orin USB / CSI 相机        │
│  └─ OrinCamera 后台线程     │ 持续 cap.read(),丢旧只留最新一帧
│     (CAP_PROP_BUFFERSIZE=1) │
└──────────────┬──────────────┘
               │  read_bgr() max_age=0.5s
               ▼
   ┌─────────────────────────────────────────┐
   │  Lite3RealBridge.render_camera()        │  返回 PIL.Image
   └─────────────┬─────────────────┬─────────┘
                 │                 │
       (推理路径) │                 │ (显示路径,异步独立线程)
                 ▼                 ▼
   ┌─────────────────┐   ┌──────────────────────┐
   │ ContextBuffer   │   │ _async_viewer_loop   │
   │  push 最新帧    │   │  按 --camera-viewer- │
   │  保留 4 帧上下文│   │  fps 调用 imshow     │
   └────────┬────────┘   │  (默认 15 Hz)        │
            │            └──────────┬───────────┘
            │                       │
            ▼                       ▼
   ┌─────────────────┐   ┌──────────────────────┐
   │ NoMaD 高层      │   │ NoMachine 远程显示    │
   │ predict_        │   │ ↑ 用户看到的画面      │
   │ navigation()    │   │                       │
   └────────┬────────┘   │ 滞后来源: WiFi 网络   │
            │            │   + OpenCV imshow    │
            ▼            │   + 远程桌面带宽      │
   ┌─────────────────┐   └──────────────────────┘
   │ Lite3MiddleLayer│
   │ PD 中层         │  waypoint → (v, w)
   │ pd_controller() │
   └────────┬────────┘
            │
            ▼
   ┌─────────────────┐
   │ Lite3RealBridge │  发 Twist UDP → Lite3 运动主机
   │ apply_command() │
   └─────────────────┘
```

### 2.2 软件正确性逐条核对

| 关注点 | 实现位置 | 实际行为 |
|---|---|---|
| 相机不积压旧帧 | `OrinCamera._capture_loop` | 后台线程独占相机,单槽缓冲,新帧覆盖旧帧 |
| NoMaD 用最新帧 | `Lite3System.step_navigation` | 每 tick `render_camera()` 现取现用 |
| 显示与推理解耦 | `Lite3System._async_viewer_loop` | 独立线程,自己再调一次 `render_camera()` |
| 上下文 4 帧滚动 | `ContextBuffer` | `deque(maxlen=context_size+1)`,顺序有序 |
| 图像预处理一致 | `NoMaDInferenceModule._resize_image` | 实时帧、目标帧、topomap 用同一种 resize_mode |
| topomap 域校验 | `validate_topomap_dir_for_domain` | 真机 backend 拒绝加载仿真 topomap |
| 起立幂等 | `Lite3RealBridge.standup` | 已站立时只做模式切换,不再触发起立 |
| 命令限幅二级 | `Lite3MiddleLayerPD` + `Lite3RealBridge.apply_command` | PD 缩放 → PD clamp → 桥接 clamp,NaN/Inf 强制零速度 |

### 2.3 navigate 模式定位机制(必须了解)

> **关键事实先说在前面:topomap 不是"一张目标图像",而是一条"路标链"。每个 tick 模型同时看窗口里的多个候选节点,既用来定位,又用来挑下一站作为扩散条件。**

#### 2.3.1 每个 tick 实际发生的事

NoMaD navigate 不是开环跑一条预设轨迹,而是闭环局部目标跟踪。一个 tick 内:

```text
a. 取候选窗口 topomap[closest - radius : closest + radius + 1]
   默认 radius=4 → 一次取最多 9-10 个节点
   
b. 把当前 obs 复制 N 份,与窗口内每个候选配对
   → encode_condition 一次算出 N 个 (obs, candidate_i) 联合编码
   
c. dist_pred_net 给每个对算一个标量距离
   → argmin = "我现在视觉上最像哪个 topomap 节点" → 新 closest_node
   (含 anti-jump 限幅: clipped_closest 每 tick 最多前进 3,后退 1)
   
d. selected_index = closest_node + (1 if 距离 < close_threshold(默认 3.0) else 0)
   → "下一站"作为本 tick 的局部目标
   
e. 扩散策略条件 = (obs, topomap[selected_index]) 的联合编码
   → 输出 8 步轨迹,取第 waypoint_index(默认 2)步作为 waypoint
   
f. PD: waypoint(dx, dy) → (v = dx/dt, w = atan2(dy,dx)/dt)
   v 截到 [0, MAX_V],w 截到 [-MAX_W, MAX_W]
   
g. 桥接层再做一次安全限幅 max_linear_x / max_yaw_rate
   
h. 终止判据: selected_node ≥ goal_node 且预测距离 < close_threshold
```

#### 2.3.2 topomap 节点的三种用途

看完上面流程,可以把 topomap 节点的角色说清楚:

| 用途 | 用谁 | 谁看 |
|---|---|---|
| **视觉定位** | 滑动窗口内**所有**候选节点 | 模型(dist 头) |
| **扩散条件** | `topomap[selected_index]`,即下一站 | 模型(diffusion 头) |
| **终止判据** | `topomap[goal_node]` | 状态机 |
| **窗口右侧可视化** | `mission.goal_view = topomap[goal_node]` | 只给人看 |

> ⚠️ **常见误解:** 看到可视化左右分屏会以为"模型只看实时帧 + 那张右边的 goal"。实际上模型每 tick 看的是 9-10 张候选节点(用于定位),扩散条件用的是滑动到当前位置的"下一站",**而不是右边那张最终 goal**。最终 goal 仅在机器人滚到接近 `goal_node` 时才进入扩散条件。

#### 2.3.3 为什么必须指定 `--topomap-dir`

NoMaD 训练时学的是"从当前视野到 1-2 秒后的子目标"的**短距离**轨迹分布。直接给一张几十米外的最终 goal,模型没在训练分布里见过这种超远 goal,会输出无意义的 waypoint。

topomap 的作用就是把"长距离导航"切成一串"短距离子目标跟踪",每一段都落在模型训练分布内。没有 topomap:

- 没法视觉定位(无候选集合)
- 没法分段(每段距离都超出训练分布)
- 终止条件无从判断

所以即使你只想去"目录里第 23 张图",目录里前 22 张也不可缺 — 它们是中间路标。

#### 2.3.4 三个隐含约束(出问题时优先怀疑)

**约束 1:机器人开机位置必须接近 topomap 节点 0**

`MissionQueue.closest_node = 0` 写死,且 anti-jump clip 让 closest_node 每 tick 最多前进 3、后退 1。把机器人放到 topomap 节点 50 的位置启动 navigate:

| Tick | closest_node | 搜索窗口 | 实际匹配能力 |
|---|---|---|---|
| 0 | 0 | [0, 5] | obs@50 与 topomap[0..5] 都不像 → argmin 是垃圾 |
| 1 | ≤ 3 | [0, 8] | 仍然垃圾 |
| ... | | | |
| ~17 | 50 | [46, 55] | 终于对齐 |

期间机器人都在朝错误方向漂移。**开始 navigate 前要让机器人朝向、视野尽量贴合 `topomap/000.png`**(第一帧)。

**约束 2:`--goal-image` 只能改 goal_node,不能跳过中间节点**

`--goal-image 023.png` 会把 `goal_node = 23`、可视化右侧显示 `topomap[23]`。但 closest_node 仍从 0 开始,中间 1-22 仍作为定位锚点参与每个 tick 的搜索。**它不会**让你直接从节点 0 一步条件到节点 23。

**约束 3:PD 只输出前向速度(v ≥ 0)**

[pd_controller](scripts/nomad_mujoco_lite3_nav.py#L1014-L1028) 有 `v = clip(dx/dt, 0, MAX_V)`。模型预测"后退"会被截成 0,只能站住转向。窄走廊或需要倒车的场景会因此卡住。

### 2.4 NoMachine 看到的画面为什么会滞后

| 链路环节 | 频率 / 延迟 |
|---|---|
| 相机驱动 → OpenCV 缓冲 | 25-30 FPS,缓冲 1 帧 |
| `OrinCamera` 后台线程 | 等同相机帧率,单槽更新 |
| **`_async_viewer_loop` 取帧 + `imshow`** | **由 `--camera-viewer-fps` 决定,默认 15 FPS** |
| OpenCV 在 X11 上渲染 | 受 X11 协议、显存拷贝影响 |
| **NoMachine 远程桌面编码 + 网络传输** | **WiFi 带宽 / 延迟 / 抖动决定的瓶颈** |
| 客户端解码与刷新 | 通常 30-60 FPS |

> **结论:NoMaD 推理使用的图像始终是最新的相机帧,与 NoMachine 看到的画面无关。** NoMachine 上的滞后是远程桌面 + OpenCV 显示线程的固有特性,不影响控制环。如果想缩小显示滞后,见 [§9.3](#93-真机画面帧率低或拖拽感强)。

### 2.5 navigate 不准的可能原因(按概率排序)

1. **topomap 起点未对齐**:机器人开机视野与 `000.png` 差异过大。
2. **topomap 采集间距不合理**:节点过疏(>1.5 m/节点)或过密(<0.3 m/节点)都会让局部距离预测失真。
3. **图像 resize 模式与训练分布不一致**:目前默认 `stretch`,与 NoMaD 原始预处理一致;切到 `letterbox/center_crop` 时分布会变。
4. **真实场景与训练分布差异大**:走廊、室内、光照、纹理与 GoStanford / SACSoN 等训练数据偏差越大,越易跑偏。
5. **PD 出力被卡死**:模型让其后退 → 被截成 0 → 表面看像"卡住但还在转"。
6. **WiFi 链路不稳**:`Lite3Controller` 心跳丢包会让运动主机回退到上一条 Twist,但**这不会让相机滞后**,只会让动作执行延迟。

软件层我已经核过,这些原因里只有 1、2、4、5、6 是模型/采集/硬件,其余可在配置里调,不需要改代码。

---

## 3. 部署前准备

### 3.1 硬件清单

1. Jetson Orin NX 或 Orin Nano。
2. USB 相机或 CSI 相机一台。
3. Lite3 机器人本体。
4. Orin ↔ Lite3 网络:推荐有线直连,也可使用 Lite3 自带 WiFi 热点。
5. 独立供电(Orin 与 Lite3 都不要靠 USB 反向供电)。

### 3.2 Orin 软件环境

> **关键事实:** Jetson Orin 自带可用于 CUDA 推理的集成 GPU。不需要额外独立显卡,但**必须使用 Jetson 官方版 PyTorch**,不能直接 `pip install torch`。

**安装顺序:先确认 JetPack / Python 版本 → 装 Jetson 版 PyTorch → 装其余依赖。**

```bash
# 创建独立 conda 环境
conda create -n nomad-deploy python=3.8 -y
conda activate nomad-deploy
cd /path/to/visualnav-transformer

# 确认环境
python -V
nvcc --version
dpkg-query --show nvidia-jetpack

# PyTorch 系统依赖
sudo apt-get update
sudo apt-get install -y python3-pip libopenblas-dev

# Jetson 版 PyTorch (示例: JetPack 5.1.1, CUDA 11.4, Python 3.8)
export TORCH_INSTALL=https://developer.download.nvidia.com/compute/redist/jp/v511/pytorch/torch-2.0.0+nv23.05-cp38-cp38-linux_aarch64.whl
python -m pip install --upgrade pip
python -m pip install numpy==1.24.4
python -m pip install --no-cache-dir $TORCH_INSTALL

# 项目运行依赖
pip install diffusers==0.11.1 huggingface-hub==0.10.1
pip install efficientnet-pytorch
pip install prettytable lmdb warmup-scheduler
pip install opencv-python pillow matplotlib tqdm h5py "numpy<2"
pip install timm

cd train && pip install -e . && cd ..
```

> ⚠️ **关于 JetPack 版本不是 5.1.1**:不要照抄 v511 链接。改成 NVIDIA 官方通式 `https://developer.download.nvidia.com/compute/redist/jp/v$JP_VERSION/pytorch/$PYT_VERSION`。

> ⚠️ **关于 NumPy 版本**:Python 3.8 用 `numpy==1.24.4`;`numpy==1.26.x` 不支持 3.8。

> ⚠️ **关于 torchvision**:真机部署链路已不依赖 `torchvision`。**不要直接 `pip install torchvision`**,会触发 PyPI 重新解析覆盖 Jetson 版 `torch`,导致 `cuda False`。如确需研究脚本,单独建实验环境并源码编译 `v0.15.1`。

**验证 PyTorch + CUDA:**

```bash
python -c "import torch; print('torch', torch.__version__); print('cuda', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

理想结果:`torch 2.0.0+nv23.05` `cuda True` `device Orin`。

### 3.3 关键脚本检查

```bash
ls scripts/deployment/orin_standalone_test.py
ls scripts/deployment/lite3_real_bridge.py
ls scripts/deployment/nomad_navigation_host.py
ls scripts/configs/navigation_host/lite3_real_bridge_config.json
ls lite3_host_control/lite3_controller.py
```

文件全部存在 → 仓库完整;若缺 `lite3_host_control/`,先不要进入真机阶段。

### 3.4 模型权重与配置

```bash
mkdir -p deployment/model_weights/nomad
# 把要用的 checkpoint 放到这里
ls deployment/model_weights/nomad/nomad.pth
# 视觉编码器配置
ls scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml
```

替换其他视觉编码器只需改 `--policy-config` 与 `--policy-checkpoint`,桥接层和状态机不需要改。

### 3.5 真实 topomap 准备

> **navigate 模式必须用真实场景采集的 topomap。** 训练数据集图像、MuJoCo 仿真图像都不能用于真机 navigate。

**目录命名约定:**

```text
deployment/topomaps/images/
  real_hallway/      ← 默认真实场景标签
  lab_corridor/      ← 其他真实场景同级目录
  office_loop/
```

`--map` 是运行标签(用于 capture 命名 / 状态显示),`--topomap-dir` 才是导航实际加载的目录。两者可以不同名。

**采集流程(用 Orin 相机,保证视角与部署时一致):**

```bash
mkdir -p deployment/topomaps/images/real_hallway

# 自动间隔模式: 适合开阔场地
python scripts/deployment/capture_real_topomap.py \
  --output-dir deployment/topomaps/images/real_hallway \
  --map-name real_hallway \
  --camera-device 0 \
  --count 40 \
  --interval 5

# 手动模式: 适合需要每节点停稳的窄场地
python scripts/deployment/capture_real_topomap.py \
  --output-dir deployment/topomaps/images/real_hallway \
  --map-name real_hallway \
  --camera-device 0 \
  --count 40 \
  --manual
```

手动模式下预览窗口的按键:

| 键 | 动作 |
|---|---|
| Enter / Space / s | 保存当前帧 |
| ESC | 结束采集并写入 `topomap_meta.json` |

> **如果想边用导航主机的 keyboard 模式开机器人边采集**,先在另一个终端启动主机,**务必加 `--camera off`** 防止两个进程抢同一台相机。导航主机进入 `keyboard` 模式后再在采集终端运行上面的命令。

**采集要点:**

1. 先确定起点(将作为 `000.png`)和目标点。
2. 相机高度、朝向贴合 Lite3 头部视角。
3. 沿机器人未来要走的路线缓慢前进。
4. 节点间距 0.5-1.0 m。
5. 文件名按顺序自动编为 `000.png`、`001.png`、...

**采集后检查:**

```bash
ls deployment/topomaps/images/real_hallway | head
cat deployment/topomaps/images/real_hallway/topomap_meta.json
```

`topomap_meta.json` 中应有 `"domain": "real"`;否则真机 backend 会拒绝加载。

---

## 4. 阶段一:Orin 独立调试

> 本阶段不连接 Lite3,只验证 Orin 侧软硬件链路。每一步都给出可验收的输出范例。

### 4.1 步骤 1:相机读取测试

```bash
# USB
python scripts/deployment/orin_standalone_test.py --test camera --camera-device 0
# CSI
python scripts/deployment/orin_standalone_test.py --test camera --use-csi
```

**预期输出(USB 实测参考):**

```text
[OrinCamera] 相机已打开: requested=640x480@30fps, actual=640x480@30.0fps,
             CSI=False, backend=v4l2, fourcc=MJPG
  帧 0: size=(640, 480), latency=41.4ms
  ...
✅ 相机正常 | 平均延迟: 40.0ms | FPS: 25.0
```

桥接层默认用 V4L2 + MJPG + buffer=1,目的就是不让旧帧积压。

**失败排查:**

| 现象 | 处理 |
|---|---|
| `Camera index out of range` | `ls /dev/video*` 找正确编号,改 `--camera-device` |
| CSI 报 `相机打开失败` | GStreamer 未就绪,先 `nvgstcapture-1.0` 自测 |
| `device=0` 被占用 | `sudo fuser -k /dev/video0` |

### 4.2 步骤 2:USB 相机尺寸与编码器输入对齐

USB 输出 `640x480`,NoMaD baseline 输入 `96x96`。**不需要在相机层改尺寸**,代码会在送入 NoMaD 前根据 `image_resize_mode` 自动预处理。重要的是实时帧、目标帧、topomap 帧用同一种模式。

| 模式 | 行为 | 适用 |
|---|---|---|
| `stretch`(默认) | 直接拉伸到 96x96 | 与 NoMaD 训练预处理一致,首选 |
| `center_crop` | 中心裁方再缩放 | 几何比例真实,但裁掉左右视场 |
| `letterbox` | 保持比例 + 黑边补齐 | 几何最真实,但训练分布无黑边 |

```bash
python scripts/deployment/orin_standalone_test.py \
  --test pipeline \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5 \
  --image-resize-mode stretch
```

> **首轮真机部署用 `stretch`。** 只有发现机器人对横向距离、转弯幅度判断明显异常时,再分别测试 `center_crop` 和 `letterbox`。

> 相机标定解决的是镜头畸变,不是宽高比对齐。两件事不要混。普通 USB 相机畸变小可不标定;边缘明显弯曲再标定。

### 4.3 步骤 3:模型加载测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test model \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

**预期输出(实测参考):**

```text
PyTorch: 2.0.0+nv23.05
CUDA available: True
GPU: Orin
✅ 模型加载成功 | 耗时: 2.87s
Device: cuda
Image size: (96, 96)
Resize mode: stretch
Context size: 3
Trajectory length: 8
```

`Device: cuda` 是阶段一是否成立的关键证据。首次加载 ~2.9s 正常(checkpoint 读 + CUDA 初始化)。

> 若 `CUDA available: False`,**先怀疑 Jetson 版 PyTorch 未对齐**,而不是 Orin 没有 GPU。

### 4.4 步骤 4:推理基准测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test benchmark \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5
```

**预期输出:**

```text
预热 (5 次)...
运行 20 次基准测试...

📊 结果 (DDIM-5):
   视觉编码:   56.9 ± 1.4 ms
   扩散采样:   86.7 ± 1.9 ms
   端到端:     143.6 ± 2.8 ms
   推理频率:   ~7.0 Hz
```

| 端到端延迟 | 状态 |
|---|---|
| 120-300 ms | 可继续真机尝试 |
| > 333 ms (< 3 Hz) | 不要进入真机闭环 |
| 显示 `cpu` | 即使能跑也不算正式 Orin 部署 |

**性能优化:**

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
# 重测,或改 --ddim-steps 2
```

> 若运行中出现 `UserWarning: Converting mask without torch.bool dtype to bool`,只要数值稳定即可视为非阻塞 warning。

### 4.5 步骤 5:端到端流水线测试

```bash
python scripts/deployment/orin_standalone_test.py \
  --test pipeline \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --ddim-steps 5
```

**预期输出:**

```text
[1/10] 2895.5ms | waypoint=(1.245, 0.150)
[2/10] 151.5ms | waypoint=(1.341, 0.161)
...
[10/10] 153.0ms | waypoint=(1.330, 0.031)

📊 端到端流水线结果:
   平均延迟: 427.3 ± 822.7 ms
   推理频率: ~2.3 Hz
```

> **首轮 ~2.9s 是冷启动开销,不是稳态。** 应关注第 2-10 轮,稳定在 ~150 ms ≈ 6.5 Hz 即 OK。当前版本已额外输出"稳态延迟(不含首轮)"。

### 4.6 步骤 6:桥接相机独立测试

```bash
python scripts/deployment/lite3_real_bridge.py --test-camera-only --camera-device 0
```

应连续打印 30 帧 `(640, 480)`。这一步证明桥接层调用的相机封装与阶段一一致。

### 4.7 步骤 7:部署清单检查

```bash
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3
```

会打印 Markdown 表格列出 checkpoint、导航主机、桥接链路等。必需文件缺失时退出码非零。

### 4.8 阶段一通过标准

| 项 | 标准 |
|---|---|
| 相机读取 | 正常,FPS ≥ 20 |
| 模型加载 | 设备 = `cuda` |
| 基准频率 | ≥ 3 Hz(理想 ≥ 5 Hz) |
| pipeline 稳态 | 首轮之后稳定,不含首轮平均 ≤ 250 ms |
| waypoint 数值 | 非零、不发散 |

阶段一通过后,后续怀疑应集中在网络、桥接、安全限幅,不再怀疑 Orin 推理能力。

---

## 5. 阶段二:Orin + Lite3 联动

### 5.1 网络连接

**当前默认无线方案:** Orin 连接 Lite3 WiFi 热点,Lite3 自动分配 IP,实测 Orin = `192.168.2.17`,运动主机 = `192.168.2.1`。

| 地址 | 角色 | 用途 |
|---|---|---|
| `192.168.2.17` | Orin 导航主机 | 你电脑 SSH 登录目标 |
| `192.168.2.1` | Lite3 运动主机 | `Lite3RealBridge` 的 `robot_ip` |

> ⚠️ **不要把 `robot_ip` 改成 `192.168.2.17`**,否则控制包会发回 Orin 自己。

```bash
ssh guest@192.168.2.17

# 进 Orin 后核对:
ip addr show wlan0          # wlan0 应有 192.168.2.17
ping 192.168.2.1            # 应能收到回复且延迟稳定
```

**如改回有线直连:**

```bash
sudo ifconfig eth0 192.168.1.100 netmask 255.255.255.0
ping 192.168.1.120
# 同步把 bridge config 中 robot_ip 改回有线段
```

### 5.2 桥接配置

```bash
cat scripts/configs/navigation_host/lite3_real_bridge_config.json
```

参考默认值(已按首次真机调试调小):

```json
{
  "description": "Lite3 真机桥接配置 — Orin 部署用",
  "network": {
    "orin_ip": "192.168.2.17",
    "lite3_motion_host_ip": "192.168.2.1",
    "legacy_motion_host_ip": "192.168.1.120"
  },
  "bridge_kwargs": {
    "robot_ip": "192.168.2.1",
    "robot_port": 43893,
    "local_port": 43897,
    "camera_device": 0,
    "camera_width": 640,
    "camera_height": 480,
    "camera_fps": 30,
    "camera_backend": "v4l2",
    "camera_fourcc": "MJPG",
    "use_csi": false,
    "csi_sensor_id": 0,
    "csi_flip": 0,
    "twist_hz": 25.0,
    "max_linear_x": 0.12,
    "max_yaw_rate": 0.35,
    "gait": "low",
    "standup_wait": 3.0
  }
}
```

| 字段 | 说明 |
|---|---|
| `network` | 仅供人工核对,不传给桥接代码 |
| `bridge_kwargs.robot_ip` | 真正传给桥接的运动主机地址 |
| `camera_backend / camera_fourcc` | 默认 `v4l2 + MJPG`,绕开 YUYV 带宽不足问题 |
| `max_linear_x = 0.12` | 桥接层最终线速度限幅 |
| `max_yaw_rate = 0.35` | 桥接层最终偏航角速度限幅 |
| `standup_wait = 3.0` | 实际等待时间 = `max(0, standup_wait - 3.0)`(底层固定 3 秒已包含) |

> 速度还可在 PD 中层再缩,见 [§9.4](#94-速度指令太大或机器人动作太猛)。

### 5.3 通讯与相机联通测试(不起立)

```bash
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.2.1 --camera-device 0
```

**预期输出:**

```text
[Lite3RealBridge] 控制器已连接: 192.168.2.1:43893
=== 测试相机 ===
图像尺寸: (640, 480)
位置: (0.000, 0.000), yaw=0.000
```

### 5.4 起立测试

```bash
python scripts/deployment/lite3_real_bridge.py \
  --robot-ip 192.168.2.1 --camera-device 0 --test-standup
```

**安全要求:**

1. 机器人四周至少 1 m 安全空间。
2. 一人扶持,一人操作。
3. 遥控器急停始终可用。

**预期输出:** `[Lite3RealBridge] 起立完成,已进入自主+移动模式`。

### 5.5 低速前进测试

```bash
python scripts/deployment/lite3_real_bridge.py \
  --robot-ip 192.168.2.1 --camera-device 0 \
  --test-standup --test-twist
```

起立后低速前进约 2 秒,终端打印 `=== 测试低速前进 (2秒) ===` → `停止`。

> 若旧版本报 `ModuleNotFoundError: No module named 'lite3_system'`,说明 `--test-twist` 还在依赖仿真侧接口;同步到当前版本即可。

---

## 6. 交互式导航主机

### 6.1 启动命令

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --map real_hallway \
  --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --pd-linear-scale 0.6 --pd-yaw-scale 0.6 \
  --pd-max-v 0.12 --pd-max-w 0.35 \
  --profile-timing --profile-interval 5
```

> ⚠️ **启动即自动起立!** `InteractiveNavigationHost.run()` 进 idle 前会调用一次 `_ensure_standing()`,机器人在你按下 Enter 的瞬间起立。所以**回车前**必须:
>
> 1. 四周 ≥ 1 m 安全空间
> 2. 一人扶持
> 3. 遥控急停可用
> 4. 桥接配置 `max_linear_x / max_yaw_rate` 已改小
>
> §6.2 第一条 `stand` 实质是"维持站立",不是首次起立。

成功进入后会出现:

```text
[Host] IDLE — 等待命令
```

### 6.2 推荐首次交互顺序

```text
stand --stand-steps 20
status
capture
explore --max-steps 30 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
estop
```

| 命令 | 期望结果 |
|---|---|
| `stand --stand-steps 20` | 机器人保持站立,任务结束 → idle,**不应**触发 `soft_estop` |
| `status` | 打印 pos/yaw/height、tasks/captures、`Backend: real, Map: real_hallway` |
| `capture` | 当前帧落盘到 `captures/`,打印 `[Capture] #0: ...` |
| `explore --max-steps 30 ...` | 短程无目标探索,完成后 → idle |
| `estop` | 打印 `⚠️ 执行急停!` 与 `⚠️ 软急停!`,主机退出 idle |

> 若 `status` 显示 `Map: easy`,说明 Orin 上代码未同步到当前版本,或启动命令显式写了 `map: easy`。临时修正加 `--map real_hallway`。

> `--profile-timing` 会在导航/探索时按 `--profile-interval` 周期打印 `camera/ui/preprocess/infer/command/total` 各阶段耗时。排查完可去掉。

---

## 7. navigate 模式:真实 topomap 目标导航

### 7.1 启动前检查

```bash
ls deployment/topomaps/images/real_hallway | head
cat deployment/topomaps/images/real_hallway/topomap_meta.json
```

`topomap_meta.json` 必须有 `"domain": "real"`。

> ⚠️ **必须把机器人放在 topomap 起点位置上。** 见 [§2.3](#23-navigate-模式定位机制必须了解)的隐含约束:`closest_node` 起始为 0,只在 `[0, 0+radius+1]` 内搜索。开机视野偏离 `000.png` 太远会让定位锁死。

### 7.2 启动主机(同 §6.1)

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive --backend real --map real_hallway --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

### 7.3 下发导航任务

```text
navigate --topomap-dir deployment/topomaps/images/real_hallway \
         --max-steps 200 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
```

**运行期窗口键位:**

| 键 | 动作 |
|---|---|
| `[` / `]` | 把当前 topomap 目标节点向前/向后切换 |
| `c` | 立即拍当前帧到 `captures/` |
| `,` / `.` | 切换 capture 队列选中项 |
| `ESC` | 退出当前任务,回到主机 idle |
| `e` | 软急停 |

**指定特定目标图像(可选):**

```text
navigate --topomap-dir deployment/topomaps/images/real_hallway \
         --goal-image 023.png \
         --max-steps 200 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
```

`--goal-image` 匹配文件名或不带后缀的 stem;不指定则用 topomap 最后一张。

### 7.4 启动报错速查

| 报错 | 原因 |
|---|---|
| `Real backend requires an explicit real-world --topomap-dir; dataset fallback is disabled.` | 命令里漏写 `--topomap-dir` |
| `Real backend requires --bridge-module and --bridge-class.` | 启动主机时缺桥接参数 |
| `Topomap directory not found` | 路径写错 |
| `Topomap directory '...' is tagged as 'mujoco', but the current backend requires 'real'.` | 传入了仿真 topomap |

### 7.5 navigate 不准的现场调参建议

按 [§2.5](#25-navigate-不准的可能原因按概率排序) 的顺序逐项排:

1. **起点对齐**:把机器人摆到 `000.png` 拍摄位置同视角,再开 navigate。
2. **重新采集 topomap**:节点间距 0.5-1.0 m,转弯处加密。
3. **改 image_resize_mode**:确保实时帧、topomap、采集时三个地方的 resize 模式一致。`navigate` 命令里加 `--image-resize-mode stretch`(或与采集时相同)。
4. **缩小 close_threshold**:让 selected_node 更晚前推,减少"模型以为已经到了下一个节点"的误判。
5. **增大 radius**:让局部搜索窗更宽,容忍 closest_node 估计偏差,例如 `add --radius 6`(交互模式下当前不直接支持,需在 plan-file 里改)。
6. **降速**:`--pd-linear-scale 0.4 --pd-max-v 0.08`,给模型更多决策时间。
7. **看 profile_timing**:若 `infer` 耗时大且不稳,先解决推理速度;若 `total` 稳定但机器人方向乱,基本是模型/采集问题。

---

## 8. 运行结果保存

### 8.1 仅可视化(默认)

`--camera on` 即可,左实时 FPV、右当前 goal,不写视频。

### 8.2 保存 MP4

加 `--save-images`(参数名兼容旧命令,行为已改成保存视频):

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive --backend real --map real_hallway --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --pd-linear-scale 0.6 --pd-yaw-scale 0.6 \
  --pd-max-v 0.12 --pd-max-w 0.35 \
  --save-images
```

| 任务类型 | 视频内容 |
|---|---|
| `navigate` / `mission` 有目标图像 | 实时 FPV + 当前 goal 左右拼接 |
| `explore` / `keyboard` 等无目标 | 仅实时 FPV |

视频路径会在任务启动和结束时打印,格式:

```text
results/deployment/<session>/videos/<timestamp>_<label>/navigation_record.mp4
```

旁边还会写 `recording_meta.json`,含完整 `video_path`。

### 8.3 无可视化(正式实验)

```bash
... --no-gui --camera off --save-images
```

| 选项 | 效果 |
|---|---|
| `--no-gui` | 不弹 OpenCV 窗口 |
| `--camera off` | 桥接层不打开相机(此时不能跑 navigate/explore,因为模型需要图像) |
| `--save-images` | 仍写视频(但 `--camera off` 时实时画面是空的,通常仅用于纯命令脚本测试) |

> 实战常用组合:`--camera on --no-gui --save-images` — 推理用相机,但不弹窗,后台写视频。

### 8.4 切换其他真实场景

```bash
mkdir -p deployment/topomaps/images/lab_corridor

python scripts/deployment/nomad_navigation_host.py \
  --interactive --backend real --map lab_corridor --camera on ...
```

进入交互后:

```text
navigate --topomap-dir deployment/topomaps/images/lab_corridor --max-steps 200 ...
```

### 8.5 运行目录布局

每次启动主机会建一个新目录:

```text
results/deployment/<timestamp>_lite3_real_interactive/
├── captures/     # 执行 capture 的图像
├── goal_views/   # navigate 任务的当前 goal 图像
└── videos/<timestamp>_..../
    ├── navigation_record.mp4
    └── recording_meta.json
```

```bash
ls results/deployment
find results/deployment/<latest_run_dir> -maxdepth 2 -type f | head
```

---

## 9. 常见问题排查

### 9.1 相机打不开

```bash
ls /dev/video*
v4l2-ctl --list-devices
fuser -v /dev/video0          # 看占用
sudo fuser -k /dev/video0     # 强制释放占用
python scripts/deployment/lite3_real_bridge.py --test-camera-only --camera-device 0
```

`OrinCamera` 打开时会重试 3 次。仍失败时拔插 USB 或换 `--camera-device`。

### 9.2 Orin 推理太慢

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

任务命令里改:

```text
explore --max-steps 30 --scheduler ddim --ddim-steps 2 --cfg-weight 0.0
```

### 9.3 真机画面帧率低或拖拽感强

> ⚠️ **先看 [§2.4](#24-nomachine-看到的画面为什么会滞后):软件层 NoMaD 推理用的一定是最新帧。** NoMachine 上看到的延迟是显示链路的,不影响控制环。

**排查相机采集 vs 显示的方法:**

```text
status
```

会打印:

```text
Camera: camera_fps=28.7, frame_age=0.012s, frames=1234
```

| 现象 | 含义 | 处理 |
|---|---|---|
| `camera_fps` 远低于设定值 | 相机采集瓶颈 | 检查 USB 是否支持 MJPG: `v4l2-ctl --device=/dev/video0 --list-formats-ext`;不支持就把 `camera_fourcc` 改成 `null` |
| `camera_fps` 正常,窗口显示卡 | NoMachine / OpenCV 显示瓶颈 | 降 `--camera-viewer-fps 8`,或加 `--no-async-camera-viewer` 走串行 |
| `frame_age > 0.5s` | `read_bgr()` 会抛异常 | 多半是相机硬件问题或 USB 抖动 |

**完整 timing 排查:**

主机启动加 `--profile-timing --profile-interval 5`,导航时会按周期打印:

```text
[Timing] navigate tick=10 camera=2.1ms ui=0.2ms preprocess=1.0ms
         infer=148.6ms command=0.5ms total=152.4ms
```

| 列 | 偏大说明 |
|---|---|
| `camera` | 主循环取图慢,后端问题 |
| `ui` | 异步显示禁用或走串行,正常 ≈ 0 |
| `preprocess` | 图像 resize 异常,topomap 应已缓存 |
| `infer` | 模型推理慢,改 ddim-steps |
| `command` | 桥接 / UDP 异常,正常 < 5ms |

**正式实验不看窗口:** `--camera on --no-gui` 或 `--camera on --no-gui --save-images`。

### 9.4 速度指令太大或机器人动作太猛

两处可调:

| 层 | 参数 | 文件 / 位置 |
|---|---|---|
| 桥接最终限幅 | `max_linear_x` / `max_yaw_rate` | `lite3_real_bridge_config.json` |
| PD 中层缩放 + 限幅 | `--pd-linear-scale` / `--pd-yaw-scale` / `--pd-max-v` / `--pd-max-w` | 启动主机命令行 |

**首轮真机推荐组合:**

```text
--pd-linear-scale 0.6 --pd-yaw-scale 0.6 --pd-max-v 0.12 --pd-max-w 0.35
```

仍过快 → `pd-linear-scale 0.4` 或 `pd-max-v 0.08`。只是转向猛 → 单独降 `pd-yaw-scale` 或 `pd-max-w`,不要同时大幅降低线速度(否则只剩转弯)。

### 9.5 `navigate` 启动报 topomap 错误

| 报错 | 检查 |
|---|---|
| 漏 `--topomap-dir` | 命令里显式写出 |
| 目录不存在 | `ls deployment/topomaps/images/<name>` |
| domain 不匹配 | `topomap_meta.json` 是否 `"domain": "real"` |
| 含仿真图像 | 重新用 Orin 相机采集 |

### 9.6 没有保存 `videos/`

需要同时满足:

1. 启动主机时带 `--save-images`。
2. 实际执行了 `explore` 或 `navigate` 等循环任务(只 idle / capture 不会触发录像)。

视频路径会在任务启动和结束时打印。

### 9.7 任务报 `No module named 'torchvision'`

桥接通信本身没问题,是任务系统加载旧版 MuJoCo 工具脚本时触发了历史 `torchvision` import。**当前真机链路不依赖 torchvision**。

> 不要直接 `pip install torchvision`,会替换 Jetson 版 `torch`。

验证当前代码已无该依赖:

```bash
python - <<'PY'
import sys
sys.path.insert(0, "scripts/simulation")
from lite3_system.legacy_bridge import load_legacy
legacy = load_legacy()
print("legacy loaded")
print("pd", legacy.pd_controller([1.0, 0.0]))
PY
```

预期输出 `legacy loaded` 与一组 PD 数值。仍报错 → 同步代码到最新提交。

### 9.8 输入 `stand` 后立刻软急停

旧版状态机把 `completed` 也当 `safe_stop`,触发 `soft_estop`。**当前代码已修复:**

| 状态 | 行为 |
|---|---|
| `completed`(普通任务结束) | `controlled_stop()` 仅发零速度 |
| `estop` / `failed` | `safe_stop()` 调用 `soft_estop` |

修复后 `stand --stand-steps 20` 应正常结束并回 idle,**不再**打印 `⚠️ 软急停!`。

---

## 10. 快速启动序列

```bash
# 1. 环境与文件检查
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3

# 2. 相机
python scripts/deployment/orin_standalone_test.py --test camera --camera-device 0

# 3. 模型加载
python scripts/deployment/orin_standalone_test.py --test model \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth

# 4. 推理基准
python scripts/deployment/orin_standalone_test.py --test benchmark \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth --ddim-steps 5

# 5. 采集真实 topomap (可选: 另开终端启动 keyboard 模式 + --camera off)
python scripts/deployment/capture_real_topomap.py \
  --output-dir deployment/topomaps/images/real_hallway \
  --map-name real_hallway --camera-device 0 --count 40 --manual

# 6. 桥接通讯
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.2.1 --camera-device 0

# 7. 起立
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.2.1 --camera-device 0 --test-standup

# 8. 低速前进
python scripts/deployment/lite3_real_bridge.py --robot-ip 192.168.2.1 --camera-device 0 --test-standup --test-twist

# 9. 交互式主机
python scripts/deployment/nomad_navigation_host.py \
  --interactive --backend real --map real_hallway --camera on \
  --bridge-module deployment.lite3_real_bridge \
  --bridge-class Lite3RealBridge \
  --bridge-config scripts/configs/navigation_host/lite3_real_bridge_config.json \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --save-images
```

进入交互模式后:

```text
stand --stand-steps 20
status
keyboard      # W/S/A/D/Q/E 控制,Esc 回 idle
capture
explore --max-steps 30 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
navigate --topomap-dir deployment/topomaps/images/real_hallway \
         --max-steps 200 --scheduler ddim --ddim-steps 5 --cfg-weight 0.0
estop
```

---

## 11. 关键结论与检查清单

### 11.1 必须遵守的口径

1. **navigate 必须显式 `--topomap-dir`**,真机不会自动回退。
2. **真实 topomap 必须 Orin 相机采集**,目录中需有 `topomap_meta.json` 含 `"domain": "real"`。
3. **真机 `--map` 默认 `real_hallway`**,其他真实场景同级目录。`easy/medium/hard` 仅 MuJoCo。
4. **`bridge_kwargs.robot_ip` 必须写运动主机地址(`192.168.2.1`)**,绝不能写 Orin 自身。
5. **`--bridge-config` 用项目自带 JSON**,代码已兼容 `bridge_kwargs` 包装格式。默认限幅 `max_linear_x=0.12 / max_yaw_rate=0.35`,默认 `v4l2 + MJPG`。
6. **首轮真机用 `image_resize_mode=stretch`**;闭环出现横向几何异常再测 `center_crop` / `letterbox`。
7. **`--save-images` 现保存 MP4**;有目标的 navigate/mission 是 FPV+goal 拼接,无目标任务为单 FPV。
8. **`captures/` `goal_views/` `videos/`** 都对应实际代码路径,不是文档设计。
9. **交互式主机启动即自动起立**(`InteractiveNavigationHost.run()` 进 idle 前调用 `_ensure_standing()`)。回车前必须做完安全准备。
10. **机器人动作过猛先调 PD**,再调桥接最终限幅。

### 11.2 调试 navigate 不准的优先顺序

按 [§2.5](#25-navigate-不准的可能原因按概率排序):

1. 起点视野 vs `topomap/000.png` 是否对齐
2. topomap 节点间距 0.5-1.0 m
3. `image_resize_mode` 与采集一致
4. 真实场景 vs 训练分布差异
5. `pd-linear-scale` 降到 0.4
6. WiFi 链路 ping 延迟与丢包

### 11.3 调试 NoMachine 滞后的优先顺序

> **再次提醒:这与控制环和模型推理无关。**

1. `--camera-viewer-fps 8`(默认 15,降低到 8 通常更稳)
2. `--no-async-camera-viewer` 改用串行显示
3. NoMachine 客户端降低分辨率/编码质量
4. 改用有线网络
5. 最终方案:`--camera on --no-gui --save-images`,不看实时图像,只看视频复盘

### 11.4 推荐流程小结

独立调试 → 通讯仅连接 → `--test-standup/--test-twist` → 交互式主机(`--camera off` 进 keyboard 采 topomap) → 切回 `--camera on` 做探索与基于真实 topomap 的单目标导航。
