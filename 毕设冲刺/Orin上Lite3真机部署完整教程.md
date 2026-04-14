# Orin 上 Lite3 真机部署完整教程

## 1. 文档定位

这份文档专门对应 Jetson Orin 上的 Lite3 真机部署。目标不是只让程序启动，而是形成一条完整可执行的部署链：

1. 在 Orin 上完成 NoMaD 推理。
2. 支持可视化视觉图像，也支持无可视化运行。
3. 能保存运行过程中的实时图像。
4. 能保存当前任务对应的目标图像。

---

## 2. 部署目标与边界

当前这条链路适合承担三类任务：

1. 静态感知检查。
2. 低速短程视觉目标导航。
3. 探索模式和任务切换验证。

当前真机链路的原则是：

1. 高层使用 NoMaD。
2. 中层使用 PD 控制映射。
3. 底层使用 Lite3 运控桥接。
4. 真机导航默认使用真实采集的 topomap 目录，不使用 MuJoCo 的随机目标生成。

---

## 3. 代码组成

### 3.1 关键入口

1. 导航主机：`scripts/deployment/nomad_navigation_host.py`
2. 状态机系统：`scripts/simulation/lite3_system/system.py`
3. 状态机定义：`scripts/simulation/lite3_system/states.py`
4. 真机平台桥接：`scripts/simulation/lite3_system/interfaces.py`
5. 推理模块：`scripts/shared/nomad_inference.py`
6. 部署检查清单：`scripts/deployment/nomad_real_deployment_checklist.py`

### 3.2 真机桥接必须提供的方法

当前 `ExternalBridgePlatform` 默认要求桥接类至少实现：

1. `render_camera`
2. `standup`
3. `get_pose`
4. `apply_command` 或 `send_command`

建议同时实现：

1. `get_height`
2. `get_forward_speed`
3. `is_fallen`
4. `viewer_alive`
5. `close`
6. `emergency_stop`

---

## 4. 部署前准备

### 4.1 模型与配置

准备以下内容：

1. 策略配置文件，例如 `scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml`
2. 对应 checkpoint，例如 `deployment/model_weights/nomad/nomad.pth`
3. 真实环境采集得到的 topomap 目录

### 4.2 真实 topomap

真机部署时应使用真实采集的 topomap 目录，而不是 MuJoCo 自动生成目录。当前代码对这一点已经做了域检查，避免误把仿真 topomap 拿去真机运行。

### 4.3 部署检查

推荐先执行：

```bash
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3
```

这一步的目标不是运行导航，而是先确认：

1. 权重路径存在。
2. 配置路径存在。
3. 顶层目录结构符合部署要求。

---

## 5. 可视化模式与无可视化模式

## 5.1 可视化调试模式

适合调试与录屏。特点是：

1. 打开相机显示窗口。
2. 可同时显示实时图像与目标图像。
3. 可通过交互式命令保存 capture。

推荐命令：

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --camera on \
  --bridge-module your_bridge_module \
  --bridge-class YourLite3Bridge \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

## 5.2 无可视化运行模式

适合正式闭环运行。特点是：

1. 不打开 GUI。
2. 仍可保持推理与控制闭环。
3. 可继续保存运行过程图像。

推荐命令：

```bash
python scripts/deployment/nomad_navigation_host.py \
  --interactive \
  --backend real \
  --no-gui \
  --camera off \
  --bridge-module your_bridge_module \
  --bridge-class YourLite3Bridge \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth
```

### 5.3 两种模式的区别

1. `--camera on/off` 控制是否显示统一相机窗口。
2. `--no-gui` 控制是否启用图形界面。
3. 推理模块、状态机和任务调度逻辑在两种模式下保持一致。

---

## 6. 图像保存机制

当前系统已经形成三类图像保存目录，全部位于一次 session 的结果目录下：

1. `results/deployment/<timestamp>_lite3_real_host/fpv/`
   用于保存运行过程中的实时图像帧。
2. `results/deployment/<timestamp>_lite3_real_host/goal_views/`
   用于保存当前任务对应的目标图像。
3. `results/deployment/<timestamp>_lite3_real_host/captures/`
   用于保存交互式人工 capture 的图像。

### 6.1 保存实时图像

只要在状态机参数中加入 `--save-fpv`，系统会定期把实时相机帧保存到 `fpv/` 目录。

### 6.2 保存目标图像

当前任务的 `goal_view` 会在进入任务后自动保存到 `goal_views/` 目录。这样后续无论是写实验记录、做论文图，还是排查“为什么没到达目标”，都能直接找到本次导航对应的目标图像。

### 6.3 保存交互式目标图像

在交互式模式中执行 `capture`，系统会把当下画面保存到 `captures/` 目录，并加入共享 session 队列中，便于后续切换和对照。

---

## 7. 推荐部署流程

## 7.1 第一层：静态感知检查

目标：

1. 确认相机图像可以正常读取。
2. 确认推理模块可正常加载权重。
3. 确认 GUI 或无 GUI 模式都能启动。

建议动作：

1. 先运行 `stand`
2. 再运行 `status`
3. 必要时执行 `capture`

## 7.2 第二层：原地闭环检查

目标：

1. 让系统完成相机采集、推理、控制指令生成的完整闭环。
2. 观察是否存在控制量突变、相机卡顿或明显延迟。

建议命令：

1. 进入交互模式。
2. 先执行 `stand`。
3. 再执行一次短步数 `navigate --max-steps 30`。

## 7.3 第三层：低速短距离导航

目标：

1. 验证状态机 `stand -> navigate -> completed/failed` 流程。
2. 验证目标图像显示与实际到达行为是否一致。

建议使用真实 topomap 的短路线。

## 7.4 第四层：正式短程导航

目标：

1. 验证多次重复运行的稳定性。
2. 保留 `fpv/`、`goal_views/`、`captures/` 和运行摘要。

正式阶段建议默认：

1. 低速
2. `DDIM-2`
3. 经过 MuJoCo 验证的 checkpoint
4. 必须保留 `estop` 通道

---

## 8. 交互式导航主机使用方式

启动交互模式后，可使用以下命令：

1. `stand`
2. `navigate`
3. `explore`
4. `estop`
5. `capture`
6. `status`
7. `help`
8. `quit`

推荐工作流是：

1. 启动后先 `stand`
2. 执行 `status`
3. 如需保留当前场景视图，执行 `capture`
4. 再执行 `navigate`
5. 如需无目标探索，执行 `explore`
6. 任何异常立即执行 `estop`

---

## 9. 关于“探索模式”和“多目标导航”的说明

### 9.1 探索模式

探索模式不是“随机乱走”，而是使用 goal mask 为空时的策略输出推进 waypoint 队列。但真机上仍应增加额外工程约束：

1. 低速上限
2. 区域边界限制
3. 时间上限
4. 人工监护与急停

### 9.2 多目标导航

当前真机系统正确的多目标实现方式，不是让 NoMaD 一次预测全部目标顺序，而是：

1. 导航主机维护任务队列。
2. 状态机逐个执行子目标。
3. 每一段子目标都由当前 `goal_view` 和 topomap 支持。

---

## 10. 推荐命令模板

### 10.1 单目标导航

```bash
python scripts/deployment/nomad_navigation_host.py \
  --backend real \
  --bridge-module your_bridge_module \
  --bridge-class YourLite3Bridge \
  --camera on \
  --policy-config scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml \
  --policy-checkpoint deployment/model_weights/nomad/nomad.pth \
  --mode navigate \
  --topomap-dir deployment/topomaps/images/your_real_topomap \
  --scheduler ddim --ddim-steps 2 --cfg-weight 0.5 --save-fpv
```

### 10.2 Mission 计划模式

```bash
python scripts/deployment/nomad_navigation_host.py \
  --backend real \
  --bridge-module your_bridge_module \
  --bridge-class YourLite3Bridge \
  --plan-file scripts/configs/navigation_host/lite3_navigation_host_plan.json \
  --save-plan
```

---

## 11. 最终结论

当前 Orin 真机部署链路已经具备以下能力：

1. 在 Orin 上加载 NoMaD 策略并完成推理。
2. 支持有可视化和无可视化两种运行模式。
3. 支持保存实时图像、目标图像和人工 capture 图像。
4. 支持通过统一导航主机管理任务切换与状态机执行。

因此，这一部分已经可以单独作为 Lite3 真机部署章节的完整教程与系统实现说明。
