# 自绘图 AI 生成提示词

> 用途：为论文中“自绘图”类缺失图片提供图像生成提示词。
> 更新时间：2026-04-28
>
> 建议流程：先用 AI 生成干净的架构草图，再用 draw.io/PPT/TikZ 人工重排文字与箭头。图像生成模型不一定能稳定写中文，因此可要求“少量英文标签”，最后手工替换为论文中文术语。

## 通用风格要求

所有图统一采用：

- 白色或浅灰背景，学术论文信息图风格；
- 矢量图、扁平化、细线条、少阴影；
- 模块框 5 到 8 个为宜，避免过密；
- 箭头方向清楚，横向链路从左到右，分层链路从上到下；
- 主色建议：蓝色表示视觉/感知，绿色表示 topomap/距离预测，紫色表示扩散模型，橙色表示控制/速度指令，红色或深灰表示安全/急停/底层运控；
- 不要生成科幻背景、复杂纹理、卡通风格机器人；
- 文字尽量短，例如 Camera、Topomap、Visual Encoder、Distance Predictor、Diffusion Policy、PD Control、UDP Bridge、Lite3。

---

## 图1.1 研究背景与目标场景示意图

Label：`fig:intro-background-placeholder`

提示词：

```text
Create a clean academic vector-style illustration for a thesis introduction. The topic is visual goal navigation for a quadruped robot. Show a Lite3-like quadruped robot in an indoor corridor, with a front-facing camera view, a target location ahead, and a small topomap image sequence floating above the path. Add subtle arrows showing the robot moving from start to goal using visual observations. Include three conceptual labels: Visual Observation, Goal Image / Topomap, Robot Action. Use white background, blue and green accents, simple line art, no photorealistic rendering, no cartoon style, suitable for an undergraduate thesis figure.
```

建议后期标注：真实视觉观测、目标图像 / topomap、扩散策略、四足机器人执行。

---

## 图1.2 本文总体技术路线与章节关系图

Label：`fig:intro-roadmap-placeholder`

提示词：

```text
Create a clean thesis roadmap diagram with five connected stages from left to right. Stage 1: NoMaD baseline and problem definition. Stage 2: inference-side optimization, including DDIM, CFG, TTS, and visual encoder comparison. Stage 3: unified closed-loop framework, including inference module, state machine, navigation host, and platform interface. Stage 4: MuJoCo Lite3 simulation validation. Stage 5: Jetson Orin and Lite3 real-robot deployment. Use a horizontal flowchart with chapter numbers above each stage, simple icons, academic vector style, white background, consistent blue-green-purple-orange color palette.
```

建议后期标注：第二章前置知识、第三章算法实验、第四章仿真闭环、第五章真机部署。

---

## 图2.1 导航任务输入输出与 topomap 关系示意图

Label：`fig:prelim-nav-pipeline-placeholder`

提示词：

```text
Create a vector diagram explaining visual navigation with topomap. On the left, show a robot receiving a sequence of recent camera frames: I(t-3), I(t-2), I(t-1), I(t). Above or below, show an ordered topomap sequence I0, I1, I2, ..., Ig. Highlight a sliding window around the current closest node and mark one selected subgoal node. In the center, show a NoMaD policy block. On the right, show predicted local waypoints and a final velocity command. Make clear that the final goal is the sequence endpoint, while the real-time subgoal is selected inside the sliding window. Academic vector style, white background, minimal text, blue for observations, green for topomap, purple for policy, orange for commands.
```

建议后期标注：历史观察、局部窗口、closest node、selected subgoal、waypoint。

---

## 图2.2 扩散模型、扩散策略、ViT 编码器与 NoMaD 技术演化关系

Label：`fig:prelim-diffusion-placeholder`

提示词：

```text
Create an academic conceptual evolution diagram from left to right. Module 1: DDPM / DDIM diffusion model, showing noise-to-sample denoising steps. Module 2: classifier-free guidance, showing conditional and unconditional branches merging. Module 3: diffusion policy, showing trajectory or waypoint generation instead of image generation. Module 4: visual encoder and Transformer, showing image tokens and goal token. Module 5: NoMaD visual navigation system, showing observation images plus goal image generating waypoints. Use a clean horizontal timeline layout, simple arrows, white background, subtle blue-purple palette, no equations except short labels.
```

建议后期标注：DDPM/DDIM、CFG、Diffusion Policy、ViNT/ViT、NoMaD。

---

## 图2.3 Lite3 平台与控制链路示意图

Label：`fig:prelim-lite3-placeholder`

提示词：

```text
Create a layered control diagram for a Lite3 quadruped robot visual navigation platform. Top layer: high-level visual policy running at low frequency. Middle layer: velocity command interface and safety checks. Bottom layer: Lite3 locomotion controller and joint-level execution. On the side, show camera observation going upward to the high-level policy and twist command going downward to the robot. Include different frequencies conceptually: vision frequency, velocity refresh frequency, joint control frequency. Use a clean vertical layered architecture style, white background, simple robot silhouette at the bottom, blue/orange/gray colors.
```

建议后期标注：高层视觉策略、中层速度接口、底层步态控制、相机观测、twist 指令。

---

## 图3.1 NoMaD 算法研究对象与数据流示意图

Label：`fig:algo-overview-placeholder`

提示词：

```text
Create a NoMaD algorithm dataflow diagram. Inputs on the left: four recent RGB observation frames and one goal image. Center block: Visual Encoder / ViNT Transformer. From the shared visual condition, split into two heads: Distance Predictor and Diffusion Policy Head. Distance Predictor outputs relative distance to topomap candidate. Diffusion Policy Head performs denoising trajectory generation and outputs a sequence of local waypoints. Add optional inference-side modules around the diffusion head: DDIM step reduction, CFG guidance, TTS candidate selection. Use academic vector style, white background, blue for encoder, green for distance head, purple for diffusion, orange for waypoint output.
```

建议后期标注：视觉编码器、距离预测头、扩散策略头、DDIM/CFG/TTS。

---

## 图4.1 统一推理模块、状态机与平台后端关系图

Label：`fig:framework-modules-placeholder`

提示词：

```text
Create a software architecture diagram for a unified visual navigation framework. Use three horizontal layers. Top: navigation host and task plan, including navigate, explore, keyboard, estop. Middle: state machine, context buffer, NoMaD inference module, PD waypoint-to-command mapping. Bottom: platform backend abstraction with two branches: MuJoCo backend and real Lite3 backend. Show that both backends provide camera image, pose, safety status, and receive velocity command. Use clean boxes and arrows, white background, academic vector style, blue for inference, orange for control, gray for backend, red for safety.
```

建议后期标注：Navigation Host、State Machine、NoMaD Inference、PD Control、MuJoCo Backend、Real Backend。

---

## 图4.2 状态机状态流转图

Label：`fig:framework-fsm-placeholder`

提示词：

```text
Create a finite state machine diagram for a quadruped visual navigation system. Use circular or rounded rectangular nodes. States: idle, stand, navigate, explore, keyboard, recovery, completed, failed, estop. Show main transitions: idle to stand; stand to navigate or explore; navigate/explore to completed; navigate/explore to recovery when stuck; recovery back to navigate/explore; any running state to estop; failure checks to failed. Use clean arrows, white background, no decorative elements. Highlight estop in red, completed in green, active control states in blue/orange.
```

建议后期标注：idle、stand、navigate、explore、recovery、completed、failed、estop。

---

## 图5.1 Orin 与 Lite3 真机部署总体架构图

Label：`fig:real-overview-placeholder`

提示词：

```text
Create a thesis-style overall architecture diagram for real-robot deployment with Jetson Orin and Lite3. The figure must emphasize two crossed chains. Horizontal chain: Camera Images -> Visual Encoder -> Distance Predictor -> Diffusion Policy -> PD Mapping -> Twist Command. Vertical chain: High-level NoMaD Host on Jetson Orin -> State Machine and PD Control -> UDP Bridge with heartbeat -> Lite3 Motion Controller -> Lite3 Robot Locomotion. Add a small topomap sequence with a highlighted sliding window and selected subgoal feeding the policy. Use white background, vector style, blue for visual modules, green for topomap and distance prediction, purple for diffusion, orange for PD/twist, red or dark gray for safety and low-level control.
```

备注：这张图应与第五章开头“双链路设计”直接对应。

---

## 图5.2 Orin 推理、桥接层与 Lite3 运动主机数据流图

Label：`fig:real-bridge-placeholder`

提示词：

```text
Create a clean dataflow diagram focused on Jetson Orin to Lite3 bridge. Left side: Orin camera and NoMaD inference. Middle: waypoint-to-command mapping, velocity clamp, NaN/Inf guard, state machine safety check. Right side: UDP bridge, heartbeat thread at 5 Hz, twist refresh thread at 25 Hz, Lite3 motion host, locomotion controller. Show that NoMaD inference runs slower than twist refresh, so the bridge repeats the latest limited command between inference steps. Use layered arrows, white background, academic vector style. Use orange for velocity command, red for safety checks, gray for UDP and low-level motion host.
```

建议后期标注：心跳、速度刷新、软急停、最近一次速度命令。

---

## 图5.3 真实环境 topomap 采集与目标图像组织示意图

Label：`fig:real-topomap-placeholder`

提示词：

```text
Create a diagram explaining real-world topomap collection and organization. Show a corridor path from start to goal. Along the path, place a sequence of camera snapshot thumbnails labeled 000.png, 001.png, 002.png, ..., goal.png. Show a person or robot-mounted camera collecting images in order. Then show the saved directory structure: deployment/topomaps/images/real_hallway/ with ordered PNG files and topomap_meta.json. On the navigation side, show the robot starting near 000.png, a sliding window over nearby topomap nodes, and a highlighted real-time subgoal. White background, clean vector style, green for topomap sequence, blue for camera images, orange arrows for robot movement.
```

建议后期标注：按通行方向采集、节点顺序、滑动窗口、实时子目标。

