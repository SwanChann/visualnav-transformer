# 通用导航模型：GNM、ViNT 和 NoMaD

**贡献者**：Dhruv Shah、Ajay Sridhar、Nitish Dashora、Catherine Glossop、Kyle Stachowicz、Arjun Bhorkar、Kevin Black、Noriaki Hirose、Sergey Levine

_伯克利人工智能研究院（Berkeley AI Research）_

[项目主页](https://general-navigation-models.github.io) | [引用方式](https://github.com/robodhruv/visualnav-transformer#citing) | [预训练模型](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)

---

通用导航模型（General Navigation Models）是基于目标条件的通用视觉导航策略，在多样化的跨实体训练数据上进行训练，能够零样本控制多种不同的机器人。它们还可以高效地微调或适配到新机器人和下游任务。我们的模型系列在以下研究论文中进行了描述（并持续增长中）：
1. [GNM: A General Navigation Model to Drive Any Robot](https://sites.google.com/view/drive-any-robot)（_2022年10月_，发表于 ICRA 2023）
2. [ViNT: A Foundation Model for Visual Navigation](https://general-navigation-models.github.io/vint/index.html)（_2023年6月_，发表于 CoRL 2023）
3. [NoMaD: Goal Masking Diffusion Policies for Navigation and Exploration](https://general-navigation-models.github.io/nomad/index.html)（_2023年10月_）

## 概述
本仓库包含使用自定义数据训练模型系列的代码、预训练模型检查点，以及在 TurtleBot2/LoCoBot 机器人上部署的示例代码。仓库结构遵循 [GNM](https://github.com/PrieureDeSion/drive-any-robot) 的组织方式。

- `./train/train.py`：训练脚本，用于在自定义数据上训练或微调 ViNT 模型。
- `./train/vint_train/models/`：包含 GNM、ViNT 和一些基线模型的模型文件。
- `./train/process_*.py`：将 rosbag 或其他格式的机器人轨迹处理成训练数据的脚本。
- `./deployment/src/record_bag.sh`：在机器人目标环境中收集演示轨迹作为 ROS bag 的脚本。该轨迹会被下采样以生成环境的拓扑图。
- `./deployment/src/create_topomap.sh`：将演示轨迹的 ROS bag 转换为机器人可用于导航的拓扑图的脚本。
- `./deployment/src/navigate.sh`：在机器人上部署训练好的 GNM/ViNT/NoMaD 模型，导航到生成的拓扑图中的期望目标的脚本。相关配置设置请参见下面的相关章节。
- `./deployment/src/explore.sh`：在机器人上部署训练好的 NoMaD 模型，随机探索环境的脚本。相关配置设置请参见下面的相关章节。

## 训练

该子文件夹包含用于处理数据集和从自定义数据训练模型的代码。

### 前置条件

代码库假定可以访问运行 Ubuntu（在 18.04 和 20.04 上测试过）、Python 3.7+ 和配备 CUDA 10+ 的 GPU 的工作站。它还假定可以访问 conda，但你可以修改它以与其他虚拟环境包或原生设置配合使用。

### 设置
在 `vint_release/`（最顶层）目录中运行以下命令：

1. 设置 conda 环境：
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. 激活 conda 环境：
    ```
    conda activate vint_train
    ```
3. 安装 vint_train 包：
    ```bash
    pip install -e train/
    ```
4. 从此[仓库](https://github.com/real-stanford/diffusion_policy)安装 `diffusion_policy` 包：
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```

### 数据处理
在[论文](https://general-navigation-models.github.io)中，我们在公开可用和未发布的数据集组合上进行训练。以下是用于训练的公开可用数据集列表；请联系相应的作者以访问未发布的数据。
- [RECON](https://sites.google.com/view/recon-robot/dataset)
- [TartanDrive](https://github.com/castacks/tartan_drive)
- [SCAND](https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html#Links)
- [GoStanford2（修改版）](https://drive.google.com/drive/folders/1RYseCpbtHEFOsmSX2uqNY_kvSxwZLVP_?usp=sharing)
- [SACSoN/HuRoN](https://sites.google.com/view/sacson-review/huron-dataset)

我们建议你下载这些数据集（以及你可能想要训练的任何其他数据集）并运行下面的处理步骤。

#### 数据处理步骤

我们提供了一些示例脚本来处理这些数据集，可以直接从 rosbag 或像 HDF5 这样的自定义格式处理：
1. 使用相关参数运行 `process_bags.py`，或运行 `process_recon.py` 来处理 RECON HDF5 文件。你也可以按照我们下面的结构手动添加自己的数据集（如果要添加自定义数据集，请查看[自定义数据集](#自定义数据集)部分）。
2. 使用相关参数在数据集文件夹上运行 `data_split.py`。

完成数据处理的第 1 步后，处理后的数据集应具有以下结构：

```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_1.jpg
│   │   └── traj_data.pkl
│   ├── <name_of_traj2>
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── T_2.jpg
│   │   └── traj_data.pkl
│   ...
└── └── <name_of_trajN>
    	├── 0.jpg
    	├── 1.jpg
    	├── ...
        ├── T_N.jpg
        └── traj_data.pkl
```  

每个 `*.jpg` 文件包含来自机器人的前向 RGB 观测，并按时间标记。`traj_data.pkl` 文件是轨迹的里程计数据。它是一个包含以下键的 pickle 字典：
- `"position"`：一个 np.ndarray [T, 2]，表示每个图像观测时机器人的 xy 坐标。
- `"yaw"`：一个 np.ndarray [T,]，表示每个图像观测时机器人的偏航角。

完成数据处理的第 2 步后，处理后的数据分割应在 `vint_release/train/vint_train/data/data_splits/` 中具有以下结构：

```
├── <dataset_name>
│   ├── train
|   |   └── traj_names.txt
└── └── test
        └── traj_names.txt 
``` 

### 训练通用导航模型
在 `vint_release/train` 目录中运行以下命令：
```bash
python train.py -c <path_of_train_config_file>
```
预制的配置 yaml 文件位于 `train/config` 目录中。

#### 自定义配置文件
你可以使用预制的 yaml 文件之一作为起点，并根据需要更改值。`config/vint.yaml` 是个不错的选择，因为它包含注释的参数。`config/defaults.yaml` 包含默认配置值（不要直接使用此配置文件进行训练，因为它没有指定任何训练数据集）。

#### 自定义数据集
确保你的数据集和数据分割目录遵循[数据处理](#数据处理步骤)部分提供的结构。找到 `train/vint_train/data/data_config.yaml` 并追加以下内容：

```
<dataset_name>:
    metric_waypoints_distance: <数据集中路标点之间的平均距离（米）>
```

找到你的训练配置文件并在 `datasets` 参数下添加以下文本（可以随意更改 `end_slack`、`goals_per_obs` 和 `negative_mining` 的值）：
```
<dataset_name>:
    data_folder: <数据集路径>
    train: data/data_splits/<dataset_name>/train/ 
    test: data/data_splits/<dataset_name>/test/ 
    end_slack: 0 # 从每个轨迹末尾截断的时间步数（以防许多轨迹以碰撞结束）
    goals_per_obs: 1 # 每个观测采样的目标数量
    negative_mining: True # 来自 ViNG 论文（Shah et al.）的负样本挖掘
```

#### 从检查点训练模型
你也可以从现有的检查点加载模型，而不是从头开始训练。
在 `vint_release/train/config/` 中的 .yaml 配置文件中添加 `load_run: <project_name>/<log_run_name>`。你要加载的 `*.pth` 文件需要保存在此文件结构中并重命名为 "latest"：`vint_release/train/logs/<project_name>/<log_run_name>/latest.pth`。这使得从先前运行的检查点训练变得容易，因为日志默认以这种方式保存。注意：如果要从先前运行的检查点加载，请检查 `vint_release/train/logs/<project_name>/` 中的运行名称，因为代码会在每个运行的配置 yaml 文件中指定的 run_name 后附加日期字符串，以避免重复的运行名称。

如果你想使用我们的检查点，可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载 `*.pth` 文件。

## 部署
该子文件夹包含加载预训练 ViNT 模型并将其部署在开源 [LoCoBot 室内机器人平台](http://www.locobot.org/)上的代码，该平台配备 [NVIDIA Jetson Orin Nano](https://www.amazon.com/NVIDIA-Jetson-Orin-Nano-Developer/dp/B0BZJTQ5YP/ref=asc_df_B0BZJTQ5YP/?tag=hyprod-20&linkCode=df0&hvadid=652427572954&hvpos=&hvnetw=g&hvrand=12520404772764575478&hvpone=&hvptwo=&hvqmt=&hvdev=c&hvdvcmdl=&hvlocint=&hvlocphy=1013585&hvtargid=pla-2112361227514&psc=1&gclid=CjwKCAjw4P6oBhBsEiwAKYVkq7dqJEwEPz0K-H33oN7MzjO0hnGcAJDkx2RdT43XZHdSWLWHKDrODhoCmnoQAvD_BwE)。它可以轻松适配到其他机器人上，研究人员已能够独立地将其部署在以下机器人上——Clearpath Jackal、DJI Tello、Unitree A1、TurtleBot2、Vizbot——以及像 CARLA 这样的模拟环境中。

### LoCoBot 设置

该软件在运行 Ubuntu 20.04 的 LoCoBot 上进行了测试。

#### 软件安装（按此顺序）
1. ROS：[ros-noetic](https://wiki.ros.org/noetic/Installation/Ubuntu)
2. ROS 包：
    ```bash
    sudo apt-get install ros-noetic-usb-cam ros-noetic-joy
    ```
3. [kobuki](http://wiki.ros.org/kobuki/Tutorials/Installation)
4. Conda 
    - 安装 anaconda/miniconda 等用于管理环境
    - 使用 environment.yml 创建 conda 环境（在 `vint_release/` 目录中运行）
        ```bash
        conda env create -f deployment/deployment_environment.yaml
        ```
    - 激活环境
        ```bash
        conda activate vint_deployment
        ```
    - （推荐）添加到 `~/.bashrc`：
        ```bash
        echo "conda activate vint_deployment" >> ~/.bashrc 
        ```
5. 安装 `vint_train` 包（在 `vint_release/` 目录中运行）：
    ```bash
    pip install -e train/
    ```
6. 从此[仓库](https://github.com/real-stanford/diffusion_policy)安装 `diffusion_policy` 包：
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
7. （推荐）如果未安装，请安装 [tmux](https://github.com/tmux/tmux/wiki/Installing)。
    许多 bash 脚本依赖 tmux 来启动多个屏幕并执行不同的命令。这对于调试很有用，因为你可以看到每个屏幕的输出。

#### 硬件要求
- LoCoBot：http://locobot.org（仅导航堆栈）
- 广角 RGB 摄像头：[示例](https://www.amazon.com/ELP-170degree-Fisheye-640x480-Resolution/dp/B00VTHD17W)。`vint_locobot.launch` 文件使用适用于 ELP 鱼眼广角等摄像头的参数，可以根据自己的情况进行修改。相应地调整 `vint_release/deployment/config/camera.yaml` 中的摄像头参数（用于可视化）。
- 适用于 Linux 的[操纵杆](https://www.amazon.com/Logitech-Wireless-Nano-Receiver-Controller-Vibration/dp/B0041RR0TW)/[键盘遥控](http://wiki.ros.org/teleop_twist_keyboard)。将操纵杆上 _deadman_switch_ 的索引映射添加到 `vint_release/deployment/config/joystick.yaml`。你可以在 [wiki](https://wiki.ros.org/joy) 中找到常见操纵杆的按钮到索引的映射。

### 加载模型权重

将模型权重 *.pth 文件保存在 `vint_release/deployment/model_weights` 文件夹中。我们的模型权重在[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)中。

### 收集拓扑地图

_确保在 `vint_release/deployment/src/` 目录中运行这些脚本。_

本节讨论为部署创建目标环境拓扑地图的简单方法。为简单起见，我们将使用"路径跟随"模式的机器人，即给定环境中的单个轨迹，任务是沿着相同的轨迹到达目标。环境可能有新的/动态障碍物、光照变化等。

#### 记录 rosbag：
```bash
./record_bag.sh <bag_name>
```

运行此命令以使用操纵杆和摄像头遥控机器人。此命令打开三个窗口：
1. `roslaunch vint_locobot.launch`：此启动文件打开摄像头的 `usb_cam` 节点、操纵杆的 joy 节点以及机器人移动底座的节点。
2. `python joy_teleop.py`：此 python 脚本启动一个节点，从 joy 主题读取输入并在主题上输出以遥控机器人底座。
3. `rosbag record /usb_cam/image_raw -o <bag_name>`：此命令不会立即运行（你必须按 Enter）。它将在 vint_release/deployment/topomaps/bags 目录中运行，我们建议你在那里存储 rosbag。

当你准备好记录 bag 时，运行 `rosbag record` 脚本并在你希望机器人跟随的地图上遥控机器人。当你完成记录路径后，终止 `rosbag record` 命令，然后终止 tmux 会话。

#### 制作拓扑地图：
```bash
./create_topomap.sh <topomap_name> <bag_filename>
```

此命令打开 3 个窗口：
1. `roscore`
2. `python create_topomap.py —dt 1 —dir <topomap_dir>`：此命令在 `/vint_release/deployment/topomaps/images` 中创建一个目录，并在播放 bag 时每秒将图像保存为地图中的节点。
3. `rosbag play -r 1.5 <bag_filename>`：此命令以 x5 速度播放 rosbag，因此 python 脚本实际上每 1.5 秒记录一次节点。`<bag_filename>` 应该是带有 .bag 扩展名的完整 bag 名称。你可以在 `make_topomap.sh` 文件中更改此值。该命令在你按 Enter 之前不会运行，只有在 python 脚本给出等待消息后才应按 Enter。播放 bag 后，移至 python 脚本运行的屏幕，以便在 rosbag 停止播放时可以终止它。

当 bag 停止播放时，终止 tmux 会话。

### 运行模型
#### 导航
_确保在 `vint_release/deployment/src/` 目录中运行此脚本。_

```bash
./navigate.sh "--model <model_name> --dir <topomap_dir>"
```

要部署已发布结果中的模型之一，我们发布了可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载的模型检查点。

`<model_name>` 是 `vint_release/deployment/config/models.yaml` 文件中模型的名称。在此文件中，你为每个模型指定这些参数（使用默认值）：
- `config_path`（str）：用于训练模型的 `vint_release/train/config/` 中 *.yaml 文件的路径
- `ckpt_path`（str）：`vint_release/deployment/model_weights/` 中 *.pth 文件的路径

确保这些配置与你训练模型时使用的配置匹配。我们提供的权重模型的配置已在 yaml 文件中提供供你参考。

`<topomap_dir>` 是 `vint_release/deployment/topomaps/images` 中目录的名称，该目录包含与拓扑地图中节点对应的图像。图像按名称从 0 到 N 排序。

此命令打开 4 个窗口：

1. `roslaunch vint_locobot.launch`：此启动文件打开摄像头的 usb_cam 节点、操纵杆的 joy 节点以及机器人移动底座的多个节点。
2. `python navigate.py --model <model_name> -—dir <topomap_dir>`：此 python 脚本启动一个节点，从 `/usb_cam/image_raw` 主题读取图像观测，将观测和地图输入模型，并将动作发布到 `/waypoint` 主题。
3. `python joy_teleop.py`：此 python 脚本启动一个节点，从 joy 主题读取输入并在主题上输出以遥控机器人底座。
4. `python pd_controller.py`：此 python 脚本启动一个节点，从 `/waypoint` 主题（来自模型的路标点）读取消息，并输出速度以导航机器人底座。

当机器人完成导航时，终止 `pd_controller.py` 脚本，然后终止 tmux 会话。如果你想在机器人导航时控制它，`joy_teleop.py` 脚本允许你使用操纵杆进行控制。

#### 探索
_确保在 `vint_release/deployment/src/` 目录中运行此脚本。_

```bash
./exploration.sh "--model <model_name>"
```

要部署已发布结果中的模型之一，我们发布了可以从[此链接](https://drive.google.com/drive/folders/1a9yWR2iooXFAqjQHetz263--4_2FFggg?usp=sharing)下载的模型检查点。

`<model_name>` 是 `vint_release/deployment/config/models.yaml` 文件中模型的名称（注意只有 NoMaD 适用于探索）。在此文件中，你为每个模型指定这些参数（使用默认值）：
- `config_path`（str）：用于训练模型的 `vint_release/train/config/` 中 *.yaml 文件的路径
- `ckpt_path`（str）：`vint_release/deployment/model_weights/` 中 *.pth 文件的路径

确保这些配置与你训练模型时使用的配置匹配。我们提供的权重模型的配置已在 yaml 文件中提供供你参考。

`<topomap_dir>` 是 `vint_release/deployment/topomaps/images` 中目录的名称，该目录包含与拓扑地图中节点对应的图像。图像按名称从 0 到 N 排序。

此命令打开 4 个窗口：

1. `roslaunch vint_locobot.launch`：此启动文件打开摄像头的 usb_cam 节点、操纵杆的 joy 节点以及机器人移动底座的多个节点。
2. `python explore.py --model <model_name>`：此 python 脚本启动一个节点，从 `/usb_cam/image_raw` 主题读取图像观测，将观测和地图输入模型，并将探索动作发布到 `/waypoint` 主题。
3. `python joy_teleop.py`：此 python 脚本启动一个节点，从 joy 主题读取输入并在主题上输出以遥控机器人底座。
4. `python pd_controller.py`：此 python 脚本启动一个节点，从 `/waypoint` 主题（来自模型的路标点）读取消息，并输出速度以导航机器人底座。

当机器人完成导航时，终止 `pd_controller.py` 脚本，然后终止 tmux 会话。如果你想在机器人导航时控制它，`joy_teleop.py` 脚本允许你使用操纵杆进行控制。

### 适配此代码到不同的机器人

我们希望此代码库足够通用，允许你将其部署到你喜欢的基于 ROS 的机器人上。你可以在 `vint_release/deployment/config/robot.yaml` 中更改机器人配置参数，如机器人的最大角速度和线速度以及用于遥控和控制机器人的主题。请随时创建 Github Issue 或通过 shah@cs.berkeley.edu 联系作者。

## 引用方式
```
@inproceedings{shah2022gnm,
  author    = {Dhruv Shah and Ajay Sridhar and Arjun Bhorkar and Noriaki Hirose and Sergey Levine},
  title     = {{GNM: A General Navigation Model to Drive Any Robot}},
  booktitle = {International Conference on Robotics and Automation (ICRA)},
  year      = {2023},
  url       = {https://arxiv.org/abs/2210.03370}
}

@inproceedings{shah2023vint,
  title     = {Vi{NT}: A Foundation Model for Visual Navigation},
  author    = {Dhruv Shah and Ajay Sridhar and Nitish Dashora and Kyle Stachowicz and Kevin Black and Noriaki Hirose and Sergey Levine},
  booktitle = {7th Annual Conference on Robot Learning},
  year      = {2023},
  url       = {https://arxiv.org/abs/2306.14846}
}

@article{sridhar2023nomad,
  author  = {Ajay Sridhar and Dhruv Shah and Catherine Glossop and Sergey Levine},
  title   = {{NoMaD: Goal Masked Diffusion Policies for Navigation and Exploration}},
  journal = {arXiv pre-print},
  year    = {2023},
  url     = {https://arxiv.org/abs/2310.xxxx}
}
```
