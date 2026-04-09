# NoMaD 中期冲刺实验脚本

> **环境要求**：Ubuntu 22.04 · Python 3.8 · RTX 4090 (24 GB) · CUDA 12.1

## 脚本列表

| 脚本 | Day | 功能 | 依赖 |
|------|-----|------|------|
| `check_dataset.py` | Day 1 | 数据集完整性检查 | — |
| `offline_inference.py` | Day 1 | NoMaD 离线推理基线 | 预训练权重 |
| `ddim_experiment.py` | Day 2 | DDIM 加速采样对比实验 | `offline_inference.py` |
| `realtime_inference.py` | Day 2 | 实时摄像头推理 | `offline_inference.py` |
| `cfg_experiment.py` | Day 3 | CFG 目标引导实验 | `offline_inference.py` |
| `gait_shake_robustness.py` | Day 3 | 步态抖动鲁棒性实验 | `offline_inference.py` |
| `nomad_vint_dinov2.py` | Day 4 | DINOv2 视觉编码器适配模块 | `timm` |
| `train_dinov2.py` | Day 4 | DINOv2 独立训练脚本 | `nomad_vint_dinov2.py` |
| `lite3_sim.py` | Day 5 | PyBullet 仿真 + 轮式 vs 足式对比 | `pybullet`（仿真机） |

## 统一运行方式

```bash
# SSH 连接训练机，激活环境
ssh yifei@<训练机IP>
conda activate nomad

# 所有脚本均在项目根目录下运行
cd <repo_root>
python scripts/<脚本名>.py
```

## 安装依赖

```bash
# 1. 安装 PyTorch (CUDA 12.1)
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# 2. 安装项目训练包
cd train && pip install -e .

# 3. 安装实验依赖
cd <repo_root>
pip install -r scripts/requirements.txt
```

## 输出目录

所有实验结果保存在 `results/` 目录下，按 Day 分组，每次运行生成带时间戳的子目录：

```
results/
├── day1/YYYYMMDD_HHMMSS_offline_inference/
├── day2/YYYYMMDD_HHMMSS_ddim_experiment/
├── realtime/YYYYMMDD_HHMMSS_realtime/
├── day3/YYYYMMDD_HHMMSS_cfg_experiment/
├── day3/YYYYMMDD_HHMMSS_gait_shake/
├── day4/  (DINOv2 训练日志在 train/logs/)
├── day5/  (仿真结果)
└── day6/  (消融实验汇总)
```
