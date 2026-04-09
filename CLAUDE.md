# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VisuaNav-Transformer implements three general-purpose goal-conditioned visual navigation models from Berkeley AI Research: **GNM**, **ViNT**, and **NoMaD**. These models are trained on cross-embodiment data and can control diverse robots in zero-shot (LoCoBot, TurtleBot2, DJI Tello, Unitree A1, etc.).

## Environment Setup

```bash
# Training environment
conda env create -f train/train_environment.yml
conda activate vint_train
pip install -e train/

# Required external dependency (for NoMaD diffusion)
git clone git@github.com:real-stanford/diffusion_policy.git
pip install -e diffusion_policy/

# Deployment environment (separate, for robot deployment)
conda env create -f deployment/deployment_environment.yaml
conda activate vint_deployment
pip install -e train/
```

Python 3.8.5, PyTorch with CUDA 10+. Key deps: diffusers==0.11.1, efficientnet-pytorch, vit-pytorch, wandb.

## Common Commands

### Training

All training commands run from inside `train/`:

```bash
# Train ViNT (default)
python train.py -c config/vint.yaml

# Train other models
python train.py -c config/gnm.yaml
python train.py -c config/nomad.yaml
```

Config merging: `config/defaults.yaml` is loaded first, then the specified config overrides it. Don't train directly with `defaults.yaml`.

### Robot Deployment

All deployment scripts run from inside `deployment/src/`:

```bash
./record_bag.sh <bag_name>                              # Collect demo trajectory
./create_topomap.sh <topomap_name> <bag_filename>      # Build topological map
./navigate.sh "--model <model_name> --dir <topomap_dir>" # Deploy navigation
./exploration.sh "--model <model_name>"                  # Deploy exploration (NoMaD only)
```

### Experiment Scripts

Scripts in `scripts/` are standalone experiment files (offline inference, DDIM acceleration, classifier-free guidance, DINOv2 adaptation, ablation studies). Each has its own dependencies listed in `scripts/requirements.txt`.

## Architecture

### Three Model Families

- **GNM** (`train/vint_train/models/gnm/`): MobileNetV2 encoder, outputs distance prediction + action trajectory. Simplest model.
- **ViNT** (`train/vint_train/models/vint/`): EfficientNet-b0 encoder + multi-head attention transformer decoder. Foundation model.
- **NoMaD** (`train/vint_train/models/nomad/`): ViNT-based vision encoder + ConditionalUnet1D diffusion head (from diffusion_policy). Uses DDPM noise scheduler, goal masking for exploration.

All models inherit from `BaseModel` (in `base_model.py`) and share: `context_size` (number of history frames), `len_traj_pred` (waypoint prediction length), `learn_angle` (whether to predict heading).

### Training Pipeline

`train.py` is the single entry point. It:
1. Merges `defaults.yaml` with user config
2. Creates `ViNT_Dataset` instances per dataset (handles all three model types)
3. Instantiates model based on `model_type` field in config
4. Dispatches to `train_eval_loop()` (GNM/ViNT) or `train_eval_loop_nomad()` (NoMaD)

Logs and checkpoints go to `train/logs/<project_name>/<run_name>_<timestamp>/`. Checkpoint loading uses `load_run: <project_name>/<log_run_name>` in config, expecting `latest.pth` in that path.

### Data Format

Each trajectory is a directory of numbered JPEGs + `traj_data.pkl` (dict with `"position"` [T,2] and `"yaw"` [T]). Data splits are `traj_names.txt` files under `train/vint_train/data/data_splits/<dataset>/train|test/`. Custom datasets must also be registered in `train/vint_train/data/data_config.yaml` with their `metric_waypoints_distance`.

Images are normalized with ImageNet statistics (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).

### Deployment Pipeline

The deployment stack is ROS-based (noetic). The inference loop in `deployment/src/navigate.py`: reads camera images from `/usb_cam/image_raw` -> runs model forward pass -> publishes waypoints to `/waypoint` -> PD controller converts to velocity commands. Model configs are in `deployment/config/models.yaml`, robot params in `deployment/config/robot.yaml`.

### Key Config Files

- `train/config/*.yaml` - Training hyperparameters per model type
- `train/vint_train/data/data_config.yaml` - Dataset-specific waypoint spacing and action stats
- `deployment/config/models.yaml` - Maps model names to checkpoint paths
- `deployment/config/robot.yaml` - Robot-specific velocity limits and ROS topics

## Important Patterns

- The `alpha` parameter (default 0.5) controls the tradeoff between distance prediction loss and action prediction loss
- NoMaD uses `goal_mask_prob` for classifier-free guidance during training (masks goal to enable exploration mode)
- Context frames use either `temporal` (sequential) or `randomized` sampling via `context_type` config
- Wandb entity is hardcoded as `"gnmv2"` in `train.py` -- change this for your own runs
- Multi-GPU uses `nn.DataParallel` via `gpu_ids` config list
