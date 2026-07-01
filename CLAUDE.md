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

### Data Processing

Before training, raw trajectories must be converted to the dataset layout and split. Run from inside `train/`:

```bash
python process_bags.py    # ROS bags → images + traj_data.pkl
python process_recon.py   # RECON HDF5 → images + traj_data.pkl
python data_split.py      # produces traj_names.txt under data/data_splits/<dataset>/{train,test}/
```

`process_bag_diff.py` is a diagnostic helper for inspecting bag differences.

### Training

```bash
python train.py -c config/vint.yaml          # ViNT
python train.py -c config/gnm.yaml           # GNM
python train.py -c config/nomad.yaml         # NoMaD (DDPM diffusion head)
python train.py -c config/nomad_dinov2.yaml  # NoMaD with DINOv2 vision encoder
python train.py -c config/late_fusion.yaml   # ViNT late-fusion variant
```

`train.py` resolves config paths and dataset paths against the cwd, `train/`, and the repo root (see `_resolve_existing_or_repo_path`/`_resolve_config_path`), so you can launch training from either `train/` or the repo root with the same arguments. Config merging: `config/defaults.yaml` is loaded first, then the specified config overrides it. Don't train directly with `defaults.yaml`.

### Tests & Linting

There is no automated test suite or linter configured for this repo. `train/setup.py` and `scripts/requirements.txt` are package-install manifests, not test runners. Validation is done via the `scripts/analysis/` (e.g. `check_dataset.py`, `offline_inference.py`, `realtime_inference.py`) and `scripts/experiments/` entrypoints, plus `scripts/tooling/env_check.py` and `path_audit.py` for environment / path sanity checks.

### Robot Deployment

All deployment scripts run from inside `deployment/src/`:

```bash
./record_bag.sh <bag_name>                               # Collect demo trajectory
./create_topomap.sh <topomap_name> <bag_filename>        # Build topological map
./navigate.sh "--model <model_name> --dir <topomap_dir>" # Deploy navigation
./explore.sh "--model <model_name>"                      # Deploy exploration (NoMaD only)
```

### Experiment Scripts

`scripts/` contains standalone experiments, organized into:
- `scripts/analysis/` — `offline_inference.py`, `realtime_inference.py`, `check_dataset.py`, `result_collector.py`, `thesis_result_summary.py`
- `scripts/experiments/` — DDIM acceleration (`ddim_experiment.py`), classifier-free guidance (`cfg_experiment.py`), TTS, gait-shake robustness, encoder comparison, ablations
- `scripts/models/` — DINOv2 backbone (`nomad_vint_dinov2.py`), backbone suite, encoder weight downloads
- `scripts/training/` — `train_dinov2.py`, backbone suite training, encoder seed sweep
- `scripts/shared/` — cross-script inference helpers (`nomad_eval_common.py`, `nomad_inference.py`); import from here rather than duplicating loaders
- `scripts/tooling/` — repo hygiene helpers (`env_check.py`, `path_audit.py`, `project_paths.py` which is the canonical path-resolution module)
- `scripts/configs/` — vision encoder and navigation host configs
- `scripts/deployment/` and `scripts/simulation/` — MuJoCo sim entrypoints (e.g. `nomad_mujoco_lite3_nav.py`, `nomad_mujoco_tron1_nav.py`, `mujoco_encoder_benchmark.py`)

Root-level scripts are kept as legacy-compatible entrypoints; prefer the categorized paths. See `scripts/README.md` for the authoritative inventory. Install deps via `scripts/requirements.txt`.

### Quadruped Deployment (Lite3)

Beyond the ROS LoCoBot stack in `deployment/`, the repo ships three Lite3-specific stacks:

- `Lite3_rl_deploy/` — C++/CMake deployment for the Unitree Lite3 quadruped (policy/, run_policy/, state_machine/, vendored MuJoCo + ONNX Runtime + MotionSDK under its own `third_party/`). Build via `cmake -S Lite3_rl_deploy -B Lite3_rl_deploy/build` then `cmake --build Lite3_rl_deploy/build`. Toggleable CMake options: `BUILD_PLATFORM` (`x86` for desktop, `arm` cross-compiles with `aarch64-linux-gnu-g++` for the on-robot SDK), `BUILD_SIM` + one of `USE_PYBULLET` / `USE_RAISIM` / `USE_MJCPP` for sim backend, and `SEND_REMOTE` to trigger `scripts/sftp_to_remote.sh` post-build push to the robot.
- `third_party/lite3/sdk_deploy/src/` — Python-side SDK integration glue.
- `third_party/lite3/lite3_host_control/` — upper-computer control scripts (`lite3_controller.py`, `lite3_command.py`, `keyboard_demo.py`). See [third_party/lite3/lite3_host_control/Lite3上位机通讯控制文档.md](third_party/lite3/lite3_host_control/Lite3上位机通讯控制文档.md).

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

### Lite3 Sim & Navigation Host

`scripts/simulation/nomad_mujoco_lite3_state_machine.py` is the modular Lite3 MuJoCo stack and supersedes the legacy monolithic `scripts/simulation/nomad_mujoco_lite3_nav.py`. It splits inference (DDPM/DDIM/CFG/TTS), high-level NoMaD topomap localization + waypoint generation, mid-level waypoint-to-velocity PD bridge, and low-level Lite3 ONNX locomotion + MuJoCo physics into separate components, driven by a state machine: `idle / standup / navigate / explore / recovery_back / recovery_turn / completed / failed`. Supporting modules live under `scripts/simulation/lite3_system/` (`interfaces.py`, `topomap.py`, `states.py`, `system.py`).

`scripts/deployment/nomad_navigation_host.py` is a **separate** concern from the state machine: the host handles task scheduling, backend selection (`mujoco` vs `real`), and mission switching (single CLI task or multi-task plan from `scripts/configs/navigation_host/lite3_navigation_host_plan.json`). The state machine handles closed-loop execution, recovery, and result logging.

Topomap nodes are pre-collected reference images (loaded from GoStanford trajectories via `load_topomap_from_dataset`, or generated online/offline from real/MuJoCo scenes); NoMaD matches the current observation against them — they are **not** generated from encoder features at deploy time.

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
