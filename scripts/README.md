# Scripts Layout

## Canonical Paths

The `scripts/` directory now keeps one canonical location for each script.
Root-level Python implementations were moved into the categorized paths below.

### `scripts/analysis/`

- `check_dataset.py`
- `offline_inference.py`
- `realtime_inference.py`
- `thesis_result_summary.py`

### `scripts/experiments/`

- `ablation_experiment.py`
- `ddim_experiment.py`
- `ddim_stat_experiment.py`
- `cfg_experiment.py`
- `cfg_stat_experiment.py`
- `tts_experiment.py`
- `tts_stat_experiment.py`
- `gait_shake_robustness.py`
- `encoder_comparison_experiment.py`
- `encoder_joint_ablation_experiment.py`
- `joint_ddim_cfg_experiment.py`
- `mujoco_encoder_benchmark.py`

### `scripts/models/`

- `nomad_vint_dinov2.py`
- `nomad_vint_backbone_suite.py`
- `download_encoder_weights.py`

### `scripts/training/`

- `train_dinov2.py`
- `train_backbone_suite.py`
- `run_encoder_seed_sweep.py`

### `scripts/configs/vision_encoder/`

- `nomad_encoder_dinov2_small.yaml`
- `nomad_encoder_convnext_tiny.yaml`
- `nomad_encoder_resnet50.yaml`
- `nomad_encoder_efficientnet_b0.yaml`

### `scripts/configs/navigation_host/`

- `lite3_navigation_host_plan.json`

### `scripts/shared/`

- `nomad_eval_common.py`
- `nomad_inference.py`

### `scripts/tooling/`

- `env_check.py`
- `path_audit.py`
- `project_paths.py`

### `scripts/tooling/data/`

- `check_gostanford.py`: GoStanford dataset integrity check.
- `fix_go_stanford_dtype.py`: GoStanford dataset repair utility that rewrites `traj_data.pkl` after creating `.backup` files.

### `scripts/simulation/`

- `lite3_sim.py`
- `nomad_mujoco_lite3_nav.py`
- `nomad_mujoco_lite3_state_machine.py`
- `nomad_mujoco_tron1_nav.py`

### `scripts/deployment/`

- `nomad_real_deployment_checklist.py`
- `nomad_navigation_host.py`

### `scripts/legacy/`

- `quick_check.py`: one-off hard-coded GoStanford trajectory smoke test kept for historical debugging context.
- `result_collector.py`: legacy alias for `scripts/analysis/thesis_result_summary.py`.

## Lite3 MuJoCo Structure

There are two Lite3 MuJoCo implementations:

1. Legacy monolithic implementation:
   `scripts/simulation/nomad_mujoco_lite3_nav.py`
2. New modular state-machine implementation:
   `scripts/simulation/nomad_mujoco_lite3_state_machine.py`

The new modular system splits the stack into:

- Inference module: shared policy loading plus DDPM/DDIM/CFG/TTS control
- High level: NoMaD topomap localization and waypoint generation
- Middle level: waypoint to velocity PD bridge
- Low level: Lite3 ONNX locomotion plus MuJoCo physics execution
- State machine: `idle / standup / navigate / explore / recovery_back / recovery_turn / completed / failed`

Supporting modules live in:

- `scripts/simulation/lite3_system/interfaces.py`
- `scripts/simulation/lite3_system/topomap.py`
- `scripts/simulation/lite3_system/states.py`
- `scripts/simulation/lite3_system/system.py`

## Navigation Host

There is now a unified deployment host:

- `scripts/deployment/nomad_navigation_host.py`

Its role is different from the Lite3 state machine:

- Navigation host: task scheduling, backend selection, mission switching
- Lite3 state machine: closed-loop execution, recovery, result logging

The host supports:

- `mujoco` backend for direct simulation runs
- `real` backend through a Python bridge adapter
- single-task launch via CLI arguments
- multi-task launch via `scripts/configs/navigation_host/lite3_navigation_host_plan.json`

## Topomap Sources

`topomap` candidate node images can come from two different sources:

1. Dataset source:
   loaded from GoStanford trajectories by `load_topomap_from_dataset`
2. Deployment / simulation source:
   generated online or offline from the real scene / MuJoCo scene by `generate-topomap`

In other words, topomap nodes are not generated from encoder features during deployment.  
They are pre-collected or pre-generated reference images, and NoMaD uses the current observation to match against them.

## Example Commands

```bash
python scripts/analysis/offline_inference.py
python scripts/experiments/ddim_stat_experiment.py --cases-per-suite 8 --num-runs 8
python scripts/experiments/tts_stat_experiment.py --cases-per-suite 8 --num-runs 8 --ddim-steps-list 5,2 --cfg-weights 0.0,0.5,1.0 --tts-budgets 0,16,32
python scripts/models/download_encoder_weights.py --backbone all
python scripts/training/train_backbone_suite.py --config config/nomad_dinov2.yaml --backbone dinov2_small --freeze-backbone --pretrained-backbone
python scripts/training/run_encoder_seed_sweep.py --base-config scripts/configs/vision_encoder/nomad_encoder_dinov2_small.yaml --backbone dinov2_small --seeds 0 1 2 --freeze-backbone --pretrained-backbone --dry-run
cd train && python train.py --config ../scripts/configs/vision_encoder/nomad_encoder_efficientnet_b0.yaml
python scripts/simulation/nomad_mujoco_lite3_nav.py --mode navigate --map easy
python scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode mission --map medium --mission-topomap topomaps/medium
python scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode navigate --map easy --scheduler ddim --ddim-steps 2 --cfg-weight 1.0 --policy-config scripts/configs/vision_encoder/nomad_encoder_dinov2_small.yaml --policy-checkpoint train/logs/nomad-backbone-suite/your_run/ema_latest.pth
python scripts/deployment/nomad_navigation_host.py --backend mujoco --plan-file scripts/configs/navigation_host/lite3_navigation_host_plan.json --dry-run
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3 --save
```
