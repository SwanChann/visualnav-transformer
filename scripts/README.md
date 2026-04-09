# Scripts Layout

## Canonical Paths

The `scripts/` directory now has categorized entrypoints.  
Root-level scripts are still kept as legacy-compatible entrypoints, but new work and docs should prefer the categorized paths below.

### `scripts/analysis/`

- `check_dataset.py`
- `offline_inference.py`
- `realtime_inference.py`
- `result_collector.py`
- `thesis_result_summary.py`

### `scripts/experiments/`

- `ablation_experiment.py`
- `ddim_experiment.py`
- `ddim_stat_experiment.py`
- `cfg_experiment.py`
- `cfg_stat_experiment.py`
- `gait_shake_robustness.py`
- `encoder_comparison_experiment.py`

### `scripts/models/`

- `nomad_vint_dinov2.py`
- `nomad_vint_backbone_suite.py`

### `scripts/training/`

- `train_dinov2.py`
- `train_backbone_suite.py`

### `scripts/shared/`

- `nomad_eval_common.py`

### `scripts/tooling/`

- `env_check.py`
- `path_audit.py`
- `project_paths.py`

### `scripts/simulation/`

- `lite3_sim.py`
- `nomad_mujoco_lite3_nav.py`
- `nomad_mujoco_lite3_state_machine.py`
- `nomad_mujoco_tron1_nav.py`

### `scripts/deployment/`

- `nomad_real_deployment_checklist.py`

## Lite3 MuJoCo Structure

There are now two Lite3 MuJoCo paths:

1. Legacy monolithic implementation:
   `scripts/nomad_mujoco_lite3_nav.py`
2. Categorized legacy-compatible entrypoint:
   `scripts/simulation/nomad_mujoco_lite3_nav.py`
3. New modular state-machine implementation:
   `scripts/simulation/nomad_mujoco_lite3_state_machine.py`

The new modular system splits the stack into:

- High level: NoMaD topomap localization and waypoint generation
- Middle level: waypoint to velocity PD bridge
- Low level: Lite3 ONNX locomotion plus MuJoCo physics execution
- State machine: `idle / standup / navigate / explore / recovery_back / recovery_turn / completed / failed`

Supporting modules live in:

- `scripts/simulation/lite3_system/interfaces.py`
- `scripts/simulation/lite3_system/topomap.py`
- `scripts/simulation/lite3_system/states.py`
- `scripts/simulation/lite3_system/system.py`

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
python scripts/training/train_backbone_suite.py --config config/nomad_dinov2.yaml --backbone dinov2_small --freeze-backbone --pretrained-backbone
python scripts/simulation/nomad_mujoco_lite3_nav.py --mode navigate --map easy
python scripts/simulation/nomad_mujoco_lite3_state_machine.py --mode mission --map medium --mission-topomap topomaps/medium
python scripts/deployment/nomad_real_deployment_checklist.py --platform lite3 --save
```
