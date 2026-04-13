# Thesis Result Summary

- Time: 2026-04-13 15:06:42
- Results root: results

## Baseline
- Source: `results/day1/20260410_143032_offline_inference/summary.txt`
- Model params: 19,049,675
- DDPM inference time: 0.1250 s
- Inference frequency: 8.0 Hz
- Predicted distance: 11.6589

## DDIM Statistical Result

- Source: `results/day2/20260410_143136_ddim_stat_experiment/ddim_overall_summary.csv`

| Config | Cases | Latency(ms) | CI95(ms) | MSE vs DDPM-10 | Diversity | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| ddim_1 | 24 | 4.51 | 0.14 | 3.516785 | 4.7357 | 8.87x |
| ddim_10 | 24 | 38.03 | 0.60 | 0.006175 | 0.3175 | 1.05x |
| ddim_2 | 24 | 8.26 | 0.28 | 0.020992 | 0.3772 | 4.84x |
| ddim_3 | 24 | 11.66 | 0.34 | 0.012409 | 0.3062 | 3.43x |
| ddim_5 | 24 | 19.26 | 0.43 | 0.004776 | 0.2706 | 2.08x |
| ddpm_10 | 24 | 39.98 | 1.37 | 0.000000 | 0.2512 | 1.00x |

## CFG Statistical Result

- Source: `results/day3/20260410_143225_cfg_stat_experiment/cfg_overall_summary.csv`

| w | Cases | Latency(ms) | Shift vs w=0 | Diversity | Forward Progress | Lateral Abs |
|---:|---:|---:|---:|---:|---:|---:|
| -1.0 | 24 | 42.28 | 0.027832 | 0.2825 | 8.500 | 2.876 |
| 0.0 | 24 | 42.65 | 0.000000 | 0.2879 | 8.523 | 2.886 |
| 0.5 | 24 | 78.27 | 0.015959 | 0.2850 | 8.515 | 2.875 |
| 1.0 | 24 | 79.75 | 0.047619 | 0.2774 | 8.573 | 2.901 |
| 2.0 | 24 | 78.81 | 0.143830 | 0.2990 | 8.527 | 2.952 |
| 4.0 | 24 | 78.92 | 0.513301 | 0.3041 | 8.483 | 3.089 |

## Joint DDIM x CFG

- Source: `results/day4/20260410_143709_joint_ddim_cfg_experiment/overall_summary.csv`

| Rank | Config | Latency(ms) | Forward Progress | Smoothness | Diversity | Score |
|---:|---|---:|---:|---:|---:|---:|
| 1 | ddim2_w0.0 | 9.99 | 8.089 | 0.1237 | 0.3897 | 5.0919 |
| 2 | ddim3_w0.0 | 14.60 | 8.076 | 0.1075 | 0.3208 | 4.9942 |
| 3 | ddim5_w0.0 | 24.63 | 8.032 | 0.1043 | 0.2906 | 4.7481 |
| 4 | ddim3_w1.0 | 26.62 | 8.109 | 0.1096 | 0.3070 | 4.7227 |
| 5 | ddim3_w0.5 | 27.04 | 8.093 | 0.1083 | 0.3130 | 4.7174 |

## Encoder Comparison

- Source: `results/day4/20260410_143729_encoder_comparison_experiment/encoder_overall_summary.csv`

| Model | Cases | Latency(ms) | Diversity | Forward Progress | Smoothness |
|---|---:|---:|---:|---:|---:|
| dinov2_small | 24 | 42.97 | 0.5417 | 8.777 | 0.1215 |
| efficientnet_b0 | 24 | 47.93 | 0.2609 | 7.353 | 0.1276 |

## Lite3 MuJoCo Results

| Source | Mode | Steps | Distance | Reached Goal |
|---|---|---:|---:|---|
| `results/nomad_mujoco/20260410_143820_navigate_mujoco/summary.txt` | navigate_mujoco | 45 | 4.705 m | True |
| `results/nomad_mujoco/20260410_143829_navigate_mujoco/summary.txt` | navigate_mujoco | 87 | 7.360 m | True |
| `results/nomad_mujoco/20260410_143840_navigate_mujoco/summary.txt` | navigate_mujoco | 70 | 7.474 m | True |
| `results/nomad_mujoco/20260410_144025_navigate_mujoco/summary.txt` | navigate_mujoco | 42 | 4.354 m | True |
| `results/nomad_mujoco/20260410_144035_navigate_mujoco/summary.txt` | navigate_mujoco | 61 | 6.433 m | True |
| `results/nomad_mujoco/20260410_144047_navigate_mujoco/summary.txt` | navigate_mujoco | 70 | 7.403 m | True |
| `results/nomad_mujoco/20260410_144106_explore_mujoco/summary.txt` | explore_mujoco | 100 | 7.763 m | N/A |
| `results/nomad_mujoco/20260410_144120_walk_test/summary.txt` | walk_test | 48 | 7.094 m | N/A |
| `results/nomad_mujoco/20260410_144134_lite3_state_machine_navigate/summary.txt` | lite3_state_machine_navigate | 43 | 4.622 m | True |
| `results/nomad_mujoco/20260410_144148_lite3_state_machine_mission/summary.txt` | lite3_state_machine_mission | 129 | 12.601 m | True |
| `results/nomad_mujoco/20260413_150349_lite3_state_machine_explore/summary.txt` | lite3_state_machine_explore | 99 | 8.711 m | N/A |
| `results/nomad_mujoco/20260413_150349_lite3_state_machine_walk-test/summary.txt` | lite3_state_machine_walk-test | 48 | 6.398 m | N/A |

## Navigation Host

- Source: `results/deployment/20260413_150346_lite3_mujoco_navigation_host/host_plan.json`
- Platform: `lite3`
- Backend: `mujoco`

| Task | Mode | Map | Status | Exit Code |
|---:|---|---|---|---:|
| 1 | navigate | easy | pending(dry-run) | 0 |
| 2 | mission | easy | pending(dry-run) | 0 |

## Tron1 Asset Status

- Source: `results/day6/20260413_150410_tron1_tron1_plan/tron1_manifest.json`

| Asset | Exists | Required | Path |
|---|---|---|---|
| nomad_weights | yes | yes | `/home/swanchan/visualnav-transformer/deployment/model_weights/nomad.pth` |
| tron1_mujoco_model | no | yes | `/home/swanchan/visualnav-transformer/assets/tron1/mujoco/tron1.xml` |
| tron1_policy | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/policy/policy.onnx` |
| tron1_camera_config | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/config/camera.yaml` |
| tron1_bridge_notes | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/README.md` |