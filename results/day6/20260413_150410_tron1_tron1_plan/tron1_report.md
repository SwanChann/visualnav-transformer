# NoMaD Tron1 MuJoCo Integration Report

- Robot: `tron1`
- Repo Root: `/home/swanchan/visualnav-transformer`
- Generated At: `2026-04-13T15:04:10`

## Asset Checks

| Item | Exists | Required | Path | Note |
|---|---|---|---|---|
| nomad_weights | yes | yes | `/home/swanchan/visualnav-transformer/deployment/model_weights/nomad.pth` | Shared NoMaD checkpoint used by the high-level navigation policy. |
| tron1_mujoco_model | no | yes | `/home/swanchan/visualnav-transformer/assets/tron1/mujoco/tron1.xml` | MuJoCo XML or MJCF model for the Tron1 wheel-legged robot. |
| tron1_policy | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/policy/policy.onnx` | Optional low-level wheel-legged locomotion policy exported to ONNX. |
| tron1_camera_config | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/config/camera.yaml` | Optional camera intrinsics/extrinsics for the MuJoCo or real robot bridge. |
| tron1_bridge_notes | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/README.md` | Recommended place to document wheel, leg, and controller topic mappings. |

## Implementation Stages

| Stage | Goal | Outputs |
|---|---|---|
| stage_1_assets | Prepare the Tron1 MuJoCo model, camera settings, and NoMaD checkpoint. | assets/tron1/mujoco/tron1.xml<br>assets/tron1/config/camera.yaml |
| stage_2_perception | Render the onboard RGB view and align it with the NoMaD observation transform. | RGB frame stream at 4 Hz<br>goal or topomap image loader |
| stage_3_navigation | Run NoMaD localization, goal matching, and DDPM or DDIM action sampling. | sampled trajectories<br>selected waypoint |
| stage_4_bridge | Convert the selected waypoint into wheel-legged chassis commands and safety-constrained references. | forward velocity command<br>yaw rate command<br>wheel-leg mode switch or guard rails |
| stage_5_execution | Connect the high-level command bridge to Tron1 locomotion control in MuJoCo. | wheel speed interface<br>leg posture or stabilizer interface |
| stage_6_evaluation | Export navigation logs for thesis figures, comparison tables, and deployment readiness review. | trajectory.txt<br>summary.txt<br>platform comparison notes |

## Recommended Next Commands

```bash
python3 scripts/simulation/nomad_mujoco_tron1_nav.py --mode validate-assets
python3 scripts/simulation/nomad_mujoco_tron1_nav.py --mode export-manifest --save
```