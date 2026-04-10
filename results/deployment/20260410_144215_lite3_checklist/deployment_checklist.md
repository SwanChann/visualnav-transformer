# NoMaD Real Deployment Checklist

- Platform: `lite3`
- Generated At: `2026-04-10T14:42:15`

| Category | Item | Exists | Required | Required File | Note |
|---|---|---|---|---|---|
| base | NoMaD checkpoint | yes | yes | `/home/swanchan/visualnav-transformer/deployment/model_weights/nomad.pth` | High-level navigation checkpoint |
| base | navigation node | yes | yes | `/home/swanchan/visualnav-transformer/deployment/src/navigate.py` | Topomap localization and waypoint generation |
| base | pd controller | yes | yes | `/home/swanchan/visualnav-transformer/deployment/src/pd_controller.py` | Waypoint-to-velocity bridge |
| base | model config | yes | yes | `/home/swanchan/visualnav-transformer/deployment/config/models.yaml` | Checkpoint and model parameter registry |
| base | robot config | yes | yes | `/home/swanchan/visualnav-transformer/deployment/config/robot.yaml` | Velocity bounds and topic names |
| platform | Lite3 MuJoCo model | yes | yes | `/home/swanchan/visualnav-transformer/sdk_deploy/src/Lite3_sdk_deploy/Lite3_description/lite3_mjcf/mjcf/Lite3.xml` | Simulation asset for integrated validation |
| platform | Lite3 locomotion policy | yes | yes | `/home/swanchan/visualnav-transformer/sdk_deploy/src/Lite3_sdk_deploy/policy/policy.onnx` | Low-level RL locomotion policy |
| platform | Lite3 MuJoCo runner | yes | yes | `/home/swanchan/visualnav-transformer/scripts/nomad_mujoco_lite3_nav.py` | Integrated NoMaD + MuJoCo + RL navigation entry |

## Deployment Review Steps

1. Confirm the camera topic, robot command topic, and topomap input path.
2. Verify the NoMaD checkpoint and the platform-specific low-level controller are both available.
3. Run a static camera test before enabling closed-loop locomotion.
4. Enable velocity limits and recovery mode before the first autonomous run.
5. Record logs, images, and platform exceptions for thesis evidence collection.