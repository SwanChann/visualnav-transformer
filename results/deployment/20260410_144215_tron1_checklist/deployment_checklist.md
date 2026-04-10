# NoMaD Real Deployment Checklist

- Platform: `tron1`
- Generated At: `2026-04-10T14:42:15`

| Category | Item | Exists | Required | Required File | Note |
|---|---|---|---|---|---|
| base | NoMaD checkpoint | yes | yes | `/home/swanchan/visualnav-transformer/deployment/model_weights/nomad.pth` | High-level navigation checkpoint |
| base | navigation node | yes | yes | `/home/swanchan/visualnav-transformer/deployment/src/navigate.py` | Topomap localization and waypoint generation |
| base | pd controller | yes | yes | `/home/swanchan/visualnav-transformer/deployment/src/pd_controller.py` | Waypoint-to-velocity bridge |
| base | model config | yes | yes | `/home/swanchan/visualnav-transformer/deployment/config/models.yaml` | Checkpoint and model parameter registry |
| base | robot config | yes | yes | `/home/swanchan/visualnav-transformer/deployment/config/robot.yaml` | Velocity bounds and topic names |
| platform | Tron1 MuJoCo helper | yes | yes | `/home/swanchan/visualnav-transformer/scripts/nomad_mujoco_tron1_nav.py` | Asset validation and implementation staging entry |
| platform | Tron1 MuJoCo model | no | yes | `/home/swanchan/visualnav-transformer/assets/tron1/mujoco/tron1.xml` | Wheel-legged MuJoCo XML or MJCF asset |
| platform | Tron1 low-level controller | no | no | `/home/swanchan/visualnav-transformer/assets/tron1/policy/policy.onnx` | Optional wheel-legged controller export |

## Deployment Review Steps

1. Confirm the camera topic, robot command topic, and topomap input path.
2. Verify the NoMaD checkpoint and the platform-specific low-level controller are both available.
3. Run a static camera test before enabling closed-loop locomotion.
4. Enable velocity limits and recovery mode before the first autonomous run.
5. Record logs, images, and platform exceptions for thesis evidence collection.