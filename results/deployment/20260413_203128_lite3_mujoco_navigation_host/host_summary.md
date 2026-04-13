# NoMaD Navigation Host Summary

- Platform: `lite3`
- Backend: `mujoco`
- Generated At: `2026-04-13T20:31:56`

| Task | Mode | Map | Backend | Status | Exit Code |
|---:|---|---|---|---|---:|
| 1 | stand | easy | mujoco | success | 0 |
| 2 | navigate | easy | mujoco | success | 0 |
| 3 | explore | easy | mujoco | success | 0 |
| 4 | estop | easy | mujoco | success | 0 |

## Host Design Notes

1. The host is responsible for task scheduling and backend selection.
2. Supported host tasks are `stand`, `navigate`, `explore`, and `estop`; `mission` remains a backward-compatible alias.
3. Capture is host-managed: the host keeps one shared session and enables c/[ / ] only during host runs.
4. The MuJoCo backend keeps one scene instance alive across task switches, so the viewer stays open and the robot pose is continuous.
5. The real backend uses one persistent bridge instance; there is no map reload concept during task switching.
6. Task topomaps must match the backend domain: MuJoCo tasks use MuJoCo topomaps, and real-robot tasks use real-world topomaps.