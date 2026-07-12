# Deployment paths

The canonical Ubuntu/Lite3 path is `scripts/deployment/nomad_navigation_host.py`
plus the modular `scripts/simulation/lite3_system/` runtime. It uses the shared
NoMaD inference module, backend injection, explicit policy-action scaling and the
same middle/low-level interfaces as MuJoCo validation.

`deployment/src/` is the upstream ROS-oriented legacy path. It retains an inline
DDPM loop, first-sample reduction and separate configuration/preprocessing code;
therefore it is not evidence of parity with the canonical runtime. This audit fixes
its unambiguous PD quadrant error (`arctan2`), but the directory remains legacy
until a ROS environment can run an end-to-end parity test. Do not use it to claim
CFG/DDIM/TTS parity, target-device deadlines, or real-robot success.
