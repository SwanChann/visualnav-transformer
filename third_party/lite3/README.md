# Lite3 Vendor Assets

This directory groups Lite3 platform assets that are not part of the core
NoMaD navigation logic.

- `sdk_deploy/`: Lite3 MuJoCo model and low-level locomotion policy assets.
- `lite3_host_control/`: upper-computer Lite3 velocity-control helpers.

`Lite3_rl_deploy/` is still kept at the repository root for now because it is an
embedded Git repository and was locked by another process during this cleanup.
The shared path helper in `scripts/project_paths.py` already supports the future
location `third_party/lite3/Lite3_rl_deploy/` once it can be moved cleanly.
