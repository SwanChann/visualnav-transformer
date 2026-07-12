# U10 First-Launch Input Failure

Status: failed before model loading or inference.

The default agent-worktree dataset root `nomad_dataset/go_stanford` contained `traj_data.pkl` metadata but no image frames for the grouped-v2 test trajectories. Case freezing expected 15 cases and produced zero, so execution stopped with `RuntimeError` before any checkpoint inference.

No result was silently retried in this directory. The subsequent successful run is isolated under `u10-independent-20260711-v2` and explicitly records the complete read-only dataset root `/home/swanchan/visualnav-transformer/nomad_dataset/go_stanford` in `frozen_cases.json`.
