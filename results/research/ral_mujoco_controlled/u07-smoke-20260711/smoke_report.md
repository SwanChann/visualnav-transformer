# U07 Mechanical and Determinism Smoke Report

Date: 2026-07-11. Environment: `nomad_train`, MuJoCo 3.2.3, `MUJOCO_GL=egl`, CUDA device `NVIDIA GeForce RTX 3050 Ti Laptop GPU`. No training, backward pass, optimizer step, or real-robot operation occurred.

## Mechanical smoke

The state-machine `stand` task ran headlessly for five held control steps after the 3 s locomotion-policy warm-up. The Lite3 model, tracked ONNX locomotion policy, FPV renderer, controller, and result finalizer all completed. The measured stand path drift was 0.014 m.

This smoke initially exposed a reproducible Linux path defect in `lite3_system/legacy_bridge.py`: it searched for `scripts/nomad_mujoco_lite3_nav.py` instead of `scripts/simulation/nomad_mujoco_lite3_nav.py`. The path was repaired with `Path(__file__)`-relative resolution and covered by `test_lite3_legacy_bridge.py`.

## Controlled closed-loop smoke

Frozen trial key: `ddim2_cfg0_standard_k8__easy__s11__g0__policy_only_off`.

- outcome: success;
- cycles recorded: 43 full loops, 42 sampler calls, 42 trajectory points;
- final goal distance: 0.4572086272 m (`< 0.5 m`);
- executed path length: 4.4091276310 m;
- collision/fall/stuck/timeout/intervention: all false;
- recovery count: 0;
- median sampler time after the first three warm-up calls: 10.0376 ms;
- trial JSON validation: pass;
- batch manifest validation: pass.

The raw trial, timing CSV, trajectory, and manifest are retained in this run directory. Historical MuJoCo results were not read into the controlled record.

## Exact repeat

The same frozen trial was executed again under run ID `u07-repeat-20260711`. Both runs had 42 trajectory points and identical outcome objects. Maximum absolute coordinate difference across the two trajectories was `0.0 m`, and endpoint L2 difference was `0.0 m`. This smoke is bitwise deterministic at the stored trajectory precision on this host; the full matrix still uses seeded episode replication and does not generalize bitwise determinism to other GPUs or software versions.

## Failure detector exercises

Synthetic, no-policy unit exercises passed for all required detector paths:

- robot-to-scene collision latches `collision_detected` during `record_step`;
- platform fall returns the failed state and latches `fall_detected`;
- the frozen stuck detector enters recovery, latches `stuck_detected`, and increments recovery count;
- reaching `max_steps` returns completed-without-success, which the runner classifies as timeout.

These constrained cases are test evidence only and are excluded from the controlled comparison.
