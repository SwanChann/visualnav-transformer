# Stage 3 — simulation and robot promotion

Environment boundary: promoted-checkpoint closed-loop simulation runs on the Ubuntu
computer. Dataset-dependent offline evaluation remains on the 4090 server. Target-
device timing and robot execution are unassigned until the actual device/control
host is named; they must not be assumed to run on Ubuntu or 4090.

## Simulation

Promote from 4090 to Ubuntu only the frozen B2/B3, B4 and non-dominated H1 checkpoints.
Bind each checkpoint to its SHA-256, Git SHA, config and offline result package. Use identical
scene sets, >=3 seeds, goal definitions, timeouts, bridge and stabilizer. Log seed,
checkpoint hash, success, final distance, path/SPL, collision, fall, stuck,
intervention, per-call inference and full-loop latency. Replace scenes whose success
is saturated across all methods.

## Target-device timing

Measure warm and cold startup separately. For sustained runs report p50/p95/p99,
deadline, miss rate, thermal/power state, device model, precision and batch/K/NFE.
An inference-only Hz estimate is not a control-loop rate.

## Robot

Pre-register start/goal poses, success radius, timeout, operator intervention,
collision/fall/stuck taxonomy and safe stop. Randomize method order and retain raw
video/logs. Do not treat automatic process exit as navigation success. Promotion to
paper Table IV requires repeated trials and confidence intervals; qualitative video
is supplementary evidence only.
