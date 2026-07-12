# V0.2 is not the deployment-consistent primary matrix

The v0.2 runner passed native NoMaD action units directly to waypoint control as
if they were meters. The legacy deployment contract applies a target-platform
scale before control. V0.2 remains an immutable audit trail, but its 60-trial
matrix must not be used as the primary deployment-consistent result.

The corrected replacement is `u09-v03-controlled-20260711`, whose frozen protocol
derives `policy_action_scale_m=0.1` from the Lite3 simulation controller contract
(`0.4 m/s / 4 Hz`) and repeats blind scene gating before the cross-method matrix.
