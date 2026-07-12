# MuJoCo calibration v0.3

This overlay corrects the policy-native action-unit to target-meter conversion.
The scale is derived from the frozen Lite3 simulation controller contract:
`bridge_max_v_mps / nomad_frequency_hz = 0.4 / 4 = 0.1 m`.

Here 4 Hz is the nominal controller contract. Headless execution advances 0.24 s
of simulation per navigation call (12 whole 20 ms locomotion steps), while the
reported `loop_hz_mean` measures wall-clock compute throughput rather than the
simulated control cadence. That discretization is fixed across all compared arms.

Only DDPM-10 baseline episodes may be used for scene gating. Cross-method results
must not be inspected until a final v0.3 protocol is frozen. The v0.2 matrix is
retained as an audit of the former scale-mismatched deployment contract and is not
the primary deployment-consistent matrix.
