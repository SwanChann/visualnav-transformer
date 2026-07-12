# Controlled MuJoCo protocol v0.3

V0.3 is the deployment-semantic scale-corrected successor to v0.2. It adds an
explicit `policy_action_scale_m=0.1`, derived from the frozen MuJoCo controller
contract's 0.4 m/s maximum linear speed and nominal 4 Hz NoMaD update rate.
Native NoMaD action units are converted before waypoint-to-velocity control.

The headless implementation advances 12 locomotion steps at 20 ms per navigation
call (0.24 simulated seconds, or 4.167 calls per simulated second) because the
nominal 250 physics steps are discretized into whole locomotion steps. This does
not turn the recorded `loop_hz_mean` into a control rate: that field is wall-clock
compute throughput in headless mode. The 0.1 m scale remains the pre-frozen nominal
controller contract; the 4% cadence discretization is disclosed and common to all
v0.3 configurations.

The scene gate was rerun with DDPM-10 only. Easy (3/3 policy-only) and medium
(1/3 policy-only) are retained; hard (0/3 in both strata) is rejected before any
v0.3 cross-method result. The final matrix remains 60 trials and all comparisons
remain descriptive because only three diffusion seeds are used.
