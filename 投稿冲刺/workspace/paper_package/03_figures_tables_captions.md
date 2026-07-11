# Figure and table plan

## Fig. 1 — deployment stack

**Caption draft.** Lite3 deployment stack for the frozen diffusion visual navigation
policy. The host receives images and goal context, produces candidate waypoint
sequences, optionally ranks them, and sends bounded commands through the robot
bridge. Dashed boxes identify measured policy inference and sampled full-loop timing;
they must not be conflated.

## Fig. 2 — source-scoped offline Pareto analysis

Use `../offline_pareto/fig_offline_pareto.pdf` and optionally
`fig_tts_marginal_gain.pdf`.

**Caption draft.** Mean historical GPU inference latency versus a standardized
trajectory-statistics proxy for six offline experiment sources. Proxy terms are
normalized within each source, and Pareto frontiers are computed within source;
positions are therefore not comparable as a global ranking across panels. Error
bars are descriptive 95% normal intervals over sampled cases. TTS shares progress,
lateral-motion and smoothness terms with the reported proxy, so apparent TTS gains
are not independent navigation-quality evidence. All cases are from Go Stanford.

## Table II — representative budget modes

Use `../offline_pareto/table_budget_modes.md`. Label entries “representative
candidates,” not “recommended deployment modes.” Preserve the latency values and
limitations verbatim. A mode can be called recommended only after closed-loop
validation.

## Fig. 3 / Table III — MuJoCo

**Blocked for quantitative paper claims.** Existing summary is an audit appendix,
not a main result. Do not plot a fabricated path: historical records contain no
trajectories. A future Table III must report configuration, map, seed count, success,
SPL/path efficiency, collision/fall/stuck, latency, and stabilizer version from a
clean rerun.

## Fig. 4 / Table IV — real robot

Use representative multi-view frames only as qualitative integration evidence and
obtain consent/privacy clearance for visible people. Quantitative rows may currently
contain sampled loop timing from `../real_robot/table_real_timing.md`; outcome cells
must read “pending protocol-based human review.” Do not compute success percentage
from directory names or automatic `status=success`.
