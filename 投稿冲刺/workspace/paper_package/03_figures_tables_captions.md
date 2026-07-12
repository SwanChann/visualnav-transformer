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

Use `../controlled_results/table_iii.csv`, `fig_controlled_success.pdf`,
`fig_controlled_latency_spl.pdf`, and `fig_controlled_trajectories.pdf`.

**Caption draft.** Controlled MuJoCo results for a frozen NoMaD checkpoint across
easy/medium scenes, diffusion seeds 11/23/47 and a 15.5 s budget. Each configuration
uses K=8. Error bars are exact 95% binomial intervals over six descriptive episodes;
three seeds do not support significance claims. Policy-only and route-stabilized
systems are faceted and never pooled. The route-stabilized stratum is 0/6 for every
configuration, while policy-only success ranges from 4/6 to 6/6. Trajectories are
drawn only from stored episode coordinates. Full-loop and sampler latency retain
separate scopes.
An exact-key repeat changed one of five medium-scene outcomes, so the table is a
frozen descriptive realization rather than a deterministic method ranking.

## Fig. 4 / Table IV — real robot

Use representative multi-view frames only as qualitative integration evidence and
obtain consent/privacy clearance for visible people. Quantitative rows may currently
contain sampled loop timing from `../real_robot/table_real_timing.md`; outcome cells
must read “pending protocol-based human review.” Do not compute success percentage
from directory names or automatic `status=success`.
