# Title, abstract, and contributions

## Working title

**Deployment-Time Trade-offs of a Frozen Diffusion Navigation Policy on a Lite3 Quadruped**

Avoid “real-time” in the title until the target device, end-to-end deadline, miss
rate, and sustained closed-loop rate are reported.

## Evidence-bounded abstract draft

Diffusion policies offer multimodal action generation for visual navigation, but
their iterative sampling and candidate selection create a deployment trade-off
between inference cost and trajectory statistics. We present a Lite3 quadruped
deployment stack for a frozen NoMaD-style visual navigation policy and a unified
offline analysis of sampler steps, classifier-free guidance, trajectory-selection
budget, and visual encoder. Across six historical Go Stanford experiments, our
reproducible analysis covers 131 source–configuration records (91 distinct
encoder/sampler/guidance/selection settings) and 2,004 case records and exposes
source-scoped Pareto frontiers between mean inference latency and a transparent
trajectory-statistics proxy. Historical robot logs further show that sampled
per-loop duration means vary across configurations, including 6.6 Hz for
DDIM-2/CFG-0/TTS-8 (33 samples) and 3.4 Hz for DDIM-2/CFG-2/TTS-8 (65 samples).
Those two trials have invalid wall-clock timestamps and assume a monotonic profiler;
they are not sustained control-loop rates. We further report a frozen, seeded
MuJoCo study with two time-constrained scenes and three diffusion seeds. In the
policy-only stratum, DDPM-10 reached 4/6 goals, DDIM-2 and DDIM-3 reached 5/6, and
DDIM-2/CFG-2/TTS-8 reached 6/6; these small-sample results are descriptive, not
significance claims. Enabling the route stabilizer yielded 0/6 for every mode under
the same 15.5 s budget, showing that system layers can reverse or erase policy
rankings. Corrected independent held-out ADE rankings, using the GO Stanford
training-data scale of 0.12 m, were negatively associated with the
policy-only simulation ranking (Spearman rho about -0.67, exact p=0.30), so offline
error did not validate closed-loop superiority. This study therefore provides a
reproducible deployment characterization and evidence protocol while withholding
statistically supported real-robot navigation claims.

A targeted v0.3 exact-key repeat preserved all five binary outcomes but produced up
to 0.444 m common-prefix trajectory deviation, so the reported fractions remain one
seeded-stochastic realization, not a stable method ranking.

## Defensible contributions

1. A reproducible Lite3 deployment stack connecting a frozen diffusion navigation
   policy to image transport, host-side scheduling, and low-level robot control.
2. A source-scoped offline analysis that unifies 131 source–configuration records
   (91 distinct settings) and explicitly separates inference cost from a non-success
   trajectory proxy.
3. An evidence audit and reporting protocol linking offline, sampled real timing,
   historical simulation, and video records without treating missing provenance as
   positive validation.
4. A provenance-complete controlled simulation matrix that separates policy-only
   and route-stabilized systems and exposes a stabilizer-dependent ranking failure.

## Forbidden claims at current evidence state

- new navigation policy or training method;
- state-of-the-art success, robustness, or generalization;
- cross-dataset or cross-embodiment validation;
- TTS independently improves navigation quality or is statistically superior;
- per-call p95 or guaranteed real-time control;
- statistically supported real-robot success rate.
- simulation results generalize to real-robot success.
