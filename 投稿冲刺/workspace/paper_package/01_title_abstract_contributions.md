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
they are not sustained control-loop rates. We audit the limits of
these measurements: offline tail values are percentiles of case-level means, the
selection verifier overlaps with the quality proxy, and existing closed-loop and
real-robot records are insufficient for comparative success claims. This study
therefore provides a reproducible deployment characterization and an evidence
protocol, while reserving navigation-effect claims for controlled closed-loop
validation.

The final sentence must be strengthened only after new controlled experiments.

## Defensible contributions

1. A reproducible Lite3 deployment stack connecting a frozen diffusion navigation
   policy to image transport, host-side scheduling, and low-level robot control.
2. A source-scoped offline analysis that unifies 131 source–configuration records
   (91 distinct settings) and explicitly separates inference cost from a non-success
   trajectory proxy.
3. An evidence audit and reporting protocol linking offline, sampled real timing,
   historical simulation, and video records without treating missing provenance as
   positive validation.

## Forbidden claims at current evidence state

- new navigation policy or training method;
- state-of-the-art success, robustness, or generalization;
- cross-dataset or cross-embodiment validation;
- TTS independently improves navigation quality;
- per-call p95 or guaranteed real-time control;
- statistically supported real-robot success rate.
