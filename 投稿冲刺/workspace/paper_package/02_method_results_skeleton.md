# Method and results skeleton

## I. Introduction

Frame one question: how should a frozen diffusion navigation policy be configured
under deployment latency constraints, and what evidence is required before an
offline trade-off becomes a navigation claim? Do not frame TinyNavBrain or future
multi-dataset work as completed contributions.

## II. System

- Frozen policy and observation/goal interfaces.
- Sampler family and denoising steps.
- CFG scale and TTS candidate-selection budget.
- Encoder choice.
- Lite3 host/robot bridge, command rate, safety and termination interfaces.

Report exact checkpoint, hardware, software revision and preprocessing before
submission. Distinguish policy inference latency from the full control loop.

## III. Offline protocol

- Six source experiments; 131 source–configuration records representing 91 distinct
  settings; 2,004 case records.
- All cases are Go Stanford; 12- and 24-case sources remain separate.
- Quality proxy: z(progress) - z(smoothness) - z(lateral motion) + 0.5 z(diversity),
  normalized within source.
- Pareto dominance is computed within source only.
- p50/p90/p95 are across case-level mean latencies, not individual calls.
- TTS/verifier overlap is disclosed as a circularity threat.

## IV. Controlled closed-loop evidence

Protocol v0.3 freezes five K=8 configurations, target-meter action scale 0.1,
easy/medium scenes, seeds 11/23/47,
a strict 0.5 m success radius, a 62-cycle (15.5 s) time budget, and separate
policy-only/system-stabilizer strata. The 60 planned episodes reconcile exactly to
25 successes and 35 failures, with no crash, infrastructure invalidity or missing
trial. Historical MuJoCo records are excluded from this aggregate.

Policy-only success is DDPM-10 4/6, DDIM-2 5/6, DDIM-3 5/6, DDIM-2/TTS-8 5/6 and
DDIM-2/CFG-2/TTS-8 6/6. Report Clopper-Pearson intervals and paired cluster
bootstrap as descriptive because only three seeds are available. The
system-stabilizer stratum is 0/6 for every configuration and has a 0.50 stuck rate,
so it must not be pooled with policy-only results. Corrected U10 v5 offline ADE
(GO Stanford scale 0.12 m loaded from the frozen training data config) versus policy-only
success has rho=-0.67 (exact permutation p=0.30), which is negative evidence against
using offline error as a deployment ranking.

An exact-key v0.3 medium/seed-23 repeat preserved all five binary outcomes but showed
up to 0.444 m common-prefix trajectory deviation. Treat seeds as stochastic
replications, report this audit beside Table III, and do not claim a stable ordering
from the primary 60 episodes.

The 228 historical MuJoCo records remain audit-only because they lack explicit
seed, latency, stuck, fall and stabilizer provenance and contain 27 semantic
conflicts.

## V. Existing real-robot evidence

Use the 528 sampled loop records for timing characterization. `N` is autocorrelated
profiler samples, not independent observations or trials. Five of seven absolute
clocks are invalid 1970 timestamps; relative durations are usable only if the
profiler clock remained monotonic. Twenty videos from 14 trials establish physical
execution and motion; they do not expose enough goal/termination information for
success-rate assignment. Table IV is timing-only until human protocol-based labels
are completed.

## VI. Discussion and limitations

Separate three layers: offline trajectory proxy, simulation navigation outcome,
and real-robot navigation outcome. Explicitly state that the first does not imply
the latter two. Discuss source dependence, circular verifier/proxy terms, sampled
profiling, missing provenance, single dataset, and absent statistical robot trials.

## VII. Required new evidence

1. Expand controlled simulation to more scenes/seeds if inferential claims are
   required; the present 3-seed matrix remains descriptive.
2. Pre-register real trial goals, success radius, timeout, intervention and failure
   taxonomy; then run repeated trials for at least baseline/fast/balanced modes.
3. Measure full-loop sustained timing and deadline misses on the named deployment
   computer. These are external execution tasks, not locally inferred results.
