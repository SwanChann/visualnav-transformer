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

## IV. Existing closed-loop evidence

The 228 historical MuJoCo records may document integration coverage only. They may
not serve as the main controlled comparison: all lack explicit seed, latency,
stuck, fall, and stabilizer provenance; 27 records conflict between stored success
and final-distance fields; several result signatures repeat across configurations.
Table III remains blocked until a clean rerun with frozen seeds and provenance.

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

1. Re-run a compact non-saturated MuJoCo matrix with >=3 frozen seeds, raw seed and
   stabilizer fields, latency, success, final distance, collision/fall/stuck reason.
2. Pre-register real trial goals, success radius, timeout, intervention and failure
   taxonomy; then run repeated trials for at least baseline/fast/balanced modes.
3. Measure full-loop sustained timing and deadline misses on the named deployment
   computer. These are external execution tasks, not locally inferred results.
