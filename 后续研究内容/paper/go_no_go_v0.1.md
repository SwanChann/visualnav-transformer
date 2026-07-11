# Future RA-L go/no-go v0.1

## Current decision

**NO-GO as a method paper.** The repository contains a useful data contract,
leakage-safe split tooling, benchmark schema, fair-baseline matrix and a shape-tested
TinyNavBrain feature scaffold. It contains no multi-dataset training result, no
method-effect result, and no new controlled closed-loop evidence.

## Why the direction is still worth pursuing

The strongest research question is not “replace diffusion with a small model.” It is:

> Under fixed data exposure, encoder, parameter, sample-count and inference budgets,
> which mixture strategy and generative head produce the best worst-domain and
> leave-one-domain-out navigation behavior?

This couples the user's interests—benchmarking, multi-dataset training, framework
design and small generative policies—while remaining falsifiable. The benchmark is
the foundation; TinyNavBrain is a candidate answer, not the assumed winner.

## Gates

1. Acquire checksum-bound RECON and HuRoN pilots under explicit license snapshots.
2. Process >=3 domains with session-safe grouped splits and reversible metric action
   normalization.
3. Run B0 specialist, B1 proportional, B2 balanced, B3 sampler-matched, B4 H0 and B5
   DDIM controls before claiming H1 gains.
4. Use >=3 seeds and report per-domain, worst-domain, IID, corruption and LODO results.
5. Advance to simulation only for non-dominated candidates; reject saturated scenes.
6. Advance to robot only after full-loop timing on the target device and a frozen
   safety/outcome protocol.

## Kill and downgrade rules

- Fewer than three usable datasets: no “multi-dataset generalization” claim.
- H1 approximately equals H0: generative mechanism is unnecessary; retain the
  benchmark or deterministic small policy.
- H1 only improves latency: efficiency paper, not navigation-method paper.
- Offline ranking fails in non-saturated simulation: do not proceed to robot claims.
- Robot evidence is underpowered or deadline-limited: withhold RA-L method submission.

## Novelty judgment

Potential novelty is moderate-to-strong only if factor-controlled mixture diagnosis
and a low-NFE generative head both survive the gates. A new head alone is crowded;
a benchmark alone needs unusually strong protocols and public artifacts. Their
combination is promising, but no novelty claim is earned yet.
