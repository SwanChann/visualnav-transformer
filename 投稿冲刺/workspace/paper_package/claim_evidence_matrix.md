# Claim–evidence matrix

| ID | Claim | Status | Evidence boundary |
|---|---|---|---|
| CL-C1 | DDIM-2 reduces sampler/full-loop cost relative to DDPM-10 | descriptive support | two scenes, three seeds |
| CL-C2 | Policy-only DDIM-2/3 reached 5/6 vs DDPM-10 4/6 in the v0.3 primary matrix | single-realization support | exact-key repeat preserved outcomes but trajectories varied; no stable ranking |
| CL-C3 | Policy-only CFG-2+TTS-8 reached 6/6 | descriptive support | offline ADE worsens; transfer untested |
| CL-C4 | Route stabilizer produced byte-identical method trajectories and 0/6 | supported negative evidence | one overridden controller behavior; rows not independent |
| CL-C5 | Corrected U10 v5 offline ADE ranking did not validate closed-loop ranking | negative evidence | GO Stanford scale 0.12 m; rho=-0.67, exact p=0.30; v2 invalidated |
| CL-C6 | Statistically supported real-robot success | blocked | protocolized repeated human outcomes absent |

Every simulation claim traces to the 60 immutable trial JSONs, protocol SHA
`102bf5192a404f8585153aecf4d5c8885d8bc6940e5105e5f19b02f6e40b4c1b`,
the U11 controlled episode CSV and Table III. No historical MuJoCo record enters the
controlled aggregate.
