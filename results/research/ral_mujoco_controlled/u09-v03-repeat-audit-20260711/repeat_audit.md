# V0.3 exact-key repeatability audit

Scope: medium scene, diffusion seed 23, goal seed 0, policy-only, all five
configurations. Primary and repeat use the same frozen v0.3 protocol and
`policy_action_scale_m=0.1`.

| Configuration | Primary → repeat | Final distance m (primary/repeat) | Maximum common-prefix XY delta m |
|---|---|---:|---:|
| DDIM-2 | success → success | 0.4093 / 0.4855 | 0.1031 |
| DDIM-2 + TTS8 | success → success | 0.4092 / 0.4575 | 0.1212 |
| DDIM-2 + CFG2 + TTS8 | success → success | 0.4139 / 0.4873 | 0.2220 |
| DDIM-3 | success → success | 0.4240 / 0.4865 | 0.1125 |
| DDPM-10 | failure → failure | 0.5084 / 0.8598 | 0.4439 |

All five binary outcomes were preserved for this one exact-key repeat, but the
continuous trajectories and final distances were not byte-identical. This narrow
audit does not establish deterministic execution or a stable method ranking; the
primary matrix remains one seeded-stochastic realization with only three seeds.
