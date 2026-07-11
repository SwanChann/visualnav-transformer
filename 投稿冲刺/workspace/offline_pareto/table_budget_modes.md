# Deployment Budget Modes

These are reproducible candidates for paper discussion. Closed-loop validation is still required before a mode can be called a deployment recommendation.

| Mode | Operational interpretation | Representative candidates | Evidence limitation |
|---|---|---|---|
| Baseline | Original DDPM-10 reference cost | tts_budget:ddpm10_cfg0.0_tts0 (34.1 ms) | reference only |
| Fast | Low-step DDIM, CFG=0, no candidate expansion | tts_budget:ddim2_cfg0.0_tts0 (7.0 ms); tts_budget:ddim3_cfg0.0_tts0 (10.0 ms) | offline proxy; source-specific normalization and case set |
| Balanced | Low-step DDIM with heuristic selection/modest candidate budget | tts_budget:ddim2_cfg0.0_tts8 (6.9 ms); tts_budget:ddim2_cfg0.0_tts16 (13.9 ms); tts_budget:ddim3_cfg0.0_tts8 (10.0 ms) | offline proxy; source-specific normalization and case set |
| Quality | Low-step DDIM with moderate CFG/candidate budget | tts_budget:ddim2_cfg1.0_tts8 (12.5 ms); tts_budget:ddim2_cfg2.0_tts8 (12.5 ms); tts_budget:ddim2_cfg2.0_tts16 (24.6 ms) | offline proxy; source-specific normalization and case set |
| Dominated | High-budget/strong-guidance negative control | cfg_sweep:ddpm10_cfg4_tts0 (78.9 ms); tts_budget:ddim10_cfg1.0_tts32 (231.5 ms) | offline proxy; source-specific normalization and case set |
