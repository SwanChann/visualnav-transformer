# Future RA-L figure/table plan

| Item | Question | Required data | Gate |
|---|---|---|---|
| Fig. 1 | What factors are controlled? | dataset/normalization/head/NFE system diagram | contracts frozen |
| Fig. 2 | Does mixture choice affect domains differently? | B0-B3 per-domain and worst-domain CIs | >=3 datasets, >=3 seeds |
| Table I | Are datasets legally and statistically auditable? | license, sessions, trajectories, split leakage | receipts + zero leakage |
| Table II | Is H1 needed? | B4 H0, B5 DDIM, H1 under matched budget | fair matrix passes |
| Fig. 3 | Is the quality-latency frontier better? | per-call device latency and primary metric | same hardware/NFE/K |
| Fig. 4 | Does it survive shift? | corruption and LODO curves | no target tuning |
| Table III | Does offline ranking survive closed loop? | seeded non-saturated sim failures | provenance complete |
| Table IV | Is deployment credible? | robot success/efficiency/failures/deadline misses | safety protocol + repeats |

No placeholder cell may be filled from a planning document. If a gate fails, remove
the corresponding item rather than narrating an unsupported result.
