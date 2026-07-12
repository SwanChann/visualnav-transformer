| Configuration | Stabilizer | Success | SPL | Final dist. m | Full-loop p95 ms | Stuck | Timeout |
|---|---|---:|---:|---:|---:|---:|---:|
| DDPM-10 | policy_only_off | 4/6 | 0.666 | 0.569 | 101.7 | 0.00 | 0.33 |
| DDIM-2 | policy_only_off | 5/6 | 0.823 | 0.787 | 58.2 | 0.17 | 0.17 |
| DDIM-3 | policy_only_off | 5/6 | 0.833 | 0.854 | 66.3 | 0.17 | 0.17 |
| DDIM-2 + TTS-8 | policy_only_off | 5/6 | 0.833 | 0.484 | 59.8 | 0.00 | 0.17 |
| DDIM-2 + CFG-2 + TTS-8 | policy_only_off | 6/6 | 1.000 | 0.443 | 81.7 | 0.00 | 0.00 |
| DDPM-10 | system_stabilizer_on | 0/6 | 0.000 | 1.121 | 91.3 | 0.50 | 1.00 |
| DDIM-2 | system_stabilizer_on | 0/6 | 0.000 | 1.121 | 60.3 | 0.50 | 1.00 |
| DDIM-3 | system_stabilizer_on | 0/6 | 0.000 | 1.121 | 63.6 | 0.50 | 1.00 |
| DDIM-2 + TTS-8 | system_stabilizer_on | 0/6 | 0.000 | 1.121 | 59.4 | 0.50 | 1.00 |
| DDIM-2 + CFG-2 + TTS-8 | system_stabilizer_on | 0/6 | 0.000 | 1.121 | 84.0 | 0.50 | 1.00 |

Note: native NoMaD actions are converted with target-platform scale 0.1 m before waypoint control. All system-stabilizer trajectories are byte-identical across the five methods for every scene/seed; those rows differ only in measured compute latency and are not independent navigation outcomes.
