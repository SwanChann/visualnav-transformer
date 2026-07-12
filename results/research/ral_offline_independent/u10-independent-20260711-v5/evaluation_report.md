# U10 v5 corrected independent offline evaluation

Status: completed, 75/75 valid records, no training.

## Correction

U10 v2 used `metric_waypoint_spacing=0.25 m`, which belongs to RECON. This
evaluation uses GO Stanford's frozen training-data value `0.12 m`, loaded from
`train/vint_train/data/data_config.yaml` and bound by SHA-256 in `summary.json`.

## Mean results over 15 paired frozen cases

| configuration | ADE (m) | FDE (m) | sampler latency (ms) |
|---|---:|---:|---:|
| DDIM-2 | 0.101775 | 0.197393 | 9.661 |
| DDIM-3 | 0.099174 | 0.195644 | 12.272 |
| DDPM-10 | 0.096914 | 0.197308 | 40.677 |
| DDIM-2 + TTS8 | 0.116765 | 0.231163 | 9.602 |
| DDIM-2 + CFG2 + TTS8 | 0.113624 | 0.214841 | 17.438 |

DDPM-10 has the lowest mean ADE in this small paired set but is roughly four
times slower than DDIM-2. The heuristic TTS variants do not improve mean ADE.
These are descriptive frozen-checkpoint results from one dataset and 15 cases;
they do not establish statistical superiority, cross-dataset generalization, or
closed-loop navigation success.
