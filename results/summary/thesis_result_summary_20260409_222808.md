# Thesis Result Summary

- Time: 2026-04-09 22:28:08
- Results root: results

## Baseline
- Source: `results/day1/20260409_221944_offline_inference/summary.txt`
- Model params: 19,049,675
- DDPM inference time: 0.0817 s
- Inference frequency: 12.2 Hz
- Predicted distance: 11.6589

## DDIM

| Config | Time(ms) | Std(ms) | MSE | Diversity | Steps | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| DDPM-10 (baseline) | 29.47 | 1.94 | 0.000047 | 0.0279 | 10 | 1.00x |
| DDIM-10 | 29.52 | 3.30 | 0.000032 | 0.0389 | 10 | 1.00x |
| DDIM-5 | 15.16 | 1.45 | 0.000033 | 0.0274 | 5 | 1.94x |
| DDIM-3 | 7.87 | 1.87 | 0.000033 | 0.0298 | 3 | 3.75x |
| DDIM-2 | 6.24 | 1.33 | 0.000024 | 0.0367 | 2 | 4.73x |
| DDIM-1 | 3.00 | 0.90 | 0.003936 | 0.5808 | 1 | 9.82x |

## CFG

| w | Diversity | Mean Displacement | Time(ms) |
|---:|---:|---:|---:|
| -1.0 | 0.0296 | 0.0493 | 605.6 |
| 0.0 | 0.0245 | 0.0163 | 48.6 |
| 0.5 | 0.0236 | 0.0234 | 76.0 |
| 1.0 | 0.0174 | 0.0410 | 92.2 |
| 2.0 | 0.0220 | 0.0577 | 90.2 |
| 4.0 | 0.0122 | 0.1041 | 84.8 |