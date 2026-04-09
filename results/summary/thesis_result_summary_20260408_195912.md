# Thesis Result Summary

- Time: 2026-04-08 19:59:12
- Results root: results

## Baseline
- Source: `results\day1\20260328_151728_offline_inference\summary.txt`
- Model params: 19,049,675
- DDPM inference time: 0.0780 s
- Inference frequency: 12.8 Hz
- Predicted distance: 10.2580

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
| -1.0 | 0.0259 | 0.0735 | 267.3 |
| 0.0 | 0.0201 | 0.0333 | 30.9 |
| 0.5 | 0.0220 | 0.0448 | 47.4 |
| 1.0 | 0.0265 | 0.0371 | 46.8 |
| 2.0 | 0.0236 | 0.0637 | 46.8 |
| 4.0 | 0.0139 | 0.0929 | 46.6 |