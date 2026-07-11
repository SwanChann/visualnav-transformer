# Table II — Representative Offline Latency–Quality Configurations

Pareto flags and quality proxy values are valid within each source experiment only; the experiments used different sampled case sets.

| Mode | Source | Configuration | Mean / case-p95 latency (ms) | Hz | Forward | Lateral | Smoothness | Diversity | Quality proxy ± CI95 | Pareto |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| Baseline | tts_budget | ddpm10\_cfg0.0\_tts0 | 34.11 / 44.73 | 29.3 | 9.103 | 1.388 | 0.069 | 0.564 | -2.646 ± 0.435 | no |
| Fast | tts_budget | ddim2\_cfg0.0\_tts0 | 7.00 / 7.56 | 143.0 | 9.010 | 1.404 | 0.081 | 0.610 | -3.407 ± 0.460 | no |
| Fast | tts_budget | ddim3\_cfg0.0\_tts0 | 10.00 / 11.17 | 100.0 | 8.997 | 1.414 | 0.080 | 0.606 | -3.400 ± 0.545 | no |
| Balanced | tts_budget | ddim2\_cfg0.0\_tts8 | 6.89 / 7.27 | 145.1 | 9.697 | 0.909 | 0.069 | 0.372 | -1.075 ± 0.617 | yes |
| Balanced | tts_budget | ddim2\_cfg0.0\_tts16 | 13.86 / 15.12 | 72.2 | 10.050 | 0.641 | 0.063 | 0.332 | 0.417 ± 0.624 | yes |
| Balanced | tts_budget | ddim3\_cfg0.0\_tts8 | 9.98 / 10.67 | 100.2 | 9.743 | 0.978 | 0.060 | 0.387 | -0.542 ± 0.570 | yes |
| Quality | tts_budget | ddim2\_cfg1.0\_tts8 | 12.52 / 13.67 | 79.8 | 9.941 | 0.865 | 0.063 | 0.328 | -0.277 ± 0.581 | yes |
| Quality | tts_budget | ddim2\_cfg2.0\_tts8 | 12.55 / 13.53 | 79.7 | 10.173 | 1.127 | 0.055 | 0.281 | -0.094 ± 1.087 | yes |
| Quality | tts_budget | ddim2\_cfg2.0\_tts16 | 24.60 / 26.30 | 40.6 | 10.495 | 0.770 | 0.052 | 0.243 | 1.340 ± 0.850 | yes |
| Dominated | cfg_sweep | ddpm10\_cfg4\_tts0 | 78.92 / 83.73 | 12.7 | 8.483 | 3.089 | 0.134 | 0.304 | -0.412 ± 1.025 | no |
| Dominated | tts_budget | ddim10\_cfg1.0\_tts32 | 231.46 / 246.22 | 4.3 | 10.416 | 0.617 | 0.039 | 0.188 | 2.112 ± 0.556 | no |

`Baseline/Fast/Balanced/Quality` are descriptive latency-budget labels, not claims of global optimality. The two dominated rows are retained as negative controls.
