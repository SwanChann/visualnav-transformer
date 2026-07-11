# Existing MuJoCo Closed-loop Summary

This table is recomputed from existing `raw_results.json`; no simulation was run.
Rows remain source-scoped because historical controller/stabilizer provenance is missing.

| Source | Encoder | Map | Sampler | CFG | Success | Steps | Path (m) | Final dist. (m) | Wall time (s) | Flagged records |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 20260414_003556_encoder_benchmark | convnext_tiny | easy | ddim-2 | 0 | 1/1 (100.0%) | 37.0±0.0 | 3.74±0.00 | 1.57±0.00 | 17.8±0.0 | 1 |
| 20260414_003556_encoder_benchmark | convnext_tiny | hard | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 4.20±0.00 | 4.25±0.00 | 20.3±0.0 | 1 |
| 20260414_003556_encoder_benchmark | convnext_tiny | medium | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 4.98±0.00 | 2.38±0.00 | 19.1±0.0 | 1 |
| 20260414_003556_encoder_benchmark | dinov2_small | easy | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 4.49±0.00 | 0.98±0.00 | 14.6±0.0 | 1 |
| 20260414_003556_encoder_benchmark | dinov2_small | hard | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 5.24±0.00 | 2.65±0.00 | 18.7±0.0 | 1 |
| 20260414_003556_encoder_benchmark | dinov2_small | medium | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 4.84±0.00 | 2.42±0.00 | 16.6±0.0 | 1 |
| 20260414_003556_encoder_benchmark | efficientnet_b0 | easy | ddim-2 | 0 | 1/1 (100.0%) | 44.0±0.0 | 4.72±0.00 | 0.18±0.00 | 13.4±0.0 | 1 |
| 20260414_003556_encoder_benchmark | efficientnet_b0 | hard | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 5.17±0.00 | 2.68±0.00 | 13.6±0.0 | 1 |
| 20260414_003556_encoder_benchmark | efficientnet_b0 | medium | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 3.85±0.00 | 3.75±0.00 | 12.4±0.0 | 1 |
| 20260414_003556_encoder_benchmark | resnet50 | easy | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 3.53±0.00 | 2.60±0.00 | 19.1±0.0 | 1 |
| 20260414_003556_encoder_benchmark | resnet50 | hard | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 3.59±0.00 | 5.42±0.00 | 18.5±0.0 | 1 |
| 20260414_003556_encoder_benchmark | resnet50 | medium | ddim-2 | 0 | 0/1 (0.0%) | 49.0±0.0 | 3.33±0.00 | 4.75±0.00 | 18.7±0.0 | 1 |
| 20260414_005304_encoder_benchmark | convnext_tiny | easy | ddim-2 | 0 | 3/3 (100.0%) | 113.0±89.4 | 8.93±5.60 | 1.70±0.51 | 11.5±4.1 | 3 |
| 20260414_005304_encoder_benchmark | convnext_tiny | hard | ddim-2 | 0 | 0/3 (0.0%) | 248.3±87.8 | 18.97±3.78 | 3.10±1.39 | 20.7±4.7 | 3 |
| 20260414_005304_encoder_benchmark | convnext_tiny | medium | ddim-2 | 0 | 2/3 (66.7%) | 205.0±128.9 | 12.60±5.75 | 3.68±4.06 | 15.5±6.0 | 3 |
| 20260414_005304_encoder_benchmark | dinov2_small | easy | ddim-2 | 0 | 0/3 (0.0%) | 299.0±0.0 | 17.23±0.96 | 1.65±0.11 | 20.8±0.3 | 3 |
| 20260414_005304_encoder_benchmark | dinov2_small | hard | ddim-2 | 0 | 0/3 (0.0%) | 216.3±71.8 | 16.38±3.27 | 3.83±1.01 | 21.7±4.0 | 3 |
| 20260414_005304_encoder_benchmark | dinov2_small | medium | ddim-2 | 0 | 0/3 (0.0%) | 299.0±0.0 | 18.00±0.50 | 2.74±0.20 | 24.7±0.3 | 3 |
| 20260414_005304_encoder_benchmark | efficientnet_b0 | easy | ddim-2 | 0 | 3/3 (100.0%) | 43.3±1.2 | 4.70±0.07 | 0.32±0.08 | 6.4±1.1 | 3 |
| 20260414_005304_encoder_benchmark | efficientnet_b0 | hard | ddim-2 | 0 | 2/3 (66.7%) | 91.0±35.5 | 9.20±2.80 | 1.61±2.16 | 10.4±2.7 | 3 |
| 20260414_005304_encoder_benchmark | efficientnet_b0 | medium | ddim-2 | 0 | 3/3 (100.0%) | 77.3±12.4 | 7.40±0.55 | 0.30±0.05 | 7.1±0.5 | 3 |
| 20260414_005304_encoder_benchmark | resnet50 | easy | ddim-2 | 0 | 0/3 (0.0%) | 299.0±0.0 | 19.27±3.12 | 5.41±1.23 | 23.8±0.2 | 3 |
| 20260414_005304_encoder_benchmark | resnet50 | hard | ddim-2 | 0 | 0/3 (0.0%) | 289.0±17.3 | 19.99±2.72 | 7.41±4.56 | 24.9±0.8 | 3 |
| 20260414_005304_encoder_benchmark | resnet50 | medium | ddim-2 | 0 | 0/3 (0.0%) | 299.3±0.6 | 17.45±3.35 | 3.59±4.43 | 24.4±1.6 | 3 |
| 20260421_seeded_mujoco_cfg_medium_ddim3 | efficientnet_b0 | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 7.7±0.8 | 3 |
| 20260421_seeded_mujoco_cfg_medium_ddim3 | efficientnet_b0 | medium | ddim-3 | 0.5 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 8.7±0.0 | 3 |
| 20260421_seeded_mujoco_cfg_medium_ddim3 | efficientnet_b0 | medium | ddim-3 | 1 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 9.1±0.3 | 3 |
| 20260421_seeded_mujoco_cfg_medium_ddim3 | efficientnet_b0 | medium | ddim-3 | 2 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 9.2±0.3 | 3 |
| 20260421_seeded_mujoco_ddim10_medium_hard | efficientnet_b0 | hard | ddim-10 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 13.5±0.9 | 3 |
| 20260421_seeded_mujoco_ddim10_medium_hard | efficientnet_b0 | medium | ddim-10 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 9.9±0.6 | 3 |
| 20260421_seeded_mujoco_ddim2_medium_hard | efficientnet_b0 | hard | ddim-2 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 10.0±1.1 | 3 |
| 20260421_seeded_mujoco_ddim2_medium_hard | efficientnet_b0 | medium | ddim-2 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 7.5±0.8 | 3 |
| 20260421_seeded_mujoco_ddim3_medium_hard | efficientnet_b0 | hard | ddim-3 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 10.2±0.7 | 3 |
| 20260421_seeded_mujoco_ddim3_medium_hard | efficientnet_b0 | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 7.9±0.3 | 3 |
| 20260421_seeded_mujoco_ddim5_medium_hard | efficientnet_b0 | hard | ddim-5 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 11.0±0.7 | 3 |
| 20260421_seeded_mujoco_ddim5_medium_hard | efficientnet_b0 | medium | ddim-5 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 8.4±0.1 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | convnext_tiny | easy | ddim-3 | 0 | 3/3 (100.0%) | 75.7±0.6 | 5.48±0.02 | 0.45±0.01 | 9.8±0.2 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | convnext_tiny | hard | ddim-3 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 13.4±0.6 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | convnext_tiny | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 10.1±0.3 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | dinov2_small | easy | ddim-3 | 0 | 3/3 (100.0%) | 75.7±0.6 | 5.48±0.02 | 0.45±0.01 | 9.8±0.2 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | dinov2_small | hard | ddim-3 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 12.9±0.8 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | dinov2_small | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 10.2±0.1 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | efficientnet_b0 | easy | ddim-3 | 0 | 3/3 (100.0%) | 75.7±0.6 | 5.48±0.02 | 0.45±0.01 | 7.9±0.7 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | efficientnet_b0 | hard | ddim-3 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 10.3±0.7 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | efficientnet_b0 | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 7.9±0.1 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | resnet50 | easy | ddim-3 | 0 | 3/3 (100.0%) | 75.7±0.6 | 5.48±0.02 | 0.45±0.01 | 10.8±0.3 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | resnet50 | hard | ddim-3 | 0 | 3/3 (100.0%) | 130.0±17.3 | 9.21±0.67 | 0.48±0.01 | 15.2±1.0 | 3 |
| 20260421_seeded_mujoco_encoder_benchmark | resnet50 | medium | ddim-3 | 0 | 3/3 (100.0%) | 68.0±0.0 | 6.32±0.00 | 0.49±0.00 | 11.2±0.1 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddim-2 | 0 | 3/3 (100.0%) | 153.3±110.6 | 12.69±7.96 | 1.16±1.00 | 13.2±5.3 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddim-2 | 0.5 | 3/3 (100.0%) | 38.3±1.2 | 3.72±0.15 | 1.53±0.10 | 8.4±0.1 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddim-2 | 1 | 3/3 (100.0%) | 197.0±73.0 | 13.98±4.65 | 2.15±0.92 | 18.0±3.5 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddpm | 0 | 0/3 (0.0%) | 242.7±97.6 | 15.96±5.16 | 3.57±2.82 | 20.1±5.5 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddpm | 0.5 | 2/3 (66.7%) | 278.7±20.0 | 18.57±1.03 | 1.18±0.57 | 31.1±1.7 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | easy | ddpm | 1 | 2/3 (66.7%) | 114.3±117.2 | 8.29±6.32 | 1.70±1.01 | 16.1±8.5 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddim-2 | 0 | 0/3 (0.0%) | 299.0±0.0 | 21.91±4.58 | 3.11±2.13 | 24.1±2.6 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddim-2 | 0.5 | 0/3 (0.0%) | 257.3±72.2 | 18.07±7.94 | 9.27±1.16 | 24.4±6.1 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddim-2 | 1 | 0/3 (0.0%) | 255.7±75.1 | 18.49±5.22 | 2.90±1.46 | 24.5±5.2 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddpm | 0 | 0/3 (0.0%) | 243.7±53.6 | 15.95±3.08 | 10.54±2.30 | 22.2±4.7 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddpm | 0.5 | 0/3 (0.0%) | 299.0±0.0 | 16.30±2.17 | 8.31±1.56 | 30.6±0.4 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | hard | ddpm | 1 | 0/3 (0.0%) | 299.3±0.6 | 20.63±4.64 | 5.28±3.58 | 35.7±4.4 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddim-2 | 0 | 1/3 (33.3%) | 218.7±139.1 | 17.75±10.51 | 3.97±3.18 | 16.9±6.9 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddim-2 | 0.5 | 2/3 (66.7%) | 172.7±103.8 | 15.21±8.58 | 2.15±0.98 | 17.5±7.3 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddim-2 | 1 | 0/3 (0.0%) | 183.7±102.3 | 16.95±7.37 | 3.47±0.55 | 19.3±6.6 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddpm | 0 | 0/3 (0.0%) | 289.7±16.2 | 21.09±5.39 | 7.07±2.80 | 24.6±4.6 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddpm | 0.5 | 1/3 (33.3%) | 227.7±123.6 | 16.09±8.02 | 1.66±0.78 | 28.5±13.3 | 3 |
| ddim_cfg_encoder_ablation | convnext_tiny | medium | ddpm | 1 | 1/3 (33.3%) | 270.0±47.7 | 17.37±7.01 | 6.71±3.46 | 30.9±6.4 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddim-2 | 0 | 3/3 (100.0%) | 43.3±0.6 | 4.65±0.05 | 0.32±0.06 | 7.3±0.2 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddim-2 | 0.5 | 3/3 (100.0%) | 45.0±2.0 | 4.84±0.19 | 0.33±0.17 | 8.4±0.3 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddim-2 | 1 | 2/3 (66.7%) | 129.7±146.7 | 11.63±11.78 | 0.60±0.64 | 14.0±9.0 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddpm | 0 | 3/3 (100.0%) | 44.7±1.2 | 4.80±0.12 | 0.34±0.04 | 10.2±1.4 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddpm | 0.5 | 3/3 (100.0%) | 43.3±0.6 | 4.64±0.02 | 0.22±0.04 | 12.5±0.5 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | easy | ddpm | 1 | 3/3 (100.0%) | 43.7±0.6 | 4.68±0.05 | 0.24±0.08 | 10.5±0.1 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddim-2 | 0 | 3/3 (100.0%) | 121.3±80.3 | 12.73±8.03 | 2.63±3.68 | 10.4±4.0 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddim-2 | 0.5 | 3/3 (100.0%) | 73.7±2.3 | 7.92±0.23 | 0.42±0.07 | 9.1±0.2 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddim-2 | 1 | 3/3 (100.0%) | 69.7±1.5 | 7.47±0.13 | 0.39±0.09 | 8.7±0.1 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddpm | 0 | 3/3 (100.0%) | 144.0±102.7 | 13.29±9.08 | 0.50±0.11 | 14.1±6.6 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddpm | 0.5 | 3/3 (100.0%) | 72.0±2.6 | 7.74±0.30 | 0.36±0.02 | 12.3±0.3 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | hard | ddpm | 1 | 3/3 (100.0%) | 76.7±15.9 | 7.70±0.78 | 0.55±0.18 | 12.5±0.8 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddim-2 | 0 | 3/3 (100.0%) | 96.7±55.8 | 8.32±2.29 | 0.41±0.25 | 10.2±2.5 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddim-2 | 0.5 | 3/3 (100.0%) | 60.3±0.6 | 6.46±0.04 | 0.37±0.04 | 9.6±0.1 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddim-2 | 1 | 3/3 (100.0%) | 61.3±1.5 | 6.61±0.17 | 0.22±0.16 | 8.7±0.4 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddpm | 0 | 3/3 (100.0%) | 62.3±0.6 | 6.67±0.04 | 0.23±0.07 | 10.6±0.1 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddpm | 0.5 | 3/3 (100.0%) | 61.7±0.6 | 6.61±0.04 | 0.28±0.03 | 13.3±0.2 | 3 |
| ddim_cfg_encoder_ablation | efficientnet_b0 | medium | ddpm | 1 | 3/3 (100.0%) | 62.3±1.5 | 6.70±0.14 | 0.21±0.08 | 13.7±0.5 | 3 |

Table note: these runs can support closed-loop integration evidence only. Missing stabilizer metadata prevents attribution to the navigation policy alone.
