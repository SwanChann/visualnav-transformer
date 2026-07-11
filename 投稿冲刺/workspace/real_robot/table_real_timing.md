# Frozen Real-robot Timing Summary

Computed from existing sampled loop records; no robot experiment was run.

| Trial label | Config ID | N | Total mean / p50 / p95 (ms) | Infer p95 (ms) | Loop FPS mean | Clock valid |
|---|---|---:|---:|---:|---:|---|
| 0428下午基线，后方走廊，失败 | unknown | 95 | 336.9 / 336.5 / 363.7 | 359.6 | 2.99 | True |
| 0428下午最优方案，后方走廊。成功 | unknown | 166 | 153.2 / 154.9 / 168.6 | 164.6 | 6.56 | True |
| ddim2cfg0tts8 | ddim2_cfg0_tts8 | 33 | 160.4 / 148.0 / 170.1 | 166.4 | 6.61 | False |
| ddim2cfg2tts8 | ddim2_cfg2_tts8 | 65 | 294.9 / 286.3 / 317.5 | 313.0 | 3.43 | False |
| ddim3cfg0tts0 | ddim3_cfg0_tts0 | 108 | 169.9 / 168.6 / 187.1 | 182.9 | 5.90 | False |
| ddpm | ddpm_steps_unknown_cfg_unknown_tts_unknown | 10 | 346.8 / 351.1 / 369.1 | 364.5 | 2.89 | False |
| pingpong-road | unknown | 51 | 142.8 / 141.9 / 153.9 | 150.3 | 7.02 | False |

`N` is the number of profiled loop samples, not navigation trials. The profiler records one loop every configured interval (`profile_interval`), so these rows are sampled per-loop timings rather than a continuous trace.
