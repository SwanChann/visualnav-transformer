# Offline Metric Notes

## Scope and source data

This analysis unifies 131 configurations from six existing offline experiments. It does not run the navigation policy again.

- `ddim_sweep`: `results\day2\20260410_143136_ddim_stat_experiment\ddim_overall_summary.csv` + `results\day2\20260410_143136_ddim_stat_experiment\ddim_case_records.csv`
- `cfg_sweep`: `results\day3\20260410_143225_cfg_stat_experiment\cfg_overall_summary.csv` + `results\day3\20260410_143225_cfg_stat_experiment\cfg_case_records.csv`
- `ddim_cfg_joint`: `results\day4\20260410_143709_joint_ddim_cfg_experiment\overall_summary.csv` + `results\day4\20260410_143709_joint_ddim_cfg_experiment\case_records.csv`
- `tts_budget`: `results\day4\20260421_140601_tts_stat_experiment\tts_overall_summary.csv` + `results\day4\20260421_140601_tts_stat_experiment\tts_case_records.csv`
- `encoder_comparison`: `results\day4\20260421_151228_encoder_comparison_experiment\encoder_overall_summary.csv` + `results\day4\20260421_151228_encoder_comparison_experiment\encoder_case_records.csv`
- `encoder_joint`: `results\day5\20260422_171021_encoder_joint_ablation_experiment\joint_overall_summary.csv` + `results\day5\20260422_171021_encoder_joint_ablation_experiment\joint_case_records.csv`

## Raw fields

- `num_cases`, mean latency and its CI95, forward progress, smoothness, absolute lateral motion, diversity, and endpoint norm come from each source summary CSV.
- p50/p90/p95 latency are recomputed from the matching case-record `latency_ms` values with linear (R-7) interpolation. Each case-record value is already a mean over 3 or 8 inference runs, so these are percentiles **across case-level means**, not per-call tail-latency percentiles.
- Encoder comparison is recorded as DDPM-10/CFG-0/TTS-0 because that experiment script defaults to DDPM and the observed run uses the default; CFG sweep uses DDPM-10 as explicitly defined by `cfg_stat_experiment.py`.

## Derived fields

- `hz_mean = 1000 / latency_mean_ms`; this is an inference-only upper bound, not full perception-control loop frequency.
- Within each source experiment, every case record is z-normalized using the mean and population standard deviation over all case/config records in that source.
- `quality_proxy = z(forward_progress) - z(smoothness) - z(lateral_abs) + 0.5*z(diversity)`.
- `quality_proxy_ci95` is `1.96 * sample_std(case_proxy) / sqrt(num_cases)` and therefore includes covariance between component metrics at case level.
- `pareto_frontier` minimizes mean latency and maximizes quality proxy **within the same source experiment**. Cross-experiment dominance is deliberately not claimed because the sampled case sets and collection dates differ.
- Budget modes are descriptive rather than globally optimized: Baseline is DDPM-10/CFG-0/TTS-0; Fast prioritizes low-step CFG-0 sampling without candidate expansion; Balanced adds a modest candidate budget; Quality adds moderate guidance/candidate budget. Non-frontier rows are marked Dominated.

## Missing fields and limitations

- `invalid_or_saturation` is unavailable in all six source datasets and is written as `NA`; its penalty is omitted from the proxy.
- The proxy is a transparent trajectory-statistics summary, **not navigation success rate**, collision rate, safety, or real-world utility. Higher diversity is not universally beneficial, and forward progress can reward unsafe straight motion.
- TTS uses a heuristic verifier containing forward progress, lateral deviation, smoothness, and path efficiency, while the reported proxy reuses three of those terms. TTS proxy improvements are therefore partly true **by construction** and are not independent evidence that navigation improved.
- `TTS=0` still samples the standard batch of 8 NoMaD trajectories without heuristic top-k selection. `TTS=8` ranks that same-size batch (top-4 here); only budgets 16/32 expand generation beyond the standard batch.
- The 12-case TTS and encoder-joint experiments are not directly interchangeable with the 24-case sweeps.
- All offline cases come from `go_stanford`; the results do not establish cross-dataset or outdoor generalization.
- Timings are historical GPU inference measurements. They do not include image transport, preprocessing, robot communication, low-level control, or current Lite3 on-board hardware latency.
- Normal-approximation CIs are descriptive for the sampled cases; they are not independent repeated-training confidence intervals.

## Integrity audit

- ddim_sweep: 6 configs, 144 case records, 0 config count mismatches; Pareto points: 2
- cfg_sweep: 6 configs, 144 case records, 0 config count mismatches; Pareto points: 2
- ddim_cfg_joint: 20 configs, 480 case records, 0 config count mismatches; Pareto points: 1
- tts_budget: 65 configs, 780 case records, 0 config count mismatches; Pareto points: 13
- encoder_comparison: 4 configs, 96 case records, 0 config count mismatches; Pareto points: 1
- encoder_joint: 30 configs, 360 case records, 0 config count mismatches; Pareto points: 4

## Reproduction

```powershell
python scripts/analysis/ral_offline_pareto.py --results results --out 投稿冲刺/workspace/offline_pareto
```

The script uses only the Python standard library and writes vector PDF directly, so it does not depend on the repository's currently inconsistent NumPy/pandas/matplotlib environment.
