# U10 Independent Offline Evaluation Report

Status: completed with 75/75 valid paired case/configuration records and no training operations.

## Frozen evaluation inputs

- dataset images and `traj_data.pkl`: read-only external dataset at `/home/swanchan/visualnav-transformer/nomad_dataset/go_stanford` because the agent worktree transfer contains pose metadata but not images;
- leakage-safe split: grouped-v2 Go Stanford test split, SHA-256 `3f2a27d941d016be7854d8530feee674cd4e55dfd1c1aa6cbc499028729bd581`;
- frozen cases: 5 cases each from short/medium/long goal-offset suites, 15 total;
- frozen checkpoint: SHA-256 `66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79`;
- candidate count: K=8 for every configuration;
- latency: `sampler_per_call`, three warm-up calls excluded, CUDA synchronized before and after every measured call.

The failed first launch under `u10-independent-20260711` is retained: it stopped before inference because the agent worktree dataset had no images. The `-v2` run makes the complete external dataset path explicit rather than silently substituting it.

## Independent metrics

Prediction waypoints are converted to meters with the Go Stanford `metric_waypoint_spacing=0.25 m`. Ground truth is computed directly from held-out `traj_data.pkl` poses in the observation-frame local coordinate system. ADE/FDE, forward-progress error, wrapped heading error, minADE@8, and pairwise candidate diversity do not reuse the heuristic TTS verifier.

| Configuration | ADE m | FDE m | minADE@8 m | sampler mean ms | sampler p95 ms |
|---|---:|---:|---:|---:|---:|
| DDPM-10 / CFG-0 / mean K=8 | 0.6004 | 1.0394 | 0.5203 | 39.98 | 41.98 |
| DDIM-2 / CFG-0 / mean K=8 | 0.6018 | 1.0429 | 0.5057 | 9.28 | 10.76 |
| DDIM-3 / CFG-0 / mean K=8 | 0.6028 | 1.0458 | 0.5206 | 13.09 | 14.70 |
| DDIM-2 / CFG-0 / ranked TTS-8 | 0.6944 | 1.2201 | 0.5057 | 9.48 | 11.16 |
| DDIM-2 / CFG-2 / ranked TTS-8 | 0.6875 | 1.2110 | 0.5294 | 17.12 | 18.68 |

## Interpretation

DDIM-2 preserves the mean-trajectory ADE/FDE of DDPM-10 on this small frozen case set while reducing sampler latency by roughly fourfold. Ranked heuristic TTS-8 is negative evidence: it uses the same eight DDIM-2 candidates and therefore has the same minADE@8 opportunity, but its selected top-1 trajectory worsens independent ADE/FDE relative to averaging the batch. CFG-2 approximately doubles DDIM-2 sampler latency and does not recover the independent error.

These are descriptive results over 15 held-out cases, not a closed-loop success claim and not a multi-dataset benchmark. The U08 closed-loop gate rejected every available scene as success-saturated, so offline and closed-loop rankings cannot yet be correlated.
