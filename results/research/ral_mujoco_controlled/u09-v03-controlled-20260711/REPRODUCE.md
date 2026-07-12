# Reproduce deployment-scale-corrected controlled evidence

Environment: Ubuntu, conda `nomad_train`, `MUJOCO_GL=egl`. No command below trains.

Immutable primary protocol:

- `后续研究内容/benchmark/ral_mujoco_protocol_v0.3.json`
- SHA-256 `102bf5192a404f8585153aecf4d5c8885d8bc6940e5105e5f19b02f6e40b4c1b`
- checkpoint SHA-256 `66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79`
- `policy_action_scale_m=0.1`, derived from the frozen nominal MuJoCo contract
  `0.4 m/s / 4 Hz`
- each headless navigation call advances 12 x 20 ms = 0.24 s simulated time;
  `loop_hz_mean` is wall-clock compute throughput, not the simulated control rate

Verify the checkpoint:

```bash
test "$(sha256sum deployment/model_weights/nomad/nomad.pth | awk '{print $1}')" = \
  "66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79"
```

Run a new matrix with a new run id:

```bash
MUJOCO_GL=egl /home/swanchan/anaconda3/envs/nomad_train/bin/python \
  scripts/experiments/ral_mujoco_controlled.py \
  --protocol 后续研究内容/benchmark/ral_mujoco_protocol_v0.3.json \
  --run-id <new-run-id>
```

Analyze only that run and corrected U10 v5:

```bash
python scripts/analysis/ral_controlled_results.py \
  --run-dir results/research/ral_mujoco_controlled/<new-run-id> \
  --offline-summary results/research/ral_offline_independent/u10-independent-20260711-v5/summary.json \
  --protocol 后续研究内容/benchmark/ral_mujoco_protocol_v0.3.json \
  --out results/research/ral_mujoco_controlled/<new-run-id>/analysis \
  --bootstrap-replicates 2000 --seed 20260711 --strict
```

U09 v0.2 and U10 v2 are audit-only invalidated predecessors. They must not be
ingested into the primary aggregate or offline correlation.
