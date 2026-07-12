# Reproduce controlled MuJoCo v0.2

Run from the `agent/ubuntu-sim-handoff` worktree with conda environment
`nomad_train` and `MUJOCO_GL=egl`. No command below trains a model.

## Immutable inputs

- protocol overlay: `后续研究内容/benchmark/ral_mujoco_protocol_v0.2.json`
- protocol SHA-256: `4df19bb8d1d15d6969c1df52c2b2f339e5f489dc3bb6db140fe3586fdd9497d6`
- base protocol SHA-256: `0dffb341642027af89a51bf81df0053e6682647e062e41a7db0e75a4ef68e62f`
- checkpoint SHA-256: `66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79`
- environment lock: `results/research/ubuntu_environment/pip_freeze.txt`

The checkpoint is intentionally ignored by Git and must exist at
`deployment/model_weights/nomad/nomad.pth` with the stated hash.

Verify it before any run:

```bash
test "$(sha256sum deployment/model_weights/nomad/nomad.pth | awk '{print $1}')" = \
  "66edf3902382b29e795e64dc6c8dda98932061297b1e55fb0e529332f8540c79"
```

## Execute a new immutable matrix

```bash
MUJOCO_GL=egl /home/swanchan/anaconda3/envs/nomad_train/bin/python \
  scripts/experiments/ral_mujoco_controlled.py \
  --protocol 后续研究内容/benchmark/ral_mujoco_protocol_v0.2.json \
  --run-id <new-run-id>
```

Never reuse `u09-v02-controlled-20260711` as the new run ID; the runner does not
overwrite completed primary attempts.

## Validate and analyze

```bash
for f in results/research/ral_mujoco_controlled/<new-run-id>/trials/*.json; do
  python scripts/analysis/validate_ral_mujoco_trial.py --result "$f" || exit 1
done
python scripts/analysis/validate_ral_mujoco_trial.py \
  --result results/research/ral_mujoco_controlled/<new-run-id>/batch_manifest.json
python scripts/analysis/ral_controlled_results.py \
  --run-dir results/research/ral_mujoco_controlled/<new-run-id> \
  --offline-summary results/research/ral_offline_independent/u10-independent-20260711-v5/summary.json \
  --protocol 后续研究内容/benchmark/ral_mujoco_protocol_v0.2.json \
  --out results/research/ral_mujoco_controlled/<new-run-id>/analysis \
  --bootstrap-replicates 2000 --seed 20260711 --strict
python scripts/analysis/ral_controlled_figures.py \
  --analysis-dir results/research/ral_mujoco_controlled/<new-run-id>/analysis \
  --output-dir 投稿冲刺/workspace/controlled_results
```

U11 analysis was independently regenerated into `/tmp/ral-u11-repro` and compared
recursively with the stored analysis; the diff was empty. PDF validity is checked
separately because PDF metadata need not be byte-identical.

## Evidence boundary

The controlled aggregate reads only the named run's primary trial JSONs. Historical
MuJoCo records are rejected by path allowlisting. Policy-only and system-stabilizer
strata are never pooled. With three seeds, all method effects are descriptive.

The U10 image dataset is an external read-only input and is not bundled here. Set a
host-specific dataset path when reproducing U10; do not encode API keys or absolute
dataset paths into result claims.

U10 v5 loads `go_stanford.metric_waypoint_spacing=0.12 m` from
`train/vint_train/data/data_config.yaml`. U10 v2 is invalidated because it used the
RECON scale `0.25 m`; it must not be analyzed or cited.
