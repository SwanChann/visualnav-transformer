# Ubuntu Transfer Audit

- Recorded at: `2026-07-11T17:06:33+08:00`
- Handoff: `visualnav-transformer/ubuntu-sim-offline/2026-07-11-1231`
- Agent role: `ubuntu-simulation-offline-agent`
- Repository: `/home/swanchan/visualnav-transformer-worktrees/agent-ubuntu-sim-handoff`
- Branch/upstream: `agent/ubuntu-sim-handoff` / `origin/agent/ubuntu-sim-handoff`
- HEAD: `463ac70129c6afec71fa84e3eba457d436781642`
- Ahead/behind at audit: `0/0`
- Worktree state before audit writes: clean

## Transfer integrity

The current checkout is the published transfer branch, not default `main`. The
following critical directories are present:

- `.agents/`
- `results/research/`
- `scripts/research/`
- `后续研究内容/`
- `投稿冲刺/workspace/`

The latest handoff, exhaustive Ubuntu backlog, persistent workstream records, and
project snapshot were read before execution. The older `SESSION_HANDOFF.md` and
2026-07-09 RA-L packet are background only.

## Host

- OS: Ubuntu 20.04.6 LTS (`focal`)
- Kernel: `5.15.0-139-generic`
- Architecture: `x86_64`
- CPU: Intel Core i5-12500H, 16 logical CPUs visible
- RAM: 15 GiB total, approximately 9.5 GiB available during audit
- Home filesystem: ext4, approximately 30 GiB free during audit
- GPU: NVIDIA GeForce RTX 3050 Ti Laptop GPU, 4096 MiB
- Driver: `535.183.01`
- CUDA toolkit: `12.2`; PyTorch runtime CUDA: `12.1`
- Git: `2.25.1`
- Conda executable: `/home/swanchan/anaconda3/bin/conda` (not on default PATH)
- Requested environment: `nomad_train`, Python `3.8.5`

## Git/worktree observations

Two registered worktrees exist:

1. `/home/swanchan/visualnav-transformer` on `main` at `0f1284a...`.
2. This worktree on `agent/ubuntu-sim-handoff` at `463ac70...`.

Git LFS is not installed. `git submodule status --recursive` fails because the
index contains historical gitlinks such as `Lite3_rl_deploy` while the current
`.gitmodules` has no matching mapping. Controlled simulation must use the tracked
assets under `third_party/lite3/`, not the empty gitlink directory.

## Initial blockers and resolutions

- The transfer branch intentionally excludes `*.pth`. The expected NoMaD baseline
  was found in the main worktree as `deployment/model_weights/nomad_baseline.pth`,
  verified against the handoff SHA-256, and copied to the ignored local path
  `deployment/model_weights/nomad/nomad.pth`.
- The grouped-v2 manifest hash differs only by line endings: the Ubuntu LF bytes hash
  to `b9ea199f...`, while a CRLF normalization hashes to the handoff value
  `ad75c573...`. The CSV content is therefore accounted for; both hashes are retained
  in the input inventory.
- `vint_train` initially resolved to the other worktree. U02 must re-point the
  editable install to this worktree before any controlled run.

## Scope boundary

Allowed here: environment repair, frozen-checkpoint inference, offline evaluation,
MuJoCo/PyBullet simulation, analysis, tests and paper evidence integration.
Training/fine-tuning/distillation, real-robot execution and automatic real-trial
outcome labeling remain prohibited.
