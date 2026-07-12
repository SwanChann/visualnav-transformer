# Ubuntu No-Training Environment Report

- Environment: `nomad_train`
- Conda executable: `/home/swanchan/anaconda3/bin/conda`
- Python: `3.8.5`
- Renderer selected: MuJoCo EGL (`MUJOCO_GL=egl`)
- Scope: frozen inference, offline analysis and simulation only; no optimizer,
  backward pass, training, fine-tuning or real-robot execution was run.

## Core runtime

| Component | Observed value | Status |
|---|---|---|
| PyTorch | `2.4.1+cu121` | pass |
| CUDA available | RTX 3050 Ti, CUDA runtime 12.1 | pass |
| NumPy | `1.24.3` | pass (`<2`) |
| torchvision | `0.19.1+cu121` | pass |
| OpenCV | `4.6.0` | pass |
| diffusers | `0.11.1` | pass |
| MuJoCo | `3.2.3` | pass |
| PyBullet | `3.2.7` package import | pass |
| ONNX Runtime | `1.19.2` | pass |
| h5py | `3.6.0` | installed/pass |
| lmdb | `1.7.5` | installed/pass |
| vit-pytorch | `1.15.7` | installed/pass |
| pandas | `2.0.3` | installed/pass |
| seaborn | `0.13.2` | installed/pass |

## Editable sources

- `vint_train` resolves to
  `/home/swanchan/visualnav-transformer-worktrees/agent-ubuntu-sim-handoff/train/vint_train`.
- `diffusion_policy` resolves from `/home/swanchan/diffusion_policy`, commit
  `5ba07ac6661db573af695b419a7947ecb704690f`.

The environment initially contained a stale `easy-install.pth` entry for the other
worktree. It was removed after reinstalling this worktree in editable mode; an
assertion now verifies the resolved module path before experiments.

## Verification executed

1. Imported torch, torchvision, OpenCV, NumPy, YAML, diffusers, MuJoCo,
   PyBullet, ONNX Runtime, h5py, lmdb, pandas and seaborn.
2. Imported `ConditionalUnet1D`, `NoMaD` and `scripts.shared.nomad_inference`.
3. Loaded the tracked Lite3 MJCF model with MuJoCo 3.2.3.
4. Created a `64x64` EGL renderer and rendered an RGB array with shape
   `(64, 64, 3)` and dtype `uint8`.
5. Ran `nomad_mujoco_lite3_state_machine.py --help`; the seed, TTS and
   route-stabilizer controls are exposed.
6. Ran `scripts/tooling/env_check.py`; project structure, CUDA, 3696 Go Stanford
   trajectories and the restored local checkpoint passed. The only structure warning
   was the intentionally absent legacy `毕设冲刺/` directory.

## Environment lock

`pip_freeze.txt` was generated with `pip list --format=freeze` so it contains
package/version pairs without user-local editable paths or credentials. Editable
source commits and paths are recorded above and in `input_inventory.json`.

## Known runtime warnings

- `conda` is not on the default non-interactive PATH; automation must invoke
  `/home/swanchan/anaconda3/bin/conda run -n nomad_train ...` or initialize conda.
- Git LFS is not installed, but the selected controlled inputs are present and
  independently hashed.
- The historical gitlink metadata is inconsistent; controlled simulation uses
  tracked `third_party/lite3/` assets.
