# Ubuntu Cross-Platform Regression Report

Recorded: `2026-07-11`

## Test results

| Suite/check | Result |
|---|---|
| `scripts/analysis/test_*.py` | 5/5 pass |
| `scripts/research/data/test_*.py` | 11/11 pass |
| benchmark validator tests | 7/7 pass |
| benchmark example JSON | pass, 0 errors, 0 warnings |
| TinyNavBrain synthetic shape/forward tests | 5/5 pass |
| legacy MuJoCo navigation `--help` | pass |
| state-machine MuJoCo `--help` | pass |
| `git diff --check` | pass |

No training, optimizer or backward call was run. The first direct benchmark-test
invocation failed because its directory was not on `sys.path`; rerunning with
`unittest discover -s scripts/research` passed 7/7. The first example-validator
invocation passed an unsupported `--schema` CLI flag; using the documented
`--result` form passed. Neither was an implementation regression.

## Path audit

`scripts/tooling/path_audit.py` completed and reported 170 findings in 31 files.
They are primarily:

- historical documentation and archived real-robot metadata;
- intentional `投稿冲刺/workspace/...` prose paths;
- handoff/backlog prose;
- one legacy ROS launch path (`/home/racecar/...`).

No controlled simulator, new analysis implementation, or research validator was
found to depend on a Windows drive letter. The older documentation paths remain as
historical evidence and were not rewritten.

## Linux environment repairs

1. Re-pointed the `vint_train` editable install to this agent worktree.
2. Removed the stale prior-worktree entry from the conda environment's
   `easy-install.pth`.
3. Added the documented missing Python dependencies without upgrading NumPy,
   PyTorch or Diffusers.

The resolved `vint_train.__file__` is asserted to be under this worktree.
