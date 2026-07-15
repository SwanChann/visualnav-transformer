# TinyNavBrain Go Stanford B0 Audit

- Passed: True
- Git: `45d4d14e4481819b8317173f6420bef5f7beec09`
- Optimizer steps / backward calls: 400 / 800
- Local checkpoint bytes: 807960660
- Evaluation performed: False
- Pretrained weights downloaded: False

## Runs

### h0-200

- Steps/backward/examples: 200 / 400 / 1600
- Finite / GradScaler no-skip / exact-resume: True / True / True
- Scoped steady step median/p95: 78.931 / 100.271 ms
- Scoped effective examples/s: 101.355
- Training-step peak CUDA memory: 478.69 MiB

### h1-200

- Steps/backward/examples: 200 / 400 / 1600
- Finite / GradScaler no-skip / exact-resume: True / True / True
- Scoped steady step median/p95: 79.467 / 96.497 ms
- Scoped effective examples/s: 100.671
- Training-step peak CUDA memory: 488.52 MiB

## Gates

- config_hash_exact: True
- runner_hash_exact: True
- execution_budget_exact: True
- no_evaluation: True
- no_pretrained_download: True
- two_runs_present: True
- each_run_budget_exact: True
- all_values_finite: True
- no_grad_scaler_skip: True
- all_exact_resume_gates_pass: True
- all_samples_unique_within_each_run: True
- meter_targets_finite_and_under_10m: True
- checkpoint_retention_exact: True
- all_checkpoint_receipts_exact: True
- checkpoint_space_under_authorized_1gib: True

## Evidence boundary

This audit establishes B0 plumbing, finite recorded values, exact resume, local checkpoint receipts, and scoped smoke timing/memory only. Loss ranges are not convergence or model-quality evidence; the <10 m check is not independent physical calibration.
