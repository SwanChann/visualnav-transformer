# TinyNavBrain TRAIN-STEP Resume-Only Recovery

- GPU: NVIDIA GeForce RTX 4090 (cuda:0)
- Total backward/optimizer steps across the gate: 2
- Recovery backward/optimizer steps: 0
- Original numeric step losses retained: False
- Post-checkpoint forward-only H0 loss: 0.03625239
- Post-checkpoint forward-only H1 loss: 1.02227986
- Checkpoint: `results/research/pretraining/train_step_gate_20260716/exact_resume_step2.pt` (97.79 MiB)
- Checkpoint SHA-256: `c732eb398462e6af0d9428df550c5bfbe09672eaf7f47cd97c33351e44f8fa35`
- Ready for B0 execution: False

## Failure and repair

The two authorized steps and checkpoint write completed. The first reload failed because
`map_location=cuda:0` moved a CPU RNG ByteTensor to CUDA. Recovery normalizes RNG state
to CPU and performs resume-only verification without replaying either training step.

## Gates

- checkpoint_records_exactly_two_steps: True
- checkpoint_explicitly_not_b0: True
- h0_optimizer_path_evidenced: True
- h1_optimizer_path_evidenced: True
- shared_encoder_two_step_evidenced: True
- finite_optimizer_state: True
- model_differs_from_seeded_initialization: True
- model_state_exact_after_reload: True
- optimizer_state_exact_after_reload: True
- rng_state_exact_after_reload: True
- next_batch_exact_after_reload: True
- finite_postcheckpoint_forward_losses: True
- reload_did_not_change_metadata: True

## Remaining before B0

- review and freeze the TRAIN-STEP implementation/recovery report in Git
- obtain separate explicit authorization for the frozen 200-step B0 run

## Evidence boundary

The checkpoint proves two optimizer-path integration steps and exact resume. Original step loss values were not retained; reported losses are post-checkpoint forward-only checks. This is not B0, convergence, quality, or method evidence.
