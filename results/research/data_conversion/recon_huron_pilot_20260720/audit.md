# RECON / HuRoN conversion pilot audit

- Conversion pilot passed: True
- Promotion ready: False
- Peak stage storage upper bound: 1.275230 GB / 10 GB
- Model/backward/optimizer/training/evaluation/simulation: 0

## RECON

- Candidate prefix audited: 1711
- Selected: `recon_release/jackal_2019-09-17-16-09-02_3_r02.hdf5`
- Frames/windows: 14 / 1
- dt: 0.2537814255429789 s (kinematic_estimate_not_timestamp_validated)
- Observed median step: 0.2437799913341394 m

## HuRoN / SACSoN

- Synchronized/retained frames: 67 / 66
- Canonical windows: 53
- dt: 0.28725385665893555 s (measured_from_rosbag_record_time_after_sampling)
- Observed median step: 0.20304962050501574 m

## Promotion blockers

- collection policy/version metadata is not encoded in the selected raw artifact
- per-frame timestamps are absent from the selected RECON HDF5; dt is kinematically inferred

## Evidence boundary

Two isolated processed pilots, integrity checks, and a train-only leakage-safe manifest. RECON dt remains inferred because per-frame timestamps are absent; collection policy/version metadata is absent. No model execution, training, evaluation, or simulation was performed.
