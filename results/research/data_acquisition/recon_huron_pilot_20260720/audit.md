# RECON / HuRoN raw pilot acquisition audit

- Passed: True
- Payload written: 53.246385 GB decimal / 350 GB authorized (15.21%)
- Semantic conversion/model/training/evaluation/simulation: 0

## RECON

- Official archive bytes: 53235196027
- Inventory: 11836 HDF5 members, 76383437178 uncompressed bytes
- Selected raw member: `nomad_dataset/recon_raw/recon_release/jackal_2019-12-19-14-24-48_0_r00.hdf5`
- Selected SHA-256: `d850e8cf3b6a3613d7fce54312f19109412edf816031791fdf0838febbcffbb6`
- Strict receipt/preflight/content gates: passed

## HuRoN / SACSoN

- Selected raw bag: `nomad_dataset/huron_raw/Dec-09-2022-bww8/00000010.bag`
- Bytes: 11044659
- SHA-256: `71ac9844157bee7e90f9a6ab1d91af10ebad7e39073fd2885ccabc6d66034397`
- Required fisheye-image and odometry topics/content: passed

## Remaining blockers

- Semantic conversion of either pilot has not been authorized or executed.
- Processed manifests, group-safe splits, dt/scale verification, and processor receipts do not exist yet.
- These two raw pilots do not establish full-dataset completeness or a multi-dataset training result.

## Evidence boundary

Official-page snapshot, raw-byte integrity, minimum content readability, and conversion readiness for one RECON HDF5 and one HuRoN bag only. No semantic conversion, model execution, training, evaluation, or simulation was performed.
