# RECON / HuRoN Local Presence Audit

- Passed: False
- Materialized datasets: 0 / 2
- External download performed: False
- Data conversion performed: False
- Model/training/evaluation performed: False

## Verified local facts

### recon

- Expected root: `nomad_dataset/recon`
- Root exists: False
- Processed trajectories: 0
- Manifest rows: False
- Raw receipt / live license snapshot: False / False

### huron_sacson

- Expected root: `nomad_dataset/sacson`
- Root exists: False
- Processed trajectories: 0
- Manifest rows: False
- Raw receipt / live license snapshot: False / False

### recon_datavis

- Size/files: 34451 bytes / 12 files
- Classification: visualization_source_code_not_dataset_payload

## Processor readiness (not data readiness)

- Entrypoints present: True
- Read-only `--help` passed: True
- Dependency specs: `{"PIL": true, "cv2": true, "h5py": true, "numpy": true, "rosbag": true, "rosbags": false, "yaml": true}`

## Blocking gates

- recon_expected_root_present
- recon_processed_trajectories_present
- recon_manifest_rows_present
- recon_raw_receipt_present
- huron_expected_root_present
- huron_processed_trajectories_present
- huron_manifest_rows_present
- huron_raw_receipt_present

## Static processor risks

- processors create output directly and do not emit raw artifact receipts or checksums
- processors do not pin their own code/config version in each processed trajectory
- RECON session identity depends only on HDF5 filename and is not independently registered
- HuRoN output name preserves one parent directory plus bag filename but not collection policy/version metadata
- HuRoN human-containing imagery still requires an official terms/privacy snapshot attached to any manifest

## Evidence boundary

This is a local presence audit only. Processor code and registry records do not count as data. No RECON/HuRoN content, license payload, physical scale, timestamp, or session split was validated.
