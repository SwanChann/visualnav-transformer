# huron_sacson raw DATA-PILOT dry-run

- Passed: False
- Candidates: 0
- Candidate bytes: 0
- Checksum mode: `metadata`
- Conversion executed: False
- Model/training/evaluation/simulation executed: False

## Planned conversion command (not executed)

Working directory: `/home/yifei/codespace/visualnav-transformer/train`

```text
python process_bags.py --dataset-name sacson --input-dir /home/yifei/codespace/visualnav-transformer/nomad_dataset/huron_raw --output-dir /home/yifei/codespace/visualnav-transformer/nomad_dataset/sacson
```

## Blocking gates

- input_root_present
- expected_raw_layout_present
- raw_candidates_present
- raw_artifact_receipt_schema_binding_valid
- license_snapshot_hash_matches_receipt
- raw_file_sha256_captured

## Evidence boundary

Dry-run inventory and conversion-command rendering only. Raw candidate presence and hashes do not validate payload semantics, physical scale, timestamps, session grouping, conversion correctness, or model behavior.
