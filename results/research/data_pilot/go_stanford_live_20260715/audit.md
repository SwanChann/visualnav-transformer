# Go Stanford Live DATA-PILOT Audit

- Passed: **True**
- Git commit: `ffa353351e5b24eb83522b1c21bf6439d81ce658`
- Dataset root: `/home/yifei/codespace/visualnav-transformer/nomad_dataset/go_stanford`
- Manifest rows / trajectory directories: 3696 / 3696
- Expected / decoded images: 198126 / 198126
- Canonical-sample-eligible trajectories: 2294
- Canonical samples available under the frozen 6+8 frame geometry: 163058
- Manifest SHA-256: `b9ea199f6b9ba8ea9d2433863efd17684193da2b544f2307510d58ae314add7e`
- Split SHA-256: `f10c2570f67107b8fcd56dcaef6a9cdfffd0ec4d7bc4eafffbb4015304e42753`
- Canonical dataset tree SHA-256: `2ca863b8a79645afe31860d3b59a436e5a5e8b4e4032bc530b37535c661b661b`

## Contract checks

- Scale agreement: True (manifest/registry/loader = 0.12 m).
- Metadata backups matching primary pickle: 3696.
- Per-frame timestamps present: False.
- Timestamp monotonicity verified: False.
- Raw artifact receipt present: False.
- Processor version pinned: False.

## Errors

- None.

## Evidence boundary

This audit establishes processed-data readability and contract consistency only; it is not training, model-quality evidence, independent physical calibration, or a raw-artifact provenance receipt.

The full DATA-PILOT backlog item remains incomplete while RECON and HuRoN are not authorized/materialized.
