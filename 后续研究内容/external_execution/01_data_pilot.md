# Stage 1 — licensed data pilots

## Order

1. RECON: acquire the smallest official HDF5 shard that exercises conversion.
2. HuRoN/SACSoN: acquire a small official bag spanning image and odometry topics.
3. Do not acquire SCAND or TartanDrive payloads until dataset-payload terms are
   recorded explicitly; repository code licenses are insufficient.

For each artifact, save the official license/terms page as a dated text or PDF
snapshot and create a receipt:

```powershell
python scripts/research/data/register_raw_artifact.py `
  --registry 后续研究内容/data/dataset_registry.json `
  --dataset-id recon `
  --artifact <raw-file> `
  --source-url <exact-official-file-url> `
  --license-snapshot <dated-license-file> `
  --license-name MIT `
  --operator <name-or-id> `
  --out <receipts>/recon_<artifact>.json
```

Then convert into a quarantine directory, build the unified manifest, and run the
dataset audit before moving anything into the canonical processed root. Preserve raw
session/bag identity. Acceptance: checksum receipt, decodable images, metric pose or
odometry, monotonic timestamps, non-empty trajectories, and zero grouped-split
leakage. Reject silent frame drops, per-frame random splits, and missing provenance.

Before releasing any derived Go Stanford artifact, include attribution, the
CC-BY-NC-SA-3.0 notice, non-commercial restriction, and share-alike obligations.
