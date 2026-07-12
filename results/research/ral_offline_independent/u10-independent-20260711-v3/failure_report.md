# U10 v3 pre-inference failure

- Status: invalid infrastructure attempt; no model inference executed.
- Cause: the integration-worktree `nomad_dataset/go_stanford` copy contains
  trajectory metadata but no image frames, so case freezing returned 0/15.
- Correction: v4 explicitly uses the complete main-worktree dataset as a read-only
  input while retaining the integration-worktree split and data configuration.
- Scientific use: none. Do not ingest this directory into any analysis.
