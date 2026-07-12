# U10 v4 pre-inference failure

- Status: invalid configuration attempt; no model inference executed.
- Cause: the overlay-only MuJoCo v0.2 protocol does not contain a standalone
  `configurations` array. The independent offline runner requires the full v0.1
  base protocol used by U10 v2.
- Correction: v5 uses the full frozen v0.1 protocol and the corrected 0.12 m
  dataset scale.
- Scientific use: none. Do not ingest this directory into any analysis.
