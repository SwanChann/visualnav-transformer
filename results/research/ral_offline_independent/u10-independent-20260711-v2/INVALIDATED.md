# INVALIDATED — do not use U10 v2

U10 v2 converted normalized NoMaD outputs with `0.25 m`, which is the RECON
`metric_waypoint_spacing`. The evaluated dataset is GO Stanford, whose frozen
training data config specifies `0.12 m`. Consequently all v2 physical-unit
trajectory errors and candidate-diversity values are invalid.

The corrected complete replacement is `u10-independent-20260711-v5` (75/75 valid,
no training). U10 v2 remains only as an immutable audit trail and must not enter
analysis, figures, tables, correlations, or claims.
