# Single RTX 4090 TinyNavBrain plan

This package contains design and static validation only. No training, backward pass,
or optimizer was executed.

`TinyNavBrain-ScaleAdaptive` uses a shared EfficientNet-B0 encoder and explicit
physical-scale/action-history tokens. A deterministic NFE=1 action-chunk head is the
strong efficiency baseline; a small meter-space rectified-flow head is tested at NFE
1/2/4/8. If it cannot outperform the deterministic head at equal data, parameters,
steps and inference budget, the generative-head direction stops.

Repository records currently verify only GO Stanford as materialized; the live 4090
inventory has not been checked in this Windows session. Cross-dataset evaluation remains locked
until at least three datasets have legal, processed, trajectory-disjoint splits.
Configured batch sizes and memory accounting are candidates, not 4090 measurements;
a future smoke profile is mandatory.

All datasets reside only on the 4090 server. Windows owns static contracts and Git/paper
integration; 4090 owns real-data adapter integration, training and cross-dataset offline
evaluation; Ubuntu owns closed-loop simulation of promoted checkpoints.
