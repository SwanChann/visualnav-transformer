# Policy backend exploration

The current integration code is partially decoupled: model loading and sampling
are centralized in `NoMaDInferenceModule`, but `Lite3System` constructs the concrete
NoMaD high-level class and accesses its preprocessing internals. The structural
`NavigationPolicyBackend` boundary enables constructor injection without importing
Torch, MuJoCo, ROS, or Lite3 into the contract.

Candidate-diversity adaptive routing was replayed on corrected U10 v5 (15 paired
cases, GO Stanford scale 0.12 m). TTS fallbacks were dominated; DDIM-3/DDPM routing
showed only post-hoc accuracy/latency tradeoffs and no pre-registered deployable
threshold. The router is therefore rejected. No selector is a safety mechanism.
