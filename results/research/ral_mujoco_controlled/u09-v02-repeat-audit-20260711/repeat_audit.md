# Seeded Stochastic Repeat Audit

The exact `medium / seed=23 / policy_only_off` key was repeated for all five
configurations after the primary U09 matrix. The repeat is a robustness audit and is
not pooled into the frozen primary denominator.

| Configuration | Primary | Repeat | Primary final m | Repeat final m | max common-prefix coordinate delta m |
|---|---|---|---:|---:|---:|
| DDPM-10 | fail | fail | 0.605 | 0.461 | 0.270 |
| DDIM-2 | success | success | 0.424 | 0.465 | 0.103 |
| DDIM-3 | success | fail | 0.431 | 3.647 | 3.300 |
| DDIM-2 + TTS-8 | success | success | 0.484 | 0.498 | 0.098 |
| DDIM-2 + CFG-2 + TTS-8 | success | success | 0.487 | 0.490 | 0.107 |

The earlier easy-scene smoke was bitwise repeatable, but medium is not. The likely
source is compounded GPU/ONNX/physics numerical nondeterminism near a tight terminal
time boundary. Therefore the three frozen seeds are treated as seeded stochastic
replicates; primary success fractions describe one immutable matrix realization and
do not establish stable method rankings or significance. The repeat records remain
separate and visible rather than replacing primary failures/successes.
