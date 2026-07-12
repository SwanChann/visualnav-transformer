# RA-L MuJoCo Calibration v0.2

冻结状态：baseline-only 场景校准前冻结。

- 权威 overlay：`ral_mujoco_calibration_v0.2.json`
- SHA-256：`3c2eec1b286879a06601c240035491dbb5adf19bb8c3a518775e9a0ef5aa75f3`
- base protocol：v0.1，SHA-256 `0dffb341642027af89a51bf81df0053e6682647e062e41a7db0e75a4ef68e62f`
- 唯一规则变化：所有 scene/config/seed/stabilizer 统一使用 `max_steps=62`，即 15.5 s simulated navigation time。
- 不变：checkpoint、scene/topomap、0.5 m success radius、K=8、controller、failure taxonomy、三 seed 和 stabilizer 分层。

选择该预算只查看了 DDPM-10 baseline 的 v0.1 calibration timing；未查看其他受控方法结果。校准阶段仍只运行 DDPM-10。最终 v0.2 protocol 只能根据本文件预声明的 method-blind gate 保留/拒绝场景，随后必须重新计算 SHA 才能运行跨方法矩阵。
