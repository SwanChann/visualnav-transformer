# RA-L Controlled MuJoCo Protocol v0.2

状态：完整跨方法矩阵运行前已冻结。

- 权威 overlay SHA-256：`4df19bb8d1d15d6969c1df52c2b2f339e5f489dc3bb6db140fe3586fdd9497d6`
- base v0.1 SHA-256：`0dffb341642027af89a51bf81df0053e6682647e062e41a7db0e75a4ef68e62f`
- calibration v0.2 SHA-256：`3c2eec1b286879a06601c240035491dbb5adf19bb8c3a518775e9a0ef5aa75f3`
- retained scenes：`easy`、`medium`；`hard` 在 baseline-only 校准中 0/6，已在任何跨方法结果产生前淘汰。
- 全局 timeout：62 NoMaD cycles / 15.5 s；success radius 仍为严格 `<0.5 m`。
- 矩阵：5 configs × 2 scenes × 3 diffusion seeds × 1 goal seed × 2 stabilizer strata = 60 trials。

所有配置继续固定 K=8；standard batch 取 8 条均值，TTS-8 用 heuristic 排序取 top-1。policy-only 与 system-stabilizer 两层分别分析，禁止池化。三 seed 结果只作描述性证据。历史 MuJoCo 数据不得进入 controlled aggregate。
