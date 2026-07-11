# TinyNavBrain Fair-baseline Matrix v0.1

| Factor | B2 Multi-NoMaD | B3 Balanced NoMaD | B4 Deterministic H0 | B5 DDIM-2/3 | TinyNavBrain H1 |
|---|---|---|---|---|---|
| Manifest/split | same | same | same | same | same |
| Dataset sampler | proportional | balanced | balanced | balanced | balanced |
| Encoder | frozen comparison choice | same | same shared encoder | same | same shared encoder |
| Fusion width/depth | report | report | same as H1 | report | 256 / 4 layers |
| Action target | meter after reversible normalization | same | meter | meter | meter |
| Action history/scale | baseline off/on ablation | baseline off/on ablation | same as H1 | same as configured | on |
| Head | DDPM noise UNet | DDPM noise UNet | MLP chunk | DDIM sampler | rectified flow |
| NFE | 10 reference | 10 reference | 1 | 2 / 3 | 1 / 2 / 4 / 8 |
| Candidate K | 1 | 1 | 1 | 1 | 1 primary；K>1 separate |
| Params/data exposure | report/fix | fix | ±10% or scaling curve | report | <20M/fix |
| Primary role | more-data control | sampler control | generative-necessity control | low-NFE control | proposed candidate |

## 禁止的对比

- TinyNavBrain 使用 action history/scale，而所有 baseline 不使用。
- TinyNavBrain 用 balanced sampling，对比 proportional NoMaD 后把收益归因于模型。
- best-of-8 TinyNavBrain 对比 single-sample baseline。
- flow NFE=8 对比 DDIM NFE=2 却只报告成功率。
- 不同 encoder、resolution 或 bridge/stabilizer 的结果合并成“head 对比”。
