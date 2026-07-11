# RA-L Claim–Evidence Ledger

更新日期：2026-07-10
机器可读版本：[`claim_evidence_matrix.csv`](claim_evidence_matrix.csv)

## 使用规则

1. 正文中的每个定量或贡献性陈述必须能映射到 CSV 中一行。
2. `unsupported` 不得进入摘要、贡献或结论；`partial` / `conditional_circular` 必须同时带限制。
3. `source_paths` 是证据入口，不代表目录内所有自动生成结论都可信；以 raw/aggregated 数据和脚本复核为准。
4. 规划文档、教程和拟议实验不是实验结果。

## 当前证据判决

| 结论 | 判决 | 原因 |
|---|---|---|
| Lite3 部署栈存在 | 代码层支持 | 主机调度、真实桥接、SDK 控制和配置均有实现；实际鲁棒性仍需证据。 |
| 离线 latency–proxy trade-off | 有限支持 | 统一脚本可复现，但 proxy 不是成功率，且只能在同一 source 内解释。 |
| TTS 改善导航质量 | 不可独立成立 | verifier 与 proxy 共用统计项，存在结构性循环。 |
| p95 是实时系统尾延迟 | 不成立 | 当前 p95 是 case-level mean latency 的分位数。 |
| MuJoCo 比较证明某编码器导航更优 | 当前不成立 | encoder benchmark 有四种 encoder，但行为指标饱和/重复，主要只显示 wall-time 差异；部分其他报告还误用了“多编码器”模板。 |
| 真机系统跑通过 | 代表性支持 | 有视频、元数据和部分 timing；结果必须人工复核，不能当统计 benchmark。 |
| 跨数据集/跨形态泛化 | 不支持 | 统一离线数据全部来自 Go Stanford。 |

## 投稿硬门槛

- G2/G3：已经具备可复现产物，但 Fig. 2/caption 必须避免跨 source 全局比较暗示。
- G4：尚需从 existing raw JSON 生成保守的 Table III；若只剩饱和结果和 stabilizer 主导，RA-L 仍是硬风险。
- G5：尚需人工视频标注与 timing CSV 汇总。未完成人工复核时，只能证明有素材，不能证明成功率。
- G6：限制必须逐条写进正文，不可仅放补充材料。

## 已完成复现检查

2026-07-10 使用临时目录重跑：

```powershell
python scripts/analysis/ral_offline_pareto.py --results results --out <temporary-output>
```

结果：4 个文本/CSV 产物与 `投稿冲刺/workspace/offline_pareto/` 逐字节一致；生成 131 行统一配置、2004 条 case record 审计记录、11 行 Table II 代表配置；两个 PDF 均为有效 `%PDF-1.4` 文件。临时目录在比对后删除，未覆盖现有产物。
