# Navigation Research Task Queue

状态：`ready` / `in_progress` / `blocked_external` / `done` / `stopped`。

| ID | 优先级 | 状态 | 任务 | 本地交付物 | 验收标准 |
|---|---:|---|---|---|---|
| GOV-01 | P0 | done | 建立主控、worktree、worker 与审计治理 | `AGENT.md`、本文件、`log.md` | 职责/边界/工作树/模型锁定均可审计 |
| RAL-01 | P0 | done | 审计现有 RA-L 证据与主张 | claims ledger、来源/缺口表 | 每条主张有来源、限制和投稿状态 |
| RAL-02 | P0 | done | 汇总已有 MuJoCo 结果，不补跑 | 聚合脚本、CSV/Markdown/audit | 对缺失字段显式报错，不臆造结果；真机人工标注仍待完成 |
| RAL-03 | P0 | done | 固化 Fig. 2 / Table II 数据说明 | schema、复现命令、限制说明 | 区分 case-mean tail 与 per-call tail；披露代理循环性 |
| RAL-04 | P0 | done | 汇总冻结真机 timing 与标注入口 | timing CSV/Table/notes + 14-trial annotation template | per-loop 口径明确；自动 status 不作为人工 outcome |
| DATA-01 | P0 | done | 多数据集来源、许可与格式审计 | dataset registry + 审计报告 | 5 个候选均有官方入口、许可状态、格式和阻塞风险 |
| DATA-02 | P0 | done | 统一 manifest 与 split 泄漏检查 | 3 个 Python 工具 + 4 个合成测试 + Go manifest | 旧 split 检出 238 个跨 split session；候选 grouped split 零泄漏 |
| BENCH-01 | P0 | done | 冻结诊断 benchmark v0.1 | protocol、result schema、metric spec + validator | 5 tracks/预算/metric/泄漏与 head-NFE 规则可机器验证 |
| MODEL-01 | P1 | done | 冻结 TinyNavBrain 接口与公平 baseline | architecture contract + configs | 输入/输出/尺度/NFE/参数预算/对照公平性明确 |
| MODEL-02 | P1 | done | 构建 feature-level 模型脚手架 | fusion/H0/H1/progress + 5 shape tests | 合成 tensor、seed、非法输入和参数预算通过；图像 adapter 未实现 |
| EXP-01 | P1 | blocked_external | 训练 baseline 与 TinyNavBrain | 外部训练结果 | 多 seed、固定预算、完整日志 |
| EXP-02 | P1 | blocked_external | 仿真闭环 benchmark | 外部仿真结果 | 冻结场景、种子、失败分类 |
| EXP-03 | P1 | blocked_external | 真机非饱和闭环验证 | 外部真机结果 | 安全协议、原始日志、成功率/效率/失败模式 |
| PAPER-01 | P1 | done | 当前 RA-L 写作证据包 | 图表 caption、method/results skeleton | 本地包完成且不过界；投稿仍受外部 G4/G5 门阻塞 |
| VIDEO-01 | P1 | done | 历史真机视频 AI 辅助审查 | 20-video manifest、contact sheets、14-trial provisional review | 仅记录可见运动；正式 outcome 保持待人工目标核验 |
| PAPER-02 | P2 | done | 下一篇 RA-L claims/go-no-go 包 | claims ledger、图表计划、反证门 | benchmark 与方法贡献都可被单独证伪 |
| DATA-03 | P1 | done | 外部数据/训练/仿真/真机执行包 | receipt 工具、阶段 runbook、promotion/kill gates | 许可未决硬阻断；工具测试通过；不在本机执行外部任务 |
| U00-04 | P0 | done | Ubuntu 转移、输入、环境、回归与冻结证据复现 | `results/research/ubuntu_{environment,reproduction}/` | 权重/CRLF/依赖/EGL/测试/历史数值均核验 |
| U05-06 | P0 | done | 冻结受控 MuJoCo 协议并实现 trial schema/runner | protocol、schema、validator、runner、合成测试 | 90-key 展开；缺 provenance 失败；异常/重试/原子写可审计 |
| U07 | P0 | done | 机械、闭环、确定性与失败触发 smoke | `u07-smoke-20260711/` | 受控 trial 有轨迹与双 scope timing；完全重复轨迹差 0；四类失败触发通过 |
| U08 | P0 | done | baseline-only 非饱和场景门 | `u08-scene-gate-20260711/scene_gate.json` | 18/18 记录有效；三场景均因 6/6 success 饱和被拒绝 |
| U09 | P0 | done | 执行部署尺度修正的冻结受控 MuJoCo 主矩阵 | `u09-v03-controlled-20260711/` | protocol v0.3；action scale 0.1 m；60/60 recorded；0 crash/invalid/missing |
| U10 | P0 | done | 修正独立物理单位离线推理评估 | `ral_offline_independent/u10-independent-20260711-v5/` | GO Stanford 0.12 m；15 cases × 5 configs = 75，invalid=0；v2 作废 |
| U11-15 | P1 | done | 统计、图表、claims、复现包与最终审计 | v0.3 analysis、controlled_results、paper_package、final_audit | strata 分离、三 seed 描述性、重复性审计、138-file repro bundle |
| U16 | P0 | done | 全盘结果纠错与无效证据隔离 | U09 v0.2 / U10 v2 invalidation markers | 下游仅绑定 v0.3/v5；协议与 summary SHA 可追溯 |
| U17 | P1 | done | Policy backend 解耦与无训练探索 | `scripts/research/policy_backend/` | constructor injection；scope/test 通过；adaptive router 因证据不足拒绝 |
| U18 | P1 | done | 单卡 4090 TinyNavBrain 训练设计 | `scripts/research/training_plans/` | 机器可读计划与 validator 通过；未执行训练 |

## 当前执行顺序

Ubuntu 授权范围内 U00–U18 已完成。后续只剩外部顺序：`RECON/HuRoN 小样本获取 -> B0..B5/TinyNavBrain 训练 -> 目标设备 sustained deadline/miss timing -> protocolized repeated robot trials -> privacy/consent review`。当前论文仍不能提交；本地不得用更多仿真替代真机硬门。

Ubuntu 可以继续 MuJoCo 和冻结 checkpoint 离线推理，但仍禁止训练、真机和自动真机 outcome 标注。
