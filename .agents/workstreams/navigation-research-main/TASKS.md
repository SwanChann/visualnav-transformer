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

## 当前执行顺序

所有本地允许任务已完成。下一顺序由外部环境执行：`RECON/HuRoN 小样本获取 -> B0..B5 训练 -> 非饱和仿真 -> 目标设备 timing -> 真机`。当前论文还需协议化人工 outcome 与新受控实验，不能提交。

外部任务 `EXP-01..03` 只准备执行包，不在本机启动。
