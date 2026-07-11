# 双 Agent 协作复盘

更新时间：2026-07-09

## 1. 复盘对象

本复盘审查最近两次使用 `codex-volc-orchestrator` 的协作：

1. **Run 1：后续研究规划**
   - Codex/OpenAI 主控负责项目理解、联网查新、文件创建、最终判断和验证。
   - 方舟 worker 负责后续研究方向、4090 可行性和路线扩展。

2. **Run 2：RA-L 可投性评估与离线补强**
   - Codex/OpenAI 主控负责读取投稿冲刺、结果资产、真机素材，判断当前是否可投，并修改投稿冲刺文件。
   - 方舟 worker 负责审稿视角的可投性判断和补实验建议。

本复盘基于当前会话中的可见过程、工具结果和已落地文件，不假设隐藏记录。

## 2. 评分

### 2.1 Codex 主控初评

| 轮次 | 主控 agent | 方舟 worker | 协作流程 | 总评 |
|---|---:|---:|---:|---|
| Run 1：后续研究规划 | 9.0 / 10 | 8.5 / 10 | 8.5 / 10 | 成功的主控并行推进 + worker 扩展型协作 |
| Run 2：RA-L 可投性评估 | 8.5 / 10 | 6.5 / 10 | 6.5 / 10 | 主控结果可靠，但 worker 调度成本偏高 |

### 2.2 方舟 worker 独立评分

方舟助理给出的评分更严格：

| 轮次 | 主控 agent | 方舟 worker | 协作流程 | 主要理由 |
|---|---:|---:|---:|---|
| Run 1：后续研究规划 | 8 / 10 | 5 / 10 | 7 / 10 | 主控能并行推进，但 worker 迟到，增量有限 |
| Run 2：RA-L 可投性评估 | 5 / 10 | 3 / 10 | 4 / 10 | 两个 worker 超时，第三个短 worker 兜底，流程控制弱 |

### 2.3 整合评分

综合主控事实检查、文件落地质量和 worker 调度成本后，最终采用以下评分：

| 轮次 | 主控 agent | 方舟 worker | 协作流程 | 总评 |
|---|---:|---:|---:|---|
| Run 1：后续研究规划 | 8.5 / 10 | 6.5 / 10 | 7.5 / 10 | 产出有效，但 worker 应更早返回结构化中间结果 |
| Run 2：RA-L 可投性评估 | 7.0 / 10 | 4.0 / 10 | 5.0 / 10 | 主控文件产出可用，但多 agent 流程本身执行偏弱 |
| 综合 | 7.8 / 10 | 5.3 / 10 | 6.3 / 10 | 可用但需要显著改进 task packet、超时和心跳机制 |

## 3. Run 1 优缺点

### 优点

- 主控没有被 worker 阻塞：worker 跑的同时，Codex 继续读项目、联网查新、创建 `后续研究内容/`。
- worker 输出有增量：明确提出 navigation brain、world model、RL fine-tuning、few-step / consistency distillation 等路线。
- 集成质量较高：Codex 没有照搬 worker，而是把 `few-step / consistency distillation` 作为近期快赢，把 `navigation brain / world-action model` 作为长期主线。
- 验证完整：新目录结构、行尾空格、`git diff --check`、`git status` 都做了。

### 缺点

- worker 任务包偏大，导致首次等待超时。
- worker 没有本地文件访问，部分判断只能基于主控摘要；这适合做方向扩展，不适合做最终事实判断。
- 最终文件较多，虽然结构清楚，但后续需要防止 `后续研究内容/` 和 `投稿冲刺/` 两条线混淆。

## 4. Run 2 优缺点

### 优点

- 主控判断扎实：实际检查了 `投稿冲刺/`、`后续研究内容/`、`results/`、`history-doc/毕设冲刺/video/`，不是只按文档愿景下结论。
- 结论明确且有边界：给出“黄灯可投”，不建议立刻换新工作，也不把 ranker / distillation / navigation brain 塞入当前 RA-L。
- 文件修改有效：新增 `04_RA-L可投性评估与离线补强.md`，更新 README，给 `03_实验与写作任务清单.md` 增加可执行的离线补强包。
- 最终 worker 的简短判断与主控一致，增强了结论可信度。

### 缺点

- 前两个 worker 任务包仍然偏宽，等待后关闭时仍在 running，造成方舟资源浪费。
- 主控在 worker 最终反馈前已经完成主要文件修改，worker 变成了事后确认，而不是全过程助理。
- 使用 `close_agent` 检查状态会直接关闭 running worker；这种操作应该更谨慎。
- Run 2 的协作流程满足“最终有 worker 反馈”，但没有达到理想的“worker 先给结构化审稿意见，主控再整合改文档”。

## 5. 对 skill 流程的符合度

| 要求 | Run 1 | Run 2 | 评价 |
|---|---|---|---|
| 读取 skill | 满足 | 满足 | 两次都先读技能说明 |
| 明确主控与 worker 角色 | 满足 | 满足 | Codex 保持最终判断和文件控制 |
| 给出 process frame | 满足 | 满足 | 用户可见过程清楚 |
| worker task packet | 基本满足 | 基本满足但过宽 | Run 2 需要更窄任务包 |
| worker 只读 | 满足 | 满足 | worker 未改文件 |
| Codex 整合判断 | 满足 | 满足 | 主控没有盲从 worker |
| 文件验证 | 满足 | 满足 | `rg`、`git diff --check`、`git status` 已跑 |
| 等待与关闭策略 | 中等 | 偏弱 | Run 2 明显需要改进 |

## 6. 改进建议

### 6.1 worker 任务包要更小

更适合 worker 的任务：

- 给一个 6-10 行评分表。
- 从审稿人角度列 3 个最大风险。
- 对一个已知方案给 yes/no/yellow 判断。
- 生成一版文档提纲或对照表。

不适合一次性丢给 worker 的任务：

- “阅读整个项目并判断能否投稿”。
- “调研最新论文并规划完整一年路线”。
- “既查资料又设计实验又判断投稿策略”。

### 6.2 先本地推进，但保留 worker 真正入口

推荐流程：

1. 主控先做 5-10 分钟本地事实检查。
2. 把事实压缩成短 task packet 发给 worker。
3. 主控继续做不依赖 worker 的文件索引、结果检查、草稿搭建。
4. worker 返回后，只整合有增量的观点。

### 6.3 不要用关闭来轮询 running worker

如果工具面没有可靠 `wait_agent`，应优先：

- 把 worker 输出要求压到极短。
- 等待 subagent notification。
- 只有确定不需要结果时才关闭。
- 如果必须按时推进，在最终输出中明确写 `worker did not return within window`。

### 6.4 worker 输出必须结构化

推荐输出模板：

```text
decision:
score:
top_3_risks:
must_do:
should_not_do:
next_step:
confidence:
```

这样比长篇分析更容易被主控整合。

## 7. 总结

两次协作总体有效：Codex/OpenAI 主控掌握了项目事实、最终判断和文件写入，方舟 worker 在 Run 1 提供了明显增量，在 Run 2 至少提供了独立确认。

主要问题不是“多 agent 没用”，而是 **worker 任务包过宽 + 等待/关闭策略不稳定**。下一次应把方舟 worker 当成短促、结构化、低权限的审稿助理，而不是完整研究代理。
