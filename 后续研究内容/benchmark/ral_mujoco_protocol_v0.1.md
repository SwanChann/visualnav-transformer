# RA-L Controlled MuJoCo Protocol v0.1

状态：已冻结，冻结时间 `2026-07-11T09:30:00Z`。机器可读的唯一权威版本是同目录的 `ral_mujoco_protocol_v0.1.json`；本文件只解释设计和记录冻结哈希。

## 研究问题与边界

在同一个冻结 NoMaD checkpoint、相同候选数 `K=8` 下，比较 DDPM-10 基线、DDIM-2/3、CFG 和 ranked TTS 是否改变闭环成功率、路径效率与延迟。这里只允许冻结 checkpoint 推理、MuJoCo 仿真和只读分析；不允许训练、微调、反向传播、optimizer step 或真机运行。

## 冻结矩阵

五个配置为：

1. DDPM-10 / CFG-0 / 标准 batch K=8，8 条轨迹取均值；
2. DDIM-2 / CFG-0 / 标准 batch K=8，8 条轨迹取均值；
3. DDIM-3 / CFG-0 / 标准 batch K=8，8 条轨迹取均值；
4. DDIM-2 / CFG-0 / TTS budget=8，heuristic 排序取 top-1；
5. DDIM-2 / CFG-2 / TTS budget=8，heuristic 排序取 top-1。

候选生成数始终为 8。标准 batch 与 ranked TTS 的区别是选择规则，不得把 TTS 的额外选择操作描述成 K=1 推理。

候选场景为 `easy`、`medium`、`hard`，使用各自的 MuJoCo 域 topomap 和固定末端目标；diffusion seeds 为 `11, 23, 47`，固定 `goal_seed=0`。route stabilizer 的 off/on 是两个独立层，禁止池化。U08 只能以 baseline 或 method-blind aggregate 做统一 scene gate，不能逐方法调整障碍、半径、timeout 或控制器。

冻结前计划量为 `5 configs × 3 scenes × 3 seeds × 1 goal seed × 2 stabilizer strata = 90 trials`。一个闭环 episode 是统计单位。

## 控制与成功定义

- 输入：stretch 到 `96×96`，4 帧上下文，8 waypoint horizon，执行 waypoint index 2。
- NoMaD：4 Hz；MuJoCo physics dt 0.001 s；locomotion policy dt 0.02 s。
- waypoint bridge：线速度上限 0.4 m/s，角速度上限 0.8 rad/s；额外 PD scale 均为 1，无额外 PD clamp。
- 成功：机器人平面位置在 timeout 前满足 `distance < 0.5 m`，且没有 terminal fall。
- timeout：240 个 NoMaD cycle，即 60 s 仿真导航时间。
- stuck：命令运动时，本体速度连续 3 个 NoMaD cycle 低于 0.05 m/s；发生 recovery 也不能擦除 stuck 事件。

## 失败、重试与统计

collision、fall、stuck、timeout、intervention、crash 都保留在计划试验分母中。只允许把在仿真开始前因输入哈希、schema 或设备检查失败的记录标为 `infrastructure_invalid` 并排除于科学统计；原记录仍必须保存。恢复运行不覆盖旧记录，同一 trial key 的后续 attempt 必须显式链接。

主指标为 episode success。SPL、final distance、path length、各失败率、recovery 和 latency 是 secondary。按 scene/goal seed/diffusion seed/stabilizer 配对；stabilizer off/on 分别分析。三 seed 统计功效不足时只作描述性结论，不声称显著性。历史 MuJoCo 记录不得进入 controlled aggregate。

延迟分别记录 `sampler_per_call` 与 `full_loop_per_call` 的原始数组，并报告 mean/p50/p95/p99；前三次调用作为 warm-up，不进入延迟统计。CUDA sampler 计时前后必须同步。

## 冻结哈希

JSON 协议 SHA-256：`0dffb341642027af89a51bf81df0053e6682647e062e41a7db0e75a4ef68e62f`。

该哈希在查看完整受控矩阵结果之前计算。任何 JSON 字节变化都会产生新协议版本，不能在看到完整矩阵结果后静默修改。
