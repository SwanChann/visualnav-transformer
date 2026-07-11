# Multi-dataset Visual Navigation Benchmark Protocol v0.1

状态：冻结候选（2026-07-10）
适用对象：NoMaD、同编码器 deterministic baseline、TinyNavBrain 及后续生成式策略
前置数据契约：[`../data/dataset_registry.json`](../data/dataset_registry.json)

## 1. 研究问题

Benchmark 不回答“哪个模型平均误差最低”这一单点问题，而要拆开四个混杂因素：

1. 训练数据是 single、proportional mixture、balanced mixture 还是 held-out transfer；
2. action 的物理尺度和历史条件是否被显式建模；
3. 输出头是 deterministic、DDPM/DDIM 还是 flow/rectified-flow；
4. 性能差异是否在固定参数量、数据曝光量和 NFE 预算后仍存在。

只有当至少 3 个许可明确、processed、manifest audit 通过的数据集存在时，Track B/C 才能被称为 multi-dataset benchmark。当前只有 Go Stanford，因此本协议不代表已有实验结果。

## 2. 冻结输入

每次实验必须记录：

- dataset registry 版本和 SHA-256；
- trajectory manifest SHA-256；
- split ID 和 split 文件 SHA-256；
- Git commit、dirty 状态和 diff SHA-256；
- model/config ID、参数量、trainable 参数量、输入分辨率、context/action horizon；
- sampler、每数据集曝光量、optimizer steps、seed；
- 推理 device、batch、warm-up、NFE 和 latency scope。

禁止逐帧随机切分。trajectory 的裁剪、相邻 segment、不同 camera 或派生版本必须共享同一 `leakage_group`。

## 3. Tracks

### Track A — IID specialist

- 每个数据集单独训练和测试。
- 使用 group-safe train/test；测试 trajectory/session 不出现在训练集。
- 作用：给出 dataset specialist 上限和数据管线 sanity。
- 不得把 A 称为跨数据集泛化。

### Track B — Mixed-domain training

- 训练集包含所有可用数据集；对每个 dataset 的 group-safe test 单独报告。
- 比较 proportional、dataset-balanced、temperature sampling。
- primary aggregation 是 dataset macro average；同时报告 micro average，不能只报 micro。
- 固定 optimizer steps 与总样本曝光量；额外报告每个 dataset 实际看到的样本数。

### Track C — Leave-one-dataset-out (LODO)

- 每次训练 N-1 个数据集，在完全 held-out dataset 测试。
- held-out dataset 不参与模型选择、normalization 统计、early stopping 或 calibration。
- 如果模型需要物理尺度 metadata，可使用数据集发布的传感器/尺度元数据，但不能读取 held-out action 分布来调参。
- 少于 3 个数据集时只叫 cross-domain case study，不叫 LODO benchmark。

### Track D — Controlled corruption

- 只在冻结 test 数据上施加 deterministic corruption。
- v0.1 类别：brightness、contrast、Gaussian noise、motion blur、frame drop、temporal jitter、goal-image mismatch。
- severity 至少 3 级；每个 corruption seed 固定并记录。
- corruption transform 不能改变 ground-truth action 或泄漏目标轨迹。
- 报告 clean-relative degradation 和 absolute metric，不只报相对百分比。

### Track E — Non-saturated closed loop

- 本地环境不能执行，该 track 由外部仿真/真机环境完成。
- 场景必须先用 baseline 校准到非饱和：至少一个强 baseline 的 success 不得为 0% 或 100%。
- 固定地图、spawn、目标、障碍、seed、success radius、timeout、collision/stuck 定义。
- route stabilizer/recovery 必须有 `off/weak/normal` 明确枚举；不同 stabilizer 不得直接池化。
- 结果按 episode 保存，不能只留 aggregate prose。

## 4. Baseline 顺序

| ID | Baseline | 目的 |
|---|---|---|
| B0 | Go-only NoMaD pipeline sanity | 验证代码可运行，不作为创新对照 |
| B1 | per-dataset NoMaD specialists | 单数据集上限 |
| B2 | proportional multi-dataset NoMaD | 排除“只因更多数据” |
| B3 | balanced multi-dataset NoMaD | 排除 sampler 混杂 |
| B4 | same-encoder deterministic chunk predictor | 检验生成式 head 是否必要 |
| B5 | same-encoder DDIM-2/3 | 固定低 NFE 推理预算 |
| M1 | TinyNavBrain | 仅在 B0–B5 稳定后进入 |

TinyNavBrain 不得先于 B2/B3/B4 实验。无法训练公平的 multi-dataset NoMaD 时，任何提升都不能归因于新框架。

## 5. 公平预算

### 数据/采样比较

- 固定总 optimizer steps、global batch 和总样本曝光量。
- 记录 `examples_seen_by_dataset`。
- proportional/balanced/temperature 只改变 sampler，不改变模型或 augmentation。

### 模型比较

- 固定 manifest、split、sampler、augmentation、encoder 和输入分辨率。
- trainable 参数量优先控制在 ±10%；超出时单独报告 scaling curve。
- 固定 action horizon、meter-scale postprocess 和控制接口。
- latency 对比必须同时给 NFE、batch、device、precision 和 scope。

### 随机性

- 科学结果至少 3 个 training seeds；资源允许时 5 个。
- corruption/closed-loop seed 与 training seed 分开记录。
- 置信区间按 trajectory/session 或 episode bootstrap，禁止把相邻 frame 当独立样本。

## 6. 输出与验证

每个 run 写一个符合 [`result_schema_v0.1.json`](result_schema_v0.1.json) 的 JSON。结果必须经过：

```powershell
python scripts/research/validate_benchmark_result.py --result <run.json>
```

`status=completed` 但缺 raw artifact、manifest hash、关键 metric 或 track metadata 时判失败。validator 通过只证明结果契约完整，不证明指标真实。

## 7. Go / No-Go

- Gate A：3+ processed datasets，许可与 manifest audit 通过。
- Gate B：B1/B2/B3 可复现，macro/micro 与每数据集结果齐全。
- Gate C：B4/B5 公平；TinyNavBrain 在至少两个 held-out/corruption 指标上跨 seed 改善。
- Gate D：closed-loop 非饱和，改善不完全依赖 stabilizer。
- Gate E：主张能被 result JSON、raw artifact 和脚本逐项追溯。

任何 Gate 失败都触发降级：优先退到 scale/corruption diagnostic，而不是继续堆模型模块。
