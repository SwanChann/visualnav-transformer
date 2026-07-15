# TinyNavBrain Architecture Contract v0.1

状态：接口冻结候选，不代表已训练或有效
目标：在固定数据、encoder、action horizon 和推理预算下，检验小型条件生成 head 是否比 deterministic / NoMaD-DDIM baseline 更能跨 dataset 保持导航行为。

## 1. 不变项

- 输入仍是 observation history + goal image，不引入语言或地图特权信息。
- 输出仍是局部 waypoint chunk，保持现有 waypoint-to-velocity / Lite3 bridge 接口。
- 所有监督和指标在 meter/radian 空间定义；dataset normalization 只允许出现在可逆预处理层。
- `dataset_id` 用于 sampler、分层报告和审计，默认不作为模型 token，避免模型只识别数据集身份。
- v0.1 不包含 world model、learned safety critic、RL、preference fine-tuning 或大 VLM。

## 2. Canonical batch contract

| Field | Shape / type | Unit | Required | 说明 |
|---|---|---:|---|---|
| `obs_images` | `[B, C+1, 3, H, W] float32` | ImageNet normalized | yes | 当前帧含在 history 末尾；adapter 可从现有 `[B,3(C+1),H,W]` 转换 |
| `goal_image` | `[B,3,H,W] float32` | ImageNet normalized | yes | goal-masked 时仍保留 shape |
| `goal_mask` | `[B] bool` | — | yes | `true` 表示 exploration / goal absent |
| `action_history` | `[B,A_hist,3] float32` | `dx_m,dy_m,dyaw_rad` | yes | 过去实际/示教相对动作；缺失处为 0 |
| `action_history_mask` | `[B,A_hist] bool` | — | yes | 区分 padding 与真实静止 |
| `dt_s` | `[B,1] float32` | s | yes | 必须来自 manifest/processor，不从 dataset ID 猜 |
| `waypoint_spacing_m` | `[B,1] float32` | m | yes | 物理尺度 token；不替代 action 的 meter 表示 |
| `target_waypoints` | `[B,T,2] float32` | m | train/eval | 当前局部坐标系中的 future xy |
| `target_heading` | `[B,T,2] float32` | sin/cos | optional | 不把 wrapped angle 直接做 MSE |
| `target_mask` | `[B,T] bool` | — | train/eval | padding/negative goal 不参与 action loss |
| `dataset_id` | `list[str]` metadata | — | audit | 不进入 v0.1 forward |
| `trajectory_id` | `list[str]` metadata | — | audit | bootstrap/leakage unit |

默认：`C=5`（6 帧含当前帧）、`A_hist=4`、`T=8`、`H=W=96`。修改这些值会产生新 contract version。

## 3. Encoder and tokenization

### Shared visual encoder

- 一个 EfficientNet-B0 级共享 RGB encoder，对 observation frames 和 goal image 复用权重。
- 每张图输出一个 `d_model=256` token；不保留现有 NoMaD 的第二个 6-channel goal encoder，以控制参数量。
- goal relation 由 `[current_token, goal_token, current-goal difference, elementwise product]` 经 MLP 投影成 relation token。
- goal-masked 时使用 learned null-goal token；不能把全零图当作 goal absent 的唯一信号。

### History and scale tokens

- `action_history` 每步由 2-layer MLP 投影为 token，并加 temporal position。
- `log(dt_s)`、`log(waypoint_spacing_m)` 经过 MLP 形成一个 scale token。
- 不在 v0.1 输入 robot name、dataset index 或 camera filename。

### Temporal fusion

- 4-layer pre-norm Transformer encoder；`d_model=256`、4 heads、FFN=1024、dropout=0.1。
- token 顺序固定：`[CLS, obs_0..obs_C, goal_relation, action_hist_0.., scale]`。
- 参数预算目标：总 trainable `<20M`；每次实现必须由脚本报告实际值，不能按模块名估算。

## 4. Heads

### H0 — deterministic baseline

- 与 TinyNavBrain 完全共享 encoder/fusion。
- MLP 一次输出 `[B,T,2]` meter waypoints。
- 这是生成式 head 必须击败的同表征 baseline。

### H1 — rectified-flow waypoint head

- 状态 `x_t ∈ R^[B,T,2]`。
- 训练采样 `x_0 = target_waypoints`、`x_1 ~ N(0,I)`，`x_t=(1-t)x_0+t x_1`。
- 网络预测 velocity target `v*=x_1-x_0`；mask 后使用 MSE。
- 推理从 `x_1` 沿反向时间 `t:1->0` 积分；NFE 只能取 `{1,2,4,8}`。
- 所有 candidate 使用同一 encoder context；candidate count 与 NFE 分开记录。
- v0.1 不宣称 rectified flow 本身是创新，贡献只能来自受控跨数据集实证与小模型设计。

### Auxiliary heads

- `progress_m [B,1]`：预测 goal progress/distance，用于目标选择与分析。
- `confidence [B,1]`：训练为 held-out waypoint error/calibration target 时才启用；没有 target 时必须关闭，不能当安全分数。
- auxiliary loss 不得读取碰撞地图、未来图像或 test-only label。

## 5. Loss contract

```text
L = L_action
  + lambda_progress * L_progress
  + lambda_heading * L_heading
  + lambda_confidence * L_confidence
```

- deterministic：`L_action = masked SmoothL1(pred_xy_m, gt_xy_m)`。
- flow：`L_action = masked MSE(v_pred, x1-x0)`。
- progress：meter-space Huber。
- heading：`1 - cosine(pred_sincos, gt_sincos)`。
- 每个 dataset 先算均值，再按 sampler/frozen objective 做 macro aggregation；禁止大数据集靠 frame 数支配 loss 日志。

## 6. Sampling and deployment contract

统一 API：

```text
encode(batch) -> context [B,256]
sample(context, nfe, candidate_count, seed) -> waypoints_m [B,K,T,2]
predict_aux(context) -> progress_m, confidence_optional
```

- seed 必须控制初始噪声；同一个 context 的 candidate 可复现。
- `K=1` 是默认公平对比；best-of-K 只能作为独立 TTS/candidate-budget 实验。
- 输出进入 bridge 前应用与 baseline 相同的速度/曲率限制；模型内部不偷偷加入 baseline 没有的 stabilizer。
- 导出接口至少支持 deterministic H0 和 fixed-NFE H1；动态 Python sampler 不算部署完成。

## 7. 与现有 NoMaD 的兼容层

当前 `ViNT_Dataset` 返回 7 元组，缺 action history、`dt_s` 和 string dataset ID。不得直接改 tuple 顺序破坏现有训练。推荐新增 dict-based adapter：

```text
LegacyViNTAdapter(existing_tuple, manifest_metadata) -> canonical batch dict
```

adapter 的第一阶段只做 shape/metadata 变换；真正的 action history 必须由 dataset index 读取过去 pose 构造，不能用 future action 泄漏。

## 8. 公平性检查

TinyNavBrain 只有同时满足下列条件才进入论文方法表：

- same encoder/fusion deterministic H0 已训练；
- multi-dataset NoMaD B2/B3 已训练；
- 输入 history、resolution、meter postprocess、sampler 和 data exposure 固定；
- 参数量、MACs、NFE、K 和 latency scope 均报告；
- 提升在至少两个 dataset/robustness 维度跨 seed 存在，而不是只改善 macro average；
- closed-loop 非饱和结果不完全由 bridge/stabilizer 产生。

## 9. Stop conditions

- `<3` 个 processed datasets：停止 cross-dataset 方法主张。
- H0 与 H1 同预算无显著差异：保留 benchmark，删除“生成式 head 更优”贡献。
- action history/scale baseline 已解释全部收益：将其作为主要工程结论，不包装 TinyNavBrain 新架构。
- NFE=1/2 不稳定且 NFE=8 无部署优势：停止小模型实时生成主线。

## 10. 当前代码状态

`scripts/research/models/tinynavbrain_scaffold.py` 已实现 feature-level fusion、H0、H1 velocity/sample 和 progress API。
4090 IMAGE-FORWARD gate 新增 `tinynavbrain_image_policy.py`，以单一共享、随机初始化且禁止下载权重的 EfficientNet-B0
接受 canonical 图像 batch；Go Stanford 的真实 Dataset/DataLoader 位于 `scripts/research/pretraining/go_stanford_adapter.py`。
当前仍缺 loss/train loop、checkpoint/exact-resume 执行和部署导出，且任何 forward-only 结果都不是训练或方法效果证据。
