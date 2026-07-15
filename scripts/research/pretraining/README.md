# Pre-training infrastructure

本目录只包含 TinyNavBrain / multi-dataset pre-training 的静态合同、配置、backlog 和单元测试。
导入模块或运行验证器不会读取外部数据，不会执行 forward/backward，不会创建 optimizer，也不会写 checkpoint。
这些静态工作归 Windows；完整数据只驻留 4090，真实 adapter 接线、smoke、训练和跨数据集离线评估均在 4090 执行。Ubuntu 只接收通过离线门的 checkpoint 做闭环仿真。

本地验证：

```powershell
conda run -n prp python -m unittest discover -s scripts/research/pretraining -p test_*.py -v
python scripts/research/pretraining/validate_pretraining_backlog.py `
  scripts/research/pretraining/pretraining_backlog_v0.1.yaml `
  --smoke-config scripts/research/pretraining/smoke_config_v0.1.yaml
```

`smoke_config_v0.1.yaml` 只是待执行协议。只有获得用户授权并产生完整 provenance 后，
才能在 4090 把 `B0-SMOKE` 从 `blocked_external` 改为已执行状态。

4090 上的 Go Stanford 真实数据 pilot 使用 `go_stanford_adapter.py` 和
`audit_go_stanford_pilot.py`。adapter 直接产生以米为单位的 canonical batch，
不创建旧 loader 的 LMDB/index；audit 可全量解码图片并生成 manifest、split 和
processed-tree 哈希。它们不加载模型，也不执行 forward/backward 或训练。
`preflight_go_stanford_batches.py` 按冻结 smoke 配置构造确定性 5% train 子集，
只在 CPU 上迭代 canonical DataLoader，不实例化或调用模型。

经独立 `IMAGE-FORWARD` 授权后，`../models/tinynavbrain_image_policy.py` 已将单一共享、
随机初始化的 EfficientNet-B0 接到 H0/H1 image-to-waypoint API；它拒绝非空预训练
权重参数。`../models/preflight_tinynav_image_forward.py` 可在真实 canonical batch 上
执行 inference-only gate 并生成带哈希的 JSON/Markdown 报告。该门不创建 optimizer、
不执行 backward/训练，也不写 checkpoint；其延迟值不能作为训练或部署结论。

`train_step_runtime.py` 实现 H0/H1 action loss 的可执行 train step、确定性 batch
cursor，以及包含 model/optimizer/Python/NumPy/Torch CPU/CUDA RNG 的 exact-resume
checkpoint。`preflight_tinynav_train_step.py` 只用于单独授权的两步集成门；
`recover_tinynav_train_step_resume.py` 是零 backward、零 optimizer step 的恢复验证路径。
两步门与 200-step B0 是不同阶段，readiness 或 post-checkpoint forward loss 不得写成
收敛、训练可行性、模型质量或方法结果。
