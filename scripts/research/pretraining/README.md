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
