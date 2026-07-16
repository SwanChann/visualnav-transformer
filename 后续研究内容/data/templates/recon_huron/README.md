# RECON / HuRoN acquisition evidence templates

这些文件只是模板，不是数据、许可或获取证据。不得把仓库 registry 中的许可陈述直接
复制成“已核验的 live snapshot”。每个真实 artifact 必须先由操作者保存官方页面快照，
再使用 `scripts/research/data/register_raw_artifact.py` 生成不可变 receipt。

建议的人工顺序：

1. 将官方下载页、许可/隐私条款和采集日期填写到
   `license_snapshot.template.md` 的副本，并保留页面原文或导出文件的 SHA-256。
2. 对官方下载 artifact 运行 `register_raw_artifact.py`；不要手工伪造 receipt。
3. 解包后，以 `raw_checksum_manifest.template.json` 为 schema 保存每个 HDF5/ROS bag
   的相对路径、字节数和 SHA-256。
4. 对本地 raw root 运行 `preflight_recon_huron_raw.py --checksum-mode sha256`。
   它只盘点并生成计划命令，不调用转换器。
5. 转换仍需新的逐轮授权，并应写入隔离的新 output root。

`raw_artifact_receipt.template.json` 只展示字段形状；权威生成器仍是
`register_raw_artifact.py`。所有 `<PLACEHOLDER>` 和 `_template_only` 字段都必须在真实
receipt 中消失。
