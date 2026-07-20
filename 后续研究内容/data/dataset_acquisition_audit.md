# 多数据集获取与许可审计

核验日期：2026-07-10
机器可读注册表：[`dataset_registry.json`](dataset_registry.json)

## 结论

当前不能说项目已经具备多数据集训练条件。只有 `nomad_dataset/go_stanford` 是可直接被 `ViNT_Dataset` 使用的 processed dataset；`recon_datavis`、`tartan_drive` 等代码目录不能计为训练数据。

推荐获取顺序：

1. 保留 Go Stanford 作为本地 pipeline anchor，但传播时遵守 CC BY-NC-SA 3.0。
2. 获取 RECON 小样本/首个分片，验证 HDF5 -> ViNT trajectory 转换和 session-level split。
3. 获取 HuRoN 小样本，验证 rosbag topic、压缩图像、里程计、policy-version metadata。
4. 只有在前三步形成 3 个 processed datasets 后，才解锁 multi-dataset benchmark 和 TinyNavBrain 实现主线。
5. SCAND 在官方 Dataverse 许可人工确认前保持阻塞；TartanDrive 只作为后续极端尺度/domain-stress，不作为首批混合训练数据。

## 官方证据

| 数据集 | 官方入口 | 官方可见许可 | 格式/规模 | 当前判决 |
|---|---|---|---|---|
| GO Stanford | <https://svl.stanford.edu/projects/gonet/dataset/> | CC BY-NC-SA 3.0 | 校园机器人图像与运动信号；本地已转换 | 可用于学术研究，传播与商业用途受限 |
| RECON | <https://sites.google.com/view/recon-robot/dataset> | MIT，数据页明确 | HDF5，约 50 GB | Tier A，优先获取 |
| HuRoN/SACSoN | <https://sites.google.com/view/sacson-review/huron-dataset> | MIT，数据页明确 | ROS bag；75 h / 58 km | Tier A，第二获取 |
| SCAND | <https://www.cs.utexas.edu/~xiao/SCAND/SCAND.html> / DOI `10.18738/T8/0PRYRH` | 官方项目页未显示 | 138 trajectories / 8.7 h / 双平台 | 许可阻塞，不作默认下载 |
| TartanDrive 1.0 | <https://github.com/castacks/tartan_drive> | 仓库 MIT；payload 边界不清 | 多 ROS bag 压缩包，单包约 100 GB | Tier C，先做小样本可行性 |

## 为什么首批不是 TartanDrive / SCAND

SCAND 的研究价值很高：同一数据集包含轮式与 Spot、室内外和拥挤人群，适合检验跨 morphology/domain。但没有明确许可时，下载、二次处理和公开 manifest 都存在合规风险。许可未知不是“默认可用”。

TartanDrive 的价值主要是 off-road 和物理尺度 stress test。它来自高速 ATV，`metric_waypoint_spacing=0.72 m`，与 Go Stanford 的 `0.12 m`、HuRoN 的 `0.255 m` 差异显著。若直接混入训练，模型可能主要学到数据集/尺度身份，反而使所谓 cross-dataset gain 失去解释力。

## 获取前的硬门槛

- 先下载最小分片，不下载完整语料。
- 为原始文件保存 URL、访问日期、官方许可快照位置、SHA-256、字节数和原始层级。
- 转换后保留 `source_dataset`、`source_session`、`source_trajectory`、robot、camera、`dt`、meter scale 和 processor version。
- split 以 trajectory/session/building 为单位；禁止逐帧随机切分。
- 同一物理轨迹的不同 camera、裁剪、降采样或派生版本必须属于同一 leakage group。
- 未知许可、缺 pose/action、无法恢复 meter scale 或无法生成稳定 trajectory ID，任一项都可触发 no-go。

## 代码侧已确认风险

1. `train/vint_train/data/data_config.yaml` 重复声明 `carla_intvns`；YAML loader 会静默覆盖，说明当前注册表缺少 schema validation。
2. `ViNT_Dataset` 的构造检查允许 `randomized` / `randomized_temporal`，但 `__getitem__` 实际只实现 `temporal`。
3. `dataset_index` 来自 YAML key 排序；新增/删除数据集会改变整数 ID，缓存或模型若依赖该值可能发生漂移。
4. `train.py` 使用 `ConcatDataset`，默认采样概率随数据集大小变化；这不是 dataset-balanced sampling。
5. 全局 `action_stats` 与不同数据集的 meter spacing 并存；训练前必须同时报告 normalized action 和 physical-meter action 分布。

这些问题不要求现在修改训练逻辑，但必须由下一步 manifest/audit 工具显式检测。否则 benchmark 会在数据进入模型之前就失去可解释性。

2026-07-16 的本地只读盘点确认两个授权工作区内 RECON/HuRoN 为 0/2
materialized。后续获取前使用 `templates/recon_huron/` 的 receipt/checksum/license
snapshot 模板，并用 `scripts/research/data/preflight_recon_huron_raw.py` 生成只读 raw
inventory 和计划转换命令。该 dry-run 不调用转换器；任何下载或转换仍需另行授权。

2026-07-20 的获取轮已闭合 raw pilot：RECON 官方 archive 为
53,235,196,027 bytes，完整目录含 11,836 个 HDF5，仅保留已解码验证的
143,826-byte 最小成员；HuRoN 保留 11,044,659-byte bag，所需 fisheye
compressed image 与 odometry topic 均可读。两者的官方页快照、receipt、SHA-256、
raw 内容审计与 strict preflight 通过，合计 raw payload 53.246385 GB，低于
350 GB 授权上限。权威证据为
`results/research/data_acquisition/recon_huron_pilot_20260720/audit.{json,md}`。

该结果仍不满足 Gate A：没有执行 HDF5/bag 语义转换，也没有 processed
trajectory、dt/米制尺度验证、group-safe manifest/split 或处理器 receipt。

### 2026-07-20 CONVERSION-PILOT

后续授权的隔离转换已完成。RECON 先审计 archive 中全部不大于
979,778 bytes 的 1,711 个 HDF5，0 读取错误，最小满足 14-frame canonical
window 的成员为 934,600 bytes。它转换为 14 帧 / 1 window，中位步距
0.243780 m，与注册值 0.25 m 的相对差 2.49%。HuRoN bag 以 rosbag record
time 为权威时钟，4 Hz 同步 67 点并保留 66 帧完整前向段 / 53 windows；
中位步距 0.203050 m，与 0.255 m 注册值相差 20.37%。

两个 processed tree 的逐文件 SHA-256、RGB 解码、position/yaw finite、连续帧号、
canonical window、source session 和尺度门通过。联合 manifest 只有 2 行 / 2
leakage groups，两者均为 pilot-only train 分配，0 error / 0 warning；它不是评测
holdout。保守峰值磁盘上界 1.275230 GB / 10 GB。证据为
`results/research/data_conversion/recon_huron_pilot_20260720/audit.{json,md}`。

晋级仍阻塞：RECON HDF5 没有逐帧 timestamp，0.253781 s dt 是位移/线速度运动学
估计；RECON 和 HuRoN 的选定 artifact 都未编码 collection policy/version，不允许用默认
值补造。因此 conversion integrity 通过，但 `promotion_ready=false`，不解锁训练或评测。

## Gate A

只有同时满足以下条件，才把下一篇论文继续定位为 multi-dataset benchmark：

- 至少 3 个许可可用、processed、可追溯的数据集；
- 每个数据集都能恢复 RGB、位置/yaw、`dt` 与 meter scale；
- session/trajectory-level split 通过泄漏审计；
- proportional、balanced、temperature sampling 可以从同一 manifest 精确定义；
- 所有数据集都能进入同一 result schema，且不把 dataset-relative normalization 当成跨数据集绝对性能。

若只能得到 1–2 个数据集，立即降级为 corruption/scale diagnostic study，不再使用“multi-dataset benchmark”或“leave-one-dataset-out”作为核心贡献。
