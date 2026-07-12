# Policy 后端解耦、无训练探索与单卡 4090 小模型路线

## 当前结论

当前项目是“部分解耦”，不是完整解耦。`NoMaDInferenceModule` 已统一模型加载、图像预处理、
条件编码和 DDPM/DDIM/CFG/TTS 采样；高层 `Lite3HighLevelNoMaD`、中层 waypoint→速度控制和
低层平台接口也已经分层。但 `Lite3System` 原先直接构造具体 NoMaD 类并访问
`high_level.inference.pil_to_tensor`，旧 `deployment/src` 又保留另一套内联推理与 PD 逻辑。

本轮引入结构化 `NavigationPolicyBackend`、构造器注入和 `preprocess_frame` 方法，使状态机可以
替换高层 policy backend，而不依赖 NoMaD 内部对象。旧 deployment ROS 脚本尚未删除或重写；
在真机重新验证之前只能标记为 legacy，不能宣称两条部署链路完全一致。

## 研究依据与可行方向

- [NoMaD](https://arxiv.org/abs/2310.07896) 证明 goal-masked diffusion policy 可以统一目标导航与探索；
  本项目继续把它作为冻结 teacher/baseline，而不是把已有机制包装成新贡献。
- [Diffusion Policy](https://arxiv.org/abs/2303.04137) 强调 action chunk 与 receding-horizon control；
  因此后端研究应保留完整轨迹和候选分布，而不是只围绕单 waypoint 调参。
- [Consistency Policy](https://arxiv.org/abs/2405.07503) 和
  [ManiCM](https://arxiv.org/abs/2406.01586) 表明 consistency distillation 可显著减少扩散推理步数；
  这需要未来训练，当前只能进入训练计划，不能作为已实现结果。
- [FlowPolicy](https://arxiv.org/abs/2412.04987) 使用 consistency flow matching 加速动作生成；
  本项目将 flow head 设为必须与确定性 head 等预算比较的候选，而不是默认优胜者。
- [ViNT](https://arxiv.org/abs/2306.14846) 展示跨机器人导航数据和可替换目标编码器的价值；
  当前只有 GO Stanford 完整落地，因此 cross-dataset 结论继续锁定。
- [MiniVLN](https://arxiv.org/abs/2409.18800) 支持分阶段知识蒸馏的小模型路线；
  但其任务是 VLN，不能直接当作本项目视觉目标导航的性能证据。

## 本轮无训练实验

在纠正 GO Stanford 物理尺度后，使用 U10 v5 的 15 个配对 case 对
`candidate_diversity` 驱动的 adaptive-compute router 做后验全阈值回放：

- cheap DDIM-2：ADE 0.101775 m，sampler 9.661 ms；
- 路由到 DDIM-3 仅出现小幅、后验的精度/延迟交换；
- 路由到 DDPM-10 只有在大量 case 走慢路径时才接近 DDPM 精度；
- TTS fallback 在该小样本上被直接使用 DDIM-2 支配。

结论：不合并 adaptive router，不作正面方法 claim。合并的是后端接口、诊断工具和负结果，
因为它们可以阻止后续把后验阈值误当成有效策略。

## TinyNavBrain-ScaleAdaptive

未来训练候选由共享 EfficientNet-B0、物理尺度/时间/action-history token、确定性 action-chunk head
和小型 meter-space residual-flow head 构成。确定性 head 是 NFE=1 的强基线；flow head 只允许在
相同 encoder、数据、参数、optimizer steps、seed 与推理预算下比较 NFE 1/2/4。

创新假设是：显式物理尺度与动作历史能减少不同 embodiment/dataset 的尺度混淆，并让小模型
在受控计算预算下保留生成式修正能力。否证条件同样明确：

1. flow head 不超过确定性 head，则停止生成式 head；
2. scale token 收益不超过 seed 波动，则停止尺度泛化 claim；
3. 离线改善不能在不少于 20 个仿真 seed 的闭环指标复现，则不进入论文主结果；
4. 少于三个合法且完整处理的数据集时，不解锁 mixed/LODO 或 cross-dataset claim。

完整机器可读计划及静态验证器位于 `scripts/research/training_plans/`。其中显存数字是部分解析记账，
不是 4090 实测；正式训练前必须先运行 200-step smoke、记录峰值显存和吞吐，再确定 batch size。

## 本轮发现并纠正的 U10 错误

U10 v2 错把 RECON 的 `metric_waypoint_spacing=0.25 m` 用于 GO Stanford；冻结训练配置实际为
`0.12 m`。v2 已作废。U10 v5 从数据配置加载尺度并完成 75/75 有效记录；所有下游相关性、
复现包和论文证据必须绑定 v5。旧目录仅保留审计，不允许进入任何分析。
