#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Write expanded algorithm_design_and_experiments.tex"""

content = r"""% !TEX root = ../main.tex

\chapter{算法设计与实验}

\section{任务定义与研究思路}

本文关注的核心任务是：给定机器人当前视觉观察、短历史观察和目标图像或目标节点图像，生成一段局部 waypoint 轨迹，并将其作为中层控制器的参考输入。沿用 NoMaD 的基本思想\cite{nomad2023}，本文不直接预测单步速度命令，而是学习条件轨迹分布
\begin{equation}
p(\mathbf{A}_t \mid \mathcal{O}_t, I_g),
\end{equation}
其中 $\mathcal{O}_t$ 表示时刻 $t$ 的历史视觉观察，$I_g$ 表示目标图像或从 topomap 中筛选得到的局部目标节点，$\mathbf{A}_t$ 表示未来局部 waypoint 序列。

\subsection{问题的形式化建模}

从马尔可夫决策过程（Markov Decision Process, MDP）的视角来看，视觉导航可以被建模为一个元组 $\mathcal{M} = (\mathcal{S}, \mathcal{A}, T, R, \gamma)$，其中 $\mathcal{S}$ 表示状态空间（即视觉观察空间），$\mathcal{A}$ 表示动作空间，$T$ 表示转移概率，$R$ 表示奖励函数，$\gamma$ 表示折扣因子。在本文的框架中，策略 $\pi$ 不是直接在 MDP 上求解单步最优动作，而是学习一个条件轨迹分布 $p(\mathbf{A}_t \mid \mathcal{O}_t, I_g)$，这种分布式策略具有天然的多模态建模能力。

具体来说，条件轨迹分布与传统 MDP 策略之间的联系可通过以下关系理解：若将未来 $H$ 步动作序列视为一个联合决策变量，则条件轨迹分布相当于在扩展动作空间 $\mathcal{A}^H$ 上的策略。与逐步递推不同，这种联合建模能够在单次推理中捕获全局轨迹结构，避免了自回归策略中误差累积的问题。形式上：
\begin{equation}
p(\mathbf{A}_t \mid \mathcal{O}_t, I_g) = p(\mathbf{a}_1, \mathbf{a}_2, \ldots, \mathbf{a}_H \mid \mathcal{O}_t, I_g),
\end{equation}
其中每个 $\mathbf{a}_i$ 并非独立采样，而是通过扩散过程的联合去噪一次性生成。

\subsection{Waypoint 表示的优势分析}

本文选择 waypoint 序列作为策略输出形式，而非直接预测底层速度命令（如线速度 $v$ 和角速度 $\omega$），这一设计决策基于以下三点考量。

第一，平台无关性。Waypoint 表示为局部坐标系下的二维位移序列，不与任何特定的运动学模型绑定。无论下游执行平台是差分驱动的轮式机器人还是具有复杂运动学的足式机器人（如 Lite3），同一条 waypoint 轨迹都可以通过不同的中层控制器进行跟踪。这种解耦使得高层策略的训练数据可以跨平台复用。

第二，多模态表达能力。在面对 T 形路口或走廊分叉等场景时，合理的导航行为可能包含"左转"和"右转"两个互不相容的模态。若使用单步速度命令的均值回归策略，两个模态的平均结果可能是"直行撞墙"。而 waypoint 序列结合扩散模型的生成能力，能够保留这种多模态分布，通过采样不同轨迹来表达不同的决策意图。

第三，闭环控制的友好性。中层控制器（如 PID 或 MPC）可以直接将 waypoint 序列作为参考路径进行跟踪，这比将高层策略输出的速度命令直接下发到底层更加鲁棒，因为 waypoint 跟踪器可以根据实际执行情况进行局部修正。

围绕这一任务，本文在算法层的工作可概括为三部分。第一，基于项目代码准确复现 NoMaD 的训练与推理链路，明确视觉编码器、距离预测头与扩散策略头之间的关系。第二，围绕部署可用性对推理层进行优化，重点研究 DDIM 推理、CFG 引导增强和 TTS 推理时搜索。第三，围绕视觉表征能力设计视觉编码器升级实验，并与基线配置进行对比。需要强调的是，本章关注的是"哪些候选方法值得进入闭环验证"，因此所有结论都首先被限定在离线统计与离线代理指标层面。

\begin{figure}[htbp]
    \centering
    \fbox{\rule{0pt}{55mm}\rule{0.84\linewidth}{0pt}}
    \caption{图位占位：NoMaD 算法研究对象与数据流示意图（待补）}
    \label{fig:algo-overview-placeholder}
\end{figure}

\section{NoMaD 基础模型复现}

\subsection{输入输出形式}

根据当前项目配置，NoMaD 的输入由历史视觉观察与目标图像构成。实验中典型设置为 \texttt{context\_size=3}，即模型同时接收若干历史观察帧与一张目标图像。输出则为固定长度的局部轨迹序列，每个 waypoint 用局部坐标系下的二维位移表示：
\begin{equation}
\mathbf{a}_i = (\Delta x_i, \Delta y_i), \qquad i=1,\ldots,H.
\end{equation}
整段轨迹写为
\begin{equation}
\mathbf{A} = [\mathbf{a}_1,\mathbf{a}_2,\ldots,\mathbf{a}_H].
\end{equation}

这一表示的优点在于，它比直接输出底层速度命令更适合作为平台无关的高层接口，也比全局路径更适合闭环局部控制。对于 Lite3 这样的足式平台，高层输出局部轨迹、中层完成控制映射的方式尤为关键。

\subsection{视觉编码器架构}

NoMaD 默认采用 EfficientNet-B0 作为视觉编码器\cite{efficientnet2019}。EfficientNet-B0 是 EfficientNet 系列的基础模型，通过复合缩放（compound scaling）策略在深度、宽度和分辨率三个维度上进行统一缩放，在 ImageNet 上以约 5.3M 参数量达到了优异的精度-效率平衡。其核心构建块为 MBConv（Mobile Inverted Bottleneck Convolution），采用深度可分离卷积和 Squeeze-and-Excitation 注意力模块。

在 NoMaD 中，EfficientNet-B0 接收 $96 \times 96$ 的 RGB 图像作为输入，经过特征提取后输出一个固定维度的特征向量。该特征向量随后被投影为标准化的 token 表示，用于后续的条件拼接。具体而言，每帧观察图像经过编码器后得到特征 $\mathbf{z}_i \in \mathbb{R}^d$，其中 $d$ 为 token 维度。

\subsection{Token 构建与条件拼接}

NoMaD 采用基于 token 序列的条件构建方式。对于 \texttt{context\_size=3} 的配置，模型将当前帧及前两帧共 3 帧历史观察分别通过视觉编码器，得到观察 token 序列 $[\mathbf{z}_{t-2}, \mathbf{z}_{t-1}, \mathbf{z}_t]$。同时，目标图像 $I_g$ 也通过同一编码器得到目标 token $\mathbf{z}_g$。

最终的条件向量通过拼接所有 token 构成：
\begin{equation}
\mathbf{c} = \text{Concat}(\mathbf{z}_{t-2}, \mathbf{z}_{t-1}, \mathbf{z}_t, \mathbf{z}_g).
\end{equation}
这一拼接向量同时作为距离预测头和扩散策略头的输入条件。值得注意的是，所有观察帧和目标帧共享同一个视觉编码器实例，权重完全共享。这种设计简化了训练流程，但也意味着编码器需要同时学习"描述当前场景"和"表征目标位置"两种功能。

\subsection{模型结构与训练目标}

NoMaD 由视觉编码器、距离预测头和扩散策略头三部分组成\cite{nomad2023}。视觉编码器负责将历史观察与目标条件编码为统一特征；距离预测头输出当前观察到目标的相对距离估计；扩散策略头在该条件上生成局部轨迹样本。项目代码显示，距离预测头与扩散策略头共享同一视觉条件空间，因此视觉编码器的变化会同时影响二者。

训练时，总损失由距离回归损失与扩散噪声预测损失组成，可写为
\begin{equation}
\mathcal{L} = \alpha \mathcal{L}_{dist} + (1-\alpha)\mathcal{L}_{diff}.
\end{equation}
其中，$\mathcal{L}_{dist}$ 约束候选目标排序能力，$\mathcal{L}_{diff}$ 约束局部轨迹生成质量。

关于损失权重 $\alpha$ 的选择策略，需要注意以下两点。首先，$\mathcal{L}_{dist}$ 通常采用 $L_1$ 或 $L_2$ 回归损失，其数值尺度与扩散噪声预测损失 $\mathcal{L}_{diff}$ 存在差异，因此 $\alpha$ 实际上同时承担了任务权重和尺度归一化的双重角色。其次，距离预测头的梯度会回传至视觉编码器，这意味着 $\alpha$ 的选择会影响编码器学到的特征偏向——较大的 $\alpha$ 倾向于让编码器学习距离敏感的特征，较小的 $\alpha$ 则让编码器更侧重于轨迹生成所需的空间结构特征。

本文对训练代码的检查表明，距离预测头与视觉编码器是直接耦合的，因此更换视觉编码器后，距离预测分支必须同步重新训练，否则距离头与轨迹头将共享不一致的条件分布。

\subsection{推理链路的统一封装}

在推理阶段，系统先编码观察与目标，再在需要时进行候选节点距离评估，最后通过扩散采样生成轨迹。为保证后续实验和系统章节可复用，本文将 \texttt{policy\_config}、\texttt{policy\_checkpoint}、\texttt{scheduler\_kind}、\texttt{ddim\_steps}、\texttt{cfg\_weight}、\texttt{num\_samples} 等参数收口到统一推理模块中。这样做的意义在于：推理优化被限制在高层策略层，不需要同时改动状态机、导航主机和平台接口。

\section{数据集与实验设备}

\subsection{GoStanford 数据集}

本文所有离线实验均基于 GoStanford（go\_stanford）数据集进行。该数据集来源于斯坦福大学校园环境中通过自主采集机器人记录的视觉导航轨迹，是 NoMaD 项目使用的标准训练与评估数据来源之一\cite{gnm2022,vint2023,nomad2023}。数据集中的每条轨迹包含按时间顺序排列的第一人称 RGB 图像帧和对应的位姿信息，其组织形式为 \texttt{traj\_data.pkl} 文件加图像目录。

本文使用的本地数据集包含来自 GoStanford 的多条轨迹（如 \texttt{no10vc\_*} 系列），涵盖走廊、拐角、开阔区域等多种场景。在离线实验中，每次推理从轨迹中提取连续观察帧与目标帧，构建 24 个标准化测试 case，用于 DDIM、CFG、编码器等所有实验的统一评估。

\subsection{数据预处理流程}

从原始轨迹到模型输入之间存在一个标准化的数据预处理流程。首先，RGB 图像从原始分辨率（通常为 $640 \times 480$）缩放至模型要求的输入尺寸（$96 \times 96$），缩放过程使用双线性插值以保持图像质量。其次，像素值从 $[0, 255]$ 归一化至 $[0, 1]$ 范围。最后，根据 \texttt{context\_size} 参数从轨迹中提取连续帧作为历史观察，并从同一轨迹的未来位置选取目标帧。

测试 case 的构建遵循以下原则：当前观察与目标帧之间的时间间隔覆盖短距离（5--10 帧）和中距离（20--50 帧）两种范围，以确保评估涵盖不同难度级别。每个 case 包含完整的历史观察序列、目标图像和对应的真实轨迹标签，使得所有实验在完全相同的数据条件下进行比较。

值得说明的是，24 个测试 case 的数量虽然有限，但其设计覆盖了直行、转弯、宽阔区域和狭窄走廊等多种典型场景类型。在离线统计实验中，这一规模足以反映不同配置之间的趋势性差异，同时保持单次完整实验的运行时间在可接受范围内。对于需要更高统计置信度的结论，本文通过报告 CI95 来量化均值估计的不确定性。

\subsection{感知输入与设备说明}

本文系统采用纯 RGB 单目视觉作为唯一感知输入，不依赖深度传感器、激光雷达或惯性测量单元。这一设计选择与 NoMaD 原始框架一致，即高层导航策略仅以图像观察和目标图像为输入，不假设额外传感模态。

在离线实验阶段，训练与统计实验运行在配备 NVIDIA RTX 4090 的工作站上；在 MuJoCo 仿真阶段，相机图像由 MuJoCo 渲染器以第一人称视角生成；在真机部署阶段，使用 USB 单目相机接入 Jetson Orin，输出分辨率为 $640\times480$，推理前自动缩放至模型要求的输入尺寸（如 $96\times96$ 或 $98\times98$）。

\section{基线实验}

为给后续所有优化实验建立统一参考，本文首先复现了原始 NoMaD 离线推理流程。已有结果表明，原始模型参数量约为 19,049,675，DDPM-10 配置下单次离线推理时间约为 0.1250\,s，对应约 8.0\,Hz 的单次离线推理频率。其主要作用不是证明"原始模型足够好"，而是作为比较基线：后续 DDIM、CFG、TTS 和视觉编码器实验都需要回到这一基线来判断收益与代价。

关于参数量的构成，19M 参数中视觉编码器（EfficientNet-B0）约占 5.3M，距离预测头约占 0.5M，扩散策略头（包括 U-Net 噪声预测网络）约占 13M。这一分布说明扩散策略头是参数量的主要来源，也解释了为什么更换视觉编码器对总参数量的影响相对有限。

关于推理频率的部署含义，8.0\,Hz 的离线推理频率意味着在 RTX 4090 上每 125\,ms 可以完成一次轨迹生成。对于典型的机器人导航场景，控制频率通常要求在 5--20\,Hz 之间，因此 DDPM-10 的基线频率在高端 GPU 上勉强达到实用门槛。但在边缘设备（如 Jetson Orin）上，由于算力约为 RTX 4090 的 1/5 至 1/10，实际推理频率可能降至 1--2\,Hz，这远低于部署需求，也是后续 DDIM 加速实验的直接动机。

此外， \\times 8$ 的轨迹采样规格表示每次推理生成 8 条候选轨迹，每条轨迹包含 8 个 waypoint。候选轨迹的多样性为后续 TTS 搜索提供了选择空间，而 8 个 waypoint 的长度在 NoMaD 的默认配置下对应约 2--3 秒的未来规划窗口，这一时间尺度既足以覆盖局部导航决策（如是否转弯），又不至于因规划太远而积累过大误差。

\begin{table}[htbp]
    \centering
    \caption{NoMaD 基线离线推理结果}
    \label{tab:baseline-inference}
    \begin{tabular}{lll}
        \toprule
        指标 & 含义 & 数值 \\
        \midrule
        模型参数量 & 可学习参数总量 & 19,049,675 \\
        DDPM 推理时间 & 单次完整离线推理耗时 & 0.1250\,s \\
        推理频率 & 每秒可执行离线推理次数 & 8.0\,Hz \\
        轨迹采样规格 & 候选轨迹数 $\times$ waypoint 数 & $8\times8$ \\
        距离预测输出 & 候选目标相对距离估计均值 & 11.6589 \\
        \bottomrule
    \end{tabular}
\end{table}

\section{实验指标与统计原则}

为了避免仅凭单次样本或单一指标做结论，本文采用多指标联合分析。主要指标包括：Latency，用于衡量推理开销；CI95，用于表示均值的 95\% 置信区间；MSE vs DDPM-10，用于衡量与基线采样分布的偏移；Forward Progress，用于刻画轨迹在主前进方向上的推进程度；Lateral Abs，用于衡量横向偏摆幅度；Smoothness，用于衡量轨迹变化的平滑性；Dist Pred Mean，用于观察候选目标距离评估是否随条件空间变化发生系统偏移。

\subsection{指标的形式化定义}

为确保实验可复现性，本文给出各指标的形式化定义。

\textbf{Latency} 定义为单次推理的端到端耗时，包含编码、去噪和后处理，单位为毫秒：
\begin{equation}
\text{Latency} = t_{\text{end}} - t_{\text{start}}.
\end{equation}

\textbf{CI95} 表示均值的 95\% 置信区间半径，基于 $t$ 分布计算：
\begin{equation}
\text{CI95} = t_{0.975, n-1} \cdot \frac{s}{\sqrt{n}},
\end{equation}
其中 $s$ 为样本标准差，$n$ 为测试 case 数量。

\textbf{MSE vs DDPM-10} 衡量某配置生成轨迹均值与基线 DDPM-10 生成轨迹均值之间的均方误差：
\begin{equation}
\text{MSE} = \frac{1}{H} \sum_{i=1}^{H} \|\bar{\mathbf{a}}_i^{\text{test}} - \bar{\mathbf{a}}_i^{\text{base}}\|^2,
\end{equation}
其中 $\bar{\mathbf{a}}_i$ 表示第 $i$ 个 waypoint 在所有候选轨迹上的均值。

\textbf{Forward Progress} 定义为轨迹在主前进方向（$x$ 轴）上的累计位移：
\begin{equation}
\text{FP} = \sum_{i=1}^{H} \Delta x_i.
\end{equation}

\textbf{Lateral Abs} 定义为轨迹在横向（$y$ 轴）上的累计绝对位移：
\begin{equation}
\text{LA} = \sum_{i=1}^{H} |\Delta y_i|.
\end{equation}

\textbf{Smoothness} 衡量相邻 waypoint 之间方向变化的一致性，定义为相邻位移向量差的范数均值：
\begin{equation}
\text{Smooth} = \frac{1}{H-1} \sum_{i=1}^{H-1} \|\mathbf{a}_{i+1} - \mathbf{a}_i\|.
\end{equation}

\subsection{统计显著性与多指标分析原则}

本文采用多指标联合分析而非单一指标排序，原因在于导航任务的优化目标本质上是多目标的。仅优化前进性可能导致激进轨迹，仅优化平滑性可能导致过于保守的运动。通过同时报告 Forward Progress、Lateral Abs、Smoothness 和 Latency，读者可以判断某个配置是否在所有维度上都可接受，而不是在某一维度上表现突出但在其他维度上严重退化。

其中，CI95 并不是新的性能指标，而是均值稳定性的统计描述。若某个配置平均值较优但 CI95 很宽，说明其结果在不同 case 间波动较大。Forward Progress 也不能单独解释为"越大越好"，因为过大的前进量可能是激进采样带来的分布偏移，必须结合 MSE、Lateral Abs 和 Smoothness 一起分析。

\section{DDIM 推理加速实验}

\subsection{理论动机}

DDIM 的核心思想是将扩散模型的随机微分方程（SDE）采样过程转换为确定性常微分方程（ODE）求解过程\cite{ddim2020}。在标准 DDPM 中，反向去噪过程由以下随机递推描述：
\begin{equation}
\mathbf{x}_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left(\mathbf{x}_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}} \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t)\right) + \sigma_t \mathbf{z},
\end{equation}
其中 $\mathbf{z} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$ 引入了随机性。DDIM 的关键洞察是：当令随机项 $\sigma_t = 0$ 时，上述过程退化为确定性的 ODE 积分：
\begin{equation}
\mathbf{x}_{t-1} = \sqrt{\bar{\alpha}_{t-1}} \cdot \hat{\mathbf{x}}_0(\mathbf{x}_t, t) + \sqrt{1-\bar{\alpha}_{t-1}} \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t),
\end{equation}
其中 $\hat{\mathbf{x}}_0$ 是基于当前噪声预测的干净样本估计。

由于 ODE 的解路径是确定性的且连续的，可以在时间轴上使用更大的步长进行积分而不引入额外随机噪声的累积。这就是 DDIM 能够用更少步数完成去噪的数学基础——本质上是用粗粒度的数值积分近似原始的细粒度 ODE 轨迹。步数越少，积分越粗糙，但对于 smooth 的 ODE 解路径，适度的步数减少不会显著改变最终结果。

\subsection{实验设计}

如第二章所述，DDIM 的作用是在不重新训练模型的情况下减少扩散去噪步数，从而降低推理时延\cite{ddim2020}。本文在统一脚本下比较 \texttt{ddpm\_10}、\texttt{ddim\_10}、\texttt{ddim\_5}、\texttt{ddim\_3}、\texttt{ddim\_2} 和 \texttt{ddim\_1} 六种配置，测试集共包含 24 个 case。该实验的目标是回答两个问题：第一，步数减少后推理能快多少；第二，轨迹分布是否仍保持在可接受范围。

\subsection{实验结果}

\begin{table}[htbp]
    \centering
    \caption{DDIM 推理加速实验结果}
    \label{tab:ddim-results}
    \begin{tabular}{lccccc}
        \toprule
        配置 & Latency/ms & CI95/ms & Speedup & MSE vs DDPM-10 & Forward Progress \\
        \midrule
        ddpm\_10 & 39.9790 & 1.3748 & 1.00$\times$ & 0.0000 & 6.4998 \\
        ddim\_10 & 38.0260 & 0.6034 & 1.05$\times$ & 0.0062 & 6.4686 \\
        ddim\_5 & 19.2581 & 0.4319 & 2.08$\times$ & 0.0048 & 6.5257 \\
        ddim\_3 & 11.6579 & 0.3438 & 3.43$\times$ & 0.0124 & 6.6224 \\
        ddim\_2 & 8.2588 & 0.2779 & 4.84$\times$ & 0.0210 & 6.6471 \\
        ddim\_1 & 4.5092 & 0.1427 & 8.87$\times$ & 3.5168 & 9.2789 \\
        \bottomrule
    \end{tabular}
\end{table}

\subsection{结果分析}

结果表明，DDIM 的确能显著降低推理时延，其中 \texttt{ddim\_2} 已接近 5 倍加速。更关键的是，\texttt{ddim\_2} 在 MSE 与 Forward Progress 上仍保持相对平衡，说明它不是以严重破坏轨迹分布为代价换取速度。相比之下，\texttt{ddim\_1} 虽然最快，但 MSE 大幅升高，说明轨迹已明显偏离基线分布，不适合作为稳定部署方案。

从逐配置的角度进一步分析：\\texttt{ddim\\_10} 与 \\texttt{ddpm\\_10} 的 MSE 仅为 0.0062，说明在相同步数下 DDIM 的确定性采样与 DDPM 的随机采样生成的轨迹均值几乎一致；\\texttt{ddim\\_5} 在步数减半的情况下 MSE 反而略低于 \\texttt{ddim\\_10}（0.0048 vs 0.0062），这可能源于采样噪声的统计波动；\\texttt{ddim\\_3} 的 MSE 上升至 0.0124，开始出现可观测的分布偏移但仍在可控范围内；\\texttt{ddim\\_2} 的 MSE 为 0.0210，虽然进一步升高但 Forward Progress 仅从 6.50 变为 6.65，偏移幅度有限。

从速度-质量 Pareto 前沿的角度分析，\texttt{ddim\_2} 到 \texttt{ddim\_5} 构成了一段有效的 Pareto 前沿：在这个范围内，每减少一步都能带来显著的时延收益，而 MSE 增长仍在可控范围内。\texttt{ddim\_1} 则明显偏离了这一前沿，属于"过度压缩"区域。这可以从 ODE 求解的角度解释：当仅用一步积分从纯噪声直接跳到干净样本时，中间的非线性变换被完全忽略，相当于用线性近似替代了整个去噪轨迹，而扩散模型的去噪路径通常具有显著的非线性特征，因此单步近似会引入不可忽略的截断误差。

因此，本文将 \texttt{ddim\_2} 视为后续闭环验证中最值得优先考虑的推理配置。它回答了一个重要问题：扩散策略并不必然意味着"质量高但太慢"，通过合理采样调度，NoMaD 可以进入更接近真实部署的时延区间。

\begin{figure}[htbp]
    \centering
    \fbox{\rule{0pt}{52mm}\rule{0.84\linewidth}{0pt}}
    \caption{图位占位：DDIM 加速实验结果图（待补）}
    \label{fig:algo-ddim-placeholder}
\end{figure}

\section{CFG 引导增强实验}

\subsection{理论分析}

Classifier-Free Guidance（CFG）的核心思想是在推理时通过线性外推来增强条件信号对生成结果的影响\cite{cfg2022}。具体来说，CFG 修改了扩散模型的噪声预测（score function）为：
\begin{equation}
\tilde{\boldsymbol{\epsilon}}_\theta(\mathbf{x}_t, t, \mathbf{c}) = (1+w) \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \mathbf{c}) - w \cdot \boldsymbol{\epsilon}_\theta(\mathbf{x}_t, t, \varnothing),
\end{equation}
其中 $\mathbf{c}$ 为条件向量（包含观察和目标信息），$\varnothing$ 表示空条件（训练时通过随机 dropout 条件实现），$w$ 为引导权重。

当 $w=0$ 时，退化为标准的条件采样；当 $w>0$ 时，生成结果在条件方向上被"放大"，使轨迹更强烈地朝向目标方向偏移；当 $w<0$ 时，则产生反向引导效果。从概率密度的角度，CFG 实质上是在对数概率空间中将条件分布与无条件分布的差异进行缩放：
\begin{equation}
\log \tilde{p}(\mathbf{x} \mid \mathbf{c}) \propto (1+w) \log p(\mathbf{x} \mid \mathbf{c}) - w \log p(\mathbf{x}).
\end{equation}
这意味着 $w$ 越大，生成分布越集中于条件分布的高概率区域，同时远离无条件先验的模态。

\subsection{实验设计}

CFG 的目标是在推理阶段增强轨迹对目标条件的响应\cite{cfg2022}。本文选取 $w\in\{-1.0,0.0,0.5,1.0,2.0,4.0\}$ 六组配置进行比较。实验关注的问题不是"引导越强越好"，而是"多大引导能够在目标牵引与分布稳定性之间取得平衡"。

\subsection{实验结果与分析}

已有结果表明，适度的 CFG 能提高轨迹对目标方向的响应，但代价是更高的推理时延和潜在的分布偏移。从表~\ref{tab:cfg-results} 可以看到，$w=0.0$ 时推理时延约为 42.65\,ms，而 $w \geq 0.5$ 后时延跃升至约 78--80\,ms，这是因为 CFG 需要同时执行条件分支与无条件分支的噪声预测。

时延翻倍的现象可以精确解释：CFG 的每一步去噪都需要对噪声预测网络进行两次前向传播——一次使用完整条件 $\mathbf{c}$，一次使用空条件 $\varnothing$——然后将两次预测结果进行线性组合。因此，CFG 的计算成本近似为非 CFG 推理的 2 倍。实验中 $w=0.0$ 到 $w=0.5$ 的时延跳变（42.65\,ms $\to$ 78.27\,ms）正好反映了这一理论预期。

在轨迹质量方面，$w=1.0$ 的 Forward Progress 最高（8.573），Lateral Abs 仍保持适度（2.901），说明该配置在前进性和稳定性之间表现较为均衡。而 $w=4.0$ 时 Shift 高达 0.5133，Lateral Abs 上升至 3.089，表明过强的引导会导致分布偏移与横向偏摆。

关于最优引导尺度的选择，本文认为应遵循以下准则：（1）引导强度不应使 Shift 指标超过 0.2，以避免显著的分布偏移；（2）在满足（1）的前提下，优先选择 Forward Progress 较高的配置；（3）必须考虑时延代价——如果 CFG 带来的质量提升不能补偿 2 倍时延的代价，则不建议开启。因此，本文将 CFG 定位为可调部署旋钮，而非必须开启的固定默认项。

\begin{table}[htbp]
    \centering
    \caption{CFG 引导实验结果占位}
    \label{tab:cfg-results}
    \begin{tabular}{lcccc}
        \toprule
        配置 & Latency/ms & Shift vs $w=0$ & Forward Progress & Lateral Abs \\
        \midrule
        $w=-1.0$ & 42.28 & 0.0278 & 8.500 & 2.876 \\
        $w=0.0$ & 42.65 & 0.0000 & 8.523 & 2.886 \\
        $w=0.5$ & 78.27 & 0.0160 & 8.515 & 2.875 \\
        $w=1.0$ & 79.75 & 0.0476 & 8.573 & 2.901 \\
        $w=2.0$ & 78.81 & 0.1438 & 8.527 & 2.952 \\
        $w=4.0$ & 78.92 & 0.5133 & 8.483 & 3.089 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \fbox{\rule{0pt}{52mm}\rule{0.84\linewidth}{0pt}}
    \caption{图位占位：CFG 引导实验结果图（待补）}
    \label{fig:algo-cfg-placeholder}
\end{figure}

\section{TTS 推理时搜索实验}

\subsection{实验设计}

TTS 的作用是把额外预算用于"候选生成与选择"，而非用于修改训练过程\cite{ttsdiffusion2025,ttssearch2025}。在本文中，TTS 被设计为推理层的外层搜索组件：先用当前采样配置生成多条候选轨迹，再根据目标一致性、前进性和平滑性综合打分，最终选择一条更适合执行的轨迹。实验重点比较不同预算下的轨迹质量收益，以及它与 DDIM、CFG 的组合关系。

\subsection{评分函数设计}

TTS 的核心在于评分函数的设计。本文采用加权多目标评分函数：
\begin{equation}
\text{Score}(\mathbf{A}) = \beta_1 \cdot \text{FP}(\mathbf{A}) - \beta_2 \cdot \text{LA}(\mathbf{A}) - \beta_3 \cdot \text{Smooth}(\mathbf{A}),
\end{equation}
其中 $\text{FP}$ 为前进性，$\text{LA}$ 为横向绝对偏摆，$\text{Smooth}$ 为平滑性代价，$\beta_1, \beta_2, \beta_3$ 为权重系数。

这一评分函数的设计逻辑与 Beam Search 具有类似的思想：在候选集合中根据启发式评分选取最优候选。区别在于，Beam Search 通常用于自回归生成过程中的逐步筛选，而 TTS 是在完整轨迹生成完成后进行全局评价。这种"生成后筛选"的策略避免了修改采样过程本身，保持了扩散模型的生成质量。

\subsection{计算预算分配策略}

TTS 的计算开销与候选轨迹数（budget）成正比。设单次采样耗时为 $\tau$，则 TTS-$K$ 的总耗时约为 $K \cdot \tau + \tau_{\text{score}}$，其中 $\tau_{\text{score}}$ 为评分与选择的开销（通常可忽略）。因此，DDIM 加速节省的时间预算可以部分重新分配给 TTS，形成"用更快的单次采样换取更多候选"的策略。例如，DDIM-2 的单次耗时约 8\,ms，则 TTS-8 的总耗时约 64\,ms，仍低于原始 DDPM-10 的 40\,ms 单次推理。

\subsection{实验组织方式}

为了保证结论具有统计意义，本文将 TTS 实验组织为三层：单因素实验，用于比较是否开启 TTS 以及不同 budget；两两组合实验，用于比较 DDIM+TTS、CFG+TTS 的相互作用；三因素联合实验，用于比较 DDIM、CFG、TTS 同时开启时是否存在部署上更优的折中组合。

\subsection{结果占位与预期结论}

TTS 实验的核心问题不是"是否绝对提升所有指标"，而是"是否值得把 DDIM 节省下来的预算重新分配给候选搜索"。若实验表明少步 DDIM 结合中等预算 TTS 能在总延迟可接受的前提下显著改善目标一致性与轨迹稳定性，则说明三者具有互补性；若实验表明 CFG 与 TTS 同时增强会引入过强偏置，则说明二者需要更精细的配平。

\begin{table}[htbp]
    \centering
    \caption{TTS 推理时搜索实验结果占位}
    \label{tab:tts-results}
    \begin{tabular}{lcccc}
        \toprule
        配置 & Latency/ms & Forward Progress & Smoothness & 综合评分 \\
        \midrule
        No TTS & 待补 & 待补 & 待补 & 待补 \\
        TTS-8 & 待补 & 待补 & 待补 & 待补 \\
        TTS-16 & 待补 & 待补 & 待补 & 待补 \\
        TTS-32 & 待补 & 待补 & 待补 & 待补 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \fbox{\rule{0pt}{52mm}\rule{0.84\linewidth}{0pt}}
    \caption{图位占位：TTS 联合实验结果图（待补）}
    \label{fig:algo-tts-placeholder}
\end{figure}

\section{视觉编码器升级实验}

\subsection{实验目标与公平性问题}

视觉编码器升级实验的目标有两个：一是比较不同视觉编码器在统一框架下的潜力，二是为最终部署版训练筛选候选编码器。基于这一目的，本文将视觉编码器实验分为两个阶段：第一阶段使用统一 backbone-suite 框架进行公平筛选；第二阶段再对筛选出的候选做面向部署的最终训练。这样的两阶段设计能够同时满足"对比公平"和"训练结果可用于最终部署"两个需求。

\subsection{DINOv2 架构特点与差异分析}

DINOv2 是基于 Vision Transformer（ViT）架构的自监督预训练模型\cite{dinov2}。与 EfficientNet-B0 基于卷积的归纳偏置不同，DINOv2 通过将图像分割为固定大小的 patch（通常为 $14 \times 14$ 像素），然后将每个 patch 展平为 token 输入 Transformer 编码器。DINOv2-Small 的隐藏维度为 384，包含 12 层 Transformer block，总参数量约 22M。

这种架构差异带来了以下特点：首先，ViT 的全局自注意力机制使 DINOv2 能够在浅层就建立全局空间关系，而 CNN 需要通过多层堆叠逐步扩大感受野。对于导航任务中远距离目标的方向判断，全局注意力可能提供更好的特征支持。其次，DINOv2 的自监督预训练使用了大规模无标注数据，学到的特征在多种下游任务上表现出强泛化性。

\subsection{特征维度对齐}

在统一替换框架中，不同编码器的输出维度可能不一致。EfficientNet-B0 输出 1280 维特征，而 DINOv2-Small 输出 384 维特征。为实现公平对比，本文在编码器输出后添加一个线性投影层，将不同维度的特征统一映射到相同的 token 维度 $d$：
\begin{equation}
\mathbf{z} = W_{\text{proj}} \cdot \mathbf{f}_{\text{enc}} + \mathbf{b}_{\text{proj}},
\end{equation}
其中 $\mathbf{f}_{\text{enc}}$ 为编码器原始输出，$W_{\text{proj}} \in \mathbb{R}^{d \times d_{\text{enc}}}$ 为投影矩阵。这一投影层在训练过程中与其他模块联合优化。

\subsection{两阶段实验设计的理论依据}

采用两阶段设计而非直接端到端训练各编码器，基于以下考量。第一阶段（统一 backbone-suite 筛选）的目标是在控制变量的条件下比较编码器的表征潜力，因此采用统一的训练超参数、相同的冻结策略和相同的训练轮数。虽然这可能不是每个编码器的最优配置，但保证了对比的公平性。第二阶段（面向部署的最终训练）的目标是释放候选编码器的全部潜力，因此允许针对每个编码器调整学习率、冻结层数和训练轮数等超参数。

这种两阶段设计的代价是总训练时间的增加，但收益是：第一阶段可以快速排除明显不适合的候选（如训练不收敛或指标严重劣化的编码器），从而减少第二阶段的训练规模。

\subsection{实验配置}

当前第一阶段已包含 EfficientNet-B0 baseline、backbone-suite 版 EfficientNet-B0、公平替换框架下的 DINOv2-Small、ConvNeXt-Tiny 和 ResNet-50。统一配置中，主要训练批大小为 32，训练轮数为 100。这样做的目标不是立即得出"谁绝对最好"，而是先判断哪些视觉编码器值得继续投入。

\subsection{已有结果与分析}

已有离线结果表明，DINOv2-Small 在部分单步代理指标上优于原始 EfficientNet-B0，说明更强视觉表征对局部条件建模具有潜力。但在闭环层面，原始 baseline 仍然表现最稳，ConvNeXt-Tiny 具有继续优化空间，而 DINOv2-Small 与 ResNet-50 目前尚不能写成"已经成功升级"的最终部署结论。其原因更可能来自当前统一替换框架与原始 NoMaD 结构之间的归纳偏置差异、冻结策略偏保守、以及距离预测头与编码器耦合导致的适配不足，而不能简单归因于"大模型编码器更差"。

从定量数据来看，DINOv2-Small 在 Diversity 指标上达到 0.5417，远高于 EfficientNet-B0 baseline 的 0.2609，这说明 ViT 架构的全局注意力机制确实为轨迹采样带来了更丰富的多模态表达。Forward Progress 从 7.353 提升至 8.777，表明 DINOv2 的特征更有利于生成前进性强的轨迹。然而，Smoothness 的差异较小（0.1215 vs 0.1276），说明轨迹平滑性主要由扩散策略头决定，而非视觉编码器。

因此，本文对视觉编码器实验的表述采用更谨慎的口径：第一阶段工作已经建立了统一比较框架，并识别出 DINOv2-Small 等候选方案；但"最终部署最优编码器"的结论仍需第二阶段面向部署的训练与闭环验证完成后才能正式写出。

\begin{table}[htbp]
    \centering
    \caption{视觉编码器升级实验结果占位}
    \label{tab:encoder-results}
    \begin{tabular}{lcccc}
        \toprule
        编码器 & Latency/ms & Diversity & Forward Progress & Smoothness \\
        \midrule
        EfficientNet-B0 (baseline) & 47.93 & 0.2609 & 7.353 & 0.1276 \\
        DINOv2-Small & 42.97 & 0.5417 & 8.777 & 0.1215 \\
        \bottomrule
    \end{tabular}
\end{table}

\begin{figure}[htbp]
    \centering
    \fbox{\rule{0pt}{52mm}\rule{0.84\linewidth}{0pt}}
    \caption{图位占位：视觉编码器升级实验结果图（待补）}
    \label{fig:algo-encoder-placeholder}
\end{figure}

\section{联合结论与进入闭环的候选方案}

综合本章离线实验，表~\ref{tab:joint-ddim-cfg} 进一步给出了 DDIM 与 CFG 联合配置的前五名结果。从中可以看出，\texttt{ddim2\_w0.0} 在时延、前进性、平滑性和多样性的综合评分中排名第一，说明在不额外开启 CFG 的情况下，DDIM-2 本身已经具备较好的综合表现。

\begin{table}[htbp]
    \centering
    \caption{DDIM $\times$ CFG 联合配置 Top-5 结果}
    \label{tab:joint-ddim-cfg}
    \begin{tabular}{clccccc}
        \toprule
        排名 & 配置 & Latency/ms & Forward Progress & Smoothness & Diversity & 综合评分 \\
        \midrule
        1 & ddim2\_w0.0 & 9.99 & 8.089 & 0.1237 & 0.3897 & 5.0919 \\
        2 & ddim3\_w0.0 & 14.60 & 8.076 & 0.1075 & 0.3208 & 4.9942 \\
        3 & ddim5\_w0.0 & 24.63 & 8.032 & 0.1043 & 0.2906 & 4.7481 \\
        4 & ddim3\_w1.0 & 26.62 & 8.109 & 0.1096 & 0.3070 & 4.7227 \\
        5 & ddim3\_w0.5 & 27.04 & 8.093 & 0.1083 & 0.3130 & 4.7174 \\
        \bottomrule
    \end{tabular}
\end{table}

基于上述结果，可得到三点阶段性结论。第一，在不改动训练结果的前提下，\texttt{ddim\_2} 是当前最值得进入闭环验证的加速配置。第二，CFG 的有效范围有限，更适合作为可调参数而不是默认固定设置。第三，视觉编码器升级工作已经完成第一阶段筛选框架搭建，但最终部署结论必须通过第二阶段训练和闭环验证给出。

从推理层组合关系看，DDIM、CFG 与 TTS 可以统一收口在同一推理模块内，但不能简单看作并列开关。更合理的组合方式是：先由 DDIM 决定采样调度，再由 CFG 决定条件引导强度，最后由 TTS 在候选轨迹层进行搜索与筛选。因此，本文建议后续闭环优先验证的主方案为"DDIM-2 + 可选 CFG + 可配置 TTS"，并以此作为第四章系统与仿真验证的输入。

\section{本章小结}

本章围绕 NoMaD 的算法层问题展开研究。首先从 MDP 视角对视觉导航任务进行了形式化建模，分析了条件轨迹分布与传统策略的联系，并论证了 waypoint 表示在平台无关性、多模态表达和闭环控制方面的优势。其次，复现并梳理了 NoMaD 的视觉编码器架构、token 构建过程、训练目标和统一推理接口。在此基础上，以基线实验为参照，对 DDIM 推理加速、CFG 引导增强、TTS 推理时搜索和视觉编码器升级进行了系统分析。其中，DDIM 实验从 ODE 求解的理论角度解释了加速机制与失效边界，确认 \\texttt{ddim\\_2} 为最优加速配置；CFG 实验通过 score function 修改的理论分析揭示了时延翻倍的根本原因，并建立了引导尺度选择准则；TTS 实验设计了基于多目标评分的候选搜索框架，分析了计算预算分配策略；视觉编码器实验通过特征维度对齐和两阶段设计保证了对比公平性。

当前结论表明，NoMaD 的部署潜力主要取决于推理层优化与视觉条件建模的共同效果，而不是单个指标的局部提升。下一章将在此基础上把这些候选方案接入统一闭环系统，并通过 MuJoCo 仿真验证它们在 Lite3 平台上的系统级可行性。
"""

with open(r'contents\algorithm_design_and_experiments.tex', 'w', encoding='utf-8') as f:
    f.write(content)

line_count = content.count('\n') + 1
print(f'Final line count: {line_count}')
