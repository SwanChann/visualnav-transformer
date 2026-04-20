#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fourth round - final push to ~380 lines."""
import os

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contents', 'preliminaries.tex')
with open(outpath, 'r', encoding='utf-8') as f:
    c = f.read()

print(f'Before: {c.count(chr(10))+1} lines')

replacements = [
    # 1. Add enumeration for diffusion model advantages in navigation
    (
        '这种建模方式的优势在于，它不强迫模型在多峰分布中只输出一个均值解，而是允许模型通过去噪过程逐步生成多种合理候选。导航与控制任务中的未来轨迹本身就常常具有多模态特性，因此扩散建模天然适合此类问题。',
        '这种建模方式在导航领域具有以下关键优势：\n'
        '\\begin{enumerate}\n'
        '    \\item \\textbf{多模态建模能力}：扩散模型不强迫在多峰分布中只输出一个均值解，而是通过去噪过程逐步生成多种合理候选。在十字路口等存在多条合理路径的场景中，这种能力尤为重要；\n'
        '    \\item \\textbf{样本质量可控}：通过调节去噪步数和引导强度，可以在生成质量与计算开销之间灵活权衡；\n'
        '    \\item \\textbf{条件生成自然}：条件信息（如目标图像）可以自然地融入去噪过程，无需额外的条件注入机制；\n'
        '    \\item \\textbf{与采样后选择兼容}：天然支持生成多条候选轨迹并进行后续筛选（即 TTS），这在确定性策略中难以实现。\n'
        '\\end{enumerate}\n'
        '导航与控制任务中的未来轨迹本身就常常具有多模态特性，因此扩散建模天然适合此类问题。'
    ),
    # 2. Add more about CFG in NoMaD context
    (
        '需要注意的是，在 NoMaD 的实际实现中，CFG 的无条件分支并非完全无信息',
        '在 NoMaD 中 CFG 的具体实现流程如下：\n'
        '\\begin{enumerate}\n'
        '    \\item 在每个去噪步骤中，分别以目标 token 和掩码 token 作为条件，执行两次前向传播；\n'
        '    \\item 获得条件噪声预测 $\\epsilon_{\\text{cond}}$ 和无条件噪声预测 $\\epsilon_{\\text{uncond}}$；\n'
        '    \\item 按照 CFG 公式计算引导后的噪声预测 $\\epsilon_{\\text{cfg}}$；\n'
        '    \\item 使用 $\\epsilon_{\\text{cfg}}$ 执行一步去噪更新。\n'
        '\\end{enumerate}\n'
        '需要注意的是，在 NoMaD 的实际实现中，CFG 的无条件分支并非完全无信息'
    ),
    # 3. Expand Diffuser/Diffusion Policy discussion
    (
        '因此被视为扩散模型在具身控制中的代表性应用。',
        '因此被视为扩散模型在具身控制中的代表性应用。\n\n'
        '具体而言，Diffuser\\cite{diffuser2022} 将整个轨迹（包括状态和动作序列）作为扩散对象，'
        '在推理时通过引导函数注入奖励信息来生成高回报轨迹。'
        'Diffusion Policy\\cite{diffusionpolicy2023} 则只对动作序列进行扩散，'
        '将状态观察作为条件信息，更适合视觉观察驱动的控制任务。'
        'NoMaD 的扩散策略头采用了类似 Diffusion Policy 的设计——仅对 waypoint 轨迹'
        '进行扩散，视觉特征作为条件信息注入。这种设计使得扩散过程仅需在低维轨迹空间中'
        '进行，显著降低了计算开销。'
    ),
    # 4. Add details about the reverse process parameterization
    (
        '在条件生成场景中，$\\epsilon_\\theta$ 额外接收条件信息 $c$，但损失函数的形式保持不变。',
        '在条件生成场景中，$\\epsilon_\\theta$ 额外接收条件信息 $c$，但损失函数的形式保持不变。'
        '在 NoMaD 的扩散策略头中，噪声预测网络 $\\epsilon_\\theta$ 采用一维卷积架构'
        '（1D ConvNet），以加噪轨迹 $x_t$、时间步嵌入和视觉条件特征的拼接作为输入，'
        '输出与 $x_t$ 同维度的噪声预测。选择 1D ConvNet 而非全连接网络的原因在于，'
        '轨迹具有天然的时序结构，卷积操作能够捕捉相邻 waypoint 之间的局部依赖关系。'
    ),
    # 5. Add more to chapter summary with structure
    (
        '各部分预备知识与后续章节的对应关系如下：',
        '\n\n各部分预备知识与后续章节的对应关系如下：\n'
        '\\begin{itemize}\n'
        '    \\item '
    ),
    (
        '导航基础与 topomap 为第四章的系统部署提供了任务框架；',
        '导航基础与 topomap 为第四章的系统部署提供了任务框架；\n'
        '    \\item '
    ),
    (
        '扩散模型原理与 DDIM、CFG 理论为第三章的推理优化实验提供了方法基础；',
        '扩散模型原理与 DDIM、CFG 理论为第三章的推理优化实验提供了方法基础；\n'
        '    \\item '
    ),
    (
        'NoMaD 的结构与耦合分析为第三章的视觉编码器升级实验提供了设计依据；',
        'NoMaD 的结构与耦合分析为第三章的视觉编码器升级实验提供了设计依据；\n'
        '    \\item '
    ),
    (
        '平台差异与 Lite3 参数为第四章的足式部署设计提供了约束条件。基于这些预备知识',
        '平台差异与 Lite3 参数为第四章的足式部署设计提供了约束条件。\n'
        '\\end{itemize}\n'
        '基于这些预备知识'
    ),
]

for old, new in replacements:
    if old in c:
        c = c.replace(old, new, 1)
        print(f'OK: {old[:50]}...')
    else:
        print(f'MISS: {old[:50]}...')

with open(outpath, 'w', encoding='utf-8') as f:
    f.write(c)

print(f'After: {c.count(chr(10))+1} lines')
