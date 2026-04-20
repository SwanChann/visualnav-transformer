#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Third round of expansion."""
import os

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contents', 'preliminaries.tex')
with open(outpath, 'r', encoding='utf-8') as f:
    c = f.read()

print(f'Before: {c.count(chr(10))+1} lines')

replacements = [
    # 1. Add itemize for navigation pipeline components
    (
        '一个完整导航系统通常包含环境感知、状态表示、目标表达、局部决策和运动执行五个环节',
        '一个完整导航系统通常包含以下五个环节',
    ),
    (
        '五个环节\\cite{thrun2005probabilistic,anderson2018evaluation}。在输出形式上',
        '五个环节\\cite{thrun2005probabilistic,anderson2018evaluation}：\n'
        '\\begin{enumerate}\n'
        '    \\item \\textbf{环境感知}：通过传感器（相机、激光雷达等）获取环境的原始观测数据；\n'
        '    \\item \\textbf{状态表示}：将原始观测编码为适合决策的紧凑表征；\n'
        '    \\item \\textbf{目标表达}：以坐标、图像或语言等形式指定导航目标；\n'
        '    \\item \\textbf{局部决策}：基于当前状态和目标信息生成局部动作或路径规划；\n'
        '    \\item \\textbf{运动执行}：将高层决策转换为底层控制指令并驱动机器人运动。\n'
        '\\end{enumerate}\n'
        '在输出形式上'
    ),
    # 2. Add itemize for goal navigation types
    (
        '常见形式包括 point-goal、image-goal、object-goal 和 language-goal 导航。',
        '常见形式包括：\n'
        '\\begin{itemize}\n'
        '    \\item \\textbf{point-goal 导航}：目标以坐标点形式给出，要求机器人到达指定位置；\n'
        '    \\item \\textbf{image-goal 导航}：目标以一张或多张图像给出，要求机器人到达图像所对应的位置；\n'
        '    \\item \\textbf{object-goal 导航}：目标以物体类别给出（如"找到冰箱"），需要语义理解能力；\n'
        '    \\item \\textbf{language-goal 导航}：目标以自然语言指令给出，是最灵活但也最具挑战的形式。\n'
        '\\end{itemize}\n'
    ),
    # 3. Add more about noise schedule in diffusion
    (
        '余弦调度使信噪比在整个扩散过程中变化更均匀，在图像生成和轨迹生成任务中均展现了更好的样本质量。',
        '余弦调度使信噪比在整个扩散过程中变化更均匀，在图像生成和轨迹生成任务中均展现了更好的样本质量。'
        '在 NoMaD 的实现中，默认使用平方余弦调度（squaredcos\\_cap\\_v2），'
        '总扩散步数 $T = 100$。这一配置在轨迹生成质量与训练稳定性之间取得了良好的平衡。'
    ),
    # 4. Add more about scoring functions with equations
    (
        '其中 $\\lambda_{\\text{goal}}$、$\\lambda_{\\text{smooth}}$ 和 $\\lambda_{\\text{prog}}$ 为各维度的权重系数。',
        '其中 $\\lambda_{\\text{goal}}$、$\\lambda_{\\text{smooth}}$ 和 $\\lambda_{\\text{prog}}$ 为各维度的权重系数。'
        '具体地，平滑性评分可以定义为相邻 waypoint 方向变化角的负值之和：\n'
        '\\begin{equation}\n'
        's_{\\text{smooth}}(\\mathbf{A}) = -\\sum_{i=2}^{H-1} \\left|\\angle(\\mathbf{a}_{i+1} - \\mathbf{a}_i) - \\angle(\\mathbf{a}_i - \\mathbf{a}_{i-1})\\right|,\n'
        '\\end{equation}\n'
        '其中 $\\angle(\\cdot)$ 表示二维向量的方向角。前向进展评分则可以定义为轨迹终点在目标方向上的投影距离：\n'
        '\\begin{equation}\n'
        's_{\\text{prog}}(\\mathbf{A}) = \\mathbf{a}_H \\cdot \\hat{\\mathbf{d}}_{\\text{goal}},\n'
        '\\end{equation}\n'
        '其中 $\\hat{\\mathbf{d}}_{\\text{goal}}$ 为指向目标的单位方向向量。'
    ),
    # 5. Add ViNT backbone detail
    (
        '经过多层自注意力后，输出的 token 表征即为共享条件空间中的统一条件特征。这一统一条件特征同时被距离预测头和扩散策略头使用。',
        '经过多层自注意力后，输出的 token 表征即为共享条件空间中的统一条件特征。\n\n'
        '在 ViNT 的原始设计中，Transformer 编码器采用 4 层标准自注意力结构，'
        '隐藏维度为 256，注意力头数为 4。每帧图像经过 EfficientNet-B0 前端后产生 '
        '$7 \\times 7 = 49$ 个空间 token，经过线性投影后维度对齐为 256。'
        '对于 $K = 5$ 帧观察和 1 帧目标图像，Transformer 的输入序列长度为 '
        '$(5 + 1) \\times 49 = 294$ 个 token。为了区分不同帧的 token，'
        '系统为每帧添加可学习的帧级位置编码（frame-level positional encoding），'
        '使模型能够区分不同时间步的观察。Transformer 的输出 token 经过全局平均池化后，'
        '产生一个固定维度的条件向量，该向量同时被距离预测头和扩散策略头使用。'
    ),
    # 6. Add more about distance prediction head
    (
        '距离预测头估计当前观察与目标之间的相对距离；',
        '距离预测头估计当前观察与目标之间的相对距离（以时间步数为单位，而非物理距离）；',
    ),
    # 7. Expand the summary with more structure
    (
        '基于这些预备知识，下一章将进一步进入 NoMaD 的算法设计与离线实验分析。',
        '各部分预备知识与后续章节的对应关系如下：'
        '导航基础与 topomap 为第四章的系统部署提供了任务框架；'
        '扩散模型原理与 DDIM、CFG 理论为第三章的推理优化实验提供了方法基础；'
        'NoMaD 的结构与耦合分析为第三章的视觉编码器升级实验提供了设计依据；'
        '平台差异与 Lite3 参数为第四章的足式部署设计提供了约束条件。'
        '基于这些预备知识，下一章将进一步进入 NoMaD 的算法设计与离线实验分析。'
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
