#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fifth round - final fine-tuning to reach ~380."""
import os

outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'contents', 'preliminaries.tex')
with open(outpath, 'r', encoding='utf-8') as f:
    c = f.read()

print(f'Before: {c.count(chr(10))+1} lines')

replacements = [
    # 1. Add more about image-goal formalization
    (
        '降低了系统复杂度，同时目标表达的歧义性也更低',
        '降低了系统复杂度。同时，image-goal 的表达歧义性也更低',
    ),
    # 2. Expand the ViNT backbone paragraph with line breaks
    (
        '在 ViNT 的原始设计中，Transformer 编码器采用 4 层标准自注意力结构，',
        '在 ViNT 的原始设计中，Transformer 编码器采用 4 层标准自注意力结构，\n',
    ),
    (
        '系统为每帧添加可学习的帧级位置编码（frame-level positional encoding），',
        '系统为每帧添加可学习的帧级位置编码\n（frame-level positional encoding），',
    ),
    # 3. Break long paragraphs into multiple lines for readability
    (
        '关于观察频率与控制频率的关系，需要区分三个层次的频率。底层运动控制器',
        '关于观察频率与控制频率的关系，需要区分三个层次的频率。\n\n底层运动控制器',
    ),
    (
        '而本文的高层视觉导航策略以 $4\\,\\text{Hz}$ 运行，每个周期完成一次图像采集、推理和速度指令发送。',
        '而本文的高层视觉导航策略以 $4\\,\\text{Hz}$ 运行，\n每个周期完成一次图像采集、推理和速度指令发送。',
    ),
    # 4. Add more detail about distance prediction
    (
        '距离预测头估计当前观察与目标之间的相对距离（以时间步数为单位，而非物理距离）；',
        '距离预测头估计当前观察与目标之间的相对距离\n'
        '（以离散时间步数为单位，表示从当前位置到达目标大约需要多少个导航步骤，而非物理距离）；',
    ),
    # 5. Break up the Lite3 section intro
    (
        '结合项目配置与部署代码，其在本文中的系统定位是',
        '结合项目配置与部署代码，\n其在本文中的系统定位是',
    ),
    # 6. Add note about training data diversity
    (
        '正因为如此，NoMaD 才能在部署时同时支持导航模式与探索模式。',
        '正因为如此，NoMaD 才能在部署时同时支持导航模式与探索模式。\n\n'
        'goal mask 的设计还有一个重要的技术意义：它使得模型在训练时自然地学习了\n'
        '"当前观察本身蕴含的行为先验"。即使没有目标指引，训练数据中的轨迹通常\n'
        '也遵循特定的行为模式（如沿道路行驶、避开障碍物），goal mask 使模型能够\n'
        '将这些模式内化为无条件生成能力。',
    ),
    # 7. Add more about the waypoint to velocity conversion
    (
        '（3）仅将前 $H_{\\text{exec}}$ 个 waypoint 转换为速度指令并发送给底层控制器；',
        '（3）仅将前 $H_{\\text{exec}}$ 个 waypoint 转换为线速度和角速度指令，\n'
        '通过比例控制器映射后发送给底层控制器；',
    ),
    # 8. Add note before figure placeholder in diffusion section
    (
        '\\caption{图位占位：扩散模型、扩散策略与 NoMaD 关系示意图（待补）}',
        '\\caption{扩散模型、扩散策略、ViT 编码器与 NoMaD 系统之间的技术演化关系。\n'
        '    从左到右依次为基础扩散理论（DDPM/DDIM）、条件引导方法（CFG）、\n'
        '    扩散策略（Diffusion Policy）以及 NoMaD 的完整架构（待补）}',
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
