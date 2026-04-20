#!/usr/bin/env python3
# -*- coding: utf-8 -*-
with open('contents/preliminaries.tex', 'r', encoding='utf-8') as f:
    c = f.read()

c = c.replace(
    '选择当前更合适的局部子目标。由此可见，',
    '选择当前更合适的局部子目标。\n\n由此可见，'
)
c = c.replace(
    '在推理阶段，对同一输入分别执行条件和无条件前向传播，',
    '在推理阶段，对同一输入分别执行条件和无条件前向传播，\n'
)
c = c.replace(
    '省去了 CNN 前端，token 来自 patch 的线性嵌入。',
    '省去了 CNN 前端，\ntoken 来自 patch 的线性嵌入。'
)
c = c.replace(
    '这种"按需扩展"的',
    '这种"按需扩展"的\n'
)

with open('contents/preliminaries.tex', 'w', encoding='utf-8') as f:
    f.write(c)
print(f'Final: {c.count(chr(10))+1} lines')
