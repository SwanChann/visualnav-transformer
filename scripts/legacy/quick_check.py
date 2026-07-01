#!/usr/bin/env python3
"""
快速验证单个轨迹文件
解决yaw数据类型问题
"""
import pickle
import numpy as np
import os

# 设置路径 - 使用expanduser处理~符号
pkl_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford/no10vcF_10_0/traj_data.pkl")

print("="*70)
print("GoStanford轨迹数据验证")
print("="*70)
print(f"路径: {pkl_path}")
print()

# 检查文件是否存在
if not os.path.exists(pkl_path):
    print(f"❌ 文件不存在: {pkl_path}")
    print()
    print("排查步骤:")
    print("1. 确认数据集位置:")
    print("   find ~ -name 'go_stanford' -type d 2>/dev/null")
    print()
    print("2. 如果在其他位置,修改脚本中的pkl_path变量")
    exit(1)

print("✅ 文件存在")
print()

# 读取数据
try:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print("✅ 成功读取pickle文件")
except Exception as e:
    print(f"❌ 读取失败: {e}")
    exit(1)

print()
print("-"*70)
print("数据结构:")
print("-"*70)
print(f"Keys: {list(data.keys())}")
print()

# Position数据
print("Position数据:")
print(f"  - Shape: {data['position'].shape}")
print(f"  - Dtype: {data['position'].dtype}")
print(f"  - X范围: [{data['position'][:, 0].min():.2f}, {data['position'][:, 0].max():.2f}]")
print(f"  - Y范围: [{data['position'][:, 1].min():.2f}, {data['position'][:, 1].max():.2f}]")
print()

# Yaw数据 - 处理可能的类型问题
print("Yaw数据:")
print(f"  - Shape: {data['yaw'].shape}")
print(f"  - Dtype: {data['yaw'].dtype}")

try:
    # 尝试转换为浮点数
    yaw = np.array(data['yaw'], dtype=float)
    print(f"  - 范围(弧度): [{yaw.min():.2f}, {yaw.max():.2f}]")
    print(f"  - 范围(度数): [{np.degrees(yaw.min()):.1f}°, {np.degrees(yaw.max()):.1f}°]")
    print(f"  - 平均值: {yaw.mean():.2f} rad ({np.degrees(yaw.mean()):.1f}°)")
except Exception as e:
    print(f"  - ⚠️  类型转换失败: {e}")
    print(f"  - 原始数据类型: {type(data['yaw'][0])}")
    # 尝试显示前几个值
    print(f"  - 前5个值: {data['yaw'][:5]}")

print()
print("-"*70)
print("数据完整性:")
print("-"*70)
num_positions = len(data['position'])
num_yaws = len(data['yaw'])
print(f"  - Position点数: {num_positions}")
print(f"  - Yaw点数: {num_yaws}")

if num_positions == num_yaws:
    print(f"  ✅ 数量匹配")
else:
    print(f"  ⚠️  数量不匹配! Position={num_positions}, Yaw={num_yaws}")

print()

# 检查对应的图像文件
traj_dir = os.path.dirname(pkl_path)
jpg_files = sorted([f for f in os.listdir(traj_dir) if f.endswith('.jpg')])
print(f"  - 图像文件数: {len(jpg_files)}")

if len(jpg_files) == num_positions:
    print(f"  ✅ 图像数量与位置点数量匹配")
else:
    print(f"  ⚠️  图像数量({len(jpg_files)})与位置点数量({num_positions})不匹配")

print()
print("="*70)
print("✅ 验证完成!")
print("="*70)
