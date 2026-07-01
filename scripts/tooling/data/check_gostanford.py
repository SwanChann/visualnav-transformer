#!/usr/bin/env python3
"""
GoStanford数据集验证脚本
用于检查数据集的完整性和正确性
"""
import pickle
import numpy as np
import os
import sys

def check_trajectory(traj_path):
    """检查单个轨迹的完整性"""
    traj_path = os.path.expanduser(traj_path)
    
    if not os.path.exists(traj_path):
        print(f"❌ 轨迹目录不存在: {traj_path}")
        return False
    
    # 检查pickle文件
    pkl_file = os.path.join(traj_path, "traj_data.pkl")
    if not os.path.exists(pkl_file):
        print(f"❌ 缺少traj_data.pkl: {traj_path}")
        return False
    
    # 检查图像文件
    jpg_files = sorted([f for f in os.listdir(traj_path) if f.endswith('.jpg')])
    if len(jpg_files) == 0:
        print(f"❌ 没有图像文件: {traj_path}")
        return False
    
    # 读取pickle数据
    try:
        with open(pkl_file, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"❌ 读取pickle文件失败: {e}")
        return False
    
    # 检查数据结构
    if 'position' not in data or 'yaw' not in data:
        print(f"❌ pickle文件缺少必要的键: {data.keys()}")
        return False
    
    num_images = len(jpg_files)
    num_positions = len(data['position'])
    
    print(f"✅ {os.path.basename(traj_path)}")
    print(f"   - 图像数量: {num_images}")
    print(f"   - 位置点数量: {num_positions}")
    print(f"   - Position shape: {data['position'].shape}")
    print(f"   - Yaw shape: {data['yaw'].shape}")
    
    if num_images != num_positions:
        print(f"   ⚠️  警告: 图像数量({num_images})与位置点数量({num_positions})不匹配")
    
    # 检查数据范围
    print(f"   - Position range: X[{data['position'][:, 0].min():.2f}, {data['position'][:, 0].max():.2f}], "
          f"Y[{data['position'][:, 1].min():.2f}, {data['position'][:, 1].max():.2f}]")
    
    # 处理yaw可能是对象数组的情况
    try:
        yaw = np.array(data['yaw'], dtype=float)  # 确保转换为浮点数
        print(f"   - Yaw range: [{yaw.min():.2f}, {yaw.max():.2f}] radians "
              f"([{np.degrees(yaw.min()):.1f}, {np.degrees(yaw.max()):.1f}] degrees)")
    except Exception as e:
        print(f"   - Yaw数据类型: {data['yaw'].dtype} (可能包含非数值)")
    
    return True

def check_dataset(dataset_root, sample_size=5):
    """检查整个数据集"""
    dataset_root = os.path.expanduser(dataset_root)
    
    if not os.path.exists(dataset_root):
        print(f"❌ 数据集根目录不存在: {dataset_root}")
        print(f"请确认路径是否正确!")
        return False
    
    print(f"数据集根目录: {dataset_root}")
    print("="*70)
    
    # 获取所有轨迹目录
    try:
        trajectories = sorted([d for d in os.listdir(dataset_root) 
                              if os.path.isdir(os.path.join(dataset_root, d))])
    except Exception as e:
        print(f"❌ 读取目录失败: {e}")
        return False
    
    # 过滤掉train/test等特殊目录
    trajectories = [t for t in trajectories if t not in ['train', 'test', 'val']]
    
    print(f"找到 {len(trajectories)} 个轨迹目录\n")
    
    if len(trajectories) == 0:
        print("❌ 没有找到轨迹目录!")
        return False
    
    # 检查样本
    print(f"检查前 {sample_size} 个轨迹...\n")
    success_count = 0
    
    for i, traj_name in enumerate(trajectories[:sample_size]):
        traj_path = os.path.join(dataset_root, traj_name)
        if check_trajectory(traj_path):
            success_count += 1
        print()
    
    print("="*70)
    print(f"样本检查结果: {success_count}/{sample_size} 成功")
    
    # 统计信息
    print("\n数据集统计:")
    print(f"  - 总轨迹数: {len(trajectories)}")
    
    # 统计不同类型的轨迹
    real_count = len([t for t in trajectories if t.startswith('no')])
    sim_count = len([t for t in trajectories if t.startswith('sim')])
    
    print(f"  - 真实机器人轨迹 (no*): {real_count}")
    print(f"  - 仿真轨迹 (sim*): {sim_count}")
    
    return True

def check_split_files(dataset_root):
    """检查数据分割文件"""
    dataset_root = os.path.expanduser(dataset_root)
    
    print("\n" + "="*70)
    print("检查数据分割文件...")
    print("="*70)
    
    train_file = os.path.join(dataset_root, "train", "traj_names.txt")
    test_file = os.path.join(dataset_root, "test", "traj_names.txt")
    
    if os.path.exists(train_file):
        with open(train_file, 'r') as f:
            train_names = [line.strip() for line in f if line.strip()]
        print(f"✅ train/traj_names.txt: {len(train_names)} 个训练轨迹")
    else:
        print(f"⚠️  train/traj_names.txt 不存在 (需要运行 data_split.py)")
    
    if os.path.exists(test_file):
        with open(test_file, 'r') as f:
            test_names = [line.strip() for line in f if line.strip()]
        print(f"✅ test/traj_names.txt: {len(test_names)} 个测试轨迹")
    else:
        print(f"⚠️  test/traj_names.txt 不存在 (需要运行 data_split.py)")

if __name__ == "__main__":
    print("="*70)
    print("GoStanford数据集验证工具")
    print("="*70)
    print()
    
    # 默认路径
    default_path = "~/visualnav-transformer/nomad_dataset/go_stanford"
    
    if len(sys.argv) > 1:
        dataset_root = sys.argv[1]
    else:
        dataset_root = default_path
    
    print(f"使用路径: {dataset_root}")
    print(f"展开后: {os.path.expanduser(dataset_root)}")
    print()
    
    # 检查整个数据集
    if check_dataset(dataset_root, sample_size=5):
        # 检查分割文件
        check_split_files(dataset_root)
        print("\n✅ 数据集验证完成!")
    else:
        print("\n❌ 数据集验证失败!")
        print("\n排查建议:")
        print("1. 检查路径是否正确:")
        print(f"   find ~ -name 'go_stanford' -type d 2>/dev/null")
        print("2. 确认数据已下载:")
        print(f"   ls -la {os.path.expanduser(dataset_root)}")
        print("3. 如果使用其他路径,运行:")
        print(f"   python {sys.argv[0]} /your/actual/path/to/go_stanford")
