#!/usr/bin/env python3
"""
修复 GoStanford2 数据集的数据类型问题
将 object 类型转换为 float64 类型

使用方法:
    python fix_go_stanford_dtype.py

功能:
    1. 自动检测所有轨迹文件
    2. 备份原始数据（.backup）
    3. 转换 position 和 yaw 为 float64
    4. 验证修复结果

错误信息:
    如果训练时遇到以下错误，请运行此脚本：
    ValueError: setting an array element with a sequence. 
    The requested array has an inhomogeneous shape after 2 dimensions.
"""
import pickle
import numpy as np
import os
from tqdm import tqdm
import shutil
import sys

def fix_trajectory_dtype(traj_folder):
    """修复单个轨迹的数据类型"""
    pkl_file = os.path.join(traj_folder, "traj_data.pkl")
    
    if not os.path.exists(pkl_file):
        return False, "pkl文件不存在"
    
    # 备份原文件
    backup_file = pkl_file + ".backup"
    if not os.path.exists(backup_file):
        try:
            shutil.copy2(pkl_file, backup_file)
        except Exception as e:
            return False, f"备份失败: {str(e)}"
    
    try:
        # 读取数据
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        # 检查并转换数据类型
        needs_fix = False
        
        if 'position' in data:
            if data['position'].dtype == 'object':
                data['position'] = np.array(data['position'], dtype=np.float64)
                needs_fix = True
            elif data['position'].dtype != np.float64:
                data['position'] = data['position'].astype(np.float64)
                needs_fix = True
        
        if 'yaw' in data:
            if data['yaw'].dtype == 'object':
                data['yaw'] = np.array(data['yaw'], dtype=np.float64)
                needs_fix = True
            elif data['yaw'].dtype != np.float64:
                data['yaw'] = data['yaw'].astype(np.float64)
                needs_fix = True
        
        # 如果需要修复，保存修复后的数据
        if needs_fix:
            with open(pkl_file, 'wb') as f:
                pickle.dump(data, f)
            return True, "已修复"
        else:
            return True, "无需修复"
    
    except Exception as e:
        return False, f"修复失败: {str(e)}"

def check_single_trajectory(traj_folder):
    """检查单个轨迹的数据类型"""
    pkl_file = os.path.join(traj_folder, "traj_data.pkl")
    
    if not os.path.exists(pkl_file):
        return None, None
    
    try:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        
        pos_dtype = data.get('position', np.array([])).dtype
        yaw_dtype = data.get('yaw', np.array([])).dtype
        
        return pos_dtype, yaw_dtype
    except:
        return None, None

def main():
    print("="*60)
    print("GoStanford2 数据集类型修复工具")
    print("="*60)
    
    # 默认数据集路径
    default_path = os.path.expanduser("~/visualnav-transformer/nomad_dataset/go_stanford")
    
    # 尝试当前目录
    if not os.path.exists(default_path):
        default_path = os.path.join(os.getcwd(), "nomad_dataset", "go_stanford")
    
    # 让用户确认或输入路径
    print(f"\n默认数据集路径: {default_path}")
    user_input = input("按回车使用默认路径，或输入自定义路径: ").strip()
    
    if user_input:
        dataset_root = os.path.expanduser(user_input)
    else:
        dataset_root = default_path
    
    if not os.path.exists(dataset_root):
        print(f"\n❌ 错误: 数据集路径不存在: {dataset_root}")
        print("请检查路径是否正确")
        sys.exit(1)
    
    # 获取所有轨迹文件夹
    print(f"\n扫描轨迹文件夹...")
    traj_folders = []
    for item in os.listdir(dataset_root):
        traj_path = os.path.join(dataset_root, item)
        if os.path.isdir(traj_path):
            pkl_file = os.path.join(traj_path, "traj_data.pkl")
            if os.path.exists(pkl_file):
                traj_folders.append(traj_path)
    
    if not traj_folders:
        print(f"❌ 错误: 在 {dataset_root} 中未找到任何轨迹数据")
        sys.exit(1)
    
    print(f"✅ 找到 {len(traj_folders)} 个轨迹文件夹\n")
    
    # 检查第一个轨迹的数据类型
    print("检查数据类型...")
    sample_traj = traj_folders[0]
    pos_dtype, yaw_dtype = check_single_trajectory(sample_traj)
    
    if pos_dtype is None:
        print("❌ 无法读取数据，请检查文件格式")
        sys.exit(1)
    
    print(f"  示例轨迹: {os.path.basename(sample_traj)}")
    print(f"  position dtype: {pos_dtype}")
    print(f"  yaw dtype: {yaw_dtype}")
    
    needs_fix = (pos_dtype == 'object' or yaw_dtype == 'object' or 
                 pos_dtype != np.float64 or yaw_dtype != np.float64)
    
    if not needs_fix:
        print("\n✅ 数据类型正确，无需修复！")
        return
    
    print("\n⚠️  检测到数据类型问题，需要修复")
    print(f"   目标类型: float64")
    
    # 确认修复
    confirm = input("\n开始修复? (y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    print("\n开始修复数据类型...")
    print("(原文件将自动备份为 *.backup)\n")
    
    success_count = 0
    fixed_count = 0
    error_count = 0
    errors = []
    
    # 处理每个轨迹
    for traj_folder in tqdm(traj_folders, desc="处理进度"):
        success, message = fix_trajectory_dtype(traj_folder)
        
        if success:
            success_count += 1
            if message == "已修复":
                fixed_count += 1
        else:
            error_count += 1
            errors.append((os.path.basename(traj_folder), message))
            if len(errors) <= 5:  # 只显示前5个错误
                tqdm.write(f"❌ {os.path.basename(traj_folder)}: {message}")
    
    print("\n" + "="*60)
    print("修复完成！")
    print("="*60)
    print(f"  总轨迹数: {len(traj_folders)}")
    print(f"  成功处理: {success_count}")
    print(f"  已修复: {fixed_count}")
    print(f"  无需修复: {success_count - fixed_count}")
    print(f"  失败: {error_count}")
    
    if error_count > 0:
        print(f"\n⚠️  有 {error_count} 个轨迹处理失败")
        if len(errors) > 5:
            print(f"   (仅显示前5个错误)")
    
    print("="*60)
    
    if fixed_count > 0:
        print("\n✅ 数据类型已修复！")
        print(f"   备份文件: {dataset_root}/**/traj_data.pkl.backup")
    
    # 验证修复结果
    print("\n验证修复结果...")
    sample_traj = traj_folders[0]
    pos_dtype, yaw_dtype = check_single_trajectory(sample_traj)
    
    print(f"  示例轨迹: {os.path.basename(sample_traj)}")
    print(f"  position dtype: {pos_dtype}")
    print(f"  yaw dtype: {yaw_dtype}")
    
    if pos_dtype == np.float64 and yaw_dtype == np.float64:
        print("\n✅ 验证通过！数据类型正确，可以开始训练")
    else:
        print("\n⚠️  验证失败，部分数据可能仍有问题")
        print("   建议检查错误日志")
    
    print("\n" + "="*60)
    print("提示:")
    print("  1. 如需恢复原始数据，删除 traj_data.pkl 并重命名 .backup 文件")
    print("  2. 确认无误后可删除所有 .backup 文件以节省空间")
    print("  3. 现在可以运行训练命令: python train.py -c config/nomad.yaml")
    print("="*60)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
