#!/usr/bin/env python3
"""
简单的HDF5文件查看器
快速查看HDF5文件中的机器人轨迹数据
"""

import h5py
import sys
import os
import numpy as np


def quick_view_hdf5(file_path):
    """快速查看HDF5文件内容"""
    print(f"📁 文件: {file_path}")
    print("=" * 60)
    
    try:
        with h5py.File(file_path, 'r') as f:
            # 基本信息
            file_size = os.path.getsize(file_path) / (1024*1024)
            print(f"📊 文件大小: {file_size:.2f} MB")
            print(f"📂 根级项目: {list(f.keys())}")
            print()
            
            # 遍历所有数据集
            print("🔍 数据集详情:")
            print("-" * 40)
            
            def print_dataset(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"📋 {name}")
                    print(f"   形状: {obj.shape}")
                    print(f"   类型: {obj.dtype}")
                    print(f"   元素数: {obj.size}")
                    
                    # 显示数据样本
                    if obj.size is not None and obj.size <= 10:
                        print(f"   数据: {obj[:]}")
                    elif obj.size is not None and len(obj.shape) == 1:
                        print(f"   前5个: {obj[:5]}")
                        print(f"   后5个: {obj[-5:]}")
                    elif obj.size is not None and len(obj.shape) == 2:
                        print(f"   前3行: {obj[:3]}")
                    elif obj.size is not None:
                        # 使用numpy的flatten方法
                        data_array = np.array(obj)
                        print(f"   数据形状复杂，显示前几个元素: {data_array.flatten()[:5]}")
                    print()
            
            f.visititems(print_dataset)
            
            # 检查常见的机器人轨迹字段
            print("🤖 机器人轨迹字段:")
            print("-" * 40)
            
            robot_fields = {
                'actions': '动作数据',
                'observations': '观测数据', 
                'states': '状态数据',
                'joint_positions': '关节位置',
                'joint_velocities': '关节速度',
                'end_effector_positions': '末端执行器位置',
                'gripper_positions': '夹爪位置',
                'images': '图像数据',
                'camera_images': '相机图像',
                'rgb_images': 'RGB图像',
                'depth_images': '深度图像',
                'timestamps': '时间戳',
                'episode_lengths': '轨迹长度',
                'rewards': '奖励',
                'dones': '完成标志'
            }
            
            found_fields = []
            for field, description in robot_fields.items():
                if field in f:
                    dataset = f[field]
                    if isinstance(dataset, h5py.Dataset):
                        found_fields.append(f"✅ {field} ({description}): {dataset.shape}")
            
            if found_fields:
                for field_info in found_fields:
                    print(field_info)
            else:
                print("❌ 未找到常见的机器人轨迹字段")
            
            print()
            
            # 检查episode结构
            if 'episodes' in f:
                print("📚 Episode结构:")
                print("-" * 40)
                episodes = f['episodes']
                if isinstance(episodes, h5py.Group):
                    episode_names = list(episodes.keys())
                    print(f"Episode数量: {len(episode_names)}")
                    if episode_names:
                        print(f"前3个Episode: {episode_names[:3]}")
                        # 显示第一个episode的结构
                        first_ep = episodes[episode_names[0]]
                        if isinstance(first_ep, h5py.Group):
                            print(f"第一个Episode字段: {list(first_ep.keys())}")
            
    except FileNotFoundError:
        print(f"❌ 错误: 文件不存在 - {file_path}")
    except Exception as e:
        print(f"❌ 错误: {e}")


def main():
    if len(sys.argv) != 2:
        print("用法: python quick_hdf5_viewer.py <hdf5文件路径>")
        print("示例: python quick_hdf5_viewer.py trajectory.h5")
        sys.exit(1)
    
    file_path = sys.argv[1]
    quick_view_hdf5(file_path)


if __name__ == "__main__":
    main() 