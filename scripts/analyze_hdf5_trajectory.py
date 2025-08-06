#!/usr/bin/env python3
"""
HDF5轨迹数据分析脚本
用于打开HDF5文件并分析其中的机器人轨迹数据
"""

import h5py
import numpy as np
import argparse
import os
from pathlib import Path
from typing import Any


def print_dataset_info(name: str, obj: Any) -> None:
    """打印数据集信息的辅助函数"""
    if isinstance(obj, h5py.Dataset):
        print(f"  Dataset: {name}")
        print(f"    Shape: {obj.shape}")
        print(f"    Dtype: {obj.dtype}")
        print(f"    Size: {obj.size} elements")
        if obj.size is not None and obj.size < 10:  # 如果数据量不大，显示具体内容
            print(f"    Data: {obj[:]}")
        elif obj.size is not None:
            print(f"    First 5 elements: {obj[:5]}")
            print(f"    Last 5 elements: {obj[-5:]}")
        print()


def analyze_hdf5_file(file_path: str) -> None:
    """分析HDF5文件的结构和内容"""
    print(f"正在分析HDF5文件: {file_path}")
    print("=" * 50)
    
    try:
        with h5py.File(file_path, 'r') as f:
            print(f"文件大小: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
            print(f"根组数量: {len(f.keys())}")
            print()
            
            # 递归遍历所有组和数据集
            def visit_all(name: str, obj: Any) -> None:
                if isinstance(obj, h5py.Group):
                    print(f"Group: {name}")
                    print(f"  子项目数量: {len(obj.keys())}")
                    if len(obj.keys()) > 0:
                        print(f"  子项目: {list(obj.keys())}")
                    print()
                elif isinstance(obj, h5py.Dataset):
                    print_dataset_info(name, obj)
            
            f.visititems(visit_all)
            
            # 特别关注常见的轨迹字段
            print("=" * 50)
            print("轨迹数据字段分析:")
            print("=" * 50)
            
            # 查找常见的轨迹字段
            common_fields = ['actions', 'observations', 'states', 'joint_positions', 
                           'joint_velocities', 'end_effector_positions', 'gripper_positions',
                           'images', 'camera_images', 'rgb_images', 'depth_images',
                           'timestamps', 'episode_lengths', 'rewards', 'dones']
            
            for field in common_fields:
                if field in f:
                    print(f"找到字段: {field}")
                    dataset = f[field]
                    if isinstance(dataset, h5py.Dataset):
                        print(f"  形状: {dataset.shape}")
                        print(f"  数据类型: {dataset.dtype}")
                        if dataset.size is not None and dataset.size < 20:
                            print(f"  数据: {dataset[:]}")
                        elif dataset.size is not None:
                            print(f"  前5个元素: {dataset[:5]}")
                    print()
            
            # 如果文件有episode结构
            if 'episodes' in f:
                print("发现episodes组:")
                episodes = f['episodes']
                if isinstance(episodes, h5py.Group):
                    print(f"  Episode数量: {len(episodes.keys())}")
                    for ep_name in list(episodes.keys())[:3]:  # 只显示前3个episode
                        print(f"  Episode: {ep_name}")
                        ep_data = episodes[ep_name]
                        if isinstance(ep_data, h5py.Group):
                            for key in ep_data.keys():
                                item = ep_data[key]
                                if isinstance(item, h5py.Dataset):
                                    print(f"    {key}: {item.shape} - {item.dtype}")
                    print()
            
            # 显示文件属性
            print("=" * 50)
            print("文件属性:")
            print("=" * 50)
            for key, value in f.attrs.items():
                print(f"{key}: {value}")
            
    except FileNotFoundError:
        print(f"错误: 文件 {file_path} 不存在")
    except Exception as e:
        print(f"错误: 无法打开文件 {file_path}")
        print(f"错误信息: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description='分析HDF5轨迹文件')
    parser.add_argument('file_path', help='HDF5文件路径')
    parser.add_argument('--output', '-o', help='输出结果到文件')
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not os.path.exists(args.file_path):
        print(f"错误: 文件 {args.file_path} 不存在")
        return
    
    # 分析文件
    if args.output:
        import sys
        original_stdout = sys.stdout
        with open(args.output, 'w', encoding='utf-8') as f:
            sys.stdout = f
            analyze_hdf5_file(args.file_path)
            sys.stdout = original_stdout
        print(f"分析结果已保存到: {args.output}")
    else:
        analyze_hdf5_file(args.file_path)


if __name__ == "__main__":
    main() 