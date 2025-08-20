#!/usr/bin/env python3
"""
下载 aloha_pen_uncap_diverse 数据集并保存前300条数据的特定维度
"""

import numpy as np
import matplotlib.pyplot as plt
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME as LEROBOT_HOME
print(LEROBOT_HOME)

def _normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)

def _unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val

def _gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with pi0 which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = _unnormalize(value, min_val=0.01844, max_val=0.05800)

    

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return np.arcsin(np.clip(value, -1.0, 1.0))

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)
    return value
    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return _normalize(value, min_val=0.4, max_val=1.5)

def main():
    print("开始下载 aloha_pen_uncap_diverse 数据集...")
    
    # 下载数据集
    dataset = LeRobotDataset("physical-intelligence/aloha_pen_uncap_diverse", revision="v2.0")
    # dataset.pull_from_repo()
    print(f"数据集下载成功！")
    
    # 准备保存数据
    num_episodes = min(600, len(dataset))
    state_dim7 = []
    state_dim14 = []
    action_dim7 = []
    action_dim14 = []
    
    print(f"开始处理前 {num_episodes} 条数据...")
    
    # 遍历前300条数据
    for i in range(num_episodes):
        episode = dataset[i]
        
        # 获取observation.state数据
        state = episode['observation.state']
        state_dim7.append(state[6])
        state_dim14.append(state[13])
        
        # 获取action数据
        action = episode['action']
        action_dim7.append(action[6])
        action_dim14.append(action[13])
        
        if (i + 1) % 50 == 0:
            print(f"已处理 {i + 1}/{num_episodes} 条数据")
    
    # 转换为numpy数组
    state_dim7 = np.array(state_dim7)
    state_dim14 = np.array(state_dim14)
    action_dim7 = np.array(action_dim7)
    action_dim14 = np.array(action_dim14)
    
    # 保存到txt文件
    output_file = "aloha_data_dimensions.txt"
    
    # with open(output_file, 'w') as f:
    #     # 写入表头
    #     f.write("state_dim7,state_dim14,action_dim7,action_dim14\n")
        
    #     # 每行对应一条数据
    #     for i in range(len(state_dim7)):
    #         f.write(f"{state_dim7[i]},{state_dim14[i]},{action_dim7[i]},{action_dim14[i]}\n")
    
    # print(f"\n数据已保存到 {output_file}")
    
    # 绘制四条曲线
    plt.figure(figsize=(12, 8))
    
    x = range(len(state_dim7))
    
    plt.subplot(2, 2, 1)
    plt.plot(x, state_dim7, 'b-', linewidth=1.5)
    plt.title('observation.state 第7维度 (索引6)')
    plt.xlabel('数据索引')
    plt.ylabel('数值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    plt.plot(x, state_dim14, 'r-', linewidth=1.5)
    plt.title('observation.state 第14维度 (索引13)')
    plt.xlabel('数据索引')
    plt.ylabel('数值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    plt.plot(x, action_dim7, 'g-', linewidth=1.5)
    plt.title('action 第7维度 (索引6)')
    plt.xlabel('数据索引')
    plt.ylabel('数值')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 4)
    plt.plot(x, action_dim14, 'm-', linewidth=1.5)
    plt.title('action 第14维度 (索引13)')
    plt.xlabel('数据索引')
    plt.ylabel('数值')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    plot_file = "aloha_data_curves.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"曲线图已保存到 {plot_file}")
    
    # 显示图片
    plt.show()
    
    print(f"\nobservation.state 第7维度: {state_dim7.shape}")
    print(f"observation.state 第14维度: {state_dim14.shape}")
    print(f"action 第7维度: {action_dim7.shape}")
    print(f"action 第14维度: {action_dim14.shape}")
    
    # 显示前几条数据作为示例
    print(f"\n前5条数据示例:")
    print(f"state_dim7: {state_dim7[:5]}")
    print(f"state_dim14: {state_dim14[:5]}")
    print(f"action_dim7: {action_dim7[:5]}")
    print(f"action_dim14: {action_dim14[:5]}")

if __name__ == "__main__":
    main() 