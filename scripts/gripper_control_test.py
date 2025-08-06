#!/usr/bin/env python3
"""
独立的Interbotix夹爪控制测试脚本
支持位置控制、PWM控制和状态查询
"""

import time
import sys
import rospy
import numpy as np
from interbotix_xs_modules.arm import InterbotixManipulatorXS
from interbotix_xs_msgs.msg import JointSingleCommand

# SLEEPY_POSTION = [0.0, -1.8799999952316284, 1.600000023841858, 0.0, -1.600000023841858, 0.0]
SLEEPY_POSTION = [0.0, -1.8399999952316284, 1.600000023841858, 0.0, -1.600000023841858, 0.0]
START_POSITION = [0, -1.36, 1.16, 0, -0.3, 0]

class GripperController:
    def __init__(self, robot_model="vx300s", robot_name="puppet_left", init_node=True):
        """初始化夹爪控制器"""
        self.robot_model = robot_model
        self.robot_name = robot_name
        
        # 夹爪位置常量
        self.GRIPPER_JOINT_OPEN = 1.4910    # 完全打开
        self.GRIPPER_JOINT_CLOSE = -0.6213  # 完全关闭
        self.GRIPPER_JOINT_MID = (self.GRIPPER_JOINT_OPEN + self.GRIPPER_JOINT_CLOSE) / 2

        # 机器人运动控制常量
        self.DT = 0.02  # 控制周期，单位秒
        self.HOME_POSITION = [0, 0, 0, 0, 0]  # 机器人home位置
        
        # 初始化机器人
        print(f"正在初始化机器人 {robot_name}...")
        self.bot = InterbotixManipulatorXS(
            robot_model=robot_model,
            group_name="arm",
            gripper_name="gripper",
            robot_name=robot_name,
            init_node=init_node
        )
        
        # 设置夹爪操作模式
        self.setup_gripper()
        
        # 创建命令对象
        self.gripper_command = JointSingleCommand(name="gripper")
        
        print(f"机器人 {robot_name} 初始化完成!")
    
    def setup_gripper(self):
        """设置夹爪操作模式"""
        try:
            # 重启夹爪电机
            # self.bot.dxl.robot_reboot_motors("group", "arm", True)
            # print("arm电机重启完成")
            self.bot.dxl.robot_reboot_motors("single", "gripper", True)
            print("夹爪电机重启完成")
            
            # 设置夹爪操作模式为PWM
            # self.bot.dxl.robot_set_profile_type("group", "arm", "time")
            # self.bot.dxl.robot_set_operating_modes("group", "arm", "position")
            # print("set operating modes for arm")
            self.bot.dxl.robot_set_operating_modes("single", "gripper", "current_based_position")
            # print("set operating modes for gripper")
            
            # 获取当前操作模式
            arm_operating_mode = self.bot.dxl.robot_get_motor_registers("group", "arm", "Operating_Mode")
            gripper_operating_mode = self.bot.dxl.robot_get_motor_registers("single", "gripper", "Operating_Mode")
            print(f"Current arm operating mode: {arm_operating_mode}")
            print(f"Current gripper operating mode: {gripper_operating_mode}")

            # 获取profile type
            # arm_profile_type = self.bot.dxl.robot_get_motor_registers("group", "arm", "profile_type")
            # gripper_profile_type = self.bot.dxl.robot_get_motor_registers("single", "gripper", "profile_type")
            # print(f"Current arm profile type: {arm_profile_type}")
            # print(f"Current gripper profile type: {gripper_profile_type}")
            
        except Exception as e:
            print(f"设置夹爪时出错: {e}")
    
    def get_gripper_position(self):
        """获取当前夹爪位置"""
        try:
            joint_position = self.bot.gripper.core.joint_states.position[6]
            return joint_position
        except Exception as e:
            print(f"获取夹爪位置时出错: {e}")
            return None
    
    def set_gripper_position(self, position):
        """设置夹爪位置（关节角度）"""
        try:
            print(f"设置夹爪位置: {position:.3f}")
            self.gripper_command.cmd = position
            self.bot.gripper.core.pub_single.publish(self.gripper_command)
            # time.sleep(1)  # 等待命令执行
            
            # current_pos = self.get_gripper_position()
            # print(f"当前夹爪位置: {current_pos:.3f}")
            
        except Exception as e:
            print(f"设置夹爪位置时出错: {e}")
    
    def set_gripper_pwm(self, pwm_value):
        """设置夹爪PWM值"""
        try:
            # 限制PWM值范围
            pwm_value = max(-1000, min(1000, pwm_value))
            print(f"设置夹爪PWM: {pwm_value}")
            
            self.gripper_command.cmd = pwm_value
            self.bot.gripper.core.pub_single.publish(self.gripper_command)
            time.sleep(0.1)
            
        except Exception as e:
            print(f"设置夹爪PWM时出错: {e}")
    
    def open_gripper(self):
        """打开夹爪"""
        print("打开夹爪...")
        self.set_gripper_position(self.GRIPPER_JOINT_OPEN)
    
    def close_gripper(self):
        """关闭夹爪"""
        print("关闭夹爪...")
        self.set_gripper_position(self.GRIPPER_JOINT_CLOSE)
    
    def set_gripper_mid(self):
        """设置夹爪到中间位置"""
        print("设置夹爪到中间位置...")
        self.set_gripper_position(self.GRIPPER_JOINT_MID)
    
    def gripper_test_sequence(self):
        """夹爪测试序列"""
        print("\n=== 开始夹爪测试序列 ===")
        
        # 测试位置控制
        print("\n1. 测试位置控制:")
        start_pos = 0.6
        self.set_gripper_position(start_pos)
        time.sleep(1)
        # for i in range(50):
        #     self.set_gripper_position(start_pos)
        #     time.sleep(0.02)
        #     start_pos += 0.02
        
        # 测试PWM控制
        # print("\n2. 测试PWM控制:")
        # self.set_gripper_pwm(500)   # 正向PWM
        # time.sleep(2)
        
        # self.set_gripper_pwm(0)     # 停止
        # time.sleep(1)
        
        # self.set_gripper_pwm(-500)  # 负向PWM
        # time.sleep(2)
        
        # self.set_gripper_pwm(0)     # 停止
        # time.sleep(1)
        
        print("\n=== 夹爪测试序列完成 ===")
    
    def get_arm_joint_positions(self):
        """获取机器人当前关节位置"""
        return self.bot.arm.core.joint_states.position[:6]

    def move_to_position(self, target_pose, move_time=1.0):
        """
        移动机器人到目标位置
        :param target_pose: 目标关节角度列表 [j1, j2, j3, j4, j5]
        :param move_time: 移动时间(秒)
        """
        try:
            print(f"移动机器人到位置: {target_pose}")
            
            # 计算轨迹点数
            num_steps = int(move_time / self.DT)
            
            # 获取当前位置
            curr_pose = self.get_arm_joint_positions()
            
            # 生成轨迹
            traj = np.linspace(curr_pose, target_pose, num_steps)
            
            # 执行轨迹
            for t in range(num_steps):
                self.bot.arm.set_joint_positions(traj[t], blocking=False)
                time.sleep(self.DT)
                
            print("移动完成")
            
        except Exception as e:
            print(f"移动机器人时出错: {e}")

    def move_to_home(self):
        """移动机器人到home位置"""
        print("移动到home位置...")
        self.move_to_position(self.HOME_POSITION)

    def interactive_control(self):
        """交互式控制"""
        print("\n=== 交互式夹爪控制 ===")
        print("命令:")
        print("  o - 打开夹爪")
        print("  c - 关闭夹爪")
        print("  m - 中间位置")
        print("  p <value> - 设置PWM值 (例如: p 500)")
        print("  g - 获取当前位置")
        print("  h - 移动到home位置")
        print("  mv j1 j2 j3 j4 j5 - 移动到指定关节角度")
        print("  q - 退出")
        
        while True:
            try:
                cmd = input("\n请输入命令: ").strip().lower()
                
                if cmd == 'q':
                    break
                elif cmd == 'o':
                    self.open_gripper()
                elif cmd == 'c':
                    self.close_gripper()
                elif cmd == 'm':
                    self.set_gripper_mid()
                elif cmd == 'g':
                    pos = self.get_gripper_position()
                    print(f"当前夹爪位置: {pos:.3f}")
                    arm_pos = self.get_arm_joint_positions()
                    print(f"当前机械臂位置: {[f'{p:.3f}' for p in arm_pos]}")
                elif cmd.startswith('p '):
                    try:
                        pwm_value = int(cmd.split()[1])
                        self.set_gripper_pwm(pwm_value)
                    except (IndexError, ValueError):
                        print("PWM值格式错误，请使用: p <数值>")
                elif cmd == 'h':
                    self.move_to_home()
                elif cmd.startswith('mv '):
                    try:
                        # 解析关节角度
                        joint_values = [float(x) for x in cmd.split()[1:]]
                        if len(joint_values) != 5:
                            raise ValueError("需要5个关节角度值")
                        self.move_to_position(joint_values)
                    except (IndexError, ValueError) as e:
                        print(f"关节角度格式错误: {e}")
                        print("请使用: mv j1 j2 j3 j4 j5")
                else:
                    print("未知命令")
                    
            except KeyboardInterrupt:
                print("\n退出...")
                break
            except Exception as e:
                print(f"执行命令时出错: {e}")

def main():
    """主函数"""
    print("Interbotix夹爪控制测试脚本")
    print("=" * 40)
    
    # 检查命令行参数
    right_robot_name = "puppet_right"
    left_robot_name = "puppet_left"
    if len(sys.argv) > 1:
        right_robot_name = sys.argv[1]
        left_robot_name = sys.argv[2]
    
    try:
        # 初始化控制器
        controller_right = GripperController(robot_name=right_robot_name, init_node=True)
        controller_left = GripperController(robot_name=left_robot_name, init_node=False)
        
        # 运行测试序列
        controller_right.open_gripper()
        controller_left.open_gripper()
        
        # DEFAULT_RESET_POSITION = [0, -1.36, 1.16, 0, -0.3, 0]
        # controller.move_to_position(DEFAULT_RESET_POSITION)
        controller_right.move_to_position(SLEEPY_POSTION)
        controller_left.move_to_position(SLEEPY_POSTION)
        # # 交互式控制
        # controller.interactive_control()
        
    except Exception as e:
        print(f"程序出错: {e}")
        return 1
    
    print("程序结束")
    return 0

if __name__ == "__main__":
    exit(main()) 