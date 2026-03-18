#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import torch
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

# 导入写的协同框架
from collaborative_framework import DualUAVSystem

class SwarmRescueROSNode:
    def __init__(self):
        rospy.init_node('swarm_rescue_snn_controller', anonymous=True)
        
        # 初始化双机系统模型 (假设已经加载了预训练权重或处于在线训练模式)
        self.model = DualUAVSystem()
        
        # 状态存储字典
        self.obs_uav1 = {'velocity': torch.zeros(1, 3), 'env_features': torch.zeros(1, 128), 'lidar_spikes': torch.zeros(1, 24)}
        self.obs_uav2 = {'velocity': torch.zeros(1, 3), 'env_features': torch.zeros(1, 128), 'lidar_spikes': torch.zeros(1, 24)}

        # ---- 订阅者 Subscribers (获取仿真环境数据) ----
        # 假设在 launch 文件中，两架无人机的命名空间分别为 /uav1 和 /uav2
        rospy.Subscriber('/uav1/scan', LaserScan, self.lidar_callback_uav1)
        rospy.Subscriber('/uav1/ground_truth/state', Odometry, self.odom_callback_uav1)
        
        rospy.Subscriber('/uav2/scan', LaserScan, self.lidar_callback_uav2)
        rospy.Subscriber('/uav2/ground_truth/state', Odometry, self.odom_callback_uav2)

        # ---- 发布者 Publishers (发送控制指令给仿真环境) ----
        self.cmd_pub_uav1 = rospy.Publisher('/uav1/cmd_vel', Twist, queue_size=10)
        self.cmd_pub_uav2 = rospy.Publisher('/uav2/cmd_vel', Twist, queue_size=10)

        # 控制循环频率 (例如 10Hz)
        self.rate = rospy.Rate(10)

    # --- 回调函数：处理传感器数据并转换为神经网络输入 ---
    def lidar_callback_uav1(self, msg):
        """处理雷达数据，下采样为24维，并转换为SNN输入脉冲(简单实现: 距离越近，脉冲频率越高/值越大)"""
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max # 处理无穷大值
        # 假设降采样到 24 根射线
        downsampled = np.linspace(0, len(ranges)-1, 24, dtype=int)
        lidar_data = ranges[downsampled]
        # 归一化并取反，使得距离越近激活越强
        spikes = 1.0 - (lidar_data / msg.range_max) 
        self.obs_uav1['lidar_spikes'] = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)

    def lidar_callback_uav2(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max
        downsampled = np.linspace(0, len(ranges)-1, 24, dtype=int)
        lidar_data = ranges[downsampled]
        spikes = 1.0 - (lidar_data / msg.range_max)
        self.obs_uav2['lidar_spikes'] = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)

    def odom_callback_uav1(self, msg):
        """获取无人机当前速度输入给网格细胞"""
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        self.obs_uav1['velocity'] = torch.tensor([vx, vy, vz], dtype=torch.float32).unsqueeze(0)

    def odom_callback_uav2(self, msg):
        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z
        self.obs_uav2['velocity'] = torch.tensor([vx, vy, vz], dtype=torch.float32).unsqueeze(0)

    # --- 主控制循环 ---
    def control_loop(self):
        while not rospy.is_shutdown():
            # 1. 将当前状态送入模型进行前向推理
            # 注意：此处 env_features (洪涝环境特征) 在物理引擎中无法直接读取，
            # 需结合你定义的 environment.py 和全局坐标系进行补充计算。这里先用假数据占位。
            self.obs_uav1['env_features'] = torch.randn(1, 128) 
            self.obs_uav2['env_features'] = torch.randn(1, 128)

            with torch.no_grad(): # 推理阶段不计算梯度
                action_uav1, action_uav2 = self.model(self.obs_uav1, self.obs_uav2)

            # 2. 将网络输出解析为控制指令 (假设动作维度为 4：vx, vy, vz, yaw_rate)
            cmd1 = self.tensor_to_twist(action_uav1)
            cmd2 = self.tensor_to_twist(action_uav2)

            # 3. 发布指令
            self.cmd_pub_uav1.publish(cmd1)
            self.cmd_pub_uav2.publish(cmd2)

            self.rate.sleep()

    def tensor_to_twist(self, action_tensor):
        """将网络的输出 Tensor 映射到实际的 Twist 速度对象上"""
        action = action_tensor.squeeze(0).numpy()
        twist = Twist()
        # 假设网络输出已经经过了 tanh 等激活函数，映射到 [-1, 1] 区间
        max_speed = 1.5 # 最大速度 1.5 m/s
        max_yaw = 0.5   # 最大偏航角速度 0.5 rad/s
        
        twist.linear.x = action[0] * max_speed
        twist.linear.y = action[1] * max_speed
        twist.linear.z = action[2] * max_speed
        twist.angular.z = action[3] * max_yaw
        return twist

if __name__ == '__main__':
    try:
        node = SwarmRescueROSNode()
        rospy.loginfo("双机协同 SNN 救援控制节点已启动...")
        node.control_loop()
    except rospy.ROSInterruptException:
        pass