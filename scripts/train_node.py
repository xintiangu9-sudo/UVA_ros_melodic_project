#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import torch
import torch.optim as optim
import numpy as np

from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from std_srvs.srv import Empty

from torch.utils.tensorboard import SummaryWriter

# ✅ 正确导入
from collaborative_framework import DualUAVSystem
from environment import FloodRescueEnvironment


class SNNTrainerNode:
    def __init__(self):
        rospy.init_node('snn_online_trainer', anonymous=True)

        # ===== 模型 =====
        self.model = DualUAVSystem()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4)
        self.env_logic = FloodRescueEnvironment()

        self.writer = SummaryWriter(
            log_dir='/home/guxintian/UVA/src/swarm_rescue/runs/snn_training'
        )

        # ===== 状态（初始化完整）=====
        self.obs_uav1 = {
            'velocity': torch.zeros(1, 3),
            'lidar_spikes': torch.zeros(1, 24),
            'env_features': torch.zeros(1, 128)
        }

        self.obs_uav2 = {
            'velocity': torch.zeros(1, 3),
            'lidar_spikes': torch.zeros(1, 24),
            'env_features': torch.zeros(1, 128)
        }

        self.ready_uav1 = False
        self.ready_uav2 = False

        self.uav1_pos = [0.0, 0.0]
        self.uav2_pos = [0.0, 0.0]

        self.min_laser_dist1 = 10.0
        self.min_laser_dist2 = 10.0

        # ===== ROS 订阅 =====
        rospy.Subscriber('/uav1/scan', LaserScan, self.lidar_cb_1)
        rospy.Subscriber('/uav1/ground_truth/state', Odometry, self.odom_cb_1)

        rospy.Subscriber('/uav2/scan', LaserScan, self.lidar_cb_2)
        rospy.Subscriber('/uav2/ground_truth/state', Odometry, self.odom_cb_2)

        # ===== 发布 =====
        self.cmd_pub_1 = rospy.Publisher('/uav1/cmd_vel', Twist, queue_size=1)
        self.cmd_pub_2 = rospy.Publisher('/uav2/cmd_vel', Twist, queue_size=1)

        # ===== Gazebo 服务 =====
        rospy.loginfo("等待 Gazebo 服务...")
        rospy.wait_for_service('/gazebo/reset_world')
        self.reset_gazebo = rospy.ServiceProxy('/gazebo/reset_world', Empty)

        self.control_rate = rospy.Rate(5)

    # ================= Lidar =================
    def lidar_cb_1(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max

        self.min_laser_dist1 = np.min(ranges)

        downsampled = ranges[np.linspace(0, len(ranges) - 1, 24, dtype=int)]
        spikes = 1.0 - (downsampled / msg.range_max)

        self.obs_uav1['lidar_spikes'] = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)
        self.ready_uav1 = True

    def lidar_cb_2(self, msg):
        ranges = np.array(msg.ranges)
        ranges[np.isinf(ranges)] = msg.range_max

        self.min_laser_dist2 = np.min(ranges)

        downsampled = ranges[np.linspace(0, len(ranges) - 1, 24, dtype=int)]
        spikes = 1.0 - (downsampled / msg.range_max)

        self.obs_uav2['lidar_spikes'] = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)
        self.ready_uav2 = True

    # ================= Odometry =================
    def odom_cb_1(self, msg):
        self.uav1_pos = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        self.obs_uav1['velocity'] = torch.tensor(
            [vx, vy, vz], dtype=torch.float32
        ).unsqueeze(0)

    def odom_cb_2(self, msg):
        self.uav2_pos = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y
        ]

        vx = msg.twist.twist.linear.x
        vy = msg.twist.twist.linear.y
        vz = msg.twist.twist.linear.z

        self.obs_uav2['velocity'] = torch.tensor(
            [vx, vy, vz], dtype=torch.float32
        ).unsqueeze(0)

    # ================= 控制 =================
    def publish_action(self, a1, a2):
        cmd1 = Twist()
        cmd2 = Twist()

        # UAV1
        cmd1.linear.x = a1[0]
        cmd1.linear.y = a1[1]
        cmd1.linear.z = a1[2] * 0.2 + 0.6
        cmd1.angular.z = a1[3] * 0.5

        # UAV2
        cmd2.linear.x = a2[0]
        cmd2.linear.y = a2[1]
        cmd2.linear.z = a2[2] * 0.2 + 0.6
        cmd2.angular.z = a2[3] * 0.5

        self.cmd_pub_1.publish(cmd1)
        self.cmd_pub_2.publish(cmd2)

    def check_collision(self):
        return (self.min_laser_dist1 < 0.3 or
                self.min_laser_dist2 < 0.3)

    # ================= 训练主循环 =================
    def train_loop(self):
        rospy.loginfo("开始训练...")

        while not rospy.is_shutdown():

            # ✅ 关键修复：等待传感器
            if not (self.ready_uav1 and self.ready_uav2):
                rospy.logwarn("等待传感器数据...")
                rospy.sleep(0.5)
                continue

            # 随机环境特征（占位）
            self.obs_uav1['env_features'] = torch.randn(1, 128)
            self.obs_uav2['env_features'] = torch.randn(1, 128)

            # ===== 模型前向 =====
            action1, action2 = self.model(self.obs_uav1, self.obs_uav2)

            action1 = action1.squeeze().detach().numpy()
            action2 = action2.squeeze().detach().numpy()

            # ===== 执行控制 =====
            self.publish_action(action1, action2)

            # ===== 奖励 =====
            r1, r2 = self.env_logic.calculate_reward(
                self.uav1_pos,
                self.uav2_pos,
                action1,
                action2,
                False
            )

            reward = r1 + r2

            self.writer.add_scalar("reward", reward)

            if self.check_collision():
                rospy.logwarn("发生碰撞！")

            self.control_rate.sleep()


if __name__ == '__main__':
    try:
        node = SNNTrainerNode()
        node.train_loop()
    except rospy.ROSInterruptException:
        pass