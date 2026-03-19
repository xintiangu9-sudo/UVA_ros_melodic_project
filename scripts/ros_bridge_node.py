\#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import socket
import json
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped


class ROSBridgeNode:
    def __init__(self):
        rospy.init_node('ros_bridge_node')

        rospy.loginfo("等待 /clock...")
        rospy.wait_for_message("/clock", rospy.AnyMsg)

        rospy.loginfo("等待 /uav1/scan...")
        rospy.wait_for_message("/uav1/scan", LaserScan)

        rospy.loginfo("等待 /uav1/state...")
        rospy.wait_for_message("/uav1/ground_truth/state", Odometry)

        # socket
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect(("127.0.0.1", 9999))

        self.scan = None
        self.state = None

        rospy.Subscriber("/uav1/scan", LaserScan, self.scan_callback)
        rospy.Subscriber("/uav1/ground_truth/state", Odometry, self.state_callback)

        # ✅ 注意这里：TwistStamped
        self.cmd_pub = rospy.Publisher("/uav1/command/twist", TwistStamped, queue_size=10)

        rospy.loginfo("ROS Bridge Node Started")

    def scan_callback(self, msg):
        self.scan = list(msg.ranges[:10])

    def state_callback(self, msg):
        self.state = {
            "x": msg.pose.pose.position.x,
            "y": msg.pose.pose.position.y
        }

    def send_and_receive(self):
        if self.scan is None or self.state is None:
            rospy.logwarn("等待传感器数据...")
            return

        data = {
            "scan": self.scan,
            "state": self.state
        }

        try:
            self.sock.sendall((json.dumps(data) + "\n").encode())

            response = self.sock.recv(1024).decode()
            action = json.loads(response.strip())

            # ✅ 构造 TwistStamped
            twist_msg = TwistStamped()
            twist_msg.header.stamp = rospy.Time.now()

            twist_msg.twist.linear.x = action.get("linear", 0.0)
            twist_msg.twist.linear.z = 0.5   # 稍微给点升力
            twist_msg.twist.angular.z = action.get("angular", 0.0)

            self.cmd_pub.publish(twist_msg)

        except Exception as e:
            rospy.logerr("Socket error: {}".format(e))

    def run(self):
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            self.send_and_receive()
            rate.sleep()


if __name__ == "__main__":
    node = ROSBridgeNode()
    node.run()
