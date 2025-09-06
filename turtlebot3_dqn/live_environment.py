#!/usr/bin/env python3
#################################################################################
# Copyright 2019 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#################################################################################
#
# Authors: Ryan Shim, Gilbert, ChanHyeong Lee

import math
import os
import sys
import time

from geometry_msgs.msg import Twist
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
import numpy
import rclpy
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from std_srvs.srv import Empty

from turtlebot3_msgs.srv import Dqn
from turtlebot3_msgs.srv import Goal


ROS_DISTRO = os.environ.get('ROS_DISTRO')


class RLEnvironment(Node):

    def __init__(self, return_pose):
        super().__init__('rl_environment')
        self.goal_pose_x = 2.0
        self.goal_pose_y = 0.0
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0

        self.action_size = 5
        self.max_step = 8000

        self.done = False
        self.fail = False
        self.succeed = False
        self.return_pose = return_pose

        self.goal_tolerance = 0.2 # 0.5
        self.collision_tolerance = 0.15

        self.goal_angle = 0.0
        self.goal_distance = 1.0
        self.init_goal_distance = 0.4
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.is_front_min_actual_front = False

        self.local_step = 0
        self.stop_cmd_vel_timer = None
        self.linear_velocity = 0.05
        self.angular_vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        qos = QoSProfile(depth=10)

        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)
        else:
            self.cmd_vel_pub = self.create_publisher(TwistStamped, 'cmd_vel', qos)

        self.odom_sub = self.create_subscription(
            Odometry,
            'odom',
            self.odom_sub_callback,
            qos
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_sub_callback,
            qos_profile_sensor_data
        )

        self.clients_callback_group = MutuallyExclusiveCallbackGroup()
        self.task_succeed_client = self.create_client(
            Goal,
            'task_succeed',
            callback_group=self.clients_callback_group
        )
        self.task_failed_client = self.create_client(
            Goal,
            'task_failed',
            callback_group=self.clients_callback_group
        )
        self.initialize_environment_client = self.create_client(
            Goal,
            'initialize_env',
            callback_group=self.clients_callback_group
        )

        self.rl_agent_interface_service = self.create_service(
            Dqn,
            'rl_agent_interface',
            self.rl_agent_interface_callback
        )
        self.make_environment_service = self.create_service(
            Empty,
            'make_environment',
            self.make_environment_callback
        )
        self.reset_environment_service = self.create_service(
            Dqn,
            'reset_environment',
            self.reset_environment_callback
        )

    def make_environment_callback(self, request, response):
        # self.get_logger().info('Make environment called')
        # while not self.initialize_environment_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().warn(
        #         'service for initialize the environment is not available, waiting ...'
        #     )
        # future = self.initialize_environment_client.call_async(Goal.Request())
        # rclpy.spin_until_future_complete(self, future)
        # response_goal = future.result()
        # if not response_goal.success:
        #     self.get_logger().error('initialize environment request failed')
        # else:
        #     self.goal_pose_x = response_goal.pose_x
        #     self.goal_pose_y = response_goal.pose_y
        #     self.get_logger().info(
        #         'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
        #     )
        self.get_logger().info('dummy make environment called')
        self.get_logger().info(
            'goal initialized at [%f, %f]' % (self.goal_pose_x, self.goal_pose_y)
        )

        return response

    def reset_environment_callback(self, request, response):
        state = self.calculate_state()
        self.init_goal_distance = state[0]
        self.prev_goal_distance = self.init_goal_distance
        response.state = state
        if self.return_pose:
            response.state.append(self.robot_pose_x)
            response.state.append(self.robot_pose_y)
            response.state.append(self.robot_pose_theta)

        return response

    def call_task_succeed(self):
        self.get_logger().info('dummy task succeed call')
        # while not self.task_succeed_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().warn('service for task succeed is not available, waiting ...')
        # future = self.task_succeed_client.call_async(Goal.Request())
        # rclpy.spin_until_future_complete(self, future)
        # if future.result() is not None:
        #     response = future.result()
        #     self.goal_pose_x = response.pose_x
        #     self.goal_pose_y = response.pose_y
        #     self.get_logger().info('service for task succeed finished')
        # else:
        #     self.get_logger().error('task succeed service call failed')

    def call_task_failed(self):
        self.get_logger().info('dummy task failed call')
        # while not self.task_failed_client.wait_for_service(timeout_sec=1.0):
        #     self.get_logger().warn('service for task failed is not available, waiting ...')
        # future = self.task_failed_client.call_async(Goal.Request())
        # rclpy.spin_until_future_complete(self, future)
        # if future.result() is not None:
        #     response = future.result()
        #     self.goal_pose_x = response.pose_x
        #     self.goal_pose_y = response.pose_y
        #     self.get_logger().info('service for task failed finished')
        # else:
        #     self.get_logger().error('task failed service call failed')

    def scan_sub_callback(self, scan):
        self.scan_ranges = []
        self.front_ranges = []
        self.front_angles = []

        num_of_lidar_rays = len(scan.ranges)
        angle_min = scan.angle_min
        angle_increment = scan.angle_increment

        self.front_distance = scan.ranges[0]

        for i in range(num_of_lidar_rays):
            angle = angle_min + i * angle_increment
            distance = scan.ranges[i]
            if distance == -float('Inf'):
                # print(f"{i} / F{num_of_lidar_rays}, {distance}")
                distance = 0.0
            if distance == float('Inf'):
                distance = 3.5
            elif numpy.isnan(distance):
                distance = 0.0

            self.scan_ranges.append(distance)

            if (0 <= angle <= math.pi/2) or (3*math.pi/2 <= angle <= 2*math.pi):
                self.front_ranges.append(distance)
                self.front_angles.append(angle)

        self.min_obstacle_distance = min(self.scan_ranges)
        self.front_min_obstacle_distance = min(self.front_ranges) if self.front_ranges else 10.0

    def odom_sub_callback(self, msg):
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi

        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def calculate_state(self):
        state = []
        state.append(float(self.goal_distance))
        state.append(float(self.goal_angle))
        for var in self.front_ranges:
            state.append(float(var))
        self.local_step += 1

        if self.goal_distance < self.goal_tolerance:
            self.get_logger().info('Goal Reached')
            self.succeed = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_succeed()

        if self.min_obstacle_distance < self.collision_tolerance:
            self.get_logger().info('Collision happened')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        if self.local_step == self.max_step:
            self.get_logger().info('Time out!')
            self.fail = True
            self.done = True
            if ROS_DISTRO == 'humble':
                self.cmd_vel_pub.publish(Twist())
            else:
                self.cmd_vel_pub.publish(TwistStamped())
            self.local_step = 0
            self.call_task_failed()

        return state

    def compute_directional_weights(self, relative_angles, max_weight=10.0):
        power = 4
        raw_weights = (numpy.cos(relative_angles))**power + 0.1
        scaled_weights = raw_weights * (max_weight / numpy.max(raw_weights))
        normalized_weights = scaled_weights / numpy.sum(scaled_weights)
        return normalized_weights

    def compute_weighted_obstacle_reward(self):
        if not self.front_ranges or not self.front_angles:
            return 0.0

        front_ranges = numpy.array(self.front_ranges)
        front_angles = numpy.array(self.front_angles)

        valid_mask = front_ranges <= 0.5
        if not numpy.any(valid_mask):
            return 0.0

        front_ranges = front_ranges[valid_mask]
        front_angles = front_angles[valid_mask]

        relative_angles = numpy.unwrap(front_angles)
        relative_angles[relative_angles > numpy.pi] -= 2 * numpy.pi

        weights = self.compute_directional_weights(relative_angles, max_weight=10.0)

        safe_dists = numpy.clip(front_ranges - 0.25, 1e-2, 3.5)
        decay = numpy.exp(-3.0 * safe_dists)

        weighted_decay = numpy.dot(weights, decay)

        # reward = - (1.0 + 4.0 * weighted_decay)
        reward = - weighted_decay

        return reward

    def calculate_reward(self):
        yaw_reward = - abs(self.goal_angle / math.pi / 2)
        dist_reward = 0 # -abs(self.goal_distance) / 3.5 # note 3.5 is max radar dist
        obstacle_reward = self.compute_weighted_obstacle_reward()
        info_str = f"directional_reward: {yaw_reward:.3f}, goal_dist: {abs(self.goal_distance):.3f}, obstacle_reward: {obstacle_reward:.3f}"
        self.get_logger().info(info_str)
        reward = yaw_reward + dist_reward + obstacle_reward

        if self.succeed:
            reward += 100.0
        elif self.fail:
            reward += -50.0

        return reward

    def rl_agent_interface_callback(self, request, response):
        action = request.action
        if ROS_DISTRO == 'humble':
            msg = Twist()
            msg.linear.x = self.linear_velocity
            msg.angular.z = self.angular_vel[action]
        else:
            msg = TwistStamped()
            msg.twist.linear.x = self.linear_velocity
            msg.twist.angular.z = self.angular_vel[action]

        self.cmd_vel_pub.publish(msg)
        if self.stop_cmd_vel_timer is None:
            self.prev_goal_distance = self.init_goal_distance
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        else:
            self.destroy_timer(self.stop_cmd_vel_timer)
            self.stop_cmd_vel_timer = self.create_timer(0.8, self.timer_callback)
        response.state = self.calculate_state()
        response.reward = self.calculate_reward()
        response.done = self.done
        if self.return_pose:
            response.state.append(self.robot_pose_x)
            response.state.append(self.robot_pose_y)
            response.state.append(self.robot_pose_theta)

        if self.done is True:
            self.done = False
            self.succeed = False
            self.fail = False
            self.goal_tolerance = max(0.2, self.goal_tolerance - 0.001)
            self.get_logger().info(f"goal_tolerance update to {self.goal_tolerance}")
        return response

    def timer_callback(self):
        self.get_logger().info('Stop called')
        if ROS_DISTRO == 'humble':
            self.cmd_vel_pub.publish(Twist())
        else:
            self.cmd_vel_pub.publish(TwistStamped())
        self.destroy_timer(self.stop_cmd_vel_timer)

    def euler_from_quaternion(self, quat):
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = numpy.arctan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = numpy.arcsin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = numpy.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw


def main(args=None):
    if args is None:
        args = sys.argv
    return_pose = args[1] if len(args) > 1 else '0'
    return_pose = int(return_pose) == 1
    print("return_pose:", return_pose)
    rclpy.init(args=args)
    rl_environment = RLEnvironment(return_pose)
    try:
        while rclpy.ok():
            rclpy.spin_once(rl_environment, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        rl_environment.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
