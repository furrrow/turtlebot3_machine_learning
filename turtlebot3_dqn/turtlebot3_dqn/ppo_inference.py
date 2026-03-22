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
# adapted from the cleanrl PPO code
# https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy

import datetime
import pathlib
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Empty

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt

from turtlebot3_msgs.srv import Dqn
from turtlebot3_dqn.ppo_agent import PPOAgent, RLNode
import h5py
import wandb

LOGGING = True
current_time = datetime.datetime.now()

"""
Note, must use tensorflow 2.18, version 2.20 results in a segfault without any warning...
"""

def append_to_dataset(h5db, label, data, data_length=None):
    if data_length is None:
        data_length = len(data)
    original_length = h5db[label].shape[0]
    new_length = original_length + data_length
    h5db[label].resize(new_length, axis=0)
    h5db[label][original_length:new_length] = data

class InferenceNode(RLNode):
    def __init__(self, ppo_agent: PPOAgent, h5_file, num_episodes, time_threshold=0.2, save_traces=False, verbose=True):
        super().__init__(ppo_agent, time_threshold, spawn_process=False)
        self.plot_scans = True
        self.h5_file = h5_file
        self.num_episodes = num_episodes
        self.save_traces = save_traces
        self.verbose = verbose
        self.rl_process()

    def rl_process(self):
        self.env_make()
        time.sleep(1.0)
        episode_start = time.time()
        episode_reward = 0
        step_duration_array = np.zeros(10)
        episode_scores = []

        current_episode = self.h5_file.attrs['num_episodes']
        state = self.reset_environment()
        if self.save_traces:
            x_y_th = state[0][-3:]
            state = state[:, :-3]
        state = self.agent.reduce_state(state)
        if self.save_traces:
            append_to_dataset(self.h5_file, "state", state)
            append_to_dataset(self.h5_file, "dones", 0, 1)
        state = np.expand_dims(state, axis=1)  # manually inject a 'channel' dim
        next_obs = torch.Tensor(state).to(self.agent.device)  # manually inject a 'channel' dim
        next_done = torch.zeros(self.agent.num_envs).to(self.agent.device)
        time.sleep(1.0)

        local_step = 0
        while current_episode < self.num_episodes:
            if self.plot_scans:
                # fig, ax = plt.subplots(subplot_kw=dict(polar=True))
                fig, ax = plt.subplots(figsize=(8,4))
            while True:
                step_start = time.time()
                self.agent.global_step += self.agent.num_envs

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    # action, logprob, _, value, explore = self.agent.network.get_action_and_value(next_obs)
                    action = self.agent.network.get_greedy_action(next_obs)

                # execute the game and log data.
                next_obs, reward, next_done, x_y_th = self.step(action.item())
                next_obs = self.agent.reduce_state(next_obs)
                if self.plot_scans:
                    goal, heading = next_obs[0][0], next_obs[0][1]
                    scans = next_obs[0, 2:]
                    spacing = (np.pi / 2 - 0) / (len(scans) // 2 + 1) / 2
                    front_angles = np.linspace(spacing, np.pi / 2 - spacing, len(scans) // 2)
                    front_angles = np.concatenate((front_angles , np.linspace(-np.pi/2 + spacing, -spacing,  len(scans) // 2)))
                    front_angles = np.rad2deg(front_angles)
                    # ax.set_theta_offset(np.pi / 2)
                    ax.set_ylim(0, 5)
                    ax.invert_xaxis()
                    # bars = ax.bar(front_angles, scans[:len(scans)//2])
                    ax.scatter(front_angles, scans, s=10)
                    ax.plot(front_angles[len(scans)//2:], scans[len(scans)//2:])
                    ax.plot(front_angles[:len(scans)//2], scans[:len(scans)//2])
                    plt.pause(0.001)
                    plt.cla()
                if self.save_traces:
                    append_to_dataset(self.h5_file, "x", x_y_th[0], 1)
                    append_to_dataset(self.h5_file, "y", x_y_th[1], 1)
                    append_to_dataset(self.h5_file, "theta", x_y_th[2], 1)
                    append_to_dataset(self.h5_file, "dt", time.time() - episode_start, 1)
                    append_to_dataset(self.h5_file, "step", local_step, 1)
                    append_to_dataset(self.h5_file, "action", action.item(), 1)
                    append_to_dataset(self.h5_file, "reward", reward, 1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.agent.device), torch.Tensor([next_done]).to(self.agent.device)
                episode_reward += reward
                local_step += 1

                time.sleep(0.01)
                step_duration = time.time() - step_start
                step_duration_array[self.agent.global_step % 10] = step_duration

                if next_done:
                    if local_step < 3:
                        print(f"too few steps, skipping ->: "
                              f"episode {current_episode} score {episode_reward:.3f} in {local_step} steps")
                        continue
                    self.agent.log({"score": episode_reward, "episode_steps": local_step})
                    if self.verbose:
                        print(f"episode {current_episode} final reward {reward:.3f} score {episode_reward:.3f} in {local_step} steps")
                    if reward > 90:
                        self.agent.success_count += 1
                    elif reward < -40:
                        self.agent.failure_count += 1
                    episode_scores.append(episode_reward)
                    current_episode += 1
                    episode_reward = 0
                    self.h5_file.attrs['num_episodes'] = current_episode
                    state = self.reset_environment()
                    episode_start = time.time()
                    local_step = 0
                    if self.save_traces:
                        x_y_th = state[0][-3:]
                        state = state[:, :-3]
                    state = self.agent.reduce_state(state)
                    if self.save_traces:
                        append_to_dataset(self.h5_file, "next_state", state)
                    state = np.expand_dims(state, axis=1)
                    next_obs = torch.Tensor(state).to(self.agent.device)
                    next_done = torch.zeros(self.agent.num_envs).to(self.agent.device)
                    time.sleep(1.0)
                    if self.plot_scans:
                        plt.clf()
                        plt.close()
                    break

            if np.median(step_duration_array) >= self.time_threshold:
                warning_msg = (f"episode_num {current_episode} median step duration {np.median(step_duration_array):.3f} "
                               f"greater than threshold {self.time_threshold}, shutting down node...")
                self.get_logger().warn(warning_msg)
                break
        return

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')
        future = self.rl_agent_interface_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state_and_pose = np.asarray(future.result().state)
            next_state = state_and_pose[:-3]
            pose_info = state_and_pose[-3:]
            next_state = np.reshape(next_state, [1, -1])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
        return next_state, reward, done, pose_info


def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '3'
    num_episodes = args[2] if len(args) > 2 else '1'
    save_traces = args[3] if len(args) > 3 else '1'

    verbose = True
    save_traces = int(save_traces) == 1
    num_episodes = int(num_episodes)
    ppo_agent = PPOAgent(stage_num, use_wandb=False, make_save_folder=False)
    model_list = [
        # "/home/jim/turtlebot3_ws/src/turtlebot3_machine_learning/saved_model/stage2__0.0025__2056__082625_0027/ppo_stage2_episode533.h5",
        "/home/jim/turtlebot3_ws/src/turtlebot3_machine_learning/saved_model/stage3__0.0003__2056__091125_2148/ppo_stage3_episode101.h5",
    ]

    for model_path in model_list:
        ppo_agent.load_checkpoint(model_path)
        ppo_agent.global_step = 0
        episode = 0
        ppo_agent.success_count = 0
        ppo_agent.failure_count = 0
        # dataset for trajectories & actions
        if verbose:
            print(f"save traces:{save_traces}" )
        h5_name = f"{os.path.split(model_path)[-1][:-3]}_traces"
        h5_name = f"{h5_name}.hdf5"
        if os.path.exists(h5_name):
            if verbose:
                print(f"{h5_name} exists, removing...")
            os.remove(f"{h5_name}")
        traj_h5 = h5py.File(f"{h5_name}", 'w')
        traj_h5.create_dataset("state", shape=(0, ppo_agent.state_size), maxshape=(None, ppo_agent.state_size), dtype=np.float32)
        traj_h5.create_dataset("next_state", shape=(0, ppo_agent.state_size), maxshape=(None, ppo_agent.state_size), dtype=np.float32)
        traj_h5.create_dataset("action", shape=(0, 1), maxshape=(None, 1), dtype=np.uint8)
        traj_h5.create_dataset("reward", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        traj_h5.create_dataset("dones", shape=(0, 1), maxshape=(None, 1), dtype=np.uint8)
        traj_h5.create_dataset("x", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        traj_h5.create_dataset("y", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        traj_h5.create_dataset("theta", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        traj_h5.create_dataset("dt", shape=(0, 1), maxshape=(None, 1), dtype=np.float32)
        traj_h5.create_dataset("step", shape=(0, 1), maxshape=(None, 1), dtype=np.uint64)
        # adding some metadata
        traj_h5.attrs['model_path'] = model_path
        traj_h5.attrs['num_episodes'] = 0

        while episode < num_episodes:
            if verbose:
                print("starting rclpy node...")
            rclpy.init()
            rl_node = InferenceNode(ppo_agent, traj_h5, num_episodes, time_threshold=0.12, save_traces=save_traces, verbose=verbose)
            # rclpy.spin(rl_node)
            rl_node.destroy_node()
            rclpy.shutdown()
            episode = traj_h5.attrs['num_episodes']
        print(f"success count: {ppo_agent.success_count} failure count {ppo_agent.failure_count} success rate {ppo_agent.success_count/num_episodes:.2f}")


if __name__ == '__main__':
    main()
