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

import collections
import datetime
import json
import math
import os
import random
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

from turtlebot3_msgs.srv import Dqn
import wandb

LOGGING = True
current_time = datetime.datetime.now()
# start_time = datetime.datetime.now()

"""
Note, must use tensorflow 2.18, version 2.20 results in a segfault without any warning...
"""

class DQNAgent():

    def __init__(self, stage_num, max_training_episodes):
        super().__init__()

        self.stage = int(stage_num)
        self.train_mode = True
        self.wandb = True
        self.state_size = 26 # 180+2 corresponds to 360 samples, originally 26
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)
        self.wandb_project_name: str = "DQN_turtlebot3"
        self.network_type = "FCNet"

        self.done = False
        self.succeed = False
        self.fail = False

        self.discount_factor = 0.999
        self.learning_rate = 0.005
        self.lr_decay_step = 50
        self.lr_decay_rate = 0.5
        self.epsilon = 0.5 # 1.0
        self.tau = 1.0 # target model update
        self.step_counter = 0
        self.epsilon_decay = 30000 # 6000 * self.stage
        self.epsilon_min = 0.05
        self.batch_size = 512
        self.memory_size = 500000
        self.device = torch.device("cuda")
        self.global_step = 0
        self.max_lidar_range = 3.5 # taken from the model sdf file, modify as needed!

        self.replay_memory = NumpyReplayBuffer(max_size=self.memory_size, batch_size=self.batch_size)
        self.min_replay_memory_size = 5000

        self.run_name = f"stage{self.stage}__{self.learning_rate}__{self.batch_size}__{current_time.strftime('%m%d%y_%H%M')}"
        config_copy = {
            "discount_factor"   :   self.discount_factor,
            "learning_rate"     :   self.learning_rate,
            "lr_decay_step"     :   self.lr_decay_step,
            "lr_decay_rate"     :   self.lr_decay_rate,
            "epsilon"           :   self.epsilon,
            "stage"             :   self.stage,
            "epsilon_decay"     :   self.epsilon_decay,
            "epsilon_min"       :   self.epsilon_min,
            "batch_size"        :   self.batch_size,
            "memory_size"       :   self.memory_size,
            "network"           :   self.network_type,
        }
        if self.wandb:
            self.run = wandb.init(
                entity="gazebo-rl",
                project=self.wandb_project_name,
                config=config_copy,
                name=self.run_name,
                save_code=True,
            )
        if self.network_type == "FCNet":
            self.q_network = FCNet(self.state_size, self.action_size).to(self.device)
            self.target_network = FCNet(self.state_size, self.action_size).to(self.device)
        elif self.network_type == "CNN_net":
            self.q_network = CNN_net(self.state_size, self.action_size).to(self.device)
            self.target_network = CNN_net(self.state_size, self.action_size).to(self.device)
        else:
            print("Error, network_type unrecognized, exiting...")
            exit()
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.lr_decay_step, gamma=self.lr_decay_rate)
        self.update_target_after = 1000
        self.target_update_after_counter = 0

        self.load_model = False
        self.load_episode = 0
        self.model_dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
            'saved_model'
        )
        self.model_path = os.path.join(
            self.model_dir_path,
            "stage1_episode200.h5"
        )

        if self.load_model:
            checkpoint = torch.load(self.model_path)
            self.q_network.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.load_episode = checkpoint['episode']
            self.epsilon = checkpoint['epsilon']
            self.global_step = checkpoint['global_step']
            self.step_counter = checkpoint['step_counter']
            print(f"model loaded from {self.model_path}")
        self.node = None

    def log(self, info:dict):
        if self.wandb:
            self.run.log(info, self.global_step)


    def get_action(self, state):
        if self.train_mode:
            lucky = random.random()
            if lucky > (1 - self.epsilon):
                result = random.randint(0, self.action_size - 1)
            else:
                q_values = self.q_network.forward(state)
                result = torch.argmax(q_values, dim=-1).cpu().numpy()[0][0]
        else:
            q_values = self.q_network.forward(state)
            result = torch.argmax(q_values, dim=-1).cpu().numpy()[0][0]

        return result

    def update_target_network(self):
        for target_network_param, q_network_param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_network_param.data.copy_(
                self.tau * q_network_param.data + (1.0 - self.tau) * target_network_param.data
            )
        print('*Target model updated*')

    def train_model(self, terminal):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return None

        experiences = self.replay_memory.sample(self.batch_size)
        states, actions, rewards, new_states, is_dones = experiences
        states = torch.from_numpy(states).float().to(self.device)  # [128, 1, 182]
        actions = torch.from_numpy(actions).float().to(torch.int32).to(self.device) # [128, 1]
        new_states = torch.from_numpy(new_states).float().to(self.device)  # [128, 1, 182]
        rewards = torch.from_numpy(rewards).float().to(self.device) # [128, 1]
        is_dones = torch.from_numpy(is_dones).float().to(self.device) # [128, 1]

        with torch.no_grad():
            target_max, target_max_indices = self.target_network(new_states).max(dim=-1)
            td_target = rewards.flatten() + self.discount_factor * target_max.flatten() * (1 - is_dones.flatten())
        old_val = self.q_network(states).squeeze(1).gather(1, actions).squeeze()
        loss = F.mse_loss(td_target, old_val)  # both [128]
        if np.isnan(np.array([loss.item()])):
            print(f"nan detected in loss {loss}")
            print(f"buffer size {td_target.shape}")
            print(f"td_target {self.replay_memory.size}")
            print(f"old_val {old_val.shape}")
            print("diff")


        # optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.target_update_after_counter += 1

        if self.target_update_after_counter > self.update_target_after and terminal:
            self.update_target_network()
        return loss.item()

class RLNode(Node):
    def __init__(self, dqn_agent:DQNAgent, time_threshold=0.2):
        super().__init__('RL_node')
        self.agent = dqn_agent
        self.time_threshold = time_threshold
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')
        # self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        # self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)
        self.rl_process()

    def rl_process(self):
        self.env_make()
        time.sleep(1.0)
        episode_num = self.agent.load_episode
        step_duration_array = np.zeros(10)

        for episode in range(self.agent.load_episode + 1, self.agent.max_training_episodes + 1):
            episode_start = time.time()
            state = self.reset_environment()
            state = np.expand_dims(state, axis=1)  # manually inject a 'channel' dim
            state_tensor = torch.Tensor(state).to(self.agent.device)  # manually inject a 'channel' dim
            episode_num += 1
            local_step = 0
            score = 0

            time.sleep(1.0)

            while True:
                step_start = time.time()
                local_step += 1
                self.agent.global_step += 1

                action = int(self.agent.get_action(state_tensor))
                next_state, reward, done = self.step(action)
                next_state = np.expand_dims(next_state, axis=1)
                score += reward

                if self.agent.train_mode:
                    self.agent.replay_memory.store((state, action, reward, next_state, done))
                    local_loss = self.agent.train_model(done)
                    self.agent.log({"reward": reward})
                    if local_loss is not None:
                        self.agent.log({"mse_loss": local_loss})
                        # updating epsilon values only after min_replay_memory_size filled
                        self.agent.step_counter += 1
                        self.agent.epsilon = self.agent.epsilon_min + (1.0 - self.agent.epsilon_min) * math.exp(
                            -1.0 * self.agent.step_counter / self.agent.epsilon_decay)
                state = next_state
                step_duration = time.time() - step_start
                step_duration_array[local_step % 10] = step_duration

                self.agent.log({
                    "timer/step_duration": step_duration,
                })
                if done:
                    episode_dict = {
                        'Episode:': episode_num,
                        'score:': score,
                        'memory length:': self.agent.replay_memory.size,
                        'epsilon:': self.agent.epsilon,
                        'lr': self.agent.scheduler.get_last_lr()[-1],
                        "timer/episode_duration": time.time() - episode_start,
                    }
                    self.agent.log(episode_dict)
                    if local_loss is not None:
                        print(
                            f"Episode {episode_num} step {self.agent.global_step} total score: {score:.3f}, loss {local_loss}")
                        self.agent.scheduler.step()
                    else:
                        print(f"Episode {episode_num} step {self.agent.global_step} total score: {score:.3f}")
                    break

                time.sleep(0.01)

            if self.agent.train_mode:
                if episode % 100 == 0:
                    model_path = os.path.join(
                        self.agent.model_dir_path,
                        'dqn_stage' + str(self.agent.stage) + '_episode' + str(episode) + '.h5')
                    torch.save({
                        'episode': episode,
                        'model_state_dict': self.agent.q_network.state_dict(),
                        'optimizer_state_dict': self.agent.optimizer.state_dict(),
                        'epsilon': self.agent.epsilon,
                        'global_step': self.agent.global_step,
                        'step_counter': self.agent.step_counter,
                    }, model_path)
                    print(f"model saved to {model_path}")
            if np.median(step_duration_array) >= self.time_threshold:
                warning_msg = (f"median step duration {np.median(step_duration_array):.3f} "
                               f"greater than threshold {self.time_threshold}, shutting down node...")
                self.agent.load_episode = episode_num
                self.get_logger().warn(warning_msg)
                break

    def env_make(self):
        while not self.make_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Environment make client failed to connect to the server, try again ...'
            )

        self.make_environment_client.call_async(Empty.Request())

    def reset_environment(self):
        while not self.reset_environment_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn(
                'Reset environment client failed to connect to the server, try again ...'
            )

        future = self.reset_environment_client.call_async(Dqn.Request())

        rclpy.spin_until_future_complete(self, future)
        if future.result() is not None:
            state = np.asarray(future.result().state)
            state = np.reshape(state, [1, -1])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state

    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')
        step_start_time = time.time()
        future = self.rl_agent_interface_client.call_async(req)
        future_call_async = time.time()
        rclpy.spin_until_future_complete(self, future)
        future_complete = time.time()
        if future.result() is not None:
            next_state = np.asarray(future.result().state)
            next_state = np.reshape(next_state, [1, -1])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))
        print_string = (f"async time: {future_call_async - step_start_time:.6f},"
              f"future complete time: {future_complete - future_call_async:.6f},"
              f"future result time: {time.time() - future_complete:.6f},"
              f"plugin_response_time: {time.time() - step_start_time:.6f}")
        # print(print_string)
        log_dict = {
            "timer/async_time": future_call_async - step_start_time,
            "timer/future_complete_time": future_complete - future_call_async,
            "timer/future_result_time": time.time() - future_complete,
            "timer/plugin_response_time": time.time() - step_start_time,
        }
        self.agent.log(log_dict)
        return next_state, reward, done

class FCNet(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions),
        )

    def forward(self, x):
        return self.network(x)


class CNN_net(nn.Module):
    def __init__(self, state_size, n_actions):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 4, 5, 3)
        self.pool1 = nn.MaxPool1d(3, 3)
        self.conv2 = nn.Conv1d(4, 16, 3, 2)
        self.pool2 = nn.MaxPool1d(2, 2)
        self.fc = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, n_actions),
        )

    def forward(self, x): # [batch, 1, 182]
        dist_and_angle = x[:, :, 0:2].squeeze(1),
        x = x[:, :, 2:]
        x = self.pool1(F.relu(self.conv1(x))) # [batch, 4, 60] -> pool1 -> # [batch, 4, 20]
        x = self.pool2(F.relu(self.conv2(x))) # [batch, 16, 9] -> pool2 -> # [batch, 16, 4]
        x = torch.flatten(x, 1) # [batch, 64]
        x = torch.cat((dist_and_angle, x), dim=0)
        torch.cat((x, x, x), 0)
        x = self.fc(x)
        return x

class NumpyReplayBuffer(object):
    def __init__(self,
                 max_size=10000,
                 batch_size=64):
        self.ss_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.as_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.rs_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.ps_mem = np.empty(shape=max_size, dtype=np.ndarray)
        self.ds_mem = np.empty(shape=max_size, dtype=np.ndarray)

        self.max_size = max_size
        self.batch_size = batch_size
        self._idx = 0
        self.size = 0

    def store(self, sample):
        s, a, r, p, d = sample
        self.ss_mem[self._idx] = s
        self.as_mem[self._idx] = a
        self.rs_mem[self._idx] = r
        self.ps_mem[self._idx] = p
        self.ds_mem[self._idx] = d

        self._idx += 1
        self._idx = self._idx % self.max_size

        self.size += 1
        self.size = min(self.size, self.max_size)

    def sample(self, batch_size=None, idxs=None):
        if batch_size is None:
            batch_size = self.batch_size
        if idxs is None:
            idxs = np.random.choice(self.size, batch_size, replace=False)
        experiences = np.vstack(self.ss_mem[idxs]), \
            np.vstack(self.as_mem[idxs]), \
            np.vstack(self.rs_mem[idxs]), \
            np.vstack(self.ps_mem[idxs]), \
            np.vstack(self.ds_mem[idxs])
        return experiences

    def __len__(self):
        return self.size

def main(args=None):
    """
    dqn_agent: has network, replay buffer, optimizer, logger
    dqn_node: contains each iteration's training, and we pass in the dqn_agent each time.
    """
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'

    dqn_agent = DQNAgent(stage_num, max_training_episodes)
    while dqn_agent.load_episode < dqn_agent.max_training_episodes + 1:
        print("starting rclpy node...")
        rclpy.init()
        rl_node = RLNode(dqn_agent, time_threshold=0.2)
        # rclpy.spin(rl_node)
        rl_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
