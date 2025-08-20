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
from torch.distributions.categorical import Categorical

from turtlebot3_msgs.srv import Dqn
import wandb

LOGGING = True
current_time = datetime.datetime.now()
# start_time = datetime.datetime.now()

"""
Note, must use tensorflow 2.18, version 2.20 results in a segfault without any warning...
"""

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(Node):

    def __init__(self, stage_num, max_training_episodes):
        super().__init__('ppo_agent')

        self.stage = int(stage_num)
        self.train_mode = True
        self.state_size = 26 # 180+2 corresponds to 360 samples, originally 26
        self.action_size = 5
        self.max_training_episodes = int(max_training_episodes)
        self.wandb_project_name: str = "PPO_turtlebot3"

        self.done = False
        self.succeed = False
        self.fail = False

        self.total_timesteps: int = 500000
        self.learning_rate: float = 2.5e-3
        self.num_envs: int = 1
        self.num_steps: int = 1024
        self.anneal_lr: bool = True
        self.gamma: float = 0.99
        self.gae_lambda: float = 0.95
        self.num_minibatches: int = 4
        self.update_epochs: int = 4
        self.norm_adv: bool = True
        self.clip_coef: float = 0.2
        self.clip_vloss: bool = True
        self.ent_coef: float = 0.01
        self.vf_coef: float = 0.5
        self.max_grad_norm: float = 0.5
        self.target_kl: float = None
        # to be filled in runtime
        self.batch_size: int = 0
        self.minibatch_size: int = 0
        self.num_iterations: int = 0

        self.device = torch.device("cuda")
        self.global_step = 0
        self.max_lidar_range = 3.5 # taken from the model sdf file, modify as needed!

        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size
        self.run_name = f"stage{self.stage}__{self.learning_rate}__{self.batch_size}__{current_time.strftime('%m%d%y_%H%M')}"
        config_copy = {
            "total_timesteps"   :   self.total_timesteps,
            "learning_rate"     :   self.learning_rate,
            "num_envs"          :   self.num_envs,
            "num_steps"         :   self.num_steps,
            "anneal_lr"         :   self.anneal_lr,
            "gamma"             :   self.gamma,
            "gae_lambda"        :   self.gae_lambda,
            "num_minibatches"   :   self.num_minibatches,
            "update_epochs"     :   self.update_epochs,
            "norm_adv"          :   self.norm_adv,
            "clip_coef"         :   self.clip_coef,
            "clip_vloss"        :   self.clip_vloss,
            "ent_coef"          :   self.ent_coef,
            "vf_coef"           :   self.vf_coef,
            "max_grad_norm"     :   self.max_grad_norm,
            "target_kl"         :   self.target_kl,
            "stage"             :   self.stage,
            "batch_size"        :   self.batch_size,
            "minibatch_size"    :   self.minibatch_size,
            "num_iterations"    :   self.num_iterations,
        }
        self.run = wandb.init(
            entity="gazebo-rl",
            project=self.wandb_project_name,
            config=config_copy,
            name=self.run_name,
            save_code=True,
        )
        self.agent = Agent(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.agent.parameters(), lr=self.learning_rate, eps=1e-5)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.num_envs) + (self.state_size,)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

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
            "stage2_episode3.h5"
        )

        if self.load_model:
            checkpoint = torch.load(self.model_path)
            self.agent.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.load_episode = checkpoint['episode']
            self.global_step = checkpoint['global_step']
            print(f"model loaded from {self.model_path}")

        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')

        # self.action_pub = self.create_publisher(Float32MultiArray, '/get_action', 10)
        # self.result_pub = self.create_publisher(Float32MultiArray, 'result', 10)

        self.process()

    def process(self):
        self.env_make()
        time.sleep(1.0)
        episode_reward = 0

        episode_num = self.load_episode
        state = self.reset_environment()
        state = np.expand_dims(state, axis=1)  # manually inject a 'channel' dim
        next_obs = torch.Tensor(state).to(self.device)  # manually inject a 'channel' dim
        next_done = torch.zeros(self.num_envs).to(self.device)
        time.sleep(1.0)

        for iteration in range(1, self.num_iterations + 1):
            episode_start = time.time()
            # Annealing the rate if instructed to do so.
            if self.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.num_iterations
                lrnow = frac * self.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            for step in range(0, self.num_steps):
                step_start = time.time()
                self.global_step += self.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, explore = self.agent.get_action_and_value(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # execute the game and log data.
                time_before = time.time()
                next_obs, reward, next_done = self.step(action.item())
                plugin_response_time = time.time() - time_before
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor([next_done]).to(self.device)
                self.run.log({"reward": reward}, self.global_step)
                episode_reward += reward

                # msg = Float32MultiArray()
                # msg.data = [float(action), float(episode_reward), float(reward)]
                # self.action_pub.publish(msg)
                if next_done:
                    self.run.log({"score": episode_reward}, self.global_step)
                    episode_num += 1
                    episode_reward = 0
                    state = self.reset_environment()
                    state = np.expand_dims(state, axis=1)
                    next_obs = torch.Tensor(state).to(self.device)
                    next_done = torch.zeros(self.num_envs).to(self.device)
                    time.sleep(1.0)
                self.run.log({
                    "step_duration": time.time() - step_start,
                    "plugin_response_time": plugin_response_time,
                }, self.global_step)
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.num_steps)):
                    if t == self.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                    advantages[
                        t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.values

            # flatten the batch
            b_obs = self.obs.reshape(-1, self.state_size)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.batch_size)
            clipfracs = []
            for epoch in range(self.update_epochs):
                np.random.shuffle(b_inds)
                optimizing_steps = 0
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, explore = self.agent.get_action_and_value(b_obs[mb_inds],
                                                                                           b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.clip_coef,
                            self.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef

                    self.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    optimizing_steps += 1

                if self.target_kl is not None and approx_kl > self.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            # record rewards for plotting purposes
            metric_dict = {
                "episode":                  episode_num,
                "charts/learning_rate":     self.optimizer.param_groups[0]["lr"],
                "losses/value_loss":        v_loss.item(),
                "losses/policy_loss":       pg_loss.item(),
                "losses/entropy":           entropy_loss.item(),
                "losses/old_approx_kl":     old_approx_kl.item(),
                "losses/approx_kl":         approx_kl.item(),
                "losses/clipfrac":          np.mean(clipfracs),
                "losses/explained_variance":explained_var,
                "optimizing_steps":         optimizing_steps,
                "episode_duration":         time.time() - episode_start,
            }
            self.run.log(metric_dict, self.global_step)

            if episode_num % 100 == 0:
                self.model_path = os.path.join(
                    self.model_dir_path,
                    'ppo_stage' + str(self.stage) + '_episode' + str(episode_num) + '.h5')
                torch.save({
                    'episode': episode_num,
                    'model_state_dict': self.agent.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'global_step': self.global_step,
                }, self.model_path)
                print(f"model saved to {self.model_path}")

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
            state = np.reshape(state, [1, self.state_size])
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return state


    def step(self, action):
        req = Dqn.Request()
        req.action = action

        while not self.rl_agent_interface_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('rl_agent interface service not available, waiting again...')

        future = self.rl_agent_interface_client.call_async(req)

        rclpy.spin_until_future_complete(self, future)

        if future.result() is not None:
            next_state = np.asarray(future.result().state)
            next_state = np.reshape(next_state, [1, self.state_size])
            reward = future.result().reward
            done = future.result().done
        else:
            self.get_logger().error(
                'Exception while calling service: {0}'.format(future.exception()))

        return next_state, reward, done


class Agent(nn.Module):
    def __init__(self, n_state, n_action):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(n_state, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(n_state, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, n_action), std=0.01),
        )

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        # my own add: to track how much agent is exploring:
        greedy_action = torch.argmax(probs.probs, axis=1)
        exploration = (action != greedy_action) * 1
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), exploration



def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '1'
    max_training_episodes = args[2] if len(args) > 2 else '1000'
    rclpy.init(args=args)

    dqn_agent = PPOAgent(stage_num, max_training_episodes)
    rclpy.spin(dqn_agent)

    dqn_agent.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
