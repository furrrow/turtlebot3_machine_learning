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
import torch.optim.lr_scheduler as lr_scheduler
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

class PPOAgent():

    def __init__(self, stage_num, use_wandb=True, make_save_folder=True):
        super().__init__()

        self.stage = int(stage_num)
        self.wandb = use_wandb
        self.state_size = 26 # 180+2 corresponds to 360 samples, originally 26
        self.action_size = 5
        self.wandb_project_name: str = "PPO_turtlebot3"

        self.done = False
        self.succeed = False
        self.fail = False
        self.load_model = True

        self.total_timesteps: int = 1000000
        self.learning_rate: float = 3e-4
        self.num_envs: int = 1
        self.num_steps: int = 2056
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
        self.iteration = 0

        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.global_step = 0

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
        if self.wandb:
            self.run = wandb.init(
                entity="gazebo-rl",
                project=self.wandb_project_name,
                config=config_copy,
                name=self.run_name,
                save_code=True,
            )
        self.network = Agent(self.state_size, self.action_size).to(self.device)
        self.optimizer = torch.optim.AdamW(self.network.parameters(), lr=self.learning_rate, eps=1e-5)
        # self.scheduler = lr_scheduler.LinearLR(self.optimizer, start_factor=1.0, end_factor=0.3, total_iters=10)
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, T_0=10, T_mult=2, eta_min=5e-8)

        # ALGO Logic: Storage setup
        self.obs = torch.zeros((self.num_steps, self.num_envs) + (self.state_size,)).to(self.device)
        self.actions = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.logprobs = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.dones = torch.zeros((self.num_steps, self.num_envs)).to(self.device)
        self.values = torch.zeros((self.num_steps, self.num_envs)).to(self.device)

        self.update_target_after = 1000
        self.target_update_after_counter = 0

        self.load_episode = 0
        self.last_save_episode = 0
        if make_save_folder:
            self.model_dir_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                'saved_model',
                self.run_name
            )
            pathlib.Path(self.model_dir_path).mkdir(parents=True, exist_ok=True)

    def load_checkpoint(self, model_path):
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(model_path, map_location=device)
        self.network.load_state_dict(checkpoint['model_state_dict'])
        # self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # self.load_episode = checkpoint['episode']
        # self.last_save_episode = checkpoint['episode']
        # self.iteration = checkpoint['iteration']
        # self.global_step = checkpoint['global_step']
        print(f"model loaded from {model_path}")

    # for when the lidar count is bigger than the state_size
    def reduce_state(self, original_state):
        lidar_state = original_state[:, 2:]
        sample_idxs = np.linspace(0, lidar_state.shape[-1]-1, self.state_size-2)
        sample_idxs = np.round(sample_idxs).astype(int)
        sampled_state = np.concatenate([original_state[:, 0:2], lidar_state[:, sample_idxs]], axis=-1)
        return sampled_state

    def log(self, info:dict):
        if self.wandb:
            self.run.log(info, self.global_step)

class RLNode(Node):
    def __init__(self, ppo_agent: PPOAgent, time_threshold=0.2, spawn_process=True):
        super().__init__('RL_node')
        self.agent = ppo_agent
        self.time_threshold = time_threshold
        self.single_step_threshold = 10
        self.rl_agent_interface_client = self.create_client(Dqn, 'rl_agent_interface')
        self.make_environment_client = self.create_client(Empty, 'make_environment')
        self.reset_environment_client = self.create_client(Dqn, 'reset_environment')
        if spawn_process:
            self.rl_process()

    def rl_process(self):
        self.env_make()
        time.sleep(1.0)
        step_duration_array = np.zeros(10)
        episode_scores = []

        episode_num = self.agent.load_episode
        iteration_num = self.agent.iteration
        state = self.reset_environment()
        state = self.agent.reduce_state(state)
        episode_reward = 0
        state = np.expand_dims(state, axis=1)  # manually inject a 'channel' dim
        next_obs = torch.Tensor(state).to(self.agent.device)  # manually inject a 'channel' dim
        next_done = torch.zeros(self.agent.num_envs).to(self.agent.device)
        time.sleep(1.0)

        for iteration in range(iteration_num, self.agent.num_iterations + 1):
            episode_start = time.time()
            local_step = 0

            for step in range(0, self.agent.num_steps):
                step_start = time.time()
                self.agent.global_step += self.agent.num_envs
                self.agent.obs[step] = next_obs
                self.agent.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value, explore = self.agent.network.get_action_and_value(next_obs)
                    self.agent.values[step] = value.flatten()
                self.agent.actions[step] = action
                self.agent.logprobs[step] = logprob

                # execute the game and log data.
                next_obs, reward, next_done = self.step(action.item())
                next_obs = self.agent.reduce_state(next_obs)
                self.agent.rewards[step] = torch.tensor(reward).to(self.agent.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.agent.device), torch.Tensor([next_done]).to(self.agent.device)
                self.agent.log({"reward": reward})
                self.agent.log({"explore": explore})
                episode_reward += reward
                local_step += 1

                # msg = Float32MultiArray()
                # msg.data = [float(action), float(episode_reward), float(reward)]
                # self.action_pub.publish(msg)
                time.sleep(0.01)
                step_duration = time.time() - step_start
                step_duration_array[self.agent.global_step % 10] = step_duration
                self.agent.log({"timer/step_duration": step_duration})

                if next_done:
                    if local_step < 2:
                        continue
                    self.agent.log({"score": episode_reward, "episode_steps": local_step})
                    print(f"iter {iteration} episode {episode_num} score {episode_reward:.3f} in {local_step} steps")
                    episode_scores.append(episode_reward)
                    episode_num += 1
                    local_step = 0
                    if np.median(step_duration_array) > self.single_step_threshold:
                        warning_msg = (f"CHECK ME...step duration {np.median(step_duration_array):.3f} "
                                      f"greater than {self.single_step_threshold}, resetting...")
                        self.get_logger().warn(warning_msg)
                        break
                    state = self.reset_environment()
                    state = self.agent.reduce_state(state)
                    episode_reward = 0
                    state = np.expand_dims(state, axis=1)
                    next_obs = torch.Tensor(state).to(self.agent.device)
                    next_done = torch.zeros(self.agent.num_envs).to(self.agent.device)
                    time.sleep(1.0)

            # Annealing the rate if instructed to do so.
            if self.agent.anneal_lr:
                self.agent.scheduler.step()

            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.network.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.agent.rewards).to(self.agent.device)
                lastgaelam = 0
                for t in reversed(range(self.agent.num_steps)):
                    if t == self.agent.num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - self.agent.dones[t + 1]
                        nextvalues = self.agent.values[t + 1]
                    delta = self.agent.rewards[t] + self.agent.gamma * nextvalues * nextnonterminal - self.agent.values[t]
                    advantages[
                        t] = lastgaelam = delta + self.agent.gamma * self.agent.gae_lambda * nextnonterminal * lastgaelam
                returns = advantages + self.agent.values

            # flatten the batch
            b_obs = self.agent.obs.reshape(-1, self.agent.state_size)
            b_logprobs = self.agent.logprobs.reshape(-1)
            b_actions = self.agent.actions.reshape(-1)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.agent.values.reshape(-1)

            # Optimizing the policy and value network
            b_inds = np.arange(self.agent.batch_size)
            clipfracs = []
            for epoch in range(self.agent.update_epochs):
                np.random.shuffle(b_inds)
                optimizing_steps = 0
                for start in range(0, self.agent.batch_size, self.agent.minibatch_size):
                    end = start + self.agent.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue, explore = self.agent.network.get_action_and_value(b_obs[mb_inds],
                                                                                           b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()

                    with torch.no_grad():
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.agent.clip_coef).float().mean().item()]

                    mb_advantages = b_advantages[mb_inds]
                    if self.agent.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.agent.clip_coef, 1 + self.agent.clip_coef)
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.agent.clip_vloss:
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.agent.clip_coef,
                            self.agent.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()
                    loss = pg_loss - self.agent.ent_coef * entropy_loss + v_loss * self.agent.vf_coef

                    self.agent.optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.agent.network.parameters(), self.agent.max_grad_norm)
                    self.agent.optimizer.step()
                    optimizing_steps += 1

                if self.agent.target_kl is not None and approx_kl > self.agent.target_kl:
                    break

            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
            avg_episode_score = np.average(np.array(episode_scores))
            # record rewards for plotting purposes
            metric_dict = {
                "episode":                  episode_num,
                "learning_rate":            self.agent.scheduler.get_last_lr()[0],
                "losses/value_loss":        v_loss.item(),
                "losses/policy_loss":       pg_loss.item(),
                "losses/entropy":           entropy_loss.item(),
                "losses/old_approx_kl":     old_approx_kl.item(),
                "losses/approx_kl":         approx_kl.item(),
                "losses/clipfrac":          np.mean(clipfracs),
                "losses/explained_variance":explained_var,
                "optimizing_steps":         optimizing_steps,
                "timer/episode_duration":   time.time() - episode_start,
            }
            self.agent.log(metric_dict)
            print(f"episode {episode_num} avg_score {avg_episode_score:.3f}"
                  f" val_loss {metric_dict['losses/value_loss']:.3f} policy_loss {metric_dict['losses/policy_loss']:.3f}")
            episode_scores = []

            if episode_num - self.agent.last_save_episode > 100:
                model_path = os.path.join(
                    self.agent.model_dir_path,
                    'ppo_stage' + str(self.agent.stage) + '_episode' + str(episode_num) + '.h5')
                torch.save({
                    'episode': episode_num,
                    'iteration': iteration,
                    'model_state_dict': self.agent.network.state_dict(),
                    'optimizer_state_dict': self.agent.optimizer.state_dict(),
                    'global_step': self.agent.global_step,
                }, model_path)
                print(f"model saved to {model_path}")
                self.agent.last_save_episode = episode_num

            if np.median(step_duration_array) >= self.time_threshold:
                warning_msg = (f"iteration {iteration} median step duration {np.median(step_duration_array):.3f} "
                               f"greater than threshold {self.time_threshold}, shutting down node...")
                self.agent.load_episode = episode_num
                self.agent.iteration = iteration + 1
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
        log_dict = {
            "timer/async_time": future_call_async - step_start_time,
            "timer/future_complete_time": future_complete - future_call_async,
            "timer/future_result_time": time.time() - future_complete,
            "timer/plugin_response_time": time.time() - step_start_time,
        }
        self.agent.log(log_dict)
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
        greedy_action = torch.argmax(probs.probs, dim=-1)
        exploration = (action != greedy_action) * 1
        return action, probs.log_prob(action), probs.entropy(), self.critic(x), exploration

    def get_greedy_action(self, x):
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        # my own add: to track how much agent is exploring:
        greedy_action = torch.argmax(probs.probs, dim=-1)
        return greedy_action



def main(args=None):
    if args is None:
        args = sys.argv
    stage_num = args[1] if len(args) > 1 else '3'

    ppo_agent = PPOAgent(stage_num, use_wandb=True)
    model_path = "/home/jim/turtlebot3_ws/src/turtlebot3_machine_learning/saved_model/stage2__0.0005__2056__082625_1058/ppo_stage2_episode1761.h5"
    ppo_agent.load_checkpoint(model_path)
    while ppo_agent.global_step < ppo_agent.total_timesteps:
        print("starting rclpy node...")
        rclpy.init()
        rl_node = RLNode(ppo_agent, time_threshold=0.12)
        # rclpy.spin(rl_node)
        rl_node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
