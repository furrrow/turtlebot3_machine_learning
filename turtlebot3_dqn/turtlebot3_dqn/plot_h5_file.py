"""
read_h5_file.py
a script to read the h5_files saved by running the inference script
installing the h5ls utility on terminal may also be helpful
h5py docs: https://docs.h5py.org/en/stable/quick.html

"""
import os

import h5py
import numpy as np
import csv
from tqdm import tqdm
from zipfile import ZipFile
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def main():
    hide_and_plot = True

    # h5_path = "ppo_stage2_episode1761_traces.hdf5"
    # goal_csv = "091025_2108_goals.csv"
    h5_path = "ppo_stage3_episode101_traces.hdf5"  # change this!
    goal_csv = "091225_1733_goals.csv"
    # goal_csv = None
    # h5_path = "ppo_stage3_episode101_traces.hdf5"
    # goal_csv = "091225_1733_goals.csv"
    # h5_path = "ppo_stage3_episode408_traces.hdf5"
    # goal_csv = "091225_1827_goals.csv"
    # h5_path = "ppo_stage3_episode309_traces.hdf5"
    # goal_csv = "091025_2132_goals.csv"
    # h5_path = "ppo_stage3_episode409_traces.hdf5"
    # goal_csv = "091025_2211_goals.csv"
    # h5_path = "ppo_stage3_episode717_traces.hdf5"
    # goal_csv = "091025_2252_goals.csv"
    # h5_path = "ppo_stage3_episode1024_traces.hdf5"
    # goal_csv = "091025_2327_goals.csv"
    # h5_path = "ppo_stage3_episode1532_traces.hdf5"
    # goal_csv = "091025_2350_goals.csv"
    # h5_path = "ppo_stage3_episode1839_traces.hdf5"
    # goal_csv = "091125_0011_goals.csv"

    f = h5py.File(h5_path, "r")
    if goal_csv:
        goal_arr = np.loadtxt(goal_csv, delimiter=",", dtype=float)
    keys_list = list(f.keys())
    # ['action', 'dones', 'dt', 'next_state', 'reward', 'state', 'step', 'theta', 'x', 'y']
    print(keys_list)
    # some metadata can be stored as well
    # for meta_key in f.attrs:
    #     print(f"metadata - {meta_key}:")
    #     print(f.attrs[meta_key])
    num_entries = f[keys_list[-1]].len()
    print(f"total of {num_entries} entries")

    # **analysis** taking a look at how many episodes, success rate, and cumulative reward for each.
    rewards_list = []
    success_tally = []
    file_paths = []
    last_i = 0
    episode_num = 0
    colors = dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS)

    keys_to_extract = ['step', 'x', 'y', 'theta', 'dt']
    figure, axes = plt.subplots(figsize=(8, 8))
    axes.set_xlim([-2.5, 2.5])
    axes.set_ylim([-2.5, 2.5])
    circle1 = plt.Circle((1, 1), 0.3, color='gray')
    circle2 = plt.Circle((-1, 1), 0.3, color='gray')
    circle3 = plt.Circle((1, -1), 0.3, color='gray')
    circle4 = plt.Circle((-1, -1), 0.3, color='gray')
    rect = plt.Rectangle((-2.3, -2.3), width=4.6, height=4.6, edgecolor='brown', fill=False)
    axes.set_aspect(1)
    axes.add_artist(circle1)
    axes.add_artist(circle2)
    axes.add_artist(circle3)
    axes.add_artist(circle4)
    axes.add_artist(rect)
    for i in tqdm(range(num_entries)):
        if i == num_entries-1:
            terminal = True
        elif f['step'][i + 1][0] == 0:
            terminal = True
        else:
            terminal = False
        if terminal:

            x = f['x'][last_i:i + 1]
            y = f['y'][last_i:i + 1]
            success_tally.append(f['reward'][i][0] > 90)
            plot_color = colors['dodgerblue'] if f['reward'][i][0] > 90 else colors['orangered']
            plt.scatter(x, y, s=0.1, color=plot_color)
            if not hide_and_plot:
                plt.title(h5_path[:-4])
            if goal_csv:
                plt.scatter(goal_arr[episode_num][0], goal_arr[episode_num][1], marker='x', color=plot_color)
            if hide_and_plot:
                axes.set_axis_off()
                plt.savefig(f"{h5_path[:-4]}png")
            cumulative_reward = sum(f['reward'][last_i:i+1])[0]
            rewards_list.append(cumulative_reward)
            last_i = i + 1
            episode_num += 1
            continue
    plt.savefig(f"{h5_path[:-4]}png")
    plt.show()
    success_rate = np.sum(np.array(success_tally) > 0) / len(rewards_list)
    print(f"success rate: {success_rate:.3f}")
    print(f"keys extracted:{keys_to_extract}")


if __name__ == '__main__':
    main()