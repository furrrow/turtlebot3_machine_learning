"""
read_h5_file.py
a script to read the h5_files saved by running the inference script
installing the h5ls utility on terminal may also be helpful
h5py docs: https://docs.h5py.org/en/stable/quick.html

"""
import h5py
import numpy as np
import csv
from tqdm import tqdm
import matplotlib.pyplot as plt

def main():
    h5_path = "ppo_stage3_sample_traces.hdf5"  # change this!
    goal_csv = "ppo_stage3_sample_goals.csv"
    visualize = False

    f = h5py.File(h5_path, "r")
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

    # one may directly access an element
    print(f['x'][100])

    # **analysis** taking a look at how many episodes, success rate, and cumulative reward for each.
    rewards_list = []
    success_tally = []
    last_i = 0
    episode_num = 0

    keys_to_extract = ['step', 'x', 'y', 'theta', 'dt']
    for i in range(num_entries):
        if i == num_entries-1:
            terminal = True
        elif f['step'][i + 1][0] == 0:
            terminal = True
        else:
            terminal = False
        if terminal:
            data = [f[key][last_i:i+1] for key in keys_to_extract]
            csv_file_name = f"../csvs/trace_records{len(rewards_list)}.csv"
            data = np.array(data).squeeze(-1).T
            np.savetxt(csv_file_name, data, delimiter=",")
            if visualize:
                x = f['x'][last_i:i + 1]
                y = f['y'][last_i:i + 1]
                figure, axes = plt.subplots()
                axes.set_xlim([-2.5, 2.5])
                axes.set_ylim([-2.5, 2.5])
                circle1 = plt.Circle((1, 1), 0.1, color='gray')
                circle2 = plt.Circle((-1, 1), 0.1, color='gray')
                circle3 = plt.Circle((1, -1), 0.1, color='gray')
                circle4 = plt.Circle((-1, -1), 0.1, color='gray')
                rect = plt.Rectangle((-2.3, -2.3), width=4.6, height=4.6, edgecolor='brown', fill=False)
                axes.set_aspect(1)
                axes.add_artist(circle1)
                axes.add_artist(circle2)
                axes.add_artist(circle3)
                axes.add_artist(circle4)
                axes.add_artist(rect)
                plt.title('Trajectory Visualization')
                plt.scatter(x, y)
                plt.scatter(goal_arr[episode_num][0], goal_arr[episode_num][1], marker='x')
                plt.show()
            cumulative_reward = sum(f['reward'][last_i:i+1])[0]
            csv_entries = i - last_i + 1
            success_tally.append(f['reward'][i][0] > 90)
            rewards_list.append(cumulative_reward)
            print(f"end of epoch {len(rewards_list)}, score: {cumulative_reward:.3f}, {csv_entries} steps saved to {csv_file_name}")
            last_i = i
            episode_num += 1
            continue
    success_rate = np.sum(np.array(success_tally) > 0) / len(rewards_list)
    print(f"success rate: {success_rate:.3f}")
    print(f"keys extracted:{keys_to_extract}")


if __name__ == '__main__':
    main()