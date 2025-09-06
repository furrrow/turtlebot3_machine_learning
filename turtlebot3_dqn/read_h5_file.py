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

def main():
    h5_path = "aug27_traces_greedy.hdf5" # change this!
    trace_number = 0
    f = h5py.File(h5_path, "r")
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
            cumulative_reward = sum(f['reward'][last_i:i+1])[0]
            csv_entries = i - last_i + 1
            success_tally.append(f['reward'][i][0] > 90)
            rewards_list.append(cumulative_reward)
            print(f"end of epoch {len(rewards_list)}, score: {cumulative_reward:.3f}, {csv_entries} steps saved to {csv_file_name}")
            continue
    success_rate = np.sum(np.array(success_tally) > 0) / len(rewards_list)
    print(f"success rate: {success_rate:.3f}")
    print(f"keys extracted:{keys_to_extract}")


if __name__ == '__main__':
    main()