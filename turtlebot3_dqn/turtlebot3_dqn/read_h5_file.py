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
    h5_path = "Aug27_traces.hdf5" # change this!
    trace_number = 0
    f = h5py.File(h5_path, "r")
    keys_list = list(f.keys())
    # ['action', 'dones', 'dt', 'next_state', 'reward', 'state', 'step', 'theta', 'x', 'y']
    print(keys_list)
    # some metadata can be stored as well
    for meta_key in f.attrs:
        print(f"metadata - {meta_key}:")
        print(f.attrs[meta_key])
    num_entries = f[keys_list[-1]].len()
    print(f"total of {num_entries} entries")

    # one may directly access an element
    print(f['x'][100])
    fieldnames = ['x', 'y', 'theta', 'dt', 'terminal']
    terminal = False
    # or we can do a loop, tqdm might make it pretty
    i = 0
    csv_entries = 0
    while i < num_entries:
        csv_file_name = f"trace_records{trace_number}.csv"
        with open(csv_file_name, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            terminal = abs(f['reward'][i]) > 45
            data = (f['x'][i], f['y'][i], f['theta'][i], f['dt'][i])
            writer.writerow(data)
            csv_entries += 1
            i += 1
            if terminal:
                print(f"{csv_entries} saved to {csv_file_name}")
                trace_number+= 1
                break



    # **analysis** taking a look at how many episodes, success rate, and cumulative reward for each.
    rewards_list = []
    cumulative_reward = 0
    success_tally = []
    for i in range(num_entries - 1):
        cumulative_reward += f['reward'][i][0]
        if f['step'][i+1][0] == 0:
            success_tally.append(f['reward'][i][0] > 90)
            print(f"end of epoch {len(rewards_list)}, score: {cumulative_reward:.3f}")
            rewards_list.append(cumulative_reward)
            cumulative_reward = 0
    cumulative_reward += f['reward'][num_entries-1][0]
    print(f"end of epoch {len(rewards_list)}, score: {cumulative_reward:.3f}")
    rewards_list.append(cumulative_reward)
    success_rate = np.sum(np.array(success_tally) > 0) / len(rewards_list)
    print(f"success rate: {success_rate:.3f}")

if __name__ == '__main__':
    main()