import os
import numpy as np
import gymnasium as gym
import random 
from highway_env.envs.common.graphics import * 
import time

# Get the current working directory
current_directory = os.getcwd()

# Specify the file name
#time = input("Enter the time stamp:", )
# file_name = 'scripts/07_17_'
# file_name = file_name + time + '.npy'
file_name = 'scripts/training_data/emg_vehicle07_25_19:50.npy'
# Combine the current directory and file name to get the file path
file_path = os.path.join(current_directory, file_name)
print(file_name)


if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")


# Load the NumPy file
data = np.load(file_path, allow_pickle=True)

# Use the data as needed


env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "manual_control": False,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 3,
        "features": ["car_id","presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 50],
            "vy": [-20, 50]
        },
        "absolute": False,
        "order": "sorted"
    }
})

obs, info = env.reset(seed = 1) #collect a single episode (replay for later)
# print("OBS: ", obs)


for d in (data):
   #action = d["action"]
#    action = handle_discrete_action_event()
   print("act", (d['man_act']))

   obs, reward, done, truncated, info = env.step(d['man_act'])
   print("Obs =", obs, d['obs'])
   #print("obs", obs)
   obs = env.render()

env.close()

