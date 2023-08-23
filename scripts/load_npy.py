import os
import numpy as np
import gymnasium as gym
import random 
from highway_env.envs.common.graphics import * 
import time

# Get the current working directory

dest = os.getcwd() + "/training_data/emg_vehicle/"
# Specify the file name
#time = input("Enter the time stamp:", )
# file_name = 'scripts/07_17_'
# file_name = file_name + time + '.npy'
# file_name = 'scripts/training_data/emg_vehicle07_25_19:54.npy'
file =  input()
seedNum = file.split(":")
seedNum = int(seedNum[0])
# Combine the current directory and file name to get the file path
file_path = os.path.join(dest, file)
print(seedNum, file_path)


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

obs, info = env.reset(seed = seedNum) #collect a single episode (replay for later)
# print("OBS: ", obs)
for d in (data):
   #action = d["action"]
#    action = handle_discrete_action_event()
   if(d['man_act']!= 1):
    print("act", (d['man_act']))
   obs, reward, done, truncated, info = env.step(d['man_act'])
   #print("Obs =", obs, d['obs'])
   #print("obs", obs)
   obs = env.render()

env.close()

