
import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import random, pygame
import cv2
import os
import time
from highway_env.envs.common import graphics 
from highway_env.envs.common import abstract



dest = os.getcwd() + "/training_data/emg_vehicle/"
hold = input()
seedNum = hold.split(":")
seedNum = int(seedNum[0])
# Combine the current directory and file name to get the file path
file_path = os.path.join(dest, hold)


if os.path.exists(file_path):
    print("File exists!")
else:
    print("File does not exist.")





# Load the NumPy file
data = np.load(file_path, allow_pickle=True)

train = []
for x in (data):
    train.append(random.randint(0,4))
keys = []
for x in data:
    print(x['man_act'])
    keys.append(x['man_act'])

#MAKE AND RUN THE EPISODE
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


#print(env.config)
obs, info = env.reset(seed= seedNum) #collect a single episode (replay for later)
check = 0
done = False
while check < len(data) and done is False:
    val = keys[check]
    val = np.int64(val)
    auto = env.action_space.sample()
    if val != 1: print(val)
    obs, reward, done, truncated, info = env.step(val)
    #print("Obs =", obs, d['obs'])
    #print("obs", obs)
    obs,val = env.render()
    check +=1

env.close()
