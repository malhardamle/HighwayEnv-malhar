
import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import random, pygame
import cv2
import os

from highway_env.envs.common import graphics 
from highway_env.envs.common import abstract



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
obs, info = env.reset() #collect a single episode (replay for later)
check = 0
done = False
while check < len(data) and done is False:
   val = 0
   for d in data:
    val = np.int64(d['man_act'])
   auto = env.action_space.sample()
   print(val)
   obs, reward, done, truncated, info = env.step(val)
   #print("Obs =", obs, d['obs'])
   #print("obs", obs)
   obs,val = env.render()
   check +=1
env.close()
