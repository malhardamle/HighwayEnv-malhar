import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import random
import os
output = datetime.now().strftime("%m_%d_%H:%M")
output =  output + ".npy"



env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "manual_control": True,
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
print(env.config)
obs, info = env.reset(seed = 1) #collect a single episode (replay for later)


#print("Obs:", obs)
done = truncated = False
global data 
data = []
man_act = []
reward = 0
while not (done or truncated):
   dummy_action = env.action_space.sample() 
   data.append({'reward': reward, 'obs': obs, 'info': info, 'done':done})
   
   obs = env.render()
   obs, reward, done, truncated, info = env.step(dummy_action)
   print(env.viewer.manual_act)
   data[-1].update({'man_act': env.viewer.manual_act})

path = "scripts/training_data/emg_vehicle"
# print(data)
np.save(path + output, data)
print("-------------------------", end='\n')
print("Saved to:", path+output)
env.close()


#command to run (python path)
# ~/Desktop/HighwayEnv-malhar/.venv/bin/python run_code.py 
