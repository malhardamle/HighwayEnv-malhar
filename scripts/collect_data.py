import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import time
import random
import cv2
import os

seed_num = random.randint(0,1000) #save seed value to replicate it later 
output =str(seed_num) + ":" + datetime.now().strftime("%m_%d_%H:%M") + ".npy"


#MAKE AND RUN THE EPISODE
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
#print(env.config)
obs, info = env.reset(seed = seed_num) #collect a single episode (replay for later)

#Run episode
done = stop_prog = False
data = []
reward = 0
counter = 0

stop_time = time.time() + 10 #run episode for 10 seconds

while not(done or stop_prog):
    if(time.time() > stop_time):
        stop_prog = True
    counter+=1
    dummy_action = env.action_space.sample() 
    data.append({'reward': reward, 'obs': obs, 'info': info, 'done':done})
   
    obs = env.render()
    obs, reward, done, truncated, info = env.step(dummy_action)
    print(obs)
    print(env.viewer.manual_act) #view action value 
    data[-1].update({'man_act': env.viewer.manual_act}) #ADD action value to data dict

      
      
print(len(data), counter) #Verify length of data (Should match total steps in that episode )

cur_path = os.getcwd()
des = "/training_data/emg_vehicle/"
save_path = cur_path+des

file = save_path + output
np.save(file, data)
print("-------------------------", end='\n')
print("Saved to:", file)
env.close()


#command to run (python path)
# ~/Desktop/HighwayEnv-malhar/.venv/bin/python run_code.py 
    