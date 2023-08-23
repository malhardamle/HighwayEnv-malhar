import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import time
import random, pygame
import cv2
import os

from highway_env.envs.common import graphics 
from highway_env.envs.common import abstract

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
action = []
reward = 0
counter = 0

arrow_key_list = []


key_mapping = {
        pygame.K_UP: "0",
        pygame.K_UNKNOWN:"1",
        pygame.K_DOWN: "2",
        pygame.K_LEFT: "4",
        pygame.K_RIGHT: "3"
    }

run_time = 10 #duration of episode
stop_time = time.time() +run_time #run episode for 10 seconds

while not(done or stop_prog): # loop to keep main program running
    if(time.time() > stop_time):
        stop_prog = True
    counter+=1
    dummy_action = env.action_space.sample() 
    data.append({'reward': reward, 'obs': obs, 'info': info, 'done':done})

  
    obs,val = env.render()
    obs, reward, done, truncated, info = env.step(dummy_action)
    print("collect:",val)
    action.append(val)
    #print(obs)
    data[-1].update({'man_act': val}) #ADD action value to data dict




print(len(graphics.keystroke), len(action),len(data), counter) #Verify length of data (Should match total steps in that episode )

for x in action: #check for None action values in list of car actions
    if x is None:
        print("invalid entry")
        break
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
    