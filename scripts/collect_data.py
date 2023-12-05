import gymnasium as gym
import numpy as np 
from datetime import datetime, time
import time
import random, pygame
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
reward = 0
counter = 0
run_time = 10 #duration of episode
stop_time = time.time() +run_time #run episode for 10 seconds


while not(done or stop_prog): # loop to keep main program running
    # print("+++++++++++++++++++++")
    if(time.time() > stop_time):
        stop_prog = True
    counter+=1
    dummy_action = env.action_space.sample() 
    
    data.append({'reward': reward, 'obs': obs, 'info': info, 'done':done})
    obs,val = env.render()
    if val !=1: print(val)
    if val != 1: print("Manual Keyboard val:", val)
    obs, reward, done, truncated, info, = env.step(dummy_action)
    data[-1].update({'man_act': val}) #ADD action value to data dict
    

print(len(data), counter) #Verify length of data (Should match total steps in that episode )
for x in data: #check for None action values in list of car actions
    if x['man_act'] is None:
        print("invalid entry")
        break

cur_path = os.getcwd()
des = "/training_data/improvement2/"
save_path = cur_path+des
file = save_path + output
np.save(file, data)
print("-------------------------", end='\n')
print("Saved to:", file)
print(output)
env.close()


#command to run (python path)
# ~/Desktop/HighwayEnv-malhar/.venv/bin/python run_code.py 
