import numpy as np
import os, gymnasium as gym
import random
#numpy file
filepath = os.getcwd() + "/training_data/emg_vehicle/"
#v = input()
v = "814:09_05_23:01.npy"
data = np.load(filepath+v,allow_pickle=True)

#txt file
#file =  input()
file = "1:09_05_23:01.txt"
f = open(os.getcwd()+"/"+file, "r")
keys = []
for l in f: 
    keys.append(l.strip())


    #MAKE AND RUN THE EPISODE
env = gym.make('highway-v0', render_mode='rgb_array')
env.configure({
    "manual_control": False,
})

train = []
for x in (data):
    train.append(random.randint(0,4))

print("Random", len(train), len(keys))
#print(env.config)
obs, info = env.reset(seed= 1) #collect a single episode (replay for later)
check = 0
done = False

print(len(data), len(keys))
if (len(data) == len(keys)):
    while check < len(data) and not done:
        #print("***********************")
        x = keys[check]
        x = np.int64(x)
        #auto = env.action_space.sample()
        obs,val = env.render()
       #print("auto", train[check])
        if val != 1:
            print ("Actual:",x)
        obs, reward, done, truncated, info = env.step(x)
        check +=1
else:
    print("DATA DOES NOT MATCH ACTION")
env.close()

print("Count:" , check)