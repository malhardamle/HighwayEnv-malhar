#########################################################################
# Author: Malhar Damle
# Description: Demonstrated Learning to update vehicle's action
#########################################################################
import numpy as np
import sys
import os 
from sklearn.ensemble import RandomForestClassifier
import gymnasium as gym


import glob

def load_files(file_path):
    obs_data = []
    action_data = []
    # assert len(obs_data.shape) == 2
    # assert len(action_data.shape) == 1
    f = 0
    for x in os.listdir(file_path):
        if x.endswith(".npy"):
            f+=1
            file = file_path + x
            print(x)
            data = np.load(file, allow_pickle=True)
            print(data)
            #obs_data.append(data)
            #for d in data:

                #print(d['obs']) load each step of obs data into 
                #print(d['man_act'])
    #print(len(obs_data), f)

def main():
    cwd = os.getcwd()
    cwd = cwd + "/training_data/"
    load_files(cwd)

if __name__ == '__main__':
    main()



def trainer():
    #get valid obs #'s (files / other method)


    #go through 15/ 20 training files
    #for each file in total files
        # obs, action = load_files()
    final_model = RandomForestClassifier(n_estimators=100) #train model 
    final_model.fit(obs_data,action)
    return final_model


# final_model = trainer() #get a new instance of trained_model

def predict(final_model, obs):
    predict_action = final_model.predict(obs)
    return predict_action


def test_env():
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

   # action = predict(final_model, obs)
    # print("act", action)
    # obs, reward, done, truncated, info = env.step(action)
    # print("obs", obs)
    # obs = env.render()

    # env.close()