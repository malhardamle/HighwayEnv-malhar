#########################################################################
# Author: Malhar Damle
# Description: Demonstrated Learning to update vehicle's action
#########################################################################
import numpy as np
import sys
import os 
from sklearn.ensemble import RandomForestClassifier
import gymnasium as gym




def load_files(file_path):
    # load all training_data (all npy files)
    # for np_name in glob.glob('*.np[yz]'):
    #     numpy_vars[np_name] = np.load(np_name, allow_pickle)


    # Load the NumPy file
    data = np.load(file_path, allow_pickle=True)

    #20  files to load  (take some validation (~5))

    #return ( a list of filled full of obs lists)
   



def trainer():
    #get valid obs #'s (files / other method)



    #go through 15/ 20 training files
    #for each file in total files
        # obs, action = load_files()
    final_model = RandomForestClassifier(n_estimators=100) #train model 
    final_model.fit(obs,action)
    return final_model


final_model = trainer() #get a new instance of trained_model

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