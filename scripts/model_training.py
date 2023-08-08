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
   
    f = 0
    for x in os.listdir(file_path):
        if x.endswith(".npy"):
            f+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            obs_data.extend([d['obs'] for d in data])
            action_data.extend([print(d['man_act']) for d in data])
        
    print(len(obs_data), len(action_data), f)
    return obs_data, action_data

def clean_model_input(obs_data, action_data):

     # Convert observation data to a 2D array
    obs_data = np.array(obs_data) #b, 3,6
    batch = obs_data.shape[0] #b 
    # Convert action data to a 1D array
    action_data = np.array(action_data)

    obs_data = obs_data.reshape((batch, -1)) #b *18


    return obs_data, action_data
    
def trainer(obs, action):
    assert obs.shape[0] == action.shape[0]
    assert len(obs.shape) == 2
    assert len(action.shape) == 1
    final_model = RandomForestClassifier(n_estimators=100) #train model 
    final_model.fit(obs,action)
    return final_model

def important_features(model):
     # Extract important features from the model
    importances = model.feature_importances_

    # Map features to their importance scores
    feature_importance_mapping = {
        f"Feature_{i+1}": importance for i, importance in enumerate(importances)
    }

    # Print the important features in descending order of importance
    sorted_features = sorted(feature_importance_mapping.items(), key=lambda x: x[1], reverse=True)
    print("Important Features:")
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")

if __name__ == '__main__':
    cwd = os.getcwd()
    cwd = cwd + "/training_data/emg_vehicle/"
    obs_data, action_data = load_files(cwd)
    obs_data, action_data = clean_model_input(obs_data, action_data)

    model = trainer(obs_data, action_data)  #train model based on obs and action val data
    print("Trained Model: ", model) #print weights of model
    important_features(model) #extract and print important features of model 
   
    


def predict(final_model, obs):
    predict_action = final_model.predict(obs)
    return predict_action


def test_env(model):
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

    action = predict(model, obs)
    print("act", action)
    obs, reward, done, truncated, info = env.step(action)
    print("obs", obs)
    obs = env.render()

    env.close()