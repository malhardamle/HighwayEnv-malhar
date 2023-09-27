#########################################################################
# Author: Malhar Damle
# Description: Demonstrated Learning to update vehicle's action
#########################################################################
import numpy as np
import sys
import os 
from sklearn.ensemble import RandomForestClassifier
import gymnasium as gym
import glob, time

obs_data = []
emg_presence = []
def load_files(file_path):
    f = 0
    for x in os.listdir(file_path):
        if x.endswith(".npy"):
            f+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            obs_data.extend([d['obs'] for d in data])
            for d in data:
                for n in d['obs']: # go through each vehicle row in data , check for emg vehicle via car id
                    if(n[0] == 1):
                        emg_presence.append(1)
                    else:
                        emg_presence.append(0)
               
    print("Total input datapoints "+ str(len(obs_data)) + " || total output label datapoints " + str(len(emg_presence))+ " || # of files:" + str(f))
    return obs_data, emg_presence

def clean_model_input(obs_data, emg_presence):
    # Convert observation data to a 2D array
    obs_data = np.array(obs_data) #b, 3,6
    batch = obs_data.shape[0] #b 
    # Convert action data to a 1D array
    emg_presence = np.array(emg_presence)
    obs_data = obs_data.reshape((batch * 3, -1)) # b*3 -> process one array (1 vehicles 6 features at once)
    # obs_data = obs_data.reshape((batch, -1)) #b *18 -> process entire batch -> 3 vehicles at a time with all 6 features each
    return obs_data, emg_presence
    
def trainer(obs, emg):
    assert obs.shape[0] == emg.shape[0] #check if the obs and output label shape match
    assert len(obs.shape) == 2 #cehck if obs is a 2d matrix
    assert len(emg.shape) == 1 #check if emg is 1d matrix
    final_model = RandomForestClassifier(n_estimators=200) #train model 
    final_model.fit(obs,emg) # obs = 
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



def predict(final_model, x_test):
    predict_action = final_model.predict(x_test)
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
    obs, info = env.reset() #collect a single episode (replay for later)

 
    dummy_action = env.action_space.sample() 
    obs, reward, done, truncated, info = env.step(dummy_action)
    print(obs)
    emg = predict(model, obs)
    print("EMG_Presence", emg)
    #print("obs", obs)
    obs = env.render()
    env.close()

#get an accuracy count by running the model against 
def accuracy_check(model):
        c = 0
        obs = []
       #open test episodes and extract obs data
        file_path = os.getcwd() + "/training_data/test/"
        for x in os.listdir(file_path):
            if x.endswith(".npy"):
                c+=1
                data = np.load(os.path.join(file_path, x), allow_pickle=True)
                obs.extend([d['obs'] for d in data])
    
        count=0
        emg_total = 0  #total emg data points
        no_emg_total = 0 #total no emg data points
        emg_correct = 0 #correct emg predictions
        no_emg_correct = 0 #correct no emg predictions
        f = open("test.txt", "w+")
        for n in obs: #go thru all obs data in x_test
            #3 row list
            for i in n: #each row of 3row list
                count +=1
                data = i
                x_test = data[None]
                v = predict(model, x_test)
                v = v[0]
                f.write("Model Prediction: "+str(v)+ ", ")
                emg_presence = False #Ground truth
                if(data[0]==1):
                    emg_presence = True
                    f.write("GT: 1" +"\n")
                    emg_total+=1
                else:
                    emg_presence = False
                    f.write("GT: 0" +"\n")
                    no_emg_total+=1

                if(emg_presence and v == 1): #emg exists and model predicts an emg presence
                    emg_correct+= 1
                elif(emg_presence == False and v==0): #no emg and model predicts 
                    no_emg_correct+=1

        #print accuracy results 
        emg_acc = (emg_correct / emg_total) * 100
        no_emg_acc = (no_emg_correct / no_emg_total) * 100

        print("EMG accuracy:" + str(emg_acc) + " || No emg Accuracy:" + str(no_emg_acc))
        print("Total EMG data points: " + str(emg_total) + " || Total No EMG Data points:" + str(no_emg_total))
        print("# of Test datapoints:" + str(len(obs)*3) +  " count of datapoints:" + str(count) + " || number of test files:" + str(c))
if __name__ == '__main__':
    start_time = time.time()
    cwd = os.getcwd()
    cwd = cwd + "/training_data/emg_vehicle/"
    obs_data, action_data = load_files(cwd)
    cwd = os.getcwd() +'/training_data/no_emg/'
    obs_data, action_data = load_files(cwd)
    obs_data, action_data = clean_model_input(obs_data, action_data)
    model = trainer(obs_data, action_data)  #train model based on obs and action val data
    important_features(model) #extract and print important features of model 
    #test_env(model)
    accuracy_check(model)
    
    elapsed_time = time.time() - start_time
    print(f"Elapsed time to run the script: {elapsed_time:.3f} seconds")
