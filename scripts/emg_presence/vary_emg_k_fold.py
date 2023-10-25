
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
from sklearn.model_selection import KFold, StratifiedKFold


def load_files(file_path, obs_data, emg_presence, c, max):
    counter = 0
    for x in os.listdir(file_path):
        if x.endswith(".npy") and counter < max:
            counter+=1
            c+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            obs_data.extend([d['obs'] for d in data])
            for d in data:
                for n in d['obs']: # go through each vehicle row in data , check for emg vehicle via car id
                    if(n[0] == 1):
                        emg_presence.append(1)
                    else:
                        emg_presence.append(0)
        else: 
            break
    return obs_data, action_data,c

def clean_model_input(obs_data, emg_presence):
    # Convert observation data to a 2D array
    obs_data = np.array(obs_data) #b, 3,6
    batch = obs_data.shape[0] #b 
    # Convert action data to a 1D array
    emg_presence = np.array(emg_presence)
    obs_data = obs_data.reshape((batch * 3, -1)) # b*3 -> process one array (1 vehicles 6 features at once)
    # obs_data = obs_data.reshape((batch, -1)) #b *18 -> process entire batch -> 3 vehicles at a time with all 6 features each
    return obs_data, emg_presence

    
def trainer(obs, emg, num_trees):
    assert obs.shape[0] == emg.shape[0] #check if the obs and output label shape match
    assert len(obs.shape) == 2 #cehck if obs is a 2d matrix
    assert len(emg.shape) == 1 #check if emg is 1d matrix
    final_model = RandomForestClassifier(n_estimators=num_trees) #train model 
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

def accuracy_calulation(model, test_data, x):
     #running prediction on test data 
    emg_total = 0  #total emg data points
    no_emg_total = 0 #total no emg data points
    emg_correct = 0 #correct emg predictions
    no_emg_correct = 0 #correct no emg predictions
    f = open(f"EMG-Presence_Fold:{x}.txt", "w+")
    for i in test_data:
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
    print(f"Fold {x} Accuracy Results:")
    print("EMG accuracy: " + str(emg_acc) + " || No Emg Accuracy: " + str(no_emg_acc))
    print("Total Test datapoints: " + str(len(test_data))+ " || Total EMG data points: " + str(emg_total) + " || Total No EMG Data points:" + str(no_emg_total))
    

def k_validation(dataset,emg, num_trees):
    #kf = KFold(n_splits=3, random_state=None,shuffle=True)
    #kf.get_n_splits(dataset)
    skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
    #print(kf)
    skf.get_n_splits(dataset,emg)
   
    for i, (train_index, test_index) in enumerate(skf.split(dataset,emg)): # go through each fold in kfold split
        print("---------------------------------------------------------------------------")
        print(f"Fold {i}:")
        print("Total datapoints:",len(obs_data),"|| # of training data:", len(train_index), "|| # of test data:", len(test_index))
        #print(f"  Train: index={train_index}") #list train index
        train_data = []
        emg_presence = []
        for x in train_index:
            train_data.append(obs_data[x])
            if(obs_data[x][0] == 1):
                emg_presence.append(1)
            else:
                emg_presence.append(0)
        #print(f"  Test:  index={test_index}")
        test_data = []
        for x in test_index:
            test_data.append(obs_data[x])       
        train_data = np.array(train_data)
        emg_presence = np.array(emg_presence)

        model = trainer(train_data,emg_presence, num_trees) #vary N-estimators parameters 
        important_features(model)
        accuracy_calulation(model,test_data, i)
       

if __name__ == '__main__':
    start_time = time.time()
    num_files = [10,20,30,40,50,60,70,80,90,100]
    for x in num_files:
        print("")
        print("---------------------------------------------------------------------------")
        limit = int(x /2)
        obs_data = []
        action_data = []
        c = 0
        cwd = "../training_data/emg_vehicle/"
        obs_data, action_data,c = load_files(cwd, obs_data,action_data,c,limit)
        cwd = '../training_data/no_emg/'
        obs_data, action_data,c = load_files(cwd,obs_data,action_data,c,limit)
        print("Total input datapoints " + str(len(obs_data)) + " || total output label datapoints " + str(len(action_data)) + " || # of files " +str(c))
        obs_data, action_data = clean_model_input(obs_data, action_data)
        num_trees = [100,150,200,250]
        for x in num_trees:
            print("N Estimators: ", x)
            k_validation(obs_data,action_data,x)
    
    elapsed_time = time.time() - start_time
    elapsed_time2 = elapsed_time/60
    print(f"Elapsed time to run the script: {elapsed_time:.3f} seconds or {elapsed_time2:.2f} minutes" )
