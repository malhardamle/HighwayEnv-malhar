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


def load_files(file_path, obs_data, action_data, c, max):
    counter = 0
    for x in os.listdir(file_path):
        if x.endswith(".npy") and counter < max:
            counter+=1
            c+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            obs_data.extend([d['obs'] for d in data])
            action_data.extend([(d['man_act']) for d in data])
        else: 
            break
    return obs_data, action_data,c

def clean_model_input(obs_data, action_data):

     # Convert observation data to a 2D array
    obs_data = np.array(obs_data) #b, 3,6
    batch = obs_data.shape[0] #b 

    # Convert action data to a 1D array
    action_data = np.array(action_data)
    obs_data = obs_data.reshape((batch, -1)) #b *18

    return obs_data, action_data

def trainer(obs, action, num):
    assert obs.shape[0] == action.shape[0]
    assert len(obs.shape) == 2
    assert len(action.shape) == 1
    final_model = RandomForestClassifier(n_estimators=num) #train model 
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

def predict(final_model, obs):
    predict_action = final_model.predict(obs)
    return predict_action



def accuracy_check(model, test_data, test_action, x):
    c = 0
    # fix sub list within list issue
    # for x in obs:
    #     temp = []
    #     test = x.tolist()
    #     for x in test:
    #         temp.extend(x)
    #     obs_final.append(temp)
    # obs_final = np.array(obs_final)
    f = open(f"Action_Fold:{x}.txt", "w+")
    assert len(test_data) == len(test_action) #ensure # of obs == # of action values in test files
    #accuracy check
    count = 0
    f1,f1t = 0,0
    f2,f2t = 0,0
    f3, f3t = 0,0
    f4 , f4t= 0,0
    f0, f0t = 0,0
    action_total = 0
    action_correct = 0
   
    for i, n in enumerate(test_data): 
        count+=1
        action_total+=1
        data = n
        data = data[None]
        v = predict(model,data)
        v = v[0]
        f.write("Model Prediction: " + str(v) + ", ")
        gt = test_action[i]
        if gt == 0: 
            f0t+=1
            if(gt == v):
                f0+=1
        elif gt == 1: 
            f1t+=1
            if(gt == v):
                f1+=1
        elif gt == 2: 
            f2t+=1
            if(gt == v):
                f2+=1
        elif gt == 3: 
            f3t+=1
            if(gt == v):
                f3+=1
        elif gt == 4: 
            f4t+=1
            if(gt == v):
                f4+=1
        f.write("GT: " +str(gt)+"\n")
        if (gt == v):
            action_correct+=1

    action_acc = (action_correct / action_total) * 100
    if f0t == 0: f0a = 0
    else: f0a = (f0/f0t)*100
    if f1t == 0: f1a = 0
    else: f1a = (f1/f1t)*100
    if f2t == 0: f2a = 0
    else: f2a = (f2 / f2t) * 100
    if f3t == 0: f3a = 0
    else: f3a = (f3 / f3t) * 100
    if f4t == 0: f4a = 0
    else: f4a = (f4/f4t) * 100
    print("")
    print("Act0 Accuracy: ",str(f0a),  "total instances: ", f0t)
    print("Act1 Accuracy: ", str(f1a),  "total instances: ", f1t)
    print("Act2 Accuracy: ", str(f2a),  "total instances: ", f2t)
    print("Act3 Accuracy: ", str(f3a),  "total instances: ", f3t)
    print("Act4 Accuracy: ", str(f4a),  "total instances: ", f4t)
    val = "{:.3f}".format(action_acc)
    print("Action accuracy: " + val)
    print("Total action data points: " + str(action_total))
    print("# of Test datapoints:" + str(len(test_data)) +  " count of datapoints:" + str(count))



def k_validation(dataset,act, num_trees):
    skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
    skf.get_n_splits(dataset,act)
   
    for i, (train_index, test_index) in enumerate(skf.split(dataset,act)): # go through each fold in kfold split
        print("---------------------------------------------------------------------------")
        print(f"Fold {i}:")
        print("Total datapoints:",len(obs_data),"|| # of training data:", len(train_index), "|| # of test data:", len(test_index))
        #print(f"  Train: index={train_index}") #list train index
        train_data, test_data = [], []
        train_action =[]
        for x in train_index:
            train_data.append(obs_data[x])
            train_action.append(act[x])
       
        #print(f"  Test:  index={test_index}")
        test_data = []
        test_action = []
        for x in test_index:
            test_data.append(obs_data[x])   
            test_action.append(action_data[x])    
        train_data = np.array(train_data)
        train_action = np.array(train_action)

        model = trainer(train_data,train_action, num_trees) #vary N-estimators parameters 
        important_features(model)
        accuracy_check(model,test_data,test_action, i)
       


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
        cwd =  "../training_data/emg_vehicle/"
        obs_data, action_data,c = load_files(cwd, obs_data, action_data,c, limit)
        cwd = '../training_data/no_emg/'
        obs_data, action_data,c = load_files(cwd, obs_data, action_data, c, limit)
        print("Total input datapoints " + str(len(obs_data)) + " || total output label datapoints " + str(len(action_data)) + " || # of files " +str(c))
        
        obs_data, action_data = clean_model_input(obs_data, action_data)
        num_trees = [100,150,200,250]
        for x in num_trees: 
            print("")
            print("N Estimators: ", x)
            k_validation(obs_data, action_data, x)

    print("")
    print("..............................")
    elapsed_time = time.time() - start_time 
    elapsed_time2 = elapsed_time/60
    print(f"Elapsed time to run the script: {elapsed_time:.3f} seconds or {elapsed_time2:.2f} minutes" )
    

