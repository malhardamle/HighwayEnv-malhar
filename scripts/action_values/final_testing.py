#########################################################################
# Author: Malhar Damle
# Description: Demonstrated Learning to update vehicle's action
#########################################################################
import numpy as np
import sys
import os 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
import gymnasium as gym
import glob, time
from sklearn.metrics import f1_score


#how many frames is the emg actually present in the obs array
def emg_presence(obs):
    c = 0
    t = len(obs)
    for x in obs:    
        if (x[2][0] == 2): 
            c+=1
    v = (c/t)*100
    v = "{:.2f}".format(v)
    print(f"EMG is in {c} frames! {v} %")



def training_stats(action):
    act0=0
    act1 = 0
    act2 = 0
    act3 = 0
    act4 = 0
    for x in action:
        num = x
        if(num == 0): act0+=1
        elif(num == 1): act1 +=1
        elif(num == 2): act2+=1
        elif(num == 3): act3 +=1 
        else: act4+=1
    print("Action Value Split: ", act0, act1, act2, act3, act4)
    
obs_data = []
action_data = []
c = 0
counters = [0, 0, 0, 0, 0]
def load_files(file_path):
    global c
    max = 107
    for x in os.listdir(file_path):
        if x.endswith(".npy"):
            c+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            for d in data:
                obs = d['obs']
                if(d['man_act'] == 0):
                    if(counters[0] <= max):
                        counters[0]+=1
                        obs_data.append(obs)
                        action_data.append((d['man_act']))
                elif (d['man_act'] == 1):
                    if(counters[1] <= max):
                        counters[1] +=1
                        obs_data.append(obs)
                        action_data.append((d['man_act']))
                elif (d['man_act'] == 2):
                    if(counters[2] <= max):
                        counters[2] +=1
                        obs_data.append(obs)
                        action_data.append((d['man_act']))
                elif (d['man_act'] == 3):
                    if(counters[3] <= 105):
                        counters[3] +=1
                        obs_data.append(obs)
                        action_data.append((d['man_act']))
                elif (d['man_act'] == 4):
                    if(counters[4] <= max):
                        counters[4] +=1
                        obs_data.append(obs)
                        action_data.append((d['man_act']))
            # obs_data.extend([d['obs'] for d in data])
            # action_data.extend([(d['man_act']) for d in data])
    return obs_data, action_data,c

def clean_model_input(obs_data, action_data):

     # Convert observation data to a 2D array
    obs_data = np.array(obs_data) #b, 3,6
    batch = obs_data.shape[0] #b 

    # Convert action data to a 1D array
    action_data = np.array(action_data)
    obs_data = obs_data.reshape((batch, -1)) #b *18

    return obs_data, action_data
    
def trainer(n,m,obs, action):
    assert obs.shape[0] == action.shape[0]
    assert len(obs.shape) == 2
    assert len(action.shape) == 1
    final_model = RandomForestClassifier(n_estimators=n,max_depth=m, random_state=0) #train model 
    final_model.fit(obs,action)
    return final_model

def important_features(model):
     # Extract important features from the model
    importances = model.feature_importances_

    feat = {
    0: "Ego car_id",
    1: "Ego x",
    2: "Ego y",
    #3: "Ego vx ",
    #4: "Ego vy",
    3: "Traffic car_id",
    4: "Traffic x",
    5: "Traffic y",
    #8: "Traffic vx",
    #9: "Traffic vy",
    6: "Emg car_id",
    7: "Emg x",
    8: "Emg y",
    #13: "Emg vx",
    #14: "Emg vy",
}
    # Map features to their importance scores
    feature_importance_mapping = {
        f"Feature_{i+1} | {feat[i]}": importance for i, importance in enumerate(importances)
    }
    
    # Print the important features in descending order of importance
    sorted_features = sorted(feature_importance_mapping.items(), key=lambda x: x[1], reverse=True)
    
    print("Important Features:")
    for index, (feature, importance) in enumerate(sorted_features):
        #print(f"{feature}", end='')
        #print(" name, " + feat[index] + " : ", end='')
        #print(f"{importance:.4f}")
        print(f"{feature}: {importance:.4f}")
   

def predict(final_model, obs):
    predict_action = final_model.predict(obs)
    return predict_action


def open_test_files(file_path):
    c = 0
    obs = []
    act = []
    obs_final = []
    max =47
    #open test episdoes and extract obs data
    count = [0,0,0,0,0]
    for x in os.listdir(file_path):
        if x.endswith(".npy"):
            c+=1
            data = np.load(os.path.join(file_path, x), allow_pickle=True)
            for d in data:
                o = d['obs']
                if(d['man_act'] == 0):
                    if(count[0] <= max):
                        count[0]+=1
                        obs.append(o)
                        act.append((d['man_act']))
                elif (d['man_act'] == 1):
                    if(count[1] <= max):
                        count[1] +=1
                        obs.append(o)
                        act.append((d['man_act']))
                elif (d['man_act'] == 2):
                    if(count[2] <= max):
                        count[2] +=1
                        obs.append(o)
                        act.append((d['man_act']))
                elif (d['man_act'] == 3):
                    if(count[3] <= max):
                        count[3] +=1
                        obs.append(o)
                        act.append((d['man_act']))
                elif (d['man_act'] == 4):
                    if(count[4] <= max):
                        count[4] +=1
                        obs.append(o)
                        act.append((d['man_act']))
            # obs.extend([d['obs'] for d in data])
            # act.extend([d['man_act'] for d in data])
    
 
    return obs, act, c


def clean_test_obs (input):
    obs_final = []
       # fix sub list within list issue
    for x in input:
        temp = []
        test = x.tolist()
        for x in test:
            temp.extend(x)
        obs_final.append(temp)
    obs_final = np.array(obs_final)
    return obs_final


def second_classification(model, input_data, gt):
   
    v = predict(model, input_data)
    v = v[0]
    if v == gt: return True, v
    else: return False, v


def conver_gt(act):
    if act == 0 or act == 2: 
        return 0
    else: return act


def final_accuracy(binary, lane, test_obs, test_act):
    acc,t = 0,0
    c =0
    y_true= []
    y_pred = []
    f = open("nested_results.txt", "w+")
    for i, n in enumerate(test_obs):
        c+=1
        t+=1
        data = n
        data = data[None]
        v = predict(binary, data)
        v = v[0]
        f.write("Model prediction: " +  str(v) + ", ")
        gt = test_act[i]
        #binary_gt = conver_gt(gt)
        y_true.append(gt)
        if (gt == 1 and v == 1):
            acc+=1
            

        if (v == 0 and gt !=1):
            accurate, v2 = second_classification(lane, data, gt)
            if accurate: acc+=1
      
        if gt ==1: y_pred.append(v)
        else: y_pred.append(v2)


    final = (acc / t ) * 100

    score = f1_score(y_true, y_pred, average='macro')
    print("Final accuracy and F1_score: ", final, score)




def reorder(obs_data): #ego, emg, trafic -> 0,1,2
    new_list = []
    for  x in (obs_data):
            if(x[1][0] == 2): 
                og = x
                moved_list = np.concatenate((og[:1], og[2:], og[1:2])) #ego, last row, middle -> 0,1,2
                new_list.append(moved_list)
            elif(x[2][0] == 1): 
                og = x
                moved_list = np.concatenate((og[:1],og[2:], og[1:2])) #ego, last, middle -> 0,1,2
                new_list.append(moved_list)
            else: 
                new_list.append(x)
  
    return new_list


def binary_con(obs, act):
    new_list = []
    new_obs = []
    for i, x in enumerate(act):
        if x!=1:
            new_list.append(0)
            new_obs.append(obs[i])
        else: 
            new_list.append(1)
            new_obs.append(obs[i])
    return new_obs, new_list


def remove_idle(obs_data, action_data):
    new_list = []
    new_obs = []
    for i, x in enumerate(action_data):
        if x==0 or x ==2 :
            new_list.append(x)
            new_obs.append(obs_data[i])
        
    return new_obs, new_list

def two_label(input, output):
    new_act = []
    new_obs = []

    for i,x in enumerate(output):
        if x == 0 or x == 1 or x==2: 
            new_act.append(x)
            new_obs.append(input[i])
    return new_obs, new_act

def remove_presence(obs_data): #remove noise features 
    new_list = []
    for x in obs_data:
        og = x
        new = np.delete(og, 1, axis=1)
        new = new[:, :-2]
        new_list.append(new)
    return new_list

def remove_test_zeros(test):
    for x in test:
        if(x[2][0] == 0 and x[1][0] == 0): #both are 0 
            x[1][0] = 1
            x[2][0] = 2
            x[1][1] = -200
            x[1][2] = -200
            x[2][1] = -200
            x[2][2] = -200
    
        elif(x[1][0] == 0): #emg 
            x[1][0] = 1
            x[1][1] = -200
            x[1][2] = -200

        elif(x[2][0] == 0): #traffic  
            x[2][0] = 2
            x[2][1] = -200
            x[2][2] = -200

    check = False
    for x in obs_data:
        if(x[1][0] == 0 or x[2][0] == 0):
            check = True
    assert check ==False, ("Zero's still exist in the input training dataset")
   
    return test


def change_zeros(obs_data):
    for x in obs_data:
        if(x[2][0] == 0 and x[1][0] == 0): #both are 0 
            x[1][0] = 1
            x[2][0] = 2
            x[1][1] = -200
            x[1][2] = -200
            x[2][1] = -200
            x[2][2] = -200
    
        elif(x[1][0] == 0): #emg 
            x[1][0] = 1
            x[1][1] = -200
            x[1][2] = -200

        elif(x[2][0] == 0): #traffic  
            x[2][0] = 2
            x[2][1] = -200
            x[2][2] = -200

    check = False
    for x in obs_data:
        if(x[1][0] == 0 or x[2][0] == 0):
            check = True
    assert check ==False, ("Zero's still exist in the input training dataset")

def two_class(obs, act):
    new_act = []
    new_obs = []
    for i, x in enumerate(act): 
        if(x == 0 or x == 2):
            new_act.append(x)
            new_obs.append(obs[i])
    return new_obs, new_act




if __name__ == '__main__':
    start_time = time.time()
    cwd = os.getcwd()
    cwd = "../training_data/emg_vehicle/"
    obs_data, action_data,c = load_files(cwd)
    cwd = '../training_data/no_emg/'
    obs_data, action_data,c = load_files(cwd)

    cwd = '../training_data/improvement/'
    obs_data, action_data,c = load_files(cwd)
    print(obs_data[-1])
    obs_data = reorder(obs_data) #standardize the order ([ego vehicle] [traffic] [emg])
    obs_data =  remove_presence(obs_data) # remove presence and other noise features
    change_zeros(obs_data)  #remove any 0 values for emg and traffic vehicle to -200
   
   
    test = '../training_data/improvement2/'
    test_obs, test_act,d = open_test_files(test)
    
    test_obs = reorder(test_obs)
    test_obs = remove_presence(test_obs)    
    test_obs = remove_test_zeros(test_obs) 
    test_obs = clean_test_obs(test_obs)
    test_obs, test_act = two_label(test_obs, test_act)


    print("Total input datapoints " + str(len(obs_data)) + " || total output label datapoints " + str(len(action_data)) + " || # of training files + test files " +str(c) + "| " + str(d))
    emg_presence(obs_data)
    obs_data, action_data = two_label(obs_data, action_data)
    training_stats(action_data)


    obs_data, action_data = clean_model_input(obs_data, action_data)
  
    
    print("testing training stats: ", end=' ')
    training_stats(test_act)


    binary_obs, binary_action = binary_con(obs_data, action_data)#train on a binary situation 
    binary_obs = np.array(binary_obs)
    binary_action = np.array(binary_action)

    binary_model = trainer(250,5, binary_obs, binary_action) 
    print("Binary Model Features: ")
    important_features(binary_model)


    two_class_obs, two_class_act = two_class(obs_data,action_data)
    two_class_obs = np.array(two_class_obs)
    two_class_act = np.array(two_class_act)

    lane_model = trainer(150,2, two_class_obs, two_class_act)
    print("Lane Yield Model Features")
    important_features(lane_model)

  
    final_accuracy(binary_model, lane_model, test_obs, test_act)


    elapsed_time = time.time() - start_time 
    elapsed_time2 = elapsed_time/60
    print(f"Elapsed time to run the script: {elapsed_time:.3f} seconds or {elapsed_time2:.2f} minutes" )
    

