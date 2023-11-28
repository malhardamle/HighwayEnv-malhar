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
        if (x[1][1] != -200 and x[1][2] != -200): 
            c+=1
    v = (c/t)*100
    v = "{:.2f}".format(v)
    print(f"Emergency vehicle is in {c} frames! {v} %")



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
    max = 102
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
                    if(counters[3] <= max):
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


def open_test_files():
    c = 0
    obs = []
    act = []
    obs_final = []
    max =10
    #open test episdoes and extract obs data
    count = [0,0,0,0,0]
    file_path = ".." + "/training_data/validate/"
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
                    if(count[1] <= max*4):
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
    training_stats(act)
    act = binary_con(act)
    training_stats(act)
    obs = reorder(obs) #standardize order in test data
    obs = remove_presence(obs) #remove all presence values in obs test data

    # fix sub list within list issue
    for x in obs:
        temp = []
        test = x.tolist()
        for x in test:
            temp.extend(x)
        obs_final.append(temp)
    obs_final = np.array(obs_final)
    return obs_final, act, c



def second_classification(model, input_data, gt):
   
    v = predict(model, input_data)
    v = v[0]
    if v == gt: return True, v
    else: return False, v


def conver_gt(act):
    if act == 0 or act == 2: 
        return 0
    else: return act


def multi_class_accuracy(binary, zero_two, test_obs, test_act):
    acc,acc2 = 0,0 
    total, total2 =0,0 #total classifiier 1 and classifier 2
    count = 0
    y_true = []
    y_pred = []
    y_true2 = []
    y_pred2=[]
    nested_true = []
    nested_pred = []
    nested_acc = 0
    nested_total = 0
    v2 = -5
    f = open("test.txt", "w+")
    for i, n in enumerate(test_obs):
        count+=1
        total +=1
        data = n
        data = data[None]
        v = predict(binary, data)
        v = v[0]
        f.write("Model Prediction: " + str(v) + ", ")
        gt = test_act[i] # either 0,1,2
        binary_gt = conver_gt(gt) # 0 or 1
        y_true.append(binary_gt)
        y_pred.append(v)
        if v == binary_gt: acc+=1  
        if (v == 0 and gt !=1):
            y_true2.append(gt)
            total2+=1
            accurate, v2 = second_classification(zero_two, data, gt) #match or no match 
            y_pred2.append(v2)
            if (accurate): acc2+=1
            

        #nested 
        nested_total += 1 
        nested_true.append(gt)
        if gt == 1: nested_pred.append(v)
        elif gt !=1: nested_pred.append(v2)

        if gt == 1 and v ==1: nested_acc+=1
        elif gt == v2: nested_acc+=1 #0 or 2 

    if nested_total !=0: nested_acc = (nested_acc / nested_total) *100 
    if total !=0: acc =( acc / total ) * 100 
    if total2 !=0: acc2 =( acc2 / total2 ) * 100 
    avg = ((acc) + (acc2)) / 2
    acc =  "{:.3f}".format(acc)
    acc2 = "{:.3f}".format(acc2) 
    print("Action accuracy: ", acc,acc2, nested_acc) 
    f_score = f1_score(y_true, y_pred)
    f_score2 = f1_score(y_true2, y_pred2, pos_label=2)
    nested_f1 = f1_score(nested_true, nested_pred, average='macro')  
    return acc, acc2, avg, f_score, f_score2, nested_acc, nested_f1 

def accuracy_check(binary, obs_final, act, c):
    assert len(obs_final) == len(act) #ensure # of obs == # of action values in test files
    #accuracy check
    count = 0
    f1,f1t = 0,0
    f2,f2t = 0,0
    f3, f3t = 0,0
    f4 , f4t= 0,0
    f0, f0t = 0,0
    action_total = 0
    action_correct = 0
    f = open("test.txt", "w+")
    for i, n in enumerate(obs_final): 
        count+=1
        action_total+=1
        data = n
        data = data[None]
        v = predict(binary,data)
        v = v[0] #model prediction 
        f.write("Model Prediction: " + str(v) + ", ")
        gt = act[i]
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
    else: f4a = ((f4/f4t) * 100)
   
    f0a= "{:.3f}".format(f0a)
    f1a= "{:.3f}".format(f1a)
    f2a= "{:.3f}".format(f2a)
    f3a= "{:.3f}".format(f3a)
    f4a= "{:.3f}".format(f4a)

    if (f0t!=0): print("Act0 Accuracy: ",str(f0a),  "total instances: ", f0t)
    else: print("Act0 Accuracy: N/A", "total instances: ", f0t)

    if (f1t!=0): print("Act1 Accuracy: ", str(f1a),  "total instances: ", f1t)
    else: print("Act1 Accuracy: N/A", "total instances: ", f1t)

    if(f2t !=0): print("Act2 Accuracy: ", str(f2a),  "total instances: ", f2t)   
    else: print("Act2 Accuracy: N/A", "total instances: ", f2t)

    if(f3t!=0):print("Act3 Accuracy: ", str(f3a),  "total instances: ", f3t)
    else: print("Act2 Accuracy: N/A", "total instances: ", f2t)
    
    if(f4t!=0):print("Act4 Accuracy: ", str(f4a),  "total instances: ", f4t)
    else: print("Act2 Accuracy: N/A", "total instances: ", f2t)
    val = "{:.3f}".format(action_acc)
    print("Action accuracy: " + val)
    print("Total action data points: " + str(action_total))
    print("# of Test datapoints:" + str(len(obs_final)) +  " count of datapoints:" + str(count) + " || number of test files:" + str(c))
    return val

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

def two_label(obs_data, action_data):
    new_act = []
    new_obs = []

    for i,x in enumerate(action_data):
        if x == 0 or x == 1 or x==2: 
            new_act.append(x)
            new_obs.append(obs_data[i])

    return new_obs, new_act

def remove_presence(obs_data): #remove noise features 
    new_list = []
    for x in obs_data:
        og = x
        new = np.delete(og, 1, axis=1)
        new = new[:, :-2]
        new_list.append(new)
    return new_list


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

def k_validation(obs, act):
    skf = StratifiedKFold(n_splits=3, random_state=None, shuffle=True)
    skf.get_n_splits(obs,act)
  
    num_trees = [1,5,10,50,100,150,200,250]
    max_depth = [1,2,3,4,5]

   
    t = 0
    final_f = 0
    final_f2 = 0
    final_f3 = 0
    a,a2,a3 = 0,0,0
    estimators,e2,e3 = 0,0,0
    d,d2 ,d3= 0,0,0

    for p in num_trees: 
        print("N_ESTIMATORS: ", p)
        for md in max_depth:
            print("DEPTH: ", md)
            avg= 0
            t+=1
            for i, (train_index, test_index) in enumerate(skf.split(obs,act)): # go through each fold in kfold split
                print("---------------------------------------------------------------------------")
                print(f"Fold {i}:")
                print("Total datapoints:",len(obs),"|| # of training data:", len(train_index), "|| # of test data:", len(test_index))
                train_data, train_action = [], []
                test_data =[]
                for x in train_index:
                    train_data.append(obs_data[x])
                    train_action.append(act[x])

                test_data = []
                test_action = []
                for x in test_index: 
                    test_data.append(obs[x])
                    test_action.append(act[x])
                
                train_data = np.array(train_data) # complete obs
                train_action = np.array(train_action) #complete action

                binary_obs, binary_action = binary_con(train_data, train_action)#train on a binary situation 
                binary_obs = np.array(binary_obs)
                binary_action = np.array(binary_action)
               
                binary_model = trainer(x, md, binary_obs, binary_action)#get model 
                important_features(binary_model)

                two_class_obs, two_class_act = two_class(train_data,train_action)
                two_class_obs = np.array(two_class_obs)
                two_class_act = np.array(two_class_act)

        
                zero_two_model = trainer(x,md, two_class_obs, two_class_act)#get model 
                important_features(zero_two_model)
                c=20
               # r1 = accuracy_check(binary_model,zero_two_model, test_data, test_action,c)
                r,r2,avg, f_one, f_two, r3, nested_f = multi_class_accuracy(binary_model, zero_two_model, test_data, test_action)
            
            
                if(float(r)> float(a)):
                    estimators = p
                    d = md
                    a = r

                if(float(r2)> float(a2)):
                    e2 = p
                    d2 = md
                    a2 = r2
                if((r3) > a3):
                    e3 = p
                    d3 = md
                    a3 = r3
            print("|||||||| ", (avg)/3, " average accuracy on this split")
            final_f += f_one
            final_f2 += f_two
            final_f3 += nested_f
    final_f = final_f / t
    final_f2 = final_f2 / t
    final_f3 = final_f3 / t
    return a,  estimators ,d, a2,e2,d2, final_f, final_f2, a3, e3, d3, final_f3

if __name__ == '__main__':
    start_time = time.time()
    cwd = os.getcwd()
    cwd = "../training_data/emg_vehicle/"
    obs_data, action_data,c = load_files(cwd)
    cwd = '../training_data/no_emg/'
    obs_data, action_data,c = load_files(cwd)

    cwd = '../training_data/improvement/'
    obs_data, action_data,c = load_files(cwd)
    
    obs_data = reorder(obs_data) #standardize the order ([ego vehicle] [traffic] [emg])
    obs_data =  remove_presence(obs_data) # remove presence and other noise features
    change_zeros(obs_data)  #remove any 0 values for emg and traffic vehicle to -200
   
   
    print (obs_data[-1])
        
    print("Total input datapoints " + str(len(obs_data)) + " || total output label datapoints " + str(len(action_data)) + " || # of files " +str(c))
    training_stats(action_data)
    emg_presence(obs_data)
    #action_data = binary_con(action_data) # move all action values (0,2,3,4 = 0)
    obs_data, action_data = two_label(obs_data, action_data)
    #obs_data, action_data = remove_idle(obs_data, action_data)
    #training_stats(action_data)
    obs_data, action_data = clean_model_input(obs_data, action_data)
    
  
    
    # #model = trainer(n, m,obs_data, action_data)  #train model based on obs and action val data
    # #important_features(model) #extract and print important features of model 
    
    # best = []
    # a, estimator, md,a2,e2,d2, f_score, f_score2, a3, e3, d3, f_3 = k_validation(obs_data, action_data)   
    # print(a,"% accuracy","n_estimator: ",estimator,"DEPTH: ",md)  
    # print(a2,"% accuracy","n_estimator: ",e2,"DEPTH: ",d2)  
    # print(a3,"% Nested accuracy","n_estimator: ",e3,"DEPTH: ",d3)  
    # print(len(obs_data))
    # final = (float(a) + float(a2)) / 2
    # print("Final accuracy", final)
    # f_avg = sum([f_score, f_score2]) / 2
    # print("F1_Score", f_score, f_score2,f_avg, f_3)
    # #obs_final, act,c = open_test_files()  
    # # obs_final = obs_data
    # # act = action_data
    # # c=20   
           

    elapsed_time = time.time() - start_time 
    elapsed_time2 = elapsed_time/60
    print(f"Elapsed time to run the script: {elapsed_time:.3f} seconds or {elapsed_time2:.2f} minutes" )
    

