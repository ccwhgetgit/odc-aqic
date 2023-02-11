#!/usr/bin/env python
# coding: utf-8

# 
# Build and publish an algorithm to predict the average concentration of one pollutant of your choice per month for the next 24 months - on average for all stations. (20 points)

# In[1]:


import pandas as pd 
import numpy as np 

data = pd.read_csv('data.csv')
data['DATA'] = pd.to_datetime(data['DATA'])
data.head()


# In[2]:


data['month'] = data['DATA'].dt.month
data['year'] = data['DATA'].dt.year
hours = [ '01h', '02h', '03h', '04h', '05h', '06h', '07h', '08h',
       '09h', '10h', '11h', '12h', '13h', '14h', '15h', '16h', '17h', '18h',
       '19h', '20h', '21h', '22h', '23h', '24h']


# In[3]:


contaminant = data['CONTAMINANT'][0]

condition = data['CONTAMINANT'] == contaminant 
pollutant_data = data.loc[condition].reset_index(drop=True)
pollutant_data = pollutant_data.sort_values(by='DATA', ascending = True).reset_index(drop=True)
pollutant_conc = pd.DataFrame(pollutant_data.groupby(['year', 'month'])[hours].mean().mean(axis=1)).reset_index()
pollutant_conc['date'] = pollutant_conc['year'] + pollutant_conc['month']
pollutant_conc = pollutant_conc.rename(columns={0:'value'})[['value', 'date']]
pollutant_conc.head()


# In[4]:


import requests
import pandas as pd 
import datetime as dt
import datetime as dt
from datetime import date
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import math
def get_mov_avg_std(df, col, N):

    mean_list = df[col].rolling(window = N, min_periods=1).mean() 
    std_list = df[col].rolling(window = N, min_periods=1).std()   
    
    # Add one timestep to the predictions
    mean_list = np.concatenate((np.array([np.nan]), np.array(mean_list[:-1])))
    std_list = np.concatenate((np.array([np.nan]), np.array(std_list[:-1])))
    
    # Append mean_list to df
    df_out = df.copy()
    df_out[col + '_mean'] = mean_list
    df_out[col + '_std'] = std_list
    
    return df_out


N= 5
shift_range = 5

for i in range(1, shift_range +1):
    pollutant_conc['lag_' + str(i)] = pollutant_conc['value'].shift(i)
pollutant_conc.dropna(inplace=True)
pollutant_conc.reset_index(drop=True, inplace=True)
pollutant_conc = get_mov_avg_std(pollutant_conc, 'value',N)
train_size = 0.6 
test_size = 0.2                                              
val_size = 0.2                                                 
N = 5 #For time lag 
num_val = int(val_size*len(pollutant_conc))
num_test = int(test_size*len(pollutant_conc))
num_train = len(pollutant_conc) - num_val - num_test
print("num_train = " + str(num_train))
print("num_val = " + str(num_val))
print("num_test = " + str(num_test))

# Split into train, cv, and test
train = pollutant_conc[:num_train]
val = pollutant_conc[num_train:num_train+num_val]
train_val = pollutant_conc[:num_train+num_val]
test = pollutant_conc[num_train+num_val:]
print("train.shape = " + str(train.shape))
print("cv.shape = " + str(val.shape))
print("train_cv.shape = " + str(train_val.shape))
print("test.shape = " + str(test.shape))

train_time = train['date']
test_time = test['date']
val_time = val['date']
train_val_time = train_val['date']

train = train.drop(columns = ['date'])
test = test.drop(columns = ['date'])
val = val.drop(columns = ['date'])
train_val = train_val.drop(columns = ['date'])

feature_pool = train.columns
output = 'value'

KNN_miss_filling = KNNImputer(n_neighbors=5).fit(test)
test = pd.DataFrame(KNN_miss_filling.transform(test))

KNN_miss_filling = KNNImputer(n_neighbors=5).fit(train_val)
train_val = pd.DataFrame(KNN_miss_filling.transform(train_val))

KNN_miss_filling = KNNImputer(n_neighbors=5).fit(val)
val = pd.DataFrame(KNN_miss_filling.transform(val))


train.columns = feature_pool
test.columns = feature_pool
train_val.columns = feature_pool
val.columns = feature_pool


standardized_features = ['value_mean', 'value_std', 'value']

for j in range(1, N+1):
    standardized_features.append("lag_"+ str(j))
    
non_standardized_features = list(set(train.columns)-set(standardized_features))
feature_pool = pollutant_conc.columns[2:]

X_train = train[feature_pool]
y_train = train[output]
X_val = val[feature_pool]
y_val = val[output]
X_train_val = train_val[feature_pool]
y_train_val = train_val[output]
X_test = test[feature_pool]
y_test = test[output]
# Get the scaler based on train set
scaler = preprocessing.MinMaxScaler().fit(train[standardized_features])

train_std=pd.DataFrame(scaler.fit_transform(train[standardized_features]))  # transform() return 'numpy.ndarray', not 'DataFrame' or 'Series'
train_nstd=pd.DataFrame(train[non_standardized_features])


train_std.columns = train_std.columns.map(lambda x: standardized_features[x])
train_std.reset_index(drop=True, inplace=True)
train_nstd.reset_index(drop=True, inplace=True)
train_scaled = pd.concat([train_std,train_nstd], sort=False,axis=1)

# Get the scaler based on cv set
scaler.val = preprocessing.MinMaxScaler().fit(val[standardized_features])


val_std=pd.DataFrame(scaler.transform(val[standardized_features]))  # transform() return 'numpy.ndarray', not 'DataFrame' or 'Series'
val_nstd=pd.DataFrame(val[non_standardized_features])
val_std.columns = val_std.columns.map(lambda x: standardized_features[x])
val_std.reset_index(drop=True, inplace=True)
val_nstd.reset_index(drop=True, inplace=True)
val_scaled = pd.concat([val_std,val_nstd], sort=False,axis=1)


scaler_trainval = preprocessing.MinMaxScaler().fit(train_val[standardized_features])

 
    
train_val_std=pd.DataFrame(scaler.transform(train_val[standardized_features]))  # transform() return 'numpy.ndarray', not 'DataFrame' or 'Series'
train_val_nstd=pd.DataFrame(train_val[non_standardized_features])
train_val_std.columns = train_val_std.columns.map(lambda x: standardized_features[x])
train_val_std.reset_index(drop=True, inplace=True)
train_val_nstd.reset_index(drop=True, inplace=True)
train_val_scaled = pd.concat([train_val_std,train_val_nstd], sort=False,axis=1)



scaler_test = preprocessing.MinMaxScaler().fit(test[standardized_features])


test_std=pd.DataFrame(scaler.transform(test[standardized_features]))  # transform() return 'numpy.ndarray', not 'DataFrame' or 'Series'
test_nstd=pd.DataFrame(test[non_standardized_features])
test_std.columns = test_std.columns.map(lambda x: standardized_features[x])
test_std.reset_index(drop=True, inplace=True)
test_nstd.reset_index(drop=True, inplace=True)
test_scaled = pd.concat([test_std,test_nstd], sort=False,axis=1)

X_train_scaled = train_scaled[feature_pool]
y_train_scaled = train_scaled['value']
X_val_scaled = val_scaled[feature_pool]
y_val_scaled= val_scaled['value']
X_train_val_scaled = train_val_scaled[feature_pool]
y_train_val_scaled = train_val_scaled['value']
X_test_scaled = test_scaled[feature_pool]
y_test_scaled = test_scaled['value']
X_train_scaled


# In[5]:


def get_mape(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) 

def get_rmse(y_true, y_pred): 
    return math.sqrt(mean_squared_error(y_true, y_pred))

def train_pred_eval_model(X_train_scaled,                           y_train_scaled,                           X_test_scaled,                           y_test,                           col_mean,                           col_std,                           seed,                           n_estimators,                           max_depth,                           learning_rate,                           min_child_weight,                           subsample,                           colsample_bytree,                           colsample_bylevel,                           gamma):

    model = XGBRegressor(objective ='reg:squarederror',seed=model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)
    

    model.fit(X_train_scaled, y_train_scaled)
    
    est_scaled = model.predict(X_test_scaled)
    est = est_scaled * col_std + col_mean

    rmse = get_rmse(y_test, est)
    mape = get_mape(y_test, est)
    
    return rmse, mape, est


#default params
n_estimators = 100                         
max_depth = 3                               
learning_rate = 0.1                         
min_child_weight = 1                  
subsample = 1                  
colsample_bytree = 1           
colsample_bylevel = 1          
gamma = 0                      
model_seed = 100


# In[6]:



def xgboost_hyperparameter_tuning():
    #Using the default parameters, test on train scaled and evaluate on validation scaled 

    rmse_bef_tuning, mape_bef_tuning, pred = train_pred_eval_model(X_train_scaled, 
                                         y_train_scaled, 
                                         X_val_scaled, 
                                         y_val, 
                                         val['value_mean'],
                                         val['value_std'],
                                         seed = model_seed,
                         n_estimators=n_estimators,
                         max_depth=max_depth,
                         learning_rate=learning_rate,
                         min_child_weight=min_child_weight,
                         subsample=subsample,
                         colsample_bytree=colsample_bytree,
                         colsample_bylevel=colsample_bylevel,
                         gamma=gamma)


    print("Phase 1")

    param_label = 'n_estimators'
    param_list = range(10, 100, 5)

    param2_label = 'max_depth'
    param2_list = [1, 2,3, 4, 5, 6, 7, 8, 9, 10]

    error_rate = {param_label: [] , param2_label: [], 'rmse': [], 'mape_pct': []}


    for param in param_list:    
        for param2 in param2_list:
            rmse, mape, pred = train_pred_eval_model(X_train_scaled, 
                                         y_train_scaled, 
                                         X_val_scaled, 
                                         y_val, 
                                         val['value_mean'],
                                         val['value_std'],
                                         seed=model_seed,
                                         n_estimators=param, 
                                         max_depth=param2, 
                                         learning_rate=learning_rate, 
                                         min_child_weight=min_child_weight, 
                                         subsample=subsample, 
                                         colsample_bytree=colsample_bytree, 
                                         colsample_bylevel=colsample_bylevel, 
                                         gamma=gamma)

            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)

    error_rate = pd.DataFrame(error_rate)
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    n_estimators_opt = temp['n_estimators'].values[0]
    max_depth_opt = temp['max_depth'].values[0]

    print("Phase 2")
    param_label = 'learning_rate'
    param_list = list(np.arange(0.01, 1, 0.01)) 


    param2_label = 'min_child_weight'
    param2_list = range(1, 21, 1)

    error_rate = {param_label: [] , param2_label: [], 'rmse': [], 'mape_pct': []}

    for param in (param_list):

        for param2 in param2_list:
            rmse, mape, pred = train_pred_eval_model(X_train_scaled, 
                                        y_train_scaled, 
                                         X_val_scaled, 
                                         y_val, 
                                          val['value_mean'],
                                         val['value_std'],
                                         seed=model_seed,
                                         n_estimators=n_estimators_opt, 
                                         max_depth=max_depth_opt, 
                                         learning_rate=param, 
                                         min_child_weight=param2, 
                                         subsample=subsample, 
                                         colsample_bytree=colsample_bytree, 
                                         colsample_bylevel=colsample_bylevel, 
                                         gamma=gamma)


            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)

    error_rate = pd.DataFrame(error_rate)
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    learning_rate_opt = temp['learning_rate'].values[0]
    min_child_weight_opt = temp['min_child_weight'].values[0]
    temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]

    print("Phase 3")
    param_label = 'subsample'
    param_list = list(np.arange(0.1, 1, 0.1))  

    param2_label = 'gamma'
    param2_list = list(np.arange(0.01, 1, 0.01))  

    error_rate = {param_label: [] , param2_label: [], 'rmse': [], 'mape_pct': []}

    for param in (param_list):
        for param2 in param2_list:
            rmse, mape, pred = train_pred_eval_model(X_train_scaled, 
                                         y_train_scaled, 
                                         X_val_scaled, 
                                         y_val, 
                                          val['value_mean'],
                                         val['value_std'],
                                         seed=model_seed,
                                         n_estimators=n_estimators_opt, 
                                         max_depth=max_depth_opt, 
                                         learning_rate=learning_rate_opt, 
                                         min_child_weight=min_child_weight_opt, 
                                         subsample=param, 
                                         colsample_bytree=colsample_bytree, 
                                         colsample_bylevel=colsample_bylevel, 
                                         gamma=param2)

            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)

    error_rate = pd.DataFrame(error_rate)
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    subsample_opt = temp['subsample'].values[0]
    gamma_opt = temp['gamma'].values[0]

    temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]

    print("Phase 4")
    param_label = 'colsample_bytree'
    param_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    param2_label = 'colsample_bylevel'
    param2_list = [0.5, 0.6, 0.7, 0.8, 0.9, 1]

    error_rate = {param_label: [] , param2_label: [], 'rmse': [], 'mape_pct': []}

    for param in (param_list):    
        for param2 in param2_list:
            rmse, mape, pred = train_pred_eval_model(X_train_scaled, 
                                         y_train_scaled, 
                                         X_val_scaled, 
                                         y_val, 
                                         val['value_mean'],
                                         val['value_std'],
                                         seed=model_seed,
                                         n_estimators=n_estimators_opt, 
                                         max_depth=max_depth_opt, 
                                         learning_rate=learning_rate_opt, 
                                         min_child_weight=min_child_weight_opt, 
                                         subsample=subsample_opt, 
                                         colsample_bytree=param, 
                                         colsample_bylevel=param2, 
                                         gamma=gamma_opt)

            error_rate[param_label].append(param)
            error_rate[param2_label].append(param2)
            error_rate['rmse'].append(rmse)
            error_rate['mape_pct'].append(mape)

    error_rate = pd.DataFrame(error_rate)
    temp = error_rate[error_rate['rmse'] == error_rate['rmse'].min()]
    colsample_bytree_opt = temp['colsample_bytree'].values[0]
    colsample_bylevel_opt = temp['colsample_bylevel'].values[0]
    # Get optimum value for param and param2, using MAPE
    # We will use RMSE to decide the final optimum params to use
    temp = error_rate[error_rate['mape_pct'] == error_rate['mape_pct'].min()]
    
    d = {'param': ['n_estimators', 'max_depth', 'learning_rate', 'min_child_weight', 'subsample', 'colsample_bytree', 'colsample_bylevel', 'gamma', 'rmse', 'mape'],
     'original': [n_estimators, max_depth, learning_rate, min_child_weight, subsample, colsample_bytree, colsample_bylevel, gamma, rmse_bef_tuning, mape_bef_tuning],
     'after_tuning': [n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt, error_rate['rmse'].min(), error_rate['mape_pct'].min()]}
    tuned_params = pd.DataFrame(d)
    tuned_params = tuned_params
    return tuned_params , [n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt]


# In[7]:


tuned_params, optimal_params = xgboost_hyperparameter_tuning()
tuned_params


# In[8]:


[n_estimators_opt, max_depth_opt, learning_rate_opt, min_child_weight_opt, subsample_opt, colsample_bytree_opt, colsample_bylevel_opt, gamma_opt] = optimal_params


# In[9]:


model = XGBRegressor(objective ='reg:squarederror',seed=model_seed,
                         n_estimators=n_estimators_opt, 
                             max_depth=max_depth_opt, 
                             learning_rate=learning_rate_opt, 
                             min_child_weight=min_child_weight_opt, 
                             subsample=subsample_opt, 
                             colsample_bytree=colsample_bytree_opt, 
                             colsample_bylevel=colsample_bylevel_opt, 
                             gamma=gamma_opt)

model.fit(X_train_val_scaled, y_train_val_scaled)

est_scaled = model.predict(X_test_scaled[-24:])
pred_vals = est_scaled * test[-24:]['value_std'].reset_index(drop=True) +    test[-24:]['value_mean'].reset_index(drop=True)
pred_vals


# In[10]:


pd.DataFrame(pred_vals)

