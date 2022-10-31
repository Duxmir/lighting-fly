# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 15:26:22 2022

@author: nickm
"""
#%% import libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

# sklearn models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


#%% Load in sample data
ecgData = pd.read_csv(r'Data\ECG_Zheng et al 2022\_Processed\AllData_ECG.csv',header=None)
eegData = pd.read_csv(r'Data\EEG_Kaya et al 2018\_Processed\AllData_EEG.csv', header=None)
emgData = pd.read_csv(r'Data\EMG_DiNardo et al 2022\_Processed\AllData_EMG.csv', header=None)

#%%
#Put all signals in array
frames = [ecgData, eegData, emgData]
allSignals = pd.concat(frames).to_numpy()

#Initialize label array
Y = np.empty((len(allSignals)),dtype=object)


#Fill label array
Y[0:len(frames[0])] = 'ECG'
Y[len(frames[0]):(len(frames[1]) + len(frames[0]))] = 'EEG'
Y[(len(frames[1]) + len(frames[0])):len(allSignals)] = 'EMG'


#Make sure the data has no nan values

allSignals[np.isnan(allSignals) == True] = 0

#%% Random forest

x_train, x_test, y_train, y_test = train_test_split(allSignals,Y,test_size=0.25,random_state=0)

forest = RandomForestClassifier()
forest.fit(x_train,y_train)
print("The accuracy of the forest classifier model is:", forest.score(x_test,y_test))

#%% n-estimators test

forestScores = []
n_estimators = range(10,300,10)
for i in n_estimators:
    estTestForest = RandomForestClassifier(n_estimators=i)
    estTestForest.fit(x_train,y_train)
    forestScores.append(estTestForest.score(x_test,y_test))


plt.plot(n_estimators,forestScores)
plt.xlabel("Number of trees")
plt.ylabel("Accuracy")





