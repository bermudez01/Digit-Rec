#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 22 09:13:02 2016

@author: Ray
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
#%% Read data and preprocessing

test=pd.read_csv('/Users/undergroundking8o8/Downloads/Digit/test.csv')
train=pd.read_csv('/Users/undergroundking8o8/Downloads/Digit/train.csv')


#%%
from sklearn import preprocessing
y=train[['label']]

del train['label']
x=preprocessing.normalize(train[0::])


#%% Cross validation function
from sklearn import cross_validation
x_train, x_test, y_train, y_test= cross_validation.train_test_split(x,y, test_size=0.3, random_state=0)

#First find out what regession types/algorethems are avaliable 

#decision tree algorythem created using random normal distributions
ranf = RandomForestRegressor()
ranf.fit(x_train, y_train)
#print(ranf.score(x_val, y_val))
# Fit the model using all the available data
#ranf.fit(x_train, y_train)

print ranf.score(x_test, y_test)

#p = ranf.predict(x_prediction)

p= ranf.predict(x)

#%%Test 
a=preprocessing.normalize(test)
#y=train[['label']]

#ranf.fit(x_train, y_train)

ranf.predict(a)
b= ranf.predict(a)
#%% Write 
#b.to_csv("/Users/undergroundking8o8/Downloads/Digit/sample_submission.csv")

