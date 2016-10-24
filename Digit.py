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

#%%
#First find out what regession types/algorethems are avaliable 

#decision tree algorithem created using random normal distributions 85%
ranf = RandomForestRegressor()
ranf.fit(x_train, y_train)
#print(ranf.score(x_val, y_val))
# Fit the model using all the available data
#ranf.fit(x_train, y_train)

print ranf.score(x_test, y_test)

#p = ranf.predict(x_prediction)

p= ranf.predict(x)

#%% SVM 91%
from sklearn.svm import LinearSVC 
LSVC= LinearSVC()
LSVC.fit(x_train, y_train)

print LSVC.score(x_test, y_test)

#%% SVC broken
from sklearn.svm import SVC
SVC= SVC()
SVC.fit(x_train, y_train)

print SVC.score(x_test, y_test)


#%% AdaBoostClassifier  74%
from sklearn.ensemble import AdaBoostClassifier
ABC=AdaBoostClassifier()
ABC.fit(x_train, y_train)

print ABC.score(x_test, y_test)

#%% Random FOrest 93%
from sklearn.ensemble import RandomForestClassifier
RF=RandomForestClassifier()
RF.fit(x_train, y_train)

print RF.score(x_test, y_test)

#%% BaggingClassifier 92%
from sklearn.ensemble import BaggingClassifier
BC=BaggingClassifier()
BC.fit(x_train, y_train)

print BC.score(x_test, y_test)

#%% ExtraTreesClassifier 94%
from sklearn.ensemble import ExtraTreesClassifier
ETC=ExtraTreesClassifier()
ETC.fit(x_train, y_train)

print ETC.score(x_test, y_test)

p= ETC.predict(test)

#%% naive_bayes.MultinomialNB 82%
from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(x_train, y_train)

print MNB.score(x_test, y_test)

#%%GaussianNB 83%
from sklearn.naive_bayes import BernoulliNB
BNB=BernoulliNB()
BNB.fit(x_train, y_train)

print BNB.score(x_test, y_test)

#%%PassiveAggressiveClassifier 89%

from sklearn.linear_model import PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier()
pac.fit(x_train, y_train)

print pac.score(x_test, y_test)

#%%Test 
a=preprocessing.normalize(test)
#y=train[['label']]

#ranf.fit(x_train, y_train)

ranf.predict(a)
b= ranf.predict(a)
#%% Write 
#b.to_csv("/Users/undergroundking8o8/Downloads/Digit/sample_submission.csv")

