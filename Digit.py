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
import csv
#p.to_csv("/Users/undergroundking8o8/Downloads/Digit/sample_submission.csv")
#
#fd = open('/Users/undergroundking8o8/Downloads/Digit/sample_submission.csv','a')
#fd.write(p)
#fd.close()

ssub=pd.read_csv('/Users/undergroundking8o8/Downloads/Digit/sample_submission.csv')
ssub['Label'] = p

ssub.to_csv('list.csv', index=False)

#%% Tensorflow https://www.kaggle.com/kangk1/digit-recognizer/image-augmentation-with-keras-and-output-into-csv
import numpy as np 
import pandas as pd 
from keras.preprocessing.image import ImageDataGenerator # for data augmentation
from matplotlib import pyplot
import csv

# data preparation of training data
train = open('Users/undergroundking8o8/Downloads/Digit/train.csv').read()
train = train.split("\n")[1:-1]
train = [i.split(",") for i in train]
X_train = np.array([[int(i[j]) for j in range(1,len(i))] for i in train])
y_train = np.array([int(i[0]) for i in train])

# for the visualization, in this notebook, we only use the first 9 images
X_train = X_train[0:9]
y_train = y_train[0:9]

for i in range(0, 9):
        pyplot.subplot(3,3,i+1)
        pyplot.imshow(X_train[i].reshape((28, 28)))
pyplot.show()

X_train = X_train.reshape(X_train.shape[0], 1, 28, 28)
X_train = X_train.astype('float32')

filename = "new_image_data.csv"
new_data = [] #store new images
new_label = [] #store the lable of new images

# more kinds of augmentation can be found at https://keras.io/preprocessing/image/
datagen = ImageDataGenerator(zoom_range=0.1,
                             width_shift_range=0.3,
                             height_shift_range=0.3,
                             rotation_range=50)
datagen.fit(X_train)

#output new images into CSV
def write_to_csv(original_data, label, filename):
    for i in range(0, len(original_data)):
        pre_process = original_data[i].reshape((28*28,1))
        single_pic = []
        single_pic.append(label[0][i])
        for j in range(0,len(pre_process)):
            temp_pix = pre_process[j][0]
            single_pic.append(temp_pix)
        with open(filename,"a") as f:
            f_csv = csv.writer(f, delimiter=',',quotechar='|', quoting=csv.QUOTE_MINIMAL)
            f_csv.writerow(single_pic)
            
number_of_batches = 100
batches = 0

for X_batch, Y_batch in datagen.flow(X_train, y_train, batch_size=10):
    new_data.append(X_batch)
    new_label.append(Y_batch) 
#         loss = model.train(X_batch, Y_batch)
    batches += 1
    if batches >= number_of_batches:
        # we need to break the loop by hand because
        # the generator loops indefinitely
        break

#show the new images
for i in range(0, 4):
        pyplot.subplot(2,2,i+1)
        pyplot.imshow(new_data[1][i].reshape((28, 28)))
    # show the plot
pyplot.show()

#write new images into CSV
for i in range(0,len(new_data)):
    write_to_csv(new_data[i], new_label, filename)
