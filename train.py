#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: ahmernajar
"""



#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the training dataset
dataset = pd.read_csv('train.csv')
X = dataset.iloc[: , 1:].values
y = dataset.iloc[: , :1].values

#Test Train Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Normalising and Reshaping
X_test = X_test / 255.0
X_test = X_test.reshape(-1,28,28,1)

X_train = X_train / 255.0
X_train = X_train.reshape(-1,28,28,1)

#encoding the output varialble
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_test = onehotencoder.fit_transform(y_test).toarray()

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
onehotencoder = OneHotEncoder(categorical_features = [0])
y_train = onehotencoder.fit_transform(y_train).toarray()


#import the libraries for CNN
from keras.layers import Dense ,BatchNormalization,Dropout
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.optimizers import Adam


#building the architecture of CNN
cnn = Sequential()

cnn.add(Convolution2D(32, 3, 3, input_shape = (28, 28, 1), activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))

cnn.add(Convolution2D(64, 3, 3, activation = 'relu'))
cnn.add(MaxPooling2D(pool_size = (2, 2)))
cnn.add(Dropout(0.4))


cnn.add(Flatten())
cnn.add(Dense(output_dim = 256, activation = 'relu'))
cnn.add(Dropout(0.4))

cnn.add(Dense(output_dim = 10, activation = 'softmax'))
cnn.add(Dropout(0.4))

cnn.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


cnn.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10)

#importing the test dataset
dataset_test = pd.read_csv('test.csv')

test_dataset = dataset_test.iloc[: , :].values

test_dataset =test_dataset.reshape(-1,28,28,1)

#predicting the results
Y_pred = cnn.predict(test_dataset)

#creating the encoded arialble to on array
results = np.argmax(x,axis = 1)

results = pd.Series(results,name="Label")

#saving the results into computer in csv format
submission = pd.concat([pd.Series(range(1,28001),name = "ImageId"),results],axis = 1)

submission.to_csv("cnn_mnist_datagen.csv",index=False)
