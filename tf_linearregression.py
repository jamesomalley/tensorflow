# -*- coding: utf-8 -*-
"""
Linear Regression to predict fuel miles for cars
Based on TensorFlow 2.0 tutorial by Python Engineer https://www.youtube.com/watch?v=LwM7xabuiIc
@author: James
"""
import os 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd

np.set_printoptions(precision=3, suppress=True)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing

#load data
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
column_names = ['MPG', 'Cylinders','Displacement','Horsepower','Weight','Acceleration','Model Year','Origin']

dataset = pd.read_csv(url, names=column_names, na_values='?',
                      comment='\t', sep=' ', skipinitialspace=True)


#clean data
dataset = dataset.dropna()

origin = dataset.pop('Origin')
dataset['USA'] = (origin == 1)*1
dataset['Europe'] = (origin == 2)*1
dataset['Japan'] = (origin == 3)*1

print(dataset.tail())

#train + test split

train_dataset = dataset.sample(frac=0.8, random_state=0)
test_dataset = dataset.drop(train_dataset.index)
print(dataset.shape, train_dataset.shape, test_dataset.shape)
print(train_dataset.describe().transpose())

#split features from labels

train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('MPG')
test_labels = test_features.pop('MPG')

#plot
def plot(feature, x=None, y=None):
    plt.figure(figsize=(10,8))
    plt.scatter(train_features[feature], train_labels, label='Data')
    if x is not None and y is not None:
        plt.plot(x,y, color='k', label='Predictions')
        plt.xlabel(feature)
        plt.ylabel('MPG')
        plt.legend()
        
plot('Weight')

#Normalize
print(train_dataset.describe().transpose()[['mean','std']])

normalizer = preprocessing.Normalization()
normalizer.adapt(np.array(train_features))
print(normalizer.mean.numpy())

first = np.array(train_features[:1])
print('First example:',first)
print('Normalized:',normalizer(first).numpy())

#Regression
feature = 'Horsepower'
single_feature = np.array(train_features[feature])

single_feature_normalizer = preprocessing.Normalization()
single_feature_normalizer.adapt(single_feature)

#Sequential model
single_feature_model = keras.models.Sequential([
    single_feature_normalizer,
    layers.Dense(units=1)
    ])

single_feature_model.summary()

#Loss and optimizer
loss = keras.losses.MeanAbsoluteError()
optim = keras.optimizers.Adam(learning_rate=0.1)

single_feature_model.compile(optimizer=optim, loss=loss)


history = single_feature_model.fit(
    train_features[feature], train_labels,
    epochs=100,
    verbose=1,
    validation_split = 0.2)

def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.ylim([0,25])
    plt.xlabel('Epoch')
    plt.ylabel('Error [MPG]')
    plt.legend()
    plt.grid(True)
plot_loss(history)

single_feature_model.evaluate(
    test_features[feature],
    test_labels, verbose=1)

#predict and plot
range_min = np.min(test_features[feature]) - 10
range_max = np.max(test_features[feature]) + 10
x = tf.linspace(range_min, range_max, 200)
y = single_feature_model.predict(x)
plot(feature, x,y)

#DNN
dnn_model = keras.Sequential([
    single_feature_normalizer,
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
    ])

dnn_model.compile(loss=loss, 
                  optimizer=tf.keras.optimizers.Adam(0.001))
dnn_model.summary()

dnn_model.fit(
    train_features[feature], train_labels,
    validation_split=0.2,
    verbose=1,
    epochs=100)

dnn_model.evaluate(test_features[feature],test_labels,verbose=1)

#predict and plot
x = tf.linspace(range_min, range_max, 200)
y=dnn_model.predict(x)

plot(feature, x,y)

#multiple inputs
linear_model = tf.keras.Sequential([
    normalizer,
    layers.Dense(units=1)
    ])

linear_model.compile(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    loss=loss)

linear_model.fit(
    train_features, train_labels,
    epochs=100,
    verbose=1,
    validation_split=0.2)

linear_model.evaluate(test_features[feature],test_labels,verbose=1)