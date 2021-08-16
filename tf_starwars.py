# -*- coding: utf-8 -*-
"""
CNN Model for image recognition of LEGO Star Wars characters using transfer learning

Based on TensorFlow 2.0 tutorial by Python Engineer https://www.youtube.com/watch?v=LwM7xabuiIc
LEGO dataset: https://www.kaggle.com/ihelon/lego-minifigures-classification

"""

import os
import math
import random
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

BASE_DIR = '\data\star-wars'
names = ["YODA","LUKE SKYWALKER","R2-D2","MACE WINDU","GENERAL GRIEVOUS"]

tf.random.set_seed(1)

#Read dataset
train_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
test_gen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_batches = train_gen.flow_from_directory(
    'data/star-wars/train',
    target_size=(256,256),
    class_mode='sparse',
    batch_size=4,
    shuffle=True,
    color_mode="rgb",
    classes=names
    )

val_batches = valid_gen.flow_from_directory(
   'data/star-wars/val',
   target_size=(256,256),
   class_mode='sparse',
   batch_size=4,
   shuffle=False,
   color_mode='rgb',
   classes=names
   )

test_batches = test_gen.flow_from_directory(
    'data/star-wars/test',
    target_size=(256,256),
    class_mode='sparse',
    batch_size=4,
    shuffle=False,
    color_mode='rgb',
    classes=names
    )

train_batch = train_batches[0]
test_batch = test_batches[0]

def show(batch, pred_labels=None):
    plt.figure(figsize=(10,10))
    for i in range(4):
        plt.subplot(2,2,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(batch[0][i], cmap=plt.cm.binary)
        lbl = names[int(batch[1][i])]
        if pred_labels is not None:
            lbl+= "/ Pred:" + names[int(pred_labels[i])]
        plt.xlabel(lbl)
    plt.show()

#Model
model = keras.models.Sequential()
model.add(layers.Conv2D(32, (3,3), strides=(1,1), padding="valid", activation='relu', input_shape=(256,256,3)))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Conv2D(64,3,activation='relu'))
model.add(layers.MaxPool2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(5))
print(model.summary())  

#Loss and optimizer
loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optim = keras.optimizers.Adam(lr=0.001)
metrics = ['accuracy']

model.compile(optimizer=optim, loss=loss, metrics=metrics)

#training
epochs = 30

#callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=5,
    verbose=2)

history = model.fit(train_batches, validation_data=val_batches,
                    callbacks=[early_stopping],epochs=epochs, verbose=2)

model.save("lego_model.h5")

#Results
plt.figure(figsize=(16,6))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='valid loss')
plt.grid()
plt.legend(fontsize=15)

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='valid acc')
plt.grid()
plt.legend(fontsize=15)

model.evaluate(test_batches, verbose=2)

predictions = model.predict(test_batches)
predictions = tf.nn.softmax(predictions)
labels = np.argmax(predictions, axis=1)
print(test_batches[0][1])
print(labels[0:4])

show(test_batches[0], labels[0:4])


