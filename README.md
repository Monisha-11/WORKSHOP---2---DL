# WORKSHOP 2 - DL

## AIM:

Developing a CNN model for CIFAR-10 dataset.

## PROBLEM STATEMENT:

Write a Python Program For CIFAR - 10 Dataset.The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.

## Loading the dataset in Tensorflow:

![image](https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/f4f63371-a98c-4a39-bd00-e7c274ea85bf)

## NETWORK MODEL:

<img width="868" alt="image" src="https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/10dd528d-99d2-41a1-8d25-1599328ab68e">



## PROGRAM:

```python

1) Write a python code to load the CIFAR-10 dataset

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import cifar10
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image
(X_train,y_train),(X_test,y_test) = cifar10.load_data()

2) Convert the output to one-hot encoded array

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[1310]
plt.imshow(single_image,cmap='gray')
y_train_onehot[850]
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled = X_train_scaled.reshape(-1,32,32,3)
X_test_scaled = X_test_scaled.reshape(-1,32,32,3)

3) Create a sequential model with appropriate number of neurons in the output layer, activation function and loss function

model = keras.Sequential()
model.add(layers.Input(shape=(32,32,3)))
model.add(layers.Conv2D(filters=62,kernel_size=(4,4),activation="relu"))
model.add(layers.MaxPool2D(pool_size=(4,4)))
model.add(layers.Flatten())
model.add(layers.Dense(77,activation="relu"))
model.add(layers.Dense(88,activation="relu"))
model.add(layers.Dense(99,activation="relu"))
model.add(layers.Dense(66,activation="relu"))
model.add(layers.Dense(10,activation="softmax"))
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics='accuracy')
model.fit(X_train_scaled ,y_train_onehot, epochs=3,
          batch_size=84,
          validation_data=(X_test_scaled,y_test_onehot))

4) Plot iteration vs accuracy and iteration vs loss for test and training data

metrics = pd.DataFrame(model.history.history)
metrics.head()
metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

5) Training the model to get more than 80% accuracy

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)
print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

```
## OUTPUT:

### DETECTED IMAGE 

<img width="396" alt="image" src="https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/545dd2f0-f553-4c96-9bac-7139a5c82395">

### ACCURACY

<img width="500" alt="image" src="https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/ea7ebe86-515f-4349-8503-b312fc15f0ce">

### LOSS 

<img width="497" alt="image" src="https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/c61ad9bc-f909-47e2-b200-e74f85d33541">

### confusion_matrix

![image](https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/e5b172d8-4829-43ed-8c8b-736f49bbd511)

### classification_report

<img width="410" alt="image" src="https://github.com/Monisha-11/WORKSHOP---2---DL/assets/93427240/dbdc879d-68d2-46a6-ac64-e8fc72690024">


## RESULT:

Thus the program sucessfully run by 63% accuracy.


