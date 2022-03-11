import tensorflow as tf
import numpy as np
import keras.layers
import tensorflow.keras.models
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model

xData = [1,2,3,4,5,6,7]
yData = [25000, 55000, 75000, 110000, 128000, 155000, 180000]
W = tf.Variable(tf.random.uniform([1],-100,100))
b = tf.Variable(tf.random.uniform([1],-100,100))





# model = Sequential()
# model.add(keras.layers.Dense(1,input_dim=1, activation='linear'))
# sgd = optimizers.SGD(lr=0.01)
# model.compile(optimizer=sgd, loss="mse", metrics="mse")
# model.fit(xData,yData, epochs = 300)
#
# print(model.predict([8]))
# model.save("./work-money.h5")

new_model = load_model("work-money.h5")
# pred = new_model.predict([9])
pred = new_model.predict([8])
print(pred)