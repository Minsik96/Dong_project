import tensorflow as tf
import numpy as np
import tensorflow.keras.optimizers
from pandas.io.parsers import read_csv
from tensorflow import keras



data = read_csv("price_predict_project/price data.csv", sep=",")

xy = np.array(data, dtype="float32")
x_data = xy[:,1:-1]
y_data = xy[:,[-1]]

W = tf.Variable(tf.random.uniform([4,1]), name="Weight")
b = tf.Variable(tf.random.uniform([1]), name="bias")



# Gradient Descent part : how to make Algorithsm model?

def model_predict():
    for i in range(1001):
        with tf.GradientTape() as tape:
            H = tf.matmul(x_data, W) + b
            cost = tf.reduce_mean(tf.square(H - y_data))
        W_grad, b_grad = tape.gradient(cost, [W,b])

        learning_rate = 0.00001
        W.assign_sub(learning_rate * W_grad)
        b.assign_sub(learning_rate * b_grad)

        if i % 500 == 0:
            print("stpe : {0}  loss : {1}\nprice : {2}".format(i, cost, H[0]))
    return


tf.saved_model(model_predict(),"saved.cpkt")