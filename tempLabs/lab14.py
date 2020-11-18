import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

data = pd.read_csv('iris.data', sep=',', header=None, dtype=str)
#Use first 100 samples to make it binary class data because only 2 classes
# are used for the first 100 samples.
data = data[: 100]
data = np.array(data)
#Cut & paste the class values after normalization
#   In the logistical regression the classes are labeled '1 & 0' not '1 & -1'
data[:, -1][data[:, -1] == 'Iris-setosa'] = 1.0
data[:, -1][data[:, -1] == 'Iris-versicolor'] = 0.0
y_data = np.hsplit(data, [-1])
data = data[:, :4]
#Normilize data
data = data.astype(np.float)
s = MinMaxScaler()
s.fit(data)
data = s.transform(data)
#paste class data into the normalized data
data = np.concatenate((data, y_data[1]), axis=1)
#the following line is needed because there is a type mismatch after concatenating
# the 1's & 0's.
data = data.astype(np.float)
#Randomize the data.
np.random.seed(4)
np.random.shuffle(data)
#create separate partions for the single test & train set (test 30, train 70)
test_data = data[: 30]
train_data = data[30:, :]

#initialization must include tf objects.
train_x = tf.constant(train_data[:, :4])
train_y = tf.constant(train_data[:, -1])
#reshape to make consistent with the size of train_x
train_y = tf.reshape(train_y, shape=[train_y.shape[0], 1])
test_x = tf.constant(test_data[:, :4])
#reshape to make consistent with the size of test_x
test_y = tf.constant(test_data[:, -1])
test_y = tf.reshape(test_y, shape=[test_y.shape[0], 1])

w = tf.Variable(tf.random.normal([train_x.shape[1], 1], dtype=tf.float64))
b = tf.Variable(tf.random.normal([1,1], dtype=tf.float64))
alpha = 0.05


def predict(x):
    #forward propagation
    out = tf.matmul(x,w)
    out = tf.add(out, b)
    out = tf.nn.sigmoid(out)
    return out

def loss(y_predict, y):
    return tf.reduce_mean(tf.square(y_predict - y))

def sigmoid(x):
    return 1 / (1 + tf.math.exp(-x))

#GD
for i in range(10000):
    with tf.GradientTape() as t:
        current_loss = loss(predict(train_x), train_y)
        dW, db = t.gradient(current_loss, [w,b])
        w.assign_sub(alpha * dW)
        b.assign_sub(alpha * db)

print(tf.math.reduce_sum(tf.cast((tf.math.round(sigmoid(tf.matmul(test_x, w))) == test_y), tf.float64)) / tf.size(test_y, out_type=tf.dtypes.float64))
