import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler, StandardScaler


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
# the 1's & 0's .
data = data.astype(np.float)
#Randomize the data.
np.random.seed(4)
np.random.shuffle(data)
#create separate partions for the single test & train set (test 30, train 70)
test_data = data[: 30]
train_data = data[30:, :]

#initialization is different compared to lab12 (linear classifier)
#notably, no 'b'.
train_x = train_data[:, :4]
train_y = train_data[:, -1]
test_x = test_data[:, :4]
test_y = test_data[:, -1]
w = np.zeros(np.size(train_x, 1))

#learning rate
lr = 0.05

def sigmoid(x):
    return 1 / (1 + np.e ** (-x))

#GD
for i in range(1000):
        w_diff = np.dot(np.transpose(train_y - sigmoid(np.dot(train_x, w))), train_x)
        w = w + lr * w_diff

#test
print(sum(np.round(sigmoid(np.dot(test_x, w))) == test_y) / np.size(test_y))
