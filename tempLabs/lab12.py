
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def hypothesis(x, w, b):
    return np.dot(x, w) + b

data = pd.read_csv('iris.data', sep=',', header=None, dtype=str)
#Will use only 2 classes.
#Use first 100 samples to make it binary class data because only 2 classes
# are used for the first 100 samples.
data = data[: 100]
data = np.array(data)
#Cut & paste the class values after normalization
data[:, -1][data[:, -1] == 'Iris-setosa'] = 1
data[:, -1][data[:, -1] == 'Iris-versicolor'] = -1
y_data = np.hsplit(data, [-1])
data = data[:, :4]
#Normilize data
data = data.astype(np.float)
s = MinMaxScaler()
s.fit(data)
data = s.transform(data)
#paste class data into the normalized data
data = np.concatenate((data, y_data[1]), axis=1)
#Randomize the data.
np.random.seed(4)
np.random.shuffle(data)
#create separate partions for the single test & train set (test 30, train 70)
test_data = data[: 30]
train_data = data[30:, :]
#initialization
train_x = train_data[:, :4]
train_y = train_data[:, -1]
test_x = test_data[:, :4]
test_y = test_data[:, -1]
w = np.zeros(np.size(train_x, 1))
b = 0
#learning rate
alpha = 0.05
#GD
for i in range(50000):
    w = w - alpha * (1 / len(data)) * np.dot(np.transpose(np.dot(train_x, w) + b - train_y), train_x)
    b = b - alpha * (1 / len(data)) * sum(np.dot(train_x,w) + b - train_y)
print(w, b)
#test accuracy
print(sum(np.sign(hypothesis(test_x, w, b)) == test_y) / len(test_x))


