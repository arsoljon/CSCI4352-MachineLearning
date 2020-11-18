from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import tensorflow as tf
import pandas as pd
import numpy as np
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
#reformat the class column to use only a '1.0' to symbolize if a class is used in a given row.
#ie. [1.0, 0.0] or [0.0, 1.0]
y_data[1] = np.resize(y_data[1], (len(y_data[1]), 2))
for a in range(len(y_data[0])):
    if y_data[1][a][0] == 1.0:
        y_data[1][a] = [1.0, 0.0]
    else:
        y_data[1][a] = [0.0, 1.0]

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
test_data = data[: 15]
train_data = data[15:, :]
#initialization must include tf objects.
#soft max requires me to format class values as such,
# [1.0, 0.0] or [0.0, 1.0].
train_x = tf.constant(train_data[:, :4])
train_y = tf.constant(train_data[:,4:6])

test_x = tf.constant(test_data[:, :4])
test_y = tf.constant(test_data[:,4:6])

model = Sequential()
model.add(Dense(2, input_dim=4, activation='softmax'))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=1000, batch_size=1)

# test
print(model.evaluate(test_x, test_y)[1])
