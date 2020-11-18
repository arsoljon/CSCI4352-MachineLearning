import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

data = pd.read_csv('auto-mpg_removed_missing_values_updated_labels.csv', delim_whitespace=True)
pd.set_option('display.max_columns', None)

# predict MPG given 7 features, excluding car-name
# had to exclude horsepower because there was a '?' in some of the rows. did not know how to rid of them before deadline.
data = data[['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model-year', 'origin']]

# better readability for the computation
data = np.array(data)

# normalize
scaler = MinMaxScaler()
scaler.fit(data)
data = scaler.transform(data)

#x is everything but the last column
#y is only the last column.
# y is what we are trying to predict?
# in this case we are trying to predict mpg
x = data[:, 1:]
y = data[:, 0]

# intialization
w = np.array([0, 0, 0, 0, 0, 0, 0])
b = 0
# learning rate
alpha = 0.05
# GD

for i in range(100000):
    w = w - alpha * (1 / len(data)) * np.dot(np.transpose(np.dot(x, w) + b - y), x)
    b = b - alpha * (1 / len(data)) * sum(np.dot(x, w) + b - y)
print(w, b)
