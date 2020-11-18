import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

x = np.array([[1.5, 2864, 2.3],
              [2.6, 8372, 1.8],
              [1.2, 6453, 2.2],
              [2.3, 9587, 3.7],
              [1.9, 2332, 3.1],
              [3.7, 8574, 1.5],
              [2.1, 7665, 2.3],
              [1.4, 2428, 1.8],
              [3.7, 9476, 3.2],
              [1.5, 3422, 2.4]])

for j in range(x):
    for i in range(3):
        x_scaled = (x[j][i] - min(x))/ (max(x) - min(x))

scaler = MinMaxScaler()
scaler.fit(x)
x = scaler.transform(x)
print(x)

