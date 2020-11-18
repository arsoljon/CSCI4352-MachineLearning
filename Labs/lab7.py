import numpy as np

a_b = np.array([3,0])
data = np.array([[2.3, 6.13], [1.2, 4.71], [4.3, 11.13], [5.7, 14.29], [3.5, 9.54],[8.9,22.43]])
x = data[:, 0]
y = data[:, 1]

mse = sum(((a_b[0] * x + a_b[1]) - y)**2) / 4

print(mse)