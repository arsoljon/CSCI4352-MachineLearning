import numpy as np
import matplotlib.pyplot as plt

data = np.array([[1.0, 2.2], [2.0,3.1], [3.0, 3.9]])
x = data[:, 0]
y = data[:, 1]

#intialization
w,b = 0, 0
#learning rate
alpha = 0.05

plt.scatter(x, y)
xl = np.linspace(0,10, 100)
#Gradient Descent
for i in range(2000):
    w = w - alpha * (1/len(data)) * sum((w * x + b - y) * x)
    b = b - alpha * (1/ len(data)) * sum(w * x + b - y)
    print(x)
    print(y)
    print(w)
    print(b)

print("w = %f, b = %f" % (w,b))
plt.plot(xl, w * xl + b)
plt.show()