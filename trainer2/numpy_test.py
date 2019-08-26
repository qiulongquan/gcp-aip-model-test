import numpy as np

y_train = np.empty((0, 5))
print(y_train)
# []

x_train = np.array((1, 5))
print(x_train, np.linspace(1, 9, num=5).reshape(1, 5))
# [1 5] [[1. 3. 5. 7. 9.]]

print(np.append(x_train, np.linspace(1, 9, num=5).reshape(1, 5)))
# [1. 5. 1. 3. 5. 7. 9.]

x = np.append(x_train, np.linspace(1, 9, num=5).reshape(1, 5))
print(x)
# [1. 5. 1. 3. 5. 7. 9.]

y_train = np.empty((0, 1))
print(np.append(y_train, np.zeros(1).reshape(1, 1), axis=0))
# [[0.]]
print(np.append(y_train, np.ones(1).reshape(1, 1), axis=0))
# [[1.]]

print(np.arange(10))
# [0 1 2 3 4 5 6 7 8 9]
