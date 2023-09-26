from operator import le
import numpy as np

# XOR example
#patterns = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1], [1, 1, 1, 1]])
#targets = np.array([-1, 1, 1, -1])

# OR example
patterns = np.array([[-1, 1, -1, 1], [-1, -1, 1, 1], [1, 1, 1, 1]])
targets = np.array([-1, 1, 1, 1])

weights = np.random.rand(3)

learning_rate = 0.001
epochs = 20

# Learning
for _ in range(epochs):
    weights = weights - learning_rate * np.matmul((np.matmul(weights, patterns) - targets), patterns.transpose())

# Checking if the perceptron as good memory trace
print(np.matmul(weights, patterns) > 0)
