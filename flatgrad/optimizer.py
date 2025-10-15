# Optomizer Functions
import numpy as np
import matplotlib.pyplot as plt
import time


# https://en.wikipedia.org/wiki/Stochastic_gradient_descent
def sgd(x, y, learningRate, epochs, batchSize):
    m = len(x)
    theta = np.random.randn(2, 1)
    xBias = np.c_[np.ones((m, 1)), x]
    costHistory = []
    for epoch in range(epochs):
        indices = np.random.permutation(m)
        xShuffled = xBias[indices]
        yShuffled = y[indices]
        for i in range(0, m, batchSize):
            xBatch = xShuffled[i:i + batchSize]
            yBatch = yShuffled[i:i + batchSize]
            gradients = 2 / batchSize * xBatch.T.dot(xBatch.dot(theta) - yBatch)
            theta -= learningRate * gradients
        predictions = xBias.dot(theta)
        cost = np.mean((predictions - y) ** 2)
        costHistory.append(cost)
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Cost: {cost}")
    return theta, costHistory

