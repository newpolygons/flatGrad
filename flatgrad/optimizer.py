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

np.random.seed(52)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)


theta, cost = sgd(x,y,learningRate=0.1,epochs=5000,batchSize=1)

plt.figure('Cost Function During Training')
plt.plot(cost)
plt.xlabel('Epochs')
plt.ylabel('Cost (MSE)')
plt.title('Cost Function during Training')

plt.figure('Linear Regression')
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, np.c_[np.ones((x.shape[0], 1)), x].dot(
    theta), color='red', label='SGD fit line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression using Stochastic Gradient Descent')
plt.legend()
plt.show()