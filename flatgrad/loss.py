# Loss Functions
import numpy as np

#https://en.wikipedia.org/wiki/Mean_absolute_error
#  [float], [float]
def mae(yTrueValue, yPredictedValue):
    return np.mean(abs(yTrueValue - yPredictedValue))
def maeGradient(yTrueValue, yPredictedValue):
    return np.where(yPredictedValue > yTrueValue, 1, -1) / yTrueValue.size

#https://en.wikipedia.org/wiki/Mean_squared_error
#  [float], [float]
def mse(yTrueValue, yPredictedValue):
    return np.mean((yTrueValue - yPredictedValue) ** 2)
def mseGradient(yTrueValue, yPredictedValue):
    return -2 * (yTrueValue - yPredictedValue) / yTrueValue.size


# https://en.wikipedia.org/wiki/Hinge_loss
def hing():
    return


# implement categ cross-entropy for mnist