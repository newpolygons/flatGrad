import random
import numpy as np
from flatgrad.engine import Value
from flatgrad.nn import Neuron, Layer, MultiLayerProceptron
from flatgrad.helpers import drawGraph


'''
np.random.seed(133)
random.seed(133)
x, y = make_moons(n_samples=100, noise=0.1)
y = y*2 - 1


model = MultiLayerProceptron(2, [16, 16, 1])
print(model)
print("Number of Parameters", len(model.parameters()))


inputs = [list(map(Value, xrow)) for xrow in x]

scores = list(map(model, input))

'''