# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 22:15:30 2022

@author: vinod
"""
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Dense_Layer import Dense
from Activations import Tanh
from losses import mse, mse_prime


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))

Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]

learning_rate = 0.1
rounds = 100


for i in range(rounds):
    error = 0
    for x, y in zip(X, Y):
        #forward
        output = x
        for layer in network:
            output = layer.forward(output)
            
        #error
        error += mse(y, output)
        
        #backward
        grad = mse_prime(y, output)
        for layer in reversed(network):
            grad = layer.backward(grad, learning_rate)
            
    error /= len(X)
    print("%d/%d, error = %f" %(i + 1, rounds, error ))

# predict output for given input
def predict(l, input_data):
    samples = len(input_data)
    result =[]
    for i in range(samples):
        output = X[i]
        for layer in l:
            output = layer.forward(output)
        result.append(output)
    return result

print(predict(network, X))

'''
# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0][0]])

points = np.array(points)
print(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="Accent")
plt.show()
'''



