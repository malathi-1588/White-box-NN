# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 16:28:45 2022

@author: vinod
"""

from Layer import Layer
import numpy as np

#input_size = i
#output_size = j
#weights W = weights matrix (jxi)
#bias B = bias matrix (jx1)
#input X = inputs x1 x2 x3... matrix (ix1)
 

class Dense(Layer):
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(output_size, input_size)
        self.bias = np.random.rand(output_size, 1)
        
    def forward(self, input):
        self.input = input
        return np.dot(self.weights, self.input) + self.bias
    
    def backward(self, output_gradient, learning_rate):
        #TODO: update parameters and return input gradient
        
        #weights_gradient dE/dW = dE/dY.X^transpose
        weights_gradient = np.dot(output_gradient, self.input.T)
        self.weights -= learning_rate * weights_gradient
        
        #bias_gradient dE/dB = output_gradient dE/dY
        self.bias -= learning_rate * output_gradient
        
        #dE/dX = W^transpose.dE/dY
        return np.dot(self.weights.T, output_gradient)
      
