# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:28:51 2022

@author: vinod
"""

from Layer import Layer
import numpy as np 

class Activation(Layer):
    def __init__(self, activation, activation_prime):
        self.activation = activation
        self.activation_prime = activation_prime
        
    def forward(self, input):
        self.input = input
        return self.activation(self.input)
    
    def backward(self, output_gradient, learning_rate):
        #TODO: update parameters and return input gradient
        return np.multiply(output_gradient, self.activation_prime(self.input))
       