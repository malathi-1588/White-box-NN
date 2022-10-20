# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 14:30:47 2022

@author: vinod
"""
"""BASE CLASS"""
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
    
    def forward(self, input):
        #TODO : return output
        pass
    
    def backward(self, output_gradient, learning_rate):
        #TODO : update parameters and return input gradient
        pass
    
    
    
