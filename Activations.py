# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 20:56:57 2022

@author: vinod
"""

from Activation_Layer import Activation
import numpy as np

class Tanh(Activation):
    def __init__(self):
        tanh = lambda x: np.tanh(x)
        tanh_prime = lambda x: 1 - np.tanh(x)**2
        super().__init__(tanh, tanh_prime)
        