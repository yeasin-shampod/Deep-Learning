import numpy as np
from Layers.Base import Baselayer

#ReLU class with no arguments in the constructor and inherited from baselayer
class ReLU(Baselayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor

        return np.maximum(0, input_tensor) #definition of RELU
    
    def backward(self, error_tensor):

        relu_derivative = self.input_tensor > 0 #1 if input > 0, else 0

        return error_tensor * relu_derivative