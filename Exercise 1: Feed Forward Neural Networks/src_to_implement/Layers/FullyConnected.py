import numpy as np
from Layers.Base import Baselayer

# inheritence from the base class
class FullyConnected(Baselayer):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.weights = np.random.rand(input_size + 1, output_size)
        self._optimizer = None          #protected variables as mentioned
        self._gradient_weights = None
        self.input_tensor = None


    #forward method with input tensor as parameter for the next layer, will return a tensor
    def forward(self, input_tensor):
        
        self.input_tensor = input_tensor    #will need it in backward pass
        bias_term = np.ones((input_tensor.shape[0],1))  #column of ones
        input_with_bias = np.hstack((input_tensor, bias_term)) 

        self._input_with_bias = input_with_bias  #for backward

        return input_with_bias @ self.weights    #matrix multiplication
    
    #backward method with error tensor as  parameter
    def backward(self, error_tensor):
        self._gradient_weights = self._input_with_bias.T @ error_tensor   #compute gradient w.r.t. weights
        error_previous = error_tensor @ self.weights[:-1, :].T #error computation for previous layer by excluding last bias row

        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)         #update weights using optimizer

        return error_previous


    #decorators as mentioned
    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, opt):
        self._optimizer = opt

    @property
    def gradient_weights(self):
        return self._gradient_weights