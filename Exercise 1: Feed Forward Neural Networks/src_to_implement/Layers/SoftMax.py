import numpy as np
from Layers.Base import Baselayer

#SoftMax class with base layer inheritence with no trainable parameters
class SoftMax(Baselayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    #will return the estimation class probability
    def forward (self, input_tensor):

        #using softmax function definiton
        input_stable = input_tensor - np.max(input_tensor, axis = 1, keepdims= True)
        exponent_input = np.exp(input_stable)
        self.output_tensor = exponent_input / np.sum(exponent_input, axis= 1, keepdims= True)
        return self.output_tensor
    
    def backward(self, error_tensor):

        #jacobian_vector product
        batch_size, num_classes = self.output_tensor.shape
        dx = np.empty_like(error_tensor)

        for i in range(batch_size):
            y = self.output_tensor[i].reshape(-1, 1)  # column vector
            jacobian = np.diagflat(y) - y @ y.T
            dx[i] = jacobian @ error_tensor[i]

        return dx