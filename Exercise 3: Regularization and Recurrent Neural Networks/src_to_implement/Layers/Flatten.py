from Layers import Base

#class with Base layer inheritence
class Flatten(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    #forward function with input tensor as parameter and returned as a single array input tensor
    def forward(self, input_tensor):
        self.lastShape = input_tensor.shape
        batch_size = self.lastShape[0]
        return input_tensor.reshape(batch_size, -1)
    
    #reshaped and returned tensor
    def backward(self, error_tensor):
        return error_tensor.reshape(self.lastShape)