import numpy as np
from Layers import Base


class Sigmoid(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.lastOut = 1 / (1 + np.exp(-input_tensor))
        return self.lastOut

    def backward(self, error_tensor):
        return error_tensor * self.lastOut * (1 - self.lastOut)