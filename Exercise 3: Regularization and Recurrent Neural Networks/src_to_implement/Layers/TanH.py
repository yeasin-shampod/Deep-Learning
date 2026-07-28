import numpy as np
from Layers import Base


class TanH(Base.BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.lastOut = np.tanh(input_tensor)
        return self.lastOut

    def backward(self, error_tensor):
        return error_tensor * (1 - self.lastOut ** 2)