import numpy as np

#Initializes all weights to a fixed constant value.
class Constant:
    def __init__(self, constant=0.1):
        self.constant = constant
    
    #Returns a tensor filled with the constant value.
    def initialize(self, shape, fan_in, fan_out):
        return np.full(shape, self.constant)

#Returns a tensor initialized from a uniform distribution in [0, 1).
class UniformRandom:
    def initialize(self, shape, fan_in, fan_out):
        return np.random.uniform(0.0, 1.0, size=shape)

class Xavier:
    def initialize(self, shape, fan_in, fan_out):
        stddev = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(*shape) * stddev


class He:
    def initialize(self, shape, fan_in, fan_out):
        stddev = np.sqrt(2.0 / fan_in)
        return np.random.randn(*shape) * stddev
